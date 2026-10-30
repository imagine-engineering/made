use std::fs;
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::mem::ManuallyDrop;
#[cfg(unix)]
use std::os::fd::FromRawFd;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use git_credential_nostr::{
    CredentialStore, CredentialStoreError, GitConfig, HelperConfig, StoredCredential,
    SystemCredentialStore,
};
use nostr::{FromBech32 as _, PublicKey as NostrPublicKey};
use nostr_compat::buzz_nips::identity::OwnerAttestation;
use nostr_compat::buzz_nips::project_workflow::{
    GitOwnerAttestation, GitSignatureEnvelope, OwnerAttestationStatus, compute_git_signing_hash,
};
use nostr_compat::{EventSignature, PublicKey};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, constants::SECRET_KEY_SIZE};

pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_REQUEST_FAILED: i32 = 1;
pub const EXIT_KEYRING_LOCKED: i32 = 2;
pub const EXIT_CREDENTIAL_MISSING: i32 = 3;
pub const EXIT_SIGNING_FAILED: i32 = 4;
pub const EXIT_VERIFICATION_FAILED: i32 = 5;

const MAX_PAYLOAD_BYTES: usize = 100 * 1024 * 1024;
const MAX_SIGNATURE_BYTES: usize = 8 * 1024;
const MAX_ARGUMENT_BYTES: usize = 4 * 1024;
const MAX_CREDENTIAL_IDENTIFIER_BYTES: usize = 1024;
const CREDENTIAL_IDENTIFIER_PREFIX: &str = "zed-nostr://credential/v1/";
const CONFIG_CREDENTIAL_IDENTIFIER: &str = "nostr.credentialIdentifier";
const CONFIG_AUTH_TAG: &str = "nostr.authTag";
const CONFIG_SIGNING_KEY: &str = "user.signingkey";
const GNUPG_PREFIX: &str = "[GNUPG:] ";

#[derive(Clone, Debug, Eq, PartialEq)]
enum Mode {
    Sign { key_id: String },
    Verify { signature_file: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Invocation {
    mode: Mode,
    status_fd: Option<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureReadError {
    Missing,
    Invalid,
}

pub trait SignatureReader {
    fn read(&self, path: &str) -> Result<String, SignatureReadError>;
}

pub trait Clock {
    fn unix_timestamp(&self) -> Result<u32, ()>;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn unix_timestamp(&self) -> Result<u32, ()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| ())?
            .as_secs();
        u32::try_from(timestamp).map_err(|_| ())
    }
}

pub struct SystemSignatureReader;

impl SignatureReader for SystemSignatureReader {
    fn read(&self, path: &str) -> Result<String, SignatureReadError> {
        if path.is_empty() || path.len() > MAX_ARGUMENT_BYTES || path.chars().any(char::is_control)
        {
            return Err(SignatureReadError::Invalid);
        }
        #[cfg(unix)]
        let file = {
            use std::os::unix::fs::OpenOptionsExt as _;
            let file = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(path)
                .map_err(|error| {
                    if error.kind() == io::ErrorKind::NotFound {
                        SignatureReadError::Missing
                    } else {
                        SignatureReadError::Invalid
                    }
                })?;
            let metadata = file.metadata().map_err(|_| SignatureReadError::Invalid)?;
            if !metadata.file_type().is_file()
                || metadata.len() > u64::try_from(MAX_SIGNATURE_BYTES).unwrap_or(u64::MAX)
            {
                return Err(SignatureReadError::Invalid);
            }
            file
        };
        #[cfg(not(unix))]
        let file = {
            let file = fs::File::open(path).map_err(|error| {
                if error.kind() == io::ErrorKind::NotFound {
                    SignatureReadError::Missing
                } else {
                    SignatureReadError::Invalid
                }
            })?;
            let metadata = file.metadata().map_err(|_| SignatureReadError::Invalid)?;
            if !metadata.file_type().is_file()
                || metadata.len() > u64::try_from(MAX_SIGNATURE_BYTES).unwrap_or(u64::MAX)
            {
                return Err(SignatureReadError::Invalid);
            }
            file
        };
        let mut contents = String::new();
        file.take(u64::try_from(MAX_SIGNATURE_BYTES + 1).unwrap_or(u64::MAX))
            .read_to_string(&mut contents)
            .map_err(|_| SignatureReadError::Invalid)?;
        if contents.len() > MAX_SIGNATURE_BYTES {
            return Err(SignatureReadError::Invalid);
        }
        Ok(contents)
    }
}

struct SecretKeyGuard(SecretKey);

impl Drop for SecretKeyGuard {
    fn drop(&mut self) {
        self.0.non_secure_erase();
    }
}

struct KeypairGuard(Keypair);

impl Drop for KeypairGuard {
    fn drop(&mut self) {
        self.0.non_secure_erase();
    }
}

#[derive(Clone, Copy)]
struct ExecutionError {
    exit_code: i32,
    message: &'static str,
}

impl ExecutionError {
    const fn new(exit_code: i32, message: &'static str) -> Self {
        Self { exit_code, message }
    }
}

pub fn run() -> i32 {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let invocation = match parse_invocation(&arguments) {
        Ok(invocation) => invocation,
        Err(error) => return emit_error(&mut io::stderr(), error),
    };
    if validate_status_fd(effective_status_fd(&invocation)).is_err() {
        return emit_error(
            &mut io::stderr(),
            ExecutionError::new(EXIT_REQUEST_FAILED, "status output is unavailable"),
        );
    }
    let mut payload = Vec::new();
    if io::stdin()
        .take(u64::try_from(MAX_PAYLOAD_BYTES + 1).unwrap_or(u64::MAX))
        .read_to_end(&mut payload)
        .is_err()
        || payload.len() > MAX_PAYLOAD_BYTES
    {
        return emit_error(
            &mut io::stderr(),
            ExecutionError::new(EXIT_REQUEST_FAILED, "git object payload is invalid"),
        );
    }
    let mut status = Vec::new();
    let exit_code = execute(
        &invocation,
        &payload,
        &GitConfig,
        &SystemCredentialStore,
        &SystemSignatureReader,
        &SystemClock,
        &mut io::stdout(),
        &mut status,
        &mut io::stderr(),
    );
    if write_status(effective_status_fd(&invocation), &status).is_err() {
        return EXIT_REQUEST_FAILED;
    }
    exit_code
}

pub fn run_with(
    arguments: &[String],
    payload: &[u8],
    config: &dyn HelperConfig,
    store: &dyn CredentialStore,
    signatures: &dyn SignatureReader,
    clock: &dyn Clock,
    stdout: &mut dyn Write,
    status: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    let invocation = match parse_invocation(arguments) {
        Ok(invocation) => invocation,
        Err(error) => return emit_error(stderr, error),
    };
    execute(
        &invocation,
        payload,
        config,
        store,
        signatures,
        clock,
        stdout,
        status,
        stderr,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute(
    invocation: &Invocation,
    payload: &[u8],
    config: &dyn HelperConfig,
    store: &dyn CredentialStore,
    signatures: &dyn SignatureReader,
    clock: &dyn Clock,
    stdout: &mut dyn Write,
    status: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    if payload.len() > MAX_PAYLOAD_BYTES {
        return emit_error(
            stderr,
            ExecutionError::new(EXIT_REQUEST_FAILED, "git object payload is invalid"),
        );
    }
    let result = match &invocation.mode {
        Mode::Sign { key_id } => sign(key_id, payload, config, store, clock, stdout, status),
        Mode::Verify { signature_file } => {
            verify(signature_file, payload, config, signatures, status)
        }
    };
    match result {
        Ok(()) => EXIT_SUCCESS,
        Err(error) => emit_error(stderr, error),
    }
}

fn parse_invocation(arguments: &[String]) -> Result<Invocation, ExecutionError> {
    if arguments.iter().any(|argument| {
        argument.len() > MAX_ARGUMENT_BYTES || argument.chars().any(char::is_control)
    }) {
        return Err(ExecutionError::new(
            EXIT_REQUEST_FAILED,
            "git signing invocation is invalid",
        ));
    }
    let mut status_fd = None;
    let mut sign_key = None;
    let mut signature_file = None;
    let mut saw_stdin_marker = false;
    let mut index = 0;
    while let Some(argument) = arguments.get(index) {
        if let Some(value) = argument.strip_prefix("--status-fd=") {
            set_status_fd(&mut status_fd, value)?;
        } else if argument == "--status-fd" {
            index += 1;
            let value = arguments.get(index).ok_or_else(invalid_invocation)?;
            set_status_fd(&mut status_fd, value)?;
        } else if argument == "-bsau" {
            if sign_key.is_some() || signature_file.is_some() {
                return Err(invalid_invocation());
            }
            index += 1;
            sign_key = Some(arguments.get(index).ok_or_else(invalid_invocation)?.clone());
        } else if argument == "--verify" {
            if signature_file.is_some() || sign_key.is_some() {
                return Err(invalid_invocation());
            }
            index += 1;
            signature_file = Some(arguments.get(index).ok_or_else(invalid_invocation)?.clone());
        } else if argument == "-" {
            saw_stdin_marker = true;
        }
        index += 1;
    }
    let mode = match (sign_key, signature_file) {
        (Some(key_id), None) if !key_id.is_empty() => Mode::Sign { key_id },
        (None, Some(signature_file)) if saw_stdin_marker && !signature_file.is_empty() => {
            Mode::Verify { signature_file }
        }
        _ => return Err(invalid_invocation()),
    };
    Ok(Invocation { mode, status_fd })
}

fn set_status_fd(target: &mut Option<i32>, value: &str) -> Result<(), ExecutionError> {
    if target.is_some() {
        return Err(invalid_invocation());
    }
    let descriptor = value.parse::<i32>().map_err(|_| invalid_invocation())?;
    if descriptor < 1 {
        return Err(invalid_invocation());
    }
    *target = Some(descriptor);
    Ok(())
}

fn invalid_invocation() -> ExecutionError {
    ExecutionError::new(EXIT_REQUEST_FAILED, "git signing invocation is invalid")
}

fn sign(
    key_id: &str,
    payload: &[u8],
    config: &dyn HelperConfig,
    store: &dyn CredentialStore,
    clock: &dyn Clock,
    stdout: &mut dyn Write,
    status: &mut dyn Write,
) -> Result<(), ExecutionError> {
    let identifiers = config.values(CONFIG_CREDENTIAL_IDENTIFIER).map_err(|_| {
        ExecutionError::new(EXIT_REQUEST_FAILED, "signing configuration is unavailable")
    })?;
    let identifier = match identifiers.as_slice() {
        [identifier] if valid_credential_identifier(identifier) => identifier,
        _ => {
            return Err(ExecutionError::new(
                EXIT_REQUEST_FAILED,
                "active signing credential is not configured",
            ));
        }
    };
    let stored = match store.read(identifier) {
        Ok(Some(stored)) => stored,
        Ok(None) => {
            return Err(ExecutionError::new(
                EXIT_CREDENTIAL_MISSING,
                "active signing credential is unavailable",
            ));
        }
        Err(CredentialStoreError::Locked) => {
            return Err(ExecutionError::new(
                EXIT_KEYRING_LOCKED,
                "protected credential storage is locked",
            ));
        }
        Err(CredentialStoreError::Unavailable) => {
            return Err(ExecutionError::new(
                EXIT_KEYRING_LOCKED,
                "protected credential storage is unavailable",
            ));
        }
    };
    let (signer, keypair) = signing_key(&stored, key_id)?;
    let timestamp = clock.unix_timestamp().map_err(|()| {
        ExecutionError::new(EXIT_SIGNING_FAILED, "signing timestamp is unavailable")
    })?;
    let owner_attestation = load_owner_attestation(config, signer, timestamp)?;
    let signing_hash = compute_git_signing_hash(timestamp, owner_attestation.as_ref(), payload)
        .map_err(|_| ExecutionError::new(EXIT_SIGNING_FAILED, "git object signing failed"))?;
    let secp256k1 = Secp256k1::signing_only();
    let signature =
        secp256k1.sign_schnorr_no_aux_rand(&Message::from_digest(signing_hash), &keypair.0);
    let signature = EventSignature::from_hex(&signature.to_string())
        .map_err(|_| ExecutionError::new(EXIT_SIGNING_FAILED, "git object signing failed"))?;
    let envelope = GitSignatureEnvelope {
        signer,
        signature,
        timestamp,
        owner_attestation,
    };
    stdout
        .write_all(envelope.to_armored().as_bytes())
        .and_then(|()| stdout.flush())
        .map_err(|_| ExecutionError::new(EXIT_REQUEST_FAILED, "signature output is unavailable"))?;
    status_line(status, "BEGIN_SIGNING")?;
    status_line(
        status,
        &format!(
            "SIG_CREATED D 8 1 00 {timestamp} {}",
            envelope.signer.to_hex()
        ),
    )?;
    Ok(())
}

fn signing_key(
    stored: &StoredCredential,
    key_id: &str,
) -> Result<(PublicKey, KeypairGuard), ExecutionError> {
    if stored.secret().len() != SECRET_KEY_SIZE {
        return Err(invalid_signing_credential());
    }
    let secret_key =
        SecretKey::from_slice(stored.secret()).map_err(|_| invalid_signing_credential())?;
    let secret_key = SecretKeyGuard(secret_key);
    let secp256k1 = Secp256k1::signing_only();
    let keypair = KeypairGuard(Keypair::from_secret_key(&secp256k1, &secret_key.0));
    let (public_key, _) = keypair.0.x_only_public_key();
    let public_key = public_key.to_string();
    if !stored.username().eq_ignore_ascii_case(&public_key)
        || normalize_key_id(key_id).as_deref() != Some(public_key.as_str())
    {
        return Err(invalid_signing_credential());
    }
    let public_key = PublicKey::from_hex(&public_key).map_err(|_| invalid_signing_credential())?;
    Ok((public_key, keypair))
}

fn invalid_signing_credential() -> ExecutionError {
    ExecutionError::new(EXIT_SIGNING_FAILED, "active signing credential is invalid")
}

fn load_owner_attestation(
    config: &dyn HelperConfig,
    signer: PublicKey,
    timestamp: u32,
) -> Result<Option<GitOwnerAttestation>, ExecutionError> {
    let values = config.values(CONFIG_AUTH_TAG).map_err(|_| {
        ExecutionError::new(EXIT_REQUEST_FAILED, "signing configuration is unavailable")
    })?;
    let value = match values.as_slice() {
        [] => return Ok(None),
        [value] => value,
        _ => {
            return Err(ExecutionError::new(
                EXIT_REQUEST_FAILED,
                "owner attestation configuration is invalid",
            ));
        }
    };
    let tag = serde_json::from_str::<Vec<String>>(value).map_err(|_| {
        ExecutionError::new(
            EXIT_REQUEST_FAILED,
            "owner attestation configuration is invalid",
        )
    })?;
    let attestation = OwnerAttestation::parse_tag(&tag).map_err(|_| {
        ExecutionError::new(
            EXIT_REQUEST_FAILED,
            "owner attestation configuration is invalid",
        )
    })?;
    attestation
        .verify_for_membership(&signer, u64::from(timestamp))
        .map_err(|_| {
            ExecutionError::new(
                EXIT_REQUEST_FAILED,
                "owner attestation configuration is invalid",
            )
        })?;
    Ok(Some(GitOwnerAttestation {
        owner: attestation.owner,
        conditions: attestation.conditions,
        signature: attestation.signature,
    }))
}

fn verify(
    signature_file: &str,
    payload: &[u8],
    config: &dyn HelperConfig,
    signatures: &dyn SignatureReader,
    status: &mut dyn Write,
) -> Result<(), ExecutionError> {
    let armored = match signatures.read(signature_file) {
        Ok(armored) => armored,
        Err(_) => {
            status_line(status, "ERRSIG 0000000000000000 0 0 00 0 9")?;
            return Err(ExecutionError::new(
                EXIT_REQUEST_FAILED,
                "detached signature is unavailable",
            ));
        }
    };
    let envelope = match GitSignatureEnvelope::parse_armored(&armored) {
        Ok(envelope) => envelope,
        Err(_) => {
            status_line(status, "ERRSIG 0000000000000000 0 0 00 0 9")?;
            return Err(ExecutionError::new(
                EXIT_VERIFICATION_FAILED,
                "detached signature is invalid",
            ));
        }
    };
    let verification = match envelope.verify(payload) {
        Ok(verification) => verification,
        Err(_) => {
            status_line(status, "NEWSIG")?;
            let signer = envelope.signer.to_hex();
            status_line(status, &format!("BADSIG {signer} {signer}"))?;
            return Err(ExecutionError::new(
                EXIT_VERIFICATION_FAILED,
                "git object signature is invalid",
            ));
        }
    };
    let signer = verification.signer.to_hex();
    let date = DateTime::<Utc>::from_timestamp(i64::from(verification.timestamp), 0)
        .map(|timestamp| timestamp.format("%Y-%m-%d").to_string())
        .ok_or_else(|| {
            ExecutionError::new(EXIT_VERIFICATION_FAILED, "signature timestamp is invalid")
        })?;
    let trust = if configured_signing_key(config).as_deref() == Some(signer.as_str()) {
        "TRUST_FULLY"
    } else {
        "TRUST_UNDEFINED"
    };
    status_line(status, "NEWSIG")?;
    status_line(status, &format!("GOODSIG {signer} {signer}"))?;
    status_line(
        status,
        &format!(
            "VALIDSIG {signer} {date} {} 0 - - - - - {signer}",
            verification.timestamp
        ),
    )?;
    status_line(status, &format!("{trust} 0 shell"))?;
    status_line(status, "NOTATION_NAME nostr-trust-model")?;
    status_line(status, "NOTATION_DATA advisory-config-match-only")?;
    status_line(status, "NOTATION_NAME nostr-oa-status")?;
    let owner_status = match verification.owner_attestation {
        OwnerAttestationStatus::Absent => "none",
        OwnerAttestationStatus::Valid => "valid",
        OwnerAttestationStatus::Invalid => "invalid_signature",
    };
    status_line(status, &format!("NOTATION_DATA {owner_status}"))?;
    if let Some(owner) = verification.owner {
        status_line(status, "NOTATION_NAME nostr-oa-owner")?;
        status_line(status, &format!("NOTATION_DATA {}", owner.to_hex()))?;
    }
    Ok(())
}

fn configured_signing_key(config: &dyn HelperConfig) -> Option<String> {
    let values = config.values(CONFIG_SIGNING_KEY).ok()?;
    match values.as_slice() {
        [value] => normalize_key_id(value),
        _ => None,
    }
}

fn normalize_key_id(value: &str) -> Option<String> {
    let value = value.trim();
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Some(value.to_ascii_lowercase());
    }
    NostrPublicKey::from_bech32(value)
        .ok()
        .map(|key| key.to_hex())
}

fn valid_credential_identifier(value: &str) -> bool {
    value.starts_with(CREDENTIAL_IDENTIFIER_PREFIX)
        && value.len() <= MAX_CREDENTIAL_IDENTIFIER_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn status_line(status: &mut dyn Write, line: &str) -> Result<(), ExecutionError> {
    writeln!(status, "{GNUPG_PREFIX}{line}")
        .and_then(|()| status.flush())
        .map_err(|_| ExecutionError::new(EXIT_REQUEST_FAILED, "status output is unavailable"))
}

fn emit_error(stderr: &mut dyn Write, error: ExecutionError) -> i32 {
    if writeln!(stderr, "error: {}", error.message)
        .and_then(|()| stderr.flush())
        .is_err()
    {
        EXIT_REQUEST_FAILED
    } else {
        error.exit_code
    }
}

fn effective_status_fd(invocation: &Invocation) -> Option<i32> {
    match (&invocation.mode, invocation.status_fd) {
        (Mode::Sign { .. }, Some(1)) => None,
        (_, status_fd) => status_fd,
    }
}

#[cfg(unix)]
fn validate_status_fd(status_fd: Option<i32>) -> Result<(), ()> {
    match status_fd {
        Some(status_fd) => {
            // This only inspects Git's inherited descriptor, so an invalid status channel
            // fails before any protected credential is read.
            if unsafe { libc::fcntl(status_fd, libc::F_GETFD) } == -1 {
                Err(())
            } else {
                Ok(())
            }
        }
        None => Ok(()),
    }
}

#[cfg(not(unix))]
fn validate_status_fd(_status_fd: Option<i32>) -> Result<(), ()> {
    Ok(())
}

#[cfg(unix)]
fn write_status(status_fd: Option<i32>, status: &[u8]) -> io::Result<()> {
    match status_fd {
        Some(status_fd) => {
            // Git owns this validated inherited descriptor; `ManuallyDrop` prevents this
            // adapter from closing it after the status bytes are written.
            let mut file = ManuallyDrop::new(unsafe { fs::File::from_raw_fd(status_fd) });
            file.write_all(status)?;
            file.flush()
        }
        None => io::stderr()
            .write_all(status)
            .and_then(|()| io::stderr().flush()),
    }
}

#[cfg(not(unix))]
fn write_status(_status_fd: Option<i32>, status: &[u8]) -> io::Result<()> {
    io::stderr()
        .write_all(status)
        .and_then(|()| io::stderr().flush())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use git_credential_nostr::{HelperConfigError, StoredCredential};
    use zeroize::Zeroizing;

    use super::*;

    #[derive(Default)]
    struct MemoryConfig(BTreeMap<String, Vec<String>>);

    impl HelperConfig for MemoryConfig {
        fn values(&self, key: &str) -> Result<Vec<String>, HelperConfigError> {
            Ok(self.0.get(key).cloned().unwrap_or_default())
        }
    }

    struct MemoryStore {
        result: Mutex<Option<Result<Option<StoredCredential>, CredentialStoreError>>>,
        reads: AtomicUsize,
    }

    impl MemoryStore {
        fn new(result: Result<Option<StoredCredential>, CredentialStoreError>) -> Self {
            Self {
                result: Mutex::new(Some(result)),
                reads: AtomicUsize::new(0),
            }
        }
    }

    impl CredentialStore for MemoryStore {
        fn read(
            &self,
            _credential_identifier: &str,
        ) -> Result<Option<StoredCredential>, CredentialStoreError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.result
                .lock()
                .expect("store lock")
                .take()
                .expect("one store read")
        }
    }

    #[derive(Default)]
    struct MemorySignatures(BTreeMap<String, String>);

    impl SignatureReader for MemorySignatures {
        fn read(&self, path: &str) -> Result<String, SignatureReadError> {
            self.0.get(path).cloned().ok_or(SignatureReadError::Missing)
        }
    }

    struct FixedClock(u32);

    impl Clock for FixedClock {
        fn unix_timestamp(&self) -> Result<u32, ()> {
            Ok(self.0)
        }
    }

    fn fixture() -> (String, MemoryConfig, MemoryStore) {
        let secret = [3; 32];
        let secret_key = SecretKey::from_slice(&secret).expect("secret key");
        let keypair = Keypair::from_secret_key(&Secp256k1::signing_only(), &secret_key);
        let public_key = keypair.x_only_public_key().0.to_string();
        let identifier =
            format!("{CREDENTIAL_IDENTIFIER_PREFIX}community/account/profile/{public_key}");
        let config = MemoryConfig(BTreeMap::from([
            (CONFIG_CREDENTIAL_IDENTIFIER.to_owned(), vec![identifier]),
            (CONFIG_SIGNING_KEY.to_owned(), vec![public_key.clone()]),
        ]));
        let store = MemoryStore::new(Ok(Some(StoredCredential::new(
            public_key.clone(),
            Zeroizing::new(secret.to_vec()),
        ))));
        (public_key, config, store)
    }

    fn sign_arguments(public_key: &str) -> Vec<String> {
        vec!["--status-fd=2".into(), "-bsau".into(), public_key.into()]
    }

    fn verify_arguments() -> Vec<String> {
        vec![
            "--status-fd=1".into(),
            "--verify".into(),
            "signature.asc".into(),
            "-".into(),
        ]
    }

    #[test]
    fn sign_and_verify_use_canonical_nip_gs_and_git_status_contracts() {
        let payload = b"tree 0123456789abcdef\n\nmessage";
        let (public_key, config, store) = fixture();
        let mut signature = Vec::new();
        let mut sign_status = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                &sign_arguments(&public_key),
                payload,
                &config,
                &store,
                &MemorySignatures::default(),
                &FixedClock(1_700_000_000),
                &mut signature,
                &mut sign_status,
                &mut stderr,
            ),
            EXIT_SUCCESS
        );
        assert!(stderr.is_empty());
        let armored = std::str::from_utf8(&signature).expect("armored signature");
        let envelope = GitSignatureEnvelope::parse_armored(armored).expect("canonical envelope");
        assert_eq!(envelope.signer.to_hex(), public_key);
        assert!(envelope.verify(payload).is_ok());
        assert_eq!(
            sign_status,
            format!(
                "[GNUPG:] BEGIN_SIGNING\n[GNUPG:] SIG_CREATED D 8 1 00 1700000000 {public_key}\n"
            )
            .as_bytes()
        );

        let signatures =
            MemorySignatures(BTreeMap::from([("signature.asc".into(), armored.into())]));
        let verify_store = MemoryStore::new(Err(CredentialStoreError::Unavailable));
        let mut stdout = Vec::new();
        let mut verify_status = Vec::new();
        assert_eq!(
            run_with(
                &verify_arguments(),
                payload,
                &config,
                &verify_store,
                &signatures,
                &FixedClock(0),
                &mut stdout,
                &mut verify_status,
                &mut stderr,
            ),
            EXIT_SUCCESS
        );
        assert_eq!(verify_store.reads.load(Ordering::SeqCst), 0);
        let status = std::str::from_utf8(&verify_status).expect("verification status");
        assert!(status.contains(&format!("[GNUPG:] GOODSIG {public_key} {public_key}\n")));
        assert!(status.contains("[GNUPG:] TRUST_FULLY 0 shell\n"));
        assert!(status.contains("[GNUPG:] NOTATION_DATA none\n"));
    }

    #[test]
    fn locked_keyring_has_exact_exit_and_redacted_error() {
        let (public_key, config, _store) = fixture();
        let store = MemoryStore::new(Err(CredentialStoreError::Locked));
        let mut stdout = Vec::new();
        let mut status = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                &sign_arguments(&public_key),
                b"payload",
                &config,
                &store,
                &MemorySignatures::default(),
                &FixedClock(1),
                &mut stdout,
                &mut status,
                &mut stderr,
            ),
            EXIT_KEYRING_LOCKED
        );
        assert!(stdout.is_empty() && status.is_empty());
        assert_eq!(stderr, b"error: protected credential storage is locked\n");
    }

    #[test]
    fn altered_object_emits_badsig_and_verification_exit() {
        let payload = b"original";
        let (public_key, config, store) = fixture();
        let mut signature = Vec::new();
        assert_eq!(
            run_with(
                &sign_arguments(&public_key),
                payload,
                &config,
                &store,
                &MemorySignatures::default(),
                &FixedClock(42),
                &mut signature,
                &mut Vec::new(),
                &mut Vec::new(),
            ),
            EXIT_SUCCESS
        );
        let signatures = MemorySignatures(BTreeMap::from([(
            "signature.asc".into(),
            String::from_utf8(signature).expect("signature UTF-8"),
        )]));
        let mut status = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                &verify_arguments(),
                b"altered",
                &config,
                &MemoryStore::new(Err(CredentialStoreError::Unavailable)),
                &signatures,
                &FixedClock(0),
                &mut Vec::new(),
                &mut status,
                &mut stderr,
            ),
            EXIT_VERIFICATION_FAILED
        );
        assert_eq!(
            status,
            format!("[GNUPG:] NEWSIG\n[GNUPG:] BADSIG {public_key} {public_key}\n").as_bytes()
        );
        assert_eq!(stderr, b"error: git object signature is invalid\n");
    }

    #[test]
    fn invalid_credential_diagnostics_redact_secret_identifier_and_key() {
        let (public_key, mut config, _store) = fixture();
        let identifier = config.0[CONFIG_CREDENTIAL_IDENTIFIER][0].clone();
        let secret_marker = b"private-key-marker".to_vec();
        let store = MemoryStore::new(Ok(Some(StoredCredential::new(
            public_key.clone(),
            Zeroizing::new(secret_marker.clone()),
        ))));
        config
            .0
            .insert(CONFIG_SIGNING_KEY.into(), vec![public_key.clone()]);
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                &sign_arguments(&public_key),
                b"payload",
                &config,
                &store,
                &MemorySignatures::default(),
                &FixedClock(1),
                &mut Vec::new(),
                &mut Vec::new(),
                &mut stderr,
            ),
            EXIT_SIGNING_FAILED
        );
        let error = std::str::from_utf8(&stderr).expect("error UTF-8");
        assert_eq!(error, "error: active signing credential is invalid\n");
        assert!(!error.contains(&identifier));
        assert!(!error.contains(&public_key));
        assert!(
            !error
                .as_bytes()
                .windows(secret_marker.len())
                .any(|window| window == secret_marker)
        );
    }

    #[test]
    fn exact_exit_codes_cover_invocation_missing_and_invalid_credentials() {
        let (public_key, config, _store) = fixture();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                &["--verify".into(), "signature.asc".into()],
                b"payload",
                &config,
                &MemoryStore::new(Ok(None)),
                &MemorySignatures::default(),
                &FixedClock(1),
                &mut Vec::new(),
                &mut Vec::new(),
                &mut stderr,
            ),
            EXIT_REQUEST_FAILED
        );
        stderr.clear();
        assert_eq!(
            run_with(
                &sign_arguments(&public_key),
                b"payload",
                &config,
                &MemoryStore::new(Ok(None)),
                &MemorySignatures::default(),
                &FixedClock(1),
                &mut Vec::new(),
                &mut Vec::new(),
                &mut stderr,
            ),
            EXIT_CREDENTIAL_MISSING
        );
        assert_eq!(stderr, b"error: active signing credential is unavailable\n");
    }
}
