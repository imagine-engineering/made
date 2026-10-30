use std::io::{self, Read, Write};

use base64::Engine as _;
use nostr::{
    EventBuilder, Keys, SecretKey, Tag,
    nips::nip98::{HttpData, HttpMethod},
    types::Url,
};
use zeroize::Zeroizing;

pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_REQUEST_REJECTED: i32 = 1;
pub const EXIT_KEYRING_LOCKED: i32 = 2;
pub const EXIT_CREDENTIAL_MISSING: i32 = 3;
pub const EXIT_SIGNING_FAILED: i32 = 4;

const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_CONFIG_BYTES: usize = 64 * 1024;
const MAX_FIELD_BYTES: usize = 4 * 1024;
const MAX_CREDENTIAL_IDENTIFIER_BYTES: usize = 1024;
const CREDENTIAL_IDENTIFIER_PREFIX: &str = "zed-nostr://credential/v1/";
const CONFIG_CREDENTIAL_IDENTIFIER: &str = "nostr.credentialIdentifier";
const CONFIG_ALLOWED_HOST: &str = "nostr.allowedHost";
const CONFIG_AUTH_TAG: &str = "nostr.authTag";

pub struct StoredCredential {
    username: String,
    secret: Zeroizing<Vec<u8>>,
}

impl StoredCredential {
    pub fn new(username: String, secret: Zeroizing<Vec<u8>>) -> Self {
        Self { username, secret }
    }

    pub fn username(&self) -> &str {
        &self.username
    }

    pub fn secret(&self) -> &[u8] {
        &self.secret
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CredentialStoreError {
    #[error("protected credential storage is locked")]
    Locked,
    #[error("protected credential storage is unavailable")]
    Unavailable,
}

pub trait CredentialStore {
    fn read(
        &self,
        credential_identifier: &str,
    ) -> Result<Option<StoredCredential>, CredentialStoreError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("Git configuration is unavailable")]
pub struct HelperConfigError;

pub trait HelperConfig {
    fn values(&self, key: &str) -> Result<Vec<String>, HelperConfigError>;
}

pub struct GitConfig;

impl HelperConfig for GitConfig {
    fn values(&self, key: &str) -> Result<Vec<String>, HelperConfigError> {
        let output = smol::block_on(
            smol::process::Command::new("git")
                .args(["config", "--get-all", key])
                .output(),
        )
        .map_err(|_| HelperConfigError)?;
        if !output.status.success() {
            return if output.status.code() == Some(1) {
                Ok(Vec::new())
            } else {
                Err(HelperConfigError)
            };
        }
        if output.stdout.len() > MAX_CONFIG_BYTES {
            return Err(HelperConfigError);
        }
        let output = std::str::from_utf8(&output.stdout).map_err(|_| HelperConfigError)?;
        output
            .lines()
            .map(|value| {
                validate_config_value(value)?;
                Ok(value.to_owned())
            })
            .collect()
    }
}

pub struct SystemCredentialStore;

impl CredentialStore for SystemCredentialStore {
    fn read(
        &self,
        credential_identifier: &str,
    ) -> Result<Option<StoredCredential>, CredentialStoreError> {
        platform::read_credential(credential_identifier)
    }
}

#[derive(Default)]
struct CredentialRequest {
    has_authtype_capability: bool,
    protocol: Option<String>,
    host: Option<String>,
    path: Option<String>,
    wwwauth: Option<String>,
}

pub fn run() -> i32 {
    let action = std::env::args().nth(1);
    let mut input = Vec::new();
    if io::stdin()
        .take(u64::try_from(MAX_INPUT_BYTES).unwrap_or(u64::MAX) + 1)
        .read_to_end(&mut input)
        .is_err()
        || input.len() > MAX_INPUT_BYTES
    {
        return emit_error(
            &mut io::stderr(),
            EXIT_REQUEST_REJECTED,
            "credential request is invalid",
        );
    }
    run_with(
        action.as_deref(),
        &input,
        &GitConfig,
        &SystemCredentialStore,
        &mut io::stdout(),
        &mut io::stderr(),
    )
}

pub fn run_with(
    action: Option<&str>,
    input: &[u8],
    config: &dyn HelperConfig,
    store: &dyn CredentialStore,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    match action {
        Some("get") | None => {}
        Some(_) => return EXIT_SUCCESS,
    }
    let request = match parse_request(input) {
        Ok(request) => request,
        Err(()) => {
            return emit_error(
                stderr,
                EXIT_REQUEST_REJECTED,
                "credential request is invalid",
            );
        }
    };
    if !request.has_authtype_capability {
        return write_silent_response(stdout);
    }
    let wwwauth = match request.wwwauth.as_deref() {
        Some(wwwauth) => wwwauth,
        None => return EXIT_SUCCESS,
    };
    let method = match parse_method(wwwauth) {
        Some(method) => method,
        None => return EXIT_SUCCESS,
    };
    let url = match build_repository_url(&request) {
        Ok(url) => url,
        Err(()) => {
            return emit_error(
                stderr,
                EXIT_REQUEST_REJECTED,
                "credential request is invalid",
            );
        }
    };
    let allowed_hosts = match config.values(CONFIG_ALLOWED_HOST) {
        Ok(values) => values,
        Err(_) => {
            return emit_error(
                stderr,
                EXIT_REQUEST_REJECTED,
                "credential configuration is unavailable",
            );
        }
    };
    let request_host = match request.host.as_deref() {
        Some(host) => host,
        None => {
            return emit_error(
                stderr,
                EXIT_REQUEST_REJECTED,
                "credential request is invalid",
            );
        }
    };
    if allowed_hosts.is_empty()
        || !allowed_hosts
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(request_host))
    {
        return emit_error(
            stderr,
            EXIT_REQUEST_REJECTED,
            "remote host is not authorized",
        );
    }
    let credential_identifiers = match config.values(CONFIG_CREDENTIAL_IDENTIFIER) {
        Ok(values) => values,
        Err(_) => {
            return emit_error(
                stderr,
                EXIT_REQUEST_REJECTED,
                "credential configuration is unavailable",
            );
        }
    };
    let credential_identifier = match credential_identifiers.as_slice() {
        [credential_identifier] if valid_credential_identifier(credential_identifier) => {
            credential_identifier
        }
        _ => {
            return emit_error(
                stderr,
                EXIT_REQUEST_REJECTED,
                "active signing credential is not configured",
            );
        }
    };
    let stored = match store.read(credential_identifier) {
        Ok(Some(stored)) => stored,
        Ok(None) => {
            return emit_error(
                stderr,
                EXIT_CREDENTIAL_MISSING,
                "active signing credential is unavailable",
            );
        }
        Err(CredentialStoreError::Locked) => {
            return emit_error(
                stderr,
                EXIT_KEYRING_LOCKED,
                "protected credential storage is locked",
            );
        }
        Err(CredentialStoreError::Unavailable) => {
            return emit_error(
                stderr,
                EXIT_KEYRING_LOCKED,
                "protected credential storage is unavailable",
            );
        }
    };
    let keys = match signing_keys(&stored) {
        Ok(keys) => keys,
        Err(()) => {
            return emit_error(
                stderr,
                EXIT_SIGNING_FAILED,
                "active signing credential is invalid",
            );
        }
    };
    let auth_tag = match load_auth_tag(config) {
        Ok(tag) => tag,
        Err(()) => {
            return emit_error(
                stderr,
                EXIT_REQUEST_REJECTED,
                "credential configuration is invalid",
            );
        }
    };
    let builder = EventBuilder::http_auth(HttpData::new(url, method));
    let builder = match auth_tag {
        Some(tag) => builder.tag(tag),
        None => builder,
    };
    let event = match builder.sign_with_keys(&keys) {
        Ok(event) => event,
        Err(_) => {
            return emit_error(stderr, EXIT_SIGNING_FAILED, "NIP-98 signing failed");
        }
    };
    let event = match serde_json::to_vec(&event) {
        Ok(event) => event,
        Err(_) => {
            return emit_error(stderr, EXIT_SIGNING_FAILED, "NIP-98 signing failed");
        }
    };
    let credential = base64::engine::general_purpose::STANDARD.encode(event);
    if writeln!(stdout, "capability[]=authtype")
        .and_then(|()| writeln!(stdout, "authtype=Nostr"))
        .and_then(|()| writeln!(stdout, "credential={credential}"))
        .and_then(|()| writeln!(stdout, "ephemeral=true"))
        .and_then(|()| writeln!(stdout, "quit=true"))
        .and_then(|()| writeln!(stdout))
        .and_then(|()| stdout.flush())
        .is_err()
    {
        return EXIT_REQUEST_REJECTED;
    }
    EXIT_SUCCESS
}

fn parse_request(input: &[u8]) -> Result<CredentialRequest, ()> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(());
    }
    let input = std::str::from_utf8(input).map_err(|_| ())?;
    let mut request = CredentialRequest::default();
    for line in input.lines() {
        if line.is_empty() {
            break;
        }
        if line.len() > MAX_FIELD_BYTES || line.chars().any(char::is_control) {
            return Err(());
        }
        if line == "capability[]=authtype" {
            request.has_authtype_capability = true;
        } else if let Some(value) = line.strip_prefix("protocol=") {
            set_once(&mut request.protocol, value)?;
        } else if let Some(value) = line.strip_prefix("host=") {
            set_once(&mut request.host, value)?;
        } else if let Some(value) = line.strip_prefix("path=") {
            set_once(&mut request.path, value)?;
        } else if let Some(value) = line.strip_prefix("wwwauth[]=")
            && value.starts_with("Nostr ")
            && request.wwwauth.is_none()
        {
            request.wwwauth = Some(value.to_owned());
        }
    }
    Ok(request)
}

fn set_once(target: &mut Option<String>, value: &str) -> Result<(), ()> {
    if target.is_some() || value.is_empty() {
        return Err(());
    }
    *target = Some(value.to_owned());
    Ok(())
}

fn parse_method(wwwauth: &str) -> Option<HttpMethod> {
    let parameters = wwwauth.strip_prefix("Nostr ")?;
    parameters.split(',').find_map(|parameter| {
        let parameter = parameter.trim();
        let value = parameter.strip_prefix("method=\"")?;
        let end = value.find('"')?;
        value[..end].parse().ok()
    })
}

fn build_repository_url(request: &CredentialRequest) -> Result<Url, ()> {
    let protocol = request.protocol.as_deref().ok_or(())?;
    if protocol != "https" && protocol != "http" {
        return Err(());
    }
    let host = request.host.as_deref().ok_or(())?;
    validate_config_value(host).map_err(|_| ())?;
    let mut url = Url::parse(&format!("{protocol}://{host}/")).map_err(|_| ())?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.host_str().is_none()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(());
    }
    let path = request.path.as_deref().ok_or(())?;
    let path = path.trim_start_matches('/');
    let repository_path = path
        .strip_suffix("/info/refs")
        .or_else(|| path.strip_suffix("/git-upload-pack"))
        .or_else(|| path.strip_suffix("/git-receive-pack"))
        .unwrap_or(path);
    if repository_path.is_empty()
        || repository_path.len() > MAX_FIELD_BYTES
        || repository_path.split('/').any(|segment| {
            segment.is_empty() || segment == "." || segment == ".." || segment.contains('\\')
        })
    {
        return Err(());
    }
    url.set_path(&format!("/{repository_path}"));
    Ok(url)
}

fn load_auth_tag(config: &dyn HelperConfig) -> Result<Option<Tag>, ()> {
    let values = config.values(CONFIG_AUTH_TAG).map_err(|_| ())?;
    match values.as_slice() {
        [] => Ok(None),
        [value] => {
            let parts: Vec<String> = serde_json::from_str(value).map_err(|_| ())?;
            if parts.len() != 4 || parts.first().map(String::as_str) != Some("auth") {
                return Err(());
            }
            Tag::parse(parts).map(Some).map_err(|_| ())
        }
        _ => Err(()),
    }
}

fn signing_keys(stored: &StoredCredential) -> Result<Keys, ()> {
    if stored.secret().len() != SecretKey::LEN
        || stored.username().len() != 64
        || !stored
            .username()
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(());
    }
    let secret = SecretKey::from_slice(stored.secret()).map_err(|_| ())?;
    let keys = Keys::new(secret);
    if !keys
        .public_key
        .to_hex()
        .eq_ignore_ascii_case(stored.username())
    {
        return Err(());
    }
    Ok(keys)
}

fn valid_credential_identifier(value: &str) -> bool {
    value.starts_with(CREDENTIAL_IDENTIFIER_PREFIX)
        && value.len() <= MAX_CREDENTIAL_IDENTIFIER_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn validate_config_value(value: &str) -> Result<(), HelperConfigError> {
    if value.is_empty()
        || value.len() > MAX_FIELD_BYTES
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(HelperConfigError);
    }
    Ok(())
}

fn write_silent_response(stdout: &mut dyn Write) -> i32 {
    if writeln!(stdout).and_then(|()| stdout.flush()).is_err() {
        EXIT_REQUEST_REJECTED
    } else {
        EXIT_SUCCESS
    }
}

fn emit_error(stderr: &mut dyn Write, exit_code: i32, message: &str) -> i32 {
    if writeln!(stderr, "error: {message}")
        .and_then(|()| stderr.flush())
        .is_err()
    {
        EXIT_REQUEST_REJECTED
    } else {
        exit_code
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ptr;

    use core_foundation::{
        base::{CFType, CFTypeRef, OSStatus, TCFType},
        boolean::CFBoolean,
        data::CFData,
        dictionary::{CFDictionary, CFDictionaryRef, CFMutableDictionary},
        string::{CFString, CFStringRef},
    };
    use zeroize::Zeroizing;

    use super::{CredentialStoreError, StoredCredential};

    pub fn read_credential(
        credential_identifier: &str,
    ) -> Result<Option<StoredCredential>, CredentialStoreError> {
        let identifier = CFString::from(credential_identifier);
        let cf_true = CFBoolean::true_value().as_CFTypeRef();
        unsafe {
            let mut attributes = CFMutableDictionary::with_capacity(5);
            attributes.set(kSecClass as *const _, kSecClassInternetPassword as *const _);
            attributes.set(kSecAttrServer as *const _, identifier.as_CFTypeRef());
            attributes.set(kSecReturnAttributes as *const _, cf_true);
            attributes.set(kSecReturnData as *const _, cf_true);
            let mut result: CFTypeRef = ptr::null();
            match SecItemCopyMatching(attributes.as_concrete_TypeRef(), &mut result) {
                ERR_SEC_SUCCESS => {}
                ERR_SEC_ITEM_NOT_FOUND => return Ok(None),
                ERR_SEC_USER_CANCELED | ERR_SEC_AUTH_FAILED | ERR_SEC_INTERACTION_NOT_ALLOWED => {
                    return Err(CredentialStoreError::Locked);
                }
                _ => return Err(CredentialStoreError::Unavailable),
            }
            let result = CFType::wrap_under_create_rule(result)
                .downcast::<CFDictionary>()
                .ok_or(CredentialStoreError::Unavailable)?;
            let username = result
                .find(kSecAttrAccount as *const _)
                .ok_or(CredentialStoreError::Unavailable)?;
            let username = CFType::wrap_under_get_rule(*username)
                .downcast::<CFString>()
                .ok_or(CredentialStoreError::Unavailable)?;
            let secret = result
                .find(kSecValueData as *const _)
                .ok_or(CredentialStoreError::Unavailable)?;
            let secret = CFType::wrap_under_get_rule(*secret)
                .downcast::<CFData>()
                .ok_or(CredentialStoreError::Unavailable)?;
            Ok(Some(StoredCredential::new(
                username.to_string(),
                Zeroizing::new(secret.bytes().to_vec()),
            )))
        }
    }

    #[link(name = "Security", kind = "framework")]
    unsafe extern "C" {
        static kSecClass: CFStringRef;
        static kSecClassInternetPassword: CFStringRef;
        static kSecAttrServer: CFStringRef;
        static kSecAttrAccount: CFStringRef;
        static kSecValueData: CFStringRef;
        static kSecReturnAttributes: CFStringRef;
        static kSecReturnData: CFStringRef;

        fn SecItemCopyMatching(query: CFDictionaryRef, result: *mut CFTypeRef) -> OSStatus;
    }

    const ERR_SEC_SUCCESS: OSStatus = 0;
    const ERR_SEC_USER_CANCELED: OSStatus = -128;
    const ERR_SEC_AUTH_FAILED: OSStatus = -25293;
    const ERR_SEC_ITEM_NOT_FOUND: OSStatus = -25300;
    const ERR_SEC_INTERACTION_NOT_ALLOWED: OSStatus = -25308;
}

#[cfg(target_os = "linux")]
mod platform {
    use zeroize::Zeroizing;

    use super::{CredentialStoreError, StoredCredential};

    const KEYRING_LABEL: &str = "zed-github-account";

    pub fn read_credential(
        credential_identifier: &str,
    ) -> Result<Option<StoredCredential>, CredentialStoreError> {
        smol::block_on(async {
            let keyring = oo7::Keyring::new()
                .await
                .map_err(|_| CredentialStoreError::Unavailable)?;
            keyring
                .unlock()
                .await
                .map_err(|_| CredentialStoreError::Locked)?;
            let items = keyring
                .search_items(&vec![("url", credential_identifier)])
                .await
                .map_err(|_| CredentialStoreError::Unavailable)?;
            for item in items {
                if !item.label().await.is_ok_and(|label| label == KEYRING_LABEL) {
                    continue;
                }
                let attributes = item
                    .attributes()
                    .await
                    .map_err(|_| CredentialStoreError::Unavailable)?;
                let username = attributes
                    .get("username")
                    .ok_or(CredentialStoreError::Unavailable)?
                    .to_owned();
                item.unlock()
                    .await
                    .map_err(|_| CredentialStoreError::Locked)?;
                let secret = item
                    .secret()
                    .await
                    .map_err(|_| CredentialStoreError::Unavailable)?;
                return Ok(Some(StoredCredential::new(
                    username,
                    Zeroizing::new(secret.to_vec()),
                )));
            }
            Ok(None)
        })
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use std::{ptr, slice};

    use windows::{
        Win32::{
            Foundation::{ERROR_ACCESS_DENIED, ERROR_CANCELLED, ERROR_NOT_FOUND},
            Security::Credentials::{CRED_TYPE_GENERIC, CREDENTIALW, CredFree, CredReadW},
        },
        core::{PCWSTR, PWSTR},
    };
    use zeroize::Zeroizing;

    use super::{CredentialStoreError, StoredCredential};

    pub fn read_credential(
        credential_identifier: &str,
    ) -> Result<Option<StoredCredential>, CredentialStoreError> {
        let target = format!("zed:url={credential_identifier}");
        let target = target.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
        let mut credentials: *mut CREDENTIALW = ptr::null_mut();
        if let Err(error) = unsafe {
            CredReadW(
                PCWSTR::from_raw(target.as_ptr()),
                CRED_TYPE_GENERIC,
                None,
                &mut credentials,
            )
        } {
            return if error.code() == ERROR_NOT_FOUND.to_hresult() {
                Ok(None)
            } else if error.code() == ERROR_ACCESS_DENIED.to_hresult()
                || error.code() == ERROR_CANCELLED.to_hresult()
            {
                Err(CredentialStoreError::Locked)
            } else {
                Err(CredentialStoreError::Unavailable)
            };
        }
        if credentials.is_null() {
            return Ok(None);
        }
        let username = unsafe { PWSTR::from_raw((*credentials).UserName.0).to_string() };
        let secret = unsafe {
            slice::from_raw_parts(
                (*credentials).CredentialBlob,
                (*credentials).CredentialBlobSize as usize,
            )
            .to_vec()
        };
        unsafe { CredFree(credentials.cast()) };
        let username = username.map_err(|_| CredentialStoreError::Unavailable)?;
        Ok(Some(StoredCredential::new(
            username,
            Zeroizing::new(secret),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use nostr::{Event, Keys, SecretKey};
    use zeroize::Zeroizing;

    use super::*;

    #[derive(Default)]
    struct MemoryConfig {
        values: BTreeMap<String, Vec<String>>,
    }

    impl MemoryConfig {
        fn valid(public_key: &str) -> Self {
            Self {
                values: BTreeMap::from([
                    (
                        CONFIG_ALLOWED_HOST.to_owned(),
                        vec!["git.example.test".to_owned()],
                    ),
                    (
                        CONFIG_CREDENTIAL_IDENTIFIER.to_owned(),
                        vec![format!(
                            "{CREDENTIAL_IDENTIFIER_PREFIX}community/account/profile/{public_key}"
                        )],
                    ),
                ]),
            }
        }
    }

    impl HelperConfig for MemoryConfig {
        fn values(&self, key: &str) -> Result<Vec<String>, HelperConfigError> {
            Ok(self.values.get(key).cloned().unwrap_or_default())
        }
    }

    struct MemoryStore {
        result: Mutex<Option<Result<Option<StoredCredential>, CredentialStoreError>>>,
        reads: AtomicUsize,
    }

    impl MemoryStore {
        fn with(result: Result<Option<StoredCredential>, CredentialStoreError>) -> Self {
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

    fn fixture() -> (MemoryConfig, MemoryStore, String) {
        let secret = [1; 32];
        let keys = Keys::new(SecretKey::from_slice(&secret).expect("secret"));
        let public_key = keys.public_key.to_hex();
        (
            MemoryConfig::valid(&public_key),
            MemoryStore::with(Ok(Some(StoredCredential::new(
                public_key.clone(),
                Zeroizing::new(secret.to_vec()),
            )))),
            public_key,
        )
    }

    fn request(host: &str) -> Vec<u8> {
        format!(
            "capability[]=authtype\nprotocol=https\nhost={host}\npath=git/owner/repository/info/refs\nwwwauth[]=Nostr method=\"GET\", realm=\"zed\"\n\n"
        )
        .into_bytes()
    }

    #[test]
    fn canonical_lookup_emits_a_valid_nip98_credential() {
        let (config, store, public_key) = fixture();
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                Some("get"),
                &request("git.example.test"),
                &config,
                &store,
                &mut stdout,
                &mut stderr,
            ),
            EXIT_SUCCESS
        );
        assert!(stderr.is_empty());
        let output = std::str::from_utf8(&stdout).expect("helper output");
        assert!(output.starts_with("capability[]=authtype\nauthtype=Nostr\n"));
        let credential = output
            .lines()
            .find_map(|line| line.strip_prefix("credential="))
            .expect("credential");
        let event: Event = serde_json::from_slice(
            &base64::engine::general_purpose::STANDARD
                .decode(credential)
                .expect("base64 event"),
        )
        .expect("NIP-98 event");
        assert_eq!(event.pubkey.to_hex(), public_key);
        event.verify().expect("valid signature");
        let event_json = serde_json::to_value(event).expect("event JSON");
        assert!(event_json["tags"].as_array().is_some_and(|tags| {
            tags.iter().any(|tag| {
                tag.as_array().is_some_and(|values| {
                    values.first().and_then(|value| value.as_str()) == Some("u")
                        && values.get(1).and_then(|value| value.as_str())
                            == Some("https://git.example.test/git/owner/repository")
                })
            })
        }));
    }

    #[test]
    fn locked_keyring_has_exact_sanitized_exit_contract() {
        let keys = Keys::new(SecretKey::from_slice(&[1; 32]).expect("secret"));
        let config = MemoryConfig::valid(&keys.public_key.to_hex());
        let store = MemoryStore::with(Err(CredentialStoreError::Locked));
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                Some("get"),
                &request("git.example.test"),
                &config,
                &store,
                &mut stdout,
                &mut stderr,
            ),
            EXIT_KEYRING_LOCKED
        );
        assert!(stdout.is_empty());
        assert_eq!(stderr, b"error: protected credential storage is locked\n");
    }

    #[test]
    fn denied_host_never_reads_the_keyring() {
        let (config, store, _public_key) = fixture();
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                Some("get"),
                &request("attacker.example"),
                &config,
                &store,
                &mut stdout,
                &mut stderr,
            ),
            EXIT_REQUEST_REJECTED
        );
        assert_eq!(store.reads.load(Ordering::SeqCst), 0);
        assert!(stdout.is_empty());
        assert_eq!(stderr, b"error: remote host is not authorized\n");
    }

    #[test]
    fn invalid_credential_redacts_identifier_and_secret() {
        let secret_marker = b"private-key-marker".to_vec();
        let public_key = "11".repeat(32);
        let config = MemoryConfig::valid(&public_key);
        let identifier = config.values[CONFIG_CREDENTIAL_IDENTIFIER][0].clone();
        let store = MemoryStore::with(Ok(Some(StoredCredential::new(
            public_key,
            Zeroizing::new(secret_marker.clone()),
        ))));
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                Some("get"),
                &request("git.example.test"),
                &config,
                &store,
                &mut stdout,
                &mut stderr,
            ),
            EXIT_SIGNING_FAILED
        );
        assert!(stdout.is_empty());
        let error = std::str::from_utf8(&stderr).expect("error output");
        assert_eq!(error, "error: active signing credential is invalid\n");
        assert!(!error.contains(&identifier));
        assert!(
            !error
                .as_bytes()
                .windows(secret_marker.len())
                .any(|value| value == secret_marker)
        );
    }

    #[test]
    fn exact_exit_codes_cover_missing_key_and_request_fallthrough() {
        let keys = Keys::new(SecretKey::from_slice(&[1; 32]).expect("secret"));
        let config = MemoryConfig::valid(&keys.public_key.to_hex());
        let store = MemoryStore::with(Ok(None));
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        assert_eq!(
            run_with(
                Some("get"),
                &request("git.example.test"),
                &config,
                &store,
                &mut stdout,
                &mut stderr,
            ),
            EXIT_CREDENTIAL_MISSING
        );
        assert_eq!(stderr, b"error: active signing credential is unavailable\n");
        stdout.clear();
        stderr.clear();
        assert_eq!(
            run_with(
                Some("store"),
                b"not parsed",
                &config,
                &MemoryStore::with(Err(CredentialStoreError::Unavailable)),
                &mut stdout,
                &mut stderr,
            ),
            EXIT_SUCCESS
        );
        assert!(stdout.is_empty() && stderr.is_empty());
        assert_eq!(
            run_with(
                Some("get"),
                b"capability[]=authtype\nprotocol=https\nhost=git.example.test\npath=git/owner/repository\n\n",
                &config,
                &MemoryStore::with(Err(CredentialStoreError::Unavailable)),
                &mut stdout,
                &mut stderr,
            ),
            EXIT_SUCCESS
        );
    }
}
