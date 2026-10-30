use std::ffi::OsString;
use std::fmt;

use clap::{Parser, Subcommand, error::ErrorKind};
use nostr_compat::pairing::{PairingQr, PairingRelayUrl, PairingSessionState};
use serde_json::{Value, json};

const MAX_PAIRING_SESSION_IDENTIFIER_BYTES: usize = 128;

#[derive(Parser)]
#[command(name = "pair", about = "Manage NIP-AB device pairing sessions")]
struct PairingArguments {
    #[command(subcommand)]
    command: PairingSubcommand,
}

#[derive(Subcommand)]
enum PairingSubcommand {
    /// Create a source session and print its QR URI.
    #[command(visible_alias = "source")]
    Create {
        /// Relay WebSocket URL. Repeat to advertise up to four relays.
        #[arg(long = "relay", required = true)]
        relays: Vec<String>,
    },
    /// Receive and import an identity from a QR URI read on standard input.
    #[command(visible_alias = "target")]
    Receive {
        /// Override the first relay encoded in the QR URI.
        #[arg(long)]
        relay: Option<String>,
    },
    /// Cancel an active pairing session.
    Cancel { session_id: String },
    /// Print the current state of a pairing session.
    Status { session_id: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairingCliSessionId(String);

impl PairingCliSessionId {
    pub fn parse(value: impl Into<String>) -> Result<Self, clap::Error> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_PAIRING_SESSION_IDENTIFIER_BYTES
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(value_error("invalid pairing session identifier"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub enum PairingCliCommand {
    Create {
        relays: Vec<PairingRelayUrl>,
    },
    Receive {
        qr: PairingQr,
        relay_override: Option<PairingRelayUrl>,
    },
    Cancel {
        session_id: PairingCliSessionId,
    },
    Status {
        session_id: PairingCliSessionId,
    },
}

impl PairingCliCommand {
    pub const fn verb(&self) -> &'static str {
        match self {
            Self::Create { .. } => "create",
            Self::Receive { .. } => "receive",
            Self::Cancel { .. } => "cancel",
            Self::Status { .. } => "status",
        }
    }

    pub fn relays(&self) -> Option<&[PairingRelayUrl]> {
        match self {
            Self::Create { relays } => Some(relays),
            _ => None,
        }
    }

    pub fn receive_qr(&self) -> Option<&PairingQr> {
        match self {
            Self::Receive { qr, .. } => Some(qr),
            _ => None,
        }
    }

    pub fn relay_override(&self) -> Option<&PairingRelayUrl> {
        match self {
            Self::Receive { relay_override, .. } => relay_override.as_ref(),
            _ => None,
        }
    }

    pub fn session_id(&self) -> Option<&PairingCliSessionId> {
        match self {
            Self::Cancel { session_id } | Self::Status { session_id } => Some(session_id),
            _ => None,
        }
    }
}

impl fmt::Debug for PairingCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Create { relays } => formatter
                .debug_struct("PairingCliCommand")
                .field("verb", &"create")
                .field("relay_count", &relays.len())
                .finish(),
            Self::Receive { relay_override, .. } => formatter
                .debug_struct("PairingCliCommand")
                .field("verb", &"receive")
                .field("qr", &"[REDACTED]")
                .field("relay_override", &relay_override)
                .finish(),
            Self::Cancel { session_id } | Self::Status { session_id } => formatter
                .debug_struct("PairingCliCommand")
                .field("verb", &self.verb())
                .field("session_id", session_id)
                .finish(),
        }
    }
}

pub fn parse_pairing_command<I, T>(
    arguments: I,
    standard_input: &str,
) -> Result<PairingCliCommand, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let arguments = PairingArguments::try_parse_from(arguments)?;
    match arguments.command {
        PairingSubcommand::Create { relays } => {
            let relays = relays
                .into_iter()
                .map(|relay| {
                    PairingRelayUrl::parse(relay)
                        .map_err(|_| value_error("invalid pairing relay URL"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if relays.len() > nostr_compat::pairing::MAX_NIP_AB_RELAYS {
                return Err(value_error("too many pairing relay URLs"));
            }
            Ok(PairingCliCommand::Create { relays })
        }
        PairingSubcommand::Receive { relay } => {
            let qr = PairingQr::parse(standard_input.trim())
                .map_err(|_| value_error("invalid NIP-AB QR URI on standard input"))?;
            let relay_override = relay
                .map(PairingRelayUrl::parse)
                .transpose()
                .map_err(|_| value_error("invalid pairing relay override"))?;
            Ok(PairingCliCommand::Receive { qr, relay_override })
        }
        PairingSubcommand::Cancel { session_id } => Ok(PairingCliCommand::Cancel {
            session_id: PairingCliSessionId::parse(session_id)?,
        }),
        PairingSubcommand::Status { session_id } => Ok(PairingCliCommand::Status {
            session_id: PairingCliSessionId::parse(session_id)?,
        }),
    }
}

fn value_error(message: &'static str) -> clap::Error {
    clap::Error::raw(ErrorKind::ValueValidation, message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairingCliImportDisposition {
    Imported,
    AlreadyPresent,
}

pub enum PairingCliOutcome {
    Created {
        session_id: PairingCliSessionId,
        qr: PairingQr,
        state: PairingSessionState,
        expires_at_millis: u64,
    },
    Received {
        session_id: PairingCliSessionId,
        credential_identifier: String,
        public_key: [u8; 32],
        disposition: PairingCliImportDisposition,
    },
    Cancelled {
        session_id: PairingCliSessionId,
    },
    Status {
        session_id: PairingCliSessionId,
        state: PairingSessionState,
        expires_at_millis: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PairingCliError {
    #[error("pairing request is invalid")]
    InvalidRequest,
    #[error("pairing service is unavailable")]
    Unavailable,
    #[error("pairing request is not authorized")]
    AuthorizationDenied,
    #[error("pairing session expired")]
    Expired,
    #[error("pairing session changed or is already terminal")]
    Conflict,
    #[error("paired identity could not be imported")]
    ImportFailed,
    #[error("pairing service returned an invalid response")]
    InvalidBackendResponse,
}

impl PairingCliError {
    pub const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "pairing_invalid_request",
            Self::Unavailable => "pairing_unavailable",
            Self::AuthorizationDenied => "pairing_authorization_denied",
            Self::Expired => "pairing_expired",
            Self::Conflict => "pairing_conflict",
            Self::ImportFailed => "pairing_import_failed",
            Self::InvalidBackendResponse => "pairing_invalid_backend_response",
        }
    }
}

pub trait PairingCliExecutor {
    fn execute(&self, command: PairingCliCommand) -> Result<PairingCliOutcome, PairingCliError>;
}

#[derive(Clone, Eq, PartialEq)]
pub struct PairingCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl PairingCliExecution {
    fn success(value: Value) -> Self {
        Self {
            stdout: format!("{value}\n"),
            stderr: String::new(),
            exit_code: 0,
        }
    }

    fn failure(value: Value, exit_code: i32) -> Self {
        Self {
            stdout: String::new(),
            stderr: format!("{value}\n"),
            exit_code,
        }
    }
}

impl fmt::Debug for PairingCliExecution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PairingCliExecution")
            .field("stdout_bytes", &self.stdout.len())
            .field("stderr_bytes", &self.stderr.len())
            .field("exit_code", &self.exit_code)
            .finish()
    }
}

pub fn execute_pairing_command(
    executor: &impl PairingCliExecutor,
    command: PairingCliCommand,
) -> PairingCliExecution {
    let verb = command.verb();
    match executor.execute(command) {
        Ok(outcome) => match success_output(verb, outcome) {
            Ok(output) => PairingCliExecution::success(output),
            Err(error) => error_output(verb, error),
        },
        Err(error) => error_output(verb, error),
    }
}

fn success_output(
    verb: &'static str,
    outcome: PairingCliOutcome,
) -> Result<Value, PairingCliError> {
    match (verb, outcome) {
        (
            "create",
            PairingCliOutcome::Created {
                session_id,
                qr,
                state,
                expires_at_millis,
            },
        ) => Ok(json!({
            "command": verb,
            "expires_at_millis": expires_at_millis,
            "ok": true,
            "qr_uri": qr.encode().map_err(|_| PairingCliError::InvalidBackendResponse)?,
            "session_id": session_id.as_str(),
            "state": session_state(state),
        })),
        (
            "receive",
            PairingCliOutcome::Received {
                session_id,
                credential_identifier,
                public_key,
                disposition,
            },
        ) => Ok(json!({
            "command": verb,
            "credential_identifier": credential_identifier,
            "disposition": import_disposition(disposition),
            "ok": true,
            "public_key": hex_bytes(public_key),
            "session_id": session_id.as_str(),
            "state": "completed",
        })),
        ("cancel", PairingCliOutcome::Cancelled { session_id }) => Ok(json!({
            "command": verb,
            "ok": true,
            "session_id": session_id.as_str(),
            "state": "aborted",
        })),
        (
            "status",
            PairingCliOutcome::Status {
                session_id,
                state,
                expires_at_millis,
            },
        ) => Ok(json!({
            "command": verb,
            "expires_at_millis": expires_at_millis,
            "ok": true,
            "session_id": session_id.as_str(),
            "state": session_state(state),
        })),
        _ => Err(PairingCliError::InvalidBackendResponse),
    }
}

fn error_output(verb: &'static str, error: PairingCliError) -> PairingCliExecution {
    let (category, retryable, exit_code) = error_contract(error);
    PairingCliExecution::failure(
        json!({
            "command": verb,
            "error": category,
            "error_code": error.diagnostic_code(),
            "ok": false,
            "retryable": retryable,
        }),
        exit_code,
    )
}

const fn error_contract(error: PairingCliError) -> (&'static str, bool, i32) {
    match error {
        PairingCliError::InvalidRequest => ("user_error", false, 1),
        PairingCliError::Unavailable => ("service_unavailable", true, 2),
        PairingCliError::AuthorizationDenied => ("authorization_denied", false, 3),
        PairingCliError::InvalidBackendResponse => ("service_error", false, 4),
        PairingCliError::Expired | PairingCliError::Conflict => ("conflict", false, 5),
        PairingCliError::ImportFailed => ("partial_failure", false, 2),
    }
}

const fn session_state(state: PairingSessionState) -> &'static str {
    match state {
        PairingSessionState::WaitingOffer => "waiting_offer",
        PairingSessionState::AwaitingSourceConfirmation => "awaiting_source_confirmation",
        PairingSessionState::AwaitingSasConfirm => "awaiting_sas_confirm",
        PairingSessionState::AwaitingTargetConfirmation => "awaiting_target_confirmation",
        PairingSessionState::Transferring => "transferring",
        PairingSessionState::PayloadSent => "payload_sent",
        PairingSessionState::PayloadReceived => "payload_received",
        PairingSessionState::Completed => "completed",
        PairingSessionState::Aborted => "aborted",
    }
}

const fn import_disposition(disposition: PairingCliImportDisposition) -> &'static str {
    match disposition {
        PairingCliImportDisposition::Imported => "imported",
        PairingCliImportDisposition::AlreadyPresent => "already_present",
    }
}

fn hex_bytes(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct FixedExecutor {
        outcome: Mutex<Option<Result<PairingCliOutcome, PairingCliError>>>,
        command_debug: Mutex<Option<String>>,
    }

    impl FixedExecutor {
        fn new(outcome: Result<PairingCliOutcome, PairingCliError>) -> Self {
            Self {
                outcome: Mutex::new(Some(outcome)),
                command_debug: Mutex::new(None),
            }
        }

        fn command_debug(&self) -> String {
            self.command_debug
                .lock()
                .expect("command debug lock")
                .clone()
                .expect("executed command")
        }
    }

    impl PairingCliExecutor for FixedExecutor {
        fn execute(
            &self,
            command: PairingCliCommand,
        ) -> Result<PairingCliOutcome, PairingCliError> {
            self.command_debug
                .lock()
                .expect("command debug lock")
                .replace(format!("{command:?}"));
            self.outcome
                .lock()
                .expect("outcome lock")
                .take()
                .expect("one execution")
        }
    }

    fn qr() -> PairingQr {
        PairingQr::parse(concat!(
            "nostrpair://79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798?",
            "secret=0707070707070707070707070707070707070707070707070707070707070707&",
            "relay=ws%3A%2F%2F127.0.0.1%3A8080%2Fpair&v=1"
        ))
        .expect("fixture QR")
    }

    fn session(value: &str) -> PairingCliSessionId {
        PairingCliSessionId::parse(value).expect("fixture session")
    }

    #[test]
    fn create_and_source_alias_keep_golden_machine_output() {
        for verb in ["create", "source"] {
            let command =
                parse_pairing_command(["pair", verb, "--relay", "ws://127.0.0.1:8080/pair"], "")
                    .expect("create syntax");
            assert_eq!(command.verb(), "create");
            assert_eq!(command.relays().map(|relays| relays.len()), Some(1));

            let qr = qr();
            let qr_uri = qr.encode().expect("fixture QR URI");
            let executor = FixedExecutor::new(Ok(PairingCliOutcome::Created {
                session_id: session("pair-a"),
                qr,
                state: PairingSessionState::WaitingOffer,
                expires_at_millis: 120_000,
            }));
            let execution = execute_pairing_command(&executor, command);

            assert_eq!(execution.exit_code, 0);
            assert_eq!(execution.stderr, "");
            assert_eq!(
                execution.stdout,
                format!(
                    "{{\"command\":\"create\",\"expires_at_millis\":120000,\"ok\":true,\"qr_uri\":\"{qr_uri}\",\"session_id\":\"pair-a\",\"state\":\"waiting_offer\"}}\n"
                )
            );
        }
    }

    #[test]
    fn receive_and_target_alias_parse_canonical_qr_and_report_verified_import() {
        let qr_uri = qr().encode().expect("fixture QR URI");
        for verb in ["receive", "target"] {
            let command = parse_pairing_command(
                ["pair", verb, "--relay", "wss://relay.example.com/pair"],
                &qr_uri,
            )
            .expect("receive syntax");
            assert_eq!(command.verb(), "receive");
            assert!(command.receive_qr().is_some());
            assert_eq!(
                command.relay_override().map(PairingRelayUrl::as_str),
                Some("wss://relay.example.com/pair")
            );
            let executor = FixedExecutor::new(Ok(PairingCliOutcome::Received {
                session_id: session("pair-b"),
                credential_identifier: "zed-nostr://credential/v1/imported".into(),
                public_key: [0xab; 32],
                disposition: PairingCliImportDisposition::Imported,
            }));

            let execution = execute_pairing_command(&executor, command);

            assert_eq!(execution.exit_code, 0);
            assert_eq!(
                execution.stdout,
                concat!(
                    "{\"command\":\"receive\",",
                    "\"credential_identifier\":\"zed-nostr://credential/v1/imported\",",
                    "\"disposition\":\"imported\",\"ok\":true,",
                    "\"public_key\":\"abababababababababababababababababababababababababababababababab\",",
                    "\"session_id\":\"pair-b\",\"state\":\"completed\"}\n"
                )
            );
        }
    }

    #[test]
    fn expiry_has_closed_golden_error_and_conflict_exit() {
        let command =
            parse_pairing_command(["pair", "status", "pair-expired"], "").expect("status syntax");
        let executor = FixedExecutor::new(Err(PairingCliError::Expired));

        let execution = execute_pairing_command(&executor, command);

        assert_eq!(execution.exit_code, 5);
        assert_eq!(execution.stdout, "");
        assert_eq!(
            execution.stderr,
            "{\"command\":\"status\",\"error\":\"conflict\",\"error_code\":\"pairing_expired\",\"ok\":false,\"retryable\":false}\n"
        );
    }

    #[test]
    fn cancel_has_golden_terminal_output() {
        let command =
            parse_pairing_command(["pair", "cancel", "pair-cancel"], "").expect("cancel syntax");
        let executor = FixedExecutor::new(Ok(PairingCliOutcome::Cancelled {
            session_id: session("pair-cancel"),
        }));

        let execution = execute_pairing_command(&executor, command);

        assert_eq!(execution.exit_code, 0);
        assert_eq!(
            execution.stdout,
            "{\"command\":\"cancel\",\"ok\":true,\"session_id\":\"pair-cancel\",\"state\":\"aborted\"}\n"
        );
    }

    #[test]
    fn status_has_golden_canonical_state_output() {
        let command =
            parse_pairing_command(["pair", "status", "pair-status"], "").expect("status syntax");
        let executor = FixedExecutor::new(Ok(PairingCliOutcome::Status {
            session_id: session("pair-status"),
            state: PairingSessionState::AwaitingTargetConfirmation,
            expires_at_millis: 220_000,
        }));

        let execution = execute_pairing_command(&executor, command);

        assert_eq!(execution.exit_code, 0);
        assert_eq!(
            execution.stdout,
            "{\"command\":\"status\",\"expires_at_millis\":220000,\"ok\":true,\"session_id\":\"pair-status\",\"state\":\"awaiting_target_confirmation\"}\n"
        );
    }

    #[test]
    fn qr_and_identity_secrets_never_reach_debug_or_error_output() {
        let qr = qr();
        let qr_uri = qr.encode().expect("fixture QR URI");
        let qr_secret = qr_uri
            .split("secret=")
            .nth(1)
            .and_then(|suffix| suffix.split('&').next())
            .expect("QR secret");
        let command = parse_pairing_command(["pair", "receive"], &qr_uri).expect("receive syntax");
        let executor = FixedExecutor::new(Err(PairingCliError::ImportFailed));

        let execution = execute_pairing_command(&executor, command);
        let debug = format!("{:?} {:?}", executor.command_debug(), execution);

        assert_eq!(execution.exit_code, 2);
        assert!(!execution.stdout.contains(qr_secret));
        assert!(!execution.stderr.contains(qr_secret));
        assert!(!debug.contains(qr_secret));
        assert_eq!(
            execution.stderr,
            "{\"command\":\"receive\",\"error\":\"partial_failure\",\"error_code\":\"pairing_import_failed\",\"ok\":false,\"retryable\":false}\n"
        );

        let identity_secret = "nsec1identity-secret-sentinel";
        assert!(!execution.stdout.contains(identity_secret));
        assert!(!execution.stderr.contains(identity_secret));
        assert!(!debug.contains(identity_secret));
    }
}
