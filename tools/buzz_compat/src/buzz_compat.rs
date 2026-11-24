use std::{collections::BTreeMap, fmt, io};

pub const BUZZ_CLI_VERSION: Version = Version::new(0, 1, 0);
pub const SHIM_PROTOCOL_VERSION: u32 = 1;

const DEFAULT_RELAY_URL: &str = "http://localhost:3000";
const MISSING_KEY_MESSAGE: &str =
    "auth error: BUZZ_PRIVATE_KEY is required (use --private-key or set env var)";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl Version {
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn parse(value: &str) -> Result<Self, ShimError> {
        let mut parts = value.split('.');
        let major = parse_version_part(parts.next())?;
        let minor = parse_version_part(parts.next())?;
        let patch = parse_version_part(parts.next())?;
        if parts.next().is_some() {
            return Err(ShimError::InvalidVersion);
        }
        Ok(Self::new(major, minor, patch))
    }
}

fn parse_version_part(value: Option<&str>) -> Result<u64, ShimError> {
    let value = value.ok_or(ShimError::InvalidVersion)?;
    if value.is_empty() || (value.len() > 1 && value.starts_with('0')) {
        return Err(ShimError::InvalidVersion);
    }
    value.parse().map_err(|_| ShimError::InvalidVersion)
}

impl fmt::Display for Version {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EndpointCompatibility {
    pub protocol_version: u32,
    pub minimum_buzz_cli_version: Version,
}

impl Default for EndpointCompatibility {
    fn default() -> Self {
        Self {
            protocol_version: SHIM_PROTOCOL_VERSION,
            minimum_buzz_cli_version: BUZZ_CLI_VERSION,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShimEnvironment(BTreeMap<String, String>);

impl ShimEnvironment {
    pub fn from_variables(values: impl IntoIterator<Item = (String, String)>) -> Self {
        Self(values.into_iter().collect())
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ForwardRequest {
    pub program: String,
    pub args: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

impl fmt::Debug for ForwardRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ForwardRequest")
            .field("program", &self.program)
            .field("arg_count", &self.args.len())
            .field("environment_keys", &self.environment.keys())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShimExecution {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
}

impl ShimExecution {
    fn stdout(value: impl Into<Vec<u8>>) -> Self {
        Self {
            stdout: value.into(),
            stderr: Vec::new(),
            exit_code: 0,
        }
    }

    fn error(category: &str, message: &str, retryable: bool, exit_code: i32) -> Self {
        let envelope = format!(
            "{{\"error\":\"{}\",\"message\":\"{}\",\"retryable\":{}}}\n",
            escape_json(category),
            escape_json(message),
            retryable,
        );
        Self {
            stdout: Vec::new(),
            stderr: envelope.into_bytes(),
            exit_code,
        }
    }
}

pub trait ForwardRunner {
    fn run(&self, request: ForwardRequest) -> io::Result<ShimExecution>;
}

pub fn execute(
    runner: &impl ForwardRunner,
    program: impl Into<String>,
    args: &[String],
    environment: &ShimEnvironment,
    compatibility: EndpointCompatibility,
) -> ShimExecution {
    if args.is_empty() || args == ["--help"] || args == ["-h"] {
        return ShimExecution::stdout(help_text().as_bytes().to_vec());
    }
    if args == ["--version"] || args == ["-V"] {
        return ShimExecution::stdout(
            format!(
                "buzz {} (Zed compatibility shim protocol {})\n",
                BUZZ_CLI_VERSION, SHIM_PROTOCOL_VERSION
            )
            .into_bytes(),
        );
    }
    if compatibility.protocol_version != SHIM_PROTOCOL_VERSION
        || BUZZ_CLI_VERSION < compatibility.minimum_buzz_cli_version
    {
        return minimum_version_error(compatibility);
    }

    let invocation = match LegacyInvocation::parse(args, environment) {
        Ok(invocation) => invocation,
        Err(error) => return error.execution(),
    };
    if invocation.requires_authentication() && invocation.private_key.is_none() {
        return ShimExecution::error("auth_error", MISSING_KEY_MESSAGE, false, 3);
    }
    let request = invocation.forward_request(program.into());
    match runner.run(request) {
        Ok(execution) => execution,
        Err(_) => ShimExecution::error(
            "error",
            "failed to execute the Zed collaboration CLI",
            false,
            4,
        ),
    }
}

fn minimum_version_error(compatibility: EndpointCompatibility) -> ShimExecution {
    let message = if compatibility.protocol_version != SHIM_PROTOCOL_VERSION {
        format!(
            "unsupported collaboration CLI protocol {}; this shim supports protocol {}",
            compatibility.protocol_version, SHIM_PROTOCOL_VERSION
        )
    } else {
        format!(
            "buzz CLI {} is unsupported; minimum version is {}",
            BUZZ_CLI_VERSION, compatibility.minimum_buzz_cli_version
        )
    };
    ShimExecution::error("upgrade_required", &message, false, 1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShimError {
    InvalidVersion,
    Usage(&'static str),
    UnknownGroup,
    UnknownCommand,
}

impl ShimError {
    fn execution(self) -> ShimExecution {
        let message = match self {
            Self::InvalidVersion => "invalid compatibility version",
            Self::Usage(message) => message,
            Self::UnknownGroup => "unknown Buzz command group",
            Self::UnknownCommand => "unknown Buzz subcommand",
        };
        ShimExecution::error("user_error", message, false, 1)
    }
}

struct LegacyInvocation {
    relay_url: String,
    output_format: String,
    private_key: Option<String>,
    auth_tag: Option<String>,
    owner: &'static str,
    group: String,
    command: Vec<String>,
    tail: Vec<String>,
}

impl LegacyInvocation {
    fn parse(args: &[String], environment: &ShimEnvironment) -> Result<Self, ShimError> {
        let mut relay_url = environment
            .get("BUZZ_RELAY_URL")
            .unwrap_or(DEFAULT_RELAY_URL)
            .to_owned();
        let mut output_format = "json".to_owned();
        let mut private_key = environment.get("BUZZ_PRIVATE_KEY").map(str::to_owned);
        let mut auth_tag = environment.get("BUZZ_AUTH_TAG").map(str::to_owned);
        let mut positional = Vec::new();
        let mut index = 0;
        while index < args.len() {
            let argument = &args[index];
            if let Some((name, value)) = argument.split_once('=') {
                match name {
                    "--relay" => relay_url = require_value(value)?.to_owned(),
                    "--format" => output_format = parse_output_format(value)?.to_owned(),
                    "--private-key" => private_key = Some(require_value(value)?.to_owned()),
                    "--auth-tag" => auth_tag = Some(require_value(value)?.to_owned()),
                    _ => positional.push(argument.clone()),
                }
                index += 1;
                continue;
            }
            match argument.as_str() {
                "--relay" | "--format" | "--private-key" | "--auth-tag" => {
                    let value = args
                        .get(index + 1)
                        .ok_or(ShimError::Usage("global option requires a value"))?;
                    match argument.as_str() {
                        "--relay" => relay_url = require_value(value)?.to_owned(),
                        "--format" => output_format = parse_output_format(value)?.to_owned(),
                        "--private-key" => private_key = Some(require_value(value)?.to_owned()),
                        "--auth-tag" => auth_tag = Some(require_value(value)?.to_owned()),
                        _ => return Err(ShimError::Usage("unsupported global option")),
                    }
                    index += 2;
                }
                _ => {
                    positional.push(argument.clone());
                    index += 1;
                }
            }
        }

        let group = positional
            .first()
            .ok_or(ShimError::Usage("missing Buzz command group"))?;
        let contract = group_contract(group).ok_or(ShimError::UnknownGroup)?;
        let command = contract
            .commands
            .iter()
            .find_map(|candidate| {
                let parts = candidate.split(' ').collect::<Vec<_>>();
                (positional.len() > parts.len()
                    && positional[1..=parts.len()]
                        .iter()
                        .map(String::as_str)
                        .eq(parts.iter().copied()))
                .then_some(parts)
            })
            .ok_or(ShimError::UnknownCommand)?;
        let tail = positional[command.len() + 1..].to_vec();
        Ok(Self {
            relay_url,
            output_format,
            private_key,
            auth_tag,
            owner: contract.owner,
            group: group.clone(),
            command: command.into_iter().map(str::to_owned).collect(),
            tail,
        })
    }

    fn requires_authentication(&self) -> bool {
        self.group != "pack"
    }

    fn forward_request(self, program: String) -> ForwardRequest {
        let mut args = vec![
            "collaboration".to_owned(),
            "--legacy-client".to_owned(),
            "buzz-cli".to_owned(),
            "--legacy-client-version".to_owned(),
            BUZZ_CLI_VERSION.to_string(),
            "--protocol-version".to_owned(),
            SHIM_PROTOCOL_VERSION.to_string(),
            "--relay".to_owned(),
            self.relay_url,
            "--format".to_owned(),
            self.output_format,
            self.owner.to_owned(),
            self.group,
        ];
        args.extend(self.command);
        args.extend(self.tail);
        let mut environment = BTreeMap::new();
        if let Some(private_key) = self.private_key {
            environment.insert("ZED_COLLABORATION_PRIVATE_KEY".to_owned(), private_key);
        }
        if let Some(auth_tag) = self.auth_tag {
            environment.insert("ZED_COLLABORATION_AUTH_TAG".to_owned(), auth_tag);
        }
        ForwardRequest {
            program,
            args,
            environment,
        }
    }
}

fn require_value(value: &str) -> Result<&str, ShimError> {
    if value.is_empty() {
        Err(ShimError::Usage("global option value must not be empty"))
    } else {
        Ok(value)
    }
}

fn parse_output_format(value: &str) -> Result<&str, ShimError> {
    match value {
        "json" | "compact" => Ok(value),
        _ => Err(ShimError::Usage("--format must be json or compact")),
    }
}

struct GroupContract {
    name: &'static str,
    owner: &'static str,
    commands: &'static [&'static str],
}

const GROUPS: &[GroupContract] = &[
    GroupContract {
        name: "agents",
        owner: "agents",
        commands: &[
            "draft-create",
            "draft-update",
            "archive",
            "unarchive",
            "archived",
        ],
    },
    GroupContract {
        name: "canvas",
        owner: "channels",
        commands: &["get", "set"],
    },
    GroupContract {
        name: "channels",
        owner: "channels",
        commands: &[
            "list",
            "get",
            "search",
            "create",
            "update",
            "topic",
            "purpose",
            "join",
            "leave",
            "archive",
            "unarchive",
            "delete",
            "members",
            "add-member",
            "remove-member",
            "set-add-policy",
        ],
    },
    GroupContract {
        name: "dms",
        owner: "social",
        commands: &["list", "open", "add-member", "hide"],
    },
    GroupContract {
        name: "emoji",
        owner: "social",
        commands: &["list", "set", "rm", "export", "import"],
    },
    GroupContract {
        name: "feed",
        owner: "social",
        commands: &["get"],
    },
    GroupContract {
        name: "issues",
        owner: "review",
        commands: &["create", "get", "list", "status"],
    },
    GroupContract {
        name: "media",
        owner: "media",
        commands: &["get"],
    },
    GroupContract {
        name: "mem",
        owner: "agents",
        commands: &["ls", "get", "hash", "set", "patch", "rm"],
    },
    GroupContract {
        name: "messages",
        owner: "messages",
        commands: &[
            "send",
            "send-diff",
            "edit",
            "delete",
            "get",
            "thread",
            "search",
            "vote",
        ],
    },
    GroupContract {
        name: "moderation",
        owner: "moderation",
        commands: &[
            "reports",
            "resolve",
            "ban",
            "unban",
            "timeout",
            "untimeout",
            "restricted",
            "audit",
        ],
    },
    GroupContract {
        name: "notes",
        owner: "social",
        commands: &["set", "get", "ls", "rm"],
    },
    GroupContract {
        name: "pack",
        owner: "agents",
        commands: &["validate", "inspect"],
    },
    GroupContract {
        name: "patches",
        owner: "review",
        commands: &["send", "get", "list", "status"],
    },
    GroupContract {
        name: "pr",
        owner: "review",
        commands: &["open", "update", "get", "list", "status"],
    },
    GroupContract {
        name: "projects",
        owner: "git",
        commands: &[
            "create",
            "get",
            "list",
            "add-repo",
            "remove-repo",
            "update",
            "delete",
        ],
    },
    GroupContract {
        name: "reactions",
        owner: "messages",
        commands: &["add", "remove", "get"],
    },
    GroupContract {
        name: "repos",
        owner: "git",
        commands: &[
            "create",
            "get",
            "list",
            "bind",
            "protect list",
            "protect set",
            "protect remove",
        ],
    },
    GroupContract {
        name: "social",
        owner: "social",
        commands: &[
            "publish",
            "set-contacts",
            "event",
            "notes",
            "contacts",
            "set-list",
            "list",
        ],
    },
    GroupContract {
        name: "upload",
        owner: "media",
        commands: &["file"],
    },
    GroupContract {
        name: "users",
        owner: "community",
        commands: &[
            "get",
            "set-profile",
            "presence",
            "set-presence",
            "set-status",
        ],
    },
    GroupContract {
        name: "workflows",
        owner: "workflows",
        commands: &[
            "list", "get", "create", "update", "delete", "trigger", "runs", "approve",
        ],
    },
];

fn group_contract(name: &str) -> Option<&'static GroupContract> {
    GROUPS.iter().find(|contract| contract.name == name)
}

pub fn help_text() -> String {
    let groups = GROUPS
        .iter()
        .map(|contract| contract.name)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Buzz CLI — interact with a Buzz relay\n\nCompatibility shim for Zed collaboration commands.\n\nGroups: {groups}\n\nExit codes: 0=ok  1=bad input  2=relay/network error  3=auth error  4=other  5=write conflict\n"
    )
}

fn escape_json(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => escaped.push_str("\\uFFFD"),
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, io};

    use super::*;

    struct RecordingRunner {
        request: RefCell<Option<ForwardRequest>>,
        result: ShimExecution,
    }

    impl RecordingRunner {
        fn returning(result: ShimExecution) -> Self {
            Self {
                request: RefCell::new(None),
                result,
            }
        }
    }

    impl ForwardRunner for RecordingRunner {
        fn run(&self, request: ForwardRequest) -> io::Result<ShimExecution> {
            self.request.replace(Some(request));
            Ok(self.result.clone())
        }
    }

    fn environment(private_key: bool) -> ShimEnvironment {
        ShimEnvironment::from_variables(private_key.then(|| {
            (
                "BUZZ_PRIVATE_KEY".to_owned(),
                "private-key-value".to_owned(),
            )
        }))
    }

    fn success() -> ShimExecution {
        ShimExecution {
            stdout: b"{\"ok\":true}\n".to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
        }
    }

    #[test]
    fn help_and_version_preserve_the_frozen_surface() {
        let runner = RecordingRunner::returning(success());
        let help = execute(
            &runner,
            "zed",
            &["--help".into()],
            &environment(false),
            EndpointCompatibility::default(),
        );
        let help = String::from_utf8(help.stdout).expect("help text");
        assert!(help.contains("Buzz CLI — interact with a Buzz relay"));
        assert!(help.contains("Exit codes: 0=ok"));
        for group in GROUPS {
            assert!(help.contains(group.name));
        }

        let version = execute(
            &runner,
            "zed",
            &["--version".into()],
            &environment(false),
            EndpointCompatibility::default(),
        );
        assert_eq!(
            String::from_utf8(version.stdout).expect("version"),
            "buzz 0.1.0 (Zed compatibility shim protocol 1)\n"
        );
        assert!(runner.request.borrow().is_none());
    }

    #[test]
    fn every_frozen_command_translates_to_its_canonical_owner() {
        for group in GROUPS {
            for command in group.commands {
                let runner = RecordingRunner::returning(success());
                let mut args = vec![group.name.to_owned()];
                args.extend(command.split(' ').map(str::to_owned));
                args.push("--fixture".into());
                let output = execute(
                    &runner,
                    "zed-test",
                    &args,
                    &environment(group.name != "pack"),
                    EndpointCompatibility::default(),
                );
                assert_eq!(output.exit_code, 0, "{} {command}", group.name);
                let request = runner.request.borrow();
                let request = request.as_ref().expect("forwarded request");
                assert_eq!(request.program, "zed-test");
                let owner_index = request
                    .args
                    .iter()
                    .position(|value| value == group.owner)
                    .expect("owner");
                assert_eq!(
                    request.args.get(owner_index + 1).map(String::as_str),
                    Some(group.name)
                );
                assert!(request.args.ends_with(&["--fixture".to_owned()]));
            }
        }
    }

    #[test]
    fn globals_are_normalized_and_secrets_move_to_child_environment() {
        let runner = RecordingRunner::returning(success());
        let args = [
            "--relay=wss://relay.example.com",
            "channels",
            "list",
            "--private-key",
            "secret-value",
            "--auth-tag=auth-secret",
            "--format",
            "compact",
        ]
        .map(str::to_owned);
        let output = execute(
            &runner,
            "zed",
            &args,
            &environment(false),
            EndpointCompatibility::default(),
        );
        assert_eq!(output.exit_code, 0);
        let request = runner.request.borrow();
        let request = request.as_ref().expect("request");
        assert!(!request.args.iter().any(|value| value == "secret-value"));
        assert!(!request.args.iter().any(|value| value == "auth-secret"));
        assert_eq!(
            request
                .environment
                .get("ZED_COLLABORATION_PRIVATE_KEY")
                .map(String::as_str),
            Some("secret-value")
        );
        assert_eq!(
            request
                .environment
                .get("ZED_COLLABORATION_AUTH_TAG")
                .map(String::as_str),
            Some("auth-secret")
        );
        assert!(
            request
                .args
                .windows(2)
                .any(|values| values[0] == "--format" && values[1] == "compact")
        );
        assert!(!format!("{request:?}").contains("secret-value"));
    }

    #[test]
    fn missing_auth_and_minimum_version_errors_are_exact_and_local() {
        let runner = RecordingRunner::returning(success());
        let missing = execute(
            &runner,
            "zed",
            &["channels".into(), "list".into()],
            &environment(false),
            EndpointCompatibility::default(),
        );
        assert_eq!(missing.exit_code, 3);
        assert_eq!(
            String::from_utf8(missing.stderr).expect("error"),
            format!(
                "{{\"error\":\"auth_error\",\"message\":\"{MISSING_KEY_MESSAGE}\",\"retryable\":false}}\n"
            )
        );

        let upgrade = execute(
            &runner,
            "zed",
            &["channels".into(), "list".into()],
            &environment(true),
            EndpointCompatibility {
                protocol_version: SHIM_PROTOCOL_VERSION,
                minimum_buzz_cli_version: Version::new(0, 2, 0),
            },
        );
        assert_eq!(upgrade.exit_code, 1);
        let error = String::from_utf8(upgrade.stderr).expect("error");
        assert!(error.contains("upgrade_required"));
        assert!(error.contains("minimum version is 0.2.0"));
        assert!(runner.request.borrow().is_none());
    }

    #[test]
    fn old_and_consolidated_endpoints_preserve_streams_and_exit_codes() {
        for expected in [
            ShimExecution {
                stdout: b"[{\"channel_id\":\"one\"}]\n".to_vec(),
                stderr: Vec::new(),
                exit_code: 0,
            },
            ShimExecution {
                stdout: Vec::new(),
                stderr: b"{\"error\":\"network_error\",\"message\":\"relay unavailable\",\"retryable\":true}\n".to_vec(),
                exit_code: 2,
            },
            ShimExecution {
                stdout: Vec::new(),
                stderr: b"{\"error\":\"delivery_unknown\",\"message\":\"completion unknown\",\"retryable\":false}\n".to_vec(),
                exit_code: 2,
            },
            ShimExecution {
                stdout: Vec::new(),
                stderr: b"{\"error\":\"conflict\",\"message\":\"stale version\",\"retryable\":false}\n".to_vec(),
                exit_code: 5,
            },
        ] {
            let old_endpoint = RecordingRunner::returning(expected.clone());
            let consolidated_endpoint = RecordingRunner::returning(expected.clone());
            let args = ["channels".into(), "list".into()];
            let old = execute(
                &old_endpoint,
                "legacy-endpoint-fixture",
                &args,
                &environment(true),
                EndpointCompatibility::default(),
            );
            let consolidated = execute(
                &consolidated_endpoint,
                "zed-endpoint-fixture",
                &args,
                &environment(true),
                EndpointCompatibility::default(),
            );
            assert_eq!(old, expected);
            assert_eq!(consolidated, expected);
        }
    }

    #[test]
    fn malformed_commands_fail_before_forwarding() {
        let runner = RecordingRunner::returning(success());
        for args in [
            vec!["unknown".into(), "list".into()],
            vec!["channels".into(), "unknown".into()],
            vec![
                "--format".into(),
                "yaml".into(),
                "channels".into(),
                "list".into(),
            ],
        ] {
            let output = execute(
                &runner,
                "zed",
                &args,
                &environment(true),
                EndpointCompatibility::default(),
            );
            assert_eq!(output.exit_code, 1);
            assert!(output.stdout.is_empty());
        }
        assert!(runner.request.borrow().is_none());
    }
}
