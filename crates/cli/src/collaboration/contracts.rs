#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollaborationOperationOwner {
    Agents,
    Channels,
    Community,
    Git,
    Media,
    Messages,
    Moderation,
    Review,
    Social,
    Workflows,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandGroupContract {
    pub legacy_name: &'static str,
    pub owner: CollaborationOperationOwner,
    pub commands: &'static [&'static str],
}

pub const COMMAND_GROUPS: &[CommandGroupContract] = &[
    CommandGroupContract {
        legacy_name: "agents",
        owner: CollaborationOperationOwner::Agents,
        commands: &[
            "draft-create",
            "draft-update",
            "archive",
            "unarchive",
            "archived",
        ],
    },
    CommandGroupContract {
        legacy_name: "canvas",
        owner: CollaborationOperationOwner::Channels,
        commands: &["get", "set"],
    },
    CommandGroupContract {
        legacy_name: "channels",
        owner: CollaborationOperationOwner::Channels,
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
    CommandGroupContract {
        legacy_name: "dms",
        owner: CollaborationOperationOwner::Social,
        commands: &["list", "open", "add-member", "hide"],
    },
    CommandGroupContract {
        legacy_name: "emoji",
        owner: CollaborationOperationOwner::Social,
        commands: &["list", "set", "rm", "export", "import"],
    },
    CommandGroupContract {
        legacy_name: "feed",
        owner: CollaborationOperationOwner::Social,
        commands: &["get"],
    },
    CommandGroupContract {
        legacy_name: "issues",
        owner: CollaborationOperationOwner::Review,
        commands: &["create", "get", "list", "status"],
    },
    CommandGroupContract {
        legacy_name: "media",
        owner: CollaborationOperationOwner::Media,
        commands: &["get"],
    },
    CommandGroupContract {
        legacy_name: "mem",
        owner: CollaborationOperationOwner::Agents,
        commands: &["ls", "get", "hash", "set", "patch", "rm"],
    },
    CommandGroupContract {
        legacy_name: "messages",
        owner: CollaborationOperationOwner::Messages,
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
    CommandGroupContract {
        legacy_name: "moderation",
        owner: CollaborationOperationOwner::Moderation,
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
    CommandGroupContract {
        legacy_name: "notes",
        owner: CollaborationOperationOwner::Social,
        commands: &["set", "get", "ls", "rm"],
    },
    CommandGroupContract {
        legacy_name: "pack",
        owner: CollaborationOperationOwner::Agents,
        commands: &["validate", "inspect"],
    },
    CommandGroupContract {
        legacy_name: "patches",
        owner: CollaborationOperationOwner::Review,
        commands: &["send", "get", "list", "status"],
    },
    CommandGroupContract {
        legacy_name: "pr",
        owner: CollaborationOperationOwner::Review,
        commands: &["open", "update", "get", "list", "status"],
    },
    CommandGroupContract {
        legacy_name: "projects",
        owner: CollaborationOperationOwner::Git,
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
    CommandGroupContract {
        legacy_name: "reactions",
        owner: CollaborationOperationOwner::Messages,
        commands: &["add", "remove", "get"],
    },
    CommandGroupContract {
        legacy_name: "repos",
        owner: CollaborationOperationOwner::Git,
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
    CommandGroupContract {
        legacy_name: "social",
        owner: CollaborationOperationOwner::Social,
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
    CommandGroupContract {
        legacy_name: "upload",
        owner: CollaborationOperationOwner::Media,
        commands: &["file"],
    },
    CommandGroupContract {
        legacy_name: "users",
        owner: CollaborationOperationOwner::Community,
        commands: &[
            "get",
            "set-profile",
            "presence",
            "set-presence",
            "set-status",
        ],
    },
    CommandGroupContract {
        legacy_name: "workflows",
        owner: CollaborationOperationOwner::Workflows,
        commands: &[
            "list", "get", "create", "update", "delete", "trigger", "runs", "approve",
        ],
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionRequirement {
    Optional,
    RelayCommands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalOptionContract {
    pub long_name: &'static str,
    pub environment_variable: Option<&'static str>,
    pub default_value: Option<&'static str>,
    pub accepted_values: &'static [&'static str],
    pub secret: bool,
    pub requirement: OptionRequirement,
}

pub const GLOBAL_OPTIONS: &[GlobalOptionContract] = &[
    GlobalOptionContract {
        long_name: "--relay",
        environment_variable: Some("BUZZ_RELAY_URL"),
        default_value: Some("http://localhost:3000"),
        accepted_values: &[],
        secret: false,
        requirement: OptionRequirement::Optional,
    },
    GlobalOptionContract {
        long_name: "--private-key",
        environment_variable: Some("BUZZ_PRIVATE_KEY"),
        default_value: None,
        accepted_values: &[],
        secret: true,
        requirement: OptionRequirement::RelayCommands,
    },
    GlobalOptionContract {
        long_name: "--auth-tag",
        environment_variable: Some("BUZZ_AUTH_TAG"),
        default_value: None,
        accepted_values: &[],
        secret: true,
        requirement: OptionRequirement::Optional,
    },
    GlobalOptionContract {
        long_name: "--format",
        environment_variable: None,
        default_value: Some("json"),
        accepted_values: &["json", "compact"],
        secret: false,
        requirement: OptionRequirement::Optional,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputStream {
    Stdout,
    Stderr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormatScope {
    AllSuccessfulCommands,
    ReadCommands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputFormatContract {
    pub name: &'static str,
    pub default: bool,
    pub scope: OutputFormatScope,
    pub normalized_json: bool,
    pub reduced_fields: bool,
}

pub const OUTPUT_FORMATS: &[OutputFormatContract] = &[
    OutputFormatContract {
        name: "json",
        default: true,
        scope: OutputFormatScope::AllSuccessfulCommands,
        normalized_json: true,
        reduced_fields: false,
    },
    OutputFormatContract {
        name: "compact",
        default: false,
        scope: OutputFormatScope::ReadCommands,
        normalized_json: true,
        reduced_fields: true,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitClass {
    Success = 0,
    UsageOrNotFound = 1,
    ServiceOrNetwork = 2,
    Authorization = 3,
    Unexpected = 4,
    Conflict = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorClass {
    Usage,
    NotFound,
    Relay { status: u16 },
    Network { retryable: bool },
    Authorization,
    Key,
    DeliveryUnknown,
    Unexpected,
    Conflict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ErrorContract {
    pub category: &'static str,
    pub exit_class: ExitClass,
    pub stream: OutputStream,
    pub retryable: bool,
}

pub fn error_contract(error: ErrorClass) -> ErrorContract {
    let (category, exit_class, retryable) = match error {
        ErrorClass::Usage => ("user_error", ExitClass::UsageOrNotFound, false),
        ErrorClass::NotFound => ("not_found", ExitClass::UsageOrNotFound, false),
        ErrorClass::Relay { status: 401 | 403 } => ("auth_error", ExitClass::Authorization, false),
        ErrorClass::Relay { status } => (
            "relay_error",
            ExitClass::ServiceOrNetwork,
            matches!(status, 429 | 502 | 503 | 504),
        ),
        ErrorClass::Network { retryable } => {
            ("network_error", ExitClass::ServiceOrNetwork, retryable)
        }
        ErrorClass::Authorization => ("auth_error", ExitClass::Authorization, false),
        ErrorClass::Key => ("key_error", ExitClass::Authorization, false),
        ErrorClass::DeliveryUnknown => ("delivery_unknown", ExitClass::ServiceOrNetwork, false),
        ErrorClass::Unexpected => ("error", ExitClass::Unexpected, false),
        ErrorClass::Conflict => ("conflict", ExitClass::Conflict, false),
    };

    ErrorContract {
        category,
        exit_class,
        stream: OutputStream::Stderr,
        retryable,
    }
}

pub const SUCCESS_STREAM: OutputStream = OutputStream::Stdout;
pub const HELP_STREAM: OutputStream = OutputStream::Stdout;
pub const ERROR_ENVELOPE_FIELDS: &[&str] = &["error", "message", "retryable"];

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, fs, path::PathBuf};

    use super::*;

    fn buzz_source_path(file: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../projects/buzz/crates/buzz-cli/src")
            .join(file)
    }

    fn read_buzz_source(file: &str) -> String {
        let path = buzz_source_path(file);
        fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!(
                "could not read frozen Buzz source {}: {error}",
                path.display()
            )
        })
    }

    fn client_fixture() -> serde_json::Value {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.agents/specs/collaborative-workspace/fixtures/clients/manifest.json");
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!(
                "could not read frozen client fixture {}: {error}",
                path.display()
            )
        });
        serde_json::from_str(&source).unwrap_or_else(|error| {
            panic!(
                "could not parse frozen client fixture {}: {error}",
                path.display()
            )
        })
    }

    fn fixture_contract<'a>(fixture: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
        fixture["contracts"]
            .as_array()
            .and_then(|contracts| contracts.iter().find(|contract| contract["id"] == id))
            .unwrap_or_else(|| panic!("missing frozen client contract {id}"))
    }

    fn kebab_case(identifier: &str) -> String {
        let mut output = String::new();
        for (index, character) in identifier.chars().enumerate() {
            if character.is_ascii_uppercase() && index != 0 {
                output.push('-');
            }
            output.push(character.to_ascii_lowercase());
        }
        output
    }

    fn enum_variants(source: &str, enum_name: &str) -> Vec<String> {
        let marker = format!("enum {enum_name} {{");
        let body = source
            .split_once(&marker)
            .unwrap_or_else(|| panic!("missing frozen enum {enum_name}"))
            .1;
        let mut command_name = None;
        let mut variants = Vec::new();

        for line in body.lines() {
            if line == "}" {
                break;
            }
            let trimmed = line.trim();
            if let Some(value) = trimmed
                .strip_prefix("#[command(name = \"")
                .and_then(|value| value.strip_suffix("\")]"))
            {
                command_name = Some(value.to_string());
                continue;
            }
            let Some(candidate) = line.strip_prefix("    ") else {
                continue;
            };
            if candidate.starts_with(' ')
                || !candidate.chars().next().is_some_and(char::is_uppercase)
            {
                continue;
            }
            let identifier = candidate
                .split(['(', '{', ',', ' '])
                .next()
                .expect("a variant line always has an identifier");
            variants.push(
                command_name
                    .take()
                    .unwrap_or_else(|| kebab_case(identifier)),
            );
        }
        variants
    }

    #[test]
    fn command_manifest_accounts_for_every_frozen_leaf_command() {
        let source = read_buzz_source("lib.rs");
        let mut frozen_groups = enum_variants(&source, "Cmd");
        let mut manifest_groups = COMMAND_GROUPS
            .iter()
            .map(|group| group.legacy_name)
            .collect::<Vec<_>>();
        frozen_groups.sort();
        manifest_groups.sort();
        assert_eq!(manifest_groups, frozen_groups);

        let mut paths = BTreeSet::new();
        for group in COMMAND_GROUPS {
            let enum_name = match group.legacy_name {
                "dms" => "DmsCmd".to_string(),
                "pr" => "PrCmd".to_string(),
                name => format!("{}Cmd", name[..1].to_ascii_uppercase() + &name[1..]),
            };
            let mut frozen_commands = enum_variants(&source, &enum_name);
            if group.legacy_name == "repos" {
                let protect_index = frozen_commands
                    .iter()
                    .position(|command| command == "protect")
                    .expect("repos protect remains a frozen nested command");
                frozen_commands.splice(
                    protect_index..=protect_index,
                    enum_variants(&source, "ReposProtectCmd")
                        .into_iter()
                        .map(|command| format!("protect {command}")),
                );
            }
            assert_eq!(
                group.commands, frozen_commands,
                "{} commands",
                group.legacy_name
            );
            for command in group.commands {
                assert!(paths.insert(format!("{} {command}", group.legacy_name)));
            }
        }
        assert_eq!(paths.len(), 113);
    }

    #[test]
    fn global_options_and_output_contract_match_the_frozen_parser() {
        let source = read_buzz_source("lib.rs");
        for option in GLOBAL_OPTIONS {
            assert!(source.contains(&format!(
                "{}: ",
                option.long_name.trim_start_matches("--").replace('-', "_")
            )));
            if let Some(environment_variable) = option.environment_variable {
                assert!(source.contains(&format!("env = \"{environment_variable}\"")));
            }
            if let Some(default_value) = option.default_value {
                assert!(source.contains(&format!("default_value = \"{default_value}\"")));
            }
            if option.secret {
                assert!(source.contains("hide_env_values = true"));
            }
        }
        assert_eq!(GLOBAL_OPTIONS.len(), 4);
        assert_eq!(OUTPUT_FORMATS.len(), 2);
        assert_eq!(
            OUTPUT_FORMATS
                .iter()
                .filter(|format| format.default)
                .count(),
            1
        );
        assert_eq!(GLOBAL_OPTIONS[3].accepted_values, ["json", "compact"]);
        assert_eq!(SUCCESS_STREAM, OutputStream::Stdout);
        assert_eq!(HELP_STREAM, OutputStream::Stdout);
        assert_eq!(ERROR_ENVELOPE_FIELDS, ["error", "message", "retryable"]);
    }

    #[test]
    fn manifest_matches_the_frozen_client_contracts() {
        let fixture = client_fixture();
        let inventory = fixture_contract(&fixture, "CLIENT-CLI-002");
        let fixture_groups = inventory["expected_output"]["groups"]
            .as_array()
            .expect("the frozen group inventory is an array")
            .iter()
            .map(|group| group.as_str().expect("a frozen group is a string"))
            .collect::<Vec<_>>();
        assert_eq!(
            COMMAND_GROUPS
                .iter()
                .map(|group| group.legacy_name)
                .collect::<Vec<_>>(),
            fixture_groups
        );

        let exits = &fixture_contract(&fixture, "CLIENT-CLI-004")["expected_output"];
        let cases = [
            ("usage", ErrorClass::Usage),
            ("not_found", ErrorClass::NotFound),
            ("network", ErrorClass::Network { retryable: true }),
            ("relay_non_auth", ErrorClass::Relay { status: 500 }),
            ("delivery_unknown", ErrorClass::DeliveryUnknown),
            ("auth", ErrorClass::Authorization),
            ("key", ErrorClass::Key),
            ("relay_401_403", ErrorClass::Relay { status: 401 }),
            ("other", ErrorClass::Unexpected),
            ("conflict", ErrorClass::Conflict),
        ];
        for (fixture_name, error) in cases {
            assert_eq!(
                i64::from(error_contract(error).exit_class as i32),
                exits[fixture_name]
            );
        }

        let envelope = &fixture_contract(&fixture, "CLIENT-CLI-005")["expected_output"];
        assert_eq!(
            envelope["format"]
                .as_object()
                .expect("the frozen error envelope is an object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            ERROR_ENVELOPE_FIELDS.iter().copied().collect()
        );
        assert_eq!(
            envelope["retryable_relay_statuses"],
            serde_json::json!([429, 502, 503, 504])
        );
        assert_eq!(envelope["delivery_unknown_retryable"], false);
    }

    #[test]
    fn every_frozen_error_has_the_retained_stream_exit_and_retry_contract() {
        let source = read_buzz_source("error.rs");
        for variant in [
            "Usage",
            "Relay",
            "Network",
            "Auth",
            "Key",
            "Conflict",
            "NotFound",
            "DeliveryUnknown",
            "Other",
        ] {
            assert!(source.contains(&format!("CliError::{variant}")));
        }

        let cases = [
            (ErrorClass::Usage, "user_error", 1, false),
            (ErrorClass::NotFound, "not_found", 1, false),
            (ErrorClass::Relay { status: 401 }, "auth_error", 3, false),
            (ErrorClass::Relay { status: 403 }, "auth_error", 3, false),
            (ErrorClass::Relay { status: 429 }, "relay_error", 2, true),
            (ErrorClass::Relay { status: 502 }, "relay_error", 2, true),
            (ErrorClass::Relay { status: 503 }, "relay_error", 2, true),
            (ErrorClass::Relay { status: 504 }, "relay_error", 2, true),
            (ErrorClass::Relay { status: 500 }, "relay_error", 2, false),
            (
                ErrorClass::Network { retryable: true },
                "network_error",
                2,
                true,
            ),
            (
                ErrorClass::Network { retryable: false },
                "network_error",
                2,
                false,
            ),
            (ErrorClass::Authorization, "auth_error", 3, false),
            (ErrorClass::Key, "key_error", 3, false),
            (ErrorClass::DeliveryUnknown, "delivery_unknown", 2, false),
            (ErrorClass::Unexpected, "error", 4, false),
            (ErrorClass::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let contract = error_contract(error);
            assert_eq!(contract.category, category);
            assert_eq!(contract.exit_class as i32, exit_code);
            assert_eq!(contract.stream, OutputStream::Stderr);
            assert_eq!(contract.retryable, retryable);
        }
        assert_eq!(ExitClass::Success as i32, 0);
    }
}
