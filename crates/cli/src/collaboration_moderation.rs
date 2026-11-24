use std::fmt;

use collab::admin::moderation::{
    ModerationOperatorCommand, ModerationOperatorError, ModerationOperatorOutcome,
    ModerationWriteReceipt,
};
use collaboration_domain::{
    ModerationReport, ModerationReportReason, ModerationReportState, ModerationReportTarget,
    ModerationResolution,
};
use serde_json::{Value, json};

use crate::collaboration::contracts::{ErrorClass, error_contract};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModerationCliVerb {
    Report,
    List,
    Resolve,
    Ban,
    Unban,
    Timeout,
    Untimeout,
    ArchiveIdentity,
    ArchiveCommunity,
}

impl ModerationCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Report => "report",
            Self::List => "list",
            Self::Resolve => "resolve",
            Self::Ban => "ban",
            Self::Unban => "unban",
            Self::Timeout => "timeout",
            Self::Untimeout => "untimeout",
            Self::ArchiveIdentity => "archive_identity",
            Self::ArchiveCommunity => "archive_community",
        }
    }
}

pub struct ModerationCliCommand {
    verb: ModerationCliVerb,
    command: ModerationOperatorCommand,
}

impl ModerationCliCommand {
    pub const fn new(command: ModerationOperatorCommand) -> Self {
        let verb = match command {
            ModerationOperatorCommand::ListReports { .. } => ModerationCliVerb::List,
            ModerationOperatorCommand::FileReport { .. } => ModerationCliVerb::Report,
            ModerationOperatorCommand::ResolveReport { .. } => ModerationCliVerb::Resolve,
            ModerationOperatorCommand::ApplyBan { .. } => ModerationCliVerb::Ban,
            ModerationOperatorCommand::LiftBan { .. } => ModerationCliVerb::Unban,
            ModerationOperatorCommand::ApplyTimeout { .. } => ModerationCliVerb::Timeout,
            ModerationOperatorCommand::LiftTimeout { .. } => ModerationCliVerb::Untimeout,
            ModerationOperatorCommand::ArchiveIdentity { .. } => ModerationCliVerb::ArchiveIdentity,
            ModerationOperatorCommand::ArchiveCommunity { .. } => {
                ModerationCliVerb::ArchiveCommunity
            }
        };
        Self { verb, command }
    }

    pub const fn verb(&self) -> &'static str {
        self.verb.as_str()
    }
}

impl fmt::Debug for ModerationCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModerationCliCommand")
            .field("verb", &self.verb.as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

pub trait ModerationCliExecutor {
    fn execute(
        &self,
        command: ModerationOperatorCommand,
    ) -> Result<ModerationOperatorOutcome, ModerationOperatorError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModerationCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl ModerationCliExecution {
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

pub fn execute_moderation_command(
    executor: &impl ModerationCliExecutor,
    command: ModerationCliCommand,
) -> ModerationCliExecution {
    let verb = command.verb;
    match executor.execute(command.command) {
        Ok(outcome) => match success_output(verb, outcome) {
            Some(output) => ModerationCliExecution::success(output),
            None => error_output(
                verb,
                "moderation_cli_invalid_outcome",
                ErrorClass::Unexpected,
            ),
        },
        Err(error) => error_output(verb, error.diagnostic_code(), common_error_class(error)),
    }
}

fn error_output(
    verb: ModerationCliVerb,
    error_code: &'static str,
    error_class: ErrorClass,
) -> ModerationCliExecution {
    let contract = error_contract(error_class);
    ModerationCliExecution::failure(
        json!({
            "command": verb.as_str(),
            "error": contract.category,
            "error_code": error_code,
            "message": error_code,
            "ok": false,
            "retryable": contract.retryable,
        }),
        contract.exit_class as i32,
    )
}

fn success_output(verb: ModerationCliVerb, outcome: ModerationOperatorOutcome) -> Option<Value> {
    match (verb, outcome) {
        (ModerationCliVerb::List, ModerationOperatorOutcome::Reports(reports)) => Some(json!({
            "command": verb.as_str(),
            "ok": true,
            "reports": reports.iter().map(report_output).collect::<Vec<_>>(),
        })),
        (ModerationCliVerb::List, ModerationOperatorOutcome::Applied(_))
        | (_, ModerationOperatorOutcome::Reports(_)) => None,
        (_, ModerationOperatorOutcome::Applied(receipt)) => Some(write_output(verb, receipt)),
    }
}

fn write_output(verb: ModerationCliVerb, receipt: ModerationWriteReceipt) -> Value {
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "version": receipt.version,
    })
}

fn report_output(report: &ModerationReport) -> Value {
    let fields = report.fields();
    json!({
        "community_id": fields.community_id,
        "filed_at_millis": fields.filed_source.occurred_at_millis,
        "filed_operation_id": fields.filed_source.operation_id,
        "private_context": fields.private_context.as_ref().map(|_| "[REDACTED]"),
        "reason": report_reason(fields.reason),
        "report_id": fields.report_id,
        "reporter_principal_id": fields.reporter_principal_id,
        "state": report_state(fields.state),
        "target": report_target(fields.target),
        "version": fields.version,
    })
}

fn report_target(target: ModerationReportTarget) -> Value {
    match target {
        ModerationReportTarget::Event(event_id) => json!({
            "kind": "event",
            "value": hex_bytes(event_id.as_bytes()),
        }),
        ModerationReportTarget::Principal(principal_id) => json!({
            "kind": "principal",
            "value": principal_id,
        }),
        ModerationReportTarget::BlobSha256(digest) => json!({
            "kind": "blob_sha256",
            "value": hex_bytes(&digest),
        }),
    }
}

fn report_state(state: ModerationReportState) -> Value {
    match state {
        ModerationReportState::Open => json!({ "status": "open" }),
        ModerationReportState::Resolved(resolution) => json!({
            "actor_principal_id": resolution.actor_principal_id,
            "operation_id": resolution.source.operation_id,
            "resolution": resolution_label(resolution.resolution),
            "resolved_at_millis": resolution.source.occurred_at_millis,
            "status": "resolved",
        }),
    }
}

const fn report_reason(reason: ModerationReportReason) -> &'static str {
    match reason {
        ModerationReportReason::Spam => "spam",
        ModerationReportReason::Profanity => "profanity",
        ModerationReportReason::IllegalContent => "illegal_content",
        ModerationReportReason::Nudity => "nudity",
        ModerationReportReason::Malware => "malware",
        ModerationReportReason::Impersonation => "impersonation",
        ModerationReportReason::Other => "other",
    }
}

const fn resolution_label(resolution: ModerationResolution) -> &'static str {
    match resolution {
        ModerationResolution::Dismissed => "dismissed",
        ModerationResolution::ContentRemoved => "content_removed",
        ModerationResolution::MemberRemoved => "member_removed",
        ModerationResolution::TimedOut => "timed_out",
        ModerationResolution::Banned => "banned",
        ModerationResolution::Escalated => "escalated",
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

const fn common_error_class(error: ModerationOperatorError) -> ErrorClass {
    match error {
        ModerationOperatorError::InvalidRequest => ErrorClass::Usage,
        ModerationOperatorError::Unavailable => ErrorClass::Network { retryable: true },
        ModerationOperatorError::PartialFailure => ErrorClass::DeliveryUnknown,
        ModerationOperatorError::AuthorizationDenied | ModerationOperatorError::TenantMismatch => {
            ErrorClass::Authorization
        }
        ModerationOperatorError::InvalidBackendResponse => ErrorClass::Unexpected,
        ModerationOperatorError::StaleAction => ErrorClass::Conflict,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use collab::admin::moderation::ArchiveVersionFence;
    use collaboration_domain::{
        AggregateId, AggregateVersion, CommunityId, CommunityMembership, MembershipRole,
        MembershipStatus, ModerationCommandSource, ModerationReportContext,
        ModerationReportRecordFields, ModerationRestriction, NostrEventId, NostrPublicKey,
        OperationId, PrincipalId,
    };

    use super::*;

    struct FixedExecutor {
        result: Mutex<Option<Result<ModerationOperatorOutcome, ModerationOperatorError>>>,
        verb: Mutex<Option<&'static str>>,
    }

    impl FixedExecutor {
        fn new(result: Result<ModerationOperatorOutcome, ModerationOperatorError>) -> Self {
            Self {
                result: Mutex::new(Some(result)),
                verb: Mutex::new(None),
            }
        }

        fn verb(&self) -> Option<&'static str> {
            *self.verb.lock().expect("verb lock")
        }
    }

    impl ModerationCliExecutor for FixedExecutor {
        fn execute(
            &self,
            command: ModerationOperatorCommand,
        ) -> Result<ModerationOperatorOutcome, ModerationOperatorError> {
            let command = ModerationCliCommand::new(command);
            *self.verb.lock().expect("verb lock") = Some(command.verb());
            self.result
                .lock()
                .expect("result lock")
                .take()
                .expect("one execution")
        }
    }

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(uuid::Uuid::from_u128(value))
    }

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(uuid::Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(uuid::Uuid::from_u128(value))
    }

    fn source(value: u128, occurred_at_millis: u64) -> ModerationCommandSource {
        ModerationCommandSource {
            operation_id: OperationId::from_uuid(uuid::Uuid::from_u128(value)),
            occurred_at_millis,
        }
    }

    fn membership(community_id: CommunityId, principal_id: PrincipalId) -> CommunityMembership {
        CommunityMembership {
            community_id,
            principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }
    }

    fn open_report(private_context: Option<&str>) -> ModerationReport {
        ModerationReport::from_record(ModerationReportRecordFields {
            report_id: aggregate(10),
            community_id: community(20),
            reporter_principal_id: principal(30),
            target: ModerationReportTarget::Event(NostrEventId::from_bytes([0xab; 32])),
            reason: ModerationReportReason::Spam,
            private_context: private_context
                .map(ModerationReportContext::new)
                .transpose()
                .expect("private context"),
            filed_source: source(40, 1_000),
            state: ModerationReportState::Open,
            version: AggregateVersion::FIRST,
        })
        .expect("report")
    }

    fn receipt() -> ModerationWriteReceipt {
        ModerationWriteReceipt {
            operation_id: OperationId::from_uuid(uuid::Uuid::from_u128(90)),
            version: AggregateVersion::new(2).expect("version"),
        }
    }

    fn commands() -> Vec<(&'static str, ModerationOperatorCommand)> {
        let community_id = community(20);
        let target_principal_id = principal(50);
        let target_membership = membership(community_id, target_principal_id);
        vec![
            (
                "report",
                ModerationOperatorCommand::FileReport {
                    report_id: aggregate(60),
                    target: ModerationReportTarget::Principal(target_principal_id),
                    reason: ModerationReportReason::Spam,
                    private_context: None,
                    source: source(61, 2_000),
                },
            ),
            (
                "list",
                ModerationOperatorCommand::ListReports {
                    limit: 50,
                    source: source(62, 2_000),
                },
            ),
            (
                "resolve",
                ModerationOperatorCommand::ResolveReport {
                    report: open_report(None),
                    expected_version: AggregateVersion::FIRST,
                    resolution: ModerationResolution::Dismissed,
                    source: source(63, 2_000),
                },
            ),
            (
                "ban",
                ModerationOperatorCommand::ApplyBan {
                    restriction: ModerationRestriction::new(community_id, target_principal_id)
                        .expect("restriction"),
                    expected_version: AggregateVersion::FIRST,
                    expires_at_millis: None,
                    target_membership,
                    current_target_membership_version: AggregateVersion::FIRST,
                    source: source(64, 2_000),
                },
            ),
            (
                "timeout",
                ModerationOperatorCommand::ApplyTimeout {
                    restriction: ModerationRestriction::new(community_id, target_principal_id)
                        .expect("restriction"),
                    expected_version: AggregateVersion::FIRST,
                    expires_at_millis: 4_000,
                    target_membership,
                    current_target_membership_version: AggregateVersion::FIRST,
                    source: source(65, 2_000),
                },
            ),
            (
                "unban",
                ModerationOperatorCommand::LiftBan {
                    restriction: ModerationRestriction::from_history(
                        community_id,
                        target_principal_id,
                        &[collaboration_domain::RestrictionTransition {
                            kind: collaboration_domain::RestrictionTransitionKind::ApplyBan {
                                expires_at_millis: None,
                            },
                            actor_principal_id: principal(51),
                            source: source(71, 1_000),
                            resulting_version: AggregateVersion::new(2).expect("version"),
                        }],
                    )
                    .expect("canonical ban history"),
                    expected_version: AggregateVersion::new(2).expect("version"),
                    target_membership,
                    current_target_membership_version: AggregateVersion::FIRST,
                    source: source(72, 2_000),
                },
            ),
            (
                "untimeout",
                ModerationOperatorCommand::LiftTimeout {
                    restriction: ModerationRestriction::from_history(
                        community_id,
                        target_principal_id,
                        &[collaboration_domain::RestrictionTransition {
                            kind: collaboration_domain::RestrictionTransitionKind::ApplyTimeout {
                                expires_at_millis: 3_000,
                            },
                            actor_principal_id: principal(51),
                            source: source(73, 1_000),
                            resulting_version: AggregateVersion::new(2).expect("version"),
                        }],
                    )
                    .expect("canonical timeout history"),
                    expected_version: AggregateVersion::new(2).expect("version"),
                    target_membership,
                    current_target_membership_version: AggregateVersion::FIRST,
                    source: source(74, 2_000),
                },
            ),
            (
                "archive_identity",
                ModerationOperatorCommand::ArchiveIdentity {
                    target_membership,
                    current_target_membership_version: AggregateVersion::FIRST,
                    identity_public_key: NostrPublicKey::from_bytes([0xcd; 32]),
                    version_fence: ArchiveVersionFence::new(None, None),
                    source: source(66, 2_000),
                },
            ),
            (
                "archive_community",
                ModerationOperatorCommand::ArchiveCommunity {
                    version_fence: ArchiveVersionFence::new(None, None),
                    source: source(67, 2_000),
                },
            ),
        ]
    }

    #[test]
    fn moderation_commands_keep_canonical_verbs_and_machine_write_output() {
        for (expected_verb, command) in commands() {
            let outcome = if expected_verb == "list" {
                ModerationOperatorOutcome::Reports(Vec::new())
            } else {
                ModerationOperatorOutcome::Applied(receipt())
            };
            let executor = FixedExecutor::new(Ok(outcome));
            let execution =
                execute_moderation_command(&executor, ModerationCliCommand::new(command));

            assert_eq!(execution.exit_code, 0);
            assert_eq!(execution.stderr, "");
            assert_eq!(executor.verb(), Some(expected_verb));
            if expected_verb == "list" {
                assert_eq!(
                    execution.stdout,
                    "{\"command\":\"list\",\"ok\":true,\"reports\":[]}\n"
                );
            } else {
                assert_eq!(
                    execution.stdout,
                    format!(
                        "{{\"command\":\"{expected_verb}\",\"ok\":true,\"operation_id\":\"00000000-0000-0000-0000-00000000005a\",\"version\":2}}\n"
                    )
                );
            }
        }
    }

    #[test]
    fn list_output_is_stable_and_redacts_private_context() {
        let private_context = "private evidence must never reach machine output";
        let executor =
            FixedExecutor::new(Ok(ModerationOperatorOutcome::Reports(vec![open_report(
                Some(private_context),
            )])));
        let execution = execute_moderation_command(
            &executor,
            ModerationCliCommand::new(ModerationOperatorCommand::ListReports {
                limit: 50,
                source: source(70, 2_000),
            }),
        );

        assert_eq!(execution.exit_code, 0);
        assert!(!execution.stdout.contains(private_context));
        assert_eq!(
            execution.stdout,
            concat!(
                "{\"command\":\"list\",\"ok\":true,\"reports\":[{",
                "\"community_id\":\"00000000-0000-0000-0000-000000000014\",",
                "\"filed_at_millis\":1000,",
                "\"filed_operation_id\":\"00000000-0000-0000-0000-000000000028\",",
                "\"private_context\":\"[REDACTED]\",",
                "\"reason\":\"spam\",",
                "\"report_id\":\"00000000-0000-0000-0000-00000000000a\",",
                "\"reporter_principal_id\":\"00000000-0000-0000-0000-00000000001e\",",
                "\"state\":{\"status\":\"open\"},",
                "\"target\":{\"kind\":\"event\",\"value\":\"abababababababababababababababababababababababababababababababab\"},",
                "\"version\":1}]}\n"
            )
        );
    }

    #[test]
    fn denied_and_stale_errors_have_compatible_exit_classes() {
        let cases = [
            (
                ModerationOperatorError::AuthorizationDenied,
                3,
                "auth_error",
                "moderation_operator_denied",
            ),
            (
                ModerationOperatorError::StaleAction,
                5,
                "conflict",
                "moderation_operator_stale_action",
            ),
        ];

        for (error, exit_code, category, error_code) in cases {
            let executor = FixedExecutor::new(Err(error));
            let execution = execute_moderation_command(
                &executor,
                ModerationCliCommand::new(ModerationOperatorCommand::ListReports {
                    limit: 50,
                    source: source(80, 2_000),
                }),
            );
            assert_eq!(execution.exit_code, exit_code);
            assert_eq!(execution.stdout, "");
            assert_eq!(
                execution.stderr,
                format!(
                    "{{\"command\":\"list\",\"error\":\"{category}\",\"error_code\":\"{error_code}\",\"message\":\"{error_code}\",\"ok\":false,\"retryable\":false}}\n"
                )
            );
        }
    }

    #[test]
    fn service_errors_keep_closed_retry_and_exit_contracts() {
        let cases = [
            (ModerationOperatorError::InvalidRequest, 1, false),
            (ModerationOperatorError::Unavailable, 2, true),
            (ModerationOperatorError::PartialFailure, 2, false),
            (ModerationOperatorError::TenantMismatch, 3, false),
            (ModerationOperatorError::InvalidBackendResponse, 4, false),
        ];

        for (error, expected_exit, expected_retryable) in cases {
            let executor = FixedExecutor::new(Err(error));
            let execution = execute_moderation_command(
                &executor,
                ModerationCliCommand::new(ModerationOperatorCommand::ListReports {
                    limit: 50,
                    source: source(81, 2_000),
                }),
            );
            let value: Value = serde_json::from_str(execution.stderr.trim()).expect("error JSON");
            assert_eq!(execution.exit_code, expected_exit);
            assert_eq!(value["retryable"], expected_retryable);
        }

        let executor = FixedExecutor::new(Ok(ModerationOperatorOutcome::Applied(receipt())));
        let execution = execute_moderation_command(
            &executor,
            ModerationCliCommand::new(ModerationOperatorCommand::ListReports {
                limit: 50,
                source: source(82, 2_000),
            }),
        );
        assert_eq!(execution.exit_code, 4);
        assert_eq!(
            execution.stderr,
            "{\"command\":\"list\",\"error\":\"error\",\"error_code\":\"moderation_cli_invalid_outcome\",\"message\":\"moderation_cli_invalid_outcome\",\"ok\":false,\"retryable\":false}\n"
        );
    }

    #[test]
    fn debug_output_redacts_command_payloads() {
        let command = ModerationCliCommand::new(ModerationOperatorCommand::FileReport {
            report_id: aggregate(82),
            target: ModerationReportTarget::Principal(principal(83)),
            reason: ModerationReportReason::Other,
            private_context: Some(
                ModerationReportContext::new("private command evidence").expect("private context"),
            ),
            source: source(84, 2_000),
        });
        let debug = format!("{command:?}");

        assert_eq!(
            debug,
            "ModerationCliCommand { verb: \"report\", payload: \"[REDACTED]\" }"
        );
        assert!(!debug.contains("private command evidence"));
        assert!(!debug.contains(&principal(83).to_string()));
    }
}
