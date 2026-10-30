use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cli::collaboration_moderation::{
    ModerationCliCommand, ModerationCliExecution, ModerationCliExecutor, execute_moderation_command,
};
use collab::{
    admin::moderation::{
        AuthorizedModerationOperation, ModerationBackendCommand, ModerationBackendError,
        ModerationBackendResponse, ModerationOperatorApi, ModerationOperatorBackend,
        ModerationOperatorCommand, ModerationOperatorError, ModerationOperatorOutcome,
        ModerationOperatorWrite,
    },
    audit::events::{ModerationAuditOperation, SecurityAuditEvent},
};
#[allow(dead_code)]
#[path = "../../collab_ui/src/moderation.rs"]
mod native_moderation;

use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, IdentityArchivePolicyState, IdentityArchiveSnapshot,
    MembershipRole, MembershipStatus, ModerationAuthorizationDecision,
    ModerationAuthorizationDenial, ModerationAuthorizationRequest, ModerationCommandOutcome,
    ModerationCommandSource, ModerationError, ModerationReport, ModerationReportContext,
    ModerationReportReason, ModerationReportRecordFields, ModerationReportState,
    ModerationReportTarget, ModerationResourceContext, ModerationSnapshot, OperationId,
    PersonalMute, PersonalMuteState, PrincipalId, PrincipalScopes, ServiceAccountId,
    TrustedTenantRoute, authorize_with_moderation,
};
use gpui::{AppContext as _, TestAppContext};
use native_moderation::{
    ModerationEvidenceSummary, ModerationQueueAction, ModerationQueueError, ModerationQueueNotice,
    ModerationQueueSnapshot, ModerationQueueView, ModerationReportPresentation,
};
use uuid::Uuid;

const PRIMARY_COMMUNITY: u128 = 1;
const FOREIGN_COMMUNITY: u128 = 2;
const OPERATOR_PRINCIPAL: u128 = 10;
const PRIVATE_EVIDENCE: &str = "private moderation evidence must remain tenant local";

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn source(value: u128) -> ModerationCommandSource {
    ModerationCommandSource {
        operation_id: OperationId::from_uuid(Uuid::from_u128(value)),
        occurred_at_millis: 10_000 + u64::try_from(value).expect("source timestamp"),
    }
}

fn membership(
    community_id: CommunityId,
    principal_id: PrincipalId,
    role: MembershipRole,
) -> CommunityMembership {
    CommunityMembership {
        community_id,
        principal_id,
        role,
        status: MembershipStatus::Active,
        version: AggregateVersion::FIRST,
    }
}

struct Fixture {
    tenant: collaboration_domain::TenantContext,
    principal: AuthenticatedPrincipal,
    report_scope: AuthorizationScope,
    moderation_scope: AuthorizationScope,
    community_scope: AuthorizationScope,
    mute_scope: AuthorizationScope,
}

impl Fixture {
    fn new(community_value: u128, principal_value: u128) -> Self {
        let community_id = community(community_value);
        let report_scope = AuthorizationScope::new("moderation:report").expect("report scope");
        let moderation_scope =
            AuthorizationScope::new("moderation:manage").expect("moderation scope");
        let community_scope =
            AuthorizationScope::new("communities:manage").expect("community scope");
        let mute_scope = AuthorizationScope::new("moderation:mute").expect("mute scope");
        let principal = AuthenticatedPrincipal::zed_account(
            principal(principal_value),
            community_id,
            ServiceAccountId::new(u64::try_from(principal_value).expect("service account")),
            PrincipalScopes::new([
                report_scope.clone(),
                moderation_scope.clone(),
                community_scope.clone(),
                mute_scope.clone(),
            ])
            .expect("principal scopes"),
        );
        let tenant = collaboration_domain::TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(
                    community_id,
                    "moderation-administration-security",
                )
                .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant");
        Self {
            tenant,
            principal,
            report_scope,
            moderation_scope,
            community_scope,
            mute_scope,
        }
    }

    fn authorization<'a>(
        &'a self,
        role: MembershipRole,
        scope: &'a AuthorizationScope,
        action: AuthorizationAction,
        kind: AuthorizationResourceKind,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: scope,
            action,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind,
                resource_id: AggregateId::from_uuid(self.tenant.community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(membership(
                self.tenant.community_id(),
                self.principal.principal_id(),
                role,
            )),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 20_000,
        }
    }

    fn authorization_for(
        &self,
        command: &ModerationOperatorCommand,
        role: MembershipRole,
    ) -> AuthorizationRequest<'_> {
        match command {
            ModerationOperatorCommand::ListReports { .. } => self.authorization(
                role,
                &self.moderation_scope,
                AuthorizationAction::Read,
                AuthorizationResourceKind::Administration,
            ),
            ModerationOperatorCommand::FileReport { .. } => self.authorization(
                role,
                &self.report_scope,
                AuthorizationAction::Write,
                AuthorizationResourceKind::Community,
            ),
            ModerationOperatorCommand::ArchiveCommunity { .. } => self.authorization(
                role,
                &self.community_scope,
                AuthorizationAction::Delete,
                AuthorizationResourceKind::Community,
            ),
            ModerationOperatorCommand::ResolveReport { .. }
            | ModerationOperatorCommand::ApplyBan { .. }
            | ModerationOperatorCommand::ApplyTimeout { .. }
            | ModerationOperatorCommand::LiftBan { .. }
            | ModerationOperatorCommand::LiftTimeout { .. }
            | ModerationOperatorCommand::ArchiveIdentity { .. } => self.authorization(
                role,
                &self.moderation_scope,
                AuthorizationAction::Manage,
                AuthorizationResourceKind::Administration,
            ),
        }
    }

    fn mute_authorization(&self, role: MembershipRole) -> AuthorizationRequest<'_> {
        self.authorization(
            role,
            &self.mute_scope,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Community,
        )
    }
}

fn open_report(
    community_id: CommunityId,
    report_value: u128,
    private_context: Option<&str>,
) -> ModerationReport {
    ModerationReport::from_record(ModerationReportRecordFields {
        report_id: aggregate(report_value),
        community_id,
        reporter_principal_id: principal(30),
        target: ModerationReportTarget::Principal(principal(40)),
        reason: ModerationReportReason::Other,
        private_context: private_context
            .map(ModerationReportContext::new)
            .transpose()
            .expect("private report context"),
        filed_source: source(report_value + 100),
        state: ModerationReportState::Open,
        version: AggregateVersion::FIRST,
    })
    .expect("open report")
}

#[derive(Clone, Copy)]
enum BackendBehavior {
    Normal,
    ForeignReports,
    PartialFailure,
}

#[derive(Clone)]
struct RecordingBackend {
    behavior: BackendBehavior,
    calls: Arc<Mutex<Vec<AuthorizedModerationOperation>>>,
}

impl RecordingBackend {
    fn new(behavior: BackendBehavior) -> Self {
        Self {
            behavior,
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn calls(&self) -> Vec<AuthorizedModerationOperation> {
        self.calls.lock().expect("backend calls").clone()
    }
}

#[async_trait]
impl ModerationOperatorBackend for RecordingBackend {
    async fn execute(
        &self,
        _community_id: CommunityId,
        operation: &AuthorizedModerationOperation,
    ) -> Result<ModerationBackendResponse, ModerationBackendError> {
        self.calls
            .lock()
            .expect("backend calls")
            .push(operation.clone());
        match self.behavior {
            BackendBehavior::PartialFailure => Err(ModerationBackendError::PartialFailure),
            BackendBehavior::ForeignReports => {
                Ok(ModerationBackendResponse::Reports(vec![open_report(
                    community(FOREIGN_COMMUNITY),
                    900,
                    Some(PRIVATE_EVIDENCE),
                )]))
            }
            BackendBehavior::Normal => match operation.command() {
                ModerationBackendCommand::ListReports { .. } => {
                    Ok(ModerationBackendResponse::Reports(Vec::new()))
                }
                ModerationBackendCommand::Write(_) => operation
                    .expected_receipt()
                    .map(ModerationBackendResponse::Written)
                    .ok_or(ModerationBackendError::InvalidData),
            },
        }
    }
}

#[derive(Clone, Copy)]
enum RequesterPolicy {
    Active,
    Archived,
    ForeignArchive,
}

struct CliOperatorExecutor {
    backend: RecordingBackend,
    role: MembershipRole,
    policy: RequesterPolicy,
}

impl CliOperatorExecutor {
    fn new(backend: RecordingBackend, role: MembershipRole, policy: RequesterPolicy) -> Self {
        Self {
            backend,
            role,
            policy,
        }
    }
}

impl ModerationCliExecutor for CliOperatorExecutor {
    fn execute(
        &self,
        command: ModerationOperatorCommand,
    ) -> Result<ModerationOperatorOutcome, ModerationOperatorError> {
        let fixture = Fixture::new(PRIMARY_COMMUNITY, OPERATOR_PRINCIPAL);
        let authorization = fixture.authorization_for(&command, self.role);
        let principal_archive = match self.policy {
            RequesterPolicy::Active => ModerationSnapshot::Absent,
            RequesterPolicy::Archived => ModerationSnapshot::Current(IdentityArchiveSnapshot {
                community_id: fixture.tenant.community_id(),
                principal_id: fixture.principal.principal_id(),
                state: IdentityArchivePolicyState::Archived,
                version: AggregateVersion::FIRST,
            }),
            RequesterPolicy::ForeignArchive => {
                ModerationSnapshot::Current(IdentityArchiveSnapshot {
                    community_id: community(FOREIGN_COMMUNITY),
                    principal_id: fixture.principal.principal_id(),
                    state: IdentityArchivePolicyState::Archived,
                    version: AggregateVersion::FIRST,
                })
            }
        };
        let policy = ModerationAuthorizationRequest {
            authorization: &authorization,
            restriction: ModerationSnapshot::Absent,
            principal_archive,
            community_archive: ModerationSnapshot::Absent,
            resource_context: ModerationResourceContext::Current,
        };
        smol::block_on(ModerationOperatorApi::new(self.backend.clone()).execute(&policy, command))
    }
}

fn list_command(operation_value: u128) -> ModerationCliCommand {
    ModerationCliCommand::new(ModerationOperatorCommand::ListReports {
        limit: 50,
        source: source(operation_value),
    })
}

fn assert_redacted_failure(
    execution: &ModerationCliExecution,
    expected_exit_code: i32,
    expected_code: &str,
) {
    assert_eq!(execution.exit_code, expected_exit_code);
    assert!(execution.stdout.is_empty());
    assert!(execution.stderr.contains(expected_code));
    assert!(!execution.stderr.contains(PRIVATE_EVIDENCE));
    assert!(
        !execution
            .stderr
            .contains(&community(FOREIGN_COMMUNITY).to_string())
    );
}

#[derive(Debug, Eq, PartialEq)]
enum PersonalMuteBoundaryError {
    Denied(ModerationAuthorizationDenial),
    Domain(ModerationError),
}

fn set_personal_mute_through_boundary(
    personal_mute: &mut PersonalMute,
    authorization: &AuthorizationRequest<'_>,
    principal_archive: ModerationSnapshot<IdentityArchiveSnapshot>,
    expected_version: AggregateVersion,
    state: PersonalMuteState,
    command_source: ModerationCommandSource,
) -> Result<ModerationCommandOutcome, PersonalMuteBoundaryError> {
    let decision = authorize_with_moderation(&ModerationAuthorizationRequest {
        authorization,
        restriction: ModerationSnapshot::Absent,
        principal_archive,
        community_archive: ModerationSnapshot::Absent,
        resource_context: ModerationResourceContext::Current,
    });
    match decision {
        ModerationAuthorizationDecision::Allowed(_) => personal_mute
            .set_state(expected_version, state, command_source, authorization)
            .map_err(PersonalMuteBoundaryError::Domain),
        ModerationAuthorizationDecision::Denied(denial) => {
            Err(PersonalMuteBoundaryError::Denied(denial))
        }
    }
}

#[test]
fn personal_mute_remains_owner_local_across_tenants_and_archive_state() {
    let owner_fixture = Fixture::new(PRIMARY_COMMUNITY, 20);
    let authorization = owner_fixture.mute_authorization(MembershipRole::Member);
    let mut personal_mute = PersonalMute::new(
        owner_fixture.tenant.community_id(),
        principal(21),
        &authorization,
    )
    .expect("personal mute");
    assert_eq!(
        set_personal_mute_through_boundary(
            &mut personal_mute,
            &authorization,
            ModerationSnapshot::Absent,
            AggregateVersion::FIRST,
            PersonalMuteState::Muted,
            source(300),
        ),
        Ok(ModerationCommandOutcome::Applied)
    );
    let muted_version = personal_mute.fields().version;

    let administrator_fixture = Fixture::new(PRIMARY_COMMUNITY, 22);
    let administrator = administrator_fixture.mute_authorization(MembershipRole::Admin);
    assert_eq!(
        personal_mute.set_state(
            muted_version,
            PersonalMuteState::Unmuted,
            source(301),
            &administrator,
        ),
        Err(ModerationError::PersonalMuteOwnerMismatch)
    );

    let foreign_fixture = Fixture::new(FOREIGN_COMMUNITY, 20);
    let foreign = foreign_fixture.mute_authorization(MembershipRole::Member);
    assert!(
        personal_mute
            .set_state(
                muted_version,
                PersonalMuteState::Unmuted,
                source(302),
                &foreign,
            )
            .is_err()
    );

    let archived = IdentityArchiveSnapshot {
        community_id: owner_fixture.tenant.community_id(),
        principal_id: owner_fixture.principal.principal_id(),
        state: IdentityArchivePolicyState::Archived,
        version: AggregateVersion::FIRST,
    };
    assert_eq!(
        set_personal_mute_through_boundary(
            &mut personal_mute,
            &authorization,
            ModerationSnapshot::Current(archived),
            muted_version,
            PersonalMuteState::Unmuted,
            source(303),
        ),
        Err(PersonalMuteBoundaryError::Denied(
            ModerationAuthorizationDenial::IdentityArchived
        ))
    );
    assert_eq!(personal_mute.fields().state, PersonalMuteState::Muted);
    assert_eq!(personal_mute.fields().version, muted_version);
    assert_eq!(personal_mute.fields().transitions.len(), 1);
    let debug = format!("{personal_mute:?}");
    assert!(!debug.contains(&principal(20).to_string()));
    assert!(!debug.contains(&principal(21).to_string()));
}

#[test]
fn operator_and_cli_fail_closed_for_roles_tenants_archives_and_partial_writes() {
    let role_backend = RecordingBackend::new(BackendBehavior::Normal);
    let role_execution = execute_moderation_command(
        &CliOperatorExecutor::new(
            role_backend.clone(),
            MembershipRole::Member,
            RequesterPolicy::Active,
        ),
        list_command(400),
    );
    assert_redacted_failure(&role_execution, 3, "moderation_operator_denied");
    assert!(role_backend.calls().is_empty());

    let archived_backend = RecordingBackend::new(BackendBehavior::Normal);
    let archived_execution = execute_moderation_command(
        &CliOperatorExecutor::new(
            archived_backend.clone(),
            MembershipRole::Owner,
            RequesterPolicy::Archived,
        ),
        list_command(401),
    );
    assert_redacted_failure(&archived_execution, 3, "moderation_operator_denied");
    assert!(archived_backend.calls().is_empty());

    let foreign_policy_backend = RecordingBackend::new(BackendBehavior::Normal);
    let foreign_policy_execution = execute_moderation_command(
        &CliOperatorExecutor::new(
            foreign_policy_backend.clone(),
            MembershipRole::Owner,
            RequesterPolicy::ForeignArchive,
        ),
        list_command(402),
    );
    assert_redacted_failure(
        &foreign_policy_execution,
        3,
        "moderation_operator_tenant_mismatch",
    );
    assert!(foreign_policy_backend.calls().is_empty());

    let foreign_response_backend = RecordingBackend::new(BackendBehavior::ForeignReports);
    let foreign_response_execution = execute_moderation_command(
        &CliOperatorExecutor::new(
            foreign_response_backend.clone(),
            MembershipRole::Owner,
            RequesterPolicy::Active,
        ),
        list_command(403),
    );
    assert_redacted_failure(
        &foreign_response_execution,
        4,
        "moderation_operator_invalid_backend_response",
    );
    assert_eq!(foreign_response_backend.calls().len(), 1);

    let partial_backend = RecordingBackend::new(BackendBehavior::PartialFailure);
    let partial_execution = execute_moderation_command(
        &CliOperatorExecutor::new(
            partial_backend.clone(),
            MembershipRole::Member,
            RequesterPolicy::Active,
        ),
        ModerationCliCommand::new(ModerationOperatorCommand::FileReport {
            report_id: aggregate(404),
            target: ModerationReportTarget::Principal(principal(41)),
            reason: ModerationReportReason::Other,
            private_context: Some(
                ModerationReportContext::new(PRIVATE_EVIDENCE).expect("private context"),
            ),
            source: source(404),
        }),
    );
    assert_redacted_failure(&partial_execution, 2, "moderation_operator_partial_failure");
    assert_eq!(partial_backend.calls().len(), 1);
    assert!(!format!("{:?}", partial_backend.calls()).contains(PRIVATE_EVIDENCE));
}

#[gpui::test]
fn native_queue_cli_and_operator_preserve_tenant_version_and_audit_attribution(
    cx: &mut TestAppContext,
) {
    let community_id = community(PRIMARY_COMMUNITY);
    let report = open_report(community_id, 500, Some(PRIVATE_EVIDENCE));
    let report_id = report.fields().report_id;
    assert_eq!(
        ModerationQueueSnapshot::new(
            community_id,
            MembershipRole::Member,
            vec![report.clone()],
            vec![ModerationReportPresentation {
                report_id,
                target_label: "Member".to_owned(),
                reporter_label: "Reporter".to_owned(),
                evidence_summary: None,
            }],
        )
        .err(),
        Some(ModerationQueueError::PermissionDenied)
    );
    let foreign_report = open_report(community(FOREIGN_COMMUNITY), 501, Some(PRIVATE_EVIDENCE));
    assert_eq!(
        ModerationQueueSnapshot::new(
            community_id,
            MembershipRole::Owner,
            vec![foreign_report.clone()],
            vec![ModerationReportPresentation {
                report_id: foreign_report.fields().report_id,
                target_label: "Foreign member".to_owned(),
                reporter_label: "Foreign reporter".to_owned(),
                evidence_summary: Some(
                    ModerationEvidenceSummary::new(PRIVATE_EVIDENCE).expect("evidence"),
                ),
            }],
        )
        .err(),
        Some(ModerationQueueError::TenantMismatch)
    );

    let snapshot = ModerationQueueSnapshot::new(
        community_id,
        MembershipRole::Owner,
        vec![report],
        vec![ModerationReportPresentation {
            report_id,
            target_label: "Member".to_owned(),
            reporter_label: "Reporter".to_owned(),
            evidence_summary: Some(
                ModerationEvidenceSummary::new(PRIVATE_EVIDENCE).expect("evidence"),
            ),
        }],
    )
    .expect("authorized queue");
    let view = cx.new(|_| ModerationQueueView::new(snapshot));
    view.update(cx, |view, cx| {
        view.request_action(report_id, ModerationQueueAction::Dismiss, cx)
    })
    .expect("confirmation");
    let request = view
        .update(cx, ModerationQueueView::confirm_action)
        .expect("action request");
    assert_eq!(request.community_id, community_id);
    assert_eq!(request.expected_version, AggregateVersion::FIRST);

    let backend = RecordingBackend::new(BackendBehavior::Normal);
    let resolution_source = source(700);
    let execution = execute_moderation_command(
        &CliOperatorExecutor::new(
            backend.clone(),
            MembershipRole::Owner,
            RequesterPolicy::Active,
        ),
        ModerationCliCommand::new(ModerationOperatorCommand::ResolveReport {
            report: request.report,
            expected_version: request.expected_version,
            resolution: request.action.resolution(),
            source: resolution_source,
        }),
    );
    assert_eq!(execution.exit_code, 0);
    assert!(execution.stderr.is_empty());
    assert!(
        execution
            .stdout
            .contains(&resolution_source.operation_id.to_string())
    );
    assert!(!execution.stdout.contains(PRIVATE_EVIDENCE));

    let calls = backend.calls();
    assert_eq!(calls.len(), 1, "one action must produce one audited write");
    let operation = calls.first().expect("audited operation");
    let SecurityAuditEvent::Moderation {
        context,
        operation: audit_operation,
        subject_principal_id,
        record_id,
    } = operation.audit_event()
    else {
        panic!("moderation write must carry audit attribution");
    };
    assert_eq!(*audit_operation, ModerationAuditOperation::ResolveReport);
    assert_eq!(context.community_id(), community_id);
    assert_eq!(context.operation_id(), resolution_source.operation_id);
    assert_eq!(
        context.actor_principal_id(),
        Some(principal(OPERATOR_PRINCIPAL))
    );
    assert_eq!(*subject_principal_id, Some(principal(40)));
    assert_eq!(*record_id, report_id.as_uuid());

    let authoritative_report = match operation.command() {
        ModerationBackendCommand::Write(ModerationOperatorWrite::Report(report)) => report.clone(),
        _ => panic!("resolution must write the canonical report"),
    };
    view.update(cx, |view, cx| {
        view.complete_action(request.request_id, authoritative_report, cx)
    })
    .expect("authoritative completion");
    assert_eq!(
        view.read_with(cx, |view, _| view.notice()),
        Some(ModerationQueueNotice::Succeeded(
            ModerationQueueAction::Dismiss
        ))
    );
    let row = view.read_with(cx, |view, _| view.rows().first().cloned());
    let row = row.expect("resolved row");
    assert!(!row.is_open());
    assert!(!format!("{row:?}").contains(PRIVATE_EVIDENCE));
}
