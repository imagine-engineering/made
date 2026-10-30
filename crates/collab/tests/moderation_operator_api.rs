use std::sync::Mutex;

use async_trait::async_trait;
use collab::{
    admin::moderation::{
        ArchiveVersionFence, AuthorizedModerationOperation, ModerationBackendCommand,
        ModerationBackendError, ModerationBackendResponse, ModerationOperatorApi,
        ModerationOperatorBackend, ModerationOperatorCommand, ModerationOperatorError,
        ModerationOperatorOutcome, ModerationOperatorWrite,
    },
    audit::events::{ModerationAuditOperation, SecurityAuditEvent},
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, IdentityArchivePolicyState, IdentityArchiveSnapshot,
    MembershipRole, MembershipStatus, ModerationAuthorizationRequest, ModerationCommandSource,
    ModerationReport, ModerationReportContext, ModerationReportReason,
    ModerationReportRecordFields, ModerationReportState, ModerationReportTarget,
    ModerationResolution, ModerationRestriction, ModerationSnapshot, NostrPublicKey, OperationId,
    PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use uuid::Uuid;

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
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    report_scope: AuthorizationScope,
    moderation_scope: AuthorizationScope,
    community_scope: AuthorizationScope,
}

impl Fixture {
    fn new() -> Self {
        let community_id = community(1);
        let report_scope = AuthorizationScope::new("moderation:report").expect("report scope");
        let moderation_scope =
            AuthorizationScope::new("moderation:manage").expect("moderation scope");
        let community_scope =
            AuthorizationScope::new("communities:manage").expect("community scope");
        let principal = AuthenticatedPrincipal::zed_account(
            principal(2),
            community_id,
            ServiceAccountId::new(2),
            PrincipalScopes::new([
                report_scope.clone(),
                moderation_scope.clone(),
                community_scope.clone(),
            ])
            .expect("scopes"),
        );
        let tenant = TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "operator-moderation-test")
                    .expect("route"),
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
        }
    }

    fn authorization<'a>(
        &'a self,
        role: MembershipRole,
        required_scope: &'a AuthorizationScope,
        action: AuthorizationAction,
        kind: AuthorizationResourceKind,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope,
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

    fn policy<'request, 'authorization>(
        &'request self,
        authorization: &'request AuthorizationRequest<'authorization>,
    ) -> ModerationAuthorizationRequest<'request, 'authorization> {
        ModerationAuthorizationRequest {
            authorization,
            restriction: ModerationSnapshot::Absent,
            principal_archive: ModerationSnapshot::Absent,
            community_archive: ModerationSnapshot::Absent,
            resource_context: collaboration_domain::ModerationResourceContext::Current,
        }
    }

    fn list_authorization(&self, role: MembershipRole) -> AuthorizationRequest<'_> {
        self.authorization(
            role,
            &self.moderation_scope,
            AuthorizationAction::Read,
            AuthorizationResourceKind::Administration,
        )
    }

    fn report_authorization(&self, role: MembershipRole) -> AuthorizationRequest<'_> {
        self.authorization(
            role,
            &self.report_scope,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Community,
        )
    }

    fn manage_authorization(&self, role: MembershipRole) -> AuthorizationRequest<'_> {
        self.authorization(
            role,
            &self.moderation_scope,
            AuthorizationAction::Manage,
            AuthorizationResourceKind::Administration,
        )
    }

    fn archive_community_authorization(&self, role: MembershipRole) -> AuthorizationRequest<'_> {
        self.authorization(
            role,
            &self.community_scope,
            AuthorizationAction::Delete,
            AuthorizationResourceKind::Community,
        )
    }
}

#[derive(Default)]
struct TestBackend {
    calls: Mutex<Vec<AuthorizedModerationOperation>>,
    failure: Mutex<Option<ModerationBackendError>>,
}

impl TestBackend {
    fn fail_with(&self, error: ModerationBackendError) {
        *self.failure.lock().expect("failure lock") = Some(error);
    }

    fn calls(&self) -> Vec<AuthorizedModerationOperation> {
        self.calls.lock().expect("calls lock").clone()
    }
}

#[async_trait]
impl ModerationOperatorBackend for TestBackend {
    async fn execute(
        &self,
        _community_id: CommunityId,
        operation: &AuthorizedModerationOperation,
    ) -> Result<ModerationBackendResponse, ModerationBackendError> {
        self.calls
            .lock()
            .expect("calls lock")
            .push(operation.clone());
        if let Some(error) = *self.failure.lock().expect("failure lock") {
            return Err(error);
        }
        match operation.command() {
            ModerationBackendCommand::ListReports { .. } => {
                Ok(ModerationBackendResponse::Reports(Vec::new()))
            }
            ModerationBackendCommand::Write(_) => Ok(ModerationBackendResponse::Written(
                operation.expected_receipt().expect("write receipt"),
            )),
        }
    }
}

fn list_command(value: u128) -> ModerationOperatorCommand {
    ModerationOperatorCommand::ListReports {
        limit: 50,
        source: source(value),
    }
}

fn open_report(community_id: CommunityId, value: u128) -> ModerationReport {
    ModerationReport::from_record(ModerationReportRecordFields {
        report_id: aggregate(value),
        community_id,
        reporter_principal_id: principal(50),
        target: ModerationReportTarget::Principal(principal(60)),
        reason: ModerationReportReason::Spam,
        private_context: None,
        filed_source: source(value + 1),
        state: ModerationReportState::Open,
        version: AggregateVersion::FIRST,
    })
    .expect("open report")
}

#[tokio::test]
async fn operator_role_matrix_is_least_privilege() {
    let fixture = Fixture::new();
    let api = ModerationOperatorApi::new(TestBackend::default());

    for role in [MembershipRole::Owner, MembershipRole::Admin] {
        let authorization = fixture.list_authorization(role);
        assert!(matches!(
            api.execute(
                &fixture.policy(&authorization),
                list_command(10 + role as u128)
            )
            .await,
            Ok(ModerationOperatorOutcome::Reports(_))
        ));
    }
    let member_list = fixture.list_authorization(MembershipRole::Member);
    assert_eq!(
        api.execute(&fixture.policy(&member_list), list_command(20))
            .await,
        Err(ModerationOperatorError::AuthorizationDenied)
    );

    let member_report = fixture.report_authorization(MembershipRole::Member);
    assert!(matches!(
        api.execute(
            &fixture.policy(&member_report),
            ModerationOperatorCommand::FileReport {
                report_id: aggregate(30),
                target: ModerationReportTarget::Principal(principal(31)),
                reason: ModerationReportReason::Spam,
                private_context: None,
                source: source(30),
            },
        )
        .await,
        Ok(ModerationOperatorOutcome::Applied(_))
    ));

    let admin_archive = fixture.archive_community_authorization(MembershipRole::Admin);
    assert_eq!(
        api.execute(
            &fixture.policy(&admin_archive),
            ModerationOperatorCommand::ArchiveCommunity {
                version_fence: ArchiveVersionFence::new(None, None),
                source: source(40),
            },
        )
        .await,
        Err(ModerationOperatorError::AuthorizationDenied)
    );
    let owner_archive = fixture.archive_community_authorization(MembershipRole::Owner);
    assert!(matches!(
        api.execute(
            &fixture.policy(&owner_archive),
            ModerationOperatorCommand::ArchiveCommunity {
                version_fence: ArchiveVersionFence::new(None, None),
                source: source(41),
            },
        )
        .await,
        Ok(ModerationOperatorOutcome::Applied(_))
    ));
}

#[tokio::test]
async fn operator_operations_keep_exact_audit_attribution() {
    let fixture = Fixture::new();
    let api = ModerationOperatorApi::new(TestBackend::default());
    let authorization = fixture.manage_authorization(MembershipRole::Owner);
    let target = membership(
        fixture.tenant.community_id(),
        principal(70),
        MembershipRole::Member,
    );

    let commands = [
        ModerationOperatorCommand::ResolveReport {
            report: open_report(fixture.tenant.community_id(), 50),
            expected_version: AggregateVersion::FIRST,
            resolution: ModerationResolution::Dismissed,
            source: source(52),
        },
        ModerationOperatorCommand::ApplyBan {
            restriction: ModerationRestriction::new(
                fixture.tenant.community_id(),
                target.principal_id,
            )
            .expect("restriction"),
            expected_version: AggregateVersion::FIRST,
            expires_at_millis: None,
            target_membership: target,
            current_target_membership_version: AggregateVersion::FIRST,
            source: source(53),
        },
        ModerationOperatorCommand::ApplyTimeout {
            restriction: ModerationRestriction::new(
                fixture.tenant.community_id(),
                target.principal_id,
            )
            .expect("restriction"),
            expected_version: AggregateVersion::FIRST,
            expires_at_millis: 30_000,
            target_membership: target,
            current_target_membership_version: AggregateVersion::FIRST,
            source: source(54),
        },
        ModerationOperatorCommand::ArchiveIdentity {
            target_membership: target,
            current_target_membership_version: AggregateVersion::FIRST,
            identity_public_key: NostrPublicKey::from_bytes([7; 32]),
            version_fence: ArchiveVersionFence::new(None, None),
            source: source(55),
        },
    ];

    for command in commands {
        assert!(matches!(
            api.execute(&fixture.policy(&authorization), command).await,
            Ok(ModerationOperatorOutcome::Applied(_))
        ));
    }

    let calls = api.into_backend().calls();
    assert_eq!(calls.len(), 4);
    let expected_operations = [
        ModerationAuditOperation::ResolveReport,
        ModerationAuditOperation::ApplyRestriction,
        ModerationAuditOperation::ApplyRestriction,
        ModerationAuditOperation::ArchiveIdentity,
    ];
    let expected_subjects = [
        principal(60),
        target.principal_id,
        target.principal_id,
        target.principal_id,
    ];
    for ((call, expected_operation), expected_subject) in
        calls.iter().zip(expected_operations).zip(expected_subjects)
    {
        let SecurityAuditEvent::Moderation {
            context,
            operation,
            subject_principal_id,
            ..
        } = call.audit_event()
        else {
            panic!("operator command must carry a moderation audit event");
        };
        assert_eq!(*operation, expected_operation);
        assert_eq!(context.community_id(), fixture.tenant.community_id());
        assert_eq!(
            context.actor_principal_id(),
            Some(fixture.principal.principal_id())
        );
        assert_eq!(*subject_principal_id, Some(expected_subject));
        assert_eq!(
            context.operation_id(),
            call.expected_receipt().expect("write receipt").operation_id
        );
    }
}

#[tokio::test]
async fn foreign_and_stale_operator_state_fails_before_backend_execution() {
    let fixture = Fixture::new();
    let api = ModerationOperatorApi::new(TestBackend::default());
    let list_authorization = fixture.list_authorization(MembershipRole::Owner);
    let foreign_archive = IdentityArchiveSnapshot {
        community_id: community(99),
        principal_id: fixture.principal.principal_id(),
        state: IdentityArchivePolicyState::Visible,
        version: AggregateVersion::FIRST,
    };
    let foreign_policy = ModerationAuthorizationRequest {
        authorization: &list_authorization,
        restriction: ModerationSnapshot::Absent,
        principal_archive: ModerationSnapshot::Current(foreign_archive),
        community_archive: ModerationSnapshot::Absent,
        resource_context: collaboration_domain::ModerationResourceContext::Current,
    };
    assert_eq!(
        api.execute(&foreign_policy, list_command(60)).await,
        Err(ModerationOperatorError::TenantMismatch)
    );

    let manage = fixture.manage_authorization(MembershipRole::Owner);
    assert_eq!(
        api.execute(
            &fixture.policy(&manage),
            ModerationOperatorCommand::ResolveReport {
                report: open_report(fixture.tenant.community_id(), 61),
                expected_version: AggregateVersion::FIRST
                    .next()
                    .expect("stale second version"),
                resolution: ModerationResolution::Dismissed,
                source: source(63),
            },
        )
        .await,
        Err(ModerationOperatorError::StaleAction)
    );
    assert_eq!(
        api.execute(
            &fixture.policy(&manage),
            ModerationOperatorCommand::ArchiveIdentity {
                target_membership: membership(
                    fixture.tenant.community_id(),
                    principal(64),
                    MembershipRole::Member,
                ),
                current_target_membership_version: AggregateVersion::FIRST,
                identity_public_key: NostrPublicKey::from_bytes([8; 32]),
                version_fence: ArchiveVersionFence::new(Some(AggregateVersion::FIRST), None,),
                source: source(64),
            },
        )
        .await,
        Err(ModerationOperatorError::StaleAction)
    );
    assert!(api.into_backend().calls().is_empty());
}

#[tokio::test]
async fn partial_failure_diagnostics_redact_private_report_context() {
    let fixture = Fixture::new();
    let backend = TestBackend::default();
    backend.fail_with(ModerationBackendError::PartialFailure);
    let api = ModerationOperatorApi::new(backend);
    let authorization = fixture.report_authorization(MembershipRole::Member);
    let secret = "private reporter evidence must not escape";
    let error = api
        .execute(
            &fixture.policy(&authorization),
            ModerationOperatorCommand::FileReport {
                report_id: aggregate(70),
                target: ModerationReportTarget::Principal(principal(71)),
                reason: ModerationReportReason::Other,
                private_context: Some(
                    ModerationReportContext::new(secret).expect("private context"),
                ),
                source: source(70),
            },
        )
        .await
        .expect_err("partial write must fail closed");

    assert_eq!(error, ModerationOperatorError::PartialFailure);
    assert!(!error.to_string().contains(secret));
    let calls = api.into_backend().calls();
    assert_eq!(calls.len(), 1);
    assert!(!format!("{:?}", calls[0]).contains(secret));
    assert!(matches!(
        calls[0].command(),
        ModerationBackendCommand::Write(ModerationOperatorWrite::Report(_))
    ));
}

#[test]
fn moderation_hydrated_restrictions_preserve_ban_actor_across_timeout_transitions() {
    use collaboration_domain::{BanState, RestrictionTransition, RestrictionTransitionKind};
    let transitions = [
        RestrictionTransition {
            kind: RestrictionTransitionKind::ApplyBan {
                expires_at_millis: None,
            },
            actor_principal_id: principal(2),
            source: source(10),
            resulting_version: AggregateVersion::new(2).expect("version"),
        },
        RestrictionTransition {
            kind: RestrictionTransitionKind::ApplyTimeout {
                expires_at_millis: 30_000,
            },
            actor_principal_id: principal(3),
            source: source(11),
            resulting_version: AggregateVersion::new(3).expect("version"),
        },
    ];
    let restriction = ModerationRestriction::from_history(community(1), principal(4), &transitions)
        .expect("history");
    assert!(
        matches!(restriction.fields().ban, BanState::Active { actor_principal_id, .. } if actor_principal_id == principal(2))
    );
    assert!(restriction.fields().ban.is_active_at(30_001));
    assert!(!restriction.fields().timeout.is_active_at(30_001));
    assert_eq!(
        ModerationRestriction::from_record(restriction.fields().clone()),
        Ok(restriction)
    );
    for error in [
        collab::admin::moderation::ModerationRestrictionReadError::Unavailable,
        collab::admin::moderation::ModerationRestrictionReadError::InvalidHistory,
        collab::admin::moderation::ModerationRestrictionReadError::TenantBoundaryViolation,
    ] {
        assert!(!error.to_string().contains("postgres://"));
        assert!(!error.to_string().contains("banned"));
    }
}
