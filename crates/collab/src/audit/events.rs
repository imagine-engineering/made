use collaboration_domain::{
    AuditAction, AuditCategory, AuditEntry, AuditError, AuditField, AuditFieldName, AuditFields,
    AuditIdentifier, AuditOutcome, AuditRecord, AuditValue, AuthenticatedPrincipal,
    AuthenticatedPrincipalKind, CommunityId, NostrAuthenticationMethod, OperationId, PrincipalId,
    TenantContext,
};
use uuid::Uuid;

use super::repository::{AuditRepository, AuditRepositoryError, ExpectedAuditHead};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuditActorKind {
    ZedAccount,
    NostrIdentity,
    OwnerAttestedAgent,
    ScopedToken,
    Service,
}

impl AuditActorKind {
    const fn name(self) -> &'static str {
        match self {
            Self::ZedAccount => "zed_account",
            Self::NostrIdentity => "nostr_identity",
            Self::OwnerAttestedAgent => "owner_attested_agent",
            Self::ScopedToken => "scoped_token",
            Self::Service => "service",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuditFailureClass {
    CredentialsRejected,
    AuthorizationDenied,
    InvalidInput,
    Conflict,
    Unavailable,
    IntegrityViolation,
    Cancelled,
}

impl AuditFailureClass {
    const fn name(self) -> &'static str {
        match self {
            Self::CredentialsRejected => "credentials_rejected",
            Self::AuthorizationDenied => "authorization_denied",
            Self::InvalidInput => "invalid_input",
            Self::Conflict => "conflict",
            Self::Unavailable => "unavailable",
            Self::IntegrityViolation => "integrity_violation",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuditEventContext {
    community_id: CommunityId,
    operation_id: OperationId,
    actor_principal_id: Option<PrincipalId>,
    actor_kind: Option<AuditActorKind>,
    outcome: AuditOutcome,
    failure_class: Option<AuditFailureClass>,
    occurred_at_millis: u64,
}

impl AuditEventContext {
    pub fn new(
        tenant: &TenantContext,
        operation_id: OperationId,
        actor: Option<&AuthenticatedPrincipal>,
        outcome: AuditOutcome,
        failure_class: Option<AuditFailureClass>,
        occurred_at_millis: u64,
    ) -> Result<Self, AuditEventError> {
        if operation_id.as_uuid().is_nil()
            || occurred_at_millis == 0
            || matches!(outcome, AuditOutcome::Succeeded) != failure_class.is_none()
        {
            return Err(AuditEventError::InvalidEvent);
        }
        if actor.is_some_and(|actor| {
            actor.community_id() != tenant.community_id() || actor.principal_id().as_uuid().is_nil()
        }) {
            return Err(AuditEventError::TenantBoundaryViolation);
        }
        let (actor_principal_id, actor_kind) = actor
            .map(|actor| (Some(actor.principal_id()), Some(actor_kind(actor))))
            .unwrap_or((None, None));
        Ok(Self {
            community_id: tenant.community_id(),
            operation_id,
            actor_principal_id,
            actor_kind,
            outcome,
            failure_class,
            occurred_at_millis,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn actor_principal_id(self) -> Option<PrincipalId> {
        self.actor_principal_id
    }

    pub const fn outcome(self) -> AuditOutcome {
        self.outcome
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthenticationAuditMethod {
    ZedAccount,
    Nip42,
    Nip98,
    ScopedToken,
    Service,
}

impl AuthenticationAuditMethod {
    pub const fn for_principal(principal: &AuthenticatedPrincipal) -> Self {
        match principal.kind() {
            AuthenticatedPrincipalKind::SimAccount { .. } => Self::ZedAccount,
            AuthenticatedPrincipalKind::NostrIdentity {
                authentication_method,
                ..
            }
            | AuthenticatedPrincipalKind::OwnerAttestedAgent {
                authentication_method,
                ..
            } => match authentication_method {
                NostrAuthenticationMethod::Nip42 => Self::Nip42,
                NostrAuthenticationMethod::Nip98 => Self::Nip98,
            },
            AuthenticatedPrincipalKind::ScopedToken { .. } => Self::ScopedToken,
            AuthenticatedPrincipalKind::Service { .. } => Self::Service,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::ZedAccount => "zed_account",
            Self::Nip42 => "nip42",
            Self::Nip98 => "nip98",
            Self::ScopedToken => "scoped_token",
            Self::Service => "service",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipAuditOperation {
    Add,
    Remove,
    ChangeRole,
}

impl MembershipAuditOperation {
    const fn action(self) -> &'static str {
        match self {
            Self::Add => "membership.add",
            Self::Remove => "membership.remove",
            Self::ChangeRole => "membership.change_role",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationAuditOperation {
    ViewQueue,
    Report,
    ResolveReport,
    ApplyRestriction,
    LiftRestriction,
    ArchiveIdentity,
    ArchiveCommunity,
}

impl ModerationAuditOperation {
    pub const fn action(self) -> &'static str {
        match self {
            Self::ViewQueue => "moderation.view_queue",
            Self::Report => "moderation.report",
            Self::ResolveReport => "moderation.resolve_report",
            Self::ApplyRestriction => "moderation.apply_restriction",
            Self::LiftRestriction => "moderation.lift_restriction",
            Self::ArchiveIdentity => "moderation.archive_identity",
            Self::ArchiveCommunity => "moderation.archive_community",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowAuditOperation {
    Trigger,
    ExecuteStep,
    DecideApproval,
    CompleteRun,
    CancelRun,
}

impl WorkflowAuditOperation {
    const fn action(self) -> &'static str {
        match self {
            Self::Trigger => "workflow.trigger",
            Self::ExecuteStep => "workflow.execute_step",
            Self::DecideApproval => "workflow.decide_approval",
            Self::CompleteRun => "workflow.complete_run",
            Self::CancelRun => "workflow.cancel_run",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationAuditOperation {
    Start,
    Checkpoint,
    Complete,
    RollBack,
}

impl MigrationAuditOperation {
    const fn action(self) -> &'static str {
        match self {
            Self::Start => "migration.start",
            Self::Checkpoint => "migration.checkpoint",
            Self::Complete => "migration.complete",
            Self::RollBack => "migration.roll_back",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationAuditSource {
    BuzzV1,
    Native,
}

impl MigrationAuditSource {
    const fn name(self) -> &'static str {
        match self {
            Self::BuzzV1 => "buzz_v1",
            Self::Native => "native",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SecurityAuditEvent {
    Authentication {
        context: AuditEventContext,
        method: AuthenticationAuditMethod,
    },
    Membership {
        context: AuditEventContext,
        operation: MembershipAuditOperation,
        subject_principal_id: PrincipalId,
    },
    Moderation {
        context: AuditEventContext,
        operation: ModerationAuditOperation,
        subject_principal_id: Option<PrincipalId>,
        record_id: Uuid,
    },
    Workflow {
        context: AuditEventContext,
        operation: WorkflowAuditOperation,
        workflow_id: Uuid,
        run_id: Uuid,
        step_index: Option<u32>,
    },
    Migration {
        context: AuditEventContext,
        operation: MigrationAuditOperation,
        source: MigrationAuditSource,
        migration_id: Uuid,
        checkpoint: Option<u64>,
    },
}

impl SecurityAuditEvent {
    pub const fn context(&self) -> AuditEventContext {
        match self {
            Self::Authentication { context, .. }
            | Self::Membership { context, .. }
            | Self::Moderation { context, .. }
            | Self::Workflow { context, .. }
            | Self::Migration { context, .. } => *context,
        }
    }

    pub fn into_record(self) -> Result<AuditRecord, AuditEventError> {
        let context = self.context();
        let mut fields = common_fields(context)?;
        let action = match self {
            Self::Authentication { method, .. } => {
                if context.outcome == AuditOutcome::Succeeded
                    && context.actor_principal_id.is_none()
                {
                    return Err(AuditEventError::MissingAttribution);
                }
                fields.push(category_field("authentication_method", method.name())?);
                "auth.authenticate"
            }
            Self::Membership {
                operation,
                subject_principal_id,
                ..
            } => {
                require_actor(context)?;
                require_non_nil(subject_principal_id.as_uuid())?;
                fields.push(identifier_field(
                    "subject_principal_id",
                    subject_principal_id.as_uuid(),
                )?);
                operation.action()
            }
            Self::Moderation {
                operation,
                subject_principal_id,
                record_id,
                ..
            } => {
                require_actor(context)?;
                require_non_nil(record_id)?;
                if let Some(subject_principal_id) = subject_principal_id {
                    require_non_nil(subject_principal_id.as_uuid())?;
                    fields.push(identifier_field(
                        "subject_principal_id",
                        subject_principal_id.as_uuid(),
                    )?);
                }
                fields.push(identifier_field("moderation_record_id", record_id)?);
                operation.action()
            }
            Self::Workflow {
                operation,
                workflow_id,
                run_id,
                step_index,
                ..
            } => {
                require_actor(context)?;
                require_non_nil(workflow_id)?;
                require_non_nil(run_id)?;
                fields.push(identifier_field("workflow_id", workflow_id)?);
                fields.push(identifier_field("workflow_run_id", run_id)?);
                if let Some(step_index) = step_index {
                    fields.push(unsigned_field(
                        "workflow_step_index",
                        u64::from(step_index),
                    )?);
                }
                operation.action()
            }
            Self::Migration {
                operation,
                source,
                migration_id,
                checkpoint,
                ..
            } => {
                require_actor(context)?;
                require_non_nil(migration_id)?;
                fields.push(category_field("migration_source", source.name())?);
                fields.push(identifier_field("migration_id", migration_id)?);
                if let Some(checkpoint) = checkpoint {
                    fields.push(unsigned_field("migration_checkpoint", checkpoint)?);
                }
                operation.action()
            }
        };
        AuditRecord::new(
            context.operation_id,
            AuditAction::new(action)?,
            context.actor_principal_id,
            context.outcome,
            context.occurred_at_millis,
            AuditFields::new(fields)?,
        )
        .map_err(Into::into)
    }
}

pub struct AuditEventRecorder {
    repository: AuditRepository,
}

impl AuditEventRecorder {
    pub const fn new(repository: AuditRepository) -> Self {
        Self { repository }
    }

    pub fn into_repository(self) -> AuditRepository {
        self.repository
    }

    pub async fn record(
        &self,
        tenant: &TenantContext,
        expected_head: ExpectedAuditHead,
        event: SecurityAuditEvent,
    ) -> Result<AuditEntry, AuditEventError> {
        if event.context().community_id != tenant.community_id() {
            return Err(AuditEventError::TenantBoundaryViolation);
        }
        let record = event.into_record()?;
        self.repository
            .append(tenant, expected_head, record)
            .await
            .map_err(Into::into)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuditEventError {
    #[error("audit event crossed its admitted community boundary")]
    TenantBoundaryViolation,
    #[error("audit event is invalid")]
    InvalidEvent,
    #[error("administrative audit event requires an admitted actor")]
    MissingAttribution,
    #[error("audit event violates the canonical record contract")]
    Domain(#[from] AuditError),
    #[error("audit event could not be persisted")]
    Repository(#[from] AuditRepositoryError),
}

fn actor_kind(principal: &AuthenticatedPrincipal) -> AuditActorKind {
    match principal.kind() {
        AuthenticatedPrincipalKind::SimAccount { .. } => AuditActorKind::ZedAccount,
        AuthenticatedPrincipalKind::NostrIdentity { .. } => AuditActorKind::NostrIdentity,
        AuthenticatedPrincipalKind::OwnerAttestedAgent { .. } => AuditActorKind::OwnerAttestedAgent,
        AuthenticatedPrincipalKind::ScopedToken { .. } => AuditActorKind::ScopedToken,
        AuthenticatedPrincipalKind::Service { .. } => AuditActorKind::Service,
    }
}

fn require_actor(context: AuditEventContext) -> Result<(), AuditEventError> {
    context
        .actor_principal_id
        .is_some()
        .then_some(())
        .ok_or(AuditEventError::MissingAttribution)
}

fn require_non_nil(value: Uuid) -> Result<(), AuditEventError> {
    (!value.is_nil())
        .then_some(())
        .ok_or(AuditEventError::InvalidEvent)
}

fn common_fields(context: AuditEventContext) -> Result<Vec<AuditField>, AuditEventError> {
    let mut fields = Vec::with_capacity(6);
    if let Some(actor_kind) = context.actor_kind {
        fields.push(category_field("actor_kind", actor_kind.name())?);
    }
    if let Some(failure_class) = context.failure_class {
        fields.push(category_field("failure_class", failure_class.name())?);
    }
    Ok(fields)
}

fn category_field(name: &'static str, value: &'static str) -> Result<AuditField, AuditEventError> {
    Ok(AuditField::new(
        AuditFieldName::new(name)?,
        AuditValue::Category(AuditCategory::new(value)?),
    ))
}

fn identifier_field(name: &'static str, value: Uuid) -> Result<AuditField, AuditEventError> {
    Ok(AuditField::new(
        AuditFieldName::new(name)?,
        AuditValue::Identifier(AuditIdentifier::new(value.to_string())?),
    ))
}

fn unsigned_field(name: &'static str, value: u64) -> Result<AuditField, AuditEventError> {
    Ok(AuditField::new(
        AuditFieldName::new(name)?,
        AuditValue::Unsigned(value),
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use collaboration_domain::{AuditValue, PrincipalScopes, ServiceAccountId, TrustedTenantRoute};
    use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};

    use super::*;
    use crate::audit::repository::{AuditHead, ExpectedAuditHead};

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn tenant(value: u128) -> TenantContext {
        let community_id = community(value);
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "audit-events")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn actor(tenant: &TenantContext, value: u128) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            PrincipalId::from_uuid(Uuid::from_u128(value)),
            tenant.community_id(),
            ServiceAccountId::new(u64::try_from(value).expect("fixture account")),
            PrincipalScopes::default(),
        )
    }

    fn repository(
        query_results: Vec<Vec<BTreeMap<String, SeaValue>>>,
        affected_rows: &[u64],
    ) -> AuditRepository {
        let database = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results(query_results)
            .append_exec_results(affected_rows.iter().copied().map(|rows_affected| {
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected,
                }
            }))
            .into_connection();
        AuditRepository::new(database).expect("repository")
    }

    fn head_row(head: AuditHead) -> BTreeMap<String, SeaValue> {
        BTreeMap::from([
            ("sequence_text".into(), head.sequence().to_string().into()),
            ("entry_hash".into(), head.hash().as_bytes().to_vec().into()),
        ])
    }

    fn context(
        tenant: &TenantContext,
        actor: Option<&AuthenticatedPrincipal>,
        operation: u128,
        outcome: AuditOutcome,
        failure: Option<AuditFailureClass>,
    ) -> AuditEventContext {
        AuditEventContext::new(
            tenant,
            OperationId::from_uuid(Uuid::from_u128(operation)),
            actor,
            outcome,
            failure,
            1_900_000_000_000 + u64::try_from(operation).expect("fixture operation"),
        )
        .expect("context")
    }

    #[tokio::test]
    async fn recorder_persists_attributable_success_and_payload_free_failure() {
        let tenant = tenant(1);
        let actor = actor(&tenant, 2);
        let success_event = SecurityAuditEvent::Authentication {
            context: context(&tenant, Some(&actor), 10, AuditOutcome::Succeeded, None),
            method: AuthenticationAuditMethod::for_principal(&actor),
        };
        let expected_success = AuditEntry::append(
            collaboration_domain::AuditChainPosition::genesis(tenant.community_id())
                .expect("genesis"),
            success_event.clone().into_record().expect("success record"),
        )
        .expect("success entry");
        let failure_event = SecurityAuditEvent::Membership {
            context: context(
                &tenant,
                Some(&actor),
                11,
                AuditOutcome::Denied,
                Some(AuditFailureClass::AuthorizationDenied),
            ),
            operation: MembershipAuditOperation::Remove,
            subject_principal_id: PrincipalId::from_uuid(Uuid::from_u128(3)),
        };
        let repository = repository(
            vec![
                vec![],
                vec![head_row(AuditHead::from_entry(&expected_success))],
            ],
            &[1, 1, 1, 1, 1, 1, 1, 1],
        );
        let recorder = AuditEventRecorder::new(repository);

        let success = recorder
            .record(&tenant, ExpectedAuditHead::Empty, success_event)
            .await
            .expect("success audit");
        let failure = recorder
            .record(
                &tenant,
                ExpectedAuditHead::Entry(AuditHead::from_entry(&success)),
                failure_event,
            )
            .await
            .expect("failure audit");

        assert_eq!(
            success.record().operation_id().as_uuid(),
            Uuid::from_u128(10)
        );
        assert_eq!(
            success.record().actor_principal_id(),
            Some(actor.principal_id())
        );
        assert_eq!(success.record().outcome(), AuditOutcome::Succeeded);
        assert_eq!(
            failure.record().operation_id().as_uuid(),
            Uuid::from_u128(11)
        );
        assert_eq!(
            failure.record().actor_principal_id(),
            Some(actor.principal_id())
        );
        assert_eq!(failure.record().outcome(), AuditOutcome::Denied);
        assert_eq!(failure.previous_hash(), Some(success.hash()));
        assert_eq!(
            failure
                .record()
                .fields()
                .as_slice()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            ["actor_kind", "failure_class", "subject_principal_id"]
        );
        assert!(
            failure
                .record()
                .fields()
                .as_slice()
                .iter()
                .all(|field| !matches!(field.value(), AuditValue::Redacted(_)))
        );

        let transaction_log = format!(
            "{:#?}",
            recorder
                .into_repository()
                .into_connection()
                .into_transaction_log()
        );
        assert!(transaction_log.contains("auth.authenticate"));
        assert!(transaction_log.contains("membership.remove"));
        assert!(transaction_log.contains("authorization_denied"));
        for private_value in ["bearer-token", "request-body", "error-detail"] {
            assert!(!transaction_log.contains(private_value));
        }
    }

    #[test]
    fn every_event_family_carries_the_supplied_operation_and_closed_fields() {
        let tenant = tenant(1);
        let actor = actor(&tenant, 2);
        let contexts = (20_u128..24).map(|operation| {
            context(
                &tenant,
                Some(&actor),
                operation,
                AuditOutcome::Succeeded,
                None,
            )
        });
        let mut contexts = contexts.into_iter();
        let events = [
            SecurityAuditEvent::Membership {
                context: contexts.next().expect("membership context"),
                operation: MembershipAuditOperation::ChangeRole,
                subject_principal_id: PrincipalId::from_uuid(Uuid::from_u128(30)),
            },
            SecurityAuditEvent::Moderation {
                context: contexts.next().expect("moderation context"),
                operation: ModerationAuditOperation::ApplyRestriction,
                subject_principal_id: Some(PrincipalId::from_uuid(Uuid::from_u128(30))),
                record_id: Uuid::from_u128(31),
            },
            SecurityAuditEvent::Workflow {
                context: contexts.next().expect("workflow context"),
                operation: WorkflowAuditOperation::DecideApproval,
                workflow_id: Uuid::from_u128(32),
                run_id: Uuid::from_u128(33),
                step_index: Some(1),
            },
            SecurityAuditEvent::Migration {
                context: contexts.next().expect("migration context"),
                operation: MigrationAuditOperation::Checkpoint,
                source: MigrationAuditSource::BuzzV1,
                migration_id: Uuid::from_u128(34),
                checkpoint: Some(5),
            },
        ];

        for (offset, event) in events.into_iter().enumerate() {
            let record = event.into_record().expect("record");
            assert_eq!(
                record.operation_id().as_uuid(),
                Uuid::from_u128(20 + u128::try_from(offset).expect("fixture offset"))
            );
            assert_eq!(record.actor_principal_id(), Some(actor.principal_id()));
            assert!(record.fields().as_slice().iter().all(|field| matches!(
                field.value(),
                AuditValue::Identifier(_) | AuditValue::Category(_) | AuditValue::Unsigned(_)
            )));
        }
    }

    #[test]
    fn event_boundary_rejects_cross_tenant_or_unattributed_administration() {
        let admitted_tenant = tenant(1);
        let other_tenant = tenant(2);
        let other_actor = actor(&other_tenant, 3);
        assert!(matches!(
            AuditEventContext::new(
                &admitted_tenant,
                OperationId::from_uuid(Uuid::from_u128(40)),
                Some(&other_actor),
                AuditOutcome::Succeeded,
                None,
                1_900_000_000_040,
            ),
            Err(AuditEventError::TenantBoundaryViolation)
        ));

        let event = SecurityAuditEvent::Migration {
            context: context(&admitted_tenant, None, 41, AuditOutcome::Succeeded, None),
            operation: MigrationAuditOperation::Start,
            source: MigrationAuditSource::BuzzV1,
            migration_id: Uuid::from_u128(42),
            checkpoint: None,
        };
        assert!(matches!(
            event.into_record(),
            Err(AuditEventError::MissingAttribution)
        ));
    }

    #[test]
    fn failed_events_require_a_closed_failure_class() {
        let tenant = tenant(1);
        let actor = actor(&tenant, 2);
        assert!(matches!(
            AuditEventContext::new(
                &tenant,
                OperationId::from_uuid(Uuid::from_u128(50)),
                Some(&actor),
                AuditOutcome::Failed,
                None,
                1_900_000_000_050,
            ),
            Err(AuditEventError::InvalidEvent)
        ));
        assert!(matches!(
            AuditEventContext::new(
                &tenant,
                OperationId::from_uuid(Uuid::from_u128(51)),
                Some(&actor),
                AuditOutcome::Succeeded,
                Some(AuditFailureClass::Unavailable),
                1_900_000_000_051,
            ),
            Err(AuditEventError::InvalidEvent)
        ));
    }
}
