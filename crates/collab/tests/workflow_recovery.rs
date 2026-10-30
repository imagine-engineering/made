use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;
use collab::workflows::{
    actions::{
        CanonicalWorkflowCommand, CanonicalWorkflowCommandSink, ResolvedActionSecret,
        WorkflowActionAttempt, WorkflowActionAuthority, WorkflowActionAuthorization,
        WorkflowActionError, WorkflowActionExecutor, WorkflowActionOutcome,
        WorkflowActionSecretResolver, WorkflowCommandDisposition, WorkflowCommandReceipt,
        bounded_action_error,
    },
    approval::{
        ApprovalDecision, ApprovalDecisionWrite, ApprovalOutboxKind, WorkflowApprovalDisposition,
        WorkflowApprovalError, WorkflowApprovalRepository,
    },
    repository::{
        RetryFailureClass as RepositoryRetryFailureClass, StoredWorkflowDefinition,
        StoredWorkflowRun, StoredWorkflowStep, WorkflowIdentity, WorkflowLifecycle,
        WorkflowProvenance, WorkflowRepository, WorkflowRepositoryError, WorkflowRetryWrite,
        WorkflowRunIdentity, WorkflowRunLease, WorkflowRunLeaseFence, WorkflowRunLeaseState,
        WorkflowRunRequest, WorkflowRunState, WorkflowScope, WorkflowStepCheckpoint,
        WorkflowStepState, WorkflowStoreOutcome, WorkflowTriggerKind,
    },
    triggers::{
        CollaborationEventTrigger, CollaborationEventTriggerKind, EVENT_TRIGGER_SCOPE,
        ScheduleClock, WORKFLOW_RUN_SCOPE, WorkflowRunClaimer, WorkflowTriggerAdmission,
        WorkflowTriggerAdmissionStatus, evaluate_schedule,
    },
    webhook::{
        ResolvedWebhookCredential, WebhookAdmissionError, WebhookAuthentication,
        WebhookCredentialReference, WebhookCredentialResolver, WorkflowWebhookAdmission,
        webhook_signature_v1,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus,
    OperationId, PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use collaboration_workflow::definition::{
    RetryFailureClass as DefinitionRetryFailureClass, WorkflowDefinition,
};
use futures::stream;
use sea_orm::{DatabaseBackend, DbErr, MockDatabase, MockExecResult, Value};
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const NOW: u64 = 1_900_000_320_000;
const WEBHOOK_SECRET: &[u8; 32] = b"0123456789abcdef0123456789abcdef";

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "workflow-recovery")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn provenance(record: &str) -> WorkflowProvenance {
    WorkflowProvenance::new("zed", record, "1", NOW, None).expect("provenance")
}

fn workflow_definition(trigger: &str, action: &str) -> WorkflowDefinition {
    WorkflowDefinition::parse_yaml(&format!(
        r#"
version: 1
name: Recovery
enabled: true
trigger:
{trigger}
steps:
  - id: recover
{action}
"#
    ))
    .expect("workflow definition")
}

fn stored_definition(
    community_id: CommunityId,
    trigger: &str,
    action: &str,
) -> StoredWorkflowDefinition {
    let definition = workflow_definition(trigger, action);
    let encoded = serde_json::to_vec(&definition).expect("definition JSON");
    StoredWorkflowDefinition {
        identity: WorkflowIdentity::new(community_id, Uuid::from_u128(10))
            .expect("workflow identity"),
        definition_version: 1,
        definition,
        definition_sha256: Sha256::digest(encoded).into(),
        creator_principal_id: principal(20),
        author_principal_id: principal(20),
        scope: WorkflowScope::Community,
        current_definition_version: 1,
        head_revision: 1,
        lifecycle: WorkflowLifecycle::Active,
        provenance: provenance("workflow:recovery:1"),
        created_at_millis: NOW,
    }
}

struct AuthorizationFixture {
    tenant: TenantContext,
    source_scope: AuthorizationScope,
    owner_scope: AuthorizationScope,
    source: AuthenticatedPrincipal,
    owner: AuthenticatedPrincipal,
    source_membership: CommunityMembership,
    owner_membership: CommunityMembership,
    source_channel_membership: ChannelMembership,
    channel_id: AggregateId,
    workflow_id: AggregateId,
}

impl AuthorizationFixture {
    fn new(community_id: CommunityId) -> Self {
        let source_principal_id = principal(30);
        let owner_principal_id = principal(20);
        let source_scope = AuthorizationScope::new(EVENT_TRIGGER_SCOPE).expect("source scope");
        let owner_scope = AuthorizationScope::new(WORKFLOW_RUN_SCOPE).expect("owner scope");
        let channel_id = aggregate(40);
        Self {
            tenant: tenant(community_id),
            source: AuthenticatedPrincipal::zed_account(
                source_principal_id,
                community_id,
                ServiceAccountId::new(30),
                PrincipalScopes::new([source_scope.clone()]).expect("source scopes"),
            ),
            owner: AuthenticatedPrincipal::zed_account(
                owner_principal_id,
                community_id,
                ServiceAccountId::new(20),
                PrincipalScopes::new([owner_scope.clone()]).expect("owner scopes"),
            ),
            source_membership: CommunityMembership {
                community_id,
                principal_id: source_principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            owner_membership: CommunityMembership {
                community_id,
                principal_id: owner_principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            source_channel_membership: ChannelMembership {
                community_id,
                channel_id,
                principal_id: source_principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            channel_id,
            workflow_id: aggregate(10),
            source_scope,
            owner_scope,
        }
    }

    fn source_request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.source,
            required_scope: &self.source_scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Channel,
                resource_id: self.channel_id,
                owner_principal_id: None,
                channel_id: Some(self.channel_id),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.source_membership),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(self.source_channel_membership),
            delegation: None,
            now_millis: NOW,
        }
    }

    fn owner_request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.owner,
            required_scope: &self.owner_scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Workflow,
                resource_id: self.workflow_id,
                owner_principal_id: Some(principal(20)),
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.owner_membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: NOW,
        }
    }
}

#[derive(Clone, Default)]
struct DurableRunClaimer {
    requests: Arc<Mutex<BTreeMap<Uuid, WorkflowRunRequest>>>,
    fail_next_claim: Arc<AtomicBool>,
}

impl DurableRunClaimer {
    fn fail_next_claim(&self) {
        self.fail_next_claim.store(true, Ordering::SeqCst);
    }

    fn requests(&self) -> Vec<WorkflowRunRequest> {
        self.requests
            .lock()
            .expect("run claims")
            .values()
            .cloned()
            .collect()
    }
}

#[async_trait]
impl WorkflowRunClaimer for DurableRunClaimer {
    async fn claim_run(
        &self,
        _tenant: &TenantContext,
        request: &WorkflowRunRequest,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        if self.fail_next_claim.swap(false, Ordering::SeqCst) {
            return Err(WorkflowRepositoryError::Unavailable(DbErr::Custom(
                "injected database restart".to_owned(),
            )));
        }
        let mut requests = self.requests.lock().expect("run claims");
        if let Some(existing) = requests.get(&request.trigger_operation_id) {
            return if existing == request {
                Ok(WorkflowStoreOutcome::Duplicate)
            } else {
                Err(WorkflowRepositoryError::IdempotencyConflict)
            };
        }
        requests.insert(request.trigger_operation_id, request.clone());
        Ok(WorkflowStoreOutcome::Applied)
    }
}

#[derive(Clone, Copy)]
struct RecoveryCredentialResolver;

#[async_trait]
impl WebhookCredentialResolver for RecoveryCredentialResolver {
    async fn resolve(
        &self,
        _tenant: &TenantContext,
        _workflow: WorkflowIdentity,
        reference: &WebhookCredentialReference,
    ) -> Result<ResolvedWebhookCredential, WebhookAdmissionError> {
        if reference.as_str() != "credentials://workflow/recovery" {
            return Err(WebhookAdmissionError::CredentialUnavailable);
        }
        ResolvedWebhookCredential::new("rotation-7", WEBHOOK_SECRET.to_vec())
    }
}

fn webhook_authentication(
    tenant: &TenantContext,
    definition: &StoredWorkflowDefinition,
    body: &[u8],
) -> WebhookAuthentication {
    let content_sha256: [u8; 32] = Sha256::digest(body).into();
    let signature = webhook_signature_v1(
        WEBHOOK_SECRET,
        tenant,
        definition.identity,
        NOW,
        "delivery-recovery-1",
        content_sha256,
    )
    .expect("signature");
    WebhookAuthentication::new(
        NOW,
        "delivery-recovery-1",
        &hex::encode(content_sha256),
        &hex::encode(signature),
    )
    .expect("authentication")
}

#[tokio::test]
async fn trigger_replay_converges_after_database_and_process_restarts() {
    let community_id = community(1);
    let authorization = AuthorizationFixture::new(community_id);
    let event_claims = DurableRunClaimer::default();
    let event_definition = stored_definition(
        community_id,
        "  on: message_posted",
        "    action: send_message\n    text: recovered",
    );
    let event = CollaborationEventTrigger::new(
        community_id,
        [5; 32],
        CollaborationEventTriggerKind::MessagePosted,
        authorization.channel_id,
        principal(30),
        [6; 32],
        NOW - 1_000,
        NOW,
        Some("recover".to_owned()),
        None,
        None,
    )
    .expect("event");

    event_claims.fail_next_claim();
    let failed_process = WorkflowTriggerAdmission::new(event_claims.clone());
    let error = failed_process
        .admit_event(
            &authorization.tenant,
            &event_definition,
            &event,
            &authorization.source_request(),
            &authorization.owner_request(),
        )
        .await
        .expect_err("database failure must not claim a run");
    assert!(error.to_string().contains("repository is unavailable"));
    assert!(event_claims.requests().is_empty());

    let restarted_process = WorkflowTriggerAdmission::new(event_claims.clone());
    let event_claim = restarted_process
        .admit_event(
            &authorization.tenant,
            &event_definition,
            &event,
            &authorization.source_request(),
            &authorization.owner_request(),
        )
        .await
        .expect("event after restart");
    let event_replay = WorkflowTriggerAdmission::new(event_claims.clone())
        .admit_event(
            &authorization.tenant,
            &event_definition,
            &event,
            &authorization.source_request(),
            &authorization.owner_request(),
        )
        .await
        .expect("event replay");
    assert_eq!(event_claim.status, WorkflowTriggerAdmissionStatus::Claimed);
    assert_eq!(
        event_replay.status,
        WorkflowTriggerAdmissionStatus::Duplicate
    );
    assert_eq!(event_claim.run_identity, event_replay.run_identity);
    assert_eq!(event_claims.requests().len(), 1);

    let cron_claims = DurableRunClaimer::default();
    let cron_definition = stored_definition(
        community_id,
        "  on: schedule\n  cron: '* * * * *'",
        "    action: send_message\n    text: scheduled",
    );
    let evaluation = evaluate_schedule(
        &cron_definition.definition,
        Some(NOW - 61_000),
        ScheduleClock::new(NOW, NOW + 30_000).expect("clock"),
    )
    .expect("cron evaluation");
    let fire = *evaluation.fires().last().expect("missed cron fire");
    let cron_claim = WorkflowTriggerAdmission::new(cron_claims.clone())
        .admit_schedule_fire(
            &authorization.tenant,
            &cron_definition,
            fire,
            &authorization.owner_request(),
        )
        .await
        .expect("cron claim");
    let cron_replay = WorkflowTriggerAdmission::new(cron_claims.clone())
        .admit_schedule_fire(
            &authorization.tenant,
            &cron_definition,
            fire,
            &authorization.owner_request(),
        )
        .await
        .expect("cron replay");
    assert_eq!(cron_claim.status, WorkflowTriggerAdmissionStatus::Claimed);
    assert_eq!(
        cron_replay.status,
        WorkflowTriggerAdmissionStatus::Duplicate
    );
    assert_eq!(cron_claim.run_identity, cron_replay.run_identity);
    assert_eq!(cron_claims.requests().len(), 1);

    let webhook_claims = DurableRunClaimer::default();
    let webhook_definition = stored_definition(
        community_id,
        "  on: webhook",
        "    action: send_message\n    text: webhook",
    );
    let body = br#"{"message":"recover"}"#;
    let authentication = webhook_authentication(&authorization.tenant, &webhook_definition, body);
    let credential = WebhookCredentialReference::new("credentials://workflow/recovery")
        .expect("credential reference");
    let webhook_claim =
        WorkflowWebhookAdmission::new(webhook_claims.clone(), RecoveryCredentialResolver)
            .admit(
                &authorization.tenant,
                &webhook_definition,
                &credential,
                &authentication,
                Some(body.len() as u64),
                stream::iter([Ok(Bytes::from_static(body))]),
                &authorization.owner_request(),
                NOW,
            )
            .await
            .expect("webhook claim");
    let webhook_replay =
        WorkflowWebhookAdmission::new(webhook_claims.clone(), RecoveryCredentialResolver)
            .admit(
                &authorization.tenant,
                &webhook_definition,
                &credential,
                &authentication,
                Some(body.len() as u64),
                stream::iter([Ok(Bytes::from_static(body))]),
                &authorization.owner_request(),
                NOW,
            )
            .await
            .expect("webhook replay");
    assert_eq!(
        webhook_claim.status,
        WorkflowTriggerAdmissionStatus::Claimed
    );
    assert_eq!(
        webhook_replay.status,
        WorkflowTriggerAdmissionStatus::Duplicate
    );
    assert_eq!(webhook_claim.run_identity, webhook_replay.run_identity);
    assert_eq!(webhook_claims.requests().len(), 1);
}

fn action_actor(community_id: CommunityId) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::service(
        principal(20),
        community_id,
        "workflow-owner",
        PrincipalScopes::new([
            AuthorizationScope::new("workflows:run").expect("workflow scope"),
            AuthorizationScope::new("messages:write").expect("message scope"),
        ])
        .expect("scopes"),
    )
    .expect("actor")
}

struct ActionFixture {
    tenant: TenantContext,
    definition: StoredWorkflowDefinition,
    run: StoredWorkflowRun,
    lease: WorkflowRunLeaseFence,
    outputs: BTreeMap<String, JsonValue>,
}

impl ActionFixture {
    fn new() -> Self {
        let community_id = community(1);
        let definition = stored_definition(
            community_id,
            "  on: message_posted",
            "    action: send_message\n    text: 'recover {{ trigger.text }}'",
        );
        let identity =
            WorkflowRunIdentity::new(community_id, Uuid::from_u128(30)).expect("run identity");
        let step = StoredWorkflowStep {
            index: 0,
            step_id: "recover".to_owned(),
            operation_id: Uuid::from_u128(40),
            state: WorkflowStepState::Running,
            attempt_count: 1,
            output: None,
            error_code: None,
            error_message: None,
            created_at_millis: NOW,
            started_at_millis: Some(NOW),
            completed_at_millis: None,
            updated_at_millis: NOW,
        };
        let run = StoredWorkflowRun {
            identity,
            workflow: definition.identity,
            definition_version: 1,
            trigger_operation_id: Uuid::from_u128(50),
            trigger_kind: WorkflowTriggerKind::Event,
            trigger_source_id: "event:recovery".to_owned(),
            trigger_context: json!({"text": "after crash"}),
            run_version: 2,
            state: WorkflowRunState::Running,
            current_step_index: 0,
            error_code: None,
            error_message: None,
            provenance: provenance("run:recovery:1"),
            created_at_millis: NOW,
            started_at_millis: Some(NOW),
            completed_at_millis: None,
            updated_at_millis: NOW,
            steps: vec![step],
            retries: Vec::new(),
        };
        let lease = WorkflowRunLease {
            identity,
            admitted_run_version: 2,
            generation: 1,
            lease_id: Uuid::from_u128(60),
            worker_id: "worker-a".to_owned(),
            state: WorkflowRunLeaseState::Active,
            acquired_at_millis: NOW,
            last_heartbeat_at_millis: NOW,
            expires_at_millis: NOW + 10_000,
            recovery_after_millis: NOW + 20_000,
            released_at_millis: None,
            release_reason: None,
        };
        Self {
            tenant: tenant(community_id),
            definition,
            run,
            lease: WorkflowRunLeaseFence::from(&lease),
            outputs: BTreeMap::new(),
        }
    }

    fn attempt(&self) -> WorkflowActionAttempt<'_> {
        WorkflowActionAttempt {
            tenant: &self.tenant,
            definition: &self.definition,
            run: &self.run,
            step: &self.run.steps[0],
            lease: &self.lease,
            default_channel: Some("00000000-0000-0000-0000-000000000070"),
            previous_step_outputs: &self.outputs,
        }
    }
}

#[derive(Clone)]
struct RecoveryAuthority {
    actor: AuthenticatedPrincipal,
    allowed: bool,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl WorkflowActionAuthority for RecoveryAuthority {
    async fn authorize(
        &self,
        _request: &WorkflowActionAuthorization<'_>,
    ) -> Result<AuthenticatedPrincipal, WorkflowActionError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.allowed {
            Ok(self.actor.clone())
        } else {
            Err(WorkflowActionError::PermissionDenied)
        }
    }
}

#[derive(Clone, Default)]
struct CommitThenDisconnectCommands {
    effects: Arc<Mutex<BTreeMap<OperationId, CanonicalWorkflowCommand>>>,
    submissions: Arc<AtomicUsize>,
}

#[async_trait]
impl CanonicalWorkflowCommandSink for CommitThenDisconnectCommands {
    async fn submit(
        &self,
        tenant: &TenantContext,
        actor: &AuthenticatedPrincipal,
        operation_id: OperationId,
        command: CanonicalWorkflowCommand,
    ) -> Result<WorkflowCommandReceipt, WorkflowActionError> {
        assert_eq!(tenant.community_id(), actor.community_id());
        self.submissions.fetch_add(1, Ordering::SeqCst);
        let mut effects = self.effects.lock().expect("effects");
        if let Some(existing) = effects.get(&operation_id) {
            assert_eq!(existing, &command);
        } else {
            effects.insert(operation_id, command);
            return Err(WorkflowActionError::CommandUnavailable);
        }
        Ok(WorkflowCommandReceipt {
            operation_id,
            disposition: WorkflowCommandDisposition::Duplicate,
            output: json!({"canonical": true}),
        })
    }
}

#[derive(Clone, Copy)]
struct NoActionSecrets;

#[async_trait]
impl WorkflowActionSecretResolver for NoActionSecrets {
    async fn resolve(
        &self,
        _tenant: &TenantContext,
        _workflow_id: Uuid,
        _secret_name: &str,
        _credential_reference: &str,
    ) -> Result<ResolvedActionSecret, WorkflowActionError> {
        Err(WorkflowActionError::SecretUnavailable)
    }
}

#[tokio::test]
async fn action_replay_reauthorizes_and_converges_after_lost_dependency_receipt() {
    let fixture = ActionFixture::new();
    let authority_calls = Arc::new(AtomicUsize::new(0));
    let commands = CommitThenDisconnectCommands::default();
    let first_process = WorkflowActionExecutor::system(
        RecoveryAuthority {
            actor: action_actor(fixture.tenant.community_id()),
            allowed: true,
            calls: Arc::clone(&authority_calls),
        },
        commands.clone(),
        NoActionSecrets,
    );

    let failure = first_process
        .execute(fixture.attempt())
        .await
        .expect_err("dependency response is lost after commit");
    assert_eq!(
        failure.retry_failure_class(),
        Some(DefinitionRetryFailureClass::TemporaryUnavailable)
    );
    assert_eq!(commands.effects.lock().expect("effects").len(), 1);

    let restarted_process = WorkflowActionExecutor::system(
        RecoveryAuthority {
            actor: action_actor(fixture.tenant.community_id()),
            allowed: true,
            calls: Arc::clone(&authority_calls),
        },
        commands.clone(),
        NoActionSecrets,
    );
    let replay = restarted_process
        .execute(fixture.attempt())
        .await
        .expect("stable operation replay");
    assert!(matches!(
        replay,
        WorkflowActionOutcome::Completed {
            disposition: WorkflowCommandDisposition::Duplicate,
            ..
        }
    ));
    assert_eq!(authority_calls.load(Ordering::SeqCst), 2);
    assert_eq!(commands.effects.lock().expect("effects").len(), 1);

    let revoked_process = WorkflowActionExecutor::system(
        RecoveryAuthority {
            actor: action_actor(fixture.tenant.community_id()),
            allowed: false,
            calls: Arc::clone(&authority_calls),
        },
        commands.clone(),
        NoActionSecrets,
    );
    assert_eq!(
        revoked_process.execute(fixture.attempt()).await,
        Err(WorkflowActionError::PermissionDenied)
    );
    assert_eq!(commands.submissions.load(Ordering::SeqCst), 2);
    assert_eq!(authority_calls.load(Ordering::SeqCst), 3);

    let mut cancelled_fixture = ActionFixture::new();
    cancelled_fixture.run.state = WorkflowRunState::Cancelled;
    let cancelled_process = WorkflowActionExecutor::system(
        RecoveryAuthority {
            actor: action_actor(cancelled_fixture.tenant.community_id()),
            allowed: true,
            calls: Arc::clone(&authority_calls),
        },
        commands.clone(),
        NoActionSecrets,
    );
    assert_eq!(
        cancelled_process.execute(cancelled_fixture.attempt()).await,
        Err(WorkflowActionError::StaleAttempt)
    );
    assert_eq!(commands.submissions.load(Ordering::SeqCst), 2);
    assert_eq!(authority_calls.load(Ordering::SeqCst), 3);

    let internal_dependency_detail = "postgres://operator:secret@private/workflows";
    let public_failure = bounded_action_error(&failure);
    assert!(!public_failure.contains(internal_dependency_detail));
    assert_eq!(public_failure, "canonical command service is unavailable");
    let secret =
        ResolvedActionSecret::new("rotation-7", internal_dependency_detail.as_bytes().to_vec())
            .expect("resolved secret");
    let redacted = format!("{secret:?}");
    assert!(redacted.contains("[REDACTED]"));
    assert!(!redacted.contains(internal_dependency_detail));
    assert!(WorkflowActionError::AmbiguousDelivery.requires_repair());
    assert_eq!(
        WorkflowActionError::AmbiguousDelivery.retry_failure_class(),
        None
    );
}

fn workflow_repository(
    query_results: Vec<Vec<BTreeMap<String, Value>>>,
    affected_rows: &[u64],
) -> WorkflowRepository {
    let connection =
        MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results(query_results)
            .append_exec_results(affected_rows.iter().copied().map(|rows_affected| {
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected,
                }
            }))
            .into_connection();
    WorkflowRepository::new(connection).expect("workflow repository")
}

fn retry_row(retry: &WorkflowRetryWrite) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("run_id".into(), retry.identity.run_id().into()),
        (
            "step_index".into(),
            i16::try_from(retry.step_index).expect("index").into(),
        ),
        (
            "attempt_number".into(),
            i16::try_from(retry.attempt_number).expect("attempt").into(),
        ),
        ("failure_class".into(), "timeout".into()),
        ("state".into(), "scheduled".into()),
        (
            "scheduled_at_millis".into(),
            i64::try_from(retry.scheduled_at_millis)
                .expect("timestamp")
                .into(),
        ),
        (
            "due_at_millis".into(),
            i64::try_from(retry.due_at_millis)
                .expect("timestamp")
                .into(),
        ),
        ("source_system".into(), "zed".into()),
        ("source_record_id".into(), "retry:recover:2".into()),
        ("source_version".into(), "1".into()),
        (
            "source_observed_at_millis".into(),
            i64::try_from(NOW).expect("timestamp").into(),
        ),
        (
            "created_at_millis".into(),
            i64::try_from(retry.created_at_millis)
                .expect("timestamp")
                .into(),
        ),
    ])
}

#[tokio::test]
async fn timeout_retry_checkpoint_is_idempotent_across_repository_restart() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let retry = WorkflowRetryWrite {
        identity: WorkflowRunIdentity::new(community_id, Uuid::from_u128(30))
            .expect("run identity"),
        step_index: 0,
        attempt_number: 2,
        retry_operation_id: Uuid::from_u128(70),
        failure_class: RepositoryRetryFailureClass::Timeout,
        scheduled_at_millis: NOW,
        due_at_millis: NOW + 1_000,
        provenance: provenance("retry:recover:2"),
        created_at_millis: NOW,
    };
    let first_process = workflow_repository(vec![vec![]], &[1, 1]);
    assert_eq!(
        first_process
            .record_retry(&tenant, &retry)
            .await
            .expect("record retry"),
        WorkflowStoreOutcome::Applied
    );

    let restarted_process = workflow_repository(vec![vec![retry_row(&retry)]], &[1]);
    assert_eq!(
        restarted_process
            .record_retry(&tenant, &retry)
            .await
            .expect("replay retry"),
        WorkflowStoreOutcome::Duplicate
    );
}

#[tokio::test]
async fn stale_lease_cannot_checkpoint_after_cancelled_worker_restart() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let identity =
        WorkflowRunIdentity::new(community_id, Uuid::from_u128(30)).expect("run identity");
    let stale_lease = WorkflowRunLease {
        identity,
        admitted_run_version: 2,
        generation: 1,
        lease_id: Uuid::from_u128(60),
        worker_id: "worker-before-restart".to_owned(),
        state: WorkflowRunLeaseState::Active,
        acquired_at_millis: NOW,
        last_heartbeat_at_millis: NOW,
        expires_at_millis: NOW + 100,
        recovery_after_millis: NOW + 200,
        released_at_millis: None,
        release_reason: None,
    };
    let repository = workflow_repository(vec![vec![]], &[1]);
    let error = repository
        .checkpoint_step(
            &tenant,
            &WorkflowStepCheckpoint {
                identity,
                expected_run_version: 2,
                step_index: 0,
                operation_id: Uuid::from_u128(40),
                expected_step_state: WorkflowStepState::Running,
                next_step_state: WorkflowStepState::Cancelled,
                next_run_state: WorkflowRunState::Cancelled,
                next_step_index: 0,
                attempt_count: 1,
                output: None,
                error_code: None,
                error_message: None,
                occurred_at_millis: NOW + 201,
                lease: WorkflowRunLeaseFence::from(&stale_lease),
            },
        )
        .await
        .expect_err("recovered worker must fence the stale lease");
    assert!(matches!(error, WorkflowRepositoryError::LeaseFenceLost));
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("ROLLBACK"));
    assert!(!log.contains("UPDATE public.collaboration_workflow_runs"));
}

fn approval_repository(
    query_results: Vec<Vec<BTreeMap<String, Value>>>,
    affected_rows: &[u64],
) -> WorkflowApprovalRepository {
    let connection =
        MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results(query_results)
            .append_exec_results(affected_rows.iter().copied().map(|rows_affected| {
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected,
                }
            }))
            .into_connection();
    WorkflowApprovalRepository::new(connection).expect("approval repository")
}

fn approval_repository_with_outbox_failure(
    query_results: Vec<Vec<BTreeMap<String, Value>>>,
) -> WorkflowApprovalRepository {
    let connection = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results(query_results)
        .append_exec_results((0..4).map(|_| MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }))
        .append_exec_errors([DbErr::Custom("injected approval outbox crash".to_owned())])
        .into_connection();
    WorkflowApprovalRepository::new(connection).expect("approval repository")
}

fn approval_actor(community_id: CommunityId) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::zed_account(
        principal(31),
        community_id,
        ServiceAccountId::new(31),
        PrincipalScopes::new([
            AuthorizationScope::new("workflows:approve").expect("approval scope")
        ])
        .expect("scopes"),
    )
}

fn approval_row(state: &str, decision_operation_id: Option<Uuid>) -> BTreeMap<String, Value> {
    let terminal = state != "pending";
    BTreeMap::from([
        ("approval_id".into(), Uuid::from_u128(70).into()),
        ("run_id".into(), Uuid::from_u128(20).into()),
        ("workflow_id".into(), Uuid::from_u128(10).into()),
        ("definition_version_text".into(), "1".into()),
        (
            "workflow_creator_principal_id".into(),
            principal(30).as_uuid().into(),
        ),
        ("step_index".into(), 0_i16.into()),
        ("step_operation_id".into(), Uuid::from_u128(50).into()),
        ("capability_sha256".into(), vec![8_u8; 32].into()),
        ("eligibility_kind".into(), "owner".into()),
        ("eligible_principal_id".into(), Option::<Uuid>::None.into()),
        ("request_message".into(), "approve recovery".into()),
        ("state".into(), state.to_owned().into()),
        ("decision_operation_id".into(), decision_operation_id.into()),
        (
            "decided_by_principal_id".into(),
            terminal.then_some(principal(31).as_uuid()).into(),
        ),
        (
            "decision_note".into(),
            terminal.then_some("approved".to_owned()).into(),
        ),
        (
            "expires_at_millis".into(),
            i64::try_from(NOW + 100_000).expect("timestamp").into(),
        ),
        (
            "created_at_millis".into(),
            i64::try_from(NOW).expect("timestamp").into(),
        ),
        (
            "decided_at_millis".into(),
            terminal
                .then_some(i64::try_from(NOW + 10).expect("timestamp"))
                .into(),
        ),
        (
            "updated_at_millis".into(),
            i64::try_from(NOW + if terminal { 10 } else { 0 })
                .expect("timestamp")
                .into(),
        ),
    ])
}

fn approval_membership_row() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("role".into(), "owner".into()),
        ("status".into(), "active".into()),
        ("membership_version_text".into(), "4".into()),
    ])
}

fn waiting_approval_row() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("run_version_text".into(), "3".into()),
        ("run_state".into(), "waiting_approval".into()),
        ("current_step_index".into(), 0_i16.into()),
        ("step_state".into(), "waiting_approval".into()),
        ("step_operation_id".into(), Uuid::from_u128(50).into()),
    ])
}

fn approval_outbox_row(kind: &str, operation_id: Uuid) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("outbox_id".into(), Uuid::from_u128(80).into()),
        ("approval_id".into(), Uuid::from_u128(70).into()),
        ("run_id".into(), Uuid::from_u128(20).into()),
        ("step_index".into(), 0_i16.into()),
        ("operation_id".into(), operation_id.into()),
        ("intent_kind".into(), kind.to_owned().into()),
        ("state".into(), "pending".into()),
        ("attempt_count".into(), 0_i16.into()),
        (
            "available_at_millis".into(),
            i64::try_from(NOW).expect("timestamp").into(),
        ),
        (
            "created_at_millis".into(),
            i64::try_from(NOW).expect("timestamp").into(),
        ),
    ])
}

#[tokio::test]
async fn approval_race_recovers_exactly_one_resume_intent_after_restart() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let approver = approval_actor(community_id);
    let grant_operation_id = Uuid::from_u128(90);
    let interrupted_process = approval_repository_with_outbox_failure(vec![
        vec![approval_row("pending", None)],
        vec![approval_membership_row()],
        vec![waiting_approval_row()],
    ]);
    let interrupted_error = interrupted_process
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: grant_operation_id,
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: Some("approved"),
            decided_at_millis: NOW + 10,
        })
        .await
        .expect_err("outbox failure must roll back the approval transition");
    assert!(matches!(
        interrupted_error,
        WorkflowApprovalError::Unavailable(_)
    ));
    let interrupted_log = format!(
        "{:#?}",
        interrupted_process.into_connection().into_transaction_log()
    );
    assert!(interrupted_log.contains("ROLLBACK"));
    assert!(!interrupted_log.contains("COMMIT"));

    let grant_process = approval_repository(
        vec![
            vec![approval_row("pending", None)],
            vec![approval_membership_row()],
            vec![waiting_approval_row()],
        ],
        &[1, 1, 1, 1, 1],
    );
    let (disposition, outbox) = grant_process
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: grant_operation_id,
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: Some("approved"),
            decided_at_millis: NOW + 10,
        })
        .await
        .expect("winning grant");
    assert_eq!(disposition, WorkflowApprovalDisposition::Applied);
    assert_eq!(outbox.kind, ApprovalOutboxKind::Resume);

    let losing_process = approval_repository(
        vec![vec![approval_row("granted", Some(grant_operation_id))]],
        &[1],
    );
    let losing_error = losing_process
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: Uuid::from_u128(91),
            decision: ApprovalDecision::Deny,
            actor: &approver,
            note: Some("denied"),
            decided_at_millis: NOW + 11,
        })
        .await
        .expect_err("losing deny must remain fenced");
    assert!(matches!(
        losing_error,
        WorkflowApprovalError::DecisionConflict
    ));

    let duplicate_process = approval_repository(
        vec![vec![approval_row("granted", Some(grant_operation_id))]],
        &[1],
    );
    let (duplicate, duplicate_outbox) = duplicate_process
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: grant_operation_id,
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: Some("approved"),
            decided_at_millis: NOW + 10,
        })
        .await
        .expect("exact replay");
    assert_eq!(duplicate, WorkflowApprovalDisposition::Duplicate);
    assert_eq!(duplicate_outbox.kind, ApprovalOutboxKind::Resume);

    let restarted_process = approval_repository(
        vec![vec![approval_outbox_row("resume", grant_operation_id)]],
        &[1],
    );
    let pending = restarted_process
        .pending_outbox(&tenant, NOW + 12, 16)
        .await
        .expect("pending outbox after restart");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].kind, ApprovalOutboxKind::Resume);
    assert_eq!(pending[0].operation_id, grant_operation_id);
}

#[tokio::test]
async fn denial_replay_recovers_exactly_one_cancel_intent() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let approver = approval_actor(community_id);
    let deny_operation_id = Uuid::from_u128(92);
    let deny_process = approval_repository(
        vec![
            vec![approval_row("pending", None)],
            vec![approval_membership_row()],
            vec![waiting_approval_row()],
        ],
        &[1, 1, 1, 1, 1],
    );
    let (disposition, outbox) = deny_process
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: deny_operation_id,
            decision: ApprovalDecision::Deny,
            actor: &approver,
            note: Some("denied"),
            decided_at_millis: NOW + 10,
        })
        .await
        .expect("winning deny");
    assert_eq!(disposition, WorkflowApprovalDisposition::Applied);
    assert_eq!(outbox.kind, ApprovalOutboxKind::Cancel);

    let late_grant_process = approval_repository(
        vec![vec![approval_row("denied", Some(deny_operation_id))]],
        &[1],
    );
    let error = late_grant_process
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: Uuid::from_u128(93),
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: Some("approved"),
            decided_at_millis: NOW + 11,
        })
        .await
        .expect_err("terminal denial must fence a late grant");
    assert!(matches!(error, WorkflowApprovalError::DecisionConflict));

    let restarted_process = approval_repository(
        vec![vec![approval_outbox_row("cancel", deny_operation_id)]],
        &[1],
    );
    let pending = restarted_process
        .pending_outbox(&tenant, NOW + 12, 16)
        .await
        .expect("pending cancellation after restart");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].kind, ApprovalOutboxKind::Cancel);
    assert_eq!(pending[0].operation_id, deny_operation_id);
}
