use std::{
    collections::{BTreeMap, BTreeSet},
    net::IpAddr,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::workflows::{
    actions::{
        CanonicalWorkflowCommand, CanonicalWorkflowCommandSink, ResolvedActionSecret,
        WorkflowActionAttempt, WorkflowActionAuthority, WorkflowActionAuthorization,
        WorkflowActionError, WorkflowActionExecutor, WorkflowActionKind, WorkflowActionOutcome,
        WorkflowActionSecretResolver, WorkflowActionTarget, WorkflowCommandDisposition,
        WorkflowCommandReceipt,
    },
    repository::{
        StoredWorkflowDefinition, StoredWorkflowRun, StoredWorkflowStep, WorkflowIdentity,
        WorkflowLifecycle, WorkflowProvenance, WorkflowRunIdentity, WorkflowRunLease,
        WorkflowRunLeaseFence, WorkflowRunLeaseState, WorkflowRunState, WorkflowScope,
        WorkflowStepState, WorkflowTriggerKind,
    },
    webhook::{WebhookDnsResolver, WebhookNetworkPolicy, WebhookTransportPolicyError},
};
use collaboration_domain::{
    AuthenticatedPrincipal, AuthorizationScope, CommunityId, OperationId, PrincipalId,
    PrincipalScopes, TenantContext, TrustedTenantRoute,
};
use collaboration_workflow::definition::WorkflowDefinition;
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const NOW: u64 = 1_781_000_000_000;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "workflow-action-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn actor(community_id: CommunityId, principal_id: PrincipalId) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::service(
        principal_id,
        community_id,
        "workflow-owner",
        PrincipalScopes::new([
            AuthorizationScope::new("workflows:run").expect("workflow scope"),
            AuthorizationScope::new("messages:write").expect("message scope"),
            AuthorizationScope::new("channels:write").expect("channel scope"),
            AuthorizationScope::new("jobs:write").expect("job scope"),
        ])
        .expect("scopes"),
    )
    .expect("actor")
}

fn provenance(record: &str) -> WorkflowProvenance {
    WorkflowProvenance::new("zed", record, "1", NOW, None).expect("provenance")
}

fn definition(action: &str) -> WorkflowDefinition {
    WorkflowDefinition::parse_yaml(&format!(
        r#"
version: 1
name: Action test
enabled: true
trigger:
  on: message_posted
steps:
  - id: action
    timeout_secs: 10
{action}
"#
    ))
    .expect("definition")
}

struct Fixture {
    tenant: TenantContext,
    definition: StoredWorkflowDefinition,
    run: StoredWorkflowRun,
    lease_fence: WorkflowRunLeaseFence,
    outputs: BTreeMap<String, JsonValue>,
}

impl Fixture {
    fn new(action: &str) -> Self {
        let community_id = community(1);
        let workflow_id =
            WorkflowIdentity::new(community_id, Uuid::from_u128(10)).expect("workflow identity");
        let definition = definition(action);
        let canonical = serde_json::to_vec(&definition).expect("canonical definition");
        let creator_principal_id = principal(20);
        let stored_definition = StoredWorkflowDefinition {
            identity: workflow_id,
            definition_version: 1,
            definition,
            definition_sha256: Sha256::digest(&canonical).into(),
            creator_principal_id,
            author_principal_id: creator_principal_id,
            scope: WorkflowScope::Community,
            current_definition_version: 1,
            head_revision: 1,
            lifecycle: WorkflowLifecycle::Active,
            provenance: provenance("workflow:action:v1"),
            created_at_millis: NOW,
        };
        let run_identity =
            WorkflowRunIdentity::new(community_id, Uuid::from_u128(30)).expect("run identity");
        let step = StoredWorkflowStep {
            index: 0,
            step_id: "action".to_owned(),
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
            identity: run_identity,
            workflow: workflow_id,
            definition_version: 1,
            trigger_operation_id: Uuid::from_u128(50),
            trigger_kind: WorkflowTriggerKind::Event,
            trigger_source_id: "event:1".to_owned(),
            trigger_context: json!({
                "author": "1111111111111111111111111111111111111111111111111111111111111111",
                "channel_id": "00000000-0000-0000-0000-000000000070",
                "message_id": "2222222222222222222222222222222222222222222222222222222222222222",
                "text": "hello"
            }),
            run_version: 2,
            state: WorkflowRunState::Running,
            current_step_index: 0,
            error_code: None,
            error_message: None,
            provenance: provenance("run:action:1"),
            created_at_millis: NOW,
            started_at_millis: Some(NOW),
            completed_at_millis: None,
            updated_at_millis: NOW,
            steps: vec![step],
            retries: Vec::new(),
        };
        let lease = WorkflowRunLease {
            identity: run_identity,
            admitted_run_version: 2,
            generation: 1,
            lease_id: Uuid::from_u128(60),
            worker_id: "worker-1".to_owned(),
            state: WorkflowRunLeaseState::Active,
            acquired_at_millis: NOW,
            last_heartbeat_at_millis: NOW,
            expires_at_millis: NOW + 10_000,
            recovery_after_millis: NOW + 20_000,
            released_at_millis: None,
            release_reason: None,
        };
        let lease_fence = WorkflowRunLeaseFence::from(&lease);
        Self {
            tenant: tenant(community_id),
            definition: stored_definition,
            run,
            lease_fence,
            outputs: BTreeMap::new(),
        }
    }

    fn attempt(&self) -> WorkflowActionAttempt<'_> {
        WorkflowActionAttempt {
            tenant: &self.tenant,
            definition: &self.definition,
            run: &self.run,
            step: &self.run.steps[0],
            lease: &self.lease_fence,
            default_channel: Some("00000000-0000-0000-0000-000000000070"),
            previous_step_outputs: &self.outputs,
        }
    }
}

#[derive(Clone)]
struct RecordingAuthority {
    actor: AuthenticatedPrincipal,
    denied: bool,
    requests: Arc<Mutex<Vec<(WorkflowActionKind, WorkflowActionTarget)>>>,
}

#[async_trait]
impl WorkflowActionAuthority for RecordingAuthority {
    async fn authorize(
        &self,
        request: &WorkflowActionAuthorization<'_>,
    ) -> Result<AuthenticatedPrincipal, WorkflowActionError> {
        self.requests
            .lock()
            .expect("authority requests")
            .push((request.action_kind, request.target.clone()));
        if self.denied {
            Err(WorkflowActionError::PermissionDenied)
        } else {
            Ok(self.actor.clone())
        }
    }
}

#[derive(Clone, Default)]
struct RecordingCommands {
    commands: Arc<Mutex<Vec<CanonicalWorkflowCommand>>>,
    operations: Arc<Mutex<BTreeSet<OperationId>>>,
}

#[async_trait]
impl CanonicalWorkflowCommandSink for RecordingCommands {
    async fn submit(
        &self,
        tenant: &TenantContext,
        actor: &AuthenticatedPrincipal,
        operation_id: OperationId,
        command: CanonicalWorkflowCommand,
    ) -> Result<WorkflowCommandReceipt, WorkflowActionError> {
        assert_eq!(tenant.community_id(), actor.community_id());
        self.commands.lock().expect("commands").push(command);
        let disposition = if self
            .operations
            .lock()
            .expect("operations")
            .insert(operation_id)
        {
            WorkflowCommandDisposition::Applied
        } else {
            WorkflowCommandDisposition::Duplicate
        };
        Ok(WorkflowCommandReceipt {
            operation_id,
            disposition,
            output: json!({ "canonical": true }),
        })
    }
}

#[derive(Clone, Copy)]
struct NoSecrets;

#[async_trait]
impl WorkflowActionSecretResolver for NoSecrets {
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

#[derive(Clone, Default)]
struct RecordingDns {
    calls: Arc<Mutex<usize>>,
}

#[async_trait]
impl WebhookDnsResolver for RecordingDns {
    async fn resolve(
        &self,
        _host: &str,
        _port: u16,
    ) -> Result<Vec<IpAddr>, WebhookTransportPolicyError> {
        *self.calls.lock().expect("DNS calls") += 1;
        Ok(vec!["203.0.113.1".parse().expect("address")])
    }
}

fn executor(
    fixture: &Fixture,
    denied: bool,
) -> (
    WorkflowActionExecutor<RecordingAuthority, RecordingCommands, NoSecrets, RecordingDns>,
    RecordingCommands,
    Arc<Mutex<Vec<(WorkflowActionKind, WorkflowActionTarget)>>>,
    RecordingDns,
) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let commands = RecordingCommands::default();
    let dns = RecordingDns::default();
    (
        WorkflowActionExecutor::new(
            RecordingAuthority {
                actor: actor(
                    fixture.tenant.community_id(),
                    fixture.definition.creator_principal_id,
                ),
                denied,
                requests: requests.clone(),
            },
            commands.clone(),
            NoSecrets,
            WebhookNetworkPolicy::new(dns.clone()),
        ),
        commands,
        requests,
        dns,
    )
}

#[tokio::test]
async fn send_dm_uses_current_permission_and_canonical_command() {
    let fixture = Fixture::new(
        r#"    action: send_dm
    to: '{{ trigger.author }}'
    text: 'reply: {{ trigger.text }}'"#,
    );
    let (executor, commands, requests, _) = executor(&fixture, false);

    let outcome = executor.execute(fixture.attempt()).await.expect("send DM");

    assert!(matches!(
        outcome,
        WorkflowActionOutcome::Completed {
            disposition: WorkflowCommandDisposition::Applied,
            ..
        }
    ));
    assert_eq!(
        commands.commands.lock().expect("commands").as_slice(),
        [CanonicalWorkflowCommand::SendDm {
            recipient: "1111111111111111111111111111111111111111111111111111111111111111"
                .to_owned(),
            text: "reply: hello".to_owned(),
        }]
    );
    assert_eq!(
        requests.lock().expect("requests").as_slice(),
        [(
            WorkflowActionKind::SendDm,
            WorkflowActionTarget::Principal(
                "1111111111111111111111111111111111111111111111111111111111111111".to_owned()
            )
        )]
    );
}

#[tokio::test]
async fn set_channel_topic_is_canonical_and_replay_safe() {
    let fixture = Fixture::new(
        r#"    action: set_channel_topic
    topic: 'status: {{ trigger.text }}'"#,
    );
    let (executor, commands, requests, _) = executor(&fixture, false);

    let first = executor
        .execute(fixture.attempt())
        .await
        .expect("set topic");
    let replay = executor
        .execute(fixture.attempt())
        .await
        .expect("replay topic");

    assert!(matches!(
        first,
        WorkflowActionOutcome::Completed {
            disposition: WorkflowCommandDisposition::Applied,
            ..
        }
    ));
    assert!(matches!(
        replay,
        WorkflowActionOutcome::Completed {
            disposition: WorkflowCommandDisposition::Duplicate,
            ..
        }
    ));
    assert_eq!(
        commands.commands.lock().expect("commands")[0],
        CanonicalWorkflowCommand::SetChannelTopic {
            channel: "00000000-0000-0000-0000-000000000070".to_owned(),
            topic: "status: hello".to_owned(),
        }
    );
    assert_eq!(requests.lock().expect("requests").len(), 2);
}

#[tokio::test]
async fn permission_denial_has_no_command_side_effect() {
    let fixture = Fixture::new(
        r#"    action: send_message
    text: denied"#,
    );
    let (executor, commands, _, _) = executor(&fixture, true);

    assert_eq!(
        executor.execute(fixture.attempt()).await,
        Err(WorkflowActionError::PermissionDenied)
    );
    assert!(commands.commands.lock().expect("commands").is_empty());
}

#[tokio::test]
async fn webhook_permission_denial_occurs_before_dns_or_secret_resolution() {
    let fixture = Fixture::new(
        r#"    action: call_webhook
    url: https://example.com/hook
    method: POST
    body: hello"#,
    );
    let (executor, commands, requests, dns) = executor(&fixture, true);

    assert_eq!(
        executor.execute(fixture.attempt()).await,
        Err(WorkflowActionError::PermissionDenied)
    );
    assert!(commands.commands.lock().expect("commands").is_empty());
    assert_eq!(*dns.calls.lock().expect("DNS calls"), 0);
    assert_eq!(
        requests.lock().expect("requests")[0].0,
        WorkflowActionKind::CallWebhook
    );
}

#[tokio::test]
async fn reaction_uses_trigger_target_without_loopback() {
    let fixture = Fixture::new(
        r#"    action: add_reaction
    emoji: eyes"#,
    );
    let (executor, commands, requests, _) = executor(&fixture, false);

    executor
        .execute(fixture.attempt())
        .await
        .expect("add reaction");

    assert_eq!(
        commands.commands.lock().expect("commands").as_slice(),
        [CanonicalWorkflowCommand::AddReaction {
            channel: "00000000-0000-0000-0000-000000000070".to_owned(),
            message: "2222222222222222222222222222222222222222222222222222222222222222".to_owned(),
            emoji: "eyes".to_owned(),
        }]
    );
    assert_eq!(
        requests.lock().expect("requests")[0],
        (
            WorkflowActionKind::AddReaction,
            WorkflowActionTarget::Message {
                channel: "00000000-0000-0000-0000-000000000070".to_owned(),
                message: "2222222222222222222222222222222222222222222222222222222222222222"
                    .to_owned(),
            }
        )
    );
}

#[tokio::test]
async fn stale_attempt_fails_before_authority() {
    let mut fixture = Fixture::new(
        r#"    action: send_dm
    to: '{{ trigger.missing }}'
    text: hello"#,
    );
    fixture.run.state = WorkflowRunState::Cancelled;
    let (executor, commands, requests, _) = executor(&fixture, false);

    assert_eq!(
        executor.execute(fixture.attempt()).await,
        Err(WorkflowActionError::StaleAttempt)
    );
    assert!(requests.lock().expect("requests").is_empty());
    assert!(commands.commands.lock().expect("commands").is_empty());
}

#[tokio::test]
async fn unresolved_template_fails_before_authority() {
    let fixture = Fixture::new(
        r#"    action: send_dm
    to: '{{ trigger.missing }}'
    text: hello"#,
    );
    let (executor, commands, requests, _) = executor(&fixture, false);

    assert_eq!(
        executor.execute(fixture.attempt()).await,
        Err(WorkflowActionError::InvalidRenderedInput)
    );
    assert!(requests.lock().expect("requests").is_empty());
    assert!(commands.commands.lock().expect("commands").is_empty());
}

#[test]
fn failure_taxonomy_never_retries_permission_or_ambiguous_delivery() {
    assert_eq!(
        WorkflowActionError::PermissionDenied.retry_failure_class(),
        None
    );
    assert_eq!(
        WorkflowActionError::RateLimited.retry_failure_class(),
        Some(collaboration_workflow::definition::RetryFailureClass::RateLimited)
    );
    assert!(WorkflowActionError::AmbiguousDelivery.requires_repair());
    assert_eq!(
        WorkflowActionError::AmbiguousDelivery.retry_failure_class(),
        None
    );
}
