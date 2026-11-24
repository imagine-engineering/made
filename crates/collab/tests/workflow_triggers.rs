use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use collab::workflows::{
    repository::{
        StoredWorkflowDefinition, WorkflowIdentity, WorkflowLifecycle, WorkflowProvenance,
        WorkflowRepositoryError, WorkflowRunRequest, WorkflowScope, WorkflowStoreOutcome,
    },
    triggers::{
        CollaborationEventTrigger, CollaborationEventTriggerKind, EVENT_TRIGGER_SCOPE,
        MAX_SCHEDULE_CATCH_UP_RUNS, ScheduleClock, WORKFLOW_RUN_SCOPE, WorkflowRunClaimer,
        WorkflowTriggerAdmission, WorkflowTriggerAdmissionError, WorkflowTriggerAdmissionStatus,
        evaluate_schedule,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus,
    PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use collaboration_workflow::definition::WorkflowDefinition;
use uuid::Uuid;

const NOW: u64 = 1_900_000_320_000;

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
            TrustedTenantRoute::from_listener(community_id, "workflow-trigger-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn workflow_definition(trigger: &str) -> WorkflowDefinition {
    WorkflowDefinition::parse_yaml(&format!(
        r#"
version: 1
name: Trigger test
enabled: true
trigger:
{trigger}
steps:
  - id: announce
    action: send_message
    text: admitted
"#
    ))
    .expect("workflow definition")
}

fn stored_definition(community_id: CommunityId, trigger: &str) -> StoredWorkflowDefinition {
    let identity =
        WorkflowIdentity::new(community_id, Uuid::from_u128(10)).expect("workflow identity");
    StoredWorkflowDefinition {
        identity,
        definition_version: 1,
        definition: workflow_definition(trigger),
        definition_sha256: [1; 32],
        creator_principal_id: principal(20),
        author_principal_id: principal(20),
        scope: WorkflowScope::Community,
        current_definition_version: 1,
        head_revision: 1,
        lifecycle: WorkflowLifecycle::Active,
        provenance: WorkflowProvenance::new("zed", "workflow:10:1", "1", NOW, None)
            .expect("provenance"),
        created_at_millis: NOW,
    }
}

#[derive(Clone, Default)]
struct FakeRunClaimer {
    requests: Arc<Mutex<Vec<WorkflowRunRequest>>>,
}

impl FakeRunClaimer {
    fn requests(&self) -> Vec<WorkflowRunRequest> {
        self.requests.lock().expect("request lock").clone()
    }
}

#[async_trait]
impl WorkflowRunClaimer for FakeRunClaimer {
    async fn claim_run(
        &self,
        _tenant: &TenantContext,
        request: &WorkflowRunRequest,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        let mut requests = self.requests.lock().expect("request lock");
        if let Some(existing) = requests
            .iter()
            .find(|existing| existing.trigger_operation_id == request.trigger_operation_id)
        {
            return if existing == request {
                Ok(WorkflowStoreOutcome::Duplicate)
            } else {
                Err(WorkflowRepositoryError::IdempotencyConflict)
            };
        }
        requests.push(request.clone());
        Ok(WorkflowStoreOutcome::Applied)
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

    fn event(&self, text: &str) -> CollaborationEventTrigger {
        CollaborationEventTrigger::new(
            self.tenant.community_id(),
            [5; 32],
            CollaborationEventTriggerKind::MessagePosted,
            self.channel_id,
            principal(30),
            [6; 32],
            NOW - 1_000,
            NOW,
            Some(text.to_owned()),
            None,
            None,
        )
        .expect("event")
    }
}

#[tokio::test]
async fn duplicate_event_claims_one_stable_run() {
    let community_id = community(1);
    let authorization = AuthorizationFixture::new(community_id);
    let definition = stored_definition(
        community_id,
        r#"  on: message_posted
  if: 'str_contains(trigger_text, "ship")'"#,
    );
    let claimer = FakeRunClaimer::default();
    let admission = WorkflowTriggerAdmission::new(claimer.clone());
    let event = authorization.event("ready to ship");

    let first = admission
        .admit_event(
            &authorization.tenant,
            &definition,
            &event,
            &authorization.source_request(),
            &authorization.owner_request(),
        )
        .await
        .expect("first admission");
    let duplicate = admission
        .admit_event(
            &authorization.tenant,
            &definition,
            &event,
            &authorization.source_request(),
            &authorization.owner_request(),
        )
        .await
        .expect("duplicate admission");

    assert_eq!(first.status, WorkflowTriggerAdmissionStatus::Claimed);
    assert_eq!(duplicate.status, WorkflowTriggerAdmissionStatus::Duplicate);
    assert_eq!(first.run_identity, duplicate.run_identity);
    assert_eq!(claimer.requests().len(), 1);
}

#[tokio::test]
async fn false_condition_is_a_truthful_filter_without_a_run() {
    let community_id = community(1);
    let authorization = AuthorizationFixture::new(community_id);
    let definition = stored_definition(
        community_id,
        r#"  on: message_posted
  if: 'str_contains(trigger_text, "ship")'"#,
    );
    let claimer = FakeRunClaimer::default();
    let admission = WorkflowTriggerAdmission::new(claimer.clone());

    let outcome = admission
        .admit_event(
            &authorization.tenant,
            &definition,
            &authorization.event("still drafting"),
            &authorization.source_request(),
            &authorization.owner_request(),
        )
        .await
        .expect("filtered admission");

    assert_eq!(outcome.status, WorkflowTriggerAdmissionStatus::Filtered);
    assert_eq!(outcome.run_identity, None);
    assert!(claimer.requests().is_empty());
}

#[tokio::test]
async fn unauthorized_event_source_is_rejected_before_condition_or_storage() {
    let community_id = community(1);
    let mut authorization = AuthorizationFixture::new(community_id);
    authorization.source_channel_membership.status = MembershipStatus::Revoked;
    let definition = stored_definition(community_id, "  on: message_posted");
    let claimer = FakeRunClaimer::default();
    let admission = WorkflowTriggerAdmission::new(claimer.clone());

    let error = admission
        .admit_event(
            &authorization.tenant,
            &definition,
            &authorization.event("attempt"),
            &authorization.source_request(),
            &authorization.owner_request(),
        )
        .await
        .expect_err("revoked source must fail");

    assert!(matches!(
        error,
        WorkflowTriggerAdmissionError::UnauthorizedSource
    ));
    assert!(claimer.requests().is_empty());
}

#[test]
fn missed_interval_is_bounded_and_worker_clock_skew_cannot_change_fires() {
    let definition = workflow_definition(
        r#"  on: schedule
  interval: 60s"#,
    );
    let previous = NOW - 5 * 60_000;
    let fast_worker = ScheduleClock::new(NOW, NOW + 60 * 60_000).expect("fast clock");
    let slow_worker = ScheduleClock::new(NOW, NOW - 60 * 60_000).expect("slow clock");

    let fast = evaluate_schedule(&definition, Some(previous), fast_worker).expect("fast plan");
    let slow = evaluate_schedule(&definition, Some(previous), slow_worker).expect("slow plan");

    assert!(fast_worker.worker_clock_skewed());
    assert!(slow_worker.worker_clock_skewed());
    assert_eq!(fast.fires(), slow.fires());
    assert_eq!(fast.fires().len(), 5);
    assert_eq!(
        fast.fires().last().expect("last").scheduled_for_millis(),
        NOW
    );

    let bounded = evaluate_schedule(
        &definition,
        Some(NOW - 2 * 24 * 60 * 60_000),
        ScheduleClock::new(NOW, NOW).expect("clock"),
    )
    .expect("bounded plan");
    assert!(bounded.backlog_before_window());
    assert_eq!(bounded.fires().len(), MAX_SCHEDULE_CATCH_UP_RUNS);
    assert!(bounded.skipped_fires() > 0);
}

#[tokio::test]
async fn missed_cron_fire_has_one_replica_stable_claim() {
    let community_id = community(1);
    let authorization = AuthorizationFixture::new(community_id);
    let definition = stored_definition(
        community_id,
        r#"  on: schedule
  cron: '* * * * *'"#,
    );
    let evaluation = evaluate_schedule(
        &definition.definition,
        Some(NOW - 61_000),
        ScheduleClock::new(NOW, NOW + 10_000).expect("clock"),
    )
    .expect("cron plan");
    let fire = *evaluation.fires().last().expect("missed cron fire");
    let claimer = FakeRunClaimer::default();
    let admission = WorkflowTriggerAdmission::new(claimer.clone());

    let first = admission
        .admit_schedule_fire(
            &authorization.tenant,
            &definition,
            fire,
            &authorization.owner_request(),
        )
        .await
        .expect("first replica");
    let second = admission
        .admit_schedule_fire(
            &authorization.tenant,
            &definition,
            fire,
            &authorization.owner_request(),
        )
        .await
        .expect("second replica");

    assert_eq!(first.status, WorkflowTriggerAdmissionStatus::Claimed);
    assert_eq!(second.status, WorkflowTriggerAdmissionStatus::Duplicate);
    assert_eq!(first.run_identity, second.run_identity);
    assert_eq!(claimer.requests().len(), 1);
}
