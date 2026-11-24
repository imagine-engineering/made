use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use collab::{
    git::{
        branch_activity::{
            BranchActivityAppendOutcome, BranchActivityEvent, BranchActivityKind,
            BranchActivityProjectionError, BranchActivityProjector, BranchActivitySink,
            BranchActivitySinkError, BranchChannelResolutionError, BranchChannelResolver,
        },
        object_store::GitContentDigest,
        smart_http_write::GitPushReceipt,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    BranchCollaboration, BranchCollaborationIdentity, BranchGeneration, BranchRefName,
    BranchUpdateKind, CommunityId, CommunityMembership, GitCommitId, MembershipRole,
    MembershipStatus, OperationId, PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext,
    TrustedTenantRoute,
};
use uuid::Uuid;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn commit(value: u64) -> GitCommitId {
    GitCommitId::parse(format!("{value:040x}")).expect("commit")
}

fn branch() -> BranchCollaboration {
    BranchCollaboration::create(
        BranchCollaborationIdentity::new(
            community(1),
            aggregate(2),
            BranchRefName::parse("refs/heads/feature/activity").expect("branch ref"),
            BranchGeneration::FIRST,
        )
        .expect("branch identity"),
        commit(10),
    )
    .expect("branch")
}

struct AuthorizationFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    scope: AuthorizationScope,
    membership: CommunityMembership,
    action: AuthorizationAction,
    resource_kind: AuthorizationResourceKind,
    resource_id: AggregateId,
}

impl AuthorizationFixture {
    fn push() -> Self {
        Self::new(
            principal(3),
            MembershipRole::Member,
            "git:write",
            AuthorizationAction::Write,
            AuthorizationResourceKind::Repository,
            aggregate(2),
        )
    }

    fn channel() -> Self {
        Self::new(
            principal(4),
            MembershipRole::Owner,
            "channels:manage",
            AuthorizationAction::Manage,
            AuthorizationResourceKind::Community,
            AggregateId::from_uuid(community(1).as_uuid()),
        )
    }

    fn new(
        principal_id: PrincipalId,
        role: MembershipRole,
        scope_value: &str,
        action: AuthorizationAction,
        resource_kind: AuthorizationResourceKind,
        resource_id: AggregateId,
    ) -> Self {
        let community_id = community(1);
        let scope = AuthorizationScope::new(scope_value).expect("scope");
        Self {
            tenant: bind_rpc_tenant(
                Some(
                    TrustedTenantRoute::from_listener(community_id, "branch-activity-test")
                        .expect("tenant route"),
                ),
                &[],
            )
            .expect("tenant"),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id,
                community_id,
                ServiceAccountId::new(1),
                PrincipalScopes::new([scope.clone()]).expect("scopes"),
            ),
            scope,
            membership: CommunityMembership {
                community_id,
                principal_id,
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            action,
            resource_kind,
            resource_id,
        }
    }

    fn request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: self.action,
            resource: AuthorizationResource {
                community_id: community(1),
                kind: self.resource_kind,
                resource_id: self.resource_id,
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 1_900_000_000_000,
        }
    }
}

struct RecoveringChannelResolver {
    channel_id: AggregateId,
    available: AtomicBool,
    calls: AtomicUsize,
}

#[async_trait]
impl BranchChannelResolver for RecoveringChannelResolver {
    async fn resolve_or_create(
        &self,
        _branch: &BranchCollaboration,
        _creator_principal_id: PrincipalId,
        _authorization: &AuthorizationRequest<'_>,
    ) -> Result<AggregateId, BranchChannelResolutionError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.available.store(true, Ordering::SeqCst);
        Ok(self.channel_id)
    }
}

#[derive(Default)]
struct MemoryActivitySink {
    events: Mutex<BTreeMap<AggregateId, BranchActivityEvent>>,
    calls: AtomicUsize,
}

#[async_trait]
impl BranchActivitySink for MemoryActivitySink {
    async fn append(
        &self,
        event: &BranchActivityEvent,
    ) -> Result<BranchActivityAppendOutcome, BranchActivitySinkError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut events = self.events.lock().expect("activity sink lock");
        if let Some(existing) = events.get(&event.event_id()) {
            return if existing == event {
                Ok(BranchActivityAppendOutcome::Existing)
            } else {
                Err(BranchActivitySinkError::Conflict)
            };
        }
        events.insert(event.event_id(), event.clone());
        Ok(BranchActivityAppendOutcome::Inserted)
    }
}

fn receipt(applied: bool) -> GitPushReceipt {
    GitPushReceipt {
        operation_id: OperationId::from_uuid(Uuid::from_u128(20)),
        parent_manifest: Some(GitContentDigest::parse("a".repeat(64)).expect("parent manifest")),
        published_manifest: applied
            .then(|| GitContentDigest::parse("b".repeat(64)).expect("published manifest")),
        applied,
    }
}

#[tokio::test]
async fn branch_activity_retry_deduplicates_and_recovers_a_missing_channel() {
    let previous = branch();
    let mut current = previous.clone();
    current
        .update_head(
            AggregateVersion::FIRST,
            &commit(10),
            commit(11),
            BranchUpdateKind::Force,
        )
        .expect("accepted force update");
    let channel_id = aggregate(30);
    let resolver = Arc::new(RecoveringChannelResolver {
        channel_id,
        available: AtomicBool::new(false),
        calls: AtomicUsize::new(0),
    });
    let sink = Arc::new(MemoryActivitySink::default());
    let projector = BranchActivityProjector::new(resolver.clone(), sink.clone());
    let push = AuthorizationFixture::push();
    let channel = AuthorizationFixture::channel();

    let first = projector
        .project_accepted_ref(
            &receipt(true),
            Some(&previous),
            &current,
            &push.request(),
            &channel.request(),
        )
        .await
        .expect("project activity");
    let retried = projector
        .project_accepted_ref(
            &receipt(true),
            Some(&previous),
            &current,
            &push.request(),
            &channel.request(),
        )
        .await
        .expect("retry activity");

    assert!(resolver.available.load(Ordering::SeqCst));
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 2);
    assert_eq!(sink.calls.load(Ordering::SeqCst), 2);
    assert_eq!(sink.events.lock().expect("events").len(), 1);
    assert_eq!(first, retried);
    assert_eq!(first.channel_id(), channel_id);
    assert_eq!(first.actor_principal_id(), principal(3));
    assert_eq!(first.branch(), &current.fields().identity);
    assert_eq!(
        first.parent_manifest().map(GitContentDigest::as_str),
        Some("a".repeat(64).as_str())
    );
    assert_eq!(first.published_manifest().as_str(), "b".repeat(64));
    assert_eq!(
        first.kind(),
        &BranchActivityKind::Updated {
            previous_commit: commit(10),
            current_commit: commit(11),
            update_kind: BranchUpdateKind::Force,
        }
    );
}

#[tokio::test]
async fn branch_activity_creation_links_the_initial_commit() {
    let current = branch();
    let resolver = Arc::new(RecoveringChannelResolver {
        channel_id: aggregate(31),
        available: AtomicBool::new(false),
        calls: AtomicUsize::new(0),
    });
    let sink = Arc::new(MemoryActivitySink::default());
    let projector = BranchActivityProjector::new(resolver, sink);
    let push = AuthorizationFixture::push();
    let channel = AuthorizationFixture::channel();

    let event = projector
        .project_accepted_ref(
            &receipt(true),
            None,
            &current,
            &push.request(),
            &channel.request(),
        )
        .await
        .expect("create activity");

    assert_eq!(
        event.kind(),
        &BranchActivityKind::Created { commit: commit(10) }
    );
    assert_eq!(event.kind().current_commit(), &commit(10));
    assert_eq!(event.branch_version(), AggregateVersion::FIRST);
}

#[tokio::test]
async fn branch_activity_rejects_unapplied_or_inconsistent_updates_before_dependencies() {
    let current = branch();
    let resolver = Arc::new(RecoveringChannelResolver {
        channel_id: aggregate(32),
        available: AtomicBool::new(false),
        calls: AtomicUsize::new(0),
    });
    let sink = Arc::new(MemoryActivitySink::default());
    let projector = BranchActivityProjector::new(resolver.clone(), sink.clone());
    let push = AuthorizationFixture::push();
    let channel = AuthorizationFixture::channel();

    assert_eq!(
        projector
            .project_accepted_ref(
                &receipt(false),
                None,
                &current,
                &push.request(),
                &channel.request(),
            )
            .await,
        Err(BranchActivityProjectionError::PushNotApplied)
    );
    assert_eq!(
        projector
            .project_accepted_ref(
                &receipt(true),
                Some(&current),
                &current,
                &push.request(),
                &channel.request(),
            )
            .await,
        Err(BranchActivityProjectionError::InvalidTransition)
    );
    assert_eq!(resolver.calls.load(Ordering::SeqCst), 0);
    assert_eq!(sink.calls.load(Ordering::SeqCst), 0);
}
