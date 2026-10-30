use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::{
    git::{
        branch_activity::{
            BranchActivityAppendOutcome, BranchActivityEvent, BranchActivityProjector,
            BranchActivitySink, BranchActivitySinkError,
        },
        branch_channel::{BranchChannelService, branch_channel_id},
        branch_lifecycle::{BranchLifecycleError, BranchLifecycleService},
        object_store::GitContentDigest,
        smart_http_write::GitPushReceipt,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    BranchArchiveReason, BranchCollaboration, BranchCollaborationIdentity, BranchGeneration,
    BranchRefName, BranchUpdateKind, ChannelMembership, CommunityId, CommunityMembership,
    GitCommitId, MembershipRole, MembershipStatus, OperationId, PrincipalId, PrincipalScopes,
    ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use sqlx::PgPool;
use uuid::Uuid;

const CHANNELS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const CHANNELS_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.down.sql"
));

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn repository_id() -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(2))
}

fn principal_id() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(3))
}

fn commit(value: u64) -> GitCommitId {
    GitCommitId::parse(format!("{value:040x}")).expect("commit")
}

fn initial_branch() -> BranchCollaboration {
    BranchCollaboration::create(
        BranchCollaborationIdentity::new(
            community_id(),
            repository_id(),
            BranchRefName::parse("refs/heads/feature/recovery").expect("branch ref"),
            BranchGeneration::FIRST,
        )
        .expect("branch identity"),
        commit(10),
    )
    .expect("branch")
}

fn receipt(operation: u128, parent: char, published: char) -> GitPushReceipt {
    GitPushReceipt {
        operation_id: OperationId::from_uuid(Uuid::from_u128(operation)),
        parent_manifest: Some(
            GitContentDigest::parse(parent.to_string().repeat(64)).expect("parent manifest"),
        ),
        published_manifest: Some(
            GitContentDigest::parse(published.to_string().repeat(64)).expect("published manifest"),
        ),
        applied: true,
    }
}

struct AuthorizationFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    push_scope: AuthorizationScope,
    channel_scope: AuthorizationScope,
    membership: CommunityMembership,
}

impl AuthorizationFixture {
    fn new() -> Self {
        let community_id = community_id();
        let principal_id = principal_id();
        let push_scope = AuthorizationScope::new("git:write").expect("push scope");
        let channel_scope = AuthorizationScope::new("channels:manage").expect("channel scope");
        Self {
            tenant: bind_rpc_tenant(
                Some(
                    TrustedTenantRoute::from_listener(community_id, "branch-recovery-test")
                        .expect("tenant route"),
                ),
                &[],
            )
            .expect("tenant"),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id,
                community_id,
                ServiceAccountId::new(1),
                PrincipalScopes::new([push_scope.clone(), channel_scope.clone()])
                    .expect("principal scopes"),
            ),
            push_scope,
            channel_scope,
            membership: CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Owner,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
        }
    }

    fn push_request(&self) -> AuthorizationRequest<'_> {
        self.request(
            &self.push_scope,
            AuthorizationAction::Write,
            AuthorizationResourceKind::Repository,
            repository_id(),
            None,
        )
    }

    fn community_request(&self) -> AuthorizationRequest<'_> {
        self.request(
            &self.channel_scope,
            AuthorizationAction::Manage,
            AuthorizationResourceKind::Community,
            AggregateId::from_uuid(community_id().as_uuid()),
            None,
        )
    }

    fn channel_request(&self, channel_id: AggregateId) -> AuthorizationRequest<'_> {
        self.request(
            &self.channel_scope,
            AuthorizationAction::Manage,
            AuthorizationResourceKind::Channel,
            channel_id,
            Some(channel_id),
        )
    }

    fn request<'a>(
        &'a self,
        scope: &'a AuthorizationScope,
        action: AuthorizationAction,
        kind: AuthorizationResourceKind,
        resource_id: AggregateId,
        channel_id: Option<AggregateId>,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: scope,
            action,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind,
                resource_id,
                owner_principal_id: None,
                channel_id,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: channel_id.map(|_| AggregateVersion::FIRST),
            channel_membership: channel_id.map(|channel_id| ChannelMembership {
                community_id: community_id(),
                channel_id,
                principal_id: principal_id(),
                role: MembershipRole::Owner,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 1_900_000_000_000,
        }
    }
}

#[derive(Default)]
struct RecoveryActivitySink {
    events: Mutex<BTreeMap<AggregateId, BranchActivityEvent>>,
}

impl RecoveryActivitySink {
    fn snapshot(&self) -> BTreeMap<AggregateId, BranchActivityEvent> {
        self.events.lock().expect("activity sink lock").clone()
    }
}

#[async_trait]
impl BranchActivitySink for RecoveryActivitySink {
    async fn append(
        &self,
        event: &BranchActivityEvent,
    ) -> Result<BranchActivityAppendOutcome, BranchActivitySinkError> {
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

async fn insert_channel_membership(pool: &PgPool, channel_id: AggregateId) {
    sqlx::query(
        "INSERT INTO public.collaboration_channel_memberships (community_id, channel_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, 'owner', 'active', 1, now(), now(), 'zed', $4, now())",
    )
    .bind(community_id().as_uuid())
    .bind(channel_id.as_uuid())
    .bind(principal_id().as_uuid())
    .bind(format!("membership:{channel_id}"))
    .execute(pool)
    .await
    .expect("insert channel membership");
}

async fn channel_state(pool: &PgPool, channel_id: AggregateId) -> (String, i64) {
    sqlx::query_as(
        "SELECT lifecycle_state, channel_version::bigint FROM public.collaboration_channels WHERE community_id = $1 AND channel_id = $2",
    )
    .bind(community_id().as_uuid())
    .bind(channel_id.as_uuid())
    .fetch_one(pool)
    .await
    .expect("load channel state")
}

async fn connect_channel_service(database_url: &str) -> Arc<BranchChannelService> {
    Arc::new(
        BranchChannelService::new(
            sea_orm::Database::connect(database_url)
                .await
                .expect("connect channel service"),
        )
        .expect("channel service"),
    )
}

async fn connect_lifecycle_service(database_url: &str) -> BranchLifecycleService {
    BranchLifecycleService::new(
        sea_orm::Database::connect(database_url)
            .await
            .expect("connect lifecycle service"),
    )
    .expect("lifecycle service")
}

#[tokio::test]
async fn branch_channel_recovery_converges_reordered_duplicate_delayed_and_reconnect_traces() {
    let Some(database_url) = std::env::var("COLLAB_BRANCH_CHANNEL_RECOVERY_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_BRANCH_CHANNEL_RECOVERY_TEST_DATABASE_URL is unset; live recovery test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(CHANNELS_UP)
        .execute(&pool)
        .await
        .expect("apply channel migration");
    sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'branch-recovery.example', 'active', 1, 'zed', 'community:branch-recovery', now(), now(), now())",
    )
    .bind(community_id().as_uuid())
    .execute(&pool)
    .await
    .expect("insert community");
    sqlx::query(
        "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, 'owner', 'active', 1, now(), now(), 'zed', 'principal:branch-recovery', now())",
    )
    .bind(community_id().as_uuid())
    .bind(principal_id().as_uuid())
    .execute(&pool)
    .await
    .expect("insert principal membership");

    let authorization = AuthorizationFixture::new();
    let initial = initial_branch();
    let mut updated = initial.clone();
    updated
        .update_head(
            AggregateVersion::FIRST,
            &commit(10),
            commit(11),
            BranchUpdateKind::FastForward,
        )
        .expect("update branch");
    let mut merged = updated.clone();
    merged
        .merge(
            updated.fields().version,
            &commit(11),
            BranchRefName::parse("refs/heads/main").expect("target branch"),
            commit(20),
        )
        .expect("merge branch");

    let channel_service = connect_channel_service(&database_url).await;
    let lifecycle_service = connect_lifecycle_service(&database_url).await;
    let original_binding = channel_service
        .bind(&initial, principal_id(), &authorization.community_request())
        .await
        .expect("bind initial branch");
    let original_channel_id = original_binding.channel_id();
    insert_channel_membership(&pool, original_channel_id).await;
    lifecycle_service
        .apply_archive_transition(
            &updated,
            &merged,
            AggregateVersion::FIRST,
            &authorization.channel_request(original_channel_id),
        )
        .await
        .expect("archive merged channel before delayed activity");

    let sink = Arc::new(RecoveryActivitySink::default());
    let projector = BranchActivityProjector::new(channel_service.clone(), sink.clone());
    let update_receipt = receipt(101, 'b', 'c');
    let create_receipt = receipt(100, 'a', 'b');
    let delayed_update = projector
        .project_accepted_ref(
            &update_receipt,
            Some(&initial),
            &updated,
            &authorization.push_request(),
            &authorization.community_request(),
        )
        .await
        .expect("project delayed update into archived channel");
    let duplicate_update = projector
        .project_accepted_ref(
            &update_receipt,
            Some(&initial),
            &updated,
            &authorization.push_request(),
            &authorization.community_request(),
        )
        .await
        .expect("deduplicate delayed update");
    assert_eq!(delayed_update, duplicate_update);
    assert_eq!(delayed_update.channel_id(), original_channel_id);
    let delayed_create = projector
        .project_accepted_ref(
            &create_receipt,
            None,
            &initial,
            &authorization.push_request(),
            &authorization.community_request(),
        )
        .await
        .expect("project reordered creation into archived channel");
    assert_eq!(delayed_create.channel_id(), original_channel_id);
    assert_eq!(sink.snapshot().len(), 2);
    assert_eq!(
        channel_state(&pool, original_channel_id).await,
        ("archived".to_owned(), 2)
    );
    assert!(matches!(
        lifecycle_service
            .apply_archive_transition(
                &updated,
                &merged,
                AggregateVersion::FIRST,
                &authorization.channel_request(original_channel_id),
            )
            .await,
        Err(BranchLifecycleError::StaleChannel)
    ));

    drop(projector);
    drop(lifecycle_service);
    drop(channel_service);
    let reconnected_channel_service = connect_channel_service(&database_url).await;
    let reconnected_lifecycle_service = connect_lifecycle_service(&database_url).await;
    let reconnected_projector =
        BranchActivityProjector::new(reconnected_channel_service.clone(), sink.clone());
    let before_replay = sink.snapshot();
    reconnected_projector
        .project_accepted_ref(
            &create_receipt,
            None,
            &initial,
            &authorization.push_request(),
            &authorization.community_request(),
        )
        .await
        .expect("replay creation after reconnect");
    reconnected_projector
        .project_accepted_ref(
            &update_receipt,
            Some(&initial),
            &updated,
            &authorization.push_request(),
            &authorization.community_request(),
        )
        .await
        .expect("replay update after reconnect");
    assert_eq!(sink.snapshot(), before_replay);

    let mut archived = merged.clone();
    archived
        .archive(
            merged.fields().version,
            &commit(11),
            BranchArchiveReason::Merged,
        )
        .expect("archive merged branch state");
    reconnected_lifecycle_service
        .apply_archive_transition(
            &merged,
            &archived,
            AggregateVersion::new(2).expect("channel version two"),
            &authorization.channel_request(original_channel_id),
        )
        .await
        .expect("apply idempotent merged archive");
    let reopened = archived
        .recreate(archived.fields().version, &commit(11), commit(30))
        .expect("reopen branch");
    let reopened_binding = reconnected_lifecycle_service
        .reopen(
            &archived,
            &reopened,
            principal_id(),
            &authorization.community_request(),
        )
        .await
        .expect("bind reopened branch generation");
    let reopened_channel_id = reopened_binding.channel_id();
    assert_ne!(reopened_channel_id, original_channel_id);
    let reopen_receipt = receipt(102, 'c', 'd');
    let reopened_event = reconnected_projector
        .project_accepted_ref(
            &reopen_receipt,
            None,
            &reopened,
            &authorization.push_request(),
            &authorization.community_request(),
        )
        .await
        .expect("project reopened generation");
    assert_eq!(reopened_event.channel_id(), reopened_channel_id);

    drop(reconnected_projector);
    drop(reconnected_lifecycle_service);
    drop(reconnected_channel_service);
    let final_channel_service = connect_channel_service(&database_url).await;
    let final_projector = BranchActivityProjector::new(final_channel_service.clone(), sink.clone());
    let converged_events = sink.snapshot();
    final_projector
        .project_accepted_ref(
            &reopen_receipt,
            None,
            &reopened,
            &authorization.push_request(),
            &authorization.community_request(),
        )
        .await
        .expect("replay reopened generation after disconnect");
    assert_eq!(sink.snapshot(), converged_events);
    assert_eq!(converged_events.len(), 3);
    assert_eq!(
        channel_state(&pool, original_channel_id).await,
        ("archived".to_owned(), 2)
    );
    assert_eq!(
        channel_state(&pool, reopened_channel_id).await,
        ("active".to_owned(), 1)
    );
    assert_eq!(
        branch_channel_id(&reopened.fields().identity).expect("reopened channel id"),
        reopened_channel_id
    );
    let channel_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_channels WHERE community_id = $1",
    )
    .bind(community_id().as_uuid())
    .fetch_one(&pool)
    .await
    .expect("count branch channels");
    assert_eq!(channel_count, 2);

    drop(final_projector);
    drop(final_channel_service);
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll back channel migration");
}
