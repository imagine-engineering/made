use collab::{
    git::{
        branch_channel::{BranchChannelService, branch_channel_id},
        branch_lifecycle::{
            BranchChannelLifecycleCause, BranchLifecycleError, BranchLifecycleService,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    BranchArchiveReason, BranchCollaboration, BranchCollaborationIdentity, BranchGeneration,
    BranchRefName, ChannelCommandOutcome, ChannelLifecycleState, ChannelMembership, CommunityId,
    CommunityMembership, GitCommitId, MembershipRole, MembershipStatus, PrincipalId,
    PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
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

fn creator_principal_id() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(3))
}

fn commit(value: u64) -> GitCommitId {
    GitCommitId::parse(format!("{value:040x}")).expect("commit")
}

fn branch(branch_ref: &str) -> BranchCollaboration {
    BranchCollaboration::create(
        BranchCollaborationIdentity::new(
            community_id(),
            repository_id(),
            BranchRefName::parse(branch_ref).expect("branch ref"),
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
}

impl AuthorizationFixture {
    fn new() -> Self {
        let community_id = community_id();
        let principal_id = creator_principal_id();
        let scope = AuthorizationScope::new("channels:manage").expect("scope");
        Self {
            tenant: bind_rpc_tenant(
                Some(
                    TrustedTenantRoute::from_listener(community_id, "branch-lifecycle-test")
                        .expect("tenant route"),
                ),
                &[],
            )
            .expect("tenant"),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id,
                community_id,
                ServiceAccountId::new(1),
                PrincipalScopes::new([scope.clone()]).expect("principal scopes"),
            ),
            scope,
            membership: CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Owner,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
        }
    }

    fn community_request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: AuthorizationAction::Manage,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(community_id().as_uuid()),
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

    fn channel_request(&self, channel_id: AggregateId) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: AuthorizationAction::Manage,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Channel,
                resource_id: channel_id,
                owner_principal_id: None,
                channel_id: Some(channel_id),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id: community_id(),
                channel_id,
                principal_id: creator_principal_id(),
                role: MembershipRole::Owner,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 1_900_000_000_100,
        }
    }
}

async fn insert_channel_membership(pool: &PgPool, channel_id: AggregateId) {
    sqlx::query(
        "INSERT INTO public.collaboration_channel_memberships (community_id, channel_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, 'owner', 'active', 1, now(), now(), 'zed', $4, now())",
    )
    .bind(community_id().as_uuid())
    .bind(channel_id.as_uuid())
    .bind(creator_principal_id().as_uuid())
    .bind(format!("membership:{channel_id}"))
    .execute(pool)
    .await
    .expect("insert channel membership");
}

async fn channel_snapshot(
    pool: &PgPool,
    channel_id: AggregateId,
) -> (String, i64, String, String, Uuid) {
    sqlx::query_as(
        "SELECT lifecycle_state, channel_version::bigint, source_record_id, integrity_value, creator_principal_id FROM public.collaboration_channels WHERE community_id = $1 AND channel_id = $2",
    )
    .bind(community_id().as_uuid())
    .bind(channel_id.as_uuid())
    .fetch_one(pool)
    .await
    .expect("load channel snapshot")
}

#[tokio::test]
async fn branch_lifecycle_archives_merge_and_delete_and_reopens_without_replacing_history() {
    let Some(database_url) = std::env::var("COLLAB_BRANCH_LIFECYCLE_TEST_DATABASE_URL").ok() else {
        eprintln!(
            "COLLAB_BRANCH_LIFECYCLE_TEST_DATABASE_URL is unset; live lifecycle test skipped"
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
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'branch-lifecycle.example', 'active', 1, 'zed', 'community:branch-lifecycle', now(), now(), now())",
    )
    .bind(community_id().as_uuid())
    .execute(&pool)
    .await
    .expect("insert community");
    sqlx::query(
        "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, 'owner', 'active', 1, now(), now(), 'zed', 'principal:branch-lifecycle', now())",
    )
    .bind(community_id().as_uuid())
    .bind(creator_principal_id().as_uuid())
    .execute(&pool)
    .await
    .expect("insert creator membership");

    let channel_service = BranchChannelService::new(
        sea_orm::Database::connect(&database_url)
            .await
            .expect("connect channel service"),
    )
    .expect("channel service");
    let lifecycle_service = BranchLifecycleService::new(
        sea_orm::Database::connect(&database_url)
            .await
            .expect("connect lifecycle service"),
    )
    .expect("lifecycle service");
    let authorization = AuthorizationFixture::new();

    let active = branch("refs/heads/feature/merge-lifecycle");
    let original_binding = channel_service
        .bind(
            &active,
            creator_principal_id(),
            &authorization.community_request(),
        )
        .await
        .expect("bind active branch");
    let original_channel_id = original_binding.channel_id();
    insert_channel_membership(&pool, original_channel_id).await;
    let active_snapshot = channel_snapshot(&pool, original_channel_id).await;

    let mut merged = active.clone();
    merged
        .merge(
            AggregateVersion::FIRST,
            &commit(10),
            BranchRefName::parse("refs/heads/main").expect("target branch"),
            commit(20),
        )
        .expect("merge branch");
    let merge_result = lifecycle_service
        .apply_archive_transition(
            &active,
            &merged,
            AggregateVersion::FIRST,
            &authorization.channel_request(original_channel_id),
        )
        .await
        .expect("archive merged branch channel");
    assert_eq!(merge_result.cause(), BranchChannelLifecycleCause::Merged);
    assert_eq!(merge_result.outcome(), ChannelCommandOutcome::Applied);
    assert_eq!(
        merge_result.channel().fields().lifecycle_state,
        ChannelLifecycleState::Archived
    );
    let archived_snapshot = channel_snapshot(&pool, original_channel_id).await;
    assert_eq!(archived_snapshot.0, "archived");
    assert_eq!(archived_snapshot.1, 2);
    assert_eq!(
        (
            &archived_snapshot.2,
            &archived_snapshot.3,
            archived_snapshot.4
        ),
        (&active_snapshot.2, &active_snapshot.3, active_snapshot.4)
    );

    assert!(matches!(
        lifecycle_service
            .apply_archive_transition(
                &active,
                &merged,
                AggregateVersion::FIRST,
                &authorization.channel_request(original_channel_id),
            )
            .await,
        Err(BranchLifecycleError::StaleChannel)
    ));

    let mut archived = merged.clone();
    archived
        .archive(
            merged.fields().version,
            &commit(10),
            BranchArchiveReason::Merged,
        )
        .expect("archive merged branch");
    let replay_result = lifecycle_service
        .apply_archive_transition(
            &merged,
            &archived,
            AggregateVersion::new(2).expect("channel version two"),
            &authorization.channel_request(original_channel_id),
        )
        .await
        .expect("apply merged archive replay");
    assert_eq!(replay_result.outcome(), ChannelCommandOutcome::Unchanged);

    let reopened = archived
        .recreate(archived.fields().version, &commit(10), commit(30))
        .expect("reopen branch");
    let reopened_binding = lifecycle_service
        .reopen(
            &archived,
            &reopened,
            creator_principal_id(),
            &authorization.community_request(),
        )
        .await
        .expect("bind reopened generation");
    assert_ne!(reopened_binding.channel_id(), original_channel_id);
    assert_eq!(
        reopened_binding.channel().fields().lifecycle_state,
        ChannelLifecycleState::Active
    );
    assert_eq!(
        channel_snapshot(&pool, original_channel_id).await,
        archived_snapshot
    );

    let delete_active = branch("refs/heads/feature/delete-lifecycle");
    let delete_binding = channel_service
        .bind(
            &delete_active,
            creator_principal_id(),
            &authorization.community_request(),
        )
        .await
        .expect("bind branch to delete");
    insert_channel_membership(&pool, delete_binding.channel_id()).await;
    let mut deleted = delete_active.clone();
    deleted
        .archive(
            AggregateVersion::FIRST,
            &commit(10),
            BranchArchiveReason::Deleted,
        )
        .expect("delete branch");
    let delete_result = lifecycle_service
        .apply_archive_transition(
            &delete_active,
            &deleted,
            AggregateVersion::FIRST,
            &authorization.channel_request(delete_binding.channel_id()),
        )
        .await
        .expect("archive deleted branch channel");
    assert_eq!(delete_result.cause(), BranchChannelLifecycleCause::Deleted);
    assert_eq!(delete_result.outcome(), ChannelCommandOutcome::Applied);
    assert_eq!(
        channel_snapshot(&pool, delete_binding.channel_id()).await.0,
        "archived"
    );

    let retained_rows: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_channels WHERE community_id = $1",
    )
    .bind(community_id().as_uuid())
    .fetch_one(&pool)
    .await
    .expect("count retained channels");
    assert_eq!(retained_rows, 3);
    assert_eq!(
        branch_channel_id(&reopened.fields().identity).expect("reopened channel id"),
        reopened_binding.channel_id()
    );

    drop(lifecycle_service);
    drop(channel_service);
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll back channel migration");
}
