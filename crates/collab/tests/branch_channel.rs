use collab::{
    git::branch_channel::{BranchChannelError, BranchChannelService, branch_channel_id},
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    BranchArchiveReason, BranchCollaboration, BranchCollaborationIdentity, BranchGeneration,
    BranchRefName, CommunityId, CommunityMembership, GitCommitId, MembershipRole, MembershipStatus,
    PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use futures::future::join_all;
use sea_orm::{DatabaseBackend, MockDatabase};
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

fn branch() -> BranchCollaboration {
    BranchCollaboration::create(
        BranchCollaborationIdentity::new(
            community_id(),
            repository_id(),
            BranchRefName::parse("refs/heads/feature/concurrent-binding").expect("branch ref"),
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
    fn new(role: MembershipRole) -> Self {
        let community_id = community_id();
        let principal_id = creator_principal_id();
        let scope = AuthorizationScope::new("channels:manage").expect("scope");
        Self {
            tenant: bind_rpc_tenant(
                Some(
                    TrustedTenantRoute::from_listener(community_id, "branch-channel-test")
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
                role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
        }
    }

    fn request(&self) -> AuthorizationRequest<'_> {
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
}

#[tokio::test]
async fn branch_channel_rejects_unapproved_or_inactive_branches_before_storage() {
    let service =
        BranchChannelService::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("service");
    let member = AuthorizationFixture::new(MembershipRole::Member);
    assert!(matches!(
        service
            .bind(&branch(), creator_principal_id(), &member.request())
            .await,
        Err(BranchChannelError::Domain(_))
    ));
    let owner = AuthorizationFixture::new(MembershipRole::Owner);
    let mut archived = branch();
    archived
        .archive(
            AggregateVersion::FIRST,
            &commit(10),
            BranchArchiveReason::Deleted,
        )
        .expect("archive");
    assert!(matches!(
        service
            .bind(&archived, creator_principal_id(), &owner.request())
            .await,
        Err(BranchChannelError::BranchUnavailable)
    ));
}

#[tokio::test]
async fn branch_channel_duplicate_concurrent_and_reconnect_creates_converge() {
    let Some(database_url) = std::env::var("COLLAB_BRANCH_CHANNEL_TEST_DATABASE_URL").ok() else {
        eprintln!(
            "COLLAB_BRANCH_CHANNEL_TEST_DATABASE_URL is unset; live concurrency test skipped"
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
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'branch-channel.example', 'active', 1, 'zed', 'community:branch-channel', now(), now(), now())",
    )
    .bind(community_id().as_uuid())
    .execute(&pool)
    .await
    .expect("insert community");
    sqlx::query(
        "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, 'owner', 'active', 1, now(), now(), 'zed', 'principal:branch-channel', now())",
    )
    .bind(community_id().as_uuid())
    .bind(creator_principal_id().as_uuid())
    .execute(&pool)
    .await
    .expect("insert creator membership");

    let connection = sea_orm::Database::connect(&database_url)
        .await
        .expect("connect service");
    let service = BranchChannelService::new(connection).expect("service");
    let branch = branch();
    let expected_channel_id = branch_channel_id(&branch.fields().identity).expect("channel id");
    let results = join_all((0..16).map(|_| {
        let service = service.clone();
        let branch = branch.clone();
        async move {
            let authorization = AuthorizationFixture::new(MembershipRole::Owner);
            service
                .bind(&branch, creator_principal_id(), &authorization.request())
                .await
        }
    }))
    .await;
    for result in results {
        let binding = result.expect("concurrent binding");
        assert_eq!(binding.branch(), &branch.fields().identity);
        assert_eq!(binding.channel_id(), expected_channel_id);
    }

    let row_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_channels WHERE community_id = $1 AND channel_id = $2",
    )
    .bind(community_id().as_uuid())
    .bind(expected_channel_id.as_uuid())
    .fetch_one(&pool)
    .await
    .expect("count branch channels");
    assert_eq!(row_count, 1);
    let authorization = AuthorizationFixture::new(MembershipRole::Owner);
    let reconnected = service
        .bind(&branch, creator_principal_id(), &authorization.request())
        .await
        .expect("reconnect binding");
    assert_eq!(reconnected.channel_id(), expected_channel_id);

    drop(service);
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll back channel migration");
}
