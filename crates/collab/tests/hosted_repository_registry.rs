use collab::{
    git::repository_registry::{
        ExternalProviderCoordinate, HostedAuthority, HostedRepositoryDraft,
        HostedRepositoryLifecycle, HostedRepositoryRegistry, HostedRepositoryRegistryError,
        RepositoryCoordinate, RepositoryPermission,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, NostrPublicKey,
    PrincipalId, PrincipalScopes, Provenance, ServiceAccountId, SourceRecordId, SourceSystem,
    TenantContext, TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase};
use sqlx::PgPool;
use url::Url;
use uuid::Uuid;

const CHANNELS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const CHANNELS_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.down.sql"
));
const GIT_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000400_collaboration_git.up.sql"
));
const GIT_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000400_collaboration_git.down.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal_id(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn repository_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "hosted-repository-registry")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn principal(community_id: CommunityId, principal_id: PrincipalId) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(7),
        PrincipalScopes::new(
            ["git:read", "git:write", "git:admin"]
                .into_iter()
                .map(|value| AuthorizationScope::new(value).expect("scope")),
        )
        .expect("principal scopes"),
    )
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

fn scope(permission: RepositoryPermission) -> AuthorizationScope {
    AuthorizationScope::new(match permission {
        RepositoryPermission::Read => "git:read",
        RepositoryPermission::Write => "git:write",
        RepositoryPermission::Admin => "git:admin",
    })
    .expect("scope")
}

fn access<'a>(
    tenant: &'a TenantContext,
    principal: &'a AuthenticatedPrincipal,
    required_scope: &'a AuthorizationScope,
    repository_id: AggregateId,
    permission: RepositoryPermission,
    membership: CommunityMembership,
) -> AuthorizationRequest<'a> {
    AuthorizationRequest {
        tenant,
        principal,
        required_scope,
        action: match permission {
            RepositoryPermission::Read => AuthorizationAction::Read,
            RepositoryPermission::Write => AuthorizationAction::Write,
            RepositoryPermission::Admin => AuthorizationAction::Manage,
        },
        resource: AuthorizationResource {
            community_id: tenant.community_id(),
            kind: AuthorizationResourceKind::Repository,
            resource_id: repository_id,
            owner_principal_id: None,
            channel_id: None,
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: Some(membership),
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis: 1_900_000_000_000,
    }
}

fn draft(
    community_id: CommunityId,
    repository_id: AggregateId,
    discriminator: &str,
    authority: HostedAuthority,
    source_system: SourceSystem,
) -> HostedRepositoryDraft {
    HostedRepositoryDraft {
        community_id,
        repository_id,
        coordinate: RepositoryCoordinate::new(NostrPublicKey::from_bytes([9; 32]), discriminator)
            .expect("repository coordinate"),
        authority,
        provenance: Provenance::new(
            source_system,
            SourceRecordId::new(format!("repository:{discriminator}")).expect("source record id"),
            1_900_000_000_000,
        )
        .with_source_version("1"),
        created_at_millis: 1_900_000_000_000,
    }
}

#[tokio::test]
async fn hosted_repository_registry_rejects_tenant_and_scope_before_database_work() {
    let community_a = community(1);
    let community_b = community(2);
    let repository_id = repository_id(3);
    let tenant_a = tenant(community_a);
    let principal_a = principal(community_a, principal_id(4));
    let wrong_scope = AuthorizationScope::new("messages:read").expect("scope");
    let wrong_scope_request = access(
        &tenant_a,
        &principal_a,
        &wrong_scope,
        repository_id,
        RepositoryPermission::Read,
        membership(
            community_a,
            principal_a.principal_id(),
            MembershipRole::Member,
        ),
    );
    let repository = HostedRepositoryRegistry::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
    )
    .expect("registry");
    assert!(matches!(
        repository
            .authorize_access(&wrong_scope_request, RepositoryPermission::Read)
            .await,
        Err(HostedRepositoryRegistryError::PermissionDenied)
    ));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );

    let tenant_b = tenant(community_b);
    let principal_b = principal(community_b, principal_id(5));
    let read_scope = scope(RepositoryPermission::Read);
    let mut cross_tenant = access(
        &tenant_b,
        &principal_b,
        &read_scope,
        repository_id,
        RepositoryPermission::Read,
        membership(
            community_b,
            principal_b.principal_id(),
            MembershipRole::Member,
        ),
    );
    cross_tenant.resource.community_id = community_a;
    let repository = HostedRepositoryRegistry::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
    )
    .expect("registry");
    assert!(matches!(
        repository
            .authorize_access(&cross_tenant, RepositoryPermission::Read)
            .await,
        Err(HostedRepositoryRegistryError::TenantBoundaryViolation)
    ));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}

#[tokio::test]
async fn hosted_repository_registry_enforces_permissions_rename_archive_and_provider_coexistence() {
    let Some(database_url) = std::env::var("COLLAB_HOSTED_REPOSITORY_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_HOSTED_REPOSITORY_TEST_DATABASE_URL is unset; live registry test skipped"
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
    sqlx::raw_sql(GIT_UP)
        .execute(&pool)
        .await
        .expect("apply Git migration");

    let community_id = community(10);
    let owner_id = principal_id(11);
    let reader_id = principal_id(12);
    sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'registry.example', 'active', 1, 'zed', 'community:registry', now(), now(), now())",
    )
    .bind(community_id.as_uuid())
    .execute(&pool)
    .await
    .expect("insert community");
    for (principal_id, role) in [(owner_id, "owner"), (reader_id, "member")] {
        sqlx::query(
            "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, 'active', 1, now(), now(), 'zed', $2::text, now())",
        )
        .bind(community_id.as_uuid())
        .bind(principal_id.as_uuid())
        .bind(role)
        .execute(&pool)
        .await
        .expect("insert membership");
    }
    sqlx::raw_sql(
        "CREATE ROLE collaboration_git_registry LOGIN PASSWORD 'registry-test' NOBYPASSRLS; \
         GRANT USAGE ON SCHEMA public TO collaboration_git_registry; \
         GRANT SELECT ON public.collaboration_community_memberships TO collaboration_git_registry; \
         GRANT SELECT, INSERT, UPDATE ON public.collaboration_hosted_repositories, \
         public.collaboration_git_storage_handles, public.collaboration_git_repository_grants \
         TO collaboration_git_registry;",
    )
    .execute(&pool)
    .await
    .expect("create registry request role");
    let mut role_url = Url::parse(&database_url).expect("database URL");
    role_url
        .set_username("collaboration_git_registry")
        .expect("set role username");
    role_url
        .set_password(Some("registry-test"))
        .expect("set role password");
    let connection = sea_orm::Database::connect(role_url.as_str())
        .await
        .expect("connect registry request role");
    let registry = HostedRepositoryRegistry::new(connection).expect("Postgres registry");
    let tenant = tenant(community_id);
    let owner = principal(community_id, owner_id);
    let reader = principal(community_id, reader_id);
    let admin_scope = scope(RepositoryPermission::Admin);
    let read_scope = scope(RepositoryPermission::Read);
    let write_scope = scope(RepositoryPermission::Write);
    let sim_repository_id = repository_id(20);
    let external_repository_id = repository_id(21);
    let sim_admin = access(
        &tenant,
        &owner,
        &admin_scope,
        sim_repository_id,
        RepositoryPermission::Admin,
        membership(community_id, owner_id, MembershipRole::Owner),
    );
    let external_admin = access(
        &tenant,
        &owner,
        &admin_scope,
        external_repository_id,
        RepositoryPermission::Admin,
        membership(community_id, owner_id, MembershipRole::Owner),
    );

    let sim = registry
        .create(
            &sim_admin,
            &draft(
                community_id,
                sim_repository_id,
                "sim-repository",
                HostedAuthority::SimHostedNip34 {
                    storage_handle_id: Uuid::from_u128(30),
                },
                SourceSystem::Buzz,
            ),
        )
        .await
        .expect("create Sim-hosted repository");
    assert!(matches!(
        sim.authority,
        HostedAuthority::SimHostedNip34 { .. }
    ));
    let external = registry
        .create(
            &external_admin,
            &draft(
                community_id,
                external_repository_id,
                "external-repository",
                HostedAuthority::ExternalProvider(
                    ExternalProviderCoordinate::new("github", "github.com", "imagineerings", "xor")
                        .expect("provider coordinate"),
                ),
                SourceSystem::ExternalGit,
            ),
        )
        .await
        .expect("create external repository");
    assert!(matches!(
        external.authority,
        HostedAuthority::ExternalProvider(_)
    ));

    registry
        .grant(
            &sim_admin,
            reader_id,
            RepositoryPermission::Read,
            1_900_000_001_000,
        )
        .await
        .expect("grant repository read");
    let reader_membership = membership(community_id, reader_id, MembershipRole::Member);
    let reader_read = access(
        &tenant,
        &reader,
        &read_scope,
        sim_repository_id,
        RepositoryPermission::Read,
        reader_membership,
    );
    registry
        .authorize_access(&reader_read, RepositoryPermission::Read)
        .await
        .expect("explicit reader may read");
    let reader_write = access(
        &tenant,
        &reader,
        &write_scope,
        sim_repository_id,
        RepositoryPermission::Write,
        reader_membership,
    );
    assert!(matches!(
        registry
            .authorize_access(&reader_write, RepositoryPermission::Write)
            .await,
        Err(HostedRepositoryRegistryError::PermissionDenied)
    ));
    assert!(matches!(
        registry
            .authorize_access(
                &access(
                    &tenant,
                    &reader,
                    &read_scope,
                    external_repository_id,
                    RepositoryPermission::Read,
                    reader_membership,
                ),
                RepositoryPermission::Read
            )
            .await,
        Err(HostedRepositoryRegistryError::PermissionDenied)
    ));

    let renamed = registry
        .rename(
            &sim_admin,
            AggregateVersion::FIRST,
            "renamed-repository",
            1_900_000_002_000,
        )
        .await
        .expect("rename repository");
    assert_eq!(renamed.repository_id, sim_repository_id);
    assert_eq!(renamed.coordinate.discriminator, "renamed-repository");
    assert_eq!(renamed.authority_version.get(), 2);
    assert!(matches!(
        registry
            .rename(
                &sim_admin,
                AggregateVersion::FIRST,
                "stale-rename",
                1_900_000_003_000,
            )
            .await,
        Err(HostedRepositoryRegistryError::VersionConflict)
    ));
    registry
        .revoke(
            &sim_admin,
            reader_id,
            RepositoryPermission::Read,
            1_900_000_004_000,
        )
        .await
        .expect("revoke repository read");
    assert!(matches!(
        registry
            .authorize_access(&reader_read, RepositoryPermission::Read)
            .await,
        Err(HostedRepositoryRegistryError::PermissionDenied)
    ));
    let archived = registry
        .archive(
            &sim_admin,
            AggregateVersion::new(2).expect("version two"),
            1_900_000_005_000,
        )
        .await
        .expect("archive repository");
    assert_eq!(archived.lifecycle, HostedRepositoryLifecycle::Archived);
    assert_eq!(archived.authority_version.get(), 3);
    assert!(matches!(
        registry
            .authorize_access(&sim_admin, RepositoryPermission::Admin)
            .await,
        Err(HostedRepositoryRegistryError::PermissionDenied)
    ));
    registry
        .into_connection()
        .close()
        .await
        .expect("close registry connection");

    sqlx::raw_sql(
        "DROP OWNED BY collaboration_git_registry; DROP ROLE collaboration_git_registry;",
    )
    .execute(&pool)
    .await
    .expect("remove registry request role");
    sqlx::raw_sql(GIT_DOWN)
        .execute(&pool)
        .await
        .expect("roll Git migration down");
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll channel migration down");
}
