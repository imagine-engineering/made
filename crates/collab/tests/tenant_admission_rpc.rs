use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use collab::tenant_admission::{AuthorizedRpcRequest, RpcAdmissionError, bind_rpc_tenant};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, PrincipalId,
    PrincipalScopes, ServiceAccountId, TrustedTenantRoute, UntrustedTenantClaim,
    UntrustedTenantClaimSource,
};
use uuid::Uuid;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn request_fixture<'a>(
    tenant: &'a collaboration_domain::TenantContext,
    authenticated: &'a AuthenticatedPrincipal,
    required_scope: &'a AuthorizationScope,
    membership: Option<CommunityMembership>,
) -> AuthorizationRequest<'a> {
    AuthorizationRequest {
        tenant,
        principal: authenticated,
        required_scope,
        action: AuthorizationAction::Read,
        resource: AuthorizationResource {
            community_id: tenant.community_id(),
            kind: AuthorizationResourceKind::Project,
            resource_id: AggregateId::from_uuid(Uuid::from_u128(20)),
            owner_principal_id: None,
            channel_id: None,
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: membership,
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis: 100,
    }
}

#[tokio::test]
async fn tenant_admission_rpc_authorizes_before_database_queries() {
    let community_id = community(1);
    let principal_id = principal(2);
    let tenant = bind_rpc_tenant(
        Some(TrustedTenantRoute::from_listener(community_id, "zed-rpc").expect("trusted route")),
        &[],
    )
    .expect("tenant");
    let required_scope = AuthorizationScope::new("projects:read").expect("scope");
    let scopes = PrincipalScopes::new([required_scope.clone()]).expect("scopes");
    let authenticated = AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(3),
        scopes,
    );
    let queries = Arc::new(AtomicUsize::new(0));

    let denied_request = request_fixture(&tenant, &authenticated, &required_scope, None);
    assert!(matches!(
        AuthorizedRpcRequest::authorize(&denied_request),
        Err(RpcAdmissionError::Denied)
    ));
    assert_eq!(queries.load(Ordering::SeqCst), 0);

    let membership = CommunityMembership {
        community_id,
        principal_id,
        role: MembershipRole::Member,
        status: MembershipStatus::Active,
        version: AggregateVersion::FIRST,
    };
    let allowed_request =
        request_fixture(&tenant, &authenticated, &required_scope, Some(membership));
    let authorized = AuthorizedRpcRequest::authorize(&allowed_request).expect("authorized");
    let result = authorized
        .run({
            let queries = queries.clone();
            move |query_tenant, query_principal| async move {
                queries.fetch_add(1, Ordering::SeqCst);
                assert_eq!(query_tenant.community_id(), community_id);
                assert_eq!(query_principal.principal_id(), principal_id);
                Ok::<_, ()>("queried")
            }
        })
        .await;
    assert_eq!(result, Ok("queried"));
    assert_eq!(queries.load(Ordering::SeqCst), 1);
}

#[test]
fn tenant_admission_rpc_rejects_payload_selected_and_conflicting_tenants() {
    let event_claim = UntrustedTenantClaim::new(community(1), UntrustedTenantClaimSource::EventTag);
    assert_eq!(
        bind_rpc_tenant(None, &[event_claim]),
        Err(RpcAdmissionError::Denied)
    );

    let route = TrustedTenantRoute::from_listener(community(1), "zed-rpc").expect("trusted route");
    let body_claim = UntrustedTenantClaim::new(community(2), UntrustedTenantClaimSource::BodyField);
    assert_eq!(
        bind_rpc_tenant(Some(route), &[body_claim]),
        Err(RpcAdmissionError::Denied)
    );
}

fn membership_tenant(value: u128) -> collaboration_domain::TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community(value), "current-membership-fixture")
                .expect("fixture route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn membership_principal(
    community_id: CommunityId,
    principal_id: PrincipalId,
) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(7),
        PrincipalScopes::new([
            AuthorizationScope::new("projects:read").expect("fixture read scope")
        ])
        .expect("fixture scopes"),
    )
}

#[tokio::test]
async fn current_principal_memberships_independent_channel_component_snapshot() -> anyhow::Result<()>
{
    use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, TransactionTrait as _, Value};
    let tenant = membership_tenant(701);
    let principal = membership_principal(tenant.community_id(), principal(702));
    let channel_id = AggregateId::from_uuid(Uuid::from_u128(703));
    let row = std::collections::BTreeMap::<String, Value>::from([
        (
            "community_membership_community_id".into(),
            Option::<Uuid>::None.into(),
        ),
        (
            "channel_membership_community_id".into(),
            tenant.community_id().as_uuid().into(),
        ),
        (
            "channel_membership_principal_id".into(),
            principal.principal_id().as_uuid().into(),
        ),
        (
            "channel_membership_channel_id".into(),
            channel_id.as_uuid().into(),
        ),
        ("channel_role".into(), "guest".into()),
        ("channel_status".into(), "active".into()),
        ("channel_version".into(), "8".into()),
    ]);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![row]])
        .append_exec_results([MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }])
        .into_connection();
    let transaction = database.begin().await?;
    let memberships =
        collab::messages::channel_admission::current_principal_memberships_in_transaction(
            &transaction,
            &tenant,
            &principal,
            Some(channel_id),
        )
        .await?;
    anyhow::ensure!(
        memberships.community.is_none(),
        "component snapshot must retain absent community membership"
    );
    let channel = memberships
        .channel
        .ok_or_else(|| anyhow::anyhow!("independent channel snapshot was incorrectly discarded"))?;
    anyhow::ensure!(
        channel.community_id == tenant.community_id()
            && channel.principal_id == principal.principal_id()
            && channel.channel_id == channel_id
            && channel.role == MembershipRole::Guest
            && channel.version.get() == 8,
        "component snapshot must return exact independently sourced channel tuple"
    );
    transaction.rollback().await?;
    let log = format!("{:#?}", database.into_transaction_log());
    anyhow::ensure!(
        log.contains("LEFT JOIN public.collaboration_community_memberships")
            && log.contains("LEFT JOIN public.collaboration_channel_memberships")
            && !log.contains("INSERT ")
            && !log.contains("UPDATE ")
            && !log.contains("DELETE "),
        "component query evidence must preserve independent joins and no membership side effects"
    );
    Ok(())
}

#[tokio::test]
async fn current_principal_memberships_rejects_invalid_version_component_snapshot()
-> anyhow::Result<()> {
    use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, TransactionTrait as _, Value};
    let tenant = membership_tenant(701);
    let principal = membership_principal(tenant.community_id(), principal(702));
    for version in ["0", "18446744073709551616"] {
        let row = std::collections::BTreeMap::<String, Value>::from([
            (
                "community_membership_community_id".into(),
                tenant.community_id().as_uuid().into(),
            ),
            (
                "community_membership_principal_id".into(),
                principal.principal_id().as_uuid().into(),
            ),
            ("community_role".into(), "member".into()),
            ("community_status".into(), "active".into()),
            ("community_version".into(), version.into()),
            (
                "channel_membership_channel_id".into(),
                Option::<Uuid>::None.into(),
            ),
        ]);
        let database = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([vec![row]])
            .append_exec_results([MockExecResult {
                last_insert_id: 0,
                rows_affected: 1,
            }])
            .into_connection();
        let transaction = database.begin().await?;
        let result =
            collab::messages::channel_admission::current_principal_memberships_in_transaction(
                &transaction,
                &tenant,
                &principal,
                None,
            )
            .await;
        anyhow::ensure!(
            result.is_err(),
            "invalid persisted version must not become membership authority"
        );
        transaction.rollback().await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn current_principal_memberships_real_transaction_and_tenant_fences() -> anyhow::Result<()> {
    use sqlx::migrate::MigrateDatabase as _;
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL").map_err(|_| {
        anyhow::anyhow!(
            "COLLAB_TEST_DATABASE_URL is required for real membership-read qualification"
        )
    })?;
    let mut url = url::Url::parse(&base_url)?;
    anyhow::ensure!(
        matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback PostgreSQL required"
    );
    url.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let url = url.to_string();
    sqlx::Postgres::create_database(&url).await?;
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&url)
        .await;
    let result = match pool {
        Ok(pool) => {
            let result = exercise_current_memberships(&pool).await;
            pool.close().await;
            result
        }
        Err(error) => Err(anyhow::Error::from(error)),
    };
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(anyhow::anyhow!(
            "owned membership database cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "membership-read qualification failed: {error:#}; owned database cleanup failed: {cleanup}"
        )),
    }
}

async fn exercise_current_memberships(pool: &sqlx::PgPool) -> anyhow::Result<()> {
    use collab::messages::channel_admission::current_principal_memberships_in_transaction;
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    sqlx::raw_sql(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/migrations/20260820000700_collaboration_channels.up.sql"
    )))
    .execute(pool)
    .await?;
    let principal_id = principal(702);
    let channel_id = AggregateId::from_uuid(Uuid::from_u128(703));
    for (community_number, community_role, community_version, channel_role, channel_version) in [
        (701, "member", 3_i64, "guest", 7_i64),
        (704, "owner", 5_i64, "admin", 9_i64),
    ] {
        let community_id = community(community_number);
        sqlx::query("INSERT INTO public.collaboration_communities (community_id,host,lifecycle_state,aggregate_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,'active',1,'zed','membership-read-fixture',now(),now(),now())")
            .bind(community_id.as_uuid()).bind(format!("membership-{community_number}.example.test")).execute(pool).await?;
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,'active',$4,now(),now(),'zed','membership-read-fixture',now())")
            .bind(community_id.as_uuid()).bind(principal_id.as_uuid()).bind(community_role).bind(community_version).execute(pool).await?;
        sqlx::query("INSERT INTO public.collaboration_channels (community_id,channel_id,name,channel_type,visibility,lifecycle_state,creator_principal_id,channel_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,'Fixture','stream','private','active',$3,1,'zed','membership-read-fixture',now(),now(),now())")
            .bind(community_id.as_uuid()).bind(channel_id.as_uuid()).bind(principal_id.as_uuid()).execute(pool).await?;
        sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,$4,'active',$5,now(),now(),'zed','membership-read-fixture',now())")
            .bind(community_id.as_uuid()).bind(channel_id.as_uuid()).bind(principal_id.as_uuid()).bind(channel_role).bind(channel_version).execute(pool).await?;
    }
    let database = sea_orm::DatabaseConnection::from(pool.clone());
    let result = async {
        let tenant_a = membership_tenant(701);
        let tenant_b = membership_tenant(704);
        let actor_a = membership_principal(tenant_a.community_id(), principal_id);
        let actor_b = membership_principal(tenant_b.community_id(), principal_id);
        let transaction = database.begin().await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres, "SET LOCAL ROLE pg_read_all_data".to_owned())).await?;
        let role = transaction.query_one(Statement::from_string(DatabaseBackend::Postgres, "SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user".to_owned())).await?.ok_or_else(|| anyhow::anyhow!("missing request role"))?;
        anyhow::ensure!(!role.try_get::<bool>("", "rolbypassrls")?, "real read fixture must run without BYPASSRLS");
        let bounded = tokio::time::timeout(std::time::Duration::from_secs(5), current_principal_memberships_in_transaction(&transaction, &tenant_a, &actor_a, Some(channel_id))).await;
        let own = bounded.map_err(|_| anyhow::anyhow!("caller-owned single-connection read deadlocked"))??;
        anyhow::ensure!(own.community.is_some_and(|member| member.role == MembershipRole::Member && member.version.get() == 3)
            && own.channel.is_some_and(|member| member.role == MembershipRole::Guest && member.version.get() == 7), "independent exact roles and versions must not inherit authority");
        let read_scope = AuthorizationScope::new("projects:read")?;
        let mut active_request = request_fixture(&tenant_a, &actor_a, &read_scope, own.community);
        active_request.current_membership_version = own.community.ok_or_else(|| anyhow::anyhow!("missing active membership"))?.version;
        anyhow::ensure!(collaboration_domain::authorize(&active_request) == collaboration_domain::AuthorizationDecision::Allowed,
            "current active member must satisfy identical scope/version policy before revocation");
        let token = AuthenticatedPrincipal::scoped_token(principal(888), tenant_a.community_id(),
            collaboration_domain::TokenId::from_uuid(Uuid::from_u128(889)), principal_id,
            PrincipalScopes::new([read_scope.clone()])?);
        anyhow::ensure!(token.principal_id() != principal_id, "token fixture envelope must differ from membership subject");
        let token_memberships = current_principal_memberships_in_transaction(&transaction, &tenant_a, &token, Some(channel_id)).await?;
        anyhow::ensure!(token_memberships == own,
            "scoped token must resolve exact canonical authorization subject, never envelope identity or guessed role");
        let mut token_read = request_fixture(&tenant_a, &token, &read_scope, token_memberships.community);
        token_read.current_membership_version = own.community.ok_or_else(|| anyhow::anyhow!("missing token subject membership"))?.version;
        anyhow::ensure!(collaboration_domain::authorize(&token_read) == collaboration_domain::AuthorizationDecision::Allowed,
            "canonical token subject lookup must preserve permitted read scope");
        let write_scope = AuthorizationScope::new("projects:write")?;
        let token_write = AuthorizationRequest { required_scope: &write_scope, action: AuthorizationAction::Write, ..token_read };
        anyhow::ensure!(collaboration_domain::authorize(&token_write) == collaboration_domain::AuthorizationDecision::Denied(collaboration_domain::AuthorizationDenial::MissingScope),
            "subject membership must not expand token scopes");
        let mismatched_token = current_principal_memberships_in_transaction(&transaction, &tenant_b, &token, Some(channel_id)).await;
        anyhow::ensure!(mismatched_token.is_err(), "same token subject must not bypass token tenant stamp");
        let no_channel = current_principal_memberships_in_transaction(&transaction, &tenant_a, &actor_a, None).await?;
        anyhow::ensure!(no_channel.community == own.community && no_channel.channel.is_none(), "community-only request must not infer any channel");
        let absent_channel = current_principal_memberships_in_transaction(&transaction, &tenant_a, &actor_a, Some(AggregateId::new())).await?;
        anyhow::ensure!(absent_channel.community == own.community && absent_channel.channel.is_none(), "missing channel must remain independently absent");
        let missing_actor = membership_principal(tenant_a.community_id(), principal(799));
        let missing = current_principal_memberships_in_transaction(&transaction, &tenant_a, &missing_actor, Some(channel_id)).await?;
        anyhow::ensure!(missing.community.is_none() && missing.channel.is_none(), "unrelated principal must not receive existing memberships");
        let mismatched = current_principal_memberships_in_transaction(&transaction, &tenant_b, &actor_a, Some(channel_id)).await;
        anyhow::ensure!(mismatched.is_err(), "principal/tenant mismatch must deny before query");
        let context = transaction.query_one(Statement::from_string(DatabaseBackend::Postgres, "SELECT current_setting('app.community_id', true) AS tenant".to_owned())).await?.ok_or_else(|| anyhow::anyhow!("missing transaction tenant"))?;
        anyhow::ensure!(context.try_get::<String>("", "tenant")? == tenant_a.community_id().to_string(), "denied mismatch must not mutate caller transaction tenant");
        let foreign_visible = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres, "SELECT count(*) AS count FROM public.collaboration_community_memberships WHERE community_id=$1", [tenant_b.community_id().as_uuid().into()])).await?.ok_or_else(|| anyhow::anyhow!("missing foreign count"))?;
        anyhow::ensure!(foreign_visible.try_get::<i64>("", "count")? == 0, "forced RLS must hide unrelated tenant even with same principal UUID");
        let foreign = current_principal_memberships_in_transaction(&transaction, &tenant_b, &actor_b, Some(channel_id)).await?;
        anyhow::ensure!(foreign.community.is_some_and(|member| member.role == MembershipRole::Owner && member.version.get() == 5)
            && foreign.channel.is_some_and(|member| member.role == MembershipRole::Admin && member.version.get() == 9), "trusted second tenant must retain its own exact snapshot");
        let foreign_token = AuthenticatedPrincipal::scoped_token(principal(888), tenant_b.community_id(),
            collaboration_domain::TokenId::from_uuid(Uuid::from_u128(889)), principal_id,
            PrincipalScopes::new([read_scope.clone()])?);
        let foreign_token_memberships = current_principal_memberships_in_transaction(&transaction, &tenant_b, &foreign_token, Some(channel_id)).await?;
        anyhow::ensure!(foreign_token_memberships == foreign && foreign_token_memberships != token_memberships,
            "same subject and envelope UUIDs must retain exact tenant-specific roles and independent channel versions");
        transaction.rollback().await?;
        let transaction = database.begin().await?;
        transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres, "UPDATE public.collaboration_community_memberships SET status='revoked',membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2", [tenant_a.community_id().as_uuid().into(), principal_id.as_uuid().into()])).await?;
        transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres, "UPDATE public.collaboration_channel_memberships SET status='archived',membership_version=membership_version+1 WHERE community_id=$1 AND channel_id=$2 AND principal_id=$3", [tenant_a.community_id().as_uuid().into(), channel_id.as_uuid().into(), principal_id.as_uuid().into()])).await?;
        let changed = current_principal_memberships_in_transaction(&transaction, &tenant_a, &actor_a, Some(channel_id)).await?;
        anyhow::ensure!(changed.community.is_some_and(|member| member.status == MembershipStatus::Revoked && member.version.get() == 4)
            && changed.channel.is_some_and(|member| member.status == MembershipStatus::Archived && member.version.get() == 8), "helper must read uncommitted caller transaction changes rather than a nested pool snapshot");
        let scope = AuthorizationScope::new("projects:read")?;
        let mut request = request_fixture(&tenant_a, &actor_a, &scope, changed.community);
        request.current_membership_version = changed.community.ok_or_else(|| anyhow::anyhow!("missing revoked membership"))?.version;
        anyhow::ensure!(collaboration_domain::authorize(&request) == collaboration_domain::AuthorizationDecision::Denied(collaboration_domain::AuthorizationDenial::InactiveMembership),
            "actual revoked state must deny independently of scope and version");
        transaction.rollback().await?;
        let transaction = database.begin().await?;
        let restored = current_principal_memberships_in_transaction(&transaction, &tenant_a, &actor_a, Some(channel_id)).await?;
        anyhow::ensure!(restored == own, "caller rollback preserves original independent memberships");
        transaction.rollback().await?;
        sqlx::raw_sql("CREATE POLICY membership_fixture_hidden_community ON public.collaboration_community_memberships AS RESTRICTIVE FOR SELECT USING (false)").execute(pool).await?;
        let visibility = async {
            let transaction = database.begin().await?;
            transaction.execute(Statement::from_string(DatabaseBackend::Postgres, "SET LOCAL ROLE pg_read_all_data".to_owned())).await?;
            let snapshot = current_principal_memberships_in_transaction(&transaction, &tenant_a, &actor_a, Some(channel_id)).await?;
            transaction.rollback().await?;
            Ok::<_, anyhow::Error>(snapshot)
        }.await;
        sqlx::raw_sql("DROP POLICY membership_fixture_hidden_community ON public.collaboration_community_memberships").execute(pool).await?;
        let visibility = visibility?;
        anyhow::ensure!(visibility.community.is_none() && visibility.channel == own.channel,
            "actual SQL must independently return exact visible channel tuple when stricter fixture SELECT policy hides its FK-valid community tuple; this is visibility evidence, not physical channel-only storage");
        Ok::<(),anyhow::Error>(())
    }.await;
    drop(database);
    result?;
    let expected_constraint: String = sqlx::query_scalar("SELECT conname FROM pg_constraint WHERE conrelid='public.collaboration_channel_memberships'::regclass AND confrelid='public.collaboration_community_memberships'::regclass")
        .fetch_one(pool).await?;
    let foreign_key_negative = sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,'guest','active',1,now(),now(),'zed','channel-only-negative',now())")
        .bind(community(701).as_uuid()).bind(channel_id.as_uuid()).bind(principal(799).as_uuid()).execute(pool).await;
    anyhow::ensure!(
        foreign_key_negative
            .as_ref()
            .err()
            .and_then(sqlx::Error::as_database_error)
            .is_some_and(|error| error.code().as_deref() == Some("23503")
                && error.constraint() == Some(expected_constraint.as_str())),
        "current canonical FK must reject physical channel-only membership; do not weaken it to manufacture parity"
    );
    eprintln!(
        "LIVE_CURRENT_MEMBERSHIP_READS_VERIFIED: supplied single-connection transaction, exact independent snapshots, no-BYPASS RLS, revocation/rollback; physical channel-only membership remains FK-blocked, component-only independence not live virtual admission"
    );
    Ok(())
}

#[tokio::test]
async fn current_principal_memberships_decoder_and_coordinate_negatives() -> anyhow::Result<()> {
    use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, TransactionTrait as _, Value};
    let tenant = membership_tenant(701);
    let principal = membership_principal(tenant.community_id(), principal(702));
    let channel_id = AggregateId::from_uuid(Uuid::from_u128(703));
    for (role, status, version, stored_principal, stored_channel) in [
        ("invented", "active", "8", 702, 703),
        ("guest", "invented", "8", 702, 703),
        ("guest", "active", "0", 702, 703),
        ("guest", "active", "18446744073709551616", 702, 703),
        ("guest", "active", "8", 799, 703),
        ("guest", "active", "8", 702, 799),
    ] {
        let row = std::collections::BTreeMap::<String, Value>::from([
            (
                "community_membership_community_id".into(),
                Option::<Uuid>::None.into(),
            ),
            (
                "channel_membership_community_id".into(),
                tenant.community_id().as_uuid().into(),
            ),
            (
                "channel_membership_principal_id".into(),
                Uuid::from_u128(stored_principal).into(),
            ),
            (
                "channel_membership_channel_id".into(),
                Uuid::from_u128(stored_channel).into(),
            ),
            ("channel_role".into(), role.into()),
            ("channel_status".into(), status.into()),
            ("channel_version".into(), version.into()),
        ]);
        let database = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([vec![row]])
            .append_exec_results([MockExecResult {
                last_insert_id: 0,
                rows_affected: 1,
            }])
            .into_connection();
        let transaction = database.begin().await?;
        let result =
            collab::messages::channel_admission::current_principal_memberships_in_transaction(
                &transaction,
                &tenant,
                &principal,
                Some(channel_id),
            )
            .await;
        anyhow::ensure!(
            result.is_err(),
            "invalid decoder or mismatched returned coordinate must not become membership authority"
        );
        transaction.rollback().await?;
    }
    Ok(())
}
