use std::sync::Arc;

use collab::{
    db::collaboration::event_repository::{
        EventRepository, EventVerificationState, VerifiedEventRecord,
    },
    identity::profile_repository::{
        IdentityProfileProjectionOutcome, IdentityProfileRepository,
        IdentityProfileRepositoryError, PersistedIdentityProfile,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    CommunityId, NostrPublicKey, ProfileKind, TenantContext, TrustedTenantRoute,
};
use nostr_compat::{CanonicalEvent, EventSignature, PublicKey, SignedEvent, TimestampPolicy};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use sha2::{Digest as _, Sha256};
use sqlx::migrate::MigrateDatabase;
use tokio::sync::Barrier;
use uuid::Uuid;

const NOW: u64 = 1_900_000_000;
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20261009000100_collaboration_identity_profiles.up.sql"
));
const REGISTRATION_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20261010000100_collaboration_identity_registration.up.sql"
));
const REGISTRATION_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20261010000100_collaboration_identity_registration.down.sql"
));
const OBSERVATION_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20261010000200_collaboration_identity_observation.up.sql"
));
const OBSERVATION_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20261010000200_collaboration_identity_observation.down.sql"
));
const MODERATION_SUBJECT_UP: &str =
    include_str!("../migrations/20261010000300_collaboration_moderation_subject_targets.up.sql");
const MODERATION_SUBJECT_DOWN: &str =
    include_str!("../migrations/20261010000300_collaboration_moderation_subject_targets.down.sql");
const ACCEPTED_NAMES_UP: &str =
    include_str!("../migrations/20261010000400_collaboration_identity_accepted_names.up.sql");
const ACCEPTED_NAMES_DOWN: &str =
    include_str!("../migrations/20261010000400_collaboration_identity_accepted_names.down.sql");
const SIGNED_REASONS_UP: &str =
    include_str!("../migrations/20261010000500_collaboration_moderation_signed_reasons.up.sql");
const SIGNED_REASONS_DOWN: &str =
    include_str!("../migrations/20261010000500_collaboration_moderation_signed_reasons.down.sql");
const MODERATION_RETIREMENT_UP: &str = include_str!(
    "../migrations/20261026000100_collaboration_moderation_generated_retirement.up.sql"
);
const MODERATION_RETIREMENT_DOWN: &str = include_str!(
    "../migrations/20261026000100_collaboration_moderation_generated_retirement.down.sql"
);
const CHANNELS_UP: &str =
    include_str!("../migrations/20260820000700_collaboration_channels.up.sql");
const AUDIT_UP: &str = include_str!("../migrations/20260824000400_collaboration_audit.up.sql");
const MODERATION_UP: &str =
    include_str!("../migrations/20260824000500_collaboration_moderation.up.sql");
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20261009000100_collaboration_identity_profiles.down.sql"
));

#[tokio::test]
async fn moderation_retirement_preserves_generated_values_and_rejects_history_mutation()
-> anyhow::Result<()> {
    let mut database = url::Url::parse(&std::env::var("COLLAB_TEST_DATABASE_URL")?)?;
    anyhow::ensure!(
        matches!(database.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback PostgreSQL required"
    );
    database.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let database = database.to_string();
    sqlx::Postgres::create_database(&database).await?;
    let pool = sqlx::PgPool::connect(&database).await?;
    let result: anyhow::Result<()> = async {
        sqlx::raw_sql(MODERATION_RETIREMENT_DOWN).execute(&pool).await?;
        sqlx::raw_sql("CREATE TABLE public.retirement_generated (id integer PRIMARY KEY, payload text NOT NULL, is_current boolean NOT NULL, target_subject_principal_id integer GENERATED ALWAYS AS (length(payload)) STORED); INSERT INTO public.retirement_generated (id,payload,is_current) VALUES (1,'original',true); CREATE TRIGGER retirement_guard BEFORE UPDATE OR DELETE ON public.retirement_generated FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_moderation_version_retirement();").execute(&pool).await?;
        let failure = sqlx::query("UPDATE public.retirement_generated SET is_current=false WHERE id=1").execute(&pool).await.expect_err("old guard rejects a legitimate generated-column retirement");
        assert_eq!(failure.as_database_error().and_then(|error| error.code()).as_deref(), Some("23514"));
        sqlx::raw_sql(MODERATION_RETIREMENT_UP).execute(&pool).await?;
        sqlx::raw_sql("CREATE TABLE public.retirement_ordinary (id integer PRIMARY KEY, payload text NOT NULL, is_current boolean NOT NULL, target_subject_principal_id text); INSERT INTO public.retirement_ordinary VALUES (1,'original',true,'ordinary'); CREATE TRIGGER retirement_guard BEFORE UPDATE OR DELETE ON public.retirement_ordinary FOR EACH ROW EXECUTE FUNCTION public.guard_collaboration_moderation_version_retirement();").execute(&pool).await?;
        for table in ["retirement_generated", "retirement_ordinary"] {
            let invalid = format!("UPDATE public.{table} SET payload='changed',is_current=false WHERE id=1");
            let failure = sqlx::query(&invalid).execute(&pool).await.expect_err("ordinary source columns remain immutable");
            assert_eq!(failure.as_database_error().and_then(|error| error.code()).as_deref(), Some("23514"));
            let valid = format!("UPDATE public.{table} SET is_current=false WHERE id=1");
            assert_eq!(sqlx::query(&valid).execute(&pool).await?.rows_affected(), 1);
            for invalid in [format!("UPDATE public.{table} SET is_current=true WHERE id=1"), format!("UPDATE public.{table} SET payload='changed' WHERE id=1"), format!("DELETE FROM public.{table} WHERE id=1")] {
                let failure = sqlx::query(&invalid).execute(&pool).await.expect_err("retired history cannot mutate or disappear");
                assert_eq!(failure.as_database_error().and_then(|error| error.code()).as_deref(), Some("23514"));
            }
        }
        let retained: (String,i32,bool) = sqlx::query_as("SELECT payload,target_subject_principal_id,is_current FROM public.retirement_generated WHERE id=1").fetch_one(&pool).await?;
        assert_eq!(retained, ("original".into(), 8, false));
        sqlx::query("INSERT INTO public.retirement_ordinary VALUES (2,'original',true,'ordinary')").execute(&pool).await?;
        let failure=sqlx::query("UPDATE public.retirement_ordinary SET target_subject_principal_id='changed',is_current=false WHERE id=2").execute(&pool).await.expect_err("ordinary columns with a generated-column name remain protected");
        assert_eq!(failure.as_database_error().and_then(|error| error.code()).as_deref(), Some("23514"));
        Ok(())
    }.await;
    pool.close().await;
    let cleanup = sqlx::Postgres::drop_database(&database).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (result, cleanup) => Err(anyhow::anyhow!(
            "retirement guard result={result:?}; owned database cleanup={cleanup:?}"
        )),
    }
}

#[test]
fn identity_profile_repository_rejects_disconnected_connection() {
    assert!(matches!(
        IdentityProfileRepository::new(sea_orm::DatabaseConnection::Disconnected),
        Err(IdentityProfileRepositoryError::UnsupportedDatabase)
    ));
}

fn tenant(number: u128) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(
                CommunityId::from_uuid(Uuid::from_u128(number)),
                "identity-profile-fixture",
            )
            .expect("trusted fixture route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn keypair(byte: u8) -> Keypair {
    Keypair::from_secret_key(
        &Secp256k1::new(),
        &SecretKey::from_slice(&[byte; 32]).expect("fixture key"),
    )
}

fn public_key(byte: u8) -> PublicKey {
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair(byte));
    PublicKey::from_bytes(public_key.serialize())
}

fn owner_tag(agent: u8, owner: u8, conditions: &str) -> Vec<String> {
    let preimage = format!(
        "nostr:agent-auth:{}:{conditions}",
        public_key(agent).to_hex()
    );
    let digest = Sha256::digest(preimage.as_bytes());
    let mut bytes = [0; 32];
    bytes.copy_from_slice(&digest);
    let signature =
        Secp256k1::new().sign_schnorr_no_aux_rand(&Message::from_digest(bytes), &keypair(owner));
    vec![
        "auth".into(),
        public_key(owner).to_hex(),
        conditions.to_owned(),
        signature.to_string(),
    ]
}

fn signed_profile(
    tenant: &TenantContext,
    agent: u8,
    created_at: u64,
    tags: Vec<Vec<String>>,
    content: &str,
) -> VerifiedEventRecord {
    let event = CanonicalEvent::new(public_key(agent), created_at, 0, tags, content.to_owned());
    let claimed_id = event.event_id().expect("event ID");
    let signature = Secp256k1::new().sign_schnorr_no_aux_rand(
        &Message::from_digest(*claimed_id.as_bytes()),
        &keypair(agent),
    );
    VerifiedEventRecord::new(
        tenant.community_id(),
        SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        },
        EventVerificationState::Live,
        NOW * 1_000,
        TimestampPolicy::Bounded {
            now: NOW,
            max_past_seconds: 60,
            max_future_seconds: 60,
        },
    )
    .expect("actual verified signed profile")
}

fn materialized(
    outcome: IdentityProfileProjectionOutcome,
) -> anyhow::Result<PersistedIdentityProfile> {
    match outcome {
        IdentityProfileProjectionOutcome::Materialized(profile)
        | IdentityProfileProjectionOutcome::Unchanged(profile) => Ok(*profile),
        _ => anyhow::bail!("expected canonical classified profile"),
    }
}

#[test]
fn identity_profile_migration_preserves_canonical_sources_and_tenant_boundaries() {
    for requirement in [
        "PRIMARY KEY (community_id, profile_id)",
        "UNIQUE (community_id, principal_id)",
        "UNIQUE (community_id, author_public_key)",
        "first_owner_proof_event_id",
        "REFERENCES public.collaboration_events (community_id, event_id)",
        "ON DELETE RESTRICT",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS PERMISSIVE",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(UP.contains(requirement), "missing {requirement}");
    }
    assert!(!DOWN.contains("CASCADE"));
    assert!(!DOWN.contains("DROP TABLE public.collaboration_events"));
    for secret in [
        "private_key",
        "secret_key",
        "nsec",
        "mnemonic",
        "raw_challenge",
    ] {
        assert!(!UP.contains(secret), "unexpected secret column {secret}");
    }
}

#[test]
fn identity_profile_source_requires_actual_outer_signature() {
    let tenant = tenant(1);
    let verified = signed_profile(&tenant, 3, NOW, vec![owner_tag(3, 7, "kind=0")], "{}");
    let mut tampered = verified.signed_event().clone();
    tampered.event.content = r#"{"name":"tampered"}"#.into();
    assert!(
        VerifiedEventRecord::new(
            tenant.community_id(),
            tampered,
            EventVerificationState::Live,
            NOW * 1_000,
            TimestampPolicy::Bounded {
                now: NOW,
                max_past_seconds: 60,
                max_future_seconds: 60
            }
        )
        .is_err(),
        "an owner tag must not substitute for the actual profile signature"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_profile_repository_signed_provenance_and_transactions() -> anyhow::Result<()> {
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL").map_err(|_| {
        anyhow::anyhow!("COLLAB_TEST_DATABASE_URL is required for live profile qualification")
    })?;
    let mut url = url::Url::parse(&base_url)?;
    anyhow::ensure!(
        matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "profile qualification requires owned loopback PostgreSQL"
    );
    url.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let url = url.to_string();
    sqlx::Postgres::create_database(&url).await?;
    let result = exercise_profiles(&url).await;
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(anyhow::anyhow!(
            "owned profile database cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "profile qualification failed: {error:#}; owned database cleanup failed: {cleanup}"
        )),
    }
}

async fn exercise_profiles(url: &str) -> anyhow::Result<()> {
    let pool = sqlx::PgPool::connect(url).await?;
    let result = async {
        for migration in [
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/migrations/20260820000100_collaboration_events.up.sql")),
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/migrations/20260820000200_collaboration_event_heads.up.sql")), UP, REGISTRATION_UP, OBSERVATION_UP, CHANNELS_UP, AUDIT_UP, MODERATION_UP, MODERATION_SUBJECT_UP, ACCEPTED_NAMES_UP, SIGNED_REASONS_UP, MODERATION_RETIREMENT_UP,
        ] { sqlx::raw_sql(migration).execute(&pool).await?; }
        let repository = Arc::new(IdentityProfileRepository::new(sea_orm::Database::connect(url).await?)?);
        let events = EventRepository::new(sea_orm::Database::connect(url).await?)?;
        let result = exercise_registered_profiles(&repository, &events, &pool).await;
        events.into_connection().close().await?;
        Arc::try_unwrap(repository).map_err(|_| anyhow::anyhow!("profile fixture leaked a repository reference"))?.into_connection().close().await?;
        result?;
        let mut rollback = pool.begin().await?;
        sqlx::raw_sql(SIGNED_REASONS_DOWN).execute(&mut *rollback).await?;
        sqlx::raw_sql(ACCEPTED_NAMES_DOWN).execute(&mut *rollback).await?;
        sqlx::raw_sql(MODERATION_SUBJECT_DOWN).execute(&mut *rollback).await?;
        sqlx::raw_sql(OBSERVATION_DOWN).execute(&mut *rollback).await?;
        sqlx::raw_sql(REGISTRATION_DOWN).execute(&mut *rollback).await?;
        rollback.commit().await?;
        sqlx::raw_sql(DOWN).execute(&pool).await?;
        let table: Option<String> = sqlx::query_scalar("SELECT to_regclass('public.collaboration_identity_profiles')::text").fetch_one(&pool).await?;
        anyhow::ensure!(table.is_none(), "profile migration must reverse its own table");
        let event_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events").fetch_one(&pool).await?;
        anyhow::ensure!(event_count > 0, "profile rollback must retain original canonical signed evidence");
        eprintln!("LIVE_IDENTITY_PROFILE_REPOSITORY_VERIFIED: signed provenance, stable IDs, owner races, renewal, tenant denial, corruption, reversible projection cleanup");
        Ok(())
    }.await;
    pool.close().await;
    result
}

async fn exercise_registered_profiles(
    repository: &Arc<IdentityProfileRepository>,
    events: &EventRepository,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    let tenant_a = tenant(1);
    let tenant_b = tenant(2);
    let agent_key = NostrPublicKey::from_bytes(*public_key(3).as_bytes());
    let unknown = signed_profile(
        &tenant_a,
        3,
        NOW,
        vec![vec!["p".into(), public_key(7).to_hex()]],
        r#"{"name":"Unclassified","nip05":"claimed@example.invalid"}"#,
    );
    anyhow::ensure!(
        repository
            .project_profile_event(&tenant_a, &unknown)
            .await?
            == IdentityProfileProjectionOutcome::Unclassified,
        "kind0 metadata or p tags must not classify Human or establish owner"
    );
    anyhow::ensure!(
        repository
            .current_by_public_key(&tenant_a, agent_key)
            .await?
            .is_none(),
        "unclassified profile must allocate no canonical identity"
    );
    anyhow::ensure!(
        events
            .exact(&tenant_a, unknown.signed_event().claimed_id)
            .await?
            .is_some(),
        "unknown original signed source must remain canonical"
    );

    let proof = signed_profile(
        &tenant_a,
        3,
        NOW + 1,
        vec![owner_tag(3, 7, "kind=0")],
        r#"{"display_name":"Agent","picture":"https://example.invalid/avatar","about":"old"}"#,
    );
    let first = materialized(repository.project_profile_event(&tenant_a, &proof).await?)?;
    anyhow::ensure!(
        !first.principal_id().as_uuid().is_nil()
            && !first.profile().profile_id().as_uuid().is_nil(),
        "canonical owner must allocate non-nil stable IDs"
    );
    anyhow::ensure!(
        first.first_owner_proof_event_id().as_bytes() == proof.signed_event().claimed_id.as_bytes(),
        "first owner must retain actual signed proof identity"
    );
    let retry = materialized(repository.project_profile_event(&tenant_a, &proof).await?)?;
    anyhow::ensure!(
        retry == first,
        "exact retry must preserve UUIDs, profile version and evidence"
    );

    let renewal = signed_profile(
        &tenant_a,
        3,
        NOW + 2,
        vec![owner_tag(3, 7, "kind=0&created_at>1")],
        r#"{"display_name":"Renewed"}"#,
    );
    let renewed = materialized(
        repository
            .project_profile_event(&tenant_a, &renewal)
            .await?,
    )?;
    anyhow::ensure!(
        renewed.principal_id() == first.principal_id()
            && renewed.profile().profile_id() == first.profile().profile_id()
            && renewed.first_owner_proof_event_id() == first.first_owner_proof_event_id()
            && renewed.profile().version().get() == 2,
        "same-owner renewal must retain canonical association and first owner"
    );
    anyhow::ensure!(
        renewed
            .profile()
            .fields()
            .metadata
            .as_ref()
            .is_some_and(
                |metadata| metadata.value.avatar_url.is_none() && metadata.value.about.is_none()
            ),
        "kind0 absolute-state omission must clear old metadata"
    );

    let updated = signed_profile(
        &tenant_a,
        3,
        NOW + 3,
        Vec::new(),
        r#"{"display_name":"","name":"fallback","nip05":"self-claim@example.invalid"}"#,
    );
    let updated = materialized(
        repository
            .project_profile_event(&tenant_a, &updated)
            .await?,
    )?;
    anyhow::ensure!(
        updated.profile().version().get() == 3 && updated.principal_id() == first.principal_id(),
        "metadata-only update must retain classification and owner association"
    );
    anyhow::ensure!(
        updated
            .profile()
            .fields()
            .metadata
            .as_ref()
            .is_some_and(|metadata| metadata.value.display_name.is_none()),
        "explicit empty display_name must clear, not fall back to name"
    );
    let ProfileKind::Agent(agent) = updated.profile().kind() else {
        anyhow::bail!("registered agent must not downgrade to Human");
    };
    anyhow::ensure!(
        agent.claimed_owner == Some(NostrPublicKey::from_bytes(*public_key(7).as_bytes())),
        "unsigned metadata cannot replace first owner"
    );
    anyhow::ensure!(
        repository
            .current_by_public_key(&tenant_a, agent_key)
            .await?
            == Some(updated.clone()),
        "fresh hydration must verify retained originals and association"
    );

    let different_owner =
        signed_profile(&tenant_a, 3, NOW + 4, vec![owner_tag(3, 8, "kind=0")], "{}");
    anyhow::ensure!(
        matches!(
            repository
                .project_profile_event(&tenant_a, &different_owner)
                .await,
            Err(IdentityProfileRepositoryError::IdentityConflict)
        ),
        "different valid owner must not replace first owner"
    );
    anyhow::ensure!(
        events
            .exact(&tenant_a, different_owner.signed_event().claimed_id)
            .await?
            .is_none(),
        "rejected association must not leave an event/head side effect"
    );
    let stale = signed_profile(
        &tenant_a,
        3,
        NOW + 1,
        vec![owner_tag(3, 7, "kind=0")],
        r#"{"name":"stale"}"#,
    );
    anyhow::ensure!(
        repository.project_profile_event(&tenant_a, &stale).await?
            == IdentityProfileProjectionOutcome::Stale,
        "source replacement ordering must stay with EventRepository"
    );

    let mut bad_tag = owner_tag(4, 7, "kind=0");
    if let Some(signature) = bad_tag.get_mut(3) {
        *signature = "00".repeat(64);
    }
    for tags in [
        vec![bad_tag],
        vec![owner_tag(4, 7, "kind=1")],
        vec![owner_tag(4, 7, "kind=0"), owner_tag(4, 7, "kind=0")],
    ] {
        let bad = signed_profile(&tenant_a, 4, NOW, tags, "{}");
        anyhow::ensure!(
            matches!(
                repository.project_profile_event(&tenant_a, &bad).await,
                Err(IdentityProfileRepositoryError::InvalidRecord)
            ),
            "actual outer signature must not substitute for invalid owner proof"
        );
        anyhow::ensure!(
            events
                .exact(&tenant_a, bad.signed_event().claimed_id)
                .await?
                .is_none(),
            "invalid owner proof must not persist"
        );
    }
    anyhow::ensure!(
        matches!(
            repository.project_profile_event(&tenant_b, &proof).await,
            Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
        ),
        "trusted tenant must precede profile data access"
    );
    let another_tenant = signed_profile(&tenant_b, 3, NOW, vec![owner_tag(3, 8, "kind=0")], "{}");
    let another = materialized(
        repository
            .project_profile_event(&tenant_b, &another_tenant)
            .await?,
    )?;
    anyhow::ensure!(
        another.principal_id() != first.principal_id()
            && another.profile().profile_id() != first.profile().profile_id(),
        "same key in another community has distinct canonical association"
    );

    exercise_profile_races(repository, &tenant_a).await?;
    exercise_rls(pool, &tenant_a, &tenant_b, &first).await?;
    let owner_change = sqlx::query("UPDATE public.collaboration_identity_profiles SET owner_public_key=$3, aggregate_version=aggregate_version+1 WHERE community_id=$1 AND author_public_key=$2")
        .bind(tenant_a.community_id().as_uuid()).bind(agent_key.as_bytes().to_vec()).bind(public_key(8).as_bytes().to_vec()).execute(pool).await;
    anyhow::ensure!(
        owner_change
            .as_ref()
            .err()
            .and_then(sqlx::Error::as_database_error)
            .and_then(|error| error.code())
            .as_deref()
            == Some("23514"),
        "database guard must retain the first verified owner even for direct writes"
    );
    anyhow::ensure!(
        repository
            .current_by_public_key(&tenant_a, agent_key)
            .await?
            == Some(updated),
        "rejected owner rewrite must preserve current identity"
    );
    let source_delete = sqlx::query(
        "DELETE FROM public.collaboration_events WHERE community_id=$1 AND event_id=$2",
    )
    .bind(tenant_a.community_id().as_uuid())
    .bind(first.first_owner_proof_event_id().as_bytes().to_vec())
    .execute(pool)
    .await;
    anyhow::ensure!(
        source_delete
            .as_ref()
            .err()
            .and_then(sqlx::Error::as_database_error)
            .and_then(|error| error.code())
            .as_deref()
            == Some("23503"),
        "retained proof foreign key must protect original signed evidence"
    );
    sqlx::query("UPDATE public.collaboration_identity_profiles SET profile_hash=$3, aggregate_version=aggregate_version+1 WHERE community_id=$1 AND author_public_key=$2")
        .bind(tenant_a.community_id().as_uuid()).bind(agent_key.as_bytes().to_vec()).bind(vec![0_u8;32]).execute(pool).await?;
    anyhow::ensure!(
        matches!(
            repository.current_by_public_key(&tenant_a, agent_key).await,
            Err(IdentityProfileRepositoryError::InvalidRecord)
        ),
        "corrupt projection must fail closed"
    );
    Ok(())
}

async fn exercise_profile_races(
    repository: &Arc<IdentityProfileRepository>,
    tenant: &TenantContext,
) -> anyhow::Result<()> {
    let shared = signed_profile(tenant, 5, NOW, vec![owner_tag(5, 7, "kind=0")], "{}");
    let barrier = Arc::new(Barrier::new(2));
    let first = {
        let barrier = barrier.clone();
        let shared = &shared;
        async move {
            barrier.wait().await;
            repository.project_profile_event(tenant, shared).await
        }
    };
    let second = {
        let barrier = barrier.clone();
        let shared = &shared;
        async move {
            barrier.wait().await;
            repository.project_profile_event(tenant, shared).await
        }
    };
    let (first, second) = tokio::join!(first, second);
    let first = materialized(first?)?;
    let second = materialized(second?)?;
    anyhow::ensure!(
        first == second && first.profile().version().get() == 1,
        "concurrent exact registration must allocate one stable association"
    );

    let first_source = signed_profile(tenant, 6, NOW, vec![owner_tag(6, 7, "kind=0")], "{}");
    let second_source = signed_profile(tenant, 6, NOW, vec![owner_tag(6, 8, "kind=0")], "{}");
    let barrier = Arc::new(Barrier::new(2));
    let first = {
        let barrier = barrier.clone();
        let first_source = &first_source;
        async move {
            barrier.wait().await;
            repository.project_profile_event(tenant, first_source).await
        }
    };
    let second = {
        let barrier = barrier.clone();
        let second_source = &second_source;
        async move {
            barrier.wait().await;
            repository
                .project_profile_event(tenant, second_source)
                .await
        }
    };
    let (first, second) = tokio::join!(first, second);
    anyhow::ensure!(
        usize::from(first.is_ok()) + usize::from(second.is_ok()) == 1,
        "two valid owner proofs must not concurrently establish different owners"
    );
    let rejected = match (first, second) {
        (Ok(_), Err(error)) | (Err(error), Ok(_)) => error,
        _ => anyhow::bail!("owner race inconsistent"),
    };
    anyhow::ensure!(
        matches!(rejected, IdentityProfileRepositoryError::IdentityConflict),
        "first-owner race must reject through exact canonical owner conflict"
    );
    Ok(())
}

async fn exercise_rls(
    pool: &sqlx::PgPool,
    tenant_a: &TenantContext,
    tenant_b: &TenantContext,
    first: &PersistedIdentityProfile,
) -> anyhow::Result<()> {
    let mut transaction = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE pg_read_all_data")
        .execute(&mut *transaction)
        .await?;
    let bypass: bool =
        sqlx::query_scalar("SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user")
            .fetch_one(&mut *transaction)
            .await?;
    anyhow::ensure!(!bypass, "RLS fixture must actually run without BYPASSRLS");
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant_a.community_id().to_string())
        .execute(&mut *transaction)
        .await?;
    let own_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1",
    )
    .bind(tenant_a.community_id().as_uuid())
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(
        own_count > 0,
        "nonbypass owner tenant must read real persisted profiles"
    );
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant_b.community_id().to_string())
        .execute(&mut *transaction)
        .await?;
    let foreign_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1",
    )
    .bind(tenant_a.community_id().as_uuid())
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(
        foreign_count == 0,
        "nonbypass foreign tenant must see no profile rows"
    );
    transaction.rollback().await?;

    let mut transaction = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE pg_write_all_data")
        .execute(&mut *transaction)
        .await?;
    let bypass: bool =
        sqlx::query_scalar("SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user")
            .fetch_one(&mut *transaction)
            .await?;
    anyhow::ensure!(
        !bypass,
        "write-denial fixture must actually run without BYPASSRLS"
    );
    let can_insert: bool = sqlx::query_scalar("SELECT has_table_privilege(current_user, 'public.collaboration_identity_profiles', 'INSERT')")
        .fetch_one(&mut *transaction).await?;
    anyhow::ensure!(
        can_insert,
        "foreign-write negative must not be a missing table privilege"
    );
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant_b.community_id().to_string())
        .execute(&mut *transaction)
        .await?;
    let rejected = sqlx::query("INSERT INTO public.collaboration_identity_profiles (community_id, profile_id, principal_id, author_public_key, profile_kind, owner_public_key, first_owner_proof_event_id, source_event_id, aggregate_version, profile_payload, profile_hash) VALUES ($1,$2,$3,$4,'agent',$5,$6,$6,1,$7,$8)")
        .bind(tenant_a.community_id().as_uuid()).bind(Uuid::new_v4()).bind(Uuid::new_v4()).bind(public_key(9).as_bytes().to_vec())
        .bind(public_key(7).as_bytes().to_vec()).bind(first.first_owner_proof_event_id().as_bytes().to_vec())
        .bind(serde_json::to_value(first.profile())?).bind(vec![1_u8;32]).execute(&mut *transaction).await;
    let denied = rejected
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .is_some_and(|error| {
            error.code().as_deref() == Some("42501")
                && error.message().contains("row-level security")
        });
    transaction.rollback().await?;
    anyhow::ensure!(
        denied,
        "nonbypass foreign tenant must be denied a real profile INSERT by RLS"
    );
    Ok(())
}

#[derive(Clone)]
struct PersistedResolver(Arc<IdentityProfileRepository>);

#[async_trait::async_trait]
impl collab::nostr::auth::NostrPrincipalResolver for PersistedResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &collab::nostr::auth::VerifiedNip42Identity,
    ) -> Result<
        collaboration_domain::AuthenticatedPrincipal,
        collab::nostr::auth::NostrPrincipalResolutionError,
    > {
        collab::nostr::auth::NostrPrincipalResolver::resolve(self.0.as_ref(), tenant, identity)
            .await
    }
}

#[derive(Clone, Default)]
struct RegistrationReplay(
    Arc<std::sync::Mutex<std::collections::BTreeSet<(CommunityId, nostr_compat::EventId)>>>,
);

#[async_trait::async_trait]
impl collab::nostr::auth::NostrAuthReplayStore for RegistrationReplay {
    async fn claim(
        &self,
        community: CommunityId,
        event: nostr_compat::EventId,
        _expires: u64,
    ) -> Result<
        collab::nostr::auth::ReplayClaim,
        collab::nostr::auth::NostrAuthenticationInfrastructureError,
    > {
        let inserted = self
            .0
            .lock()
            .map_err(|_| collab::nostr::auth::NostrAuthenticationInfrastructureError::Unavailable)?
            .insert((community, event));
        Ok(if inserted {
            collab::nostr::auth::ReplayClaim::Claimed
        } else {
            collab::nostr::auth::ReplayClaim::Replay
        })
    }
}

fn signed_registration(
    agent: u8,
    owner: Option<u8>,
    timestamp: u64,
    challenge_byte: u8,
) -> SignedEvent {
    signed_registration_with_tags(
        agent,
        owner
            .map(|owner| owner_tag(agent, owner, "kind=0"))
            .into_iter()
            .collect(),
        timestamp,
        challenge_byte,
    )
}

fn signed_registration_with_tags(
    agent: u8,
    owner_tags: Vec<Vec<String>>,
    timestamp: u64,
    challenge_byte: u8,
) -> SignedEvent {
    let mut tags = vec![
        vec!["relay".into(), "wss://relay.example.test".into()],
        vec!["challenge".into(), hex::encode([challenge_byte; 32])],
    ];
    tags.extend(owner_tags);
    let event = CanonicalEvent::new(
        public_key(agent),
        timestamp,
        nostr_compat::generated_kinds::KIND_AUTH as u16,
        tags,
        String::new(),
    );
    let claimed_id = event.event_id().expect("signed registration event ID");
    let signature = Secp256k1::new().sign_schnorr_no_aux_rand(
        &Message::from_digest(*claimed_id.as_bytes()),
        &keypair(agent),
    );
    SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    }
}

fn invalid_optional_owner_tags(agent: u8, owner: u8) -> Vec<Vec<Vec<String>>> {
    let mut invalid_signature = owner_tag(agent, owner, "");
    if let Some(signature) = invalid_signature.get_mut(3) {
        *signature = "00".repeat(64);
    }
    let duplicate = owner_tag(agent, owner, "");
    vec![
        vec![invalid_signature],
        vec![vec!["auth".into()]],
        vec![owner_tag(agent, agent, "")],
        vec![duplicate.clone(), duplicate],
        vec![owner_tag(agent, owner, &format!("created_at<{}", NOW - 1))],
    ]
}

async fn authenticate_registration(
    repository: &Arc<IdentityProfileRepository>,
    tenant: &TenantContext,
    source: &SignedEvent,
) -> anyhow::Result<collaboration_domain::AuthenticatedPrincipal> {
    let challenge = source
        .event
        .tags
        .iter()
        .find(|tag| tag.first().map(String::as_str) == Some("challenge"))
        .and_then(|tag| tag.get(1))
        .ok_or_else(|| anyhow::anyhow!("missing fixture challenge"))?;
    let mut authenticator = collab::nostr::auth::NostrConnectionAuthenticator::new(
        tenant.clone(),
        "wss://relay.example.test",
        collab::nostr::auth::NostrAuthChallenge::parse(challenge.clone())?,
        source.event.created_at,
        RegistrationReplay::default(),
        PersistedResolver(repository.clone()),
    )?;
    Ok(authenticator
        .authenticate(source, source.event.created_at)
        .await?
        .clone())
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_registration_actual_auth_single_connection_and_owner_races() -> anyhow::Result<()>
{
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL").map_err(|_| {
        anyhow::anyhow!(
            "COLLAB_TEST_DATABASE_URL is required for actual registration qualification"
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
    let result = exercise_identity_registration(&url).await;
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(anyhow::anyhow!(
            "owned registration database cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "registration qualification failed: {error:#}; owned database cleanup failed: {cleanup}"
        )),
    }
}

async fn exercise_identity_registration(url: &str) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(url)
        .await?;
    let result = async {
        for migration in [
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/migrations/20260820000100_collaboration_events.up.sql"
            )),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/migrations/20260820000200_collaboration_event_heads.up.sql"
            )),
            UP,
            REGISTRATION_UP,
            OBSERVATION_UP,
            CHANNELS_UP,
            AUDIT_UP,
            MODERATION_UP,
            MODERATION_SUBJECT_UP,
            ACCEPTED_NAMES_UP,
            SIGNED_REASONS_UP,
            MODERATION_RETIREMENT_UP,
        ] {
            sqlx::raw_sql(migration).execute(&pool).await?;
        }
        let repository = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let bounded = tokio::time::timeout(
            std::time::Duration::from_secs(20),
            exercise_single_connection_registration(&repository, &pool),
        )
        .await;
        Arc::try_unwrap(repository)
            .map_err(|_| anyhow::anyhow!("single-connection resolver fixture leaked a reference"))?
            .into_connection()
            .close()
            .await?;
        bounded.map_err(|_| {
            anyhow::anyhow!("single-connection registration/head hydration deadlocked")
        })??;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    pool.close().await;
    result?;
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(6)
        .connect(url)
        .await?;
    let result = async {
        let repository = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let result = exercise_registration_races(&repository, &pool).await;
        Arc::try_unwrap(repository)
            .map_err(|_| anyhow::anyhow!("race resolver fixture leaked a reference"))?
            .into_connection()
            .close()
            .await?;
        result?;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    pool.close().await;
    result?;
    eprintln!(
        "LIVE_IDENTITY_REGISTRATION_VERIFIED: actual signed AUTH, single-connection canonical head hydration, stable unclassified association, first owner, exact-key fallback, under-lock races, RLS and guarded rollback; transient replay fixture only, no current-member/live service admission"
    );
    Ok(())
}

async fn exercise_single_connection_registration(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collaboration_domain::AuthenticatedPrincipalKind;
    let tenant_a = tenant(31);
    let tenant_b = tenant(32);
    let key = NostrPublicKey::from_bytes(*public_key(21).as_bytes());
    let profile = signed_profile(
        &tenant_a,
        21,
        NOW,
        Vec::new(),
        r#"{"name":"before-auth","picture":"https://example.test/avatar"}"#,
    );
    anyhow::ensure!(
        repository
            .project_profile_event(&tenant_a, &profile)
            .await?
            == IdentityProfileProjectionOutcome::Unclassified,
        "unknown kind0 must not allocate or infer Human"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&tenant_a, key)
            .await?
            .is_none(),
        "unknown profile must not establish a principal"
    );
    let source = signed_registration(21, None, NOW, 31);
    let plain = authenticate_registration(repository, &tenant_a, &source).await?;
    anyhow::ensure!(
        matches!(plain.kind(), AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } if *public_key == key),
        "plain AUTH proves exact key only"
    );
    let registration = repository
        .current_registration_by_public_key(&tenant_a, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing registration"))?;
    anyhow::ensure!(
        registration.profile().is_none()
            && repository
                .current_by_public_key(&tenant_a, key)
                .await?
                .is_none(),
        "plain AUTH remains unclassified"
    );
    anyhow::ensure!(
        registration.principal_id() == plain.principal_id() && registration.version().get() == 1,
        "stable canonical association first write"
    );
    let retry = authenticate_registration(repository, &tenant_a, &source).await?;
    anyhow::ensure!(
        retry.principal_id() == plain.principal_id()
            && repository
                .current_registration_by_public_key(&tenant_a, key)
                .await?
                == Some(registration.clone()),
        "new authenticated session for exact signed key must reuse ID without version churn"
    );
    let membership_before: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships")
            .fetch_one(pool)
            .await?;
    for (index, tags) in invalid_optional_owner_tags(21, 22).into_iter().enumerate() {
        let invalid = signed_registration_with_tags(21, tags, NOW, 140 + index as u8);
        let fallback = authenticate_registration(repository, &tenant_a, &invalid).await
            .map_err(|error| anyhow::anyhow!("invalid optional owner evidence rejected actual canonical key registration: {error:#}"))?;
        anyhow::ensure!(
            matches!(fallback.kind(), AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } if *public_key == key)
                && fallback.principal_id() == plain.principal_id()
                && fallback.scopes() == plain.scopes(),
            "invalid owner evidence changed key identity or rights"
        );
        anyhow::ensure!(
            repository
                .current_registration_by_public_key(&tenant_a, key)
                .await?
                == Some(registration.clone()),
            "optional OA fallback changed canonical profile, version, observation or first owner"
        );
        anyhow::ensure!(
            repository
                .current_subject_by_public_key(
                    &tenant_a,
                    NostrPublicKey::from_bytes(*public_key(22).as_bytes())
                )
                .await?
                .is_none(),
            "invalid optional OA fabricated an observed owner"
        );
        let membership_after: i64 =
            sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships")
                .fetch_one(pool)
                .await?;
        anyhow::ensure!(
            membership_after == membership_before,
            "key fallback fabricated membership"
        );
    }
    for (index, signer) in (50..55_u8).enumerate() {
        let tags = invalid_optional_owner_tags(signer, 22)
            .into_iter()
            .nth(index)
            .ok_or_else(|| anyhow::anyhow!("missing owner fallback vector"))?;
        let invalid = signed_registration_with_tags(signer, tags, NOW, 160 + index as u8);
        let fallback = authenticate_registration(repository, &tenant_a, &invalid).await
            .map_err(|error| anyhow::anyhow!("invalid optional owner evidence rejected first actual key registration: {error:#}"))?;
        let expected_key = NostrPublicKey::from_bytes(*public_key(signer).as_bytes());
        let current = repository
            .current_registration_by_public_key(&tenant_a, expected_key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("actual key fallback failed to persist registration"))?;
        anyhow::ensure!(
            matches!(fallback.kind(), AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } if *public_key == expected_key)
                && current.principal_id() == fallback.principal_id()
                && current.version().get() == 1
                && current.profile().is_none()
                && fallback.scopes() == plain.scopes(),
            "first invalid-OA fallback fabricated classified profile or rights"
        );
        let subject = repository
            .current_subject_by_public_key(&tenant_a, expected_key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing actual fallback subject"))?;
        anyhow::ensure!(subject.status() == collab::identity::profile_repository::IdentitySubjectStatus::RegisteredUnclassified,
            "invalid optional OA inferred Human or Agent");
        let membership: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2")
            .bind(tenant_a.community_id().as_uuid()).bind(fallback.principal_id().as_uuid()).fetch_one(pool).await?;
        anyhow::ensure!(
            membership == 0,
            "first optional-OA fallback fabricated membership"
        );
    }
    let owner = signed_registration(21, Some(22), NOW, 32);
    let agent = authenticate_registration(repository, &tenant_a, &owner).await?;
    anyhow::ensure!(
        matches!(agent.kind(), AuthenticatedPrincipalKind::OwnerAttestedAgent { owner_public_key, proof_event_id, .. }
        if *owner_public_key == NostrPublicKey::from_bytes(*public_key(22).as_bytes()) && *proof_event_id.as_bytes() == *owner.claimed_id.as_bytes()),
        "actual OA registers virtual agent"
    );
    anyhow::ensure!(
        agent.principal_id() == plain.principal_id(),
        "owner backfill must retain canonical stable principal"
    );
    let classified = repository
        .current_by_public_key(&tenant_a, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing typed agent"))?;
    anyhow::ensure!(
        classified.profile().profile_id() == registration.profile_id()
            && classified
                .profile()
                .fields()
                .metadata
                .as_ref()
                .is_some_and(|metadata| metadata.value.name.as_deref() == Some("before-auth")),
        "first AUTH owner projection hydrates retained real kind0 head inside current transaction"
    );
    let payload: serde_json::Value = sqlx::query_scalar("SELECT first_owner_auth_proof FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key=$2")
        .bind(tenant_a.community_id().as_uuid()).bind(key.as_bytes().to_vec()).fetch_one(pool).await?;
    anyhow::ensure!(
        payload["exact_tag"] == serde_json::to_value(owner_tag(21, 22, "kind=0"))?,
        "retain exact original OA signature and conditions"
    );
    let durable_auth_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events WHERE kind=$1")
            .bind(nostr_compat::generated_kinds::KIND_AUTH as i32)
            .fetch_one(pool)
            .await?;
    anyhow::ensure!(
        durable_auth_count == 0,
        "transient AUTH originals must never enter durable event repository"
    );
    let row_before = repository
        .current_registration_by_public_key(&tenant_a, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing classified registration"))?;
    let exact_retry = authenticate_registration(repository, &tenant_a, &owner).await?;
    anyhow::ensure!(
        exact_retry == agent
            && repository
                .current_registration_by_public_key(&tenant_a, key)
                .await?
                == Some(row_before.clone()),
        "owner exact retry must be idempotent"
    );
    let different = signed_registration(21, Some(23), NOW, 33);
    let fallback = authenticate_registration(repository, &tenant_a, &different).await?;
    anyhow::ensure!(
        matches!(fallback.kind(), AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } if *public_key == key)
            && fallback.principal_id() == plain.principal_id(),
        "conflicting owner backfill keeps successful exact-key AUTH without delegation"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&tenant_a, key)
            .await?
            == Some(row_before.clone()),
        "different owner must not change first owner or projection"
    );
    let renewed = signed_registration(21, Some(22), NOW + 1, 34);
    let renewed_principal = authenticate_registration(repository, &tenant_a, &renewed).await?;
    anyhow::ensure!(
        renewed_principal.principal_id() == plain.principal_id(),
        "same owner renewal retains ID"
    );
    let first_after: serde_json::Value = sqlx::query_scalar("SELECT first_owner_auth_proof FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key=$2")
        .bind(tenant_a.community_id().as_uuid()).bind(key.as_bytes().to_vec()).fetch_one(pool).await?;
    anyhow::ensure!(
        first_after == payload,
        "renewal preserves original first-owner proof"
    );
    let foreign = authenticate_registration(repository, &tenant_b, &source).await?;
    anyhow::ensure!(
        foreign.principal_id() != plain.principal_id(),
        "same key has tenant-bound canonical associations"
    );
    let reloaded = IdentityProfileRepository::new(sea_orm::DatabaseConnection::from(pool.clone()))?;
    anyhow::ensure!(
        reloaded
            .current_registration_by_public_key(&tenant_a, key)
            .await?
            .is_some_and(|record| record.principal_id() == plain.principal_id()),
        "fresh repository hydration preserves actual association and proofs"
    );
    drop(reloaded);
    let mut transaction = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE pg_read_all_data")
        .execute(&mut *transaction)
        .await?;
    let bypass: bool =
        sqlx::query_scalar("SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user")
            .fetch_one(&mut *transaction)
            .await?;
    anyhow::ensure!(!bypass, "registration RLS fixture must not bypass RLS");
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant_b.community_id().to_string())
        .execute(&mut *transaction)
        .await?;
    let count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1",
    )
    .bind(tenant_a.community_id().as_uuid())
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(
        count == 0,
        "foreign registration association and proof hidden by RLS"
    );
    transaction.rollback().await?;
    let mut foreign_write = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE pg_write_all_data")
        .execute(&mut *foreign_write)
        .await?;
    let can_insert: bool = sqlx::query_scalar("SELECT has_table_privilege(current_user, 'public.collaboration_identity_profiles', 'INSERT')")
        .fetch_one(&mut *foreign_write).await?;
    anyhow::ensure!(
        can_insert,
        "unclassified foreign-write negative requires actual INSERT privilege"
    );
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant_b.community_id().to_string())
        .execute(&mut *foreign_write)
        .await?;
    let rejected_insert = sqlx::query("INSERT INTO public.collaboration_identity_profiles (community_id, profile_id, principal_id, author_public_key, aggregate_version, profile_hash, registration_source_event_id, registration_verified_at_seconds, projection_hash_version) VALUES ($1,$2,$3,$4,1,$5,$6,$7,2)")
        .bind(tenant_a.community_id().as_uuid()).bind(Uuid::new_v4()).bind(Uuid::new_v4())
        .bind(public_key(29).as_bytes().to_vec()).bind(vec![1_u8;32]).bind(source.claimed_id.as_bytes().to_vec()).bind(NOW as i64)
        .execute(&mut *foreign_write).await;
    let foreign_denied = rejected_insert
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .is_some_and(|error| {
            error.code().as_deref() == Some("42501")
                && error.message().contains("row-level security")
        });
    foreign_write.rollback().await?;
    anyhow::ensure!(
        foreign_denied,
        "foreign unclassified association INSERT must be denied specifically by RLS"
    );
    let mut limited_rollback = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE pg_write_all_data")
        .execute(&mut *limited_rollback)
        .await?;
    let rejected_limited = sqlx::raw_sql(REGISTRATION_DOWN)
        .execute(&mut *limited_rollback)
        .await;
    let limited_denied = rejected_limited
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .is_some_and(|error| {
            error.code().as_deref() == Some("42501")
                && error.message().contains("inspect every tenant")
        });
    limited_rollback.rollback().await?;
    anyhow::ensure!(
        limited_denied,
        "rollback must reject tenant-filtered evidence census before any DDL"
    );
    let mut rollback = pool.begin().await?;
    let rejected = sqlx::raw_sql(REGISTRATION_DOWN)
        .execute(&mut *rollback)
        .await;
    let refused = rejected
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .is_some_and(|error| {
            error.code().as_deref() == Some("23514")
                && error
                    .message()
                    .contains("discard retained identity or owner evidence")
        });
    rollback.rollback().await?;
    anyhow::ensure!(
        refused,
        "registration down must refuse loss of unclassified IDs and detached proof evidence"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&tenant_a, key)
            .await?
            .is_some_and(|record| record.principal_id() == plain.principal_id()),
        "failed rollback preserves retained proofs"
    );
    Ok(())
}

async fn exercise_registration_races(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collaboration_domain::AuthenticatedPrincipalKind;
    let tenant = tenant(41);
    let key = NostrPublicKey::from_bytes(*public_key(24).as_bytes());
    let first = signed_registration(24, Some(22), NOW, 41);
    let second = signed_registration(24, Some(23), NOW, 42);
    let (first_result, second_result) = tokio::join!(
        authenticate_registration(repository, &tenant, &first),
        authenticate_registration(repository, &tenant, &second)
    );
    let first_result = first_result?;
    let second_result = second_result?;
    anyhow::ensure!(
        first_result.principal_id() == second_result.principal_id(),
        "concurrent registration must allocate one canonical principal"
    );
    anyhow::ensure!(
        usize::from(matches!(
            first_result.kind(),
            AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
        )) + usize::from(matches!(
            second_result.kind(),
            AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
        )) == 1,
        "first-owner race must admit one delegation and preserve the losing key authentication"
    );
    let current = repository
        .current_by_public_key(&tenant, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing raced agent"))?;
    let ProfileKind::Agent(agent) = current.profile().kind() else {
        anyhow::bail!("raced registration not an agent");
    };
    let owner =
        if agent.claimed_owner == Some(NostrPublicKey::from_bytes(*public_key(22).as_bytes())) {
            22
        } else {
            23
        };
    let race_key = NostrPublicKey::from_bytes(*public_key(25).as_bytes());
    let plain = signed_registration(25, None, NOW, 44);
    let base = authenticate_registration(repository, &tenant, &plain).await?;
    let initial = signed_profile(&tenant, 25, NOW, Vec::new(), r#"{"name":"before-lock"}"#);
    anyhow::ensure!(
        repository.project_profile_event(&tenant, &initial).await?
            == IdentityProfileProjectionOutcome::Unclassified,
        "plain registration plus metadata must remain unclassified before actual owner evidence"
    );
    let mut lock = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(format!(
            "collaboration-profile:{}:{}",
            tenant.community_id(),
            public_key(25).to_hex()
        ))
        .execute(&mut *lock)
        .await?;
    let update = signed_profile(
        &tenant,
        25,
        NOW + 1,
        Vec::new(),
        r#"{"name":"under-lock-head"}"#,
    );
    let receipt = signed_registration(25, Some(22), NOW + 1, 43);
    let (profile_result, auth_result, reader_result, release_result) = tokio::join!(
        async {
            tokio::time::timeout(
                std::time::Duration::from_secs(20),
                repository.project_profile_event(&tenant, &update),
            )
            .await
            .map_err(|_| anyhow::anyhow!("under-lock profile update deadlocked"))?
            .map_err(anyhow::Error::from)
        },
        async {
            tokio::time::timeout(
                std::time::Duration::from_secs(20),
                authenticate_registration(repository, &tenant, &receipt),
            )
            .await
            .map_err(|_| anyhow::anyhow!("under-lock AUTH update deadlocked"))?
        },
        async {
            tokio::time::timeout(
                std::time::Duration::from_secs(20),
                repository.current_authored_metadata_by_public_key(&tenant, race_key),
            )
            .await
            .map_err(|_| anyhow::anyhow!("under-lock authored reader deadlocked"))?
            .map_err(anyhow::Error::from)
        },
        async {
            let blocked = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    let waiters: i64 = sqlx::query_scalar("SELECT count(*) FROM pg_locks WHERE locktype='advisory' AND NOT granted AND database=(SELECT oid FROM pg_database WHERE datname=current_database())")
                        .fetch_one(pool).await?;
                    if waiters >= 3 { return Ok::<(), anyhow::Error>(()); }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }).await.map_err(|_| anyhow::anyhow!("canonical writers and authored reader did not contend on the held identity lock"))?;
            blocked?;
            lock.commit().await?;
            Ok::<(), anyhow::Error>(())
        }
    );
    release_result?;
    profile_result?;
    let raced_read = reader_result?
        .ok_or_else(|| anyhow::anyhow!("under-lock authored reader hid ordinary metadata"))?;
    anyhow::ensure!(
        raced_read
            .subject()
            .is_some_and(|subject| subject.principal_id() == base.principal_id())
            && (raced_read.metadata().source_event_id.as_bytes()
                == initial.signed_event().claimed_id.as_bytes()
                || raced_read.metadata().source_event_id.as_bytes()
                    == update.signed_event().claimed_id.as_bytes()),
        "under-lock authored reader mixed author/head or changed stable ID"
    );
    let delegated = auth_result?;
    anyhow::ensure!(
        matches!(
            delegated.kind(),
            AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
        ) && delegated.principal_id() == base.principal_id(),
        "under-lock owner materialization must retain base association"
    );
    let final_profile = repository
        .current_by_public_key(&tenant, race_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing concurrently hydrated profile"))?;
    let metadata = final_profile
        .profile()
        .fields()
        .metadata
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing concurrently hydrated metadata"))?;
    anyhow::ensure!(
        metadata.value.name.as_deref() == Some("under-lock-head")
            && metadata.source_event_id.as_bytes() == update.signed_event().claimed_id.as_bytes(),
        "serialized AUTH/profile writers must retain the real latest canonical metadata head"
    );
    let final_read = repository
        .current_authored_metadata_by_public_key(&tenant, race_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("final authored head missing"))?;
    anyhow::ensure!(
        final_read.metadata() == metadata
            && final_read.owner_public_key()
                == Some(NostrPublicKey::from_bytes(*public_key(22).as_bytes()))
            && final_read
                .subject()
                .is_some_and(|subject| subject.principal_id() == base.principal_id()),
        "post-race authored read lost current head or separate canonical owner"
    );
    let winner = repository
        .current_by_public_key(&tenant, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing original raced owner"))?;
    let ProfileKind::Agent(winner_agent) = winner.profile().kind() else {
        anyhow::bail!("missing raced agent kind");
    };
    anyhow::ensure!(
        winner_agent.claimed_owner
            == Some(NostrPublicKey::from_bytes(*public_key(owner).as_bytes())),
        "first-owner race proof remains stable across unrelated head race"
    );
    Ok(())
}

#[async_trait::async_trait]
impl collab::nostr::http::Nip98PrincipalResolver for PersistedResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &collab::nostr::http::VerifiedNip98Identity,
    ) -> Result<
        collaboration_domain::AuthenticatedPrincipal,
        collab::nostr::http::NostrHttpDirectoryError,
    > {
        collab::nostr::http::Nip98PrincipalResolver::resolve(self.0.as_ref(), tenant, identity)
            .await
    }
}

#[async_trait::async_trait]
impl collab::nostr::http::Nip98ReplayStore for RegistrationReplay {
    async fn claim(
        &self,
        tenant: &TenantContext,
        event: nostr_compat::EventId,
        _expires: u64,
    ) -> Result<collab::nostr::http::Nip98ReplayClaim, collab::nostr::http::NostrHttpDirectoryError>
    {
        let inserted = self
            .0
            .lock()
            .map_err(|_| collab::nostr::http::NostrHttpDirectoryError::Unavailable)?
            .insert((tenant.community_id(), event));
        Ok(if inserted {
            collab::nostr::http::Nip98ReplayClaim::Claimed
        } else {
            collab::nostr::http::Nip98ReplayClaim::Replay
        })
    }
}

struct HttpReceiptProbe {
    repository: Arc<IdentityProfileRepository>,
    receipt: std::sync::Mutex<Option<collab::nostr::http::VerifiedNip98Identity>>,
}

#[async_trait::async_trait]
impl collab::nostr::http::Nip98PrincipalResolver for HttpReceiptProbe {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &collab::nostr::http::VerifiedNip98Identity,
    ) -> Result<
        collaboration_domain::AuthenticatedPrincipal,
        collab::nostr::http::NostrHttpDirectoryError,
    > {
        identity.for_tenant(tenant)?;
        *self
            .receipt
            .lock()
            .map_err(|_| collab::nostr::http::NostrHttpDirectoryError::Unavailable)? =
            Some(identity.clone());
        collab::nostr::http::Nip98PrincipalResolver::resolve(
            self.repository.as_ref(),
            tenant,
            identity,
        )
        .await
    }
}

fn http_tenant(number: u128) -> anyhow::Result<TenantContext> {
    Ok(TenantContext::establish(
        Some(TrustedTenantRoute::from_direct_host(
            CommunityId::from_uuid(Uuid::from_u128(number)),
            &format!("tenant-{number}.example.test"),
        )?),
        &[],
    )?)
}

fn signed_http_registration(
    key: u8,
    tenant: &TenantContext,
    path: &str,
) -> anyhow::Result<(SignedEvent, String)> {
    use base64::Engine as _;
    let body = b"canonical registration fixture";
    let event = CanonicalEvent::new(
        public_key(key),
        NOW,
        27_235,
        vec![
            vec![
                "u".into(),
                format!("https://{}{path}", tenant.route_reference()),
            ],
            vec!["method".into(), "POST".into()],
            vec!["payload".into(), hex::encode(Sha256::digest(body))],
        ],
        String::new(),
    );
    let claimed_id = event.event_id()?;
    let signature = Secp256k1::new()
        .sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair(key));
    let source = SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string())?,
    };
    let wire = serde_json::json!({
        "id": source.claimed_id.to_hex(),
        "pubkey": source.event.public_key.to_hex(),
        "created_at": source.event.created_at,
        "kind": source.event.kind,
        "tags": source.event.tags,
        "content": source.event.content,
        "sig": source.signature.to_hex(),
    });
    Ok((
        source,
        format!(
            "Nostr {}",
            base64::engine::general_purpose::STANDARD.encode(serde_json::to_vec(&wire)?),
        ),
    ))
}

fn http_registration_authenticator(
    repository: &Arc<IdentityProfileRepository>,
    replay: RegistrationReplay,
) -> collab::nostr::http::Nip98Authenticator {
    collab::nostr::http::Nip98Authenticator::new(
        true,
        Arc::new(replay),
        Arc::new(PersistedResolver(repository.clone())),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_registration_actual_nip98_stable_ids_and_rollback() -> anyhow::Result<()> {
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL").map_err(|_| {
        anyhow::anyhow!("COLLAB_TEST_DATABASE_URL is required for NIP98 persistence qualification")
    })?;
    let mut url = url::Url::parse(&base_url)?;
    anyhow::ensure!(
        matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback PostgreSQL required",
    );
    url.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let url = url.to_string();
    sqlx::Postgres::create_database(&url).await?;
    let result = exercise_http_registration(&url).await;
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(anyhow::anyhow!(
            "owned NIP98 database cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "NIP98 qualification failed: {error:#}; owned database cleanup failed: {cleanup}",
        )),
    }
}

async fn exercise_http_registration(url: &str) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(url)
        .await?;
    let result = async {
        for migration in [
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/migrations/20260820000100_collaboration_events.up.sql"
            )),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/migrations/20260820000200_collaboration_event_heads.up.sql"
            )),
            UP,
            REGISTRATION_UP,
            OBSERVATION_UP,
            CHANNELS_UP,
            AUDIT_UP,
            MODERATION_UP,
            MODERATION_SUBJECT_UP,
            ACCEPTED_NAMES_UP,
            SIGNED_REASONS_UP,
            MODERATION_RETIREMENT_UP,
        ] {
            sqlx::raw_sql(migration).execute(&pool).await?;
        }
        let repository = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let bounded = tokio::time::timeout(
            std::time::Duration::from_secs(20),
            exercise_http_registration_cases(&repository, &pool),
        )
        .await;
        Arc::try_unwrap(repository)
            .map_err(|_| anyhow::anyhow!("NIP98 fixture leaked a repository reference"))?
            .into_connection()
            .close()
            .await?;
        bounded
            .map_err(|_| anyhow::anyhow!("single-connection NIP98 registration deadlocked"))??;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    pool.close().await;
    result
}

async fn exercise_http_registration_cases(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collab::nostr::http::Nip98AuthenticationError;
    use collaboration_domain::{AuthenticatedPrincipalKind, NostrAuthenticationMethod};
    let tenant = http_tenant(501)?;
    let second_tenant = http_tenant(502)?;
    let body = b"canonical registration fixture";
    let (source, authorization) = signed_http_registration(101, &tenant, "/events")?;
    let replay = RegistrationReplay::default();
    let receipt_probe = Arc::new(HttpReceiptProbe {
        repository: repository.clone(),
        receipt: std::sync::Mutex::new(None),
    });
    let authenticator = collab::nostr::http::Nip98Authenticator::new(
        true,
        Arc::new(replay.clone()),
        receipt_probe.clone(),
    );
    let principal = authenticator
        .authenticate(&tenant, &authorization, "POST", "/events", body, true, NOW)
        .await?;
    anyhow::ensure!(
        matches!(
            principal.kind(),
            AuthenticatedPrincipalKind::NostrIdentity {
                authentication_method: NostrAuthenticationMethod::Nip98,
                public_key: key,
                ..
            } if *key.as_bytes() == *public_key(101).as_bytes()
        ),
        "canonical NIP98 must return exact-key identity, never Human/owner/member inference"
    );
    anyhow::ensure!(
        authenticator
            .authenticate(&tenant, &authorization, "POST", "/events", body, true, NOW,)
            .await
            == Err(Nip98AuthenticationError::Replay),
        "replay must reject before repository access"
    );
    let key = NostrPublicKey::from_bytes(*public_key(101).as_bytes());
    let record = repository
        .current_registration_by_public_key(&tenant, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing canonical NIP98 registration"))?;
    anyhow::ensure!(
        record.profile().is_none(),
        "new NIP98 registration must remain unclassified"
    );
    anyhow::ensure!(record.principal_id() == principal.principal_id());
    let reload = IdentityProfileRepository::new(sea_orm::DatabaseConnection::from(pool.clone()))?;
    anyhow::ensure!(
        reload
            .current_registration_by_public_key(&tenant, key)
            .await?
            == Some(record.clone()),
        "registration association must survive repository reconstruction"
    );
    drop(reload);
    let nip42 = authenticate_registration(
        repository,
        &tenant,
        &signed_registration(101, None, NOW, 101),
    )
    .await?;
    anyhow::ensure!(
        nip42.principal_id() == principal.principal_id(),
        "NIP42 and NIP98 must share stable principal"
    );
    let captured_receipt = receipt_probe
        .receipt
        .lock()
        .map_err(|_| anyhow::anyhow!("receipt lock poisoned"))?
        .clone()
        .ok_or_else(|| anyhow::anyhow!("verified HTTP receipt missing"))?;
    anyhow::ensure!(
        collab::nostr::http::Nip98PrincipalResolver::resolve(
            repository.as_ref(),
            &second_tenant,
            &captured_receipt,
        )
        .await
            == Err(collab::nostr::http::NostrHttpDirectoryError::Denied),
        "canonical resolver must deny reused foreign receipt before SQL acquisition"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&second_tenant, key)
            .await?
            .is_none(),
        "foreign receipt cannot create a principal in another community"
    );
    drop(authenticator);
    drop(receipt_probe);
    let required_scope = collaboration_domain::AuthorizationScope::new("messages:write")?;
    let request = collaboration_domain::AuthorizationRequest {
        tenant: &tenant,
        principal: &principal,
        required_scope: &required_scope,
        action: collaboration_domain::AuthorizationAction::Write,
        resource: collaboration_domain::AuthorizationResource {
            community_id: tenant.community_id(),
            kind: collaboration_domain::AuthorizationResourceKind::Community,
            resource_id: collaboration_domain::AggregateId::from_uuid(Uuid::from_u128(501)),
            owner_principal_id: None,
            channel_id: None,
        },
        current_membership_version: collaboration_domain::AggregateVersion::FIRST,
        community_membership: None,
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis: NOW * 1_000,
    };
    anyhow::ensure!(
        collaboration_domain::authorize(&request)
            == collaboration_domain::AuthorizationDecision::Denied(
                collaboration_domain::AuthorizationDenial::MissingMembership,
            ),
        "successful NIP98 identity and known scope must not grant membership"
    );
    let (_, second_authorization) = signed_http_registration(101, &second_tenant, "/events")?;
    let second = http_registration_authenticator(repository, RegistrationReplay::default())
        .authenticate(
            &second_tenant,
            &second_authorization,
            "POST",
            "/events",
            body,
            true,
            NOW,
        )
        .await?;
    anyhow::ensure!(
        second.principal_id() != principal.principal_id(),
        "same key across tenants must allocate independent IDs"
    );
    for (method, path, wrong_body, now, expected) in [
        (
            "GET",
            "/events",
            body.as_slice(),
            NOW,
            Nip98AuthenticationError::Invalid,
        ),
        (
            "POST",
            "/wrong",
            body.as_slice(),
            NOW,
            Nip98AuthenticationError::TenantBinding,
        ),
        (
            "POST",
            "/events",
            b"tampered".as_slice(),
            NOW,
            Nip98AuthenticationError::Invalid,
        ),
        (
            "POST",
            "/events",
            body.as_slice(),
            NOW + 61,
            Nip98AuthenticationError::Invalid,
        ),
    ] {
        let (_, invalid_authorization) = signed_http_registration(102, &tenant, "/events")?;
        anyhow::ensure!(
            http_registration_authenticator(repository, RegistrationReplay::default())
                .authenticate(
                    &tenant,
                    &invalid_authorization,
                    method,
                    path,
                    wrong_body,
                    true,
                    now
                )
                .await
                == Err(expected),
            "invalid HTTP binding must fail before stable ID allocation"
        );
    }
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(
                &tenant,
                NostrPublicKey::from_bytes(*public_key(102).as_bytes()),
            )
            .await?
            .is_none(),
        "failed NIP98 verification must not create registration"
    );
    let (_, race_authorization) = signed_http_registration(103, &tenant, "/race")?;
    let mut tasks = Vec::new();
    for _ in 0..4 {
        let repository = repository.clone();
        let tenant = tenant.clone();
        let authorization = race_authorization.clone();
        tasks.push(tokio::spawn(async move {
            http_registration_authenticator(&repository, RegistrationReplay::default())
                .authenticate(&tenant, &authorization, "POST", "/race", body, true, NOW)
                .await
        }));
    }
    let mut identifiers = std::collections::BTreeSet::new();
    for task in tasks {
        identifiers.insert(task.await??.principal_id().as_uuid());
    }
    anyhow::ensure!(
        identifiers.len() == 1,
        "concurrent actual receipts must allocate one canonical principal"
    );
    let delegated = authenticate_registration(
        repository,
        &tenant,
        &signed_registration(101, Some(104), NOW, 104),
    )
    .await?;
    anyhow::ensure!(
        delegated.principal_id() == principal.principal_id()
            && matches!(
                delegated.kind(),
                AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
            ),
        "unclassified HTTP registration must allow a later verified first-owner registration"
    );
    let direct = http_registration_authenticator(repository, RegistrationReplay::default())
        .authenticate(&tenant, &authorization, "POST", "/events", body, true, NOW)
        .await?;
    anyhow::ensure!(
        direct.principal_id() == principal.principal_id()
            && matches!(
                direct.kind(),
                AuthenticatedPrincipalKind::NostrIdentity {
                    authentication_method: NostrAuthenticationMethod::Nip98,
                    ..
                }
            ),
        "NIP98 key auth must not invent owner delegation from stored presentation"
    );
    let mut transaction = pool.begin().await?;
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(tenant.community_id().as_uuid().to_string())
        .execute(&mut *transaction)
        .await?;
    let (digest, verified_at): (Vec<u8>, String) = sqlx::query_as(
        "SELECT registration_source_event_id, registration_verified_at_seconds::text FROM public.collaboration_identity_profiles WHERE community_id = $1 AND author_public_key = $2",
    ).bind(tenant.community_id().as_uuid()).bind(key.as_bytes().to_vec())
        .fetch_one(&mut *transaction).await?;
    anyhow::ensure!(
        digest.as_slice() == source.claimed_id.as_bytes() && verified_at == NOW.to_string(),
        "stable registration must retain original verified digest/time across protocol renewals"
    );
    transaction.commit().await?;
    sqlx::raw_sql(&format!(
        "CREATE FUNCTION nip98_fixture_reject_insert() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.author_public_key = decode('{}', 'hex') THEN RAISE EXCEPTION 'owned insert interruption' USING ERRCODE = '40001'; END IF; RETURN NEW; END $$; CREATE TRIGGER nip98_fixture_reject_insert BEFORE INSERT ON public.collaboration_identity_profiles FOR EACH ROW EXECUTE FUNCTION nip98_fixture_reject_insert();",
        public_key(105).to_hex(),
    )).execute(pool).await?;
    let (_, interrupted_authorization) = signed_http_registration(105, &tenant, "/interrupted")?;
    let interrupted = http_registration_authenticator(repository, RegistrationReplay::default());
    anyhow::ensure!(
        interrupted
            .authenticate(
                &tenant,
                &interrupted_authorization,
                "POST",
                "/interrupted",
                body,
                true,
                NOW,
            )
            .await
            == Err(Nip98AuthenticationError::Unavailable),
        "failed canonical transaction must surface Unavailable"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(
                &tenant,
                NostrPublicKey::from_bytes(*public_key(105).as_bytes()),
            )
            .await?
            .is_none(),
        "failed registration transaction must not retain partial IDs"
    );
    sqlx::raw_sql("DROP TRIGGER nip98_fixture_reject_insert ON public.collaboration_identity_profiles; DROP FUNCTION nip98_fixture_reject_insert();").execute(pool).await?;
    anyhow::ensure!(
        interrupted
            .authenticate(
                &tenant,
                &interrupted_authorization,
                "POST",
                "/interrupted",
                body,
                true,
                NOW,
            )
            .await
            == Err(Nip98AuthenticationError::Replay),
        "failed persistence does not undo replay claim"
    );
    let (_, retry_authorization) = signed_http_registration(105, &tenant, "/interrupted?retry=1")?;
    interrupted
        .authenticate(
            &tenant,
            &retry_authorization,
            "POST",
            "/interrupted?retry=1",
            body,
            true,
            NOW,
        )
        .await?;
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(
                &tenant,
                NostrPublicKey::from_bytes(*public_key(105).as_bytes()),
            )
            .await?
            .is_some(),
        "fresh signed retry must create one complete registration after recovery"
    );
    let mut transaction = pool.begin().await?;
    sqlx::query("SET LOCAL ROLE pg_read_all_data")
        .execute(&mut *transaction)
        .await?;
    let bypass: bool =
        sqlx::query_scalar("SELECT rolbypassrls FROM pg_roles WHERE rolname=current_user")
            .fetch_one(&mut *transaction)
            .await?;
    anyhow::ensure!(!bypass, "NIP98 isolation proof must run without BYPASSRLS");
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant.community_id().as_uuid().to_string())
        .execute(&mut *transaction)
        .await?;
    let own_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(
        own_count > 0,
        "RLS positive must read actual HTTP registrations"
    );
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(second_tenant.community_id().as_uuid().to_string())
        .execute(&mut *transaction)
        .await?;
    let foreign_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(
        foreign_count == 0,
        "nonbypass foreign tenant cannot read stable HTTP principal association"
    );
    let own_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1",
    )
    .bind(second_tenant.community_id().as_uuid())
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(
        own_count > 0,
        "foreign denial cannot hide missing reader privilege"
    );
    let persisted_auth_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events WHERE community_id=$1 AND kind IN (22242,27235)")
        .bind(second_tenant.community_id().as_uuid()).fetch_one(&mut *transaction).await?;
    anyhow::ensure!(
        persisted_auth_count == 0,
        "transient auth signatures must not enter durable event storage"
    );
    transaction.rollback().await?;
    eprintln!(
        "NIP98_DISPOSABLE_COMPONENT_VERIFIED: real signed HTTP binding and canonical SQL stable association; replay fixture only, no current-membership/live-service claim"
    );
    Ok(())
}

#[derive(Clone)]
struct CountedRegistrationResolver {
    repository: Arc<IdentityProfileRepository>,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl collab::nostr::auth::NostrPrincipalResolver for CountedRegistrationResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &collab::nostr::auth::VerifiedNip42Identity,
    ) -> Result<
        collaboration_domain::AuthenticatedPrincipal,
        collab::nostr::auth::NostrPrincipalResolutionError,
    > {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        collab::nostr::auth::NostrPrincipalResolver::resolve(
            self.repository.as_ref(),
            tenant,
            identity,
        )
        .await
    }
}

async fn authenticate_with_structural_gate(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    source: &SignedEvent,
    calls: Arc<std::sync::atomic::AtomicUsize>,
) -> Result<
    collaboration_domain::AuthenticatedPrincipal,
    collab::nostr::auth::NostrAuthenticationError,
> {
    let challenge = source
        .event
        .tags
        .iter()
        .find(|tag| tag.first().map(String::as_str) == Some("challenge"))
        .and_then(|tag| tag.get(1))
        .ok_or(collab::nostr::auth::NostrAuthenticationError::InvalidChallenge)?;
    let resolver = collab::nostr::auth::NostrStructuralBanResolver::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
        CountedRegistrationResolver {
            repository: repository.clone(),
            calls,
        },
    )
    .map_err(|_| collab::nostr::auth::NostrAuthenticationError::InfrastructureUnavailable)?;
    let mut authenticator = collab::nostr::auth::NostrConnectionAuthenticator::new(
        tenant.clone(),
        "wss://relay.example.test",
        collab::nostr::auth::NostrAuthChallenge::parse(challenge.clone())?,
        source.event.created_at,
        RegistrationReplay::default(),
        resolver,
    )?;
    authenticator
        .authenticate(source, source.event.created_at)
        .await
        .cloned()
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_registered_nip42_structural_bans_owned_sql() -> anyhow::Result<()> {
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL")?;
    let mut url = url::Url::parse(&base_url)?;
    anyhow::ensure!(
        matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback PostgreSQL required"
    );
    url.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let url = url.to_string();
    sqlx::Postgres::create_database(&url).await?;
    let result = exercise_structural_bans(&url).await;
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(anyhow::anyhow!(
            "structural-ban database cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "structural-ban qualification failed: {error:#}; database cleanup failed: {cleanup}"
        )),
    }
}

async fn exercise_structural_bans(url: &str) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(url)
        .await?;
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        for migration in [
            include_str!("../migrations/20260820000100_collaboration_events.up.sql"),
            include_str!("../migrations/20260820000200_collaboration_event_heads.up.sql"),
            UP,
            REGISTRATION_UP,
            OBSERVATION_UP,
            include_str!("../migrations/20260820000700_collaboration_channels.up.sql"),
            include_str!("../migrations/20260824000400_collaboration_audit.up.sql"),
            include_str!("../migrations/20260824000500_collaboration_moderation.up.sql"),
            MODERATION_SUBJECT_UP,
            ACCEPTED_NAMES_UP,
            SIGNED_REASONS_UP,
            MODERATION_RETIREMENT_UP,
        ] {
            sqlx::raw_sql(migration).execute(&pool).await?;
        }
        let repository = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let result = exercise_structural_ban_cases(&repository, &pool).await;
        Arc::try_unwrap(repository)
            .map_err(|_| anyhow::anyhow!("structural resolver leaked repository reference"))?
            .into_connection()
            .close()
            .await?;
        result
    })
    .await;
    pool.close().await;
    result.map_err(|_| {
        anyhow::anyhow!("single-connection structural-ban reader/resolver deadlocked")
    })?
}

async fn exercise_structural_ban_cases(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collab::nostr::auth::NostrAuthenticationError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let own = tenant(71);
    let foreign = tenant(72);
    for tenant in [&own, &foreign] {
        sqlx::query("INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1,$2,'active',1,'zed',$2,'1',now(),now(),now())")
            .bind(tenant.community_id().as_uuid()).bind(format!("structural-{}.example.test", tenant.community_id())).execute(pool).await?;
    }
    let mut subjects = Vec::new();
    for key in [31_u8, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42] {
        let principal =
            authenticate_registration(repository, &own, &signed_registration(key, None, NOW, key))
                .await?;
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_version, source_observed_at) VALUES ($1,$2,$3,'active',1,now(),now(),'zed',$4,'1',now())")
            .bind(own.community_id().as_uuid()).bind(principal.principal_id().as_uuid())
            .bind(if key == 31 {"admin"} else {"member"}).bind(format!("structural-member-{key}")).execute(pool).await?;
        subjects.push((key, principal.principal_id()));
    }
    let lookup = |key| {
        subjects
            .iter()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, id)| *id)
            .ok_or_else(|| anyhow::anyhow!("missing registered fixture subject"))
    };
    let actor = lookup(31)?;
    let target = lookup(32)?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target,
            sequence: 1,
            version: 2,
            kind: "apply_ban",
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    let calls = Arc::new(AtomicUsize::new(0));
    let banned = signed_registration(32, Some(90), NOW, 81);
    anyhow::ensure!(
        authenticate_with_structural_gate(repository, pool, &own, &banned, calls.clone()).await
            == Err(NostrAuthenticationError::Banned),
        "direct ban must win before unrepresented presented owner"
    );
    anyhow::ensure!(
        calls.load(Ordering::SeqCst) == 0,
        "banned direct key reached registration delegate"
    );
    let banned_before = repository
        .current_registration_by_public_key(
            &own,
            NostrPublicKey::from_bytes(*public_key(32).as_bytes()),
        )
        .await?;
    for (index, tags) in invalid_optional_owner_tags(32, 90).into_iter().enumerate() {
        let invalid = signed_registration_with_tags(32, tags, NOW, 150 + index as u8);
        anyhow::ensure!(
            authenticate_with_structural_gate(repository, pool, &own, &invalid, calls.clone())
                .await
                == Err(NostrAuthenticationError::Banned),
            "invalid optional OA bypassed direct structural ban"
        );
        anyhow::ensure!(
            calls.load(Ordering::SeqCst) == 0,
            "banned key with invalid OA reached actual registration"
        );
        anyhow::ensure!(
            repository
                .current_registration_by_public_key(
                    &own,
                    NostrPublicKey::from_bytes(*public_key(32).as_bytes())
                )
                .await?
                == banned_before,
            "direct denial changed canonical registration or owner proof"
        );
        anyhow::ensure!(
            repository
                .current_subject_by_public_key(
                    &own,
                    NostrPublicKey::from_bytes(*public_key(90).as_bytes())
                )
                .await?
                .is_none(),
            "invalid optional OA materialized its unrepresented owner"
        );
    }
    let owner = lookup(33)?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target: owner,
            sequence: 2,
            version: 2,
            kind: "apply_ban",
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    let owner_banned = signed_registration(34, Some(33), NOW, 82);
    anyhow::ensure!(
        authenticate_with_structural_gate(repository, pool, &own, &owner_banned, calls.clone())
            .await
            == Err(NostrAuthenticationError::Banned),
        "actual presented owner ban must win before delegation"
    );
    anyhow::ensure!(
        calls.load(Ordering::SeqCst) == 0,
        "banned owner reached registration delegate"
    );
    let owner_key = NostrPublicKey::from_bytes(*public_key(33).as_bytes());
    let before_owner = repository
        .current_registration_by_public_key(&own, owner_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing owner"))?;
    let agent_key = NostrPublicKey::from_bytes(*public_key(34).as_bytes());
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&own, agent_key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing agent"))?
            .profile()
            .is_none(),
        "banned presented owner must not materialize agent profile"
    );
    authenticate_registration(
        repository,
        &own,
        &signed_registration(34, Some(33), NOW, 88),
    )
    .await?;
    let historic_owner_profile = repository
        .current_registration_by_public_key(&own, agent_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing historical agent"))?;
    authenticate_with_structural_gate(
        repository,
        pool,
        &own,
        &signed_registration(34, Some(31), NOW, 89),
        calls.clone(),
    )
    .await?;
    anyhow::ensure!(
        calls.load(Ordering::SeqCst) == 1,
        "stored historical banned owner substituted for actual presented clear owner"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&own, agent_key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing agent"))?
            == historic_owner_profile,
        "different presented owner replaced immutable first owner"
    );
    let clear_key = NostrPublicKey::from_bytes(*public_key(38).as_bytes());
    let clear_before = repository
        .current_registration_by_public_key(&own, clear_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing clear direct-key registration"))?;
    let fallback_calls = Arc::new(AtomicUsize::new(0));
    let membership_before: String = sqlx::query_scalar("SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY principal_id),'[]'::jsonb)::text FROM public.collaboration_community_memberships m WHERE community_id=$1")
        .bind(own.community_id().as_uuid()).fetch_one(pool).await?;
    for (index, tags) in invalid_optional_owner_tags(38, 33).into_iter().enumerate() {
        let invalid = signed_registration_with_tags(38, tags, NOW, 170 + index as u8);
        let fallback = authenticate_with_structural_gate(
            repository,
            pool,
            &own,
            &invalid,
            fallback_calls.clone(),
        )
        .await
        .map_err(|error| {
            anyhow::anyhow!("invalid claimed owner became structural authority: {error}")
        })?;
        anyhow::ensure!(
            matches!(fallback.kind(), collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } if *public_key == clear_key)
                && fallback.principal_id() == clear_before.principal_id(),
            "invalid optional OA conferred owner delegation"
        );
        anyhow::ensure!(
            repository
                .current_registration_by_public_key(&own, clear_key)
                .await?
                == Some(clear_before.clone()),
            "invalid optional OA changed clear key profile or owner evidence"
        );
        anyhow::ensure!(
            repository
                .current_registration_by_public_key(&own, owner_key)
                .await?
                == Some(before_owner.clone()),
            "invalid optional OA changed its claimed banned owner"
        );
        let membership_after: String = sqlx::query_scalar("SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY principal_id),'[]'::jsonb)::text FROM public.collaboration_community_memberships m WHERE community_id=$1")
            .bind(own.community_id().as_uuid()).fetch_one(pool).await?;
        anyhow::ensure!(
            membership_after == membership_before,
            "invalid OA changed current membership, role or version"
        );
    }
    anyhow::ensure!(
        fallback_calls.load(Ordering::SeqCst) == 5,
        "genuine clear key fallback did not reach actual canonical registration"
    );
    let foreign_registration = authenticate_registration(
        repository,
        &foreign,
        &signed_registration(32, None, NOW, 83),
    )
    .await?;
    anyhow::ensure!(
        foreign_registration.principal_id() != target,
        "tenant subject mappings collapsed"
    );
    authenticate_with_structural_gate(
        repository,
        pool,
        &foreign,
        &signed_registration(32, None, NOW, 84),
        calls.clone(),
    )
    .await?;
    anyhow::ensure!(
        calls.load(Ordering::SeqCst) == 2,
        "foreign clear key should reach actual registration"
    );
    let timed = lookup(35)?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target: timed,
            sequence: 3,
            version: 2,
            kind: "apply_timeout",
            ban: "none",
            ban_expiry: None,
            timeout: "active",
            timeout_expiry: Some(NOW * 1000 + 10_000),
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    authenticate_with_structural_gate(
        repository,
        pool,
        &own,
        &signed_registration(35, None, NOW, 85),
        calls.clone(),
    )
    .await?;
    let expired = lookup(36)?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target: expired,
            sequence: 4,
            version: 2,
            kind: "apply_ban",
            ban: "active",
            ban_expiry: Some(NOW * 1000 - 1),
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    authenticate_with_structural_gate(
        repository,
        pool,
        &own,
        &signed_registration(36, None, NOW, 86),
        calls.clone(),
    )
    .await?;
    anyhow::ensure!(
        calls.load(Ordering::SeqCst) == 4,
        "timeout-only and expired ban must permit AUTH"
    );
    let unknown = authenticate_with_structural_gate(
        repository,
        pool,
        &own,
        &signed_registration(91, None, NOW, 87),
        calls.clone(),
    )
    .await?;
    anyhow::ensure!(
        calls.load(Ordering::SeqCst) == 5,
        "unbanned unknown key never reached the actual key-registration delegate"
    );
    let unknown_subject = repository
        .current_subject_by_public_key(&own, NostrPublicKey::from_bytes(*public_key(91).as_bytes()))
        .await?
        .ok_or_else(|| anyhow::anyhow!("actual AUTH did not preserve its observed subject"))?;
    anyhow::ensure!(unknown_subject.principal_id() == unknown.principal_id()
        && unknown_subject.status() == collab::identity::profile_repository::IdentitySubjectStatus::RegisteredUnclassified,
        "unknown-key AUTH changed observed identity or inferred Human classification");
    let membership_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2")
        .bind(own.community_id().as_uuid()).bind(unknown.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        membership_count == 0,
        "key authentication fabricated community membership"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&own, owner_key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("owner disappeared"))?
            == before_owner,
        "denial modified presented owner"
    );
    exercise_structural_history_negatives(repository, pool, &own, actor, lookup(37)?).await?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target: lookup(38)?,
            sequence: 8,
            version: 3,
            kind: "apply_ban",
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target: lookup(39)?,
            sequence: 9,
            version: 2,
            kind: "apply_timeout",
            ban: "active",
            ban_expiry: None,
            timeout: "active",
            timeout_expiry: Some(NOW * 1000 + 1000),
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            actor,
            target: lookup(40)?,
            sequence: 10,
            version: 2,
            kind: "apply_ban",
            action_kind: Some("lift_ban"),
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            audit_action: None,
            actor,
            audit_actor: Some(lookup(34)?),
            target: lookup(41)?,
            sequence: 11,
            version: 2,
            kind: "apply_ban",
            action_kind: None,
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    seed_structural_transition(
        pool,
        &own,
        SeedStructuralTransition {
            actor,
            audit_actor: None,
            audit_action: Some("auth.authenticate"),
            target: lookup(42)?,
            sequence: 12,
            version: 2,
            kind: "apply_ban",
            action_kind: None,
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    for corrupt in [
        lookup(38)?,
        lookup(39)?,
        lookup(40)?,
        lookup(41)?,
        lookup(42)?,
    ] {
        use sea_orm::TransactionTrait as _;
        let connection = sea_orm::DatabaseConnection::from(pool.clone());
        let transaction = connection.begin().await?;
        let read = collab::admin::moderation::current_restriction_in_transaction(
            &transaction,
            &own,
            corrupt,
        )
        .await;
        transaction.rollback().await?;
        anyhow::ensure!(
            read == Err(collab::admin::moderation::ModerationRestrictionReadError::InvalidHistory),
            "noncontiguous or contradictory persisted history accepted"
        );
        drop(connection);
    }
    eprintln!(
        "OWNED_SQL_STRUCTURAL_BAN_VERIFIED: actual signed key/OA receipts; seeded represented memberships/history; direct/owner bans short-circuit registration; timeout/expiry/tenant/unknown distinctions; max-one pool; no listener/live service or nonmember parity claimed"
    );
    Ok(())
}

struct SeedStructuralTransition<'a> {
    actor: collaboration_domain::PrincipalId,
    audit_actor: Option<collaboration_domain::PrincipalId>,
    audit_action: Option<&'a str>,
    target: collaboration_domain::PrincipalId,
    sequence: i64,
    version: i64,
    kind: &'a str,
    action_kind: Option<&'a str>,
    ban: &'a str,
    ban_expiry: Option<u64>,
    timeout: &'a str,
    timeout_expiry: Option<u64>,
    occurred_at: u64,
}

async fn seed_structural_transition(
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    seed: SeedStructuralTransition<'_>,
) -> anyhow::Result<()> {
    let SeedStructuralTransition {
        audit_action,
        audit_actor,
        action_kind,
        actor,
        target,
        sequence,
        version,
        kind,
        ban,
        ban_expiry,
        timeout,
        timeout_expiry,
        occurred_at,
    } = seed;
    let audit_action = audit_action.unwrap_or(if matches!(kind, "lift_ban" | "lift_timeout") {
        collab::audit::events::ModerationAuditOperation::LiftRestriction.action()
    } else {
        collab::audit::events::ModerationAuditOperation::ApplyRestriction.action()
    });
    let operation = Uuid::new_v4();
    use collab::audit::repository::{AuditRepository, ExpectedAuditHead};
    use collaboration_domain::{AuditAction, AuditFields, AuditOutcome, AuditRecord, OperationId};
    let audit = AuditRepository::new(sea_orm::DatabaseConnection::from(pool.clone()))?;
    let head = audit.load_head(tenant).await?;
    let entry = audit
        .append(
            tenant,
            head.map(ExpectedAuditHead::Entry)
                .unwrap_or(ExpectedAuditHead::Empty),
            AuditRecord::new(
                OperationId::from_uuid(operation),
                AuditAction::new(audit_action)?,
                Some(audit_actor.unwrap_or(actor)),
                AuditOutcome::Succeeded,
                occurred_at,
                AuditFields::default(),
            )?,
        )
        .await?;
    anyhow::ensure!(
        entry.record().operation_id().as_uuid() == operation,
        "audit append changed operation"
    );
    let mut transaction = pool.begin().await?;
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant.community_id().to_string())
        .execute(&mut *transaction)
        .await?;
    sqlx::query("INSERT INTO public.collaboration_moderation_actions (community_id,action_id,operation_id,actor_principal_id,action_kind,record_kind,record_id,target_principal_id,occurred_at,source_system,source_record_id,source_version,source_observed_at) VALUES ($1,$2,$3,$4,$5,'restriction',$6,$6,to_timestamp($7::numeric/1000),'zed',$8,'1',now())")
        .bind(tenant.community_id().as_uuid()).bind(Uuid::new_v4()).bind(operation).bind(actor.as_uuid()).bind(action_kind.unwrap_or(kind)).bind(target.as_uuid()).bind(occurred_at.to_string()).bind(format!("action-{sequence}-{operation}")).execute(&mut *transaction).await?;
    sqlx::query("UPDATE public.collaboration_moderation_restriction_versions SET is_current=false WHERE community_id=$1 AND target_principal_id=$2 AND is_current")
        .bind(tenant.community_id().as_uuid()).bind(target.as_uuid()).execute(&mut *transaction).await?;
    sqlx::query("INSERT INTO public.collaboration_moderation_restriction_versions (community_id,target_principal_id,restriction_version,is_current,ban_state,ban_expires_at,timeout_state,timeout_expires_at,transition_kind,actor_principal_id,operation_id,occurred_at,source_system,source_record_id,source_version,source_observed_at) VALUES ($1,$2,$3,true,$4,to_timestamp($5::numeric/1000),$6,to_timestamp($7::numeric/1000),$8,$9,$10,to_timestamp($11::numeric/1000),'zed',$12,'1',now())")
        .bind(tenant.community_id().as_uuid()).bind(target.as_uuid()).bind(version).bind(ban).bind(ban_expiry.map(|value|value.to_string())).bind(timeout).bind(timeout_expiry.map(|value|value.to_string())).bind(kind).bind(actor.as_uuid()).bind(operation).bind(occurred_at.to_string()).bind(format!("restriction-{operation}")).execute(&mut *transaction).await?;
    transaction.commit().await?;
    Ok(())
}

async fn exercise_structural_history_negatives(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    actor: collaboration_domain::PrincipalId,
    target: collaboration_domain::PrincipalId,
) -> anyhow::Result<()> {
    use collab::admin::moderation::{
        ModerationRestrictionReadError, current_restriction_in_transaction,
    };
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    seed_structural_transition(
        pool,
        tenant,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target,
            sequence: 5,
            version: 2,
            kind: "apply_ban",
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 - 1000,
        },
    )
    .await?;
    seed_structural_transition(
        pool,
        tenant,
        SeedStructuralTransition {
            audit_action: None,
            audit_actor: None,
            action_kind: None,
            actor,
            target,
            sequence: 6,
            version: 3,
            kind: "apply_timeout",
            ban: "active",
            ban_expiry: None,
            timeout: "active",
            timeout_expiry: Some(NOW * 1000 + 5000),
            occurred_at: NOW * 1000 - 500,
        },
    )
    .await?;
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    let transaction = connection.begin().await?;
    let hydrated = current_restriction_in_transaction(&transaction, tenant, target)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing history"))?;
    anyhow::ensure!(
        hydrated.fields().version.get() == 3 && hydrated.fields().transitions.len() == 2,
        "complete retired history not replayed"
    );
    anyhow::ensure!(
        hydrated.fields().ban.is_active_at(NOW * 1000)
            && hydrated.fields().timeout.is_active_at(NOW * 1000),
        "independent ban/timeout lost"
    );
    let foreign = self::tenant(73);
    anyhow::ensure!(
        current_restriction_in_transaction(&transaction, &foreign, target).await
            == Err(ModerationRestrictionReadError::TenantBoundaryViolation),
        "transaction tenant fence overwritten"
    );
    let key = NostrPublicKey::from_bytes(*public_key(37).as_bytes());
    anyhow::ensure!(
        matches!(
            repository
                .current_registration_in_transaction(&transaction, &foreign, key)
                .await,
            Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
        ),
        "profile transaction tenant fence overwritten"
    );
    let operation = hydrated
        .fields()
        .transitions
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing audit operation"))?
        .source
        .operation_id;
    anyhow::ensure!(
        matches!(
            collab::audit::repository::AuditRepository::load_operation_in_transaction(
                &transaction,
                &foreign,
                operation
            )
            .await,
            Err(collab::audit::repository::AuditRepositoryError::TenantBoundaryViolation)
        ),
        "audit transaction tenant fence overwritten"
    );
    let entry = collab::audit::repository::AuditRepository::load_operation_in_transaction(
        &transaction,
        tenant,
        operation,
    )
    .await?
    .ok_or_else(|| anyhow::anyhow!("canonical audit entry missing"))?;
    anyhow::ensure!(
        entry.record().operation_id() == operation,
        "canonical supplied-transaction read changed operation"
    );
    transaction.rollback().await?;
    exercise_structural_audit_corruption_views(&connection, tenant, target, operation).await?;
    exercise_structural_no_bypass_rls(&connection, repository, tenant, target).await?;
    let mut retirement = pool.begin().await?;
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant.community_id().to_string())
        .execute(&mut *retirement)
        .await?;
    sqlx::query("UPDATE public.collaboration_moderation_restriction_versions SET is_current=false WHERE community_id=$1 AND target_principal_id=$2 AND is_current")
        .bind(tenant.community_id().as_uuid()).bind(target.as_uuid()).execute(&mut *retirement).await?;
    retirement.commit().await?;
    let transaction = connection.begin().await?;
    anyhow::ensure!(
        current_restriction_in_transaction(&transaction, tenant, target).await
            == Err(ModerationRestrictionReadError::InvalidHistory),
        "existing history without head treated as absent"
    );
    transaction.rollback().await?;
    let transaction = connection.begin().await?;
    transaction.execute(Statement::from_string(DatabaseBackend::Postgres, "ALTER TABLE public.collaboration_moderation_restriction_versions RENAME TO structural_fixture_unavailable".to_owned())).await?;
    anyhow::ensure!(
        current_restriction_in_transaction(&transaction, tenant, target).await
            == Err(ModerationRestrictionReadError::Unavailable),
        "SQL error treated as unrestricted or banned"
    );
    transaction.rollback().await?;
    seed_structural_transition(
        pool,
        tenant,
        SeedStructuralTransition {
            actor,
            audit_actor: None,
            audit_action: None,
            target,
            sequence: 7,
            version: 4,
            kind: "lift_timeout",
            action_kind: None,
            ban: "active",
            ban_expiry: None,
            timeout: "none",
            timeout_expiry: None,
            occurred_at: NOW * 1000 + 1000,
        },
    )
    .await?;
    let transaction = connection.begin().await?;
    let lifted = current_restriction_in_transaction(&transaction, tenant, target)
        .await?
        .ok_or_else(|| anyhow::anyhow!("missing canonical lifted head"))?;
    transaction.rollback().await?;
    anyhow::ensure!(
        lifted.fields().version.get() == 4
            && lifted.fields().ban.is_active_at(NOW * 1000 + 1000)
            && !lifted.fields().timeout.is_active_at(NOW * 1000 + 1000),
        "canonical LiftRestriction audit vocabulary or independent state lost"
    );
    drop(connection);
    Ok(())
}

async fn exercise_structural_no_bypass_rls(
    connection: &sea_orm::DatabaseConnection,
    repository: &Arc<IdentityProfileRepository>,
    tenant: &TenantContext,
    target: collaboration_domain::PrincipalId,
) -> anyhow::Result<()> {
    use collab::admin::moderation::{
        ModerationRestrictionReadError, current_restriction_in_transaction,
    };
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    let transaction = connection.begin().await?;
    let role = format!("structural_fixture_{}", Uuid::new_v4().simple());
    let result = async {
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,
            format!("CREATE ROLE {role} NOLOGIN NOBYPASSRLS"))).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,
            format!("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}"))).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,
            format!("SET LOCAL ROLE {role}"))).await?;
        let registered = repository.current_registration_in_transaction(&transaction, tenant, NostrPublicKey::from_bytes(*public_key(37).as_bytes())).await?.ok_or_else(|| anyhow::anyhow!("RLS hid own represented key"))?;
        anyhow::ensure!(registered.principal_id() == target, "RLS returned a different mapped subject");
        let restriction = current_restriction_in_transaction(&transaction, tenant, target).await?.ok_or_else(|| anyhow::anyhow!("RLS hid own restriction"))?;
        anyhow::ensure!(restriction.fields().version.get() == 3, "RLS current history not retained");
        let row = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT count(*)::text AS count FROM public.collaboration_identity_profiles WHERE community_id=$1",
            [self::tenant(72).community_id().as_uuid().into()],
        )).await?.ok_or_else(|| anyhow::anyhow!("missing RLS count"))?;
        anyhow::ensure!(row.try_get::<String>("", "count")? == "0", "no-bypass role read foreign profile");
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres, "SET LOCAL ROLE NONE".to_owned())).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,
            format!("REVOKE SELECT ON public.collaboration_moderation_restriction_versions FROM {role}"))).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,
            format!("SET LOCAL ROLE {role}"))).await?;
        anyhow::ensure!(current_restriction_in_transaction(&transaction, tenant, target).await == Err(ModerationRestrictionReadError::Unavailable), "permission failure became unrestricted or banned");
        Ok::<(),anyhow::Error>(())
    }.await;
    let cleanup = transaction.rollback().await;
    if cleanup.is_ok() {
        let retained = connection
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1) AS retained",
                [role.into()],
            ))
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing transactional role cleanup receipt"))?;
        anyhow::ensure!(
            !retained.try_get::<bool>("", "retained")?,
            "owned RLS role survived rollback"
        );
    }
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(cleanup)) => Err(anyhow::anyhow!(
            "owned transactional RLS role rollback failed: {cleanup}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "RLS qualification failed: {error:#}; owned transactional role rollback failed: {cleanup}"
        )),
    }
}

async fn exercise_structural_audit_corruption_views(
    connection: &sea_orm::DatabaseConnection,
    tenant: &TenantContext,
    target: collaboration_domain::PrincipalId,
    operation: collaboration_domain::OperationId,
) -> anyhow::Result<()> {
    use collab::admin::moderation::{
        ModerationRestrictionReadError, current_restriction_in_transaction,
    };
    use collab::audit::repository::{AuditRepository, AuditRepositoryError};
    use collaboration_domain::{AuditField, AuditFieldName, AuditFields, AuditValue};
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    let altered_fields = serde_json::to_string(&AuditFields::new(vec![AuditField::new(
        AuditFieldName::new("fixture_tamper")?,
        AuditValue::Boolean(true),
    )])?)?;
    for tamper_hash in [true, false] {
        let transaction = connection.begin().await?;
        let result = async {
            transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres, "SELECT set_config('app.community_id',$1,true)", [tenant.community_id().to_string().into()])).await?;
            let columns = if tamper_hash {
                format!("CASE WHEN operation_id = '{}'::uuid THEN decode(repeat('ff',32),'hex') ELSE entry_hash END AS entry_hash, fields", operation.as_uuid())
            } else {
                format!("entry_hash, CASE WHEN operation_id = '{}'::uuid THEN '{}'::jsonb ELSE fields END AS fields", operation.as_uuid(), altered_fields)
            };
            // Mount corrupt reads transactionally; original immutable rows, constraints and triggers remain intact.
            transaction.execute(Statement::from_string(DatabaseBackend::Postgres, "ALTER TABLE public.collaboration_audit_entries RENAME TO structural_fixture_audit_original".to_owned())).await?;
            transaction.execute(Statement::from_string(DatabaseBackend::Postgres, format!("CREATE VIEW public.collaboration_audit_entries AS SELECT community_id,sequence,previous_hash,previous_sequence,operation_id,action,actor_principal_id,outcome,occurred_at_millis,bridge_source,bridge_source_sequence,bridge_source_head,recorded_at,{columns} FROM public.structural_fixture_audit_original"))).await?;
            anyhow::ensure!(matches!(AuditRepository::load_operation_in_transaction(&transaction, tenant, operation).await, Err(AuditRepositoryError::Domain(_))), "canonical audit reader accepted altered hash/fields");
            anyhow::ensure!(current_restriction_in_transaction(&transaction, tenant, target).await == Err(ModerationRestrictionReadError::InvalidHistory), "restriction reader accepted audit hash/fields corruption");
            Ok::<(), anyhow::Error>(())
        }.await;
        let cleanup = transaction.rollback().await;
        match (result, cleanup) {
            (Ok(()), Ok(())) => {}
            (Err(error), Ok(())) => return Err(error),
            (Ok(()), Err(cleanup)) => {
                return Err(anyhow::anyhow!("audit fixture rollback failed: {cleanup}"));
            }
            (Err(error), Err(cleanup)) => {
                return Err(anyhow::anyhow!(
                    "audit qualification failed: {error:#}; rollback failed: {cleanup}"
                ));
            }
        }
        let transaction = connection.begin().await?;
        let result = async {
            let restored =
                AuditRepository::load_operation_in_transaction(&transaction, tenant, operation)
                    .await?
                    .ok_or_else(|| {
                        anyhow::anyhow!("original audit evidence missing after rollback")
                    })?;
            restored.verify()?;
            anyhow::ensure!(
                current_restriction_in_transaction(&transaction, tenant, target)
                    .await?
                    .is_some(),
                "original restriction did not recover after view rollback"
            );
            Ok::<(), anyhow::Error>(())
        }
        .await;
        let cleanup = transaction.rollback().await;
        match (result, cleanup) {
            (Ok(()), Ok(())) => {}
            (Err(error), Ok(())) => return Err(error),
            (Ok(()), Err(cleanup)) => {
                return Err(anyhow::anyhow!("audit readback rollback failed: {cleanup}"));
            }
            (Err(error), Err(cleanup)) => {
                return Err(anyhow::anyhow!(
                    "audit readback failed: {error:#}; rollback failed: {cleanup}"
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone)]
struct CaptureObservationReceipt(
    Arc<std::sync::Mutex<Option<collab::nostr::auth::VerifiedNip42Identity>>>,
);

#[async_trait::async_trait]
impl collab::nostr::auth::NostrPrincipalResolver for CaptureObservationReceipt {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &collab::nostr::auth::VerifiedNip42Identity,
    ) -> Result<
        collaboration_domain::AuthenticatedPrincipal,
        collab::nostr::auth::NostrPrincipalResolutionError,
    > {
        identity
            .for_tenant(tenant)
            .map_err(|_| collab::nostr::auth::NostrPrincipalResolutionError::Denied)?;
        *self
            .0
            .lock()
            .map_err(|_| collab::nostr::auth::NostrPrincipalResolutionError::Unavailable)? =
            Some(identity.clone());
        Err(collab::nostr::auth::NostrPrincipalResolutionError::Denied)
    }
}

async fn verified_observation_receipt(
    tenant: &TenantContext,
    source: &SignedEvent,
) -> anyhow::Result<collab::nostr::auth::VerifiedNip42Identity> {
    let capture = CaptureObservationReceipt(Arc::new(std::sync::Mutex::new(None)));
    let challenge = source
        .event
        .tags
        .iter()
        .find(|tag| tag.first().map(String::as_str) == Some("challenge"))
        .and_then(|tag| tag.get(1))
        .ok_or_else(|| anyhow::anyhow!("missing challenge"))?;
    let mut verifier = collab::nostr::auth::NostrConnectionAuthenticator::new(
        tenant.clone(),
        "wss://relay.example.test",
        collab::nostr::auth::NostrAuthChallenge::parse(challenge.clone())?,
        NOW,
        RegistrationReplay::default(),
        capture.clone(),
    )?;
    anyhow::ensure!(
        matches!(
            verifier.authenticate(source, NOW).await,
            Err(collab::nostr::auth::NostrAuthenticationError::IdentityDenied)
        ),
        "receipt capture must not manufacture accepted authentication"
    );
    let receipt = capture
        .0
        .lock()
        .map_err(|_| anyhow::anyhow!("capture poisoned"))?
        .take();
    receipt.ok_or_else(|| anyhow::anyhow!("actual verifier produced no receipt"))
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_observation_actual_receipts_promotion_and_sql_boundaries() -> anyhow::Result<()> {
    let mut url = url::Url::parse(&std::env::var("COLLAB_TEST_DATABASE_URL")?)?;
    anyhow::ensure!(
        matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback required"
    );
    url.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let url = url.to_string();
    sqlx::Postgres::create_database(&url).await?;
    let result = exercise_observation_database(&url).await;
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(cleanup)) => Err(anyhow::anyhow!("observation DB cleanup failed: {cleanup}")),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "observation qualification failed: {error:#}; DB cleanup failed: {cleanup}"
        )),
    }
}

async fn exercise_observation_database(url: &str) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(url)
        .await?;
    let result = async {
        for migration in [
            include_str!("../migrations/20260820000100_collaboration_events.up.sql"),
            include_str!("../migrations/20260820000200_collaboration_event_heads.up.sql"),
            UP,
            REGISTRATION_UP,
            OBSERVATION_UP,
            CHANNELS_UP,
            AUDIT_UP,
            MODERATION_UP,
            MODERATION_SUBJECT_UP,
            ACCEPTED_NAMES_UP,
            SIGNED_REASONS_UP,
            MODERATION_RETIREMENT_UP,
        ] {
            sqlx::raw_sql(migration).execute(&pool).await?;
        }
        let repository = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let bounded = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            exercise_observation_cases(&repository, &pool),
        )
        .await;
        Arc::try_unwrap(repository)
            .map_err(|_| anyhow::anyhow!("observation repository reference leaked"))?
            .into_connection()
            .close()
            .await?;
        bounded.map_err(|_| anyhow::anyhow!("one-connection observation deadlock"))??;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    pool.close().await;
    result?;
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(6)
        .connect(url)
        .await?;
    let result = async {
        let repository = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let outcome = exercise_observation_race(&repository, &pool).await;
        Arc::try_unwrap(repository)
            .map_err(|_| anyhow::anyhow!("observation race reference leaked"))?
            .into_connection()
            .close()
            .await?;
        outcome
    }
    .await;
    pool.close().await;
    result
}

async fn observation_row_snapshot(
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
) -> anyhow::Result<serde_json::Value> {
    Ok(sqlx::query_scalar("SELECT COALESCE(jsonb_agg(to_jsonb(profile) - 'accepted_nip05_name' - 'accepted_nip05_source_event_id' - 'nip05_processed_event_id' - 'nip05_projection_host' - 'observation_proof' - 'observation_source_event_id' - 'observation_verified_at_millis' - 'observation_moderation_operation_id' ORDER BY author_public_key), '[]'::jsonb) FROM public.collaboration_identity_profiles AS profile WHERE community_id=$1")
        .bind(tenant.community_id().as_uuid()).fetch_one(pool).await?)
}

async fn exercise_observation_cases(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collab::identity::profile_repository::IdentitySubjectStatus;
    use sea_orm::TransactionTrait as _;
    let own = tenant(801);
    let foreign = tenant(802);
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    // This genuine registration creates a legacy v2 row with no observation; reversing/reapplying observation must preserve every old byte.
    authenticate_registration(repository, &own, &signed_registration(130, None, NOW, 130)).await?;
    let legacy_source = signed_profile(&own, 129, NOW, vec![owner_tag(129, 128, "kind=0")], "{}");
    materialized(
        repository
            .project_profile_event(&own, &legacy_source)
            .await?,
    )?;
    // Seed historical v1 encoding from the actual signed canonical profile only in this owned fixture; no source or authority changes.
    sqlx::query("WITH original AS (DELETE FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key=$2 RETURNING *) INSERT INTO public.collaboration_identity_profiles (community_id,profile_id,principal_id,author_public_key,profile_kind,owner_public_key,first_owner_proof_event_id,source_event_id,aggregate_version,profile_payload,profile_hash,created_at,updated_at,first_owner_auth_proof,latest_owner_auth_proof,registration_source_event_id,registration_verified_at_seconds,projection_hash_version,legacy_profile_hash,observation_proof,observation_source_event_id,observation_verified_at_millis) SELECT community_id,profile_id,principal_id,author_public_key,profile_kind,owner_public_key,first_owner_proof_event_id,source_event_id,aggregate_version,profile_payload,legacy_profile_hash,created_at,updated_at,first_owner_auth_proof,latest_owner_auth_proof,registration_source_event_id,registration_verified_at_seconds,1,legacy_profile_hash,NULL,NULL,NULL FROM original")
        .bind(own.community_id().as_uuid()).bind(public_key(129).as_bytes().to_vec()).execute(pool).await?;
    repository
        .current_registration_by_public_key(
            &own,
            NostrPublicKey::from_bytes(*public_key(129).as_bytes()),
        )
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!("genuine historical v1 profile failed canonical hydration")
        })?;
    let old = observation_row_snapshot(pool, &own).await?;
    let mut rollback = pool.begin().await?;
    sqlx::raw_sql(ACCEPTED_NAMES_DOWN)
        .execute(&mut *rollback)
        .await?;
    sqlx::raw_sql(MODERATION_SUBJECT_DOWN)
        .execute(&mut *rollback)
        .await?;
    sqlx::raw_sql(OBSERVATION_DOWN)
        .execute(&mut *rollback)
        .await?;
    rollback.commit().await?;
    anyhow::ensure!(
        observation_row_snapshot(pool, &own).await? == old,
        "empty observation reversal changed legacy row bytes/hash"
    );
    sqlx::raw_sql(OBSERVATION_UP).execute(pool).await?;
    sqlx::raw_sql(MODERATION_SUBJECT_UP).execute(pool).await?;
    sqlx::raw_sql(ACCEPTED_NAMES_UP).execute(pool).await?;
    anyhow::ensure!(
        observation_row_snapshot(pool, &own).await? == old,
        "observation upgrade changed legacy v2 bytes/hash"
    );
    repository
        .current_registration_by_public_key(
            &own,
            NostrPublicKey::from_bytes(*public_key(130).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("v2 hash no longer hydrates"))?;
    repository
        .current_by_public_key(
            &own,
            NostrPublicKey::from_bytes(*public_key(129).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("observation upgrade broke original v1 hash/proof"))?;
    let mut rollback = pool.begin().await?;
    sqlx::raw_sql(ACCEPTED_NAMES_DOWN)
        .execute(&mut *rollback)
        .await?;
    sqlx::raw_sql(MODERATION_SUBJECT_DOWN)
        .execute(&mut *rollback)
        .await?;
    sqlx::raw_sql(OBSERVATION_DOWN)
        .execute(&mut *rollback)
        .await?;
    rollback.commit().await?;
    sqlx::raw_sql(OBSERVATION_UP).execute(pool).await?;
    sqlx::raw_sql(MODERATION_SUBJECT_UP).execute(pool).await?;
    sqlx::raw_sql(ACCEPTED_NAMES_UP).execute(pool).await?;
    anyhow::ensure!(
        observation_row_snapshot(pool, &own).await? == old,
        "repeated observation reversal changed legacy v1/v2 bytes/hash"
    );
    let source = signed_profile(&own, 131, NOW, vec![], "{\"name\":\"observed only\"}");
    let event_id = source.signed_event().claimed_id;
    anyhow::ensure!(
        repository.project_profile_event(&own, &source).await?
            == IdentityProfileProjectionOutcome::Unclassified,
        "unknown metadata inferred profile classification"
    );
    let key = NostrPublicKey::from_bytes(*public_key(131).as_bytes());
    anyhow::ensure!(
        repository
            .current_subject_by_public_key(&own, key)
            .await?
            .is_none(),
        "kind0 projection allocated an unknown subject"
    );
    let transaction = connection.begin().await?;
    let observed = repository
        .observe_retained_event_author_in_transaction(&transaction, &own, event_id)
        .await?;
    anyhow::ensure!(
        observed.status() == IdentitySubjectStatus::ObservedOnly,
        "retained source classified/authenticated observed subject"
    );
    let duplicate = repository
        .observe_retained_event_author_in_transaction(&transaction, &own, event_id)
        .await?;
    anyhow::ensure!(
        observed == duplicate,
        "duplicate observation changed immutable evidence/version"
    );
    transaction.commit().await?;
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&own, key)
            .await?
            .is_none(),
        "observation was exposed as completed registration"
    );
    anyhow::ensure!(
        repository.current_by_public_key(&own, key).await?.is_none(),
        "observation became typed profile"
    );
    let transaction = connection.begin().await?;
    anyhow::ensure!(
        repository
            .current_registration_in_transaction(&transaction, &own, key)
            .await?
            .is_none(),
        "transaction lookup exposed observed registration"
    );
    anyhow::ensure!(
        repository
            .current_subject_in_transaction(&transaction, &own, key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("observed subject missing"))?
            == observed,
        "explicit subject lookup lost association"
    );
    anyhow::ensure!(
        matches!(
            repository
                .current_subject_in_transaction(&transaction, &foreign, key)
                .await,
            Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
        ),
        "subject lookup retargeted tenant transaction"
    );
    transaction.rollback().await?;
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let through_structural_gate = authenticate_with_structural_gate(
        repository,
        pool,
        &own,
        &signed_registration(131, None, NOW, 131),
        calls.clone(),
    )
    .await?;
    anyhow::ensure!(
        calls.load(std::sync::atomic::Ordering::SeqCst) == 1
            && through_structural_gate.principal_id() == observed.principal_id(),
        "observed-key ban lookup failed to preserve identity before actual AUTH registration"
    );
    let principal =
        authenticate_registration(repository, &own, &signed_registration(131, None, NOW, 132))
            .await?;
    anyhow::ensure!(
        principal.principal_id() == observed.principal_id(),
        "genuine AUTH replaced observed stable principal"
    );
    let registered = repository
        .current_subject_by_public_key(&own, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("promoted subject missing"))?;
    anyhow::ensure!(
        registered.status() == IdentitySubjectStatus::RegisteredUnclassified
            && registered.profile_id() == observed.profile_id()
            && registered.version().get() == observed.version().get() + 1,
        "AUTH promotion did not retain IDs/version/classification boundary"
    );
    authenticate_registration(repository, &own, &signed_registration(131, None, NOW, 133)).await?;
    anyhow::ensure!(
        repository
            .current_subject_by_public_key(&own, key)
            .await?
            .as_ref()
            == Some(&registered),
        "duplicate AUTH rewrote first registration/observation/version"
    );
    let receipt =
        verified_observation_receipt(&own, &signed_registration(132, Some(133), NOW, 134)).await?;
    let transaction = connection.begin().await?;
    let signer = repository
        .observe_verified_nip42_key_in_transaction(&transaction, &own, &receipt)
        .await?;
    let owner = repository
        .observe_verified_nip42_presented_owner_in_transaction(&transaction, &own, &receipt)
        .await?
        .ok_or_else(|| anyhow::anyhow!("verified owner observation missing"))?;
    anyhow::ensure!(
        signer.status() == IdentitySubjectStatus::ObservedOnly
            && owner.status() == IdentitySubjectStatus::ObservedOnly
            && signer.principal_id() != owner.principal_id(),
        "presented OA inferred owner Human/agent/member authority"
    );
    anyhow::ensure!(
        repository
            .current_registration_in_transaction(&transaction, &own, signer.public_key())
            .await?
            .is_none()
            && repository
                .current_registration_in_transaction(&transaction, &own, owner.public_key())
                .await?
                .is_none(),
        "OA observation leaked completed registration"
    );
    anyhow::ensure!(
        matches!(
            repository
                .observe_verified_nip42_key_in_transaction(&transaction, &foreign, &receipt)
                .await,
            Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
        ),
        "foreign actual receipt crossed tenant"
    );
    transaction.commit().await?;
    let auth_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events WHERE kind=22242")
            .fetch_one(pool)
            .await?;
    anyhow::ensure!(
        auth_count == 0,
        "observation persisted transient AUTH original"
    );
    let receipt =
        verified_observation_receipt(&foreign, &signed_registration(132, None, NOW, 135)).await?;
    let transaction = connection.begin().await?;
    let foreign_subject = repository
        .observe_verified_nip42_key_in_transaction(&transaction, &foreign, &receipt)
        .await?;
    anyhow::ensure!(
        foreign_subject.principal_id() != signer.principal_id(),
        "cross-tenant key shared canonical subject"
    );
    transaction.commit().await?;
    let (_, authorization) =
        signed_http_registration(132, &http_tenant(801)?, "/observed-promotion")?;
    let http = http_registration_authenticator(repository, RegistrationReplay::default());
    let http_principal = http
        .authenticate(
            &http_tenant(801)?,
            &authorization,
            "POST",
            "/observed-promotion",
            b"canonical registration fixture",
            true,
            NOW,
        )
        .await?;
    anyhow::ensure!(
        http_principal.principal_id() == signer.principal_id(),
        "NIP98 promotion replaced observed principal"
    );
    let subject = repository
        .current_subject_by_public_key(&own, signer.public_key())
        .await?
        .ok_or_else(|| anyhow::anyhow!("HTTP-promoted subject missing"))?;
    anyhow::ensure!(
        subject.version().get() == signer.version().get() + 1
            && subject.status() == IdentitySubjectStatus::RegisteredUnclassified,
        "NIP98 did not perform actual promotion"
    );
    let first_owner = authenticate_registration(
        repository,
        &own,
        &signed_registration(132, Some(133), NOW, 136),
    )
    .await?;
    anyhow::ensure!(
        first_owner.principal_id() == signer.principal_id()
            && matches!(first_owner.kind(), collaboration_domain::AuthenticatedPrincipalKind::OwnerAttestedAgent { owner_public_key, .. } if *owner_public_key == owner.public_key()),
        "genuine OA materialization lost observation IDs or first-owner semantics"
    );
    let materialized = repository
        .current_subject_by_public_key(&own, signer.public_key())
        .await?
        .ok_or_else(|| anyhow::anyhow!("observed agent materialization missing"))?;
    anyhow::ensure!(
        materialized.status() == IdentitySubjectStatus::ClassifiedProfile
            && materialized.profile_id() == signer.profile_id()
            && materialized.version().get() == subject.version().get() + 1,
        "actual owner proof did not materialize canonical classified profile/version"
    );
    let losing_owner = authenticate_registration(
        repository,
        &own,
        &signed_registration(132, Some(134), NOW, 137),
    )
    .await?;
    anyhow::ensure!(
        matches!(
            losing_owner.kind(),
            collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { .. }
        ) && losing_owner.principal_id() == signer.principal_id()
            && repository
                .current_subject_by_public_key(&own, signer.public_key())
                .await?
                .as_ref()
                == Some(&materialized),
        "different OA rewrote first owner or rejected otherwise valid exact-key auth"
    );
    authenticate_registration(
        repository,
        &own,
        &signed_registration(132, Some(133), NOW, 138),
    )
    .await?;
    let renewed = repository
        .current_subject_by_public_key(&own, signer.public_key())
        .await?
        .ok_or_else(|| anyhow::anyhow!("renewed observed agent missing"))?;
    anyhow::ensure!(
        renewed.principal_id() == materialized.principal_id()
            && renewed.profile_id() == materialized.profile_id()
            && renewed.version().get() == materialized.version().get() + 1,
        "same-owner renewal changed stable observation association"
    );
    exercise_observation_source_boundaries(repository, &connection, &own, &foreign, event_id, key)
        .await?;
    exercise_observation_limited_rollback(pool).await?;
    exercise_observation_sql_guards(repository, pool, &own, owner.public_key(), event_id).await?;
    exercise_observation_rls(repository, &connection, &own, owner.public_key()).await?;
    exercise_signed_absent_restriction_journey(repository, pool).await?;
    Ok(())
}

async fn exercise_observation_sql_guards(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    observed_key: NostrPublicKey,
    source_event_id: nostr_compat::EventId,
) -> anyhow::Result<()> {
    let before = repository
        .current_subject_by_public_key(tenant, observed_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("guard subject missing"))?;
    let mut transaction = pool.begin().await?;
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant.community_id().to_string())
        .execute(&mut *transaction)
        .await?;
    let rejected = sqlx::query("UPDATE public.collaboration_identity_profiles SET observation_verified_at_millis=observation_verified_at_millis+1, aggregate_version=aggregate_version+1 WHERE community_id=$1 AND author_public_key=$2").bind(tenant.community_id().as_uuid()).bind(observed_key.as_bytes().to_vec()).execute(&mut *transaction).await;
    let immutable = rejected
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .is_some_and(|error| error.code().as_deref() == Some("23514"));
    transaction.rollback().await?;
    anyhow::ensure!(
        immutable
            && repository
                .current_subject_by_public_key(tenant, observed_key)
                .await?
                .as_ref()
                == Some(&before),
        "observation provenance changed"
    );
    for (proof, source, verified, expected) in [
        (
            serde_json::json!({"source_kind":"UnverifiedImportedKey"}),
            None,
            Some(1_i64),
            "23514",
        ),
        (
            serde_json::json!({"source_kind":"RetainedEventAuthor","event_id":vec![0_u8;32],"verified_at_millis":1}),
            Some(vec![0_u8; 32]),
            Some(1_i64),
            "23503",
        ),
        (
            serde_json::json!({"source_kind":"VerifiedNip42Key","event_id":vec![0_u8;32],"verified_at_seconds":1}),
            None,
            None,
            "23514",
        ),
    ] {
        let mut transaction = pool.begin().await?;
        sqlx::query("SELECT set_config('app.community_id',$1,true)")
            .bind(tenant.community_id().to_string())
            .execute(&mut *transaction)
            .await?;
        let rejected = sqlx::query("INSERT INTO public.collaboration_identity_profiles (community_id,profile_id,principal_id,author_public_key,aggregate_version,profile_hash,projection_hash_version,observation_proof,observation_source_event_id,observation_verified_at_millis) VALUES ($1,$2,$3,$4,1,$5,3,$6,$7,CAST($8 AS bigint))")
            .bind(tenant.community_id().as_uuid()).bind(Uuid::new_v4()).bind(Uuid::new_v4())
            .bind(public_key(136).as_bytes().to_vec()).bind(vec![0_u8;32]).bind(proof).bind(source).bind(verified).execute(&mut *transaction).await;
        let refused = rejected
            .as_ref()
            .err()
            .and_then(sqlx::Error::as_database_error)
            .is_some_and(|error| error.code().as_deref() == Some(expected));
        transaction.rollback().await?;
        anyhow::ensure!(
            refused,
            "SQL observation provenance/FK shape was accepted or failed with the wrong reason"
        );
        anyhow::ensure!(
            repository
                .current_subject_by_public_key(
                    tenant,
                    NostrPublicKey::from_bytes(*public_key(136).as_bytes())
                )
                .await?
                .is_none(),
            "invalid observation shape leaked persistent authority"
        );
    }
    let mut transaction = pool.begin().await?;
    let rejected = sqlx::raw_sql(OBSERVATION_DOWN)
        .execute(&mut *transaction)
        .await;
    let refused = rejected
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .is_some_and(|error| {
            error.code().as_deref() == Some("23514")
                && error
                    .message()
                    .contains("discard observed identity evidence")
        });
    transaction.rollback().await?;
    anyhow::ensure!(
        refused
            && repository
                .current_subject_by_public_key(tenant, observed_key)
                .await?
                .as_ref()
                == Some(&before),
        "observation rollback discarded evidence"
    );
    let mut transaction = pool.begin().await?;
    sqlx::query("SELECT set_config('app.community_id',$1,true)")
        .bind(tenant.community_id().to_string())
        .execute(&mut *transaction)
        .await?;
    let rejected = sqlx::query(
        "DELETE FROM public.collaboration_events WHERE community_id=$1 AND event_id=$2",
    )
    .bind(tenant.community_id().as_uuid())
    .bind(source_event_id.as_bytes().to_vec())
    .execute(&mut *transaction)
    .await;
    let retained = rejected
        .as_ref()
        .err()
        .and_then(sqlx::Error::as_database_error)
        .is_some_and(|error| error.code().as_deref() == Some("23503"));
    transaction.rollback().await?;
    anyhow::ensure!(retained, "observation event FK did not preserve source");
    Ok(())
}

async fn exercise_observation_rls(
    repository: &Arc<IdentityProfileRepository>,
    connection: &sea_orm::DatabaseConnection,
    tenant: &TenantContext,
    key: NostrPublicKey,
) -> anyhow::Result<()> {
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    let role = format!("observation_fixture_{}", Uuid::new_v4().simple());
    let transaction = connection.begin().await?;
    let result = async {
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres, format!("CREATE ROLE {role} NOLOGIN NOBYPASSRLS"))).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres, format!("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}"))).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres, format!("SET LOCAL ROLE {role}"))).await?;
        anyhow::ensure!(repository.current_subject_in_transaction(&transaction, tenant, key).await?.is_some(), "RLS hid own observed subject");
        anyhow::ensure!(repository.current_registration_in_transaction(&transaction, tenant, key).await?.is_none(), "RLS lookup conferred completed registration");
        let row = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres, "SELECT count(*)::text AS count FROM public.collaboration_identity_profiles WHERE community_id=$1", [self::tenant(802).community_id().as_uuid().into()])).await?.ok_or_else(||anyhow::anyhow!("RLS receipt absent"))?;
        anyhow::ensure!(row.try_get::<String>("", "count")? == "0", "RLS exposed foreign observation");
        Ok::<(),anyhow::Error>(())
    }.await;
    let cleanup = transaction.rollback().await;
    if cleanup.is_ok() {
        let row = connection
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1) AS retained",
                [role.into()],
            ))
            .await?
            .ok_or_else(|| anyhow::anyhow!("role cleanup receipt absent"))?;
        anyhow::ensure!(
            !row.try_get::<bool>("", "retained")?,
            "owned observation role survived rollback"
        );
    }
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(cleanup)) => Err(anyhow::anyhow!(
            "observation RLS rollback failed: {cleanup}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "observation RLS failed: {error:#}; rollback failed: {cleanup}"
        )),
    }
}

async fn exercise_observation_race(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use sea_orm::TransactionTrait as _;
    let tenant = tenant(803);
    let source = signed_registration(140, None, NOW, 140);
    let receipt = verified_observation_receipt(&tenant, &source).await?;
    let observe = || async {
        let connection = sea_orm::DatabaseConnection::from(pool.clone());
        let transaction = connection.begin().await?;
        let subject = repository
            .observe_verified_nip42_key_in_transaction(&transaction, &tenant, &receipt)
            .await?;
        transaction.commit().await?;
        Ok::<_, anyhow::Error>(subject)
    };
    let (first, second, principal) =
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            tokio::join!(
                observe(),
                observe(),
                authenticate_registration(repository, &tenant, &source)
            )
        })
        .await
        .map_err(|_| anyhow::anyhow!("observation/AUTH race deadlocked"))?;
    let first = first?;
    let second = second?;
    let principal = principal?;
    anyhow::ensure!(
        first.principal_id() == second.principal_id()
            && first.principal_id() == principal.principal_id()
            && first.profile_id() == second.profile_id(),
        "concurrent observation/AUTH allocated duplicate authorities"
    );
    let current = repository
        .current_subject_by_public_key(&tenant, first.public_key())
        .await?
        .ok_or_else(|| anyhow::anyhow!("raced subject missing"))?;
    anyhow::ensure!(
        current.status()
            != collab::identity::profile_repository::IdentitySubjectStatus::ObservedOnly
            && (1..=2).contains(&current.version().get()),
        "race failed registration or manufactured versions"
    );
    let count:i64=sqlx::query_scalar("SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key=$2").bind(tenant.community_id().as_uuid()).bind(first.public_key().as_bytes().to_vec()).fetch_one(pool).await?;
    anyhow::ensure!(count == 1, "duplicate raced subject rows");
    exercise_signed_ban_auth_race(repository, pool).await?;
    Ok(())
}

async fn exercise_observation_source_boundaries(
    repository: &Arc<IdentityProfileRepository>,
    connection: &sea_orm::DatabaseConnection,
    own: &TenantContext,
    foreign: &TenantContext,
    event_id: nostr_compat::EventId,
    observed_key: NostrPublicKey,
) -> anyhow::Result<()> {
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    let before = repository
        .current_subject_by_public_key(own, observed_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("retained observation disappeared"))?;
    let transaction = connection.begin().await?;
    anyhow::ensure!(
        matches!(
            repository
                .observe_retained_event_author_in_transaction(&transaction, foreign, event_id)
                .await,
            Err(IdentityProfileRepositoryError::InvalidRecord)
        ),
        "foreign retained event established observation"
    );
    transaction.rollback().await?;
    let transaction = connection.begin().await?;
    anyhow::ensure!(
        matches!(
            repository
                .observe_retained_event_author_in_transaction(
                    &transaction,
                    own,
                    nostr_compat::EventId::from_bytes([0; 32])
                )
                .await,
            Err(IdentityProfileRepositoryError::InvalidRecord)
        ),
        "missing retained event established observation"
    );
    transaction.rollback().await?;
    let receipt =
        verified_observation_receipt(own, &signed_registration(135, None, NOW, 139)).await?;
    let transaction = connection.begin().await?;
    repository
        .current_subject_in_transaction(&transaction, foreign, observed_key)
        .await?;
    anyhow::ensure!(
        matches!(
            repository
                .observe_verified_nip42_key_in_transaction(&transaction, own, &receipt)
                .await,
            Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
        ),
        "verified receipt retargeted an existing transaction"
    );
    transaction.rollback().await?;
    anyhow::ensure!(
        repository
            .current_subject_by_public_key(own, receipt.public_key())
            .await?
            .is_none(),
        "failed transaction boundary allocated an observation"
    );
    let transaction = connection.begin().await?;
    repository
        .observe_verified_nip42_key_in_transaction(&transaction, own, &receipt)
        .await?;
    transaction.rollback().await?;
    anyhow::ensure!(
        repository
            .current_subject_by_public_key(own, receipt.public_key())
            .await?
            .is_none(),
        "rolled back observation leaked subject/IDs"
    );
    let transaction = connection.begin().await?;
    let result = async {
        transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id',$1,true)", [own.community_id().to_string().into()])).await?;
        // Only the read surface changes inside this rolled-back fixture; retained immutable rows and their triggers remain intact.
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,
            "ALTER TABLE public.collaboration_events RENAME TO observation_fixture_events_original".to_owned())).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,
            "CREATE VIEW public.collaboration_events AS SELECT community_id,event_id,author_public_key,event_created_at,kind,tags,content,canonical_event_bytes,decode(repeat('00',64),'hex') AS signature,signature_state,verified_at,persistence_class,discriminator,received_at FROM public.observation_fixture_events_original".to_owned())).await?;
        anyhow::ensure!(matches!(repository.current_subject_in_transaction(&transaction, own, observed_key).await,
            Err(IdentityProfileRepositoryError::InvalidRecord)), "retained observation accepted corrupt outer signature");
        anyhow::ensure!(matches!(repository.observe_retained_event_author_in_transaction(&transaction, own, event_id).await,
            Err(IdentityProfileRepositoryError::InvalidRecord)), "observation creation trusted stored verification label over actual signature");
        Ok::<(),anyhow::Error>(())
    }.await;
    let cleanup = transaction.rollback().await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => {}
        (Err(error), Ok(())) => return Err(error),
        (Ok(()), Err(error)) => {
            return Err(anyhow::anyhow!("source view rollback failed: {error}"));
        }
        (Err(error), Err(cleanup)) => {
            return Err(anyhow::anyhow!(
                "source boundary failed: {error:#}; rollback failed: {cleanup}"
            ));
        }
    }
    anyhow::ensure!(
        repository
            .current_subject_by_public_key(own, observed_key)
            .await?
            .as_ref()
            == Some(&before),
        "corruption-view rollback lost original retained observation/proof"
    );
    Ok(())
}

async fn exercise_observation_limited_rollback(pool: &sqlx::PgPool) -> anyhow::Result<()> {
    let role = format!("observation_rollback_{}", Uuid::new_v4().simple());
    let mut transaction = pool.begin().await?;
    let result = async {
        sqlx::raw_sql(&format!(
            "CREATE ROLE {role} NOLOGIN NOBYPASSRLS; SET LOCAL ROLE {role}"
        ))
        .execute(&mut *transaction)
        .await?;
        let refused = sqlx::raw_sql(OBSERVATION_DOWN)
            .execute(&mut *transaction)
            .await;
        anyhow::ensure!(
            refused
                .as_ref()
                .err()
                .and_then(sqlx::Error::as_database_error)
                .is_some_and(|error| error.code().as_deref() == Some("42501")
                    && error.message().contains("every-tenant visibility")),
            "limited migration role could reverse unseen observation evidence"
        );
        Ok::<(), anyhow::Error>(())
    }
    .await;
    let cleanup = transaction.rollback().await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => {}
        (Err(error), Ok(())) => return Err(error),
        (Ok(()), Err(error)) => {
            return Err(anyhow::anyhow!(
                "limited rollback fixture cleanup failed: {error}"
            ));
        }
        (Err(error), Err(cleanup)) => {
            return Err(anyhow::anyhow!(
                "limited rollback failed: {error:#}; cleanup failed: {cleanup}"
            ));
        }
    }
    let retained: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1)")
            .bind(&role)
            .fetch_one(pool)
            .await?;
    anyhow::ensure!(
        !retained,
        "owned limited rollback role survived transaction"
    );
    Ok(())
}

fn signed_restriction_command(
    actor: u8,
    kind: u32,
    target: u8,
    now: u64,
    nonce: u8,
    expiration: Option<u64>,
) -> SignedEvent {
    let mut tags = vec![
        vec!["p".into(), public_key(target).to_hex()],
        vec!["nonce".into(), nonce.to_string()],
        vec!["reason".into(), "disposable signed restriction".into()],
    ];
    if let Some(expiration) = expiration {
        tags.push(vec!["expiration".into(), expiration.to_string()]);
    }
    let event = CanonicalEvent::new(
        public_key(actor),
        now,
        u16::try_from(kind).expect("canonical moderation kind"),
        tags,
        "not the moderation reason".into(),
    );
    sign_restriction_event(actor, event)
}

fn sign_restriction_event(actor: u8, event: CanonicalEvent) -> SignedEvent {
    let claimed_id = event.event_id().expect("canonical event ID");
    let signature = Secp256k1::new().sign_schnorr_no_aux_rand(
        &Message::from_digest(*claimed_id.as_bytes()),
        &keypair(actor),
    );
    SignedEvent {
        event,
        claimed_id,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    }
}

async fn send_signed_restriction_command<S>(
    ingress: &collab::nostr::event_ingest::NostrEventIngress<S>,
    tenant: &TenantContext,
    actor: &collaboration_domain::AuthenticatedPrincipal,
    advertised_role: collaboration_domain::MembershipRole,
    source: &SignedEvent,
    now: u64,
) -> anyhow::Result<collab::nostr::event_ingest::NostrEventIngestOutcome>
where
    S: collab::collaboration_command::DomainCommandSink<
            collab::nostr::event_ingest::NostrEventCommand,
        >,
{
    use collaboration_domain::{
        AggregateId, AggregateVersion, AuthorizationAction, AuthorizationRequest,
        AuthorizationResource, AuthorizationResourceKind, AuthorizationScope, CommunityMembership,
        MembershipStatus,
    };
    let scope = AuthorizationScope::new("messages:write")?;
    let admission =
        collab::tenant_admission::AuthorizedRpcRequest::authorize(&AuthorizationRequest {
            tenant,
            principal: actor,
            required_scope: &scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: tenant.community_id(),
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(tenant.community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: tenant.community_id(),
                principal_id: actor.principal_id(),
                role: advertised_role,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: now * 1000,
        })
        .map_err(|_| anyhow::anyhow!("fixture transport admission denied"))?;
    let frame = serde_json::json!(["EVENT", {
        "id": source.claimed_id.to_hex(), "pubkey": source.event.public_key.to_hex(), "created_at": source.event.created_at,
        "kind": source.event.kind, "tags": source.event.tags, "content": source.event.content, "sig": source.signature.to_hex(),
    }]);
    Ok(ingress
        .handle_frame(admission, &serde_json::to_string(&frame)?, now)
        .await?)
}

async fn restriction_journey_snapshot(
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
) -> anyhow::Result<serde_json::Value> {
    Ok(sqlx::query_scalar(r#"
SELECT jsonb_build_object(
 'profiles', (SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY author_public_key),'[]'::jsonb) FROM public.collaboration_identity_profiles AS row WHERE community_id=$1),
 'events', (SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY event_id),'[]'::jsonb) FROM public.collaboration_events AS row WHERE community_id=$1),
 'actions', (SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY operation_id),'[]'::jsonb) FROM public.collaboration_moderation_actions AS row WHERE community_id=$1),
 'entries', (SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY sequence),'[]'::jsonb) FROM public.collaboration_audit_entries AS row WHERE community_id=$1),
 'heads', (SELECT COALESCE(jsonb_agg(to_jsonb(row)),'[]'::jsonb) FROM public.collaboration_audit_heads AS row WHERE community_id=$1),
 'restrictions', (SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY target_subject_principal_id, restriction_version),'[]'::jsonb) FROM public.collaboration_moderation_restriction_versions AS row WHERE community_id=$1),
 'members', (SELECT COALESCE(jsonb_agg(to_jsonb(row) ORDER BY principal_id),'[]'::jsonb) FROM public.collaboration_community_memberships AS row WHERE community_id=$1))
"#).bind(tenant.community_id().as_uuid()).fetch_one(pool).await?)
}

async fn exercise_signed_absent_restriction_journey(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collab::nostr::event_ingest::NostrEventIngestStatus;
    use collaboration_domain::MembershipRole;
    use nostr_compat::generated_kinds::{
        KIND_MODERATION_BAN, KIND_MODERATION_TIMEOUT, KIND_MODERATION_UNBAN,
        KIND_MODERATION_UNTIMEOUT,
    };
    let own = tenant(173);
    let foreign = tenant(174);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    sqlx::query("INSERT INTO public.collaboration_communities (community_id,host,lifecycle_state,aggregate_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,'active',1,'zed',$2,now(),now(),now())")
        .bind(own.community_id().as_uuid()).bind("signed-absent-restriction.example.test").execute(pool).await?;
    let owner =
        authenticate_registration(repository, &own, &signed_registration(201, None, now, 201))
            .await?;
    let admin =
        authenticate_registration(repository, &own, &signed_registration(202, None, now, 202))
            .await?;
    let member =
        authenticate_registration(repository, &own, &signed_registration(203, None, now, 203))
            .await?;
    for (actor, role) in [(&owner, "owner"), (&admin, "admin"), (&member, "member")] {
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,'active',1,now(),now(),'zed',$4,now())")
            .bind(own.community_id().as_uuid()).bind(actor.principal_id().as_uuid()).bind(role).bind(actor.principal_id().to_string()).execute(pool).await?;
    }
    let sink = collab::admin::moderation::NostrRestrictionCommandSink::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?;
    let ingress = collab::nostr::event_ingest::NostrEventIngress::new(
        sink,
        collab::nostr::ingress::NostrIngressDeployment::InProcess,
    );
    let target_key = NostrPublicKey::from_bytes(*public_key(204).as_bytes());
    anyhow::ensure!(
        repository
            .current_subject_by_public_key(&own, target_key)
            .await?
            .is_none(),
        "signed-ban target was preclassified or registered"
    );
    let before = restriction_journey_snapshot(pool, &own).await?;
    for (actor, signer, target, kind, nonce) in [
        (&member, 203, 204, KIND_MODERATION_BAN, 1),
        (&admin, 202, 201, KIND_MODERATION_BAN, 2),
        (&admin, 202, 202, KIND_MODERATION_TIMEOUT, 3),
    ] {
        let event = signed_restriction_command(
            signer,
            kind,
            target,
            now,
            nonce,
            (kind == KIND_MODERATION_TIMEOUT).then_some(now + 600),
        );
        anyhow::ensure!(
            send_signed_restriction_command(
                &ingress,
                &own,
                actor,
                MembershipRole::Owner,
                &event,
                now
            )
            .await?
            .status()
                == NostrEventIngestStatus::Unauthorized,
            "caller-provided admission role bypassed current signed-command policy"
        );
        anyhow::ensure!(
            restriction_journey_snapshot(pool, &own).await? == before,
            "denied target policy leaked profile/audit/restriction authority"
        );
    }
    let ban = signed_restriction_command(202, KIND_MODERATION_BAN, 204, now, 4, None);
    anyhow::ensure!(
        send_signed_restriction_command(&ingress, &own, &admin, MembershipRole::Admin, &ban, now)
            .await?
            .status()
            == NostrEventIngestStatus::Accepted,
        "authorized admin could not ban a physically absent signed-key target"
    );
    let observed = repository
        .current_subject_by_public_key(&own, target_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("signed admin target observation missing"))?;
    anyhow::ensure!(
        observed.status()
            == collab::identity::profile_repository::IdentitySubjectStatus::ObservedOnly
            && repository
                .current_registration_by_public_key(&own, target_key)
                .await?
                .is_none(),
        "admin target observation became registration/classification"
    );
    let target_memberships: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2")
        .bind(own.community_id().as_uuid()).bind(observed.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        target_memberships == 0,
        "absent-key restriction fabricated target membership"
    );
    let reason: Option<String> = sqlx::query_scalar("SELECT public_reason FROM public.collaboration_moderation_actions WHERE community_id=$1 AND source_record_id=$2")
        .bind(own.community_id().as_uuid()).bind(ban.claimed_id.to_hex()).fetch_one(pool).await?;
    anyhow::ensure!(
        reason.as_deref() == Some("disposable signed restriction"),
        "moderation reason came from content instead of actual Buzz reason tag"
    );
    let accepted = restriction_journey_snapshot(pool, &own).await?;
    anyhow::ensure!(
        send_signed_restriction_command(&ingress, &own, &admin, MembershipRole::Admin, &ban, now)
            .await?
            .status()
            == NostrEventIngestStatus::Duplicate,
        "signed command retry lost its canonical original operation/version"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == accepted,
        "retry rewrote original event/projection/audit/restriction"
    );
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    anyhow::ensure!(
        authenticate_with_structural_gate(
            repository,
            pool,
            &own,
            &signed_registration(204, None, now, 204),
            calls.clone()
        )
        .await
            == Err(collab::nostr::auth::NostrAuthenticationError::Banned),
        "direct ban of absent observed key did not precede authentication delegate"
    );
    anyhow::ensure!(
        calls.load(std::sync::atomic::Ordering::SeqCst) == 0
            && restriction_journey_snapshot(pool, &own).await? == accepted,
        "direct banned AUTH delegated or changed immutable observed target"
    );
    let new_agent = signed_registration(205, Some(204), now, 205);
    anyhow::ensure!(
        authenticate_with_structural_gate(repository, pool, &own, &new_agent, calls.clone()).await
            == Err(collab::nostr::auth::NostrAuthenticationError::Banned),
        "actual presented-owner ban did not precede registration delegate"
    );
    anyhow::ensure!(
        calls.load(std::sync::atomic::Ordering::SeqCst) == 0
            && repository
                .current_subject_by_public_key(
                    &own,
                    NostrPublicKey::from_bytes(*public_key(205).as_bytes())
                )
                .await?
                .is_none(),
        "owner-banned AUTH leaked fresh agent observation/classification"
    );
    let unban = signed_restriction_command(202, KIND_MODERATION_UNBAN, 204, now, 5, None);
    anyhow::ensure!(
        send_signed_restriction_command(&ingress, &own, &admin, MembershipRole::Admin, &unban, now)
            .await?
            .status()
            == NostrEventIngestStatus::Accepted,
        "admin could not lift absent-target ban through canonical history"
    );
    let authenticated = authenticate_with_structural_gate(
        repository,
        pool,
        &own,
        &signed_registration(204, None, now, 206),
        calls.clone(),
    )
    .await?;
    anyhow::ensure!(
        authenticated.principal_id() == observed.principal_id()
            && repository
                .current_subject_by_public_key(&own, target_key)
                .await?
                .ok_or_else(|| anyhow::anyhow!("promoted target missing"))?
                .profile_id()
                == observed.profile_id(),
        "real AUTH promotion replaced admin-observed stable IDs"
    );
    let timeout_admin =
        signed_restriction_command(201, KIND_MODERATION_TIMEOUT, 202, now, 6, Some(now + 600));
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &owner,
            MembershipRole::Owner,
            &timeout_admin,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "owner could not timeout protected admin"
    );
    let mut self_lift =
        signed_restriction_command(202, KIND_MODERATION_UNTIMEOUT, 202, now, 7, None).event;
    self_lift
        .tags
        .push(vec!["expiration".into(), "not-a-timestamp".into()]);
    let self_lift = sign_restriction_event(202, self_lift);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &self_lift,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "timed-out admin cannot perform actual global control-plane self-lift"
    );
    let ban_admin = signed_restriction_command(201, KIND_MODERATION_BAN, 202, now, 8, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &owner,
            MembershipRole::Owner,
            &ban_admin,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "owner could not ban admin"
    );
    let banned = restriction_journey_snapshot(pool, &own).await?;
    let denied_lift = signed_restriction_command(202, KIND_MODERATION_UNBAN, 202, now, 9, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &denied_lift,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Unauthorized,
        "active ban received timeout-style control-plane exemption"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == banned,
        "banned control-plane command altered state"
    );
    let owner_lift = signed_restriction_command(201, KIND_MODERATION_UNBAN, 202, now, 10, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &owner,
            MembershipRole::Owner,
            &owner_lift,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "owner cannot lift protected-admin ban"
    );
    let before_empty_expiration = restriction_journey_snapshot(pool, &own).await?;
    let mut missing_timeout_expiration =
        signed_restriction_command(202, KIND_MODERATION_TIMEOUT, 213, now, 24, None).event;
    missing_timeout_expiration
        .tags
        .push(vec!["expiration".into()]);
    let missing_timeout_expiration = sign_restriction_event(202, missing_timeout_expiration);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &missing_timeout_expiration,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Unauthorized,
        "timeout accepted a valueless expiration tag"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == before_empty_expiration,
        "missing timeout expiration leaked subject or audit state"
    );
    for (target, nonce, reason) in [
        (249, 27, format!("{}x", "é".repeat(512))),
        (250, 28, "é".repeat(9000)),
    ] {
        let mut long_reason =
            signed_restriction_command(202, KIND_MODERATION_BAN, target, now, nonce, None).event;
        for tag in &mut long_reason.tags {
            if tag.first().is_some_and(|name| name == "reason") {
                if let Some(value) = tag.get_mut(1) {
                    *value = reason.clone();
                }
            }
        }
        let long_reason = sign_restriction_event(202, long_reason);
        anyhow::ensure!(
            send_signed_restriction_command(
                &ingress,
                &own,
                &admin,
                MembershipRole::Admin,
                &long_reason,
                now,
            )
            .await?
            .status()
                == NostrEventIngestStatus::Accepted,
            "signed restriction reason within event budget was rejected by obsolete 1024-byte cap"
        );
        let retained: (String, Uuid, String, String) = sqlx::query_as(
            "SELECT public_reason,operation_id,source_record_id,integrity_value FROM public.collaboration_moderation_actions WHERE community_id=$1 AND source_system='nostr' AND source_record_id=$2"
        ).bind(own.community_id().as_uuid()).bind(long_reason.claimed_id.to_hex()).fetch_one(pool).await?;
        anyhow::ensure!(
            retained.0 == reason
                && retained.2 == long_reason.claimed_id.to_hex()
                && retained.3 == retained.2,
            "signed restriction reason or source was truncated/replaced"
        );
        let audit = collab::audit::repository::AuditRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?
        .load_operation(
            &own,
            collaboration_domain::OperationId::from_uuid(retained.1),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("signed restriction audit missing"))?;
        audit.verify()?;
        let fields = serde_json::to_string(audit.record().fields())?;
        anyhow::ensure!(
            !fields.contains(&reason) && fields.len() < 16 * 1024,
            "public reason leaked into bounded canonical audit payload"
        );
        let subject = repository
            .current_subject_by_public_key(
                &own,
                NostrPublicKey::from_bytes(*public_key(target).as_bytes()),
            )
            .await?
            .ok_or_else(|| anyhow::anyhow!("genuine signed moderation observation missing"))?;
        anyhow::ensure!(
            repository
                .current_registration_by_public_key(
                    &own,
                    NostrPublicKey::from_bytes(*public_key(target).as_bytes())
                )
                .await?
                .is_none(),
            "long reason created registration authority"
        );
        let source: serde_json::Value = sqlx::query_scalar(
            "SELECT observation_proof FROM public.collaboration_identity_profiles WHERE community_id=$1 AND principal_id=$2"
        ).bind(own.community_id().as_uuid()).bind(subject.principal_id().as_uuid()).fetch_one(pool).await?;
        anyhow::ensure!(
            serde_json::to_string(&source)?.contains(&reason),
            "actual signed reason source was not retained"
        );
        // Constraint evaluation precedes uniqueness/FK checks; only the source budget is varied and the owned transaction always rolls back.
        for (source_system, action_kind) in [
            ("'zed'", "action_kind"),
            ("source_system", "'archive_identity'"),
        ] {
            let mut transaction = pool.begin().await?;
            let attempted = sqlx::query(&format!(
                "INSERT INTO public.collaboration_moderation_actions SELECT community_id,action_id,operation_id,actor_principal_id,{action_kind},record_kind,record_id,target_principal_id,target_event_id,reason_code,public_reason,private_reason,occurred_at,{source_system},source_record_id,source_version,source_observed_at,integrity_algorithm,integrity_value,recorded_at FROM public.collaboration_moderation_actions WHERE community_id=$1 AND operation_id=$2"
            )).bind(own.community_id().as_uuid()).bind(retained.1).execute(&mut *transaction).await;
            transaction.rollback().await?;
            anyhow::ensure!(
                matches!(attempted, Err(sqlx::Error::Database(ref error)) if error.code().as_deref()==Some("23514")
                && error.constraint()==Some("collaboration_moderation_actions_public_reason_check")),
                "general moderation reason budget was weakened"
            );
        }
    }
    let before_reason_boundaries = restriction_journey_snapshot(pool, &own).await?;
    let mut too_large =
        signed_restriction_command(202, KIND_MODERATION_BAN, 251, now, 29, None).event;
    for tag in &mut too_large.tags {
        if tag.first().is_some_and(|name| name == "reason") {
            if let Some(value) = tag.get_mut(1) {
                *value = "r".repeat(collab::nostr::MAX_NOSTR_FRAME_BYTES);
            }
        }
    }
    let too_large = sign_restriction_event(202, too_large);
    let rejected = send_signed_restriction_command(
        &ingress,
        &own,
        &admin,
        MembershipRole::Admin,
        &too_large,
        now,
    )
    .await;
    anyhow::ensure!(
        matches!(rejected, Err(ref error) if error.downcast_ref::<collab::nostr::event_ingest::NostrEventFrameError>()
        .is_some_and(|error| error.kind()==collab::nostr::event_ingest::NostrEventFrameErrorKind::FrameTooLarge)),
        "global signed EVENT frame budget was weakened"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == before_reason_boundaries,
        "oversized signed EVENT leaked subject/event/audit/restriction state"
    );
    for fields in [
        serde_json::Value::Array((0..33).map(|index| serde_json::json!({
            "name": format!("budget_{index}"), "value": {"type": "boolean", "value": true}
        })).collect()),
        serde_json::json!([{"name": "budget", "value": {"type": "identifier", "value": "x".repeat(16 * 1024 + 1)}}]),
    ] {
        let attempted = sqlx::query(
            "INSERT INTO public.collaboration_audit_entries (community_id,sequence,entry_hash,previous_hash,operation_id,action,actor_principal_id,outcome,occurred_at_millis,fields,bridge_source,bridge_source_sequence,bridge_source_head,recorded_at) SELECT community_id,sequence,entry_hash,previous_hash,operation_id,action,actor_principal_id,outcome,occurred_at_millis,$2,bridge_source,bridge_source_sequence,bridge_source_head,recorded_at FROM public.collaboration_audit_entries WHERE community_id=$1 LIMIT 1"
        ).bind(own.community_id().as_uuid()).bind(fields.clone()).execute(pool).await;
        anyhow::ensure!(matches!(attempted, Err(sqlx::Error::Database(ref error)) if error.code().as_deref()==Some("23514")
            && error.constraint()==Some("collaboration_audit_entries_fields_check")),
            "global canonical audit SQL budget was weakened");
        // Expose a corrupt read in this uniquely owned database, then restore the original immutable table before checking the actual signed command result.
        let mut mount = pool.begin().await?;
        sqlx::raw_sql("ALTER TABLE public.collaboration_audit_entries RENAME TO signed_reason_fixture_audit_original").execute(&mut *mount).await?;
        sqlx::raw_sql(&format!("CREATE VIEW public.collaboration_audit_entries AS SELECT community_id,sequence,previous_hash,previous_sequence,operation_id,action,actor_principal_id,outcome,occurred_at_millis,bridge_source,bridge_source_sequence,bridge_source_head,recorded_at,entry_hash,CASE WHEN community_id='{}'::uuid THEN '{}'::jsonb ELSE fields END AS fields FROM public.signed_reason_fixture_audit_original",
            own.community_id().as_uuid(), serde_json::to_string(&fields)?)).execute(&mut *mount).await?;
        mount.commit().await?;
        let source = signed_restriction_command(202, KIND_MODERATION_UNBAN, 249, now, 30, None);
        let denied = send_signed_restriction_command(&ingress, &own, &admin, MembershipRole::Admin, &source, now).await;
        let mut restore = pool.begin().await?;
        sqlx::raw_sql("DROP VIEW public.collaboration_audit_entries; ALTER TABLE public.signed_reason_fixture_audit_original RENAME TO collaboration_audit_entries").execute(&mut *restore).await?;
        restore.commit().await?;
        anyhow::ensure!(denied?.status()==NostrEventIngestStatus::Unavailable,
            "signed restriction consumed out-of-budget canonical audit history");
        anyhow::ensure!(restriction_journey_snapshot(pool, &own).await? == before_reason_boundaries,
            "invalid audit budget leaked signed restriction/event/subject mutation");
    }
    let mut rollback = pool.begin().await?;
    let refused = sqlx::raw_sql(SIGNED_REASONS_DOWN)
        .execute(&mut *rollback)
        .await;
    rollback.rollback().await?;
    anyhow::ensure!(
        matches!(refused, Err(sqlx::Error::Database(ref error)) if error.code().as_deref()==Some("23514")),
        "signed reason rollback discarded long retained evidence"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == before_reason_boundaries,
        "refused signed reason rollback changed retained state"
    );
    let restored_lift = signed_restriction_command(202, KIND_MODERATION_UNBAN, 249, now, 30, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &restored_lift,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "signed restriction did not recover after restoring original bounded audit history"
    );
    let mut unusual_ban =
        signed_restriction_command(202, KIND_MODERATION_BAN, 213, now, 25, None).event;
    unusual_ban
        .tags
        .insert(0, vec!["p".into(), "not-a-public-key".into()]);
    unusual_ban.tags.push(vec!["expiration".into()]);
    unusual_ban
        .tags
        .push(vec!["h".into(), foreign.community_id().to_string()]);
    for tag in &mut unusual_ban.tags {
        if tag.first().is_some_and(|name| name == "reason") {
            if let Some(reason) = tag.get_mut(1) {
                *reason = "é".repeat(512);
            }
        }
    }
    let unusual_ban = sign_restriction_event(202, unusual_ban);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &unusual_ban,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "signed ban did not preserve global h/first-valid p/valueless-expiration or 1024-byte reason behavior"
    );
    let unusual_target = repository
        .current_subject_by_public_key(
            &own,
            NostrPublicKey::from_bytes(*public_key(213).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("first-valid signed target observation was not retained"))?;
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(
                &own,
                NostrPublicKey::from_bytes(*public_key(213).as_bytes())
            )
            .await?
            .is_none(),
        "signed first-valid target proof was promoted to registration"
    );
    anyhow::ensure!(
        unusual_target.principal_id().as_uuid() != owner.principal_id().as_uuid(),
        "malformed first p tag changed selected target authority"
    );
    let mut unusual_lift =
        signed_restriction_command(202, KIND_MODERATION_UNBAN, 213, now, 26, None).event;
    unusual_lift
        .tags
        .push(vec!["expiration".into(), "not-a-timestamp".into()]);
    let unusual_lift = sign_restriction_event(202, unusual_lift);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &unusual_lift,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "signed lift parsed irrelevant malformed expiration instead of completing its journey"
    );
    let before_invalid = restriction_journey_snapshot(pool, &own).await?;
    let mut tampered = signed_restriction_command(202, KIND_MODERATION_BAN, 206, now, 11, None);
    tampered.signature = EventSignature::from_hex(&"00".repeat(64))?;
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &tampered,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Invalid,
        "tampered signature reached target observation/audit writer"
    );
    let stale = signed_restriction_command(202, KIND_MODERATION_BAN, 206, now - 121, 12, None);
    anyhow::ensure!(
        send_signed_restriction_command(&ingress, &own, &admin, MembershipRole::Admin, &stale, now)
            .await?
            .status()
            == NostrEventIngestStatus::Unauthorized,
        "actual moderation 120-second freshness fence was replaced by ingress 15-minute allowance"
    );
    let wrong_author = signed_restriction_command(201, KIND_MODERATION_BAN, 206, now, 13, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &wrong_author,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Unauthorized,
        "authenticated actor and signed command author diverged"
    );
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &foreign,
            &admin,
            MembershipRole::Admin,
            &wrong_author,
            now
        )
        .await
        .is_err(),
        "cross-tenant principal reached signed sink"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == before_invalid,
        "invalid signature/freshness/tenant command leaked state"
    );
    let timeout_observed =
        signed_restriction_command(202, KIND_MODERATION_TIMEOUT, 204, now, 14, Some(now + 600));
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &timeout_observed,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "admin cannot timeout independent observed subject"
    );
    let delegated_agent = authenticate_with_structural_gate(
        repository,
        pool,
        &own,
        &signed_registration(205, Some(204), now, 207),
        calls.clone(),
    )
    .await?;
    anyhow::ensure!(
        matches!(
            delegated_agent.kind(),
            collaboration_domain::AuthenticatedPrincipalKind::OwnerAttestedAgent { .. }
        ),
        "actual owner proof did not retain canonical agent classification"
    );
    anyhow::ensure!(
        calls.load(std::sync::atomic::Ordering::SeqCst) == 2,
        "owner timeout incorrectly cascaded to actual agent AUTH"
    );
    let owner_self_ban = signed_restriction_command(201, KIND_MODERATION_BAN, 201, now, 15, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &owner,
            MembershipRole::Owner,
            &owner_self_ban,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "actual owner self-target policy was replaced by legacy Made self restriction"
    );
    let admin_lift_owner =
        signed_restriction_command(202, KIND_MODERATION_UNBAN, 201, now, 16, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &admin_lift_owner,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Accepted,
        "admin lifting an owner restriction inherited apply-only target protection"
    );
    let altered_ingress = collab::nostr::event_ingest::NostrEventIngress::new(
        AlteredRestrictionOperation(collab::admin::moderation::NostrRestrictionCommandSink::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?),
        collab::nostr::ingress::NostrIngressDeployment::InProcess,
    );
    let before_operation_fault = restriction_journey_snapshot(pool, &own).await?;
    let exact_signature_wrong_operation =
        signed_restriction_command(202, KIND_MODERATION_BAN, 206, now, 23, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &altered_ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &exact_signature_wrong_operation,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Unauthorized,
        "genuinely verified signed payload authorized a different caller-supplied operation UUID"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == before_operation_fault,
        "signed-wrapper operation mismatch leaked profile/audit/restriction state"
    );
    drop(altered_ingress);
    let before_atomic_failure = restriction_journey_snapshot(pool, &own).await?;
    let atomic = signed_restriction_command(202, KIND_MODERATION_BAN, 206, now, 17, None);
    let atomic_constraint = format!(
        "ALTER TABLE public.collaboration_moderation_restriction_versions ADD CONSTRAINT disposable_restriction_writer_failure CHECK (source_record_id <> '{}') NOT VALID",
        atomic.claimed_id.to_hex()
    );
    sqlx::raw_sql(&atomic_constraint).execute(pool).await?;
    let failed_write = send_signed_restriction_command(
        &ingress,
        &own,
        &admin,
        MembershipRole::Admin,
        &atomic,
        now,
    )
    .await;
    let cleanup = sqlx::raw_sql("ALTER TABLE public.collaboration_moderation_restriction_versions DROP CONSTRAINT disposable_restriction_writer_failure").execute(pool).await;
    cleanup?;
    anyhow::ensure!(
        failed_write?.status() == NostrEventIngestStatus::Unavailable,
        "forced final history write failure did not propagate infrastructure failure"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == before_atomic_failure,
        "failed canonical history write leaked prepared UUID/profile, audit head/action or restriction retirement"
    );
    exercise_signed_ban_during_auth_delegate(repository, pool, &own, &ingress, &admin, now).await?;
    sqlx::query("UPDATE public.collaboration_community_memberships SET role='member', membership_version=membership_version+1, updated_at=now() WHERE community_id=$1 AND principal_id=$2")
        .bind(own.community_id().as_uuid()).bind(admin.principal_id().as_uuid()).execute(pool).await?;
    let demoted = restriction_journey_snapshot(pool, &own).await?;
    let demoted_command = signed_restriction_command(202, KIND_MODERATION_BAN, 206, now, 18, None);
    anyhow::ensure!(
        send_signed_restriction_command(
            &ingress,
            &own,
            &admin,
            MembershipRole::Admin,
            &demoted_command,
            now
        )
        .await?
        .status()
            == NostrEventIngestStatus::Unauthorized,
        "stale transport role/version authorized a currently demoted signed-command actor"
    );
    anyhow::ensure!(
        restriction_journey_snapshot(pool, &own).await? == demoted,
        "current-role denial changed target/audit/history state"
    );
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    use sea_orm::TransactionTrait as _;
    let transaction = connection.begin().await?;
    let restriction = collab::admin::moderation::current_restriction_in_transaction(
        &transaction,
        &own,
        observed.principal_id(),
    )
    .await?
    .ok_or_else(|| anyhow::anyhow!("canonical absent-target restriction disappeared"))?;
    anyhow::ensure!(
        restriction.fields().transitions.len() == 3 && restriction.fields().version.get() == 4,
        "absent-target ban/lift/timeout lost original audit-attributed history"
    );
    transaction.rollback().await?;
    let mut rollback = pool.begin().await?;
    let refusal = sqlx::raw_sql(MODERATION_SUBJECT_DOWN)
        .execute(&mut *rollback)
        .await;
    anyhow::ensure!(
        matches!(refusal, Err(sqlx::Error::Database(ref error)) if error.code().as_deref() == Some("23514")),
        "subject migration rollback discarded retained signed restriction provenance"
    );
    rollback.rollback().await?;
    drop(ingress);
    connection.close().await?;
    Ok(())
}

async fn exercise_signed_ban_auth_race(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collab::nostr::event_ingest::NostrEventIngestStatus;
    use collaboration_domain::MembershipRole;
    let own = tenant(175);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    sqlx::query("INSERT INTO public.collaboration_communities (community_id,host,lifecycle_state,aggregate_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,'active',1,'zed',$2,now(),now(),now())")
        .bind(own.community_id().as_uuid()).bind("signed-ban-auth-race.example.test").execute(pool).await?;
    let admin =
        authenticate_registration(repository, &own, &signed_registration(211, None, now, 211))
            .await?;
    sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'admin','active',1,now(),now(),'zed',$3,now())")
        .bind(own.community_id().as_uuid()).bind(admin.principal_id().as_uuid()).bind(admin.principal_id().to_string()).execute(pool).await?;
    let ingress = collab::nostr::event_ingest::NostrEventIngress::new(
        collab::admin::moderation::NostrRestrictionCommandSink::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?,
        collab::nostr::ingress::NostrIngressDeployment::InProcess,
    );
    let ban = signed_restriction_command(
        211,
        nostr_compat::generated_kinds::KIND_MODERATION_BAN,
        212,
        now,
        21,
        None,
    );
    let auth = signed_registration(212, None, now, 212);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(3));
    let first = async {
        barrier.wait().await;
        send_signed_restriction_command(&ingress, &own, &admin, MembershipRole::Admin, &ban, now)
            .await
    };
    let second = async {
        barrier.wait().await;
        send_signed_restriction_command(&ingress, &own, &admin, MembershipRole::Admin, &ban, now)
            .await
    };
    let authenticate = async {
        barrier.wait().await;
        authenticate_with_structural_gate(repository, pool, &own, &auth, calls.clone()).await
    };
    let (first, second, authenticated) =
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            tokio::join!(first, second, authenticate)
        })
        .await
        .map_err(|_| anyhow::anyhow!("actual signed ban/retry/AUTH transaction race deadlocked"))?;
    let statuses = [first?.status(), second?.status()];
    anyhow::ensure!(
        statuses
            .iter()
            .filter(|status| **status == NostrEventIngestStatus::Accepted)
            .count()
            == 1
            && statuses
                .iter()
                .filter(|status| **status == NostrEventIngestStatus::Duplicate)
                .count()
                == 1,
        "concurrent signed restriction retry lost original canonical application/version"
    );
    let subject = repository
        .current_subject_by_public_key(
            &own,
            NostrPublicKey::from_bytes(*public_key(212).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("ban/AUTH race lost canonical subject"))?;
    match authenticated {
        Ok(principal) => anyhow::ensure!(
            principal.principal_id() == subject.principal_id(),
            "AUTH-before-ban allocated a second identity authority"
        ),
        Err(collab::nostr::auth::NostrAuthenticationError::Banned) => {}
        Err(error) => anyhow::bail!("ban/AUTH race returned unexpected error: {error}"),
    }
    let before_calls = calls.load(std::sync::atomic::Ordering::SeqCst);
    anyhow::ensure!(
        before_calls <= 1,
        "ban/AUTH race delegated more than its one actual AUTH"
    );
    anyhow::ensure!(
        authenticate_with_structural_gate(
            repository,
            pool,
            &own,
            &signed_registration(212, None, now, 213),
            calls.clone()
        )
        .await
            == Err(collab::nostr::auth::NostrAuthenticationError::Banned),
        "post-commit AUTH observed stale absence after real signed ban"
    );
    anyhow::ensure!(
        calls.load(std::sync::atomic::Ordering::SeqCst) == before_calls,
        "post-commit banned AUTH reached registration/current-membership delegate"
    );
    let identities: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key=$2")
        .bind(own.community_id().as_uuid()).bind(public_key(212).as_bytes().to_vec()).fetch_one(pool).await?;
    let restrictions: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_moderation_restriction_versions WHERE community_id=$1 AND target_subject_principal_id=$2")
        .bind(own.community_id().as_uuid()).bind(subject.principal_id().as_uuid()).fetch_one(pool).await?;
    let actions: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_moderation_actions WHERE community_id=$1",
    )
    .bind(own.community_id().as_uuid())
    .fetch_one(pool)
    .await?;
    let members: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2")
        .bind(own.community_id().as_uuid()).bind(subject.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        identities == 1 && restrictions == 1 && actions == 1 && members == 0,
        "concurrent ban/AUTH forked subject/history or fabricated membership"
    );
    Ok(())
}

struct BlockedGenuineRegistrationResolver {
    repository: Arc<IdentityProfileRepository>,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl collab::nostr::auth::NostrPrincipalResolver for BlockedGenuineRegistrationResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &collab::nostr::auth::VerifiedNip42Identity,
    ) -> Result<
        collaboration_domain::AuthenticatedPrincipal,
        collab::nostr::auth::NostrPrincipalResolutionError,
    > {
        let principal = collab::nostr::auth::NostrPrincipalResolver::resolve(
            self.repository.as_ref(),
            tenant,
            identity,
        )
        .await?;
        self.entered.notify_one();
        self.release.notified().await;
        Ok(principal)
    }
}

async fn exercise_signed_ban_during_auth_delegate(
    repository: &Arc<IdentityProfileRepository>,
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    ingress: &collab::nostr::event_ingest::NostrEventIngress<
        collab::admin::moderation::NostrRestrictionCommandSink,
    >,
    admin: &collaboration_domain::AuthenticatedPrincipal,
    now: u64,
) -> anyhow::Result<()> {
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let source = signed_registration(214, None, now, 214);
    let resolver = collab::nostr::auth::NostrStructuralBanResolver::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
        BlockedGenuineRegistrationResolver {
            repository: repository.clone(),
            entered: entered.clone(),
            release: release.clone(),
        },
    )?;
    let mut authenticator = collab::nostr::auth::NostrConnectionAuthenticator::new(
        tenant.clone(),
        "wss://relay.example.test",
        collab::nostr::auth::NostrAuthChallenge::parse(hex::encode([214_u8; 32]))?,
        now,
        RegistrationReplay::default(),
        resolver,
    )?;
    let actual_auth = async { authenticator.authenticate(&source, now).await.cloned() };
    let commit_ban = async {
        entered.notified().await;
        let ban = signed_restriction_command(
            202,
            nostr_compat::generated_kinds::KIND_MODERATION_BAN,
            214,
            now,
            22,
            None,
        );
        let result = send_signed_restriction_command(
            ingress,
            tenant,
            admin,
            collaboration_domain::MembershipRole::Admin,
            &ban,
            now,
        )
        .await;
        release.notify_one();
        let outcome = result?;
        anyhow::ensure!(
            outcome.status() == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
            "genuine signed ban did not commit while actual AUTH delegate was suspended"
        );
        Ok::<(), anyhow::Error>(())
    };
    let (authenticated, ban) = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        tokio::join!(actual_auth, commit_ban)
    })
    .await
    .map_err(|_| anyhow::anyhow!("single-connection final AUTH/ban fence deadlocked"))?;
    ban?;
    anyhow::ensure!(
        authenticated == Err(collab::nostr::auth::NostrAuthenticationError::Banned),
        "AUTH succeeded after a signed ban committed during its genuine registration delegate"
    );
    anyhow::ensure!(
        matches!(
            authenticator.state(),
            collab::nostr::auth::NostrAuthenticationState::Failed
        ),
        "final restriction rejection left native authenticator authenticated"
    );
    let subject = repository
        .current_subject_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(214).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("blocked genuine registration lost canonical subject"))?;
    let members: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(subject.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        members == 0,
        "failed final AUTH decision fabricated membership"
    );
    Ok(())
}

struct AlteredRestrictionOperation(collab::admin::moderation::NostrRestrictionCommandSink);

#[async_trait::async_trait]
impl
    collab::collaboration_command::DomainCommandSink<collab::nostr::event_ingest::NostrEventCommand>
    for AlteredRestrictionOperation
{
    async fn submit(
        &self,
        command: collab::collaboration_command::DomainCommand<
            collab::nostr::event_ingest::NostrEventCommand,
        >,
    ) -> Result<
        collab::collaboration_command::DomainCommandReceipt,
        collab::collaboration_command::DomainCommandSubmissionError,
    > {
        let altered = collab::collaboration_command::DomainCommand::new(
            collaboration_domain::OperationId::from_uuid(Uuid::new_v4()),
            command.tenant().clone(),
            command.principal().clone(),
            command.expected_version(),
            command.predecessor(),
            command.originating_adapter(),
            command.into_payload(),
        );
        collab::collaboration_command::DomainCommandSink::submit(&self.0, altered).await
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_authored_metadata_actual_signed_heads_without_classification()
-> anyhow::Result<()> {
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL")?;
    let mut url = url::Url::parse(&base_url)?;
    anyhow::ensure!(
        matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback PostgreSQL required"
    );
    url.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let url = url.to_string();
    sqlx::Postgres::create_database(&url).await?;
    let result = exercise_authored_metadata_database(&url).await;
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(anyhow::anyhow!(
            "authored metadata database cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "authored metadata failed: {error:#}; database cleanup failed: {cleanup}"
        )),
    }
}

async fn exercise_authored_metadata_database(url: &str) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(url)
        .await?;
    let result = async {
        for migration in [
            include_str!("../migrations/20260820000100_collaboration_events.up.sql"),
            include_str!("../migrations/20260820000200_collaboration_event_heads.up.sql"),
            UP,
            REGISTRATION_UP,
            OBSERVATION_UP,
            CHANNELS_UP,
            AUDIT_UP,
            MODERATION_UP,
            MODERATION_SUBJECT_UP,
            ACCEPTED_NAMES_UP,
            SIGNED_REASONS_UP,
            MODERATION_RETIREMENT_UP,
        ] {
            sqlx::raw_sql(migration).execute(&pool).await?;
        }
        let repository = Arc::new(IdentityProfileRepository::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?);
        let events = EventRepository::new(sea_orm::DatabaseConnection::from(pool.clone()))?;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(20),
            exercise_authored_metadata_reader(&repository, &events, &pool),
        )
        .await
        .map_err(|_| anyhow::anyhow!("authored metadata supplied-transaction pool1 deadlocked"))?;
        events.into_connection().close().await?;
        Arc::try_unwrap(repository)
            .map_err(|_| anyhow::anyhow!("authored metadata fixture leaked a reference"))?
            .into_connection()
            .close()
            .await?;
        result
    }
    .await;
    pool.close().await;
    result
}

async fn exercise_authored_metadata_reader(
    repository: &Arc<IdentityProfileRepository>,
    events: &EventRepository,
    pool: &sqlx::PgPool,
) -> anyhow::Result<()> {
    use collab::identity::profile_repository::IdentitySubjectStatus;
    use sea_orm::{ConnectionTrait, DatabaseBackend, Statement, TransactionTrait};
    let own = tenant(1201);
    let foreign = tenant(1202);
    let key = NostrPublicKey::from_bytes(*public_key(80).as_bytes());
    anyhow::ensure!(
        repository
            .current_authored_metadata_by_public_key(&own, key)
            .await?
            .is_none(),
        "missing profile fabricated empty metadata"
    );
    let signed = signed_profile(
        &own,
        80,
        NOW,
        Vec::new(),
        r#"{"display_name":"Person","name":"alias","picture":"https://example.test/avatar","about":"before","nip05":"claimed@example.invalid"}"#,
    );
    anyhow::ensure!(
        repository.project_profile_event(&own, &signed).await?
            == IdentityProfileProjectionOutcome::Unclassified,
        "ordinary authored metadata inferred immutable Human"
    );
    let metadata = repository
        .current_authored_metadata_by_public_key(&own, key)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "ordinary signed profile metadata remained absent at canonical read boundary"
            )
        })?;
    anyhow::ensure!(
        metadata.community_id() == own.community_id()
            && metadata.metadata().source_author == key
            && metadata.metadata().source_event_id.as_bytes()
                == signed.signed_event().claimed_id.as_bytes()
            && metadata.metadata().source_created_at == NOW
            && metadata.metadata().value.display_name.as_deref() == Some("Person")
            && metadata.metadata().value.name.as_deref() == Some("alias")
            && metadata.metadata().value.about.as_deref() == Some("before"),
        "signed authored metadata attribution or fields changed"
    );
    anyhow::ensure!(
        metadata.subject().is_none()
            && metadata.owner_public_key().is_none()
            && repository
                .current_registration_by_public_key(&own, key)
                .await?
                .is_none(),
        "metadata read allocated/authenticated/classified a subject"
    );
    anyhow::ensure!(
        repository
            .current_authored_metadata_by_public_key(&foreign, key)
            .await?
            .is_none(),
        "foreign tenant saw authored metadata"
    );
    let plain =
        authenticate_registration(repository, &own, &signed_registration(80, None, NOW, 201))
            .await?;
    let registered = repository
        .current_registration_by_public_key(&own, key)
        .await?;
    let read = repository
        .current_authored_metadata_by_public_key(&own, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("registered ordinary metadata hidden"))?;
    anyhow::ensure!(
        read.subject().is_some_and(|subject| subject.status()
            == IdentitySubjectStatus::RegisteredUnclassified
            && subject.principal_id() == plain.principal_id())
            && read.owner_public_key().is_none(),
        "ordinary metadata read changed registration authority"
    );
    let empty = signed_profile(&own, 80, NOW + 1, Vec::new(), "{}");
    repository.project_profile_event(&own, &empty).await?;
    let read = repository
        .current_authored_metadata_by_public_key(&own, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("real empty profile became missing"))?;
    anyhow::ensure!(
        read.metadata().value == collaboration_domain::ProfileMetadata::default()
            && read.metadata().source_event_id.as_bytes()
                == empty.signed_event().claimed_id.as_bytes()
            && repository
                .current_registration_by_public_key(&own, key)
                .await?
                == registered,
        "absolute omissions retained old metadata or changed identity"
    );
    anyhow::ensure!(
        repository.project_profile_event(&own, &signed).await?
            == IdentityProfileProjectionOutcome::Stale,
        "old signed profile replaced current head"
    );
    let tie_a = signed_profile(&own, 80, NOW + 2, Vec::new(), r#"{"name":"tie-a"}"#);
    let tie_b = signed_profile(&own, 80, NOW + 2, Vec::new(), r#"{"name":"tie-b"}"#);
    repository.project_profile_event(&own, &tie_a).await?;
    repository.project_profile_event(&own, &tie_b).await?;
    let expected = if tie_a.signed_event().claimed_id.as_bytes()
        < tie_b.signed_event().claimed_id.as_bytes()
    {
        &tie_a
    } else {
        &tie_b
    };
    let read = repository
        .current_authored_metadata_by_public_key(&own, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("tie head absent"))?;
    anyhow::ensure!(
        read.metadata().source_event_id.as_bytes() == expected.signed_event().claimed_id.as_bytes(),
        "authored reader bypassed canonical lowest-ID tie order"
    );
    let agent = authenticate_registration(
        repository,
        &own,
        &signed_registration(80, Some(81), NOW, 202),
    )
    .await?;
    let classified = repository
        .current_registration_by_public_key(&own, key)
        .await?;
    let read = repository
        .current_authored_metadata_by_public_key(&own, key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("verified agent metadata hidden"))?;
    anyhow::ensure!(
        agent.principal_id() == plain.principal_id()
            && read
                .subject()
                .is_some_and(|subject| subject.principal_id() == plain.principal_id()
                    && subject.status() == IdentitySubjectStatus::ClassifiedProfile)
            && read.owner_public_key()
                == Some(NostrPublicKey::from_bytes(*public_key(81).as_bytes()))
            && read.metadata().source_event_id.as_bytes()
                == expected.signed_event().claimed_id.as_bytes(),
        "later real OA changed ID/head or lost separate verified owner"
    );
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    let transaction = connection.begin().await?;
    anyhow::ensure!(
        repository
            .current_authored_metadata_in_transaction(&transaction, &own, key)
            .await?
            .is_some(),
        "supplied pool1 transaction reader failed"
    );
    anyhow::ensure!(
        matches!(
            repository
                .current_authored_metadata_in_transaction(&transaction, &foreign, key)
                .await,
            Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
        ),
        "supplied transaction changed current tenant setting"
    );
    transaction.rollback().await?;
    let transaction = connection.begin().await?;
    let role = format!("metadata_reader_{}", Uuid::new_v4().simple());
    let rls_result = async {
        transaction
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                format!("CREATE ROLE {role} NOLOGIN NOBYPASSRLS"),
            ))
            .await?;
        transaction
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                format!("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}"),
            ))
            .await?;
        transaction
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                format!("SET LOCAL ROLE {role}"),
            ))
            .await?;
        anyhow::ensure!(
            repository
                .current_authored_metadata_in_transaction(&transaction, &own, key)
                .await?
                .is_some(),
            "nonbypass role lost own signed metadata"
        );
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "SELECT set_config('app.community_id',$1,true)",
                [foreign.community_id().to_string().into()],
            ))
            .await?;
        anyhow::ensure!(
            repository
                .current_authored_metadata_in_transaction(&transaction, &foreign, key)
                .await?
                .is_none(),
            "forced RLS exposed other tenant head"
        );
        anyhow::ensure!(
            matches!(
                repository
                    .current_authored_metadata_in_transaction(&transaction, &own, key)
                    .await,
                Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
            ),
            "reader bypassed foreign mounted tenant"
        );
        Ok::<(), anyhow::Error>(())
    }
    .await;
    let cleanup = transaction.rollback().await;
    match (rls_result, cleanup) {
        (Ok(()), Ok(())) => {}
        (Err(error), Ok(())) => return Err(error),
        (Ok(()), Err(error)) => return Err(error.into()),
        (Err(error), Err(cleanup)) => {
            return Err(anyhow::anyhow!(
                "metadata RLS failed: {error:#}; rollback failed: {cleanup}"
            ));
        }
    }
    let retained: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1)")
            .bind(role)
            .fetch_one(pool)
            .await?;
    anyhow::ensure!(!retained, "owned read-only role survived rollback");
    let tomb_key = NostrPublicKey::from_bytes(*public_key(82).as_bytes());
    let tomb = signed_profile(&own, 82, NOW, Vec::new(), r#"{"name":"deleted"}"#);
    repository.project_profile_event(&own, &tomb).await?;
    anyhow::ensure!(
        events.delete(&own, tomb.signed_event().claimed_id).await?,
        "canonical event deletion failed"
    );
    anyhow::ensure!(
        repository
            .current_authored_metadata_by_public_key(&own, tomb_key)
            .await?
            .is_none(),
        "tombstone exposed deleted metadata"
    );
    anyhow::ensure!(
        repository.project_profile_event(&own, &tomb).await?
            == IdentityProfileProjectionOutcome::Stale,
        "deleted head resurrected"
    );
    let fresh = signed_profile(&own, 82, NOW + 1, Vec::new(), r#"{"name":"fresh"}"#);
    repository.project_profile_event(&own, &fresh).await?;
    anyhow::ensure!(
        repository
            .current_authored_metadata_by_public_key(&own, tomb_key)
            .await?
            .is_some_and(|read| read.metadata().value.name.as_deref() == Some("fresh")),
        "newer signed head after tombstone missing"
    );
    let limit_key = NostrPublicKey::from_bytes(*public_key(83).as_bytes());
    for (index, (field, maximum)) in [
        ("display_name", 256),
        ("name", 256),
        ("picture", 2048),
        ("about", 4096),
        ("nip05", 320),
    ]
    .into_iter()
    .enumerate()
    {
        let content = serde_json::json!({(field): "x".repeat(maximum)}).to_string();
        let bounded = signed_profile(&own, 83, NOW + index as u64 * 2, Vec::new(), &content);
        repository.project_profile_event(&own, &bounded).await?;
        anyhow::ensure!(
            repository
                .current_authored_metadata_by_public_key(&own, limit_key)
                .await?
                .is_some(),
            "canonical metadata byte limit rejected exact boundary"
        );
        let content = serde_json::json!({(field): "x".repeat(maximum+1)}).to_string();
        let excessive = signed_profile(&own, 83, NOW + index as u64 * 2 + 1, Vec::new(), &content);
        repository.project_profile_event(&own, &excessive).await?;
        anyhow::ensure!(
            matches!(
                repository
                    .current_authored_metadata_by_public_key(&own, limit_key)
                    .await,
                Err(IdentityProfileRepositoryError::InvalidRecord)
            ),
            "authored read bypassed canonical metadata limit"
        );
        anyhow::ensure!(
            events
                .exact(&own, excessive.signed_event().claimed_id)
                .await?
                .is_some(),
            "metadata bounds rewrote/discarded original signed evidence"
        );
    }
    let transaction = connection.begin().await?;
    let corruption_result = async {
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres, "ALTER TABLE public.collaboration_events RENAME TO metadata_fixture_events_original".to_owned())).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres, "CREATE VIEW public.collaboration_events AS SELECT community_id,event_id,author_public_key,event_created_at,kind,tags,content,canonical_event_bytes,decode(repeat('00',64),'hex') AS signature,signature_state,verified_at,persistence_class,discriminator,received_at FROM public.metadata_fixture_events_original".to_owned())).await?;
        anyhow::ensure!(matches!(repository.current_authored_metadata_in_transaction(&transaction, &own, tomb_key).await,
            Err(IdentityProfileRepositoryError::InvalidRecord)), "authored read accepted corrupt retained signature");
        Ok::<(), anyhow::Error>(())
    }.await;
    let cleanup = transaction.rollback().await;
    match (corruption_result, cleanup) {
        (Ok(()), Ok(())) => {}
        (Err(error), Ok(())) => return Err(error),
        (Ok(()), Err(error)) => return Err(error.into()),
        (Err(error), Err(cleanup)) => {
            return Err(anyhow::anyhow!(
                "metadata corruption read failed: {error:#}; rollback failed: {cleanup}"
            ));
        }
    }
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(&own, key)
            .await?
            == classified,
        "metadata reads changed profile/owner/history"
    );
    let memberships: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships")
            .fetch_one(pool)
            .await?;
    anyhow::ensure!(memberships == 0, "metadata read fabricated membership");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_profile_actual_mounted_http_sql_redis_journey() -> anyhow::Result<()> {
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL")?;
    let redis = std::env::var("COLLAB_TEST_REDIS_URL")?;
    let redis_url = url::Url::parse(&redis)?;
    anyhow::ensure!(
        redis_url.scheme() == "redis"
            && matches!(
                redis_url.host_str(),
                Some("127.0.0.1" | "localhost" | "::1")
            )
            && redis_url.username().is_empty()
            && redis_url.password().is_none()
            && matches!(redis_url.path(), "" | "/" | "/0"),
        "owned authless loopback Redis database0 required for profile service qualification"
    );
    let mut database = url::Url::parse(&base_url)?;
    anyhow::ensure!(
        matches!(database.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback PostgreSQL required"
    );
    database.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let database = database.to_string();
    sqlx::Postgres::create_database(&database).await?;
    let community = CommunityId::from_uuid(Uuid::new_v4());
    let result = exercise_profile_http_database(&database, &redis, community).await;
    let redis_cleanup = profile_fixture_redis_cleanup(&redis_url, community).await;
    let database_cleanup = sqlx::Postgres::drop_database(&database).await;
    match (result, redis_cleanup, database_cleanup) {
        (Ok(()), Ok(()), Ok(())) => Ok(()),
        (result, redis_cleanup, database_cleanup) => Err(anyhow::anyhow!(
            "profile HTTP result={result:?}; owned Redis cleanup={redis_cleanup:?}; owned SQL cleanup={database_cleanup:?}"
        )),
    }
}

async fn profile_fixture_redis_integer(
    endpoint: &url::Url,
    parts: &[&[u8]],
) -> anyhow::Result<i64> {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let address = endpoint
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("Redis host missing"))?;
        let mut stream =
            tokio::net::TcpStream::connect((address, endpoint.port().unwrap_or(6379))).await?;
        let mut command = format!("*{}\r\n", parts.len()).into_bytes();
        for part in parts {
            command.extend_from_slice(format!("${}\r\n", part.len()).as_bytes());
            command.extend_from_slice(part);
            command.extend_from_slice(b"\r\n");
        }
        stream.write_all(&command).await?;
        let mut response = Vec::new();
        loop {
            anyhow::ensure!(response.len() < 64, "invalid owned Redis integer response");
            let byte = stream.read_u8().await?;
            response.push(byte);
            if byte == b'\n' {
                break;
            }
        }
        let response = std::str::from_utf8(&response)?;
        Ok::<_, anyhow::Error>(
            response
                .strip_suffix("\r\n")
                .and_then(|response| response.strip_prefix(':'))
                .ok_or_else(|| anyhow::anyhow!("owned Redis integer command failed"))?
                .parse()?,
        )
    })
    .await
    .map_err(|_| anyhow::anyhow!("owned Redis connect/write/read operation timed out"))?
}

async fn profile_fixture_redis_cleanup(
    endpoint: &url::Url,
    community: CommunityId,
) -> anyhow::Result<()> {
    const CLEAN: &str = "local keys=redis.call('KEYS',KEYS[1]); for _,key in ipairs(keys) do redis.call('DEL',key) end; return #redis.call('KEYS',KEYS[1])";
    for pattern in [
        format!("buzz:{community}:ratelimit:*:api"),
        format!("zed.collaboration.auth.v1:{community}:*"),
    ] {
        let left = profile_fixture_redis_integer(
            endpoint,
            &[b"EVAL", CLEAN.as_bytes(), b"1", pattern.as_bytes()],
        )
        .await?;
        anyhow::ensure!(
            left == 0,
            "owned profile admission Redis keys survived cleanup"
        );
    }
    Ok(())
}

fn profile_fixture_signed_event(
    key: u8,
    now: u64,
    tags: Vec<Vec<String>>,
    content: &str,
) -> SignedEvent {
    sign_restriction_event(
        key,
        CanonicalEvent::new(public_key(key), now, 0, tags, content.to_owned()),
    )
}

fn profile_fixture_wire(signed: &SignedEvent) -> serde_json::Value {
    serde_json::json!({"id":signed.claimed_id.to_hex(),"pubkey":signed.event.public_key.to_hex(),
        "created_at":signed.event.created_at,"kind":signed.event.kind,"tags":signed.event.tags,
        "content":signed.event.content,"sig":signed.signature.to_hex()})
}

async fn profile_fixture_http_request(
    router: &axum::Router,
    key: u8,
    host: &str,
    path: &str,
    body: &serde_json::Value,
    payload: bool,
    tamper_signature: bool,
    tamper_body: bool,
) -> anyhow::Result<(axum::http::StatusCode, serde_json::Value)> {
    use axum::body::HttpBody as _;
    use base64::Engine as _;
    use tower::ServiceExt as _;
    let bytes = serde_json::to_vec(body)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let mut tags = vec![
        vec!["u".into(), format!("https://{host}{path}")],
        vec!["method".into(), "POST".into()],
        vec!["nonce".into(), Uuid::new_v4().to_string()],
    ];
    if payload {
        tags.push(vec!["payload".into(), hex::encode(Sha256::digest(&bytes))]);
    }
    let mut signed = sign_restriction_event(
        key,
        CanonicalEvent::new(public_key(key), now, 27235, tags, String::new()),
    );
    if tamper_signature {
        signed.signature = EventSignature::from_hex(&"00".repeat(64))?;
    }
    let header = format!(
        "Nostr {}",
        base64::engine::general_purpose::STANDARD
            .encode(serde_json::to_vec(&profile_fixture_wire(&signed))?)
    );
    let transmitted = if tamper_body { b"[]".to_vec() } else { bytes };
    let response = router
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("Host", host)
                .header("Authorization", header)
                .body(axum::body::Body::from(transmitted))?,
        )
        .await?;
    let status = response.status();
    let mut body = response.into_body();
    let mut content = Vec::new();
    while let Some(chunk) = body.data().await {
        content.extend_from_slice(&chunk?);
    }
    let value = match serde_json::from_slice(&content) {
        Ok(value) => value,
        Err(error) if status.is_success() => return Err(error.into()),
        Err(_) => serde_json::Value::Null,
    };
    Ok((status, value))
}

async fn exercise_profile_http_database(
    database: &str,
    redis: &str,
    community: CommunityId,
) -> anyhow::Result<()> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(database)
        .await?;
    let result = async {
        profile_fixture_apply_canonical_schema(&pool).await?;
        tokio::time::timeout(
            std::time::Duration::from_secs(90),
            exercise_profile_http_journey(&pool, redis, community),
        )
        .await
        .map_err(|_| anyhow::anyhow!("actual profile SQL/Redis/HTTP pool1 journey deadlocked"))?
    }
    .await;
    pool.close().await;
    result
}

async fn exercise_profile_http_journey(
    pool: &sqlx::PgPool,
    redis: &str,
    community: CommunityId,
) -> anyhow::Result<()> {
    use axum::http::StatusCode;
    use collab::identity::profile_repository::IdentitySubjectStatus;
    let host = "profile-service.example.test";
    let own = TenantContext::establish(
        Some(TrustedTenantRoute::from_direct_host(community, host)?),
        &[],
    )?;
    sqlx::query("INSERT INTO public.collaboration_communities (community_id,host,lifecycle_state,aggregate_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,'active',1,'zed',$2,now(),now(),now())")
        .bind(community.as_uuid()).bind(host).execute(pool).await?;
    let repository = Arc::new(IdentityProfileRepository::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let member =
        authenticate_registration(&repository, &own, &signed_registration(81, None, now, 81))
            .await?;
    sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'member','active',1,now(),now(),'zed',$3,now())")
        .bind(community.as_uuid()).bind(member.principal_id().as_uuid()).bind(member.principal_id().to_string()).execute(pool).await?;
    let config: collab::Config = serde_json::from_value(serde_json::json!({
        "http_port":0,"database_url":"","database_max_connections":1,"zed_environment":"test",
        "zed_cloud_internal_api_key":"fixture-only","collaboration_redis_url":redis,
        "collaboration_nostr_routes":serde_json::json!([{"community_id":community.as_uuid(),"public_url":format!("https://{host}")}]).to_string(),
        "collaboration_nostr_api_calls_per_minute":1000
    }))?;
    let make_router = |config: &collab::Config| -> anyhow::Result<axum::Router> {
        collab::nostr::http::configured_profile_http_router(
            config,
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?
        .ok_or_else(|| anyhow::anyhow!("configured production profile service was not mounted"))
    };
    let router = make_router(&config)?;
    exercise_profile_public_metadata(pool, &config, &repository, community, host).await?;
    let query = serde_json::json!([{"authors":[public_key(81).to_hex()],"kinds":[0],"limit":1}]);
    let (status, response) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, false, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK && response == serde_json::json!([]),
        "mounted ordinary profile query must return actual empty list"
    );
    let signed = profile_fixture_signed_event(
        81,
        now,
        Vec::new(),
        r#"{"name":"ordinary","about":"genuine HTTP profile"}"#,
    );
    let wire = profile_fixture_wire(&signed);
    let before = repository
        .current_subject_by_public_key(&own, NostrPublicKey::from_bytes(*public_key(81).as_bytes()))
        .await?
        .ok_or_else(|| anyhow::anyhow!("registered fixture subject missing"))?;
    let (status, response) =
        profile_fixture_http_request(&router, 81, host, "/events", &wire, false, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK
            && response
                .get("accepted")
                .and_then(serde_json::Value::as_bool)
                == Some(true),
        "actual mounted profile service failed genuine signed kind0 production journey"
    );
    let (status, response) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, false, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK && response == serde_json::json!([wire.clone()]),
        "actual mounted profile query lost exact signed head"
    );
    let authored = repository
        .current_authored_metadata_by_public_key(
            &own,
            NostrPublicKey::from_bytes(*public_key(81).as_bytes()),
        )
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!("actual service ingestion did not reach canonical authored reader")
        })?;
    anyhow::ensure!(
        authored.metadata().value.name.as_deref() == Some("ordinary")
            && authored
                .subject()
                .is_some_and(|subject| subject.principal_id() == before.principal_id()
                    && subject.status() == IdentitySubjectStatus::RegisteredUnclassified)
            && authored.owner_public_key().is_none(),
        "ordinary profile inferred Human/owner or changed stable identity"
    );
    let restarted = make_router(&config)?;
    let (status, response) =
        profile_fixture_http_request(&restarted, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK && response == serde_json::json!([wire.clone()]),
        "service reconstruction lost durable signed profile"
    );
    let (_, response) =
        profile_fixture_http_request(&restarted, 81, host, "/events", &wire, true, false, false)
            .await?;
    anyhow::ensure!(
        response.get("message").and_then(serde_json::Value::as_str) == Some("duplicate:"),
        "fresh NIP98 actual EVENT retry lost duplicate semantics"
    );
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/events", &wire, true, false, true)
            .await?;
    anyhow::ensure!(
        status == StatusCode::UNAUTHORIZED,
        "present payload hash failed to bind actual transmitted body"
    );
    let redis_url = url::Url::parse(redis)?;
    let quota_key = format!("buzz:{community}:ratelimit:{}:api", public_key(81).to_hex());
    let prior = profile_fixture_redis_integer(
        &redis_url,
        &[
            b"EVAL",
            b"return tonumber(redis.call('GET',KEYS[1]) or '0')",
            b"1",
            quota_key.as_bytes(),
        ],
    )
    .await?;
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/events", &wire, true, true, false)
            .await?;
    let after = profile_fixture_redis_integer(
        &redis_url,
        &[
            b"EVAL",
            b"return tonumber(redis.call('GET',KEYS[1]) or '0')",
            b"1",
            quota_key.as_bytes(),
        ],
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::UNAUTHORIZED && prior == after,
        "invalid crypto consumed shared HTTP quota"
    );
    let unknown = profile_fixture_signed_event(82, now, Vec::new(), r#"{"name":"no membership"}"#);
    let (status, _) = profile_fixture_http_request(
        &router,
        82,
        host,
        "/events",
        &profile_fixture_wire(&unknown),
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "plain genuine key authentication fabricated profile authorization"
    );
    anyhow::ensure!(
        repository
            .current_registration_by_public_key(
                &own,
                NostrPublicKey::from_bytes(*public_key(82).as_bytes())
            )
            .await?
            .is_some(),
        "successful genuine NIP98 registration was incorrectly rolled back as profile denial"
    );
    let events: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_events WHERE community_id=$1 AND event_id=$2",
    )
    .bind(community.as_uuid())
    .bind(unknown.claimed_id.as_bytes().as_slice())
    .fetch_one(pool)
    .await?;
    let members: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1",
    )
    .bind(community.as_uuid())
    .fetch_one(pool)
    .await?;
    let denied_key = format!("buzz:{community}:ratelimit:{}:api", public_key(82).to_hex());
    let denied_count = profile_fixture_redis_integer(
        &redis_url,
        &[
            b"EVAL",
            b"return tonumber(redis.call('GET',KEYS[1]) or '0')",
            b"1",
            denied_key.as_bytes(),
        ],
    )
    .await?;
    anyhow::ensure!(
        events == 0 && members == 1 && denied_count == 1,
        "valid unauthorized HTTP admission failed quota or mutated EVENT/member state"
    );
    let member_key = NostrPublicKey::from_bytes(*public_key(81).as_bytes());
    let registration_before_metadata = repository
        .current_subject_by_public_key(&own, member_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("ordinary registration missing"))?;
    let malformed = vec!["auth".into(), "not-a-public-key".into()];
    let mut invalid_signature = owner_tag(81, 83, "");
    *invalid_signature
        .get_mut(3)
        .ok_or_else(|| anyhow::anyhow!("owner signature fixture tag absent"))? = "00".repeat(64);
    let proof = owner_tag(81, 83, "");
    for (offset, tags) in [
        vec![malformed],
        vec![invalid_signature],
        vec![proof.clone(), proof],
        vec![owner_tag(81, 81, "")],
        vec![owner_tag(
            81,
            83,
            &format!("created_at<{}", now.saturating_sub(1)),
        )],
    ]
    .into_iter()
    .enumerate()
    {
        let metadata = profile_fixture_signed_event(
            81,
            now + u64::try_from(offset)? + 1,
            tags,
            r#"{"name":"metadata without owner authority"}"#,
        );
        let (status, response) = profile_fixture_http_request(
            &router,
            81,
            host,
            "/events",
            &profile_fixture_wire(&metadata),
            true,
            false,
            false,
        )
        .await?;
        anyhow::ensure!(
            status == StatusCode::OK
                && response
                    .get("accepted")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true),
            "invalid optional OA rejected independently signed ordinary metadata"
        );
        let current = repository
            .current_subject_by_public_key(&own, member_key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("metadata subject missing"))?;
        let metadata = repository
            .current_authored_metadata_by_public_key(&own, member_key)
            .await?
            .ok_or_else(|| anyhow::anyhow!("metadata head missing"))?;
        anyhow::ensure!(
            current.principal_id() == registration_before_metadata.principal_id()
                && current.profile_id() == registration_before_metadata.profile_id()
                && current.status() == IdentitySubjectStatus::RegisteredUnclassified
                && metadata.owner_public_key().is_none(),
            "optional OA failure allocated or delegated identity authority"
        );
    }
    let owned = profile_fixture_signed_event(
        81,
        now + 6,
        vec![owner_tag(81, 83, "")],
        r#"{"name":"verified owner profile"}"#,
    );
    let (status, response) = profile_fixture_http_request(
        &router,
        81,
        host,
        "/events",
        &profile_fixture_wire(&owned),
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::OK
            && response
                .get("accepted")
                .and_then(serde_json::Value::as_bool)
                == Some(true),
        "valid owner kind0 failed canonical materialization"
    );
    let metadata = repository
        .current_authored_metadata_by_public_key(&own, member_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("owned metadata missing"))?;
    anyhow::ensure!(
        metadata.owner_public_key() == Some(NostrPublicKey::from_bytes(*public_key(83).as_bytes()))
            && metadata
                .subject()
                .is_some_and(|subject| subject.principal_id() == before.principal_id()),
        "genuine OA lost owner/stable principal provenance"
    );
    let conflict = profile_fixture_signed_event(
        81,
        now + 7,
        vec![owner_tag(81, 84, "")],
        r#"{"name":"conflicting owner"}"#,
    );
    let (status, _) = profile_fixture_http_request(
        &router,
        81,
        host,
        "/events",
        &profile_fixture_wire(&conflict),
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "different verified owner changed established owner"
    );
    let (status, response) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK && response == serde_json::json!([profile_fixture_wire(&owned)]),
        "owner-conflict rollback replaced signed metadata head"
    );
    let cleared = profile_fixture_signed_event(81, now + 8, Vec::new(), "{}");
    let (status, response) = profile_fixture_http_request(
        &router,
        81,
        host,
        "/events",
        &profile_fixture_wire(&cleared),
        false,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::OK
            && response
                .get("accepted")
                .and_then(serde_json::Value::as_bool)
                == Some(true),
        "genuine absolute metadata clear was not accepted"
    );
    let metadata = repository
        .current_authored_metadata_by_public_key(&own, member_key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("empty authored profile confused with missing"))?;
    anyhow::ensure!(
        metadata.metadata().value.name.is_none()
            && metadata.metadata().value.about.is_none()
            && metadata.owner_public_key()
                == Some(NostrPublicKey::from_bytes(*public_key(83).as_bytes())),
        "absolute clear retained stale metadata or erased verified owner"
    );
    let (status, response) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, false, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK && response == serde_json::json!([profile_fixture_wire(&cleared)]),
        "mounted query confused an actual empty profile with missing state"
    );
    let different_signer =
        profile_fixture_signed_event(85, now, Vec::new(), r#"{"name":"wrong signer"}"#);
    let (status, _) = profile_fixture_http_request(
        &router,
        81,
        host,
        "/events",
        &profile_fixture_wire(&different_signer),
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "valid outer HTTP key published another key's profile"
    );
    let mut invalid_inner =
        profile_fixture_signed_event(81, now + 9, Vec::new(), r#"{"name":"invalid inner"}"#);
    invalid_inner.signature = EventSignature::from_hex(&"00".repeat(64))?;
    let (status, _) = profile_fixture_http_request(
        &router,
        81,
        host,
        "/events",
        &profile_fixture_wire(&invalid_inner),
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::BAD_REQUEST,
        "genuine outer authorization accepted invalid inner signature"
    );
    let unsupported = sign_restriction_event(
        81,
        CanonicalEvent::new(
            public_key(81),
            now,
            1,
            Vec::new(),
            "unsupported profile kind".into(),
        ),
    );
    let (status, _) = profile_fixture_http_request(
        &router,
        81,
        host,
        "/events",
        &profile_fixture_wire(&unsupported),
        true,
        false,
        false,
    )
    .await?;
    let (query_status, _) = profile_fixture_http_request(
        &router,
        81,
        host,
        "/query",
        &serde_json::json!([{"kinds":[1]}]),
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::UNPROCESSABLE_ENTITY
            && query_status == StatusCode::UNPROCESSABLE_ENTITY,
        "profile-only adapter pretended broad EVENT/query support"
    );
    let mut exhausted = config.clone();
    let current_count = profile_fixture_redis_integer(
        &redis_url,
        &[
            b"EVAL",
            b"return tonumber(redis.call('GET',KEYS[1]) or '0')",
            b"1",
            quota_key.as_bytes(),
        ],
    )
    .await?;
    exhausted.collaboration_nostr_api_calls_per_minute = Some(u64::try_from(current_count)? + 1);
    let first_worker = make_router(&exhausted)?;
    let second_worker = make_router(&exhausted)?;
    let (first, _) = profile_fixture_http_request(
        &first_worker,
        81,
        host,
        "/query",
        &query,
        true,
        false,
        false,
    )
    .await?;
    let (second, _) = profile_fixture_http_request(
        &second_worker,
        81,
        host,
        "/query",
        &query,
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        first == StatusCode::OK && second == StatusCode::TOO_MANY_REQUESTS,
        "separate actual service workers did not share quota"
    );
    let ttl = profile_fixture_redis_integer(&redis_url, &[b"TTL", quota_key.as_bytes()]).await?;
    anyhow::ensure!((0..=60).contains(&ttl), "actual quota missed expiration");
    let detached =
        profile_fixture_redis_integer(&redis_url, &[b"PERSIST", quota_key.as_bytes()]).await?;
    anyhow::ensure!(detached == 1, "owned quota TTL fault fixture not mounted");
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, true, false, false)
            .await?;
    let repaired =
        profile_fixture_redis_integer(&redis_url, &[b"TTL", quota_key.as_bytes()]).await?;
    anyhow::ensure!(
        status == StatusCode::OK && (0..=60).contains(&repaired),
        "actual atomic quota failed TTL repair"
    );
    let administrator =
        authenticate_registration(&repository, &own, &signed_registration(86, None, now, 86))
            .await?;
    sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'admin','active',1,now(),now(),'zed',$3,now())")
        .bind(community.as_uuid()).bind(administrator.principal_id().as_uuid()).bind(administrator.principal_id().to_string()).execute(pool).await?;
    let restrictions = collab::nostr::event_ingest::NostrEventIngress::new(
        collab::admin::moderation::NostrRestrictionCommandSink::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?,
        collab::nostr::ingress::NostrIngressDeployment::InProcess,
    );
    let timeout = signed_restriction_command(86, 9042, 81, now, 87, Some(now + 60));
    let outcome = send_signed_restriction_command(
        &restrictions,
        &own,
        &administrator,
        collaboration_domain::MembershipRole::Admin,
        &timeout,
        now,
    )
    .await?;
    anyhow::ensure!(
        outcome.status() == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
        "canonical signed timeout did not commit"
    );
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/events", &wire, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "actual profile EVENT bypassed canonical active timeout"
    );
    let lift = signed_restriction_command(86, 9043, 81, now, 88, None);
    let outcome = send_signed_restriction_command(
        &restrictions,
        &own,
        &administrator,
        collaboration_domain::MembershipRole::Admin,
        &lift,
        now,
    )
    .await?;
    anyhow::ensure!(
        outcome.status() == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
        "canonical signed timeout lift did not commit"
    );
    let ban = signed_restriction_command(86, 9040, 81, now, 89, None);
    let outcome = send_signed_restriction_command(
        &restrictions,
        &own,
        &administrator,
        collaboration_domain::MembershipRole::Admin,
        &ban,
        now,
    )
    .await?;
    anyhow::ensure!(
        outcome.status() == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
        "canonical signed direct ban did not commit"
    );
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/events", &wire, true, false, false)
            .await?;
    let (read_status, _) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN && read_status == StatusCode::FORBIDDEN,
        "actual profile write/read bypassed canonical direct ban"
    );
    let lift = signed_restriction_command(86, 9041, 81, now, 90, None);
    let outcome = send_signed_restriction_command(
        &restrictions,
        &own,
        &administrator,
        collaboration_domain::MembershipRole::Admin,
        &lift,
        now,
    )
    .await?;
    anyhow::ensure!(
        outcome.status() == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
        "canonical signed ban lift did not commit"
    );
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK,
        "actual current lifted policy remained stale"
    );
    sqlx::query("UPDATE public.collaboration_communities SET host='wrong-host.example.test',aggregate_version=aggregate_version+1 WHERE community_id=$1").bind(community.as_uuid()).execute(pool).await?;
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "explicit deployment bypassed mismatched canonical SQL community host"
    );
    sqlx::query("UPDATE public.collaboration_communities SET host=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1").bind(community.as_uuid()).bind(host).execute(pool).await?;
    profile_fixture_outer_auth_boundaries(&router, host, &query).await?;
    profile_fixture_lifecycle_race(pool, &router, community, host).await?;
    for state in ["archived", "quiescing", "fenced", "tombstone"] {
        sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1")
            .bind(community.as_uuid()).bind(state).execute(pool).await?;
        let (status, _) =
            profile_fixture_http_request(&router, 81, host, "/events", &wire, true, false, false)
                .await?;
        let (read_status, _) =
            profile_fixture_http_request(&router, 81, host, "/query", &query, true, false, false)
                .await?;
        anyhow::ensure!(
            status == StatusCode::FORBIDDEN && read_status == StatusCode::FORBIDDEN,
            "nonactive community was served by actual profile owner"
        );
    }
    sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state='active',aggregate_version=aggregate_version+1 WHERE community_id=$1")
        .bind(community.as_uuid()).execute(pool).await?;
    profile_fixture_multiple_filters(pool, &router, &repository, &own, host, now).await?;
    profile_fixture_accepted_names(pool, &config, &repository, &own, host, now).await?;
    sqlx::query("UPDATE public.collaboration_community_memberships SET status='revoked',membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
        .bind(community.as_uuid()).bind(member.principal_id().as_uuid()).execute(pool).await?;
    let (status, _) =
        profile_fixture_http_request(&router, 81, host, "/events", &wire, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::FORBIDDEN,
        "cached duplicate receipt bypassed current revocation"
    );
    let mut zero = config.clone();
    zero.collaboration_nostr_api_calls_per_minute = Some(0);
    let (status, _) = profile_fixture_http_request(
        &make_router(&zero)?,
        81,
        host,
        "/events",
        &wire,
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::TOO_MANY_REQUESTS,
        "configured zero shared quota failed deny-all semantics"
    );
    drop(router);
    drop(restarted);
    drop(repository);
    Ok(())
}

async fn profile_fixture_multiple_filters(
    pool: &sqlx::PgPool,
    router: &axum::Router,
    repository: &Arc<IdentityProfileRepository>,
    tenant: &TenantContext,
    host: &str,
    now: u64,
) -> anyhow::Result<()> {
    use axum::http::StatusCode;
    use collab::nostr::subscriptions::{MAX_FILTER_LIMIT, MAX_FILTERS_PER_REQUEST};
    let mut heads = Vec::new();
    for (offset, secret) in [141u8, 142, 143].into_iter().enumerate() {
        let registered = authenticate_registration(
            repository,
            tenant,
            &signed_registration(secret, None, now, secret),
        )
        .await?;
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'member','active',1,now(),now(),'zed',$3,now())")
            .bind(tenant.community_id().as_uuid()).bind(registered.principal_id().as_uuid()).bind(registered.principal_id().to_string()).execute(pool).await?;
        let signed = profile_fixture_signed_event(
            secret,
            now.saturating_sub(u64::try_from(offset)?),
            Vec::new(),
            &serde_json::json!({"name":format!("filter head {secret}")}).to_string(),
        );
        let wire = profile_fixture_wire(&signed);
        let (status, response) = profile_fixture_http_request(
            router, secret, host, "/events", &wire, true, false, false,
        )
        .await?;
        anyhow::ensure!(
            status == StatusCode::OK
                && response
                    .get("accepted")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true),
            "multifilter fixture failed genuine signed production head admission"
        );
        heads.push(wire);
    }
    let newest = heads
        .first()
        .ok_or_else(|| anyhow::anyhow!("newest head absent"))?;
    let middle = heads
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("middle head absent"))?;
    let oldest = heads
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("oldest head absent"))?;
    let authors = [141u8, 142, 143].map(|secret| public_key(secret).to_hex());
    let newest_one =
        serde_json::json!({"kinds":[0],"authors":[public_key(141).to_hex()],"limit":1});
    let oldest_one =
        serde_json::json!({"kinds":[0],"authors":[public_key(143).to_hex()],"limit":1});
    let first_one = serde_json::json!({"kinds":[0],"authors":authors,"limit":1});
    let first_two = serde_json::json!({"kinds":[0],"authors":authors,"limit":2});
    let cases = [
        (
            serde_json::json!([first_one, oldest_one]),
            serde_json::json!([newest, oldest]),
            "independent profile filters lost their own limits or starved the later filter",
        ),
        (
            serde_json::json!([first_two, oldest_one]),
            serde_json::json!([newest, middle, oldest]),
            "unequal profile filter limits were merged",
        ),
        (
            serde_json::json!([first_two, newest_one]),
            serde_json::json!([newest, middle, newest]),
            "overlapping HTTP profile filters lost their repeated result",
        ),
        (
            serde_json::json!([oldest_one, first_two]),
            serde_json::json!([oldest, newest, middle]),
            "profile response ignored original filter order",
        ),
        (
            serde_json::json!([newest_one, newest_one]),
            serde_json::json!([newest, newest]),
            "duplicate HTTP filters were deduplicated",
        ),
    ];
    for (query, expected, failure) in cases {
        let (status, response) =
            profile_fixture_http_request(router, 81, host, "/query", &query, true, false, false)
                .await?;
        anyhow::ensure!(
            status == StatusCode::OK && response == expected,
            "{failure}"
        );
    }
    let clamped =
        serde_json::json!({"kinds":[0],"authors":authors,"limit":u64::from(MAX_FILTER_LIMIT)+1});
    let query = serde_json::Value::Array(vec![clamped.clone(); MAX_FILTERS_PER_REQUEST]);
    let expected = heads
        .iter()
        .cloned()
        .cycle()
        .take(heads.len() * MAX_FILTERS_PER_REQUEST)
        .collect::<Vec<_>>();
    let (status, response) =
        profile_fixture_http_request(router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK && response == serde_json::json!(expected),
        "canonical maximum filter count/clamped per-filter bound lost overlap or order"
    );
    let query = serde_json::Value::Array(vec![clamped; MAX_FILTERS_PER_REQUEST + 1]);
    let (status, _) =
        profile_fixture_http_request(router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::BAD_REQUEST,
        "profile route accepted more than the canonical maximum filter count"
    );
    let query = serde_json::json!([{"kinds":[0],"authors":authors,"limit":0}]);
    let (status, _) =
        profile_fixture_http_request(router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::BAD_REQUEST,
        "zero-limit profile query bypassed the canonical query constructor"
    );
    let (status, repeated) =
        profile_fixture_http_request(router, 141, host, "/events", newest, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK
            && repeated
                .get("accepted")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && repeated
                .get("message")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|message| message.starts_with("duplicate:")),
        "genuine profile EVENT retry lost its accepted duplicate receipt"
    );
    let query = serde_json::json!([newest_one, newest_one]);
    let (status, response) =
        profile_fixture_http_request(router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK && response == serde_json::json!([newest, newest]),
        "EVENT retry changed repeated-filter head semantics"
    );
    let mut aggregate_heads = Vec::new();
    let mut aggregate_authors = Vec::new();
    let aggregate_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    for (offset, secret) in (144u8..=244).enumerate() {
        let registered = authenticate_registration(
            repository,
            tenant,
            &signed_registration(secret, None, aggregate_now, secret),
        )
        .await?;
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'member','active',1,now(),now(),'zed',$3,now())")
            .bind(tenant.community_id().as_uuid()).bind(registered.principal_id().as_uuid()).bind(registered.principal_id().to_string()).execute(pool).await?;
        let event = profile_fixture_signed_event(
            secret,
            aggregate_now.saturating_sub(u64::try_from(offset)?),
            Vec::new(),
            &serde_json::json!({"name":format!("aggregate profile head {secret}")}).to_string(),
        );
        let wire = profile_fixture_wire(&event);
        let (status, response) = profile_fixture_http_request(
            router, secret, host, "/events", &wire, true, false, false,
        )
        .await?;
        anyhow::ensure!(
            status == StatusCode::OK
                && response
                    .get("accepted")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true),
            "aggregate profile fixture failed genuine signed EVENT admission"
        );
        aggregate_authors.push(public_key(secret).to_hex());
        aggregate_heads.push(wire);
    }
    anyhow::ensure!(
        aggregate_heads.len() == 101,
        "aggregate fixture did not retain all101 genuinely submitted heads"
    );
    let filter = serde_json::json!({"kinds":[0],"authors":aggregate_authors,"limit":101});
    let query = serde_json::Value::Array(vec![filter; MAX_FILTERS_PER_REQUEST]);
    let expected = aggregate_heads
        .iter()
        .cloned()
        .cycle()
        .take(aggregate_heads.len() * MAX_FILTERS_PER_REQUEST)
        .collect::<Vec<_>>();
    let (status, response) =
        profile_fixture_http_request(router, 81, host, "/query", &query, true, false, false)
            .await?;
    anyhow::ensure!(
        status == StatusCode::OK,
        "genuine1010-result profile query was not admitted"
    );
    let returned = response
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("aggregate profile response was not an event list"))?;
    anyhow::ensure!(
        returned.len() > usize::try_from(MAX_FILTER_LIMIT)? && returned.len() == 1010,
        "profile query retained unsupported aggregate1000 cap"
    );
    anyhow::ensure!(
        returned == &expected,
        "aggregate profile query lost original filter order, signed heads or repeated results"
    );
    eprintln!("PROFILE_HTTP_MULTIFILTER_OVER1000_VERIFIED");
    Ok(())
}

async fn profile_fixture_lifecycle_race(
    pool: &sqlx::PgPool,
    router: &axum::Router,
    community: CommunityId,
    host: &str,
) -> anyhow::Result<()> {
    let service_pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
        .fetch_one(pool)
        .await?;
    let policy_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect_with(pool.connect_options().as_ref().clone())
        .await?;
    let result=async {
        let mut policy=policy_pool.begin().await?;
        let policy_pid:i32=sqlx::query_scalar("SELECT pg_backend_pid()").fetch_one(&mut *policy).await?;
        anyhow::ensure!(service_pid!=policy_pid,"policy blocker did not own a separate backend");
        sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state='fenced',aggregate_version=aggregate_version+1 WHERE community_id=$1")
            .bind(community.as_uuid()).execute(&mut *policy).await?;
        let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        let event=profile_fixture_signed_event(81,now+9,Vec::new(),r#"{"name":"must not commit across fence"}"#);
        let wire=profile_fixture_wire(&event);
        let mut request=Box::pin(profile_fixture_http_request(router,81,host,"/events",&wire,true,false,false));
        let observe=async {
            loop {
                sqlx::query("SELECT pg_stat_clear_snapshot()").execute(&mut *policy).await?;
                let waiting:bool=sqlx::query_scalar(r#"
SELECT EXISTS (
 SELECT 1 FROM pg_stat_activity AS activity
 WHERE activity.pid=$1 AND activity.datid=(SELECT oid FROM pg_database WHERE datname=current_database())
 AND activity.state='active' AND activity.wait_event_type='Lock'
 AND activity.query LIKE '%FROM public.collaboration_communities%'
 AND activity.query LIKE '%FOR SHARE%'
 AND $2=ANY(pg_blocking_pids(activity.pid))
 AND EXISTS (SELECT 1 FROM pg_locks AS waiting WHERE waiting.pid=activity.pid AND NOT waiting.granted)
)
"#).bind(service_pid).bind(policy_pid).fetch_one(&mut *policy).await?;
                if waiting {return Ok::<(),anyhow::Error>(());}
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        };
        tokio::select! {
            completed=&mut request => {
                anyhow::bail!("profile request completed before its exact FOR SHARE backend waited on owned policy blocker: {completed:?}");
            }
            observed=tokio::time::timeout(std::time::Duration::from_secs(5),observe) => {
                observed.map_err(|_|anyhow::anyhow!("exact owned profile FOR SHARE lock waiter was not observed"))??;
            }
        }
        println!("PROFILE_HTTP_LIFECYCLE_LOCK_WAITER_VERIFIED: service_pid={service_pid}; policy_pid={policy_pid}; exact community FOR SHARE blocked before commit");
        policy.commit().await?;
        let (status,_)=tokio::time::timeout(std::time::Duration::from_secs(5),request).await??;
        anyhow::ensure!(status==axum::http::StatusCode::FORBIDDEN,"profile write used stale lifecycle after observed row blocker committed");
        let count:i64=sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events WHERE community_id=$1 AND event_id=$2")
            .bind(community.as_uuid()).bind(event.claimed_id.as_bytes().to_vec()).fetch_one(pool).await?;
        anyhow::ensure!(count==0,"fenced HTTP failure persisted signed EVENT despite transaction admission");
        sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state='active',aggregate_version=aggregate_version+1 WHERE community_id=$1")
            .bind(community.as_uuid()).execute(pool).await?;
        Ok::<(),anyhow::Error>(())
    }.await;
    policy_pool.close().await;
    result
}

async fn profile_fixture_fixed_authorization(
    router: &axum::Router,
    host: &str,
    path: &str,
    body: &serde_json::Value,
    signed: &SignedEvent,
) -> anyhow::Result<(axum::http::StatusCode, serde_json::Value)> {
    use axum::body::HttpBody as _;
    use base64::Engine as _;
    use tower::ServiceExt as _;
    let header = format!(
        "Nostr {}",
        base64::engine::general_purpose::STANDARD
            .encode(serde_json::to_vec(&profile_fixture_wire(signed))?)
    );
    let response = router
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("Host", host)
                .header("Authorization", header)
                .body(axum::body::Body::from(serde_json::to_vec(body)?))?,
        )
        .await?;
    let status = response.status();
    let mut body = response.into_body();
    let mut content = Vec::new();
    while let Some(chunk) = body.data().await {
        content.extend_from_slice(&chunk?);
    }
    Ok((
        status,
        match serde_json::from_slice(&content) {
            Ok(value) => value,
            Err(error) if status.is_success() => return Err(error.into()),
            Err(_) => serde_json::Value::Null,
        },
    ))
}

async fn profile_fixture_outer_auth_boundaries(
    router: &axum::Router,
    host: &str,
    query: &serde_json::Value,
) -> anyhow::Result<()> {
    use axum::http::StatusCode;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let tags = |authority: &str, method: &str| {
        vec![
            vec!["u".into(), format!("https://{authority}/query")],
            vec!["method".into(), method.into()],
            vec!["nonce".into(), Uuid::new_v4().to_string()],
        ]
    };
    for (created_at, outer_tags) in [
        (now, tags("foreign-profile.example.test", "POST")),
        (now, tags(host, "GET")),
        (now.saturating_sub(61), tags(host, "POST")),
        (now + 61, tags(host, "POST")),
    ] {
        let signed = sign_restriction_event(
            81,
            CanonicalEvent::new(public_key(81), created_at, 27235, outer_tags, String::new()),
        );
        let (status, _) =
            profile_fixture_fixed_authorization(router, host, "/query", query, &signed).await?;
        anyhow::ensure!(
            status == StatusCode::UNAUTHORIZED,
            "mounted ordinary route ignored signed URL/method/freshness boundary"
        );
    }
    let bytes = serde_json::to_vec(query)?;
    let hash = hex::encode(Sha256::digest(&bytes));
    for hashes in [
        vec![hash.clone(), hash.clone()],
        vec!["not-a-body-hash".into()],
    ] {
        let mut outer_tags = tags(host, "POST");
        outer_tags.extend(hashes.into_iter().map(|hash| vec!["payload".into(), hash]));
        let signed = sign_restriction_event(
            81,
            CanonicalEvent::new(public_key(81), now, 27235, outer_tags, String::new()),
        );
        let (status, _) =
            profile_fixture_fixed_authorization(router, host, "/query", query, &signed).await?;
        anyhow::ensure!(
            status == StatusCode::UNAUTHORIZED,
            "optional payload semantics accepted duplicate or malformed present body hash"
        );
    }
    let replay = sign_restriction_event(
        81,
        CanonicalEvent::new(
            public_key(81),
            now,
            27235,
            tags(host, "POST"),
            String::new(),
        ),
    );
    let (first, _) =
        profile_fixture_fixed_authorization(router, host, "/query", query, &replay).await?;
    let (second, _) =
        profile_fixture_fixed_authorization(router, host, "/query", query, &replay).await?;
    anyhow::ensure!(
        first == StatusCode::OK && second == StatusCode::UNAUTHORIZED,
        "actual shared replay store accepted repeated genuine NIP98 authorization"
    );
    let (status, _) = profile_fixture_fixed_authorization(
        router,
        "unmounted-profile.example.test",
        "/query",
        query,
        &replay,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::NOT_FOUND,
        "unknown Host selected an implicit profile tenant"
    );
    Ok(())
}

async fn profile_fixture_apply_canonical_schema(pool: &sqlx::PgPool) -> anyhow::Result<()> {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let package = repository.join("deploy/collaboration/migrations");
    let migrations = repository.join("crates/collab/migrations");
    let manifest = package.join("manifest.json");
    let mut validator = tokio::process::Command::new("python3");
    validator
        .kill_on_drop(true)
        .arg(package.join("migrate.py"))
        .arg("--manifest")
        .arg(&manifest)
        .arg("--migration-directory")
        .arg(&migrations)
        .arg("validate");
    let validation = tokio::time::timeout(std::time::Duration::from_secs(10), validator.output())
        .await
        .map_err(|_| anyhow::anyhow!("canonical migration package validation timed out"))??;
    anyhow::ensure!(
        validation.status.success()
            && String::from_utf8_lossy(&validation.stdout)
                .contains("migration_manifest_valid count=31 ceiling=20261026000100"),
        "canonical checked31-pair package rejected: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
    let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest)?)?;
    let entries = document
        .get("migrations")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("canonical migration plan absent"))?;
    anyhow::ensure!(
        entries.len() == 31,
        "checked canonical service schema inventory changed"
    );
    // Reuse the legacy bootstrap required by the canonical package's own smoke owner before its additive SQL pairs.
    sqlx::raw_sql(include_str!("../migrations/20251208000000_test_schema.sql"))
        .execute(pool)
        .await?;
    for migration in entries {
        let up = migration
            .get("up")
            .ok_or_else(|| anyhow::anyhow!("canonical UP entry absent"))?;
        let file = up
            .get("file")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("canonical UP path absent"))?;
        let expected = up
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("canonical UP receipt absent"))?;
        let sql = std::fs::read(migrations.join(file))?;
        anyhow::ensure!(
            hex::encode(Sha256::digest(&sql)) == expected,
            "validated canonical SQL changed before service application"
        );
        sqlx::raw_sql(std::str::from_utf8(&sql)?)
            .execute(pool)
            .await?;
    }
    println!(
        "PROFILE_HTTP_CANONICAL_SCHEMA31_VERIFIED: actual checked ordered31-pair package plus canonical legacy test bootstrap; same owned service database"
    );
    Ok(())
}

async fn exercise_profile_public_metadata(
    pool: &sqlx::PgPool,
    config: &collab::Config,
    repository: &IdentityProfileRepository,
    community: CommunityId,
    host: &str,
) -> anyhow::Result<()> {
    use axum::http::StatusCode;
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    let other = CommunityId::from_uuid(Uuid::new_v4());
    let absent = CommunityId::from_uuid(Uuid::new_v4());
    let other_host = "metadata-second.example.test";
    let absent_host = "metadata-absent.example.test";
    let icon = "https://profile-service.example.test/workspace.png";
    let read_counts = || async {
        sqlx::query_as::<_, (i64, i64, i64)>("SELECT (SELECT count(*) FROM public.collaboration_identity_profiles),(SELECT count(*) FROM public.collaboration_community_memberships),(SELECT count(*) FROM public.collaboration_events)").fetch_one(pool).await
    };
    let before_counts = read_counts().await?;

    let other_icon = "https://metadata-second.example.test/workspace.png";
    sqlx::query("UPDATE public.collaboration_communities SET icon=$2 WHERE community_id=$1")
        .bind(community.as_uuid())
        .bind(icon)
        .execute(pool)
        .await?;
    sqlx::query("INSERT INTO public.collaboration_communities (community_id,host,icon,lifecycle_state,aggregate_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,$3,'active',1,'zed',$2,now(),now(),now())")
        .bind(other.as_uuid()).bind(other_host).bind(other_icon).execute(pool).await?;
    let mut config = config.clone();
    config.collaboration_nostr_routes = Some(
        serde_json::json!([
            {"community_id":community.as_uuid(),"public_url":format!("https://{host}")},
            {"community_id":other.as_uuid(),"public_url":format!("https://{other_host}")},
            {"community_id":absent.as_uuid(),"public_url":format!("https://{absent_host}")}
        ])
        .to_string(),
    );
    let profile = collab::nostr::http::configured_profile_http_state(
        &config,
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?
    .ok_or_else(|| anyhow::anyhow!("configured metadata/profile service missing"))?;
    let metadata = profile.metadata();
    let root_metadata = metadata.clone();
    let router = collab::nostr::http::profile_http_router(profile)
        .merge(collab::nostr::http::metadata_http_router(metadata))
        .route(
            "/",
            axum::routing::get(move |headers: axum::http::HeaderMap| {
                let metadata = root_metadata.clone();
                async move {
                    collab::nostr::http::metadata_http_root_response(
                        Some(metadata.as_ref()),
                        &headers,
                        "retained status text".to_owned(),
                    )
                    .await
                }
            }),
        );
    let (status, document) = profile_fixture_metadata_request(&router, "/info", host, None).await?;
    anyhow::ensure!(
        status == StatusCode::OK && document["icon"] == icon,
        "production NIP11 must read actual canonical SQL community icon"
    );
    anyhow::ensure!(
        document["supported_nips"] == serde_json::json!([5, 11, 98])
            && document["software"] == "https://github.com/simtropolis/made"
            && document.get("supported_extensions").is_none()
            && document.get("self").is_none()
            && document["limitation"].get("max_subscriptions").is_none()
            && document["limitation"].get("max_subid_length").is_none(),
        "profile-only service advertised an unmounted relay capability"
    );
    let (status, root_document) = profile_fixture_metadata_request(
        &router,
        "/",
        host,
        Some("text/plain, application/nostr+json"),
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::OK && root_document == document,
        "production info and negotiated root must share the canonical renderer"
    );
    for accept in [None, Some("text/plain"), Some("*/*")] {
        let (status, text) = profile_fixture_metadata_request(&router, "/", host, accept).await?;
        anyhow::ensure!(
            status == StatusCode::OK && text == "retained status text",
            "public metadata displaced ordinary root status text"
        );
    }
    let (_, other_document) =
        profile_fixture_metadata_request(&router, "/info", other_host, None).await?;
    anyhow::ensure!(
        other_document["icon"] == other_icon && other_document["icon"] != document["icon"],
        "public NIP11 metadata crossed the trusted community host"
    );
    for unknown in [
        "unknown.example.test",
        absent_host,
        "profile-service.example.test@foreign.test",
    ] {
        let (status, generic) =
            profile_fixture_metadata_request(&router, "/info", unknown, None).await?;
        anyhow::ensure!(
            status == StatusCode::OK
                && generic.get("icon").is_none()
                && generic["supported_nips"] == document["supported_nips"],
            "missing/unknown/malformed host leaked optional tenant metadata"
        );
    }
    let (status, names) = profile_fixture_metadata_request(
        &router,
        "/.well-known/nostr.json?name=ordinary",
        host,
        None,
    )
    .await?;
    anyhow::ensure!(
        status == StatusCode::OK && names == serde_json::json!({"names":{},"relays":{}}),
        "production NIP05 returned an authored claim without an accepted name"
    );
    for lifecycle in ["archived", "quiescing", "fenced", "tombstone"] {
        sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1")
            .bind(community.as_uuid()).bind(lifecycle).execute(pool).await?;
        let (_, generic) = profile_fixture_metadata_request(&router, "/info", host, None).await?;
        anyhow::ensure!(
            generic.get("icon").is_none(),
            "nonserving community exposed public icon"
        );
    }
    sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state='active',host='conflicting-route.example.test',aggregate_version=aggregate_version+1 WHERE community_id=$1")
        .bind(community.as_uuid()).execute(pool).await?;
    let (_, generic) = profile_fixture_metadata_request(&router, "/info", host, None).await?;
    anyhow::ensure!(
        generic.get("icon").is_none(),
        "SQL/configured host conflict exposed community metadata"
    );
    sqlx::query("UPDATE public.collaboration_communities SET host=$2,icon='',aggregate_version=aggregate_version+1 WHERE community_id=$1")
        .bind(community.as_uuid()).bind(host).execute(pool).await?;
    let (_, cleared) = profile_fixture_metadata_request(&router, "/info", host, None).await?;
    anyhow::ensure!(
        cleared.get("icon").is_none(),
        "absolute empty community icon was not omitted"
    );
    let data_icon = format!("data:image/png;base64,{}", "A".repeat(3_000));
    sqlx::query("UPDATE public.collaboration_communities SET icon=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1")
        .bind(community.as_uuid()).bind(&data_icon).execute(pool).await?;
    let (_, current) = profile_fixture_metadata_request(&router, "/info", host, None).await?;
    anyhow::ensure!(
        current["icon"] == data_icon,
        "canonical community icon incorrectly inherited fixture metadata URL budget"
    );
    sqlx::query("UPDATE public.collaboration_communities SET icon=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1")
        .bind(community.as_uuid()).bind("bad\nicon").execute(pool).await?;
    let (_, corrupted) = profile_fixture_metadata_request(&router, "/info", host, None).await?;
    anyhow::ensure!(
        corrupted.get("icon").is_none(),
        "invalid canonical icon decoder failed to redact metadata"
    );
    sqlx::query("UPDATE public.collaboration_communities SET icon=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1")
        .bind(community.as_uuid()).bind(icon).execute(pool).await?;
    let own = TenantContext::establish(
        Some(TrustedTenantRoute::from_direct_host(community, host)?),
        &[],
    )?;
    let prior_version = repository
        .current_public_community_metadata(&own)
        .await?
        .ok_or_else(|| anyhow::anyhow!("prior serving metadata missing"))?
        .version;
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    let transaction = connection.begin().await?;
    transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "UPDATE public.collaboration_communities SET icon=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1",
        [community.as_uuid().into(), "transaction-only-icon".into()])).await?;
    let snapshot = repository
        .current_public_community_metadata_in_transaction(&transaction, &own)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!("supplied metadata transaction lost own current snapshot")
        })?;
    anyhow::ensure!(
        snapshot.icon.as_ref().map(|icon| icon.as_str()) == Some("transaction-only-icon")
            && snapshot.version.get() == prior_version.get() + 1,
        "public metadata reader used a nested pool or stale snapshot"
    );
    let foreign = TenantContext::establish(
        Some(TrustedTenantRoute::from_direct_host(other, other_host)?),
        &[],
    )?;
    anyhow::ensure!(
        matches!(
            repository
                .current_public_community_metadata_in_transaction(&transaction, &foreign)
                .await,
            Err(IdentityProfileRepositoryError::TenantBoundaryViolation)
        ),
        "public metadata reader reset an already-bound foreign transaction"
    );
    let configured = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT current_setting('app.community_id',true) AS tenant".to_owned(),
        ))
        .await?
        .ok_or_else(|| anyhow::anyhow!("metadata tenant fence setting missing"))?;
    anyhow::ensure!(
        configured.try_get::<String>("", "tenant")? == community.to_string(),
        "rejected public metadata read changed transaction tenant"
    );
    transaction.rollback().await?;
    let transaction = connection.begin().await?;
    let role = format!("public_metadata_{}", Uuid::new_v4().simple());
    let result = async {
        for command in [
            format!("CREATE ROLE {role} NOLOGIN NOBYPASSRLS"),
            format!("GRANT SELECT, UPDATE ON public.collaboration_communities TO {role}"),
            format!("SET LOCAL ROLE {role}"),
        ] {
            transaction
                .execute(Statement::from_string(DatabaseBackend::Postgres, command))
                .await?;
        }
        let snapshot = repository
            .current_public_community_metadata_in_transaction(&transaction, &own)
            .await?
            .ok_or_else(|| anyhow::anyhow!("nonbypass RLS hid own public metadata"))?;
        anyhow::ensure!(
            snapshot.icon.as_ref().map(|icon| icon.as_str()) == Some(icon),
            "nonbypass canonical metadata read changed own icon"
        );
        let row = transaction
            .query_one(Statement::from_string(
                DatabaseBackend::Postgres,
                "SELECT count(*)::text AS count FROM public.collaboration_communities".to_owned(),
            ))
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing RLS public metadata count"))?;
        anyhow::ensure!(
            row.try_get::<String>("", "count")? == "1",
            "nonbypass public metadata reader exposed foreign community rows"
        );
        let row = transaction
            .query_one(Statement::from_string(
                DatabaseBackend::Postgres,
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user".to_owned(),
            ))
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing metadata role capability check"))?;
        anyhow::ensure!(
            !row.try_get::<bool>("", "rolsuper")? && !row.try_get::<bool>("", "rolbypassrls")?,
            "public metadata fixture accidentally retained superuser RLS bypass"
        );
        Ok::<(), anyhow::Error>(())
    }
    .await;
    transaction.rollback().await?;
    result?;
    let row =
        sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1)")
            .bind(role)
            .fetch_one(pool)
            .await?;
    anyhow::ensure!(!row, "owned public metadata role survived rollback");
    let (status, final_document) =
        profile_fixture_metadata_request(&router, "/info", host, None).await?;
    anyhow::ensure!(
        status == StatusCode::OK && final_document["icon"] == icon,
        "public metadata uncommitted icon survived rollback or current read regressed"
    );
    let unavailable_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect_with(pool.connect_options().as_ref().clone())
        .await?;
    let unavailable = collab::nostr::http::configured_profile_http_state(
        &config,
        sea_orm::DatabaseConnection::from(unavailable_pool.clone()),
    )?
    .ok_or_else(|| anyhow::anyhow!("unavailable metadata service construction failed"))?;
    let unavailable_router = collab::nostr::http::metadata_http_router(unavailable.metadata());
    unavailable_pool.close().await;
    let (status, unavailable_document) =
        profile_fixture_metadata_request(&unavailable_router, "/info", host, None).await?;
    anyhow::ensure!(
        status == StatusCode::OK
            && unavailable_document.get("icon").is_none()
            && unavailable_document["supported_nips"] == document["supported_nips"],
        "SQL metadata infrastructure failure broke generic NIP11 document"
    );
    anyhow::ensure!(
        read_counts().await? == before_counts,
        "public metadata allocated identity, changed membership or persisted an event"
    );
    println!(
        "PROFILE_HTTP_PUBLIC_METADATA_SQL_VERIFIED: canonical same-pool current state/RLS/host/lifecycle/icon; genuine metadata mount/root composition; accepted NIP05 owned separately"
    );
    Ok(())
}

async fn profile_fixture_metadata_request(
    router: &axum::Router,
    path: &str,
    host: &str,
    accept: Option<&str>,
) -> anyhow::Result<(axum::http::StatusCode, serde_json::Value)> {
    use axum::body::HttpBody as _;
    use tower::ServiceExt as _;
    let mut request = axum::http::Request::builder()
        .uri(path)
        .header("Host", host)
        .header("X-Forwarded-Host", "metadata-second.example.test");
    if let Some(accept) = accept {
        request = request.header("Accept", accept);
    }
    let response = router
        .clone()
        .oneshot(request.body(axum::body::Body::empty())?)
        .await?;
    let status = response.status();
    let mut body = response.into_body();
    let mut bytes = Vec::new();
    while let Some(chunk) = body.data().await {
        bytes.extend_from_slice(&chunk?);
    }
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => serde_json::Value::String(String::from_utf8(bytes)?),
        }
    };
    Ok((status, value))
}

async fn profile_fixture_accepted_names(
    pool: &sqlx::PgPool,
    config: &collab::Config,
    repository: &Arc<IdentityProfileRepository>,
    tenant: &TenantContext,
    host: &str,
    now: u64,
) -> anyhow::Result<()> {
    use axum::http::StatusCode;
    use collab::identity::profile_repository::IdentitySubjectStatus;
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    let state = collab::nostr::http::configured_profile_http_state(
        config,
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?
    .ok_or_else(|| anyhow::anyhow!("accepted name production state missing"))?;
    let router = collab::nostr::http::profile_http_router(state.clone())
        .merge(collab::nostr::http::metadata_http_router(state.metadata()));
    let first = signed_registration(130, None, now, 130);
    let second = signed_registration(131, None, now, 131);
    let (first, second) = tokio::join!(
        authenticate_registration(repository, tenant, &first),
        authenticate_registration(repository, tenant, &second)
    );
    let first = first?;
    let second = second?;
    anyhow::ensure!(
        first.principal_id() != second.principal_id(),
        "concurrent signed registrations reused identity authority"
    );
    for principal in [&first, &second] {
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'member','active',1,now(),now(),'zed',$3,now())")
            .bind(tenant.community_id().as_uuid()).bind(principal.principal_id().as_uuid()).bind(principal.principal_id().to_string()).execute(pool).await?;
    }
    let rights_before: serde_json::Value = sqlx::query_scalar("SELECT jsonb_agg(jsonb_build_array(profile_id,principal_id,author_public_key,profile_kind,owner_public_key,first_owner_proof_event_id,first_owner_auth_proof,registration_source_event_id) ORDER BY author_public_key) FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key IN ($2,$3)")
        .bind(tenant.community_id().as_uuid()).bind(public_key(130).as_bytes().to_vec()).bind(public_key(131).as_bytes().to_vec()).fetch_one(pool).await?;
    let alice = profile_fixture_signed_event(
        130,
        now.saturating_sub(40),
        Vec::new(),
        &serde_json::json!({"name":"first", "nip05":format!(" Alice@{host} ")}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &alice).await?;
    let bob = profile_fixture_signed_event(
        131,
        now.saturating_sub(40),
        Vec::new(),
        &serde_json::json!({"name":"second", "nip05":format!("Bob@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 131, host, &bob).await?;
    profile_fixture_expect_name(&router, host, "ALICE", Some(130)).await?;
    profile_fixture_expect_name(&router, host, "bob", Some(131)).await?;
    let collision = profile_fixture_signed_event(
        131,
        now.saturating_sub(39),
        Vec::new(),
        &serde_json::json!({"name":"second changed", "nip05":format!("Alice@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 131, host, &collision).await?;
    profile_fixture_expect_name(&router, host, "alice", Some(130)).await?;
    profile_fixture_expect_name(&router, host, "bob", Some(131)).await?;
    let authored = repository
        .current_authored_metadata_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(131).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("contested signed authored metadata missing"))?;
    anyhow::ensure!(
        authored.metadata().value.name.as_deref() == Some("second changed")
            && authored.metadata().value.nip05_handle.as_deref()
                == Some(format!("Alice@{host}").as_str()),
        "accepted-name collision rejected genuine EVENT or rewrote authored display claim"
    );
    let renamed = profile_fixture_signed_event(
        130,
        now.saturating_sub(38),
        Vec::new(),
        &serde_json::json!({"nip05":format!("Carol@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &renamed).await?;
    profile_fixture_expect_name(&router, host, "alice", None).await?;
    profile_fixture_expect_name(&router, host, "carol", Some(130)).await?;
    for (key, original) in [(130, &alice), (131, &collision)] {
        let (status, response) = profile_fixture_http_request(
            &router,
            key,
            host,
            "/events",
            &profile_fixture_wire(original),
            true,
            false,
            false,
        )
        .await?;
        anyhow::ensure!(
            status == StatusCode::OK
                && response["accepted"] == true
                && response["message"]
                    .as_str()
                    .is_some_and(|message| message.starts_with("duplicate:")),
            "old name EVENT retry lost canonical duplicate receipt"
        );
    }
    profile_fixture_expect_name(&router, host, "alice", None).await?;
    profile_fixture_expect_name(&router, host, "bob", Some(131)).await?;
    profile_fixture_expect_name(&router, host, "carol", Some(130)).await?;
    for (offset, claim) in [
        None,
        Some("".to_owned()),
        Some("off@foreign.example.test".to_owned()),
        Some("without-domain".to_owned()),
    ]
    .into_iter()
    .enumerate()
    {
        let mut metadata = serde_json::json!({"name":"clearing"});
        if let Some(claim) = claim {
            metadata["nip05"] = serde_json::json!(claim);
        }
        let signed = profile_fixture_signed_event(
            130,
            now.saturating_sub(37)
                .saturating_add(u64::try_from(offset)?),
            Vec::new(),
            &metadata.to_string(),
        );
        profile_fixture_submit_name(&router, 130, host, &signed).await?;
        profile_fixture_expect_name(&router, host, "carol", None).await?;
    }
    let maximum_local = "a".repeat(
        255usize
            .checked_sub(host.chars().count() + 1)
            .ok_or_else(|| anyhow::anyhow!("fixture host too long"))?,
    );
    let maximum = profile_fixture_signed_event(
        130,
        now.saturating_sub(32),
        Vec::new(),
        &serde_json::json!({"nip05":format!("{maximum_local}@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &maximum).await?;
    profile_fixture_expect_name(&router, host, &maximum_local, Some(130)).await?;
    let over = profile_fixture_signed_event(
        130,
        now.saturating_sub(31),
        Vec::new(),
        &serde_json::json!({"nip05":format!("a{maximum_local}@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &over).await?;
    profile_fixture_expect_name(&router, host, &maximum_local, Some(130)).await?;
    profile_fixture_expect_name(&router, host, &format!("a{maximum_local}"), None).await?;
    let oversized_authored = repository
        .current_authored_metadata_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(130).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("storage-bound signed metadata head disappeared"))?;
    anyhow::ensure!(
        oversized_authored.metadata().source_event_id.as_bytes() == over.claimed_id.as_bytes()
            && oversized_authored.metadata().value.nip05_handle.as_deref()
                == Some(format!("a{maximum_local}@{host}").as_str()),
        "oversized valid same-host claim failed to retain genuine authored head"
    );
    let retained_name_source = sqlx::query_as::<_, (Vec<u8>, Vec<u8>, String, i16)>(
        "SELECT accepted_nip05_source_event_id,nip05_processed_event_id,nip05_projection_host,projection_hash_version FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key=$2",
    )
    .bind(tenant.community_id().as_uuid())
    .bind(public_key(130).as_bytes().to_vec())
    .fetch_one(pool)
    .await?;
    anyhow::ensure!(
        retained_name_source.0.as_slice() == maximum.claimed_id.as_bytes()
            && retained_name_source.1.as_slice() == over.claimed_id.as_bytes()
            && retained_name_source.2 == host
            && retained_name_source.3 == 4,
        "storage-bound claim replaced accepted source or lost current decision provenance"
    );
    let unicode = profile_fixture_signed_event(
        130,
        now.saturating_sub(30),
        Vec::new(),
        &serde_json::json!({"nip05":format!("İÉ@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &unicode).await?;
    profile_fixture_expect_name(&router, host, "İÉ", Some(130)).await?;
    let unicode_local = "🦀".repeat(maximum_local.chars().count());
    let unicode_claim = format!("{unicode_local}@{host}");
    anyhow::ensure!(
        unicode_claim.chars().count() == 255 && unicode_claim.len() > 320,
        "fixture did not cross authored byte limit with exact255 Unicode fullhandle"
    );
    let unicode_maximum = profile_fixture_signed_event(
        130,
        now.saturating_sub(29),
        Vec::new(),
        &serde_json::json!({"nip05":unicode_claim}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &unicode_maximum).await?;
    profile_fixture_expect_name(&router, host, &unicode_local, Some(130)).await?;
    let unicode_authored = repository
        .current_authored_metadata_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(130).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("bounded Unicode signed head absent"))?;
    anyhow::ensure!(
        unicode_authored.metadata().source_event_id.as_bytes()
            == unicode_maximum.claimed_id.as_bytes()
            && unicode_authored.metadata().value.nip05_handle.as_deref()
                == Some(unicode_claim.as_str()),
        "Unicode producer/read journey changed raw claim or signed head attribution"
    );
    let unicode_storage_bound = profile_fixture_signed_event(
        130,
        now.saturating_sub(28),
        Vec::new(),
        &serde_json::json!({"nip05":format!("🦀{unicode_local}@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &unicode_storage_bound).await?;
    profile_fixture_expect_name(&router, host, &unicode_local, Some(130)).await?;
    let retained_unicode_source = sqlx::query_as::<_, (Vec<u8>, Vec<u8>)>(
        "SELECT accepted_nip05_source_event_id,nip05_processed_event_id FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key=$2",
    ).bind(tenant.community_id().as_uuid()).bind(public_key(130).as_bytes().to_vec()).fetch_one(pool).await?;
    anyhow::ensure!(
        retained_unicode_source.0.as_slice() == unicode_maximum.claimed_id.as_bytes()
            && retained_unicode_source.1.as_slice() == unicode_storage_bound.claimed_id.as_bytes(),
        "oversized Unicode storage claim replaced prior source or lost current signed provenance"
    );
    for (offset, local_scalars) in [321usize, 400usize].into_iter().enumerate() {
        let local = "🦀".repeat(
            local_scalars
                .checked_sub(host.chars().count() + 1)
                .ok_or_else(|| anyhow::anyhow!("fixture host exceeds scalar budget"))?,
        );
        let rejected = profile_fixture_signed_event(
            130,
            now.saturating_sub(27)
                .saturating_add(u64::try_from(offset)?),
            Vec::new(),
            &serde_json::json!({"nip05":format!("{local}@{host}")}).to_string(),
        );
        let (status, response) = profile_fixture_http_request(
            &router,
            130,
            host,
            "/events",
            &profile_fixture_wire(&rejected),
            true,
            false,
            false,
        )
        .await?;
        anyhow::ensure!(
            status == StatusCode::FORBIDDEN && response["accepted"] == false,
            "unbounded Unicode authored metadata was admitted: status={status} response={response}"
        );
        let persisted: i64=sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events WHERE community_id=$1 AND event_id=$2")
            .bind(tenant.community_id().as_uuid()).bind(rejected.claimed_id.as_bytes().to_vec()).fetch_one(pool).await?;
        anyhow::ensure!(
            persisted == 0,
            "oversized Unicode rejection left canonical EVENT mutation"
        );
        profile_fixture_expect_name(&router, host, &unicode_local, Some(130)).await?;
        anyhow::ensure!(
            repository
                .current_authored_metadata_by_public_key(
                    tenant,
                    NostrPublicKey::from_bytes(*public_key(130).as_bytes())
                )
                .await?
                .is_some_and(|authored| authored.metadata().source_event_id.as_bytes()
                    == unicode_storage_bound.claimed_id.as_bytes()),
            "oversized Unicode rejection changed genuine current head"
        );
    }
    let unicode_restored = profile_fixture_signed_event(
        130,
        now.saturating_sub(25),
        Vec::new(),
        &serde_json::json!({"nip05":format!("İÉ@{host}")}).to_string(),
    );
    profile_fixture_submit_name(&router, 130, host, &unicode_restored).await?;
    profile_fixture_expect_name(&router, host, "İÉ", Some(130)).await?;
    profile_fixture_expect_name(&router, host, "missing", None).await?;
    profile_fixture_expect_name(&router, "unmapped.example.test", "İÉ", None).await?;
    let foreign_host = "metadata-second.example.test";
    let foreign_id: Uuid = sqlx::query_scalar(
        "SELECT community_id FROM public.collaboration_communities WHERE host=$1",
    )
    .bind(foreign_host)
    .fetch_one(pool)
    .await?;
    let foreign = TenantContext::establish(
        Some(TrustedTenantRoute::from_direct_host(
            CommunityId::from_uuid(foreign_id),
            foreign_host,
        )?),
        &[],
    )?;
    let mut other_config = config.clone();
    other_config.collaboration_nostr_routes = Some(
        serde_json::json!([
            {"community_id":tenant.community_id().as_uuid(),"public_url":format!("https://{host}")},
            {"community_id":foreign_id,"public_url":format!("https://{foreign_host}")}
        ])
        .to_string(),
    );
    let other_state = collab::nostr::http::configured_profile_http_state(
        &other_config,
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?
    .ok_or_else(|| anyhow::anyhow!("two-tenant accepted-name production state missing"))?;
    let other_router = collab::nostr::http::profile_http_router(other_state.clone()).merge(
        collab::nostr::http::metadata_http_router(other_state.metadata()),
    );
    let foreign_member = authenticate_registration(
        repository,
        &foreign,
        &signed_registration(130, None, now, 134),
    )
    .await?;
    sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'member','active',1,now(),now(),'zed',$3,now())")
        .bind(foreign.community_id().as_uuid()).bind(foreign_member.principal_id().as_uuid()).bind(foreign_member.principal_id().to_string()).execute(pool).await?;
    let foreign_name = profile_fixture_signed_event(
        130,
        now.saturating_sub(30),
        Vec::new(),
        &serde_json::json!({"nip05":format!("İÉ@{foreign_host}")}).to_string(),
    );
    let foreign_result = async {
        profile_fixture_submit_name(&other_router, 130, foreign_host, &foreign_name).await?;
        profile_fixture_expect_name(&other_router, foreign_host, "İÉ", Some(130)).await?;
        Ok::<(), anyhow::Error>(())
    }
    .await;
    let cleanup = profile_fixture_redis_cleanup(
        &url::Url::parse(
            config
                .collaboration_redis_url
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("fixture Redis missing"))?,
        )?,
        foreign.community_id(),
    )
    .await;
    match (foreign_result, cleanup) {
        (Ok(()), Ok(())) => {}
        (result, cleanup) => anyhow::bail!(
            "owned foreign NIP05 request={result:?}; owned foreign Redis cleanup={cleanup:?}"
        ),
    }
    let race_first = signed_registration(132, None, now, 132);
    let race_second = signed_registration(133, None, now, 133);
    let (race_first, race_second) = tokio::join!(
        authenticate_registration(repository, tenant, &race_first),
        authenticate_registration(repository, tenant, &race_second)
    );
    for principal in [race_first?, race_second?] {
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'member','active',1,now(),now(),'zed',$3,now())")
            .bind(tenant.community_id().as_uuid()).bind(principal.principal_id().as_uuid()).bind(principal.principal_id().to_string()).execute(pool).await?;
    }
    let race_first = profile_fixture_signed_event(
        132,
        now.saturating_sub(29),
        Vec::new(),
        &serde_json::json!({"nip05":format!("RACE@{host}")}).to_string(),
    );
    let race_second = profile_fixture_signed_event(
        133,
        now.saturating_sub(29),
        Vec::new(),
        &serde_json::json!({"nip05":format!("race@{host}")}).to_string(),
    );
    let (left, right) = tokio::join!(
        profile_fixture_submit_name(&router, 132, host, &race_first),
        profile_fixture_submit_name(&router, 133, host, &race_second)
    );
    left?;
    right?;
    let winner = repository
        .current_accepted_nip05(tenant, "race")
        .await?
        .ok_or_else(|| anyhow::anyhow!("concurrent genuine name claims lost both owners"))?;
    anyhow::ensure!(
        [public_key(132), public_key(133)].contains(&winner),
        "concurrent accepted-name claims produced invented identity"
    );
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1 AND accepted_nip05_name=$2")
        .bind(tenant.community_id().as_uuid()).bind(format!("race@{host}")).fetch_one(pool).await?;
    anyhow::ensure!(
        count == 1,
        "same-tenant concurrent name claims violated uniqueness"
    );
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    let transaction = connection.begin().await?;
    let role = format!("accepted_name_{}", Uuid::new_v4().simple());
    let result = async {
        for command in [format!("CREATE ROLE {role} NOLOGIN NOBYPASSRLS"), format!("GRANT SELECT,UPDATE ON public.collaboration_communities TO {role}"), format!("GRANT SELECT ON public.collaboration_identity_profiles,public.collaboration_events,public.collaboration_event_heads TO {role}"), format!("SET LOCAL ROLE {role}")] {
            transaction.execute(Statement::from_string(DatabaseBackend::Postgres,command)).await?;
        }
        anyhow::ensure!(repository.current_accepted_nip05_in_transaction(&transaction, tenant, "i̇é").await? == Some(public_key(130)), "nonbypass accepted-name reader hid exact own source");
        anyhow::ensure!(matches!(repository.current_accepted_nip05_in_transaction(&transaction,&foreign,"i̇é").await, Err(IdentityProfileRepositoryError::TenantBoundaryViolation)), "accepted-name reader reset supplied foreign tenant");
        let capabilities = transaction.query_one(Statement::from_string(DatabaseBackend::Postgres,"SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user".to_owned())).await?.ok_or_else(|| anyhow::anyhow!("accepted name role missing"))?;
        anyhow::ensure!(!capabilities.try_get::<bool>("","rolsuper")? && !capabilities.try_get::<bool>("","rolbypassrls")?,"accepted name role retained RLS bypass");
        let leaked = transaction.query_one(Statement::from_sql_and_values(DatabaseBackend::Postgres,"SELECT count(*)::text AS count FROM public.collaboration_identity_profiles WHERE community_id=$1",[foreign.community_id().as_uuid().into()])).await?.ok_or_else(|| anyhow::anyhow!("foreign RLS row count missing"))?;
        anyhow::ensure!(leaked.try_get::<String>("","count")? == "0", "nonbypass name read exposed foreign row");
        Ok::<(),anyhow::Error>(())
    }.await;
    transaction.rollback().await?;
    result?;
    let role_exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=$1)")
            .bind(&role)
            .fetch_one(pool)
            .await?;
    anyhow::ensure!(!role_exists, "accepted-name fixture role survived rollback");
    let transaction = connection.begin().await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id',$1,true)",
            [tenant.community_id().to_string().into()],
        ))
        .await?;
    transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,"UPDATE public.collaboration_identity_profiles SET profile_hash=decode(repeat('00',32),'hex'),aggregate_version=aggregate_version+1 WHERE community_id=$1 AND author_public_key=$2",[tenant.community_id().as_uuid().into(),public_key(130).as_bytes().to_vec().into()])).await?;
    let result = async {
        anyhow::ensure!(
            matches!(
                repository
                    .current_accepted_nip05_in_transaction(&transaction, tenant, "i̇é")
                    .await,
                Err(IdentityProfileRepositoryError::InvalidRecord)
            ),
            "accepted-name reader trusted corrupted projection hash"
        );
        Ok::<(), anyhow::Error>(())
    }
    .await;
    transaction.rollback().await?;
    result?;
    let transaction = connection.begin().await?;
    let result = async {
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,"ALTER TABLE public.collaboration_events RENAME TO accepted_name_original_events".to_owned())).await?;
        transaction.execute(Statement::from_string(DatabaseBackend::Postgres,"CREATE VIEW public.collaboration_events AS SELECT community_id,event_id,author_public_key,event_created_at,kind,tags,content,canonical_event_bytes,decode(repeat('00',64),'hex') AS signature,signature_state,verified_at,persistence_class,discriminator,received_at FROM public.accepted_name_original_events".to_owned())).await?;
        anyhow::ensure!(matches!(repository.current_accepted_nip05_in_transaction(&transaction,tenant,"i̇é").await,Err(IdentityProfileRepositoryError::InvalidRecord) | Err(IdentityProfileRepositoryError::EventRepository(_))),"accepted-name read trusted persisted verification flag over real signed source");
        Ok::<(),anyhow::Error>(())
    }.await;
    transaction.rollback().await?;
    result?;
    profile_fixture_expect_name(&router, host, "İÉ", Some(130)).await?;
    for lifecycle in ["archived", "quiescing", "fenced", "tombstone"] {
        sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state=$2,aggregate_version=aggregate_version+1 WHERE community_id=$1").bind(tenant.community_id().as_uuid()).bind(lifecycle).execute(pool).await?;
        profile_fixture_expect_name(&router, host, "İÉ", None).await?;
    }
    sqlx::query("UPDATE public.collaboration_communities SET lifecycle_state='active',aggregate_version=aggregate_version+1 WHERE community_id=$1").bind(tenant.community_id().as_uuid()).execute(pool).await?;
    let unavailable = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect_with(pool.connect_options().as_ref().clone())
        .await?;
    let unavailable_state = collab::nostr::http::configured_profile_http_state(
        config,
        sea_orm::DatabaseConnection::from(unavailable.clone()),
    )?
    .ok_or_else(|| anyhow::anyhow!("outage profile state absent"))?;
    let unavailable_router =
        collab::nostr::http::metadata_http_router(unavailable_state.metadata());
    unavailable.close().await;
    profile_fixture_expect_name(&unavailable_router, host, "İÉ", None).await?;
    let rights_after: serde_json::Value = sqlx::query_scalar("SELECT jsonb_agg(jsonb_build_array(profile_id,principal_id,author_public_key,profile_kind,owner_public_key,first_owner_proof_event_id,first_owner_auth_proof,registration_source_event_id) ORDER BY author_public_key) FROM public.collaboration_identity_profiles WHERE community_id=$1 AND author_public_key IN ($2,$3)")
        .bind(tenant.community_id().as_uuid()).bind(public_key(130).as_bytes().to_vec()).bind(public_key(131).as_bytes().to_vec()).fetch_one(pool).await?;
    anyhow::ensure!(
        rights_after == rights_before,
        "accepted-name projection changed registration IDs/kind/owner authority"
    );
    anyhow::ensure!(
        repository
            .current_subject_by_public_key(
                tenant,
                NostrPublicKey::from_bytes(*public_key(130).as_bytes())
            )
            .await?
            .is_some_and(
                |subject| subject.status() == IdentitySubjectStatus::RegisteredUnclassified
            ),
        "accepted name inferred Human/agent classification"
    );
    let mut rollback = pool.begin().await?;
    let refused = sqlx::raw_sql(ACCEPTED_NAMES_DOWN)
        .execute(&mut *rollback)
        .await;
    anyhow::ensure!(
        matches!(refused,Err(sqlx::Error::Database(ref error)) if error.code().as_deref()==Some("23514")),
        "accepted-name rollback discarded retained signed name decisions"
    );
    rollback.rollback().await?;
    println!(
        "PROFILE_HTTP_ACCEPTED_NIP05_SQL_VERIFIED: genuine admitted current heads; collision/rename/clear/Unicode255/concurrent pool1/tenant/RLS/sourcehash/signature/lifecycle/outage; no identity authority uplift"
    );
    Ok(())
}

async fn profile_fixture_submit_name(
    router: &axum::Router,
    key: u8,
    host: &str,
    signed: &SignedEvent,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        signed.event.public_key == public_key(key),
        "NIP05 fixture HTTP signer differs from genuine EVENT author"
    );
    let (status, response) = profile_fixture_http_request(
        router,
        key,
        host,
        "/events",
        &profile_fixture_wire(signed),
        true,
        false,
        false,
    )
    .await?;
    anyhow::ensure!(
        status == axum::http::StatusCode::OK && response["accepted"] == true,
        "genuine admitted kind0 name claim was rejected: status={status} body={response}"
    );
    Ok(())
}

async fn profile_fixture_expect_name(
    router: &axum::Router,
    host: &str,
    name: &str,
    key: Option<u8>,
) -> anyhow::Result<()> {
    let encoded = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("name", name)
        .finish();
    let (status, response) = profile_fixture_metadata_request(
        router,
        &format!("/.well-known/nostr.json?{encoded}"),
        host,
        None,
    )
    .await?;
    let canonical = name.to_lowercase();
    let expected = key
        .map(|key| {
            let key = public_key(key).to_hex();
            serde_json::json!({"names":{canonical:key},"relays":{key:[format!("wss://{host}")]}})
        })
        .unwrap_or_else(|| serde_json::json!({"names":{},"relays":{}}));
    anyhow::ensure!(
        status == axum::http::StatusCode::OK && response == expected,
        "production accepted NIP05 decision differs from signed projection: name={name} expected={expected} status={status} actual={response}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_virtual_agent_messages_actual_auth_immutable_author() -> anyhow::Result<()> {
    let base_url = std::env::var("COLLAB_TEST_DATABASE_URL").map_err(|_| {
        anyhow::anyhow!("COLLAB_TEST_DATABASE_URL required for virtual message qualification")
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
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(60),
                exercise_virtual_agent_messages(&pool),
            )
            .await
            .map_err(|_| anyhow::anyhow!("single-connection virtual message journey deadlocked"));
            pool.close().await;
            result.and_then(|result| result)
        }
        Err(error) => Err(error.into()),
    };
    let cleanup = sqlx::Postgres::drop_database(&url).await;
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(anyhow::anyhow!(
            "virtual message DB cleanup failed: {error}"
        )),
        (Err(error), Err(cleanup)) => Err(anyhow::anyhow!(
            "virtual message failed: {error:#}; cleanup failed: {cleanup}"
        )),
    }
}

async fn virtual_message_admission(
    pool: &sqlx::PgPool,
    profiles: &IdentityProfileRepository,
    tenant: &TenantContext,
    principal: &collaboration_domain::AuthenticatedPrincipal,
    channel_id: collaboration_domain::AggregateId,
    now: u64,
) -> anyhow::Result<collab::messages::channel_admission::AuthorizedChannel> {
    use sea_orm::TransactionTrait as _;
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    let transaction = connection.begin().await?;
    let result = collab::messages::channel_admission::admit_authenticated_channel_in_transaction(
        &transaction,
        profiles,
        tenant,
        principal,
        channel_id,
        now * 1000,
    )
    .await;
    match result {
        Ok(authorization) => {
            transaction.commit().await?;
            Ok(authorization)
        }
        Err(error) => {
            transaction.rollback().await?;
            Err(error)
        }
    }
}

fn virtual_message_command(
    tenant: &TenantContext,
    authorization: collab::messages::channel_admission::AuthorizedChannel,
    message_id: collaboration_domain::AggregateId,
    operation_id: collaboration_domain::OperationId,
    kind: collab::messages::channel_mutation::MessageOperationKind,
    expected_version: Option<collaboration_domain::AggregateVersion>,
    source: SignedEvent,
    now: u64,
) -> anyhow::Result<
    collab::collaboration_command::DomainCommand<
        collab::messages::channel_mutation::MessageOperation,
    >,
> {
    let signed_event = VerifiedEventRecord::new(
        tenant.community_id(),
        source,
        EventVerificationState::Live,
        now * 1000,
        TimestampPolicy::Bounded {
            now,
            max_past_seconds: 120,
            max_future_seconds: 120,
        },
    )?;
    Ok(collab::collaboration_command::DomainCommand::new(
        operation_id,
        tenant.clone(),
        authorization.principal.clone(),
        expected_version,
        None,
        collab::collaboration_command::CommandAdapter::NostrInProcess,
        collab::messages::channel_mutation::MessageOperation {
            authorization,
            kind,
            message_id,
            expected_version,
            signed_event: Some(signed_event),
            reaction: None,
            related_reaction_event_id: None,
            acknowledged_outbox_sequence: None,
        },
    ))
}

async fn exercise_virtual_agent_messages(pool: &sqlx::PgPool) -> anyhow::Result<()> {
    use collab::{
        collaboration_command::DomainCommandSink as _,
        db::collaboration::outbox::TransactionalCommandOutbox,
        messages::{
            channel_mutation::{CanonicalMessageMutation, MessageOperationKind},
            channel_service::CanonicalMessageService,
        },
    };
    use collaboration_domain::{
        AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
        AuthorizationScope, MembershipRole, NostrAuthenticationMethod, OperationId,
        PrincipalScopes,
    };
    profile_fixture_apply_canonical_schema(pool).await?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let tenant = tenant(501);
    let profiles = Arc::new(IdentityProfileRepository::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?);
    let owner =
        authenticate_registration(&profiles, &tenant, &signed_registration(151, None, now, 1))
            .await?;
    let first_auth = signed_registration(152, Some(151), now, 2);
    let agent = authenticate_registration(&profiles, &tenant, &first_auth).await?;
    let agent_record = profiles
        .current_registration_by_public_key(
            &tenant,
            NostrPublicKey::from_bytes(*public_key(152).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("actual agent registration absent"))?;
    let channel = AggregateId::from_uuid(Uuid::new_v4());
    sqlx::query("INSERT INTO public.collaboration_communities (community_id,host,lifecycle_state,aggregate_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,'virtual-agent.example.test','active',1,'zed','virtual-fixture',now(),now(),now())")
        .bind(tenant.community_id().as_uuid()).execute(pool).await?;
    sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'admin','active',1,now(),now(),'zed','actual-owner-fixture',now())")
        .bind(tenant.community_id().as_uuid()).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    sqlx::query("INSERT INTO public.collaboration_channels (community_id,channel_id,name,channel_type,visibility,lifecycle_state,creator_principal_id,channel_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,'actual virtual messages','stream','open','active',$3,1,'zed','virtual-fixture',now(),now(),now())")
        .bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    // Preserve a genuine ordinary author's old-schema row across DOWN/UP in this owned DB.
    sqlx::raw_sql(include_str!(
        "../migrations/20261010000600_collaboration_virtual_agent_message_authors.down.sql"
    ))
    .execute(pool)
    .await?;
    let legacy_source = sign_restriction_event(
        151,
        CanonicalEvent::new(
            public_key(151),
            now,
            nostr_compat::generated_kinds::KIND_STREAM_MESSAGE as u16,
            vec![vec!["h".into(), channel.to_string()]],
            "retained ordinary author".into(),
        ),
    );
    let legacy_record = VerifiedEventRecord::new(
        tenant.community_id(),
        legacy_source.clone(),
        EventVerificationState::Live,
        now * 1000,
        TimestampPolicy::Bounded {
            now,
            max_past_seconds: 120,
            max_future_seconds: 120,
        },
    )?;
    let events = EventRepository::new(sea_orm::DatabaseConnection::from(pool.clone()))?;
    events
        .store(
            &tenant,
            &legacy_record,
            collab::db::collaboration::persistence_policy::EventPersistencePolicy::evaluate(
                legacy_source.event.kind,
                collab::db::collaboration::persistence_policy::PrivacyAdmission::community(),
            )?,
        )
        .await?;
    let legacy_message_id = AggregateId::from_uuid(Uuid::new_v4());
    sqlx::query("INSERT INTO public.collaboration_messages (community_id,message_id,channel_id,source_event_id,current_event_id,author_principal_id,message_created_at,lifecycle_state,message_version,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,$4,$4,$5,$6::numeric,'active',1,'zed','ordinary-author-fixture',now())")
        .bind(tenant.community_id().as_uuid()).bind(legacy_message_id.as_uuid()).bind(channel.as_uuid())
        .bind(legacy_source.claimed_id.as_bytes().to_vec()).bind(owner.principal_id().as_uuid()).bind(now.to_string()).execute(pool).await?;
    let old_row: serde_json::Value = sqlx::query_scalar("SELECT to_jsonb(message) FROM public.collaboration_messages AS message WHERE message_id=$1")
        .bind(legacy_message_id.as_uuid()).fetch_one(pool).await?;
    sqlx::raw_sql(include_str!(
        "../migrations/20261010000600_collaboration_virtual_agent_message_authors.up.sql"
    ))
    .execute(pool)
    .await?;
    let unchanged_row: serde_json::Value = sqlx::query_scalar("SELECT to_jsonb(message)-ARRAY['agent_author_profile_id','persistent_member_principal_id','agent_author_owner_principal_id','agent_author_proof_event_id','agent_author_attribution'] FROM public.collaboration_messages AS message WHERE message_id=$1")
        .bind(legacy_message_id.as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        old_row == unchanged_row,
        "schema30 changed historical ordinary message bytes"
    );
    let unknown = Uuid::new_v4();
    let unknown_member = sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,'member','active',1,now(),now(),'zed','unknown-fixture',now())")
        .bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(unknown).execute(pool).await.expect_err("ordinary human membership FK must remain exact");
    anyhow::ensure!(
        unknown_member
            .as_database_error()
            .and_then(|error| error.code())
            .as_deref()
            == Some("23503"),
        "ordinary membership FK failed for wrong reason"
    );
    let wrong_profile = sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,agent_profile_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,$4,'member','active',1,now(),now(),'zed','wrong-profile-fixture',now())")
        .bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(unknown).bind(agent_record.profile_id().as_uuid()).execute(pool).await.expect_err("agent profile principal equality must be exact");
    anyhow::ensure!(
        matches!(
            wrong_profile
                .as_database_error()
                .and_then(|error| error.code())
                .as_deref(),
            Some("23503" | "23514")
        ),
        "wrong profile failed for unrelated reason"
    );
    let plain_owner_profile = profiles
        .current_registration_by_public_key(
            &tenant,
            NostrPublicKey::from_bytes(*public_key(151).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("actual plain-key owner registration missing"))?;
    let unclassified_reference = sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,agent_profile_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,$4,'member','active',1,now(),now(),'zed','unclassified-profile-fixture',now())")
        .bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(owner.principal_id().as_uuid())
        .bind(plain_owner_profile.profile_id().as_uuid()).execute(pool).await.expect_err("registration-only profile is not a verified agent");
    anyhow::ensure!(
        unclassified_reference
            .as_database_error()
            .and_then(|error| error.code())
            .as_deref()
            == Some("23514"),
        "unclassified profile rejection must come from canonical typed guard"
    );
    let no_join = virtual_message_admission(pool, &profiles, &tenant, &agent, channel, now).await;
    anyhow::ensure!(
        no_join.is_err(),
        "virtual agent bypassed explicit channel membership"
    );
    for (principal, profile) in [
        (owner.principal_id(), None),
        (
            agent.principal_id(),
            Some(agent_record.profile_id().as_uuid()),
        ),
    ] {
        sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,agent_profile_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,$4,'member','active',1,now(),now(),'zed','explicit-channel-fixture',now())")
            .bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(principal.as_uuid()).bind(profile).execute(pool).await?;
    }
    let admission =
        virtual_message_admission(pool, &profiles, &tenant, &agent, channel, now).await?;
    anyhow::ensure!(
        admission.community_membership.role == MembershipRole::Member,
        "agent inherited owner admin role"
    );
    anyhow::ensure!(
        admission
            .authorize(AuthorizationAction::Manage, now * 1000)
            .is_err(),
        "virtual membership granted Manage"
    );
    let legacy_scopes = PrincipalScopes::new([AuthorizationScope::new(
        collab::messages::channel_admission::CHANNEL_WRITE_SCOPE,
    )?])?;
    let legacy_only = AuthenticatedPrincipal::owner_attested_agent(
        agent.principal_id(),
        tenant.community_id(),
        agent_record
            .profile()
            .ok_or_else(|| anyhow::anyhow!("verified agent profile absent"))?,
        NostrAuthenticationMethod::Nip42,
        legacy_scopes,
    )?;
    let legacy_admission =
        virtual_message_admission(pool, &profiles, &tenant, &legacy_only, channel, now).await?;
    anyhow::ensure!(
        legacy_admission
            .authorize(AuthorizationAction::Write, now * 1000)
            .is_err(),
        "legacy scope uplifted Nostr message authority"
    );
    let no_scopes = AuthenticatedPrincipal::owner_attested_agent(
        agent.principal_id(),
        tenant.community_id(),
        agent_record
            .profile()
            .ok_or_else(|| anyhow::anyhow!("agent profile absent"))?,
        NostrAuthenticationMethod::Nip42,
        PrincipalScopes::new([])?,
    )?;
    let empty_admission =
        virtual_message_admission(pool, &profiles, &tenant, &no_scopes, channel, now).await?;
    for action in [AuthorizationAction::Read, AuthorizationAction::Write] {
        anyhow::ensure!(
            empty_admission.authorize(action, now * 1000).is_err(),
            "missing canonical scope authorized a message"
        );
    }
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    let sink = Arc::new(
        TransactionalCommandOutbox::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
            CanonicalMessageMutation::new(sea_orm::DatabaseConnection::from(pool.clone()))?,
        )
        .map_err(|_| anyhow::anyhow!("virtual fixture requires PostgreSQL"))?,
    );
    let message_id = AggregateId::from_uuid(Uuid::new_v4());
    let operation_id = OperationId::from_uuid(Uuid::new_v4());
    let source = sign_restriction_event(
        152,
        CanonicalEvent::new(
            public_key(152),
            now,
            nostr_compat::generated_kinds::KIND_STREAM_MESSAGE as u16,
            vec![vec!["h".into(), channel.to_string()]],
            "genuine virtual author".into(),
        ),
    );
    let command = || {
        virtual_message_command(
            &tenant,
            admission.clone(),
            message_id,
            operation_id,
            MessageOperationKind::Create,
            None,
            source.clone(),
            now,
        )
    };
    let first = sink.submit(command()?).await.map_err(|error| {
        anyhow::anyhow!("genuine virtual author create must be accepted: {error}")
    })?;
    anyhow::ensure!(
        first.authoritative_version() == AggregateVersion::FIRST,
        "genuine virtual author create must be accepted"
    );
    let counts = || async {
        sqlx::query_as::<_, (i64,i64,i64)>("SELECT (SELECT count(*) FROM public.collaboration_command_receipts),(SELECT count(*) FROM public.collaboration_outbox),(SELECT count(*) FROM public.collaboration_events)").fetch_one(pool).await
    };
    let baseline = counts().await?;
    let duplicate = sink.submit(command()?).await?;
    anyhow::ensure!(
        duplicate.operation_id() == first.operation_id()
            && duplicate.authoritative_version() == first.authoritative_version()
            && counts().await? == baseline,
        "successful virtual duplicate changed original receipt or emitted events"
    );
    {
        use nostr_compat::generated_kinds::{
            KIND_MODERATION_BAN, KIND_MODERATION_TIMEOUT, KIND_MODERATION_UNBAN,
            KIND_MODERATION_UNTIMEOUT,
        };
        let restriction_sink = collab::admin::moderation::NostrRestrictionCommandSink::new(
            sea_orm::DatabaseConnection::from(pool.clone()),
        )?;
        let ingress = collab::nostr::event_ingest::NostrEventIngress::new(
            restriction_sink,
            collab::nostr::ingress::NostrIngressDeployment::InProcess,
        );
        let ban = signed_restriction_command(151, KIND_MODERATION_BAN, 152, now, 21, None);
        anyhow::ensure!(
            send_signed_restriction_command(
                &ingress,
                &tenant,
                &owner,
                MembershipRole::Admin,
                &ban,
                now
            )
            .await?
            .status()
                == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
            "genuine agent ban not admitted"
        );
        let restricted = counts().await?;
        anyhow::ensure!(
            sink.submit(command()?).await.is_err() && counts().await? == restricted,
            "current canonical agent ban bypassed duplicate receipt admission"
        );
        let unban = signed_restriction_command(151, KIND_MODERATION_UNBAN, 152, now, 22, None);
        anyhow::ensure!(
            send_signed_restriction_command(
                &ingress,
                &tenant,
                &owner,
                MembershipRole::Admin,
                &unban,
                now
            )
            .await?
            .status()
                == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
            "genuine agent unban not admitted"
        );
        let timeout =
            signed_restriction_command(151, KIND_MODERATION_TIMEOUT, 152, now, 23, Some(now + 60));
        anyhow::ensure!(
            send_signed_restriction_command(
                &ingress,
                &tenant,
                &owner,
                MembershipRole::Admin,
                &timeout,
                now
            )
            .await?
            .status()
                == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
            "genuine agent timeout not admitted"
        );
        let restricted = counts().await?;
        anyhow::ensure!(
            sink.submit(command()?).await.is_err() && counts().await? == restricted,
            "current canonical agent timeout bypassed duplicate receipt admission"
        );
        let timeout_read =
            CanonicalMessageService::new(sea_orm::DatabaseConnection::from(pool.clone()))?
                .message(&admission, message_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("timeout blocked permitted agent read"))?;
        anyhow::ensure!(
            timeout_read.author.owner_principal_id() == Some(owner.principal_id()),
            "timeout read lost immutable attribution"
        );

        let untimeout =
            signed_restriction_command(151, KIND_MODERATION_UNTIMEOUT, 152, now, 24, None);
        anyhow::ensure!(
            send_signed_restriction_command(
                &ingress,
                &tenant,
                &owner,
                MembershipRole::Admin,
                &untimeout,
                now
            )
            .await?
            .status()
                == collab::nostr::event_ingest::NostrEventIngestStatus::Accepted,
            "genuine agent timeout lift not admitted"
        );
    }
    let sequence: i64 = sqlx::query_scalar(
        "SELECT max(outbox_sequence) FROM public.collaboration_outbox WHERE community_id=$1",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(pool)
    .await?;
    let acknowledgement_operation = OperationId::from_uuid(Uuid::new_v4());
    let sequence_u64 = u64::try_from(sequence)?;
    let acknowledge = || {
        collab::collaboration_command::DomainCommand::new(
            acknowledgement_operation,
            tenant.clone(),
            admission.principal.clone(),
            None,
            None,
            collab::collaboration_command::CommandAdapter::NostrInProcess,
            collab::messages::channel_mutation::MessageOperation {
                authorization: admission.clone(),
                kind: MessageOperationKind::Acknowledge,
                message_id,
                expected_version: None,
                signed_event: None,
                reaction: None,
                related_reaction_event_id: None,
                acknowledged_outbox_sequence: Some(sequence_u64),
            },
        )
    };
    sink.submit(acknowledge()).await?;
    let read_frontier: i64 = sqlx::query_scalar("SELECT last_outbox_sequence FROM public.collaboration_message_read_states WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(agent.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        read_frontier == sequence,
        "virtual acknowledgment lost actual outbox frontier"
    );
    counts().await?;
    let service = CanonicalMessageService::new(sea_orm::DatabaseConnection::from(pool.clone()))?;
    let owner_admission =
        virtual_message_admission(pool, &profiles, &tenant, &owner, channel, now).await?;
    let ordinary = service
        .message(&owner_admission, legacy_message_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("migration lost ordinary human message visibility"))?;
    anyhow::ensure!(
        ordinary.author == collaboration_domain::MessageAuthor::principal(owner.principal_id())
            && ordinary.body == "retained ordinary author",
        "ordinary author changed during typed migration"
    );
    let original = service
        .message(&owner_admission, message_id)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!("canonical service hid virtual message without legacy binding")
        })?;
    anyhow::ensure!(
        original.author_principal_id == agent.principal_id()
            && original.author.owner_principal_id() == Some(owner.principal_id()),
        "virtual message lost owning-human attribution"
    );
    let metadata = sign_restriction_event(152, CanonicalEvent::new(public_key(152), now,
        0, vec![], r#"{"display_name":"Actual agent display","picture":"https://virtual-agent.example.test/avatar.png"}"#.into()));
    let metadata = VerifiedEventRecord::new(
        tenant.community_id(),
        metadata,
        EventVerificationState::Live,
        now * 1000,
        TimestampPolicy::Bounded {
            now,
            max_past_seconds: 120,
            max_future_seconds: 120,
        },
    )?;
    profiles.project_profile_event(&tenant, &metadata).await?;
    let presentation = service
        .message(&owner_admission, message_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("metadata update hid virtual message"))?;
    anyhow::ensure!(
        presentation.author == original.author
            && presentation.author_display_name == "Actual agent display"
            && presentation.author_avatar_url == "https://virtual-agent.example.test/avatar.png",
        "canonical authored metadata changed identity or was not displayed"
    );
    let baseline = counts().await?;
    let attribution: serde_json::Value = sqlx::query_scalar("SELECT agent_author_attribution FROM public.collaboration_messages WHERE community_id=$1 AND message_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(message_id.as_uuid()).fetch_one(pool).await?;
    authenticate_registration(
        &profiles,
        &tenant,
        &signed_registration(152, Some(151), now + 1, 3),
    )
    .await?;
    let after = service
        .message(&owner_admission, message_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("renewal hid original message"))?;
    anyhow::ensure!(
        after.author == original.author,
        "profile renewal rewrote historical message proof"
    );
    anyhow::ensure!(
        sink.submit(command()?).await.is_err()
            && sink.submit(acknowledge()).await.is_err()
            && counts().await? == baseline,
        "stale verified principal duplicate bypassed renewed proof"
    );
    let expiry_timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let expires_at = expiry_timestamp + 3;
    let expiring_auth = signed_registration_with_tags(
        152,
        vec![owner_tag(
            152,
            151,
            &format!("kind=0&created_at<{expires_at}"),
        )],
        expiry_timestamp,
        4,
    );
    let expiring = authenticate_registration(&profiles, &tenant, &expiring_auth).await?;
    let expiring_admission = virtual_message_admission(
        pool,
        &profiles,
        &tenant,
        &expiring,
        channel,
        expiry_timestamp,
    )
    .await?;
    let expiring_message_id = AggregateId::from_uuid(Uuid::new_v4());
    let expiring_operation_id = OperationId::from_uuid(Uuid::new_v4());
    let expiring_source = sign_restriction_event(
        152,
        CanonicalEvent::new(
            public_key(152),
            expiry_timestamp,
            nostr_compat::generated_kinds::KIND_STREAM_MESSAGE as u16,
            vec![vec!["h".into(), channel.to_string()]],
            "author proof later expires".into(),
        ),
    );
    let expiring_command = || {
        virtual_message_command(
            &tenant,
            expiring_admission.clone(),
            expiring_message_id,
            expiring_operation_id,
            MessageOperationKind::Create,
            None,
            expiring_source.clone(),
            expiry_timestamp,
        )
    };
    sink.submit(expiring_command()?).await?;
    let expiring_original = service
        .message(&owner_admission, expiring_message_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("genuine expiring author message missing"))?;
    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
    let current_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    anyhow::ensure!(
        current_seconds >= expires_at,
        "expiry fixture did not reach signed condition boundary"
    );
    let baseline = counts().await?;
    anyhow::ensure!(
        sink.submit(expiring_command()?).await.is_err() && counts().await? == baseline,
        "expired verified principal duplicate must deny without mutation"
    );
    let historical_expired = service
        .message(&owner_admission, expiring_message_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("expired historical proof hid message"))?;
    anyhow::ensure!(
        historical_expired.author == expiring_original.author,
        "historical exact OA must be evaluated at original admitted source time"
    );
    let renewed = authenticate_registration(
        &profiles,
        &tenant,
        &signed_registration(152, Some(151), current_seconds, 5),
    )
    .await?;
    let mut forged_tag = owner_tag(152, 151, "kind=0");
    *forged_tag
        .get_mut(3)
        .ok_or_else(|| anyhow::anyhow!("OA signature fixture absent"))? = "00".repeat(64);
    let no_owner_authority = authenticate_registration(
        &profiles,
        &tenant,
        &signed_registration_with_tags(152, vec![forged_tag], current_seconds, 8),
    )
    .await?;
    anyhow::ensure!(
        matches!(
            no_owner_authority.kind(),
            collaboration_domain::AuthenticatedPrincipalKind::NostrIdentity { .. }
        ) && virtual_message_admission(
            pool,
            &profiles,
            &tenant,
            &no_owner_authority,
            channel,
            current_seconds
        )
        .await
        .is_err(),
        "invalid optional OA fallback granted virtual message membership"
    );
    let viewer = authenticate_registration(
        &profiles,
        &tenant,
        &signed_registration(153, None, current_seconds, 6),
    )
    .await?;
    sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,'member','active',1,now(),now(),'zed','independent-reader-fixture',now())")
        .bind(tenant.community_id().as_uuid()).bind(viewer.principal_id().as_uuid()).execute(pool).await?;
    sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,'member','active',1,now(),now(),'zed','independent-reader-fixture',now())")
        .bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(viewer.principal_id().as_uuid()).execute(pool).await?;
    let viewer_admission =
        virtual_message_admission(pool, &profiles, &tenant, &viewer, channel, current_seconds)
            .await?;
    {
        use sqlx::Connection as _;
        let request_name = format!("virtual-owner-race-{}", Uuid::new_v4().simple());
        sqlx::query("SELECT set_config('default_transaction_isolation','repeatable read',false), set_config('application_name',$1,false)")
            .bind(&request_name).execute(pool).await?;
        let isolation: String = sqlx::query_scalar("SHOW default_transaction_isolation")
            .fetch_one(pool)
            .await?;
        anyhow::ensure!(
            isolation == "repeatable read",
            "configured isolation negative fixture absent"
        );
        let race_admission =
            virtual_message_admission(pool, &profiles, &tenant, &renewed, channel, current_seconds)
                .await?;
        let race_command = virtual_message_command(
            &tenant,
            race_admission,
            message_id,
            operation_id,
            MessageOperationKind::Create,
            None,
            source.clone(),
            now,
        )?;
        let mut blocker = sqlx::PgConnection::connect_with(pool.connect_options().as_ref()).await?;
        let blocker_pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
            .fetch_one(&mut blocker)
            .await?;
        let mut blocker = blocker.begin().await?;
        sqlx::query("UPDATE public.collaboration_community_memberships SET status='revoked', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
            .bind(tenant.community_id().as_uuid()).bind(owner.principal_id().as_uuid()).execute(&mut *blocker).await?;
        let blocked_sink = sink.clone();
        let mut task = tokio::spawn(async move { blocked_sink.submit(race_command).await });
        let wait = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                sqlx::query("SELECT pg_stat_clear_snapshot()").execute(&mut *blocker).await?;
                let waiting: bool = sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE application_name=$1 AND datname=current_database() AND wait_event_type='Lock' AND $2=ANY(pg_blocking_pids(pid)) AND query LIKE '%collaboration_community_memberships%FOR SHARE%')")
                    .bind(&request_name).bind(blocker_pid).fetch_one(&mut *blocker).await?;
                if waiting { return Ok::<_, sqlx::Error>(()); }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }).await;
        if !matches!(wait, Ok(Ok(()))) {
            blocker.rollback().await?;
            task.abort();
            match task.await {
                Err(error) if error.is_cancelled() => {}
                Err(error) => return Err(error.into()),
                Ok(_) => {}
            }
            return Err(anyhow::anyhow!(
                "exact owned request did not reach current-owner FOR SHARE lock waiter: {wait:?}"
            ));
        }
        anyhow::ensure!(
            !task.is_finished(),
            "request completed before committed owner revocation"
        );
        blocker.commit().await?;
        let denied = tokio::time::timeout(std::time::Duration::from_secs(5), &mut task).await;
        let denied = match denied {
            Ok(result) => result?,
            Err(error) => {
                task.abort();
                match task.await {
                    Err(join) if join.is_cancelled() => {}
                    Err(join) => return Err(join.into()),
                    Ok(_) => {}
                }
                return Err(error.into());
            }
        };
        anyhow::ensure!(
            denied.is_err() && counts().await? == baseline,
            "post-lock owner revocation must deny duplicate under configured RepeatableRead without mutation"
        );
        let revoked_history = service
            .message(&viewer_admission, message_id)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("revoked-owner history hidden from independent reader")
            })?;
        anyhow::ensure!(
            revoked_history.author == original.author,
            "historical author changed after concurrent owner revocation"
        );
        sqlx::query("UPDATE public.collaboration_community_memberships SET status='active', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
            .bind(tenant.community_id().as_uuid()).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    }
    {
        let current_admission =
            virtual_message_admission(pool, &profiles, &tenant, &renewed, channel, current_seconds)
                .await?;
        let current_duplicate = virtual_message_command(
            &tenant,
            current_admission,
            message_id,
            operation_id,
            MessageOperationKind::Create,
            None,
            source.clone(),
            now,
        )?;
        virtual_moderation_duplicate_race(
            pool,
            &tenant,
            &owner,
            sink.clone(),
            current_duplicate,
            current_seconds,
        )
        .await?;
    }
    let baseline = counts().await?;
    {
        let fresh_agent =
            virtual_message_admission(pool, &profiles, &tenant, &renewed, channel, current_seconds)
                .await?;
        let duplicate = || {
            virtual_message_command(
                &tenant,
                fresh_agent.clone(),
                message_id,
                operation_id,
                MessageOperationKind::Create,
                None,
                source.clone(),
                now,
            )
        };
        let exact = sink.submit(duplicate()?).await?;
        anyhow::ensure!(
            exact.operation_id() == operation_id,
            "pre-archive genuine duplicate receipt changed"
        );
        let archived_source = sign_restriction_event(
            152,
            CanonicalEvent::new(
                public_key(152),
                current_seconds,
                nostr_compat::generated_kinds::KIND_STREAM_MESSAGE as u16,
                vec![vec!["h".into(), channel.to_string()]],
                "archived owner must deny".into(),
            ),
        );
        let new_command = virtual_message_command(
            &tenant,
            fresh_agent.clone(),
            AggregateId::from_uuid(Uuid::new_v4()),
            OperationId::from_uuid(Uuid::new_v4()),
            MessageOperationKind::Create,
            None,
            archived_source,
            current_seconds,
        )?;
        virtual_seed_owner_archive_policy(pool, &tenant, &owner, true, 1, current_seconds).await?;
        let before = counts().await?;
        anyhow::ensure!(
            sink.submit(duplicate()?).await.is_err()
                && sink.submit(new_command).await.is_err()
                && service.message(&fresh_agent, message_id).await.is_err()
                && counts().await? == before,
            "archived owner authorized virtual duplicate new-command or read"
        );
        let historical = service
            .message(&viewer_admission, message_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("archived owner hid independent historical read"))?;
        anyhow::ensure!(
            historical.author == original.author,
            "owner archive rewrote signed original attribution"
        );
        virtual_seed_owner_archive_policy(pool, &tenant, &owner, false, 2, current_seconds).await?;
    }
    sqlx::query("UPDATE public.collaboration_community_memberships SET status='archived', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    anyhow::ensure!(
        virtual_message_admission(pool, &profiles, &tenant, &renewed, channel, now)
            .await
            .is_err()
            && counts().await? == baseline,
        "archived owner retained virtual admission"
    );
    sqlx::query("UPDATE public.collaboration_community_memberships SET status='active', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    sqlx::query("UPDATE public.collaboration_channel_memberships SET status='revoked', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(agent.principal_id().as_uuid()).execute(pool).await?;
    let channel_denied =
        virtual_message_admission(pool, &profiles, &tenant, &renewed, channel, now).await?;
    anyhow::ensure!(
        channel_denied
            .authorize(AuthorizationAction::Write, now * 1000)
            .is_err(),
        "revoked explicit channel joined virtual author"
    );
    sqlx::query("UPDATE public.collaboration_channel_memberships SET status='active', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(agent.principal_id().as_uuid()).execute(pool).await?;
    sqlx::query("UPDATE public.collaboration_community_memberships SET status='revoked', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    let renewed_admission =
        virtual_message_admission(pool, &profiles, &tenant, &renewed, channel, now).await;
    anyhow::ensure!(
        renewed_admission.is_err()
            && sink.submit(command()?).await.is_err()
            && counts().await? == baseline,
        "owner revocation failed to deny virtual admission and duplicate"
    );
    let inactive_owner_history = service
        .message(&viewer_admission, message_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("inactive owner hid historical agent author"))?;
    anyhow::ensure!(
        inactive_owner_history.author == original.author,
        "fresh reader authority must not rewrite or suppress revoked-owner attribution"
    );

    sqlx::query("UPDATE public.collaboration_community_memberships SET status='active', membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    let owner_admission =
        virtual_message_admission(pool, &profiles, &tenant, &owner, channel, now).await?;
    let foreign = bind_rpc_tenant(
        Some(TrustedTenantRoute::from_listener(
            CommunityId::from_uuid(Uuid::from_u128(502)),
            "virtual-foreign-fixture",
        )?),
        &[],
    )
    .map_err(|error| anyhow::anyhow!("foreign fixture tenant admission failed: {error:?}"))?;
    let foreign_principal = authenticate_registration(
        &profiles,
        &foreign,
        &signed_registration(154, None, current_seconds, 7),
    )
    .await?;
    let foreign_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_identity_profiles WHERE community_id=$1 AND principal_id=$2")
        .bind(foreign.community_id().as_uuid()).bind(foreign_principal.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(foreign_count == 1, "actual foreign RLS source absent");
    {
        use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
        let transaction = connection.begin().await?;
        let role = format!("virtual_author_{}", Uuid::new_v4().simple());
        transaction
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                format!("CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS"),
            ))
            .await?;
        for statement in [
            format!("GRANT USAGE ON SCHEMA public TO {role}"),
            format!("GRANT SELECT, UPDATE ON ALL TABLES IN SCHEMA public TO {role}"),
            format!("SET LOCAL ROLE {role}"),
        ] {
            transaction
                .execute(Statement::from_string(DatabaseBackend::Postgres, statement))
                .await?;
        }
        let rls = collab::messages::channel_admission::admit_authenticated_channel_in_transaction(
            &transaction,
            &profiles,
            &tenant,
            &renewed,
            channel,
            now * 1000,
        )
        .await?;
        anyhow::ensure!(
            rls.principal.principal_id() == agent.principal_id(),
            "non-BYPASS RLS lost exact virtual subject"
        );
        let foreign = bind_rpc_tenant(
            Some(TrustedTenantRoute::from_listener(
                CommunityId::from_uuid(Uuid::from_u128(502)),
                "virtual-foreign-fixture",
            )?),
            &[],
        )
        .map_err(|error| anyhow::anyhow!("foreign fixture tenant admission failed: {error:?}"))?;
        let denied =
            collab::messages::channel_admission::admit_authenticated_channel_in_transaction(
                &transaction,
                &profiles,
                &foreign,
                &renewed,
                channel,
                now * 1000,
            )
            .await;
        anyhow::ensure!(
            denied.is_err(),
            "foreign tenant reinterpreted verified virtual principal"
        );
        let changed: i64 = transaction.query_one(Statement::from_string(DatabaseBackend::Postgres,
            format!("SELECT count(*)::bigint AS members FROM public.collaboration_identity_profiles WHERE community_id='{}'", foreign.community_id()))).await?
            .ok_or_else(|| anyhow::anyhow!("RLS fixture query missing"))?.try_get("", "members")?;
        anyhow::ensure!(changed == 0, "forced RLS exposed actual foreign identity");
        transaction.rollback().await?;
    }
    let unrelated_deletion = sign_restriction_event(
        153,
        CanonicalEvent::new(
            public_key(153),
            current_seconds,
            nostr_compat::generated_kinds::KIND_DELETION as u16,
            vec![vec!["e".into(), source.claimed_id.to_hex()]],
            String::new(),
        ),
    );
    anyhow::ensure!(
        sink.submit(virtual_message_command(
            &tenant,
            viewer_admission.clone(),
            message_id,
            OperationId::from_uuid(Uuid::new_v4()),
            MessageOperationKind::Delete,
            Some(AggregateVersion::FIRST),
            unrelated_deletion,
            current_seconds
        )?)
        .await
        .is_err()
            && counts().await? == baseline,
        "unrelated ordinary member deleted a verified agent's message"
    );
    let deletion = sign_restriction_event(
        151,
        CanonicalEvent::new(
            public_key(151),
            now + 2,
            nostr_compat::generated_kinds::KIND_DELETION as u16,
            vec![vec!["e".into(), source.claimed_id.to_hex()]],
            String::new(),
        ),
    );
    sink.submit(virtual_message_command(
        &tenant,
        owner_admission.clone(),
        message_id,
        OperationId::from_uuid(Uuid::new_v4()),
        MessageOperationKind::Delete,
        Some(AggregateVersion::FIRST),
        deletion,
        now,
    )?)
    .await?;
    let deleted = service
        .message(&owner_admission, message_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("deleted message absent"))?;
    anyhow::ensure!(
        deleted.deleted && deleted.author == original.author && deleted.body.is_empty(),
        "owning-human delete lost immutable original agent author"
    );
    let altered = sqlx::query("UPDATE public.collaboration_messages SET agent_author_owner_principal_id=author_principal_id WHERE community_id=$1 AND message_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(message_id.as_uuid()).execute(pool).await.expect_err("immutable message author update must reject");
    anyhow::ensure!(
        altered
            .as_database_error()
            .and_then(|error| error.code())
            .as_deref()
            == Some("23514"),
        "immutable author update rejected for wrong reason"
    );
    sqlx::raw_sql("ALTER TABLE public.collaboration_messages DISABLE TRIGGER collaboration_messages_virtual_agent_reference").execute(pool).await?;
    let mut corrupted = attribution.clone();
    *corrupted
        .pointer_mut("/proof/proof/exact_tag/3")
        .ok_or_else(|| anyhow::anyhow!("immutable genuine AUTH OA fixture path absent"))? =
        serde_json::Value::String("00".repeat(64));
    sqlx::query("UPDATE public.collaboration_messages SET agent_author_attribution=$3 WHERE community_id=$1 AND message_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(message_id.as_uuid()).bind(&corrupted).execute(pool).await?;
    let corrupted_read = service.message(&owner_admission, message_id).await;
    sqlx::query("UPDATE public.collaboration_messages SET agent_author_attribution=$3 WHERE community_id=$1 AND message_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(message_id.as_uuid()).bind(&attribution).execute(pool).await?;
    sqlx::raw_sql("ALTER TABLE public.collaboration_messages ENABLE TRIGGER collaboration_messages_virtual_agent_reference").execute(pool).await?;
    anyhow::ensure!(
        corrupted_read.is_err(),
        "corrupt immutable OA must not hydrate or fallback to ordinary author"
    );
    let retained: serde_json::Value = sqlx::query_scalar("SELECT agent_author_attribution FROM public.collaboration_messages WHERE community_id=$1 AND message_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(message_id.as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        retained == attribution,
        "mutation changed immutable exact OA attribution"
    );
    let fake_members: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2")
        .bind(tenant.community_id().as_uuid()).bind(agent.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        fake_members == 0,
        "virtual author fabricated persistent community membership"
    );
    let rollback = sqlx::raw_sql(include_str!(
        "../migrations/20261010000600_collaboration_virtual_agent_message_authors.down.sql"
    ))
    .execute(pool)
    .await
    .expect_err("retained virtual attribution must prevent rollback");
    anyhow::ensure!(
        rollback
            .as_database_error()
            .and_then(|error| error.code())
            .as_deref()
            == Some("23514"),
        "virtual rollback failed for wrong reason"
    );
    let final_counts = counts().await?;
    anyhow::ensure!(
        final_counts.0 == baseline.0 + 1
            && final_counts.1 == baseline.1 + 1
            && final_counts.2 == baseline.2 + 1,
        "owning-human delete or refused rollback lost data or extra side effects"
    );
    eprintln!(
        "VIRTUAL_AGENT_MESSAGE_IMMUTABLE_AUTHOR_VERIFIED: actual AUTH/OA, pool1, absent physical membership, explicit channel, current scopes/owner, duplicate fences, renewal, owning-human delete, retained proof, service visibility and fail-closed rollback; no mounted Nostr listener/live claim"
    );
    Ok(())
}

async fn virtual_moderation_duplicate_race(
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    owner: &collaboration_domain::AuthenticatedPrincipal,
    sink: Arc<
        collab::db::collaboration::outbox::TransactionalCommandOutbox<
            collab::messages::channel_mutation::CanonicalMessageMutation,
        >,
    >,
    duplicate: collab::collaboration_command::DomainCommand<
        collab::messages::channel_mutation::MessageOperation,
    >,
    now: u64,
) -> anyhow::Result<()> {
    use collab::{
        collaboration_command::DomainCommandSink as _,
        nostr::event_ingest::{NostrEventIngestStatus, NostrEventIngress},
    };
    use nostr_compat::generated_kinds::{KIND_MODERATION_BAN, KIND_MODERATION_UNBAN};
    use sqlx::Connection as _;
    let request_name = format!("virtual-ban-request-{}", Uuid::new_v4().simple());
    let ban_name = format!("virtual-ban-writer-{}", Uuid::new_v4().simple());
    sqlx::query("SELECT set_config('default_transaction_isolation','repeatable read',false),set_config('application_name',$1,false)")
        .bind(&request_name).execute(pool).await?;
    let ban_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect_with(
            pool.connect_options()
                .as_ref()
                .clone()
                .application_name(&ban_name),
        )
        .await?;
    let ingress = NostrEventIngress::new(
        collab::admin::moderation::NostrRestrictionCommandSink::new(
            sea_orm::DatabaseConnection::from(ban_pool.clone()),
        )?,
        collab::nostr::ingress::NostrIngressDeployment::InProcess,
    );
    let mut controller = sqlx::PgConnection::connect_with(pool.connect_options().as_ref()).await?;
    let controller_pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
        .fetch_one(&mut controller)
        .await?;
    let mut blocker = Some(controller.begin().await?);
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(format!(
            "collaboration-moderation:{}",
            tenant.community_id()
        ))
        .execute(
            &mut **blocker
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("moderation blocker missing"))?,
        )
        .await?;
    let owned_tenant = tenant.clone();
    let actor = owner.clone();
    let ban = signed_restriction_command(151, KIND_MODERATION_BAN, 152, now, 25, None);
    let mut ban_task = Some(tokio::spawn(async move {
        send_signed_restriction_command(
            &ingress,
            &owned_tenant,
            &actor,
            collaboration_domain::MembershipRole::Admin,
            &ban,
            now,
        )
        .await
    }));
    let mut request_task = None;
    let result: anyhow::Result<()> = async {
        // The earlier exclusive advisory waiter must acquire before the later duplicate request.
        virtual_wait_exact_moderation_lock(&mut **blocker.as_mut().ok_or_else(|| anyhow::anyhow!("moderation blocker missing"))?,
            &ban_name, controller_pid).await?;
        request_task = Some(tokio::spawn(async move { sink.submit(duplicate).await }));
        virtual_wait_exact_moderation_lock(&mut **blocker.as_mut().ok_or_else(|| anyhow::anyhow!("moderation blocker missing"))?,
            &request_name, controller_pid).await?;
        anyhow::ensure!(!ban_task.as_ref().is_none_or(|task| task.is_finished())
            && !request_task.as_ref().is_none_or(|task| task.is_finished()), "moderation race escaped held canonical lock");
        blocker.take().ok_or_else(|| anyhow::anyhow!("moderation blocker missing"))?.commit().await?;
        let ban = virtual_complete_owned_task(ban_task.take().ok_or_else(|| anyhow::anyhow!("owned ban task missing"))?).await??;
        anyhow::ensure!(ban.status() == NostrEventIngestStatus::Accepted, "actual queued signed ban did not commit");
        let denied = virtual_complete_owned_task(request_task.take().ok_or_else(|| anyhow::anyhow!("owned duplicate task missing"))?).await?;
        anyhow::ensure!(denied.is_err(), "post-lock committed ban must deny duplicate under configured RepeatableRead");
        let receipt_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_command_receipts WHERE community_id=$1 AND command_kind='collaborative_message.create.v1'")
            .bind(tenant.community_id().as_uuid()).fetch_one(pool).await?;
        anyhow::ensure!(receipt_count == 2, "denied duplicate changed original message receipts");
        let ingress = NostrEventIngress::new(collab::admin::moderation::NostrRestrictionCommandSink::new(
            sea_orm::DatabaseConnection::from(ban_pool.clone()))?, collab::nostr::ingress::NostrIngressDeployment::InProcess);
        let unban = signed_restriction_command(151, KIND_MODERATION_UNBAN, 152, now, 26, None);
        anyhow::ensure!(send_signed_restriction_command(&ingress, tenant, owner, collaboration_domain::MembershipRole::Admin,
            &unban, now).await?.status() == NostrEventIngestStatus::Accepted, "race fixture could not canonically lift owned ban");
        Ok(())
    }.await;
    let mut cleanup_error = None;
    if let Some(blocker) = blocker {
        if let Err(error) = blocker.rollback().await {
            cleanup_error = Some(anyhow::Error::from(error));
        }
    }
    if let Some(task) = ban_task {
        task.abort();
        match task.await {
            Err(error) if error.is_cancelled() => {}
            Err(error) => cleanup_error = Some(error.into()),
            Ok(Err(error)) => cleanup_error = Some(error),
            Ok(Ok(_)) => {}
        }
    }
    if let Some(task) = request_task {
        task.abort();
        match task.await {
            Err(error) if error.is_cancelled() => {}
            Err(error) => cleanup_error = Some(error.into()),
            Ok(Err(error)) => {
                cleanup_error = Some(anyhow::anyhow!(
                    "owned duplicate cleanup completed with {error}"
                ))
            }
            Ok(Ok(_)) => {}
        }
    }
    ban_pool.close().await;
    if let Some(error) = cleanup_error {
        return Err(anyhow::anyhow!(
            "moderation race={result:?}; owned cleanup={error:#}"
        ));
    }
    result
}

async fn virtual_wait_exact_moderation_lock(
    connection: &mut sqlx::PgConnection,
    request_name: &str,
    blocker_pid: i32,
) -> anyhow::Result<()> {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            sqlx::query("SELECT pg_stat_clear_snapshot()").execute(&mut *connection).await?;
            let waiting: bool = sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND application_name=$1 AND wait_event_type='Lock' AND $2=ANY(pg_blocking_pids(pid)) AND query LIKE '%pg_advisory_xact_lock%')")
                .bind(request_name).bind(blocker_pid).fetch_one(&mut *connection).await?;
            if waiting { return Ok::<_, anyhow::Error>(()); }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }).await.map_err(|error| anyhow::anyhow!("exact canonical moderation waiter not observed: {error}"))?
}

async fn virtual_complete_owned_task<T>(mut task: tokio::task::JoinHandle<T>) -> anyhow::Result<T> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), &mut task).await {
        Ok(result) => Ok(result?),
        Err(error) => {
            task.abort();
            match task.await {
                Err(join) if join.is_cancelled() => {}
                Err(join) => return Err(join.into()),
                Ok(_) => {}
            }
            Err(error.into())
        }
    }
}

async fn virtual_seed_owner_archive_policy(
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    owner: &collaboration_domain::AuthenticatedPrincipal,
    archived: bool,
    version: i64,
    now: u64,
) -> anyhow::Result<()> {
    use collab::audit::{
        events::{AuditEventContext, ModerationAuditOperation, SecurityAuditEvent},
        repository::{AuditRepository, ExpectedAuditHead},
    };
    use sea_orm::{ConnectionTrait as _, DatabaseBackend, Statement, TransactionTrait as _};
    let connection = sea_orm::DatabaseConnection::from(pool.clone());
    let transaction = connection
        .begin_with_config(Some(sea_orm::IsolationLevel::ReadCommitted), None)
        .await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT set_config('app.community_id',$1,true)",
            [tenant.community_id().to_string().into()],
        ))
        .await?;
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
            [format!("collaboration-moderation:{}", tenant.community_id()).into()],
        ))
        .await?;
    let operation = collaboration_domain::OperationId::from_uuid(Uuid::new_v4());
    let head = AuditRepository::load_head_in_transaction(&transaction, tenant).await?;
    let record = SecurityAuditEvent::Moderation {
        context: AuditEventContext::new(
            tenant,
            operation,
            Some(owner),
            collaboration_domain::AuditOutcome::Succeeded,
            None,
            now * 1000,
        )?,
        operation: ModerationAuditOperation::ArchiveIdentity,
        subject_principal_id: Some(owner.principal_id()),
        record_id: owner.principal_id().as_uuid(),
    }
    .into_record()?;
    AuditRepository::append_in_transaction(
        &transaction,
        tenant,
        head.map(ExpectedAuditHead::Entry)
            .unwrap_or(ExpectedAuditHead::Empty),
        record,
    )
    .await?;
    transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "INSERT INTO public.collaboration_moderation_actions (community_id,action_id,operation_id,actor_principal_id,action_kind,record_kind,record_id,target_principal_id,occurred_at,source_system,source_record_id,source_version,source_observed_at) VALUES ($1,$2,$3,$4,'archive_identity','identity_archive',$4,$4,to_timestamp($5::numeric/1000),'zed',$6,$7,now())",
        [tenant.community_id().as_uuid().into(), Uuid::new_v4().into(), operation.as_uuid().into(), owner.principal_id().as_uuid().into(), (now*1000).to_string().into(), operation.to_string().into(), version.to_string().into()])).await?;
    transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "UPDATE public.collaboration_identity_archive_versions SET is_current=false WHERE community_id=$1 AND target_principal_id=$2 AND is_current", [tenant.community_id().as_uuid().into(), owner.principal_id().as_uuid().into()])).await?;
    transaction.execute(Statement::from_sql_and_values(DatabaseBackend::Postgres,
        "INSERT INTO public.collaboration_identity_archive_versions (community_id,target_principal_id,identity_public_key,archive_version,is_current,archive_state,consent_path,actor_principal_id,operation_id,occurred_at,source_system,source_record_id,source_version,source_observed_at) VALUES ($1,$2,$3,CAST($4 AS numeric),true,$5,'self',$2,$6,to_timestamp($7::numeric/1000),'zed',$8,$4,now())",
        [tenant.community_id().as_uuid().into(), owner.principal_id().as_uuid().into(), public_key(151).as_bytes().to_vec().into(), version.to_string().into(), (if archived {"archived"} else {"visible"}).into(), operation.as_uuid().into(), (now*1000).to_string().into(), operation.to_string().into()])).await?;
    transaction.commit().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn identity_profile_actual_closed_nip42_restriction_socket_journey() -> anyhow::Result<()> {
    let base = std::env::var("COLLAB_TEST_DATABASE_URL")?;
    let redis = std::env::var("COLLAB_TEST_REDIS_URL")?;
    let redis_url = url::Url::parse(&redis)?;
    anyhow::ensure!(
        redis_url.scheme() == "redis"
            && matches!(
                redis_url.host_str(),
                Some("127.0.0.1" | "localhost" | "::1")
            )
            && redis_url.username().is_empty()
            && redis_url.password().is_none()
            && matches!(redis_url.path(), "" | "/" | "/0"),
        "owned authless loopback Redis0 required"
    );
    let mut database = url::Url::parse(&base)?;
    anyhow::ensure!(
        matches!(database.host_str(), Some("127.0.0.1" | "localhost" | "::1")),
        "owned loopback SQL required"
    );
    database.set_path(&format!("zed-test-{}", Uuid::new_v4()));
    let database = database.to_string();
    sqlx::Postgres::create_database(&database).await?;
    let community = CommunityId::from_uuid(Uuid::new_v4());
    let pool = match sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&database)
        .await
    {
        Ok(pool) => pool,
        Err(error) => {
            let cleanup = sqlx::Postgres::drop_database(&database).await;
            return Err(anyhow::anyhow!(
                "owned socket database connect failed: {error}; cleanup={cleanup:?}"
            ));
        }
    };
    let result = async {
        profile_fixture_apply_canonical_schema(&pool).await?;
        tokio::time::timeout(
            std::time::Duration::from_secs(90),
            closed_socket_actual_journey(&pool, &redis, community),
        )
        .await
        .map_err(|_| anyhow::anyhow!("actual closed socket pool1 journey timed out"))?
    }
    .await;
    // SQLx returns query connections asynchronously. Reserve the sole connection
    // before marking the pool closed so it cannot return after the idle drain.
    let connection = pool.acquire().await?;
    let closing = pool.close();
    let connection_close = connection.close().await;
    closing.await;
    anyhow::ensure!(
        pool.is_closed() && pool.size() == 0,
        "owned socket pool did not drain"
    );
    let redis_cleanup = profile_fixture_redis_cleanup(&redis_url, community).await;
    let sql_cleanup = sqlx::Postgres::drop_database(&database).await;
    if sql_cleanup.is_err() {
        use sqlx::Connection as _;
        let mut inspector = sqlx::PgConnection::connect(&base).await?;
        let database_name = url::Url::parse(&database)?
            .path()
            .trim_start_matches('/')
            .to_owned();
        let sessions: Vec<(i32, Option<String>, String, Option<String>, Option<String>)> = sqlx::query_as("SELECT pid,state,application_name,wait_event,left(query,180) FROM pg_stat_activity WHERE datname=$1 ORDER BY pid")
            .bind(database_name).fetch_all(&mut inspector).await?;
        eprintln!("OWNED_SOCKET_SQL_CLEANUP_ACTIVE_BACKENDS: {sessions:?}");
        inspector.close().await?;
    }
    match (result, connection_close, redis_cleanup, sql_cleanup) {
        (Ok(()), Ok(()), Ok(()), Ok(())) => Ok(()),
        (result, connection_close, redis_cleanup, sql_cleanup) => Err(anyhow::anyhow!(
            "closed socket result={result:?}; owned connection close={connection_close:?}; owned Redis cleanup={redis_cleanup:?}; owned SQL cleanup={sql_cleanup:?}"
        )),
    }
}

type ClosedFixtureSocket =
    async_tungstenite::WebSocketStream<async_tungstenite::tokio::ConnectStream>;

fn closed_socket_signed_auth(
    key: u8,
    relay: &str,
    challenge: &str,
    owner: Option<u8>,
) -> SignedEvent {
    let mut tags = vec![
        vec!["relay".into(), relay.into()],
        vec!["challenge".into(), challenge.into()],
    ];
    if let Some(owner) = owner {
        tags.push(owner_tag(key, owner, ""));
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("fixture clock")
        .as_secs();
    let event = CanonicalEvent::new(public_key(key), now, 22242, tags, String::new());
    let claimed_id = event.event_id().expect("fixture auth ID");
    let signature = Secp256k1::new()
        .sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair(key));
    SignedEvent {
        event,
        claimed_id,
        signature: EventSignature::from_hex(&signature.to_string()).expect("fixture signature"),
    }
}

async fn closed_socket_receive(
    socket: &mut ClosedFixtureSocket,
) -> anyhow::Result<serde_json::Value> {
    use futures::StreamExt as _;
    let frame = tokio::time::timeout(std::time::Duration::from_secs(8), socket.next())
        .await?
        .ok_or_else(|| anyhow::anyhow!("socket closed before required frame"))??;
    match frame {
        async_tungstenite::tungstenite::Message::Text(text) => Ok(serde_json::from_str(&text)?),
        other => anyhow::bail!("expected protocol text, received {other:?}"),
    }
}

async fn closed_socket_authenticate(
    relay: &str,
    key: u8,
    owner: Option<u8>,
) -> anyhow::Result<(ClosedFixtureSocket, serde_json::Value)> {
    let (mut socket, _) = async_tungstenite::tokio::connect_async(relay)
        .await
        .map_err(|error| anyhow::anyhow!("production closed socket Upgrade failed: {error}"))?;
    let challenge = closed_socket_receive(&mut socket).await?;
    anyhow::ensure!(
        challenge.get(0).and_then(serde_json::Value::as_str) == Some("AUTH"),
        "production connection did not issue actual AUTH challenge"
    );
    let challenge = challenge
        .get(1)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("challenge missing"))?;
    let signed = closed_socket_signed_auth(key, relay, challenge, owner);
    socket
        .send(async_tungstenite::tungstenite::Message::Text(
            serde_json::json!(["AUTH", profile_fixture_wire(&signed)])
                .to_string()
                .into(),
        ))
        .await?;
    let response = closed_socket_receive(&mut socket).await?;
    anyhow::ensure!(
        response.get(0).and_then(serde_json::Value::as_str) == Some("OK")
            && response.get(1).and_then(serde_json::Value::as_str)
                == Some(signed.claimed_id.to_hex().as_str()),
        "AUTH receipt lost exact signed identity"
    );
    Ok((socket, response))
}

async fn closed_socket_send_event(
    socket: &mut ClosedFixtureSocket,
    event: &SignedEvent,
) -> anyhow::Result<serde_json::Value> {
    socket
        .send(async_tungstenite::tungstenite::Message::Text(
            serde_json::json!(["EVENT", profile_fixture_wire(event)])
                .to_string()
                .into(),
        ))
        .await?;
    closed_socket_receive(socket).await
}

async fn closed_socket_actual_journey(
    pool: &sqlx::PgPool,
    redis: &str,
    community: CommunityId,
) -> anyhow::Result<()> {
    use axum::{
        extract::{State, ws::WebSocketUpgrade},
        http::HeaderMap,
        response::Response,
    };
    use futures::StreamExt as _;
    async fn root(
        State(state): State<Arc<collab::nostr::http::NostrProfileHttpState>>,
        upgrade: Option<WebSocketUpgrade>,
        headers: HeaderMap,
    ) -> Response {
        if let Some(response) =
            collab::nostr::http::closed_websocket_root_response(Some(&state), &headers, upgrade)
                .await
        {
            return response;
        }
        collab::nostr::http::metadata_http_root_response(
            Some(state.metadata().as_ref()),
            &headers,
            "fixture ordinary root".to_owned(),
        )
        .await
    }
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    listener.set_nonblocking(true)?;
    let endpoint = format!("ws://{}", listener.local_addr()?);
    let host = listener.local_addr()?.to_string();
    let tenant = TenantContext::establish(
        Some(TrustedTenantRoute::from_direct_host(community, &host)?),
        &[],
    )?;
    sqlx::query("INSERT INTO public.collaboration_communities (community_id,host,lifecycle_state,aggregate_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,'127.0.0.1','active',1,'zed','closed-socket',now(),now(),now())").bind(community.as_uuid()).execute(pool).await?;
    let repository = Arc::new(IdentityProfileRepository::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    for (key, role) in [(81_u8, "admin"), (82_u8, "member")] {
        let principal = authenticate_registration(
            &repository,
            &tenant,
            &signed_registration(key, None, now, key),
        )
        .await?;
        sqlx::query("INSERT INTO public.collaboration_community_memberships (community_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,'active',1,now(),now(),'zed','closed-fixture-member',now())").bind(community.as_uuid()).bind(principal.principal_id().as_uuid()).bind(role).execute(pool).await?;
    }
    let config: collab::Config = serde_json::from_value(
        serde_json::json!({"http_port":0,"database_url":"","database_max_connections":1,"zed_environment":"test","zed_cloud_internal_api_key":"fixture-only","collaboration_redis_url":redis,"collaboration_nostr_routes":serde_json::json!([{"community_id":community.as_uuid(),"public_url":endpoint}]).to_string(),"collaboration_nostr_api_calls_per_minute":1000,"collaboration_nostr_closed_websocket":true}),
    )?;
    let state = collab::nostr::http::configured_profile_http_state(
        &config,
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?
    .ok_or_else(|| anyhow::anyhow!("actual closed production state disabled"))?;
    let router = axum::Router::new()
        .route("/", axum::routing::get(root))
        .with_state(state.clone())
        .merge(collab::nostr::http::profile_http_router(state.clone()))
        .merge(collab::nostr::http::metadata_http_router(state.metadata()));
    let (shutdown, stopped) = tokio::sync::oneshot::channel::<()>();
    let mut server = tokio::spawn(
        axum::Server::from_tcp(listener)?
            .serve(router.into_make_service())
            .with_graceful_shutdown(async move {
                if stopped.await.is_err() {
                    tracing::warn!("owned socket shutdown sender dropped");
                }
            }),
    );
    let result = async {
        let client = reqwest::Client::new();
        let metadata: serde_json::Value = client.get(format!("http://{host}/")).header("Accept","application/nostr+json").send().await?.json().await?;
        anyhow::ensure!(metadata["supported_nips"].as_array().is_some_and(|nips| nips.contains(&serde_json::json!(42)) && !nips.contains(&serde_json::json!(1))), "socket metadata advertised unimplemented general subscriptions or hid actual NIP42");
        let (mut invalid, _) = async_tungstenite::tokio::connect_async(&endpoint).await?;
        let challenge = closed_socket_receive(&mut invalid).await?;
        let challenge = challenge.get(1).and_then(serde_json::Value::as_str).ok_or_else(||anyhow::anyhow!("actual invalid-vector challenge missing"))?;
        let mut tampered = closed_socket_signed_auth(85,&endpoint,challenge,Some(81));
        tampered.signature = EventSignature::from_hex(&"00".repeat(64))?;
        invalid.send(async_tungstenite::tungstenite::Message::Text(serde_json::json!(["AUTH",profile_fixture_wire(&tampered)]).to_string().into())).await?;
        anyhow::ensure!(closed_socket_receive(&mut invalid).await?.get(2).and_then(serde_json::Value::as_bool)==Some(false), "invalid outer signature gained socket authority");
        invalid.close(None).await?;
        anyhow::ensure!(repository.current_subject_by_public_key(&tenant,NostrPublicKey::from_bytes(*public_key(85).as_bytes())).await?.is_none(),"invalid outer AUTH allocated identity or owner observation");
        let (mut admin, receipt) = closed_socket_authenticate(&endpoint,81,None).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool) == Some(true), "genuine current admin socket AUTH denied");
        let (mut member, receipt) = closed_socket_authenticate(&endpoint,82,None).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool) == Some(true), "genuine current member socket AUTH denied");
        member.close(None).await?;
        let (mut agent, receipt) = closed_socket_authenticate(&endpoint,83,Some(81)).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool) == Some(true), "genuine owner-bound virtual agent socket AUTH denied");
        let agent_identity = repository.current_registration_by_public_key(&tenant,NostrPublicKey::from_bytes(*public_key(83).as_bytes())).await?.ok_or_else(|| anyhow::anyhow!("actual socket did not register agent"))?;
        let physically_inserted: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2").bind(community.as_uuid()).bind(agent_identity.principal_id().as_uuid()).fetch_one(pool).await?;
        anyhow::ensure!(physically_inserted==0, "socket AUTH fabricated physical agent membership");
        closed_socket_agent_channel_journey(pool, &repository, &tenant, &endpoint, &mut admin, &mut agent, now).await?;
        let denied_admin = signed_restriction_command(83,9040,82,now,41,None);
        let response = closed_socket_send_event(&mut agent,&denied_admin).await?;
        anyhow::ensure!(response.get(2).and_then(serde_json::Value::as_bool)==Some(false), "agent inherited owner admin role");
        agent.close(None).await?;
        admin.send(async_tungstenite::tungstenite::Message::Text(serde_json::json!(["REQ","unsupported",{"kinds":[0]}]).to_string().into())).await?;
        anyhow::ensure!(closed_socket_receive(&mut admin).await?.get(0).and_then(serde_json::Value::as_str)==Some("NOTICE"), "unsupported general subscription acquired fabricated functionality");
        let unsupported = profile_fixture_signed_event(81, now, vec![vec!["nonce".into(),"unsupported-closed-profile".into()]], "{}");
        anyhow::ensure!(closed_socket_send_event(&mut admin,&unsupported).await?.get(2).and_then(serde_json::Value::as_bool)==Some(false), "closed restriction socket accepted an unsupported profile EVENT");
        let timeout = signed_restriction_command(81,9042,82,now,45,Some(now+60));
        anyhow::ensure!(closed_socket_send_event(&mut admin,&timeout).await?.get(2).and_then(serde_json::Value::as_bool)==Some(true), "actual signed socket timeout did not reach canonical mutation");
        let (mut timed_out_member,receipt)=closed_socket_authenticate(&endpoint,82,None).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool)==Some(true), "message timeout was incorrectly treated as a structural AUTH ban");
        let ordinary_admin_command=signed_restriction_command(82,9040,81,now,46,None);
        anyhow::ensure!(closed_socket_send_event(&mut timed_out_member,&ordinary_admin_command).await?.get(2).and_then(serde_json::Value::as_bool)==Some(false), "ordinary member acquired current admin authority");
        timed_out_member.close(None).await?;
        let untimeout = signed_restriction_command(81,9043,82,now,47,None);
        anyhow::ensure!(closed_socket_send_event(&mut admin,&untimeout).await?.get(2).and_then(serde_json::Value::as_bool)==Some(true), "actual signed socket timeout lift did not reach canonical mutation");
        let ban = signed_restriction_command(81,9040,82,now,42,None);
        let receipt = closed_socket_send_event(&mut admin,&ban).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool)==Some(true), "actual signed socket ban did not reach canonical mutation");
        let duplicate = closed_socket_send_event(&mut admin,&ban).await?;
        anyhow::ensure!(duplicate.get(2).and_then(serde_json::Value::as_bool)==Some(true) && duplicate.get(3).and_then(serde_json::Value::as_str).is_some_and(|reason| reason.starts_with("duplicate:")), "actual socket retry lost canonical duplicate receipt");
        let (mut denied, receipt) = closed_socket_authenticate(&endpoint,82,None).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool)==Some(false) && receipt.get(3).and_then(serde_json::Value::as_str).is_some_and(|reason| reason.starts_with("blocked:")), "committed socket ban did not structurally reject actual new AUTH");
        let close = tokio::time::timeout(std::time::Duration::from_secs(5),denied.next()).await?.ok_or_else(|| anyhow::anyhow!("ban denial lost required Close frame"))??;
        anyhow::ensure!(matches!(close,async_tungstenite::tungstenite::Message::Close(_)), "ban socket closed before ordered denial or did not send Close");
        let unban = signed_restriction_command(81,9041,82,now,43,None);
        anyhow::ensure!(closed_socket_send_event(&mut admin,&unban).await?.get(2).and_then(serde_json::Value::as_bool)==Some(true), "actual signed lift did not reach canonical mutation");
        let (mut restored,receipt)=closed_socket_authenticate(&endpoint,82,None).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool)==Some(true), "actual AUTH did not recover after canonical lift");
        let mismatch = signed_restriction_command(82,9040,81,now,44,None);
        anyhow::ensure!(closed_socket_send_event(&mut admin,&mismatch).await?.get(2).and_then(serde_json::Value::as_bool)==Some(false), "socket admitted another signer's command");
        restored.close(None).await?;
        let (mut unknown,receipt)=closed_socket_authenticate(&endpoint,84,None).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool)==Some(false), "closed socket admitted key without current membership");
        unknown.close(None).await?;
        closed_socket_membership_delegate_race(pool, &repository, &tenant, &endpoint).await?;
        let admin_identity=repository.current_registration_by_public_key(&tenant,NostrPublicKey::from_bytes(*public_key(81).as_bytes())).await?.ok_or_else(|| anyhow::anyhow!("admin identity missing"))?;
        sqlx::query("UPDATE public.collaboration_community_memberships SET status='revoked',membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2").bind(community.as_uuid()).bind(admin_identity.principal_id().as_uuid()).execute(pool).await?;
        let (mut revoked,receipt)=closed_socket_authenticate(&endpoint,83,Some(81)).await?;
        anyhow::ensure!(receipt.get(2).and_then(serde_json::Value::as_bool)==Some(false), "revoked owner still admitted virtual agent socket");
        revoked.close(None).await?;
        admin.close(None).await?;
        let stored: i64=sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events WHERE community_id=$1 AND event_id=ANY($2)").bind(community.as_uuid()).bind(vec![ban.claimed_id.as_bytes().to_vec(),unban.claimed_id.as_bytes().to_vec(),timeout.claimed_id.as_bytes().to_vec(),untimeout.claimed_id.as_bytes().to_vec()]).fetch_one(pool).await?;
        anyhow::ensure!(stored==4,"actual socket lost original signed restrictions or retry wrote duplicates: stored={stored}");
        for original in [&ban, &unban, &timeout, &untimeout] {
            let stored: (Vec<u8>, Vec<u8>) = sqlx::query_as("SELECT canonical_event_bytes,signature FROM public.collaboration_events WHERE community_id=$1 AND event_id=$2").bind(community.as_uuid()).bind(original.claimed_id.as_bytes().to_vec()).fetch_one(pool).await?;
            anyhow::ensure!(stored.0 == original.event.canonical_bytes()? && stored.1.as_slice() == original.signature.as_bytes(), "closed socket lost exact signed restriction bytes for kind {}", original.event.kind);
        }
        eprintln!("CLOSED_NIP42_SIGNED_RESTRICTION_SOCKET_VERIFIED: real loopback Upgrade/challenge/Redis/AUTH/current SQL member/OA/signed ban/lift/retry/ordered banned Close; schema31/pool1; no general subscription/OpenRelay/allowlist/live disconnect claim");
        Ok::<(),anyhow::Error>(())
    }.await;
    let shutdown_result = shutdown
        .send(())
        .map_err(|_| anyhow::anyhow!("owned socket shutdown receiver missing"));
    match tokio::time::timeout(std::time::Duration::from_secs(10), &mut server).await {
        Ok(joined) => {
            joined??;
        }
        Err(_) => {
            server.abort();
            match server.await {
                Err(error) if error.is_cancelled() => {}
                Err(error) => return Err(error.into()),
                Ok(outcome) => {
                    outcome?;
                }
            }
            anyhow::bail!(
                "owned socket server required cancellation after grace timeout; journey={result:?}"
            );
        }
    }
    shutdown_result?;
    result
}

struct BlockedClosedSocketResolver {
    inner: collab::nostr::auth::ClosedNostrPrincipalResolver,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl collab::nostr::auth::NostrPrincipalResolver for BlockedClosedSocketResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &collab::nostr::auth::VerifiedNip42Identity,
    ) -> Result<
        collaboration_domain::AuthenticatedPrincipal,
        collab::nostr::auth::NostrPrincipalResolutionError,
    > {
        let principal = self.inner.resolve(tenant, identity).await?;
        self.entered.notify_one();
        self.release.notified().await;
        Ok(principal)
    }
    async fn validate_current_in_transaction(
        &self,
        transaction: &sea_orm::DatabaseTransaction,
        tenant: &TenantContext,
        identity: &collab::nostr::auth::VerifiedNip42Identity,
        principal: &collaboration_domain::AuthenticatedPrincipal,
        now_seconds: u64,
    ) -> Result<(), collab::nostr::auth::NostrPrincipalResolutionError> {
        self.inner
            .validate_current_in_transaction(transaction, tenant, identity, principal, now_seconds)
            .await
    }
}

async fn closed_socket_membership_delegate_race(
    pool: &sqlx::PgPool,
    repository: &Arc<IdentityProfileRepository>,
    tenant: &TenantContext,
    relay: &str,
) -> anyhow::Result<()> {
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let challenge = hex::encode([217_u8; 32]);
    let signed = closed_socket_signed_auth(82, relay, &challenge, None);
    let resolver = collab::nostr::auth::NostrStructuralBanResolver::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
        BlockedClosedSocketResolver {
            inner: collab::nostr::auth::ClosedNostrPrincipalResolver::new(
                sea_orm::DatabaseConnection::from(pool.clone()),
            )?,
            entered: entered.clone(),
            release: release.clone(),
        },
    )?;
    let mut authenticator = collab::nostr::auth::NostrConnectionAuthenticator::new(
        tenant.clone(),
        relay,
        collab::nostr::auth::NostrAuthChallenge::parse(challenge)?,
        now,
        RegistrationReplay::default(),
        resolver,
    )?;
    let member = repository
        .current_registration_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(82).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("current registered member missing"))?;
    let authenticate = async { authenticator.authenticate(&signed, now).await.cloned() };
    let revoke = async {
        entered.notified().await;
        let result=sqlx::query("UPDATE public.collaboration_community_memberships SET status='revoked',membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2").bind(tenant.community_id().as_uuid()).bind(member.principal_id().as_uuid()).execute(pool).await;
        release.notify_one();
        result.map(|_| ()).map_err(anyhow::Error::from)
    };
    let (authenticated, revoked) = tokio::time::timeout(std::time::Duration::from_secs(6), async {
        tokio::join!(authenticate, revoke)
    })
    .await
    .map_err(|_| anyhow::anyhow!("closed current-policy pool1 delegate race deadlocked"))?;
    revoked?;
    anyhow::ensure!(
        authenticated == Err(collab::nostr::auth::NostrAuthenticationError::IdentityDenied),
        "closed AUTH succeeded after current membership revoked during delegate"
    );
    sqlx::query("UPDATE public.collaboration_community_memberships SET status='active',membership_version=membership_version+1 WHERE community_id=$1 AND principal_id=$2").bind(tenant.community_id().as_uuid()).bind(member.principal_id().as_uuid()).execute(pool).await?;
    Ok(())
}

async fn closed_socket_agent_channel_journey(
    pool: &sqlx::PgPool,
    profiles: &IdentityProfileRepository,
    tenant: &TenantContext,
    endpoint: &str,
    owner_socket: &mut ClosedFixtureSocket,
    agent_socket: &mut ClosedFixtureSocket,
    now: u64,
) -> anyhow::Result<()> {
    use collaboration_domain::{
        AggregateId, AuthenticatedPrincipal, NostrAuthenticationMethod, PrincipalScopes,
    };
    let owner = profiles
        .current_registration_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(81).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("owner registration absent"))?;
    let member = profiles
        .current_registration_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(82).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("member registration absent"))?;
    let agent = profiles
        .current_registration_by_public_key(
            tenant,
            NostrPublicKey::from_bytes(*public_key(83).as_bytes()),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("agent registration absent"))?;
    let private = AggregateId::from_uuid(Uuid::new_v4());
    let open = AggregateId::from_uuid(Uuid::new_v4());
    for (channel, visibility) in [(private, "private"), (open, "open")] {
        sqlx::query("INSERT INTO public.collaboration_channels (community_id,channel_id,name,channel_type,visibility,lifecycle_state,creator_principal_id,channel_version,source_system,source_record_id,source_observed_at,created_at,updated_at) VALUES ($1,$2,'actual signed agent entry','stream',$3,'active',$4,1,'zed','existing-channel-fixture',now(),now(),now())").bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(visibility).bind(owner.principal_id().as_uuid()).execute(pool).await?;
    }
    for principal in [owner.principal_id(), member.principal_id()] {
        sqlx::query("INSERT INTO public.collaboration_channel_memberships (community_id,channel_id,principal_id,role,status,membership_version,joined_at,updated_at,source_system,source_record_id,source_observed_at) VALUES ($1,$2,$3,'member','active',1,now(),now(),'zed','existing-actor-fixture',now())").bind(tenant.community_id().as_uuid()).bind(private.as_uuid()).bind(principal.as_uuid()).execute(pool).await?;
    }
    let (mut member_socket, receipt) = closed_socket_authenticate(endpoint, 82, None).await?;
    anyhow::ensure!(
        receipt.get(2).and_then(serde_json::Value::as_bool) == Some(true),
        "ordinary producer actor AUTH denied"
    );
    let put = |key, target, channel: AggregateId, role: Option<&str>, nonce: &str| {
        let mut tags = vec![
            vec!["h".into(), channel.to_string()],
            vec!["p".into(), public_key(target).to_hex()],
            vec!["nonce".into(), nonce.into()],
        ];
        if let Some(role) = role {
            tags.push(vec!["role".into(), role.into()]);
        }
        sign_restriction_event(
            key,
            CanonicalEvent::new(public_key(key), now, 9000, tags, String::new()),
        )
    };
    let create = sign_restriction_event(
        83,
        CanonicalEvent::new(
            public_key(83),
            now,
            9,
            vec![vec!["h".into(), private.to_string()]],
            "actual signed virtual author".into(),
        ),
    );
    anyhow::ensure!(
        closed_socket_send_event(agent_socket, &create)
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "agent wrote before actual explicit producer membership"
    );
    let add = put(82, 83, private, None, "actual ordinary member add");
    anyhow::ensure!(
        closed_socket_send_event(&mut member_socket, &add)
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "actual ordinary9000 producer did not create explicit agent membership"
    );
    let row:(String,Option<Uuid>,String)=sqlx::query_as("SELECT role,agent_profile_id,source_record_id FROM public.collaboration_channel_memberships WHERE community_id=$1 AND channel_id=$2 AND principal_id=$3").bind(tenant.community_id().as_uuid()).bind(private.as_uuid()).bind(agent.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        row.0 == "member"
            && row.1 == Some(agent.profile_id().as_uuid())
            && row.2 == add.claimed_id.to_hex(),
        "actual agent producer lost role/profile/signed provenance"
    );
    let retry = closed_socket_send_event(&mut member_socket, &add).await?;
    anyhow::ensure!(
        retry.get(2).and_then(serde_json::Value::as_bool) == Some(true)
            && retry.get(3).and_then(serde_json::Value::as_str) == Some("duplicate:"),
        "membership duplicate lost original canonical receipt"
    );
    anyhow::ensure!(
        closed_socket_send_event(
            &mut member_socket,
            &put(82, 83, private, Some("admin"), "no elevation")
        )
        .await?
        .get(2)
        .and_then(serde_json::Value::as_bool)
            == Some(false),
        "ordinary channel member acquired elevated role"
    );
    anyhow::ensure!(
        closed_socket_send_event(
            &mut member_socket,
            &put(82, 85, private, None, "unknown key")
        )
        .await?
        .get(2)
        .and_then(serde_json::Value::as_bool)
            == Some(false),
        "unknown target key fabricated registration/membership"
    );
    let policy = |value: &str, offset: u64| {
        sign_restriction_event(
            83,
            CanonicalEvent::new(
                public_key(83),
                now + offset,
                10100,
                Vec::new(),
                serde_json::json!({"channel_add_policy":value}).to_string(),
            ),
        )
    };
    let nobody = policy("nobody", 1);
    anyhow::ensure!(
        closed_socket_send_event(agent_socket, &nobody)
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "genuine agent10100 policy producer denied"
    );
    anyhow::ensure!(
        closed_socket_send_event(
            owner_socket,
            &put(81, 83, private, None, "nobody third-party")
        )
        .await?
        .get(2)
        .and_then(serde_json::Value::as_bool)
            == Some(false),
        "nobody agent policy ignored by actual producer"
    );
    anyhow::ensure!(
        closed_socket_send_event(agent_socket, &put(83, 83, open, None, "self open"))
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "actual open self-add rejected or fabricated actor membership required"
    );
    let owner_only = policy("owner_only", 2);
    anyhow::ensure!(
        closed_socket_send_event(agent_socket, &owner_only)
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "owner_only policy update denied"
    );
    anyhow::ensure!(
        closed_socket_send_event(
            &mut member_socket,
            &put(82, 83, private, None, "wrong owner")
        )
        .await?
        .get(2)
        .and_then(serde_json::Value::as_bool)
            == Some(false),
        "owner_only accepted unrelated member"
    );
    anyhow::ensure!(
        closed_socket_send_event(owner_socket, &put(81, 83, private, None, "exact owner"))
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "exact verified owner could not add agent as ordinary member"
    );
    anyhow::ensure!(
        closed_socket_send_event(agent_socket, &nobody)
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "same signed policy retry lost duplicate semantics"
    );
    anyhow::ensure!(
        closed_socket_send_event(agent_socket, &create)
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "actual signed agent9 did not reach canonical mutation"
    );
    anyhow::ensure!(
        closed_socket_send_event(agent_socket, &create)
            .await?
            .get(3)
            .and_then(serde_json::Value::as_str)
            == Some("duplicate:"),
        "actual agent message duplicate lost exact receipt"
    );
    let message_id:Uuid=sqlx::query_scalar("SELECT message_id FROM public.collaboration_messages WHERE community_id=$1 AND source_event_id=$2").bind(tenant.community_id().as_uuid()).bind(create.claimed_id.as_bytes().to_vec()).fetch_one(pool).await?;
    let viewer = AuthenticatedPrincipal::nostr_identity(
        owner.principal_id(),
        tenant.community_id(),
        NostrPublicKey::from_bytes(*public_key(81).as_bytes()),
        NostrAuthenticationMethod::Nip42,
        PrincipalScopes::for_nip42_authenticated_key()?,
    );
    let admission =
        virtual_message_admission(pool, profiles, tenant, &viewer, private, now).await?;
    let service = collab::messages::channel_service::CanonicalMessageService::new(
        sea_orm::DatabaseConnection::from(pool.clone()),
    )?;
    let hydrated = service
        .message(&admission, AggregateId::from_uuid(message_id))
        .await?
        .ok_or_else(|| anyhow::anyhow!("actual production-written agent history missing"))?;
    anyhow::ensure!(
        hydrated.author_principal_id == agent.principal_id()
            && hydrated.author.owner_principal_id() == Some(owner.principal_id())
            && hydrated.source_event_id.as_bytes() == create.claimed_id.as_bytes(),
        "production agent source lost immutable author/owner history"
    );
    let deletion = |key, nonce: &str| {
        sign_restriction_event(
            key,
            CanonicalEvent::new(
                public_key(key),
                now,
                5,
                vec![
                    vec!["e".into(), create.claimed_id.to_hex()],
                    vec!["nonce".into(), nonce.into()],
                ],
                String::new(),
            ),
        )
    };
    anyhow::ensure!(
        closed_socket_send_event(&mut member_socket, &deletion(82, "ordinary denial"))
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "ordinary nonauthor deleted virtual agent message"
    );
    let owner_delete = deletion(81, "owning human");
    anyhow::ensure!(
        closed_socket_send_event(owner_socket, &owner_delete)
            .await?
            .get(2)
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        "verified owning human5 did not reach canonical delete"
    );
    anyhow::ensure!(
        closed_socket_send_event(owner_socket, &owner_delete)
            .await?
            .get(3)
            .and_then(serde_json::Value::as_str)
            == Some("duplicate:"),
        "owning human delete retry changed receipt fingerprint"
    );
    let tombstone = service
        .message(&admission, AggregateId::from_uuid(message_id))
        .await?
        .ok_or_else(|| anyhow::anyhow!("tombstone history vanished"))?;
    anyhow::ensure!(
        tombstone.deleted
            && tombstone.author == hydrated.author
            && tombstone.source_event_id == hydrated.source_event_id,
        "delete rewrote historical agent proof or hid row"
    );
    let physical:i64=sqlx::query_scalar("SELECT count(*) FROM public.collaboration_community_memberships WHERE community_id=$1 AND principal_id=$2").bind(tenant.community_id().as_uuid()).bind(agent.principal_id().as_uuid()).fetch_one(pool).await?;
    anyhow::ensure!(
        physical == 0,
        "producer fabricated physical virtual-agent community membership"
    );
    closed_socket_channel_actor_revocation_race(
        pool,
        tenant,
        member_socket,
        &put(82, 81, private, None, "fresh actor revocation"),
        member.principal_id(),
        private,
    )
    .await?;
    closed_socket_channel_identity_lock_order(pool, tenant, endpoint, private, now).await?;
    eprintln!(
        "CLOSED_NIP42_TYPED_AGENT_CHANNEL_MESSAGE_JOURNEY_VERIFIED: actual signed10100/9000/9/5; ordinary channel member producer, policy/role/unknown-key denial, virtual no physical community row, immutable history and duplicate receipts; no general REQ/fanout/native proof"
    );
    Ok(())
}

async fn closed_socket_channel_actor_revocation_race(
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    socket: ClosedFixtureSocket,
    event: &SignedEvent,
    actor: collaboration_domain::PrincipalId,
    channel: collaboration_domain::AggregateId,
) -> anyhow::Result<()> {
    use sqlx::Connection as _;
    let request_name = format!("actual-channel-put-race-{}", Uuid::new_v4().simple());
    let prior_isolation: String =
        sqlx::query_scalar("SELECT current_setting('default_transaction_isolation')")
            .fetch_one(pool)
            .await?;
    let prior_name: String = sqlx::query_scalar("SELECT current_setting('application_name')")
        .fetch_one(pool)
        .await?;
    sqlx::query("SELECT set_config('application_name',$1,false),set_config('default_transaction_isolation','repeatable read',false)").bind(&request_name).execute(pool).await?;
    let receipt_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_command_receipts WHERE community_id=$1",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(pool)
    .await?;
    let mut controller = sqlx::PgConnection::connect_with(pool.connect_options().as_ref()).await?;
    let pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
        .fetch_one(&mut controller)
        .await?;
    let mut blocker = Some(controller.begin().await?);
    sqlx::query("SELECT principal_id FROM public.collaboration_channel_memberships WHERE community_id=$1 AND channel_id=$2 AND principal_id=$3 FOR UPDATE").bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(actor.as_uuid()).fetch_one(&mut **blocker.as_mut().ok_or_else(||anyhow::anyhow!("blocker missing"))?).await?;
    let signed = event.clone();
    let mut task = Some(tokio::spawn(async move {
        let mut socket = socket;
        let result = closed_socket_send_event(&mut socket, &signed).await;
        let close = socket.close(None).await;
        close?;
        result
    }));
    let result=async {
        let mut observer=sqlx::PgConnection::connect_with(pool.connect_options().as_ref()).await?;
        tokio::time::timeout(std::time::Duration::from_secs(5),async {
            loop {
                sqlx::query("SELECT pg_stat_clear_snapshot()").execute(&mut observer).await?;
                let waiting:bool=sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND application_name=$1 AND wait_event_type='Lock' AND $2=ANY(pg_blocking_pids(pid)) AND query LIKE '%collaboration_channel_memberships%' AND query LIKE '%FOR UPDATE%')").bind(&request_name).bind(pid).fetch_one(&mut observer).await?;
                if waiting { return Ok::<(),anyhow::Error>(()); }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }).await.map_err(|error|anyhow::anyhow!("exact canonical channel actor row waiter missing: {error}"))??;
        anyhow::ensure!(!task.as_ref().ok_or_else(||anyhow::anyhow!("owned request missing"))?.is_finished(),"producer completed while actual current-policy waiter held");
        sqlx::query("UPDATE public.collaboration_channel_memberships SET status='revoked',membership_version=membership_version+1,updated_at=clock_timestamp() WHERE community_id=$1 AND channel_id=$2 AND principal_id=$3").bind(tenant.community_id().as_uuid()).bind(channel.as_uuid()).bind(actor.as_uuid()).execute(&mut **blocker.as_mut().ok_or_else(||anyhow::anyhow!("blocker missing"))?).await?;
        blocker.take().ok_or_else(||anyhow::anyhow!("blocker missing"))?.commit().await?;
        let response=virtual_complete_owned_task(task.take().ok_or_else(||anyhow::anyhow!("owned request missing"))?).await??;
        anyhow::ensure!(response.get(2).and_then(serde_json::Value::as_bool)==Some(false) && response.get(3).and_then(serde_json::Value::as_str).is_some_and(|reason|reason.starts_with("restricted:")),"actual post-lock revoked private actor was authorized");
        let stored:i64=sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events WHERE community_id=$1 AND event_id=$2").bind(tenant.community_id().as_uuid()).bind(event.claimed_id.as_bytes().to_vec()).fetch_one(pool).await?;
        let after:i64=sqlx::query_scalar("SELECT count(*) FROM public.collaboration_command_receipts WHERE community_id=$1").bind(tenant.community_id().as_uuid()).fetch_one(pool).await?;
        anyhow::ensure!(stored==0 && after==receipt_count,"post-lock rejected producer retained signed EVENT or receipt side effects");
        observer.close().await?;
        Ok::<(),anyhow::Error>(())
    }.await;
    if let Some(blocker) = blocker.take() {
        blocker.rollback().await?;
    }
    if let Some(task) = task.take() {
        task.abort();
        match task.await {
            Err(error) if error.is_cancelled() => {}
            Err(error) => return Err(error.into()),
            Ok(outcome) => {
                outcome?;
            }
        }
    }
    sqlx::query("SELECT set_config('application_name',$1,false),set_config('default_transaction_isolation',$2,false)").bind(prior_name).bind(prior_isolation).execute(pool).await?;
    drop(blocker);
    controller.close().await?;
    result
}

async fn closed_socket_channel_identity_lock_order(
    pool: &sqlx::PgPool,
    tenant: &TenantContext,
    endpoint: &str,
    channel: collaboration_domain::AggregateId,
    now: u64,
) -> anyhow::Result<()> {
    use sqlx::Connection as _;
    let target_key = (86_u8..=254)
        .find(|key| {
            *public_key(*key).as_bytes() < *public_key(81).as_bytes()
                && *public_key(*key).as_bytes() < *public_key(83).as_bytes()
        })
        .ok_or_else(|| anyhow::anyhow!("genuine lower target key unavailable"))?;
    let (mut target_socket, receipt) =
        closed_socket_authenticate(endpoint, target_key, Some(81)).await?;
    let target_close = target_socket.close(None).await;
    target_close?;
    anyhow::ensure!(
        receipt.get(2).and_then(serde_json::Value::as_bool) == Some(true),
        "real target registration did not use genuine owner-bound AUTH"
    );
    let (actor_socket, receipt) = closed_socket_authenticate(endpoint, 83, Some(81)).await?;
    anyhow::ensure!(
        receipt.get(2).and_then(serde_json::Value::as_bool) == Some(true),
        "genuine actor AUTH denied"
    );
    let event = sign_restriction_event(
        83,
        CanonicalEvent::new(
            public_key(83),
            now,
            9000,
            vec![
                vec!["h".into(), channel.to_string()],
                vec!["p".into(), public_key(target_key).to_hex()],
                vec!["nonce".into(), "actual sorted identity locks".into()],
            ],
            String::new(),
        ),
    );
    let prior_name: String = sqlx::query_scalar("SELECT current_setting('application_name')")
        .fetch_one(pool)
        .await?;
    let request_name = format!("actual-channel-key-order-{}", Uuid::new_v4().simple());
    sqlx::query("SELECT set_config('application_name',$1,false)")
        .bind(&request_name)
        .execute(pool)
        .await?;
    let backend: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
        .fetch_one(pool)
        .await?;
    let identity_lock = |key| {
        format!(
            "collaboration-profile:{}:{}",
            tenant.community_id(),
            public_key(key).to_hex()
        )
    };
    let mut controller = sqlx::PgConnection::connect_with(pool.connect_options().as_ref()).await?;
    let blocker_pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
        .fetch_one(&mut controller)
        .await?;
    let mut blocker = Some(controller.begin().await?);
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1,0))")
        .bind(identity_lock(target_key))
        .execute(
            &mut **blocker
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("owned key blocker missing"))?,
        )
        .await?;
    let mut task = Some(tokio::spawn(async move {
        let mut socket = actor_socket;
        let response = closed_socket_send_event(&mut socket, &event).await;
        socket.close(None).await?;
        response
    }));
    let result=async {
        let mut observer=sqlx::PgConnection::connect_with(pool.connect_options().as_ref()).await?;
        tokio::time::timeout(std::time::Duration::from_secs(5),async {
            loop {
                sqlx::query("SELECT pg_stat_clear_snapshot()").execute(&mut observer).await?;
                let waiting:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=$1 AND application_name=$2 AND wait_event_type='Lock' AND $3=ANY(pg_blocking_pids(pid)) AND query LIKE '%pg_advisory_xact_lock%')").bind(backend).bind(&request_name).bind(blocker_pid).fetch_one(&mut observer).await?;
                if waiting { return Ok::<(),anyhow::Error>(()); }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }).await.map_err(|error|anyhow::anyhow!("actual target-key advisory waiter missing: {error}"))??;
        let higher_held:bool=sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=$1 AND locktype='advisory' AND granted AND objsubid=1 AND ((classid::bigint<<32)|objid::bigint) IN (hashtextextended($2,0),hashtextextended($3,0)))").bind(backend).bind(identity_lock(81)).bind(identity_lock(83)).fetch_one(&mut observer).await?;
        anyhow::ensure!(!higher_held,"higher actor identity locks held before smaller target identity lock");
        anyhow::ensure!(!task.as_ref().ok_or_else(||anyhow::anyhow!("owned key request missing"))?.is_finished(),"canonical producer passed its actual target key blocker");
        blocker.take().ok_or_else(||anyhow::anyhow!("owned key blocker missing"))?.commit().await?;
        let response=virtual_complete_owned_task(task.take().ok_or_else(||anyhow::anyhow!("owned key request missing"))?).await??;
        anyhow::ensure!(response.get(2).and_then(serde_json::Value::as_bool)==Some(true),"sorted-identity producer failed after owned blocker release");
        observer.close().await?;
        Ok::<(),anyhow::Error>(())
    }.await;
    if let Some(blocker) = blocker.take() {
        blocker.rollback().await?;
    }
    if let Some(task) = task.take() {
        task.abort();
        match task.await {
            Err(error) if error.is_cancelled() => {}
            Err(error) => return Err(error.into()),
            Ok(outcome) => {
                outcome?;
            }
        }
    }
    sqlx::query("SELECT set_config('application_name',$1,false)")
        .bind(prior_name)
        .execute(pool)
        .await?;
    drop(blocker);
    controller.close().await?;
    result
}
