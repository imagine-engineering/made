use collab::{
    git::review_repository::{
        ReviewProjection, ReviewProjectionRepository, ReviewProjectionWriteOutcome,
        ReviewRepositoryError,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, BranchCollaborationIdentity, BranchGeneration, BranchRefName,
    CiCheckRunCompletionInput, CiCheckRunInput, CiCheckStatus, CiCheckSuite, CiCheckSuiteIdentity,
    CiLabel, CiOutputText, CiWorkflowLink, CommunityId, GitCommitId, IntegrityAlgorithm,
    IntegrityReference, PatchRevisionInput, PatchRevisionNumber, PrincipalId, Provenance, Review,
    ReviewDecision, ReviewDecisionInput, ReviewIdentity, SourceRecordId, SourceSystem,
    TenantContext, TrustedTenantRoute,
};
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
const REVIEW_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260823000100_collaboration_git_review.up.sql"
));
const REVIEW_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260823000100_collaboration_git_review.down.sql"
));

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
    GitCommitId::parse(format!("{value:040x}")).expect("valid commit")
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "review-repository-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn initial_review(community_id: CommunityId, repository_id: AggregateId) -> Review {
    Review::open(
        ReviewIdentity::new(
            aggregate(30),
            BranchCollaborationIdentity::new(
                community_id,
                repository_id,
                BranchRefName::parse("refs/heads/feature/review-store").expect("branch ref"),
                BranchGeneration::FIRST,
            )
            .expect("branch identity"),
        )
        .expect("review identity"),
        1,
        PatchRevisionInput {
            revision_id: aggregate(31),
            base_commit: commit(100),
            head_commit: commit(101),
            author_principal_id: principal(20),
            created_at_millis: 1_900_000_000_000,
        },
    )
    .expect("review")
}

fn ci_suite(review: &Review, suite_id: u128, run_id: u128) -> CiCheckSuite {
    let revision = review.current_revision().expect("current revision");
    let mut suite = CiCheckSuite::create(
        CiCheckSuiteIdentity::for_revision(aggregate(suite_id), revision).expect("suite identity"),
        CiWorkflowLink::new(
            aggregate(suite_id + 100),
            aggregate(suite_id + 200),
            CiLabel::from_untrusted("CI").expect("workflow label"),
            None,
        )
        .expect("workflow link"),
        1_900_000_001_000,
    );
    suite
        .add_run(
            AggregateVersion::FIRST,
            CiCheckRunInput {
                check_run_id: aggregate(run_id),
                label: CiLabel::from_untrusted("test").expect("run label"),
                queued_at_millis: 1_900_000_002_000,
            },
        )
        .expect("add run");
    suite
        .complete_run(
            AggregateVersion::new(2).expect("suite version two"),
            aggregate(run_id),
            AggregateVersion::FIRST,
            &revision.head_commit,
            CiCheckRunCompletionInput {
                status: CiCheckStatus::Success,
                output: CiOutputText::from_untrusted("all tests passed"),
                artifacts: Vec::new(),
                completed_at_millis: 1_900_000_003_000,
            },
        )
        .expect("complete run");
    suite
}

fn provenance(version: &str, observed_at_millis: u64) -> Provenance {
    Provenance::new(
        SourceSystem::Nostr,
        SourceRecordId::new("event:review-projection").expect("source record"),
        observed_at_millis,
    )
    .with_source_version(version)
    .with_integrity(IntegrityReference {
        algorithm: IntegrityAlgorithm::NostrEventId,
        value: "ab".repeat(32),
    })
}

#[test]
fn review_projection_schema_is_reversible_provenance_aware_and_tenant_fenced() {
    for table in [
        "collaboration_git_review_projections",
        "collaboration_git_ci_projections",
    ] {
        assert!(REVIEW_UP.contains(&format!("CREATE TABLE public.{table}")));
        assert!(REVIEW_DOWN.contains(&format!("DROP TABLE public.{table}")));
    }
    assert!(REVIEW_UP.contains("ALTER TABLE public.%I FORCE ROW LEVEL SECURITY"));
    for column in [
        "source_system",
        "source_record_id",
        "source_version",
        "source_observed_at",
        "integrity_algorithm",
        "integrity_value",
        "projection_generation",
        "projection_hash",
    ] {
        assert!(REVIEW_UP.contains(column));
    }
    assert!(!REVIEW_UP.contains("patch_bytes"));
    assert!(!REVIEW_UP.contains("working_tree"));
    assert!(!REVIEW_UP.contains("index_state"));
}

#[tokio::test]
async fn review_repository_replaces_rebuilds_and_isolates_tenants() {
    let Some(database_url) = std::env::var("COLLAB_REVIEW_REPOSITORY_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_REVIEW_REPOSITORY_TEST_DATABASE_URL is unset; live repository test skipped"
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
    sqlx::raw_sql(REVIEW_UP)
        .execute(&pool)
        .await
        .expect("apply review projection migration");

    let community_a = community(10);
    let community_b = community(11);
    let repository_id = aggregate(12);
    for (community_id, host) in [
        (community_a, "review-a.example"),
        (community_b, "review-b.example"),
    ] {
        sqlx::query(
            "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'active', 1, 'zed', $1::text, now(), now(), now())",
        )
        .bind(community_id.as_uuid())
        .bind(host)
        .execute(&pool)
        .await
        .expect("insert community");
    }
    sqlx::query(
        "INSERT INTO public.collaboration_hosted_repositories (community_id, repository_id, repository_owner_public_key, repository_discriminator, authority_kind, authority_version, lifecycle_state, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, $3, 'review-store', 'sim_hosted_nip34', 1, 'active', 'zed', 'repository:review-store', now(), now(), now())",
    )
    .bind(community_a.as_uuid())
    .bind(repository_id.as_uuid())
    .bind(vec![7_u8; 32])
    .execute(&pool)
    .await
    .expect("insert hosted repository");
    sqlx::raw_sql(
        "CREATE ROLE collaboration_review_repository LOGIN PASSWORD 'review-test' NOBYPASSRLS; \
         GRANT USAGE ON SCHEMA public TO collaboration_review_repository; \
         GRANT SELECT ON public.collaboration_hosted_repositories TO collaboration_review_repository; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON public.collaboration_git_review_projections, public.collaboration_git_ci_projections TO collaboration_review_repository;",
    )
    .execute(&pool)
    .await
    .expect("create review repository role");
    let mut role_url = Url::parse(&database_url).expect("database URL");
    role_url
        .set_username("collaboration_review_repository")
        .expect("set role username");
    role_url
        .set_password(Some("review-test"))
        .expect("set role password");
    let request_pool = PgPool::connect(role_url.as_str())
        .await
        .expect("connect review repository role");
    let repository = ReviewProjectionRepository::new(request_pool.clone());
    let tenant_a = tenant(community_a);
    let tenant_b = tenant(community_b);

    let review_v1 = initial_review(community_a, repository_id);
    let projection_v1 = ReviewProjection::new(
        review_v1.clone(),
        vec![ci_suite(&review_v1, 40, 50)],
        provenance("1", 1_900_000_004_000),
    )
    .expect("initial projection");
    assert_eq!(
        repository
            .replace(&tenant_a, &projection_v1)
            .await
            .expect("insert projection"),
        ReviewProjectionWriteOutcome::Inserted
    );
    assert_eq!(
        repository
            .load(&tenant_a, aggregate(30))
            .await
            .expect("load projection"),
        Some(projection_v1.clone())
    );
    assert!(
        repository
            .load(&tenant_b, aggregate(30))
            .await
            .expect("load foreign projection")
            .is_none()
    );
    assert!(matches!(
        repository.replace(&tenant_b, &projection_v1).await,
        Err(ReviewRepositoryError::TenantMismatch)
    ));

    sqlx::query(
        "DELETE FROM public.collaboration_git_ci_projections WHERE community_id = $1 AND review_id = $2",
    )
    .bind(community_a.as_uuid())
    .bind(aggregate(30).as_uuid())
    .execute(&pool)
    .await
    .expect("inject missing derived CI row");
    assert_eq!(
        repository
            .replace(&tenant_a, &projection_v1)
            .await
            .expect("rebuild projection"),
        ReviewProjectionWriteOutcome::Rebuilt
    );
    assert_eq!(
        repository
            .load(&tenant_a, aggregate(30))
            .await
            .expect("load rebuilt projection"),
        Some(projection_v1.clone())
    );

    let mut review_v2 = review_v1.clone();
    review_v2
        .record_decision(
            AggregateVersion::FIRST,
            ReviewDecisionInput {
                approval_id: aggregate(60),
                revision: PatchRevisionNumber::FIRST,
                head_commit: commit(101),
                approver_principal_id: principal(21),
                decision: ReviewDecision::Approve,
                created_at_millis: 1_900_000_005_000,
            },
        )
        .expect("approve first revision");
    review_v2
        .submit_revision(
            AggregateVersion::new(2).expect("review version two"),
            PatchRevisionNumber::FIRST,
            PatchRevisionInput {
                revision_id: aggregate(32),
                base_commit: commit(100),
                head_commit: commit(102),
                author_principal_id: principal(20),
                created_at_millis: 1_900_000_006_000,
            },
        )
        .expect("replace revision");
    let projection_v2 = ReviewProjection::new(
        review_v2.clone(),
        vec![ci_suite(&review_v2, 41, 51)],
        provenance("2", 1_900_000_007_000),
    )
    .expect("replacement projection");
    assert_eq!(
        repository
            .replace(&tenant_a, &projection_v2)
            .await
            .expect("replace projection"),
        ReviewProjectionWriteOutcome::Replaced
    );
    let loaded = repository
        .load(&tenant_a, aggregate(30))
        .await
        .expect("load replacement")
        .expect("replacement exists");
    assert_eq!(loaded, projection_v2);
    assert_eq!(
        loaded
            .review()
            .current_revision()
            .map(|revision| revision.number),
        PatchRevisionNumber::new(2)
    );
    assert_eq!(loaded.review().fields().revisions.len(), 2);
    assert_eq!(loaded.review().fields().approvals.len(), 1);
    assert_eq!(loaded.ci_suites().len(), 1);
    assert!(matches!(
        repository.replace(&tenant_a, &projection_v1).await,
        Err(ReviewRepositoryError::StaleReviewVersion)
    ));

    let mut transaction = request_pool.begin().await.expect("tenant B transaction");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_b.as_uuid().to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant B");
    let foreign_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_git_review_projections WHERE review_id = $1",
    )
    .bind(aggregate(30).as_uuid())
    .fetch_one(&mut *transaction)
    .await
    .expect("count foreign review rows");
    assert_eq!(foreign_count, 0);
    transaction
        .rollback()
        .await
        .expect("roll back tenant B read");

    drop(repository);
    request_pool.close().await;
    sqlx::raw_sql(
        "DROP OWNED BY collaboration_review_repository; DROP ROLE collaboration_review_repository;",
    )
    .execute(&pool)
    .await
    .expect("drop review repository role");
    sqlx::raw_sql(REVIEW_DOWN)
        .execute(&pool)
        .await
        .expect("roll back review migration");
    sqlx::raw_sql(GIT_DOWN)
        .execute(&pool)
        .await
        .expect("roll back Git migration");
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll back channel migration");
}
