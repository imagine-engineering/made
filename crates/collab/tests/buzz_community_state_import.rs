use collab::{
    migration::buzz::community_state::{
        BuzzChannelMembershipRecord, BuzzChannelRecord, BuzzCommunityMembershipRecord,
        BuzzCommunityRecord, BuzzCommunityStateBatch, BuzzCommunityStateImportError,
        BuzzCommunityStateImporter, BuzzCommunityStateRecord, BuzzInviteRecord,
        BuzzJoinPolicyAcceptanceRecord,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, PrincipalId, TenantContext, TrustedTenantRoute};
use sea_orm::{DatabaseBackend, MockDatabase};
use sqlx::PgPool;
use uuid::Uuid;

const CHANNEL_MIGRATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "buzz-community-import")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn records(community_id: CommunityId) -> Vec<BuzzCommunityStateRecord> {
    let owner_key = "11".repeat(32);
    let member_key = "22".repeat(32);
    vec![
        BuzzCommunityStateRecord::Community(BuzzCommunityRecord {
            community_id,
            source_sequence: 1,
            host: "community.example".to_owned(),
            icon: Some("https://community.example/icon.png".to_owned()),
            lifecycle_state: "active".to_owned(),
            join_policy_version: Some("ab".repeat(32)),
            aggregate_version: 4,
            created_at_millis: 1_700_000_000_000,
            updated_at_millis: 1_700_000_010_000,
            observed_at_millis: 1_700_000_020_000,
        }),
        BuzzCommunityStateRecord::CommunityMembership(BuzzCommunityMembershipRecord {
            community_id,
            source_sequence: 2,
            principal_id: principal(11),
            public_key: owner_key.clone(),
            role: "owner".to_owned(),
            status: "active".to_owned(),
            membership_version: 7,
            added_by_principal_id: None,
            added_by_public_key: None,
            joined_at_millis: 1_700_000_000_000,
            updated_at_millis: 1_700_000_010_000,
            observed_at_millis: 1_700_000_020_000,
        }),
        BuzzCommunityStateRecord::CommunityMembership(BuzzCommunityMembershipRecord {
            community_id,
            source_sequence: 3,
            principal_id: principal(12),
            public_key: member_key.clone(),
            role: "member".to_owned(),
            status: "active".to_owned(),
            membership_version: 12,
            added_by_principal_id: Some(principal(11)),
            added_by_public_key: Some(owner_key.clone()),
            joined_at_millis: 1_700_000_001_000,
            updated_at_millis: 1_700_000_011_000,
            observed_at_millis: 1_700_000_020_000,
        }),
        BuzzCommunityStateRecord::JoinPolicyAcceptance(BuzzJoinPolicyAcceptanceRecord {
            community_id,
            source_sequence: 4,
            principal_id: principal(12),
            public_key: member_key.clone(),
            policy_version: "ab".repeat(32),
            accepted_at_millis: 1_700_000_001_000,
            observed_at_millis: 1_700_000_020_000,
        }),
        BuzzCommunityStateRecord::Channel(BuzzChannelRecord {
            community_id,
            source_sequence: 5,
            channel_id: Uuid::from_u128(101),
            name: "builders".to_owned(),
            channel_type: "stream".to_owned(),
            visibility: "private".to_owned(),
            lifecycle_state: "active".to_owned(),
            description: Some("Build together".to_owned()),
            creator_principal_id: principal(11),
            creator_public_key: owner_key.clone(),
            nip29_group_id: Some("builders".to_owned()),
            topic_required: false,
            max_members: Some(100),
            ttl_seconds: None,
            expires_at_millis: None,
            channel_version: 9,
            created_at_millis: 1_700_000_002_000,
            updated_at_millis: 1_700_000_012_000,
            observed_at_millis: 1_700_000_020_000,
        }),
        BuzzCommunityStateRecord::Invite(BuzzInviteRecord {
            community_id,
            source_sequence: 6,
            invite_id: Uuid::from_u128(201),
            channel_id: None,
            token_hash: [3; 32],
            role: "member".to_owned(),
            status: "active".to_owned(),
            max_uses: Some(5),
            use_count: 2,
            expires_at_millis: 1_800_000_000_000,
            created_by_principal_id: principal(11),
            created_by_source_identity: owner_key.clone(),
            invite_version: 3,
            created_at_millis: 1_700_000_003_000,
            updated_at_millis: 1_700_000_013_000,
            observed_at_millis: 1_700_000_020_000,
        }),
        BuzzCommunityStateRecord::ChannelMembership(BuzzChannelMembershipRecord {
            community_id,
            source_sequence: 7,
            channel_id: Uuid::from_u128(101),
            principal_id: principal(12),
            public_key: member_key,
            role: "member".to_owned(),
            status: "active".to_owned(),
            membership_version: 15,
            invited_by_principal_id: Some(principal(11)),
            invited_by_public_key: Some(owner_key),
            joined_at_millis: 1_700_000_004_000,
            updated_at_millis: 1_700_000_014_000,
            hidden_at_millis: None,
            observed_at_millis: 1_700_000_020_000,
        }),
    ]
}

#[test]
fn buzz_community_state_import_rejects_unknown_schema_versions() {
    let result = BuzzCommunityStateBatch::new(31, records(community(1)));
    assert!(matches!(
        result,
        Err(BuzzCommunityStateImportError::UnsupportedSchemaVersion(31))
    ));
}

#[tokio::test]
async fn buzz_community_state_import_rejects_cross_tenant_before_database_io() {
    let batch = BuzzCommunityStateBatch::new(30, records(community(1))).expect("valid batch");
    let importer = BuzzCommunityStateImporter::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
    )
    .expect("PostgreSQL importer");
    let result = importer.import_batch(&tenant(community(2)), &batch).await;
    assert!(matches!(
        result,
        Err(BuzzCommunityStateImportError::TenantBoundaryViolation)
    ));
    assert!(importer.into_connection().into_transaction_log().is_empty());
}

#[tokio::test]
async fn buzz_community_state_import_preserves_versions_and_resumes_idempotently() {
    let Some(database_url) = std::env::var("COLLAB_BUZZ_COMMUNITY_IMPORT_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_BUZZ_COMMUNITY_IMPORT_TEST_DATABASE_URL is unset; live community import test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(CHANNEL_MIGRATION)
        .execute(&pool)
        .await
        .expect("apply collaboration channel migration");
    let community_id = community(1);
    let tenant = tenant(community_id);
    let records = records(community_id);
    let first_batch =
        BuzzCommunityStateBatch::new(30, records[..5].to_vec()).expect("valid first batch");
    let complete_batch =
        BuzzCommunityStateBatch::new(30, records.clone()).expect("valid complete batch");
    let importer = BuzzCommunityStateImporter::new(
        sea_orm::Database::connect(&database_url)
            .await
            .expect("connect importer"),
    )
    .expect("community-state importer");

    let interrupted = importer
        .import_batch(&tenant, &first_batch)
        .await
        .expect("import first window");
    assert_eq!(interrupted.inserted, 5);
    assert_eq!(interrupted.duplicates, 0);
    assert_eq!(interrupted.source_hash, interrupted.target_hash);

    let replayed = importer
        .import_batch(&tenant, &first_batch)
        .await
        .expect("replay first window");
    assert_eq!(replayed.inserted, 0);
    assert_eq!(replayed.duplicates, 5);
    assert_eq!(replayed.source_hash, interrupted.source_hash);
    assert_eq!(replayed.target_hash, interrupted.target_hash);

    let completed = importer
        .import_batch(&tenant, &complete_batch)
        .await
        .expect("complete overlapping import");
    assert_eq!(completed.scanned, 7);
    assert_eq!(completed.inserted, 2);
    assert_eq!(completed.duplicates, 5);
    assert_eq!(completed.final_source_sequence, 7);
    assert_eq!(completed.source_hash, completed.target_hash);

    let membership_versions: Vec<(Uuid, String)> = sqlx::query_as(
        "SELECT principal_id, membership_version::text FROM public.collaboration_community_memberships WHERE community_id = $1 ORDER BY principal_id",
    )
    .bind(community_id.as_uuid())
    .fetch_all(&pool)
    .await
    .expect("read membership versions");
    assert_eq!(
        membership_versions,
        vec![
            (principal(11).as_uuid(), "7".to_owned()),
            (principal(12).as_uuid(), "12".to_owned())
        ]
    );
    let channel_membership_version: String = sqlx::query_scalar(
        "SELECT membership_version::text FROM public.collaboration_channel_memberships WHERE community_id = $1 AND channel_id = $2 AND principal_id = $3",
    )
    .bind(community_id.as_uuid())
    .bind(Uuid::from_u128(101))
    .bind(principal(12).as_uuid())
    .fetch_one(&pool)
    .await
    .expect("read channel membership version");
    assert_eq!(channel_membership_version, "15");

    let mut conflicting_records = records;
    let BuzzCommunityStateRecord::Community(community) = &mut conflicting_records[0] else {
        panic!("first fixture record must be the community");
    };
    community.host = "changed.example".to_owned();
    let conflicting = BuzzCommunityStateBatch::new(30, conflicting_records)
        .expect("structurally valid conflicting batch");
    assert!(matches!(
        importer.import_batch(&tenant, &conflicting).await,
        Err(BuzzCommunityStateImportError::IntegrityConflict)
    ));
    println!("Buzz community import: versions, memberships and idempotent receipts verified");
}
