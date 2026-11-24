use std::collections::BTreeMap;

use collab::{
    search::indexer::{
        CollaborationSearchIndexer, SearchDocumentType, SearchExclusionReason, SearchIndexerError,
        SearchIndexingOutcome, SearchProjectionOperation,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use uuid::Uuid;

fn tenant() -> TenantContext {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "search-indexer")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn outbox_row(
    outbox_sequence: i64,
    source_version: &str,
    topic: &str,
    payload: Vec<u8>,
) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("outbox_sequence".into(), outbox_sequence.into()),
        ("topic".into(), topic.to_owned().into()),
        ("source_system".into(), "zed".to_owned().into()),
        ("source_record_id".into(), "project:alpha".to_owned().into()),
        ("source_version".into(), source_version.to_owned().into()),
        (
            "source_observed_at_millis".into(),
            1_900_000_000_000_i64.into(),
        ),
        (
            "source_integrity_algorithm".into(),
            Option::<String>::None.into(),
        ),
        (
            "source_integrity_value".into(),
            Option::<String>::None.into(),
        ),
        ("payload".into(), payload.into()),
    ])
}

fn success() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

#[tokio::test]
async fn search_indexer_applies_ordered_edits_from_authoritative_outbox_rows() {
    let first = SearchProjectionOperation::upsert_community(
        SearchDocumentType::Project,
        "Alpha",
        "first body",
    )
    .expect("first projection")
    .encode()
    .expect("first payload");
    let edited = SearchProjectionOperation::upsert_community(
        SearchDocumentType::Project,
        "Alpha edited",
        "second body",
    )
    .expect("edited projection")
    .encode()
    .expect("edited payload");
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([
            vec![outbox_row(
                1,
                "1",
                "collaboration.search.document.v1",
                first,
            )],
            vec![outbox_row(
                2,
                "2",
                "collaboration.search.document.v1",
                edited,
            )],
        ])
        .append_exec_results([
            success(),
            success(),
            success(),
            success(),
            success(),
            success(),
        ])
        .into_connection();
    let indexer = CollaborationSearchIndexer::new(database).expect("indexer");
    let tenant = tenant();

    assert_eq!(
        indexer
            .index_outbox_sequence(&tenant, 1)
            .await
            .expect("first projection"),
        SearchIndexingOutcome::Indexed
    );
    assert_eq!(
        indexer
            .index_outbox_sequence(&tenant, 2)
            .await
            .expect("edited projection"),
        SearchIndexingOutcome::Indexed
    );

    let log = format!("{:#?}", indexer.into_connection().into_transaction_log());
    assert!(log.contains("Alpha edited"));
    assert!(log.contains("second body"));
    assert!(log.contains("collaboration_search_documents.projection_version"));
    assert!(log.contains("EXCLUDED.projection_version"));
    assert!(log.contains("projection_name"));
    assert!(log.contains("collaboration_search"));
}

#[tokio::test]
async fn search_indexer_turns_delete_and_retention_into_ordered_empty_tombstones() {
    let deleted = SearchProjectionOperation::exclude(
        SearchDocumentType::Project,
        SearchExclusionReason::Deleted,
    )
    .encode()
    .expect("delete payload");
    let expired = SearchProjectionOperation::exclude(
        SearchDocumentType::Project,
        SearchExclusionReason::RetentionExpired,
    )
    .encode()
    .expect("retention payload");
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([
            vec![outbox_row(
                3,
                "3",
                "collaboration.search.document.v1",
                deleted,
            )],
            vec![outbox_row(
                4,
                "4",
                "collaboration.search.document.v1",
                expired,
            )],
        ])
        .append_exec_results([
            success(),
            success(),
            success(),
            success(),
            success(),
            success(),
        ])
        .into_connection();
    let indexer = CollaborationSearchIndexer::new(database).expect("indexer");
    let tenant = tenant();

    assert_eq!(
        indexer
            .index_outbox_sequence(&tenant, 3)
            .await
            .expect("delete projection"),
        SearchIndexingOutcome::Excluded(SearchExclusionReason::Deleted)
    );
    assert_eq!(
        indexer
            .index_outbox_sequence(&tenant, 4)
            .await
            .expect("retention projection"),
        SearchIndexingOutcome::Excluded(SearchExclusionReason::RetentionExpired)
    );

    let log = format!("{:#?}", indexer.into_connection().into_transaction_log());
    assert!(log.contains("excluded"));
    assert!(!log.contains("first body"));
    assert!(log.contains("cursor < EXCLUDED.cursor"));
}

#[tokio::test]
async fn search_indexer_never_projects_direct_message_content() {
    let private_text = "do not index this direct message";
    let excluded = SearchProjectionOperation::exclude(
        SearchDocumentType::Project,
        SearchExclusionReason::DirectMessage,
    );
    let payload = excluded.encode().expect("direct-message exclusion payload");
    assert!(!String::from_utf8_lossy(&payload).contains(private_text));
    let content_bearing_exclusion = serde_json::to_vec(&serde_json::json!({
        "contract_version": 1,
        "document_type": "project",
        "mutation": {
            "action": "exclude",
            "reason": "direct_message",
            "body": private_text,
        },
    }))
    .expect("malformed private projection payload");
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([
            vec![outbox_row(
                5,
                "5",
                "collaboration.search.document.v1",
                payload,
            )],
            vec![outbox_row(
                6,
                "6",
                "collaboration.search.document.v1",
                content_bearing_exclusion,
            )],
        ])
        .append_exec_results([success(), success(), success(), success()])
        .into_connection();
    let indexer = CollaborationSearchIndexer::new(database).expect("indexer");

    assert_eq!(
        indexer
            .index_outbox_sequence(&tenant(), 5)
            .await
            .expect("direct-message exclusion"),
        SearchIndexingOutcome::Excluded(SearchExclusionReason::DirectMessage)
    );
    assert!(matches!(
        indexer.index_outbox_sequence(&tenant(), 6).await,
        Err(SearchIndexerError::InvalidInput)
    ));

    let log = format!("{:#?}", indexer.into_connection().into_transaction_log());
    assert!(log.contains("excluded"));
    assert!(!log.contains(private_text));
}

#[tokio::test]
async fn search_indexer_suppresses_idempotent_replay_before_checkpoint_advance() {
    let payload = SearchProjectionOperation::upsert_community(
        SearchDocumentType::Project,
        "Replay",
        "same content",
    )
    .expect("projection")
    .encode()
    .expect("payload");
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([
            vec![outbox_row(
                6,
                "6",
                "collaboration.search.document.v1",
                payload.clone(),
            )],
            vec![outbox_row(
                6,
                "6",
                "collaboration.search.document.v1",
                payload,
            )],
        ])
        .append_exec_results([
            success(),
            success(),
            success(),
            success(),
            MockExecResult {
                last_insert_id: 0,
                rows_affected: 0,
            },
        ])
        .into_connection();
    let indexer = CollaborationSearchIndexer::new(database).expect("indexer");
    let tenant = tenant();

    assert_eq!(
        indexer
            .index_outbox_sequence(&tenant, 6)
            .await
            .expect("first delivery"),
        SearchIndexingOutcome::Indexed
    );
    assert_eq!(
        indexer
            .index_outbox_sequence(&tenant, 6)
            .await
            .expect("replay"),
        SearchIndexingOutcome::IgnoredReplay
    );

    let log = format!("{:#?}", indexer.into_connection().into_transaction_log());
    assert_eq!(
        log.matches("INSERT INTO public.collaboration_projection_checkpoints")
            .count(),
        1
    );
}
