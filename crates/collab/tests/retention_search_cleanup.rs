use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::retention::{
    search_cleanup::{
        RetentionSearchBackend, RetentionSearchBackendError, RetentionSearchCheckpoint,
        RetentionSearchCheckpointCommitOutcome, RetentionSearchCheckpointFields,
        RetentionSearchCleanup, RetentionSearchCleanupError, RetentionSearchDelivery,
        RetentionSearchProjection, RetentionSearchProjectionError,
        RetentionSearchProjectionOutcome,
    },
    worker::{MAX_RETENTION_BATCH_SIZE, RetentionSourcePosition},
};
use collab::search::indexer::{
    CollaborationSearchIndexer, SearchDocumentType, SearchExclusionReason,
    SearchProjectionOperation,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FailureMode {
    None,
    BeforeCheckpointOnce,
    AfterCheckpointOnce,
}

struct BackendState {
    checkpoint: Option<RetentionSearchCheckpoint>,
    deliveries: Vec<RetentionSearchDelivery>,
    failure_mode: FailureMode,
    commit_calls: usize,
}

#[derive(Clone)]
struct RecordingBackend {
    state: Arc<Mutex<BackendState>>,
}

impl RecordingBackend {
    fn new(deliveries: Vec<RetentionSearchDelivery>) -> Self {
        Self {
            state: Arc::new(Mutex::new(BackendState {
                checkpoint: None,
                deliveries,
                failure_mode: FailureMode::None,
                commit_calls: 0,
            })),
        }
    }

    fn set_failure_mode(&self, failure_mode: FailureMode) {
        self.state.lock().expect("backend state").failure_mode = failure_mode;
    }

    fn checkpoint(&self) -> Option<RetentionSearchCheckpoint> {
        self.state.lock().expect("backend state").checkpoint
    }
}

#[async_trait]
impl RetentionSearchBackend for RecordingBackend {
    async fn load_checkpoint(
        &self,
        _tenant: &TenantContext,
    ) -> Result<Option<RetentionSearchCheckpoint>, RetentionSearchBackendError> {
        Ok(self.state.lock().expect("backend state").checkpoint)
    }

    async fn load_batch(
        &self,
        _tenant: &TenantContext,
        checkpoint: RetentionSearchCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionSearchDelivery>, RetentionSearchBackendError> {
        Ok(self
            .state
            .lock()
            .expect("backend state")
            .deliveries
            .iter()
            .filter(|delivery| {
                delivery.source_position().sequence() > checkpoint.source_position().sequence()
            })
            .take(limit)
            .copied()
            .collect())
    }

    async fn advance_checkpoint(
        &self,
        _tenant: &TenantContext,
        expected: RetentionSearchCheckpoint,
        next: RetentionSearchCheckpoint,
    ) -> Result<RetentionSearchCheckpointCommitOutcome, RetentionSearchBackendError> {
        let mut state = self.state.lock().expect("backend state");
        state.commit_calls += 1;
        if state.failure_mode == FailureMode::BeforeCheckpointOnce {
            state.failure_mode = FailureMode::None;
            return Err(RetentionSearchBackendError::Unavailable);
        }
        let current = state
            .checkpoint
            .unwrap_or_else(|| RetentionSearchCheckpoint::initial(expected.community_id()));
        if current == next {
            return Ok(RetentionSearchCheckpointCommitOutcome::AlreadyCommitted);
        }
        if current != expected {
            return Err(RetentionSearchBackendError::StaleCheckpoint);
        }
        state.checkpoint = Some(next);
        if state.failure_mode == FailureMode::AfterCheckpointOnce {
            state.failure_mode = FailureMode::None;
            return Err(RetentionSearchBackendError::OutcomeUnknown);
        }
        Ok(RetentionSearchCheckpointCommitOutcome::Committed)
    }
}

#[derive(Clone, Default)]
struct RecordingProjection {
    state: Arc<Mutex<BTreeMap<u64, usize>>>,
    unavailable_once: Arc<Mutex<BTreeSet<u64>>>,
}

impl RecordingProjection {
    fn fail_once(&self, outbox_sequence: u64) {
        self.unavailable_once
            .lock()
            .expect("projection failures")
            .insert(outbox_sequence);
    }

    fn attempts(&self, outbox_sequence: u64) -> usize {
        self.state
            .lock()
            .expect("projection state")
            .get(&outbox_sequence)
            .copied()
            .unwrap_or_default()
    }
}

#[async_trait]
impl RetentionSearchProjection for RecordingProjection {
    async fn exclude_after_retention(
        &self,
        _tenant: &TenantContext,
        outbox_sequence: u64,
    ) -> Result<RetentionSearchProjectionOutcome, RetentionSearchProjectionError> {
        if self
            .unavailable_once
            .lock()
            .expect("projection failures")
            .remove(&outbox_sequence)
        {
            return Err(RetentionSearchProjectionError::Unavailable);
        }
        let mut state = self.state.lock().expect("projection state");
        let attempts = state.entry(outbox_sequence).or_default();
        *attempts += 1;
        if *attempts == 1 {
            Ok(RetentionSearchProjectionOutcome::Excluded)
        } else {
            Ok(RetentionSearchProjectionOutcome::AlreadyConverged)
        }
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "retention-search-cleanup")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn delivery(
    community_id: CommunityId,
    source_sequence: u64,
    outbox_sequence: u64,
) -> RetentionSearchDelivery {
    let token = u8::try_from(source_sequence).expect("test source sequence");
    RetentionSearchDelivery::new(
        community_id,
        RetentionSourcePosition::new(source_sequence, [token; 32]).expect("source position"),
        outbox_sequence,
    )
    .expect("delivery")
}

fn outbox_row(outbox_sequence: i64, payload: Vec<u8>) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("outbox_sequence".into(), outbox_sequence.into()),
        (
            "topic".into(),
            "collaboration.search.document.v1".to_owned().into(),
        ),
        ("source_system".into(), "zed".to_owned().into()),
        (
            "source_record_id".into(),
            "project:expired".to_owned().into(),
        ),
        ("source_version".into(), "2".to_owned().into()),
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
async fn concrete_projection_requires_retention_tombstone_before_document_write() {
    let community_id = community(1);
    let retention_payload = SearchProjectionOperation::exclude(
        SearchDocumentType::Project,
        SearchExclusionReason::RetentionExpired,
    )
    .encode()
    .expect("retention payload");
    let retention_database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![outbox_row(10, retention_payload)]])
        .append_exec_results([success(), success(), success()])
        .into_connection();
    let retention_indexer =
        CollaborationSearchIndexer::new(retention_database).expect("retention indexer");
    assert_eq!(
        retention_indexer
            .exclude_after_retention(&tenant(community_id), 10)
            .await
            .expect("retention exclusion"),
        RetentionSearchProjectionOutcome::Excluded
    );

    let public_payload = SearchProjectionOperation::upsert_community(
        SearchDocumentType::Project,
        "must not index",
        "retention cleanup accepts no public mutation",
    )
    .expect("public operation")
    .encode()
    .expect("public payload");
    let public_database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![outbox_row(11, public_payload)]])
        .append_exec_results([success()])
        .into_connection();
    let public_indexer = CollaborationSearchIndexer::new(public_database).expect("public indexer");
    assert_eq!(
        public_indexer
            .exclude_after_retention(&tenant(community_id), 11)
            .await,
        Err(RetentionSearchProjectionError::InvalidData)
    );
    let log = format!(
        "{:#?}",
        public_indexer.into_connection().into_transaction_log()
    );
    assert!(!log.contains("INSERT INTO public.collaboration_search_documents"));
}

#[tokio::test]
async fn delayed_deliveries_converge_in_source_and_outbox_order() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![
        delivery(community_id, 1, 10),
        delivery(community_id, 2, 20),
        delivery(community_id, 3, 30),
    ]);
    let projection = RecordingProjection::default();
    let cleanup = RetentionSearchCleanup::new(backend.clone(), projection.clone());

    let first = cleanup
        .run_batch(&tenant(community_id), 2)
        .await
        .expect("first delayed batch");
    assert_eq!(first.checkpoint().source_position().sequence(), 2);
    assert_eq!(first.checkpoint().outbox_sequence(), 20);
    assert!(!first.completed_batch());
    let second = cleanup
        .run_batch(&tenant(community_id), 2)
        .await
        .expect("delayed suffix");
    assert_eq!(second.checkpoint().source_position().sequence(), 3);
    assert_eq!(second.checkpoint().outbox_sequence(), 30);
    assert!(second.completed_batch());
    assert_eq!(first.counts().excluded, 2);
    assert_eq!(second.counts().excluded, 1);
    assert_eq!(backend.checkpoint(), Some(second.checkpoint()));
}

#[tokio::test]
async fn checkpoint_interruption_replays_only_the_fenced_projection() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![delivery(community_id, 1, 10)]);
    backend.set_failure_mode(FailureMode::BeforeCheckpointOnce);
    let projection = RecordingProjection::default();
    let cleanup = RetentionSearchCleanup::new(backend.clone(), projection.clone());

    assert!(matches!(
        cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionSearchCleanupError::Backend(
            RetentionSearchBackendError::Unavailable
        ))
    ));
    assert!(backend.checkpoint().is_none());
    assert_eq!(projection.attempts(10), 1);
    let resumed = cleanup
        .run_batch(&tenant(community_id), 10)
        .await
        .expect("replayed projection");
    assert_eq!(resumed.counts().already_converged, 1);
    assert_eq!(projection.attempts(10), 2);
    assert_eq!(resumed.checkpoint().converged(), 1);
}

#[tokio::test]
async fn unknown_checkpoint_outcome_resumes_after_the_committed_delivery() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![
        delivery(community_id, 1, 10),
        delivery(community_id, 2, 20),
    ]);
    backend.set_failure_mode(FailureMode::AfterCheckpointOnce);
    let projection = RecordingProjection::default();
    let cleanup = RetentionSearchCleanup::new(backend.clone(), projection.clone());

    assert!(matches!(
        cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionSearchCleanupError::Backend(
            RetentionSearchBackendError::OutcomeUnknown
        ))
    ));
    assert_eq!(
        backend.checkpoint().expect("checkpoint").outbox_sequence(),
        10
    );
    let resumed = cleanup
        .run_batch(&tenant(community_id), 10)
        .await
        .expect("resume after unknown outcome");
    assert_eq!(resumed.checkpoint().outbox_sequence(), 20);
    assert_eq!(projection.attempts(10), 1);
    assert_eq!(projection.attempts(20), 1);
}

#[tokio::test]
async fn unavailable_projection_and_invalid_batches_never_advance_checkpoint() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![delivery(community_id, 1, 10)]);
    let projection = RecordingProjection::default();
    projection.fail_once(10);
    let cleanup = RetentionSearchCleanup::new(backend.clone(), projection);
    assert!(matches!(
        cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionSearchCleanupError::Projection(
            RetentionSearchProjectionError::Unavailable
        ))
    ));
    assert!(backend.checkpoint().is_none());

    let foreign_backend = RecordingBackend::new(vec![delivery(community(2), 1, 10)]);
    let foreign_cleanup =
        RetentionSearchCleanup::new(foreign_backend.clone(), RecordingProjection::default());
    assert!(matches!(
        foreign_cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionSearchCleanupError::InvalidBatch)
    ));
    assert!(foreign_backend.checkpoint().is_none());
    assert!(matches!(
        foreign_cleanup
            .run_batch(&tenant(community_id), MAX_RETENTION_BATCH_SIZE + 1)
            .await,
        Err(RetentionSearchCleanupError::InvalidInput)
    ));
    assert!(matches!(
        RetentionSearchCheckpoint::from_record(RetentionSearchCheckpointFields {
            community_id,
            checkpoint_version: 1,
            source_position: RetentionSourcePosition::new(1, [1; 32]).expect("source position"),
            outbox_sequence: 0,
            converged: 1,
        }),
        Err(RetentionSearchCleanupError::InvalidCheckpoint)
    ));
}
