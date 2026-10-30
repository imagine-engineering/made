use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::{
    push::outbox::{PushOutboxRepository, PushRetentionCancellationOutcome},
    retention::{
        cache_push_cleanup::{
            RetentionCacheInvalidator, RetentionDerivedBackend, RetentionDerivedBackendError,
            RetentionDerivedCheckpoint, RetentionDerivedCheckpointCommitOutcome,
            RetentionDerivedCleanup, RetentionDerivedCleanupError, RetentionDerivedCleanupItem,
            RetentionDerivedMutationOutcome, RetentionDerivedSourceId, RetentionDerivedTargetError,
            RetentionFinalVisibility, RetentionPushQueue,
        },
        worker::RetentionSourcePosition,
    },
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult};
use uuid::Uuid;

#[derive(Clone)]
struct Backend {
    state: Arc<
        Mutex<(
            Option<RetentionDerivedCheckpoint>,
            Vec<RetentionDerivedCleanupItem>,
        )>,
    >,
}

#[async_trait]
impl RetentionDerivedBackend for Backend {
    async fn load_checkpoint(
        &self,
        _tenant: &TenantContext,
    ) -> Result<Option<RetentionDerivedCheckpoint>, RetentionDerivedBackendError> {
        Ok(self.state.lock().expect("backend").0)
    }

    async fn load_batch(
        &self,
        _tenant: &TenantContext,
        checkpoint: RetentionDerivedCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionDerivedCleanupItem>, RetentionDerivedBackendError> {
        Ok(self
            .state
            .lock()
            .expect("backend")
            .1
            .iter()
            .filter(|item| {
                item.source_position().sequence() > checkpoint.source_position().sequence()
            })
            .take(limit)
            .copied()
            .collect())
    }

    async fn advance_checkpoint(
        &self,
        _tenant: &TenantContext,
        expected: RetentionDerivedCheckpoint,
        next: RetentionDerivedCheckpoint,
    ) -> Result<RetentionDerivedCheckpointCommitOutcome, RetentionDerivedBackendError> {
        let mut state = self.state.lock().expect("backend");
        let current = state
            .0
            .unwrap_or_else(|| RetentionDerivedCheckpoint::initial(expected.community_id()));
        if current == next {
            return Ok(RetentionDerivedCheckpointCommitOutcome::AlreadyCommitted);
        }
        if current != expected {
            return Err(RetentionDerivedBackendError::StaleCheckpoint);
        }
        state.0 = Some(next);
        Ok(RetentionDerivedCheckpointCommitOutcome::Committed)
    }
}

#[derive(Clone, Default)]
struct Target {
    state: Arc<Mutex<BTreeMap<[u8; 32], usize>>>,
    unavailable: Arc<Mutex<bool>>,
}

impl Target {
    fn fail_once(&self) {
        *self.unavailable.lock().expect("availability") = true;
    }

    fn attempts(&self, source: [u8; 32]) -> usize {
        self.state
            .lock()
            .expect("target")
            .get(&source)
            .copied()
            .unwrap_or_default()
    }

    fn mutate(
        &self,
        item: RetentionDerivedCleanupItem,
    ) -> Result<RetentionDerivedMutationOutcome, RetentionDerivedTargetError> {
        let mut unavailable = self.unavailable.lock().expect("availability");
        if *unavailable {
            *unavailable = false;
            return Err(RetentionDerivedTargetError::Unavailable);
        }
        let mut state = self.state.lock().expect("target");
        let attempts = state.entry(item.source_id().as_bytes()).or_default();
        *attempts += 1;
        Ok(if *attempts == 1 {
            RetentionDerivedMutationOutcome::Cleared
        } else {
            RetentionDerivedMutationOutcome::AlreadyClear
        })
    }
}

#[async_trait]
impl RetentionCacheInvalidator for Target {
    async fn invalidate_cache_and_presence(
        &self,
        _tenant: &TenantContext,
        item: RetentionDerivedCleanupItem,
    ) -> Result<RetentionDerivedMutationOutcome, RetentionDerivedTargetError> {
        self.mutate(item)
    }
}

#[async_trait]
impl RetentionPushQueue for Target {
    async fn cancel_obsolete_wakes(
        &self,
        _tenant: &TenantContext,
        item: RetentionDerivedCleanupItem,
    ) -> Result<RetentionDerivedMutationOutcome, RetentionDerivedTargetError> {
        self.mutate(item)
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(TrustedTenantRoute::from_listener(community_id, "derived-retention").expect("route")),
        &[],
    )
    .expect("tenant")
}

fn item(
    community_id: CommunityId,
    sequence: u64,
    visibility: RetentionFinalVisibility,
) -> RetentionDerivedCleanupItem {
    let byte = u8::try_from(sequence).expect("sequence");
    RetentionDerivedCleanupItem::new(
        community_id,
        RetentionSourcePosition::new(sequence, [byte; 32]).expect("position"),
        RetentionDerivedSourceId::new([byte; 32]).expect("source"),
        visibility,
        1_000 + sequence,
    )
    .expect("item")
}

fn backend(items: Vec<RetentionDerivedCleanupItem>) -> Backend {
    Backend {
        state: Arc::new(Mutex::new((None, items))),
    }
}

#[tokio::test]
async fn cleanup_advances_only_after_cache_presence_and_push_converge() {
    let community_id = community(1);
    let backend = backend(vec![
        item(community_id, 1, RetentionFinalVisibility::ArchiveOnly),
        item(community_id, 2, RetentionFinalVisibility::Deleted),
    ]);
    let cache = Target::default();
    let push = Target::default();
    let cleanup = RetentionDerivedCleanup::new(backend.clone(), cache.clone(), push.clone());
    let outcome = cleanup
        .run_batch(&tenant(community_id), 10)
        .await
        .expect("cleanup");
    assert_eq!(outcome.checkpoint().source_position().sequence(), 2);
    assert_eq!(outcome.checkpoint().converged(), 2);
    assert_eq!(outcome.counts().cache_cleared, 2);
    assert_eq!(outcome.counts().push_cleared, 2);
    assert_eq!(
        outcome.final_visibility(),
        Some(RetentionFinalVisibility::Deleted)
    );
    assert!(outcome.completed_batch());
}

#[tokio::test]
async fn unavailable_cache_or_push_preserves_retry_and_checkpoint_visibility() {
    let community_id = community(1);
    let backend = backend(vec![item(
        community_id,
        1,
        RetentionFinalVisibility::Deleted,
    )]);
    let cache = Target::default();
    let push = Target::default();
    cache.fail_once();
    let cleanup = RetentionDerivedCleanup::new(backend.clone(), cache.clone(), push.clone());
    assert!(matches!(
        cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionDerivedCleanupError::Cache(
            RetentionDerivedTargetError::Unavailable
        ))
    ));
    assert!(backend.state.lock().expect("backend").0.is_none());
    assert_eq!(cache.attempts([1; 32]), 0);
    assert_eq!(push.attempts([1; 32]), 0);

    push.fail_once();
    assert!(matches!(
        cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionDerivedCleanupError::Push(
            RetentionDerivedTargetError::Unavailable
        ))
    ));
    assert!(backend.state.lock().expect("backend").0.is_none());
    let resumed = cleanup
        .run_batch(&tenant(community_id), 10)
        .await
        .expect("retry");
    assert_eq!(resumed.counts().cache_already_clear, 1);
    assert_eq!(resumed.counts().push_cleared, 1);
    assert_eq!(cache.attempts([1; 32]), 2);
    assert_eq!(push.attempts([1; 32]), 1);
}

#[tokio::test]
async fn duplicate_cleanup_is_idempotent_and_live_visibility_is_unrepresentable() {
    let community_id = community(1);
    let cleanup_item = item(community_id, 1, RetentionFinalVisibility::ArchiveOnly);
    let backend = backend(vec![cleanup_item]);
    let cache = Target::default();
    let push = Target::default();
    cache.mutate(cleanup_item).expect("seed cache cleanup");
    push.mutate(cleanup_item).expect("seed push cleanup");
    let outcome = RetentionDerivedCleanup::new(backend, cache, push)
        .run_batch(&tenant(community_id), 10)
        .await
        .expect("duplicate cleanup");
    assert_eq!(outcome.counts().cache_already_clear, 1);
    assert_eq!(outcome.counts().push_already_clear, 1);
    assert_eq!(
        outcome.final_visibility(),
        Some(RetentionFinalVisibility::ArchiveOnly)
    );
}

#[tokio::test]
async fn foreign_and_regressing_batches_fail_before_derived_targets() {
    let community_id = community(1);
    let foreign_backend = backend(vec![item(
        community(2),
        1,
        RetentionFinalVisibility::Deleted,
    )]);
    let cache = Target::default();
    let push = Target::default();
    let cleanup = RetentionDerivedCleanup::new(foreign_backend, cache.clone(), push.clone());
    assert!(matches!(
        cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionDerivedCleanupError::InvalidBatch)
    ));
    assert_eq!(cache.attempts([1; 32]), 0);
    assert_eq!(push.attempts([1; 32]), 0);

    let backend = backend(vec![
        item(community_id, 2, RetentionFinalVisibility::Deleted),
        item(community_id, 1, RetentionFinalVisibility::Deleted),
    ]);
    let cleanup = RetentionDerivedCleanup::new(backend, cache.clone(), push.clone());
    assert!(matches!(
        cleanup.run_batch(&tenant(community_id), 10).await,
        Err(RetentionDerivedCleanupError::InvalidBatch)
    ));
    assert_eq!(cache.attempts([2; 32]), 0);
    assert_eq!(push.attempts([2; 32]), 0);
}

#[tokio::test]
async fn push_repository_deletes_only_tenant_scoped_source_wakes_and_reports_duplicates() {
    let tenant = tenant(community(1));
    let source_event_id = [7; 32];
    let repository = PushOutboxRepository::new(
        MockDatabase::new(DatabaseBackend::Postgres)
            .append_exec_results([
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected: 1,
                },
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected: 2,
                },
            ])
            .into_connection(),
    )
    .expect("Postgres repository");
    assert_eq!(
        repository
            .cancel_source_wakes_after_retention(&tenant, source_event_id)
            .await
            .expect("cancel wakes"),
        PushRetentionCancellationOutcome::Cancelled(2)
    );
    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(
        log.contains("DELETE FROM public.collaboration_push_wake_jobs"),
        "{log}"
    );
    assert!(log.contains("community_id ="), "{log}");
    assert!(log.contains("source_event_id ="), "{log}");
    assert!(!log.contains("payload"), "{log}");

    let repository = PushOutboxRepository::new(
        MockDatabase::new(DatabaseBackend::Postgres)
            .append_exec_results([
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected: 1,
                },
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected: 0,
                },
            ])
            .into_connection(),
    )
    .expect("Postgres repository");
    assert_eq!(
        repository
            .cancel_source_wakes_after_retention(&tenant, source_event_id)
            .await
            .expect("repeat cancellation"),
        PushRetentionCancellationOutcome::AlreadyClear
    );
}
