use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::retention::worker::{
    MAX_RETENTION_BATCH_SIZE, RetentionAuthorityAction, RetentionAuthorityBackend,
    RetentionAuthoritySnapshot, RetentionBackendError, RetentionBatchCommit,
    RetentionCommitOutcome, RetentionSourcePosition, RetentionWorkItem, RetentionWorker,
    RetentionWorkerCheckpoint, RetentionWorkerCheckpointFields, RetentionWorkerCounts,
    RetentionWorkerError,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, ArchiveRetentionRule, CommunityId, CommunityRetentionPolicy,
    CommunityRetentionPolicyFields, LegalHoldScope, LegalHoldSnapshot, LegalHoldState,
    RetentionEventKind, RetentionPersistenceClass, RetentionPolicySchemaVersion, RetentionRecord,
    RetentionSnapshot, RetentionTtl, RetentionVisibility, TenantContext, TrustedTenantRoute,
};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FailureMode {
    None,
    BeforeCommitOnce,
    AfterCommitOnce,
}

struct BackendState {
    checkpoint: Option<RetentionWorkerCheckpoint>,
    items: Vec<RetentionWorkItem>,
    actions: BTreeMap<u64, RetentionAuthorityAction>,
    failure_mode: FailureMode,
    load_calls: usize,
    commit_calls: usize,
}

#[derive(Clone)]
struct RecordingBackend {
    state: Arc<Mutex<BackendState>>,
}

impl RecordingBackend {
    fn new(items: Vec<RetentionWorkItem>) -> Self {
        Self {
            state: Arc::new(Mutex::new(BackendState {
                checkpoint: None,
                items,
                actions: BTreeMap::new(),
                failure_mode: FailureMode::None,
                load_calls: 0,
                commit_calls: 0,
            })),
        }
    }

    fn set_failure_mode(&self, failure_mode: FailureMode) {
        self.state.lock().expect("backend state").failure_mode = failure_mode;
    }

    fn replace_authority(&self, sequence: u64, authority: RetentionAuthoritySnapshot) {
        let mut state = self.state.lock().expect("backend state");
        let item = state
            .items
            .iter_mut()
            .find(|item| item.position().sequence() == sequence)
            .expect("work item");
        *item = RetentionWorkItem::new(item.position(), item.record(), authority);
    }

    fn checkpoint(&self) -> Option<RetentionWorkerCheckpoint> {
        self.state.lock().expect("backend state").checkpoint.clone()
    }

    fn actions(&self) -> BTreeMap<u64, RetentionAuthorityAction> {
        self.state.lock().expect("backend state").actions.clone()
    }

    fn calls(&self) -> (usize, usize) {
        let state = self.state.lock().expect("backend state");
        (state.load_calls, state.commit_calls)
    }
}

#[async_trait]
impl RetentionAuthorityBackend for RecordingBackend {
    async fn load_checkpoint(
        &self,
        _tenant: &TenantContext,
    ) -> Result<Option<RetentionWorkerCheckpoint>, RetentionBackendError> {
        Ok(self.state.lock().expect("backend state").checkpoint.clone())
    }

    async fn load_batch(
        &self,
        _tenant: &TenantContext,
        checkpoint: &RetentionWorkerCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionWorkItem>, RetentionBackendError> {
        let mut state = self.state.lock().expect("backend state");
        state.load_calls += 1;
        Ok(state
            .items
            .iter()
            .filter(|item| item.position().sequence() > checkpoint.cursor().sequence())
            .take(limit)
            .cloned()
            .collect())
    }

    async fn commit_batch(
        &self,
        _tenant: &TenantContext,
        commit: &RetentionBatchCommit,
    ) -> Result<RetentionCommitOutcome, RetentionBackendError> {
        let mut state = self.state.lock().expect("backend state");
        state.commit_calls += 1;
        if state.failure_mode == FailureMode::BeforeCommitOnce {
            state.failure_mode = FailureMode::None;
            return Err(RetentionBackendError::Unavailable);
        }
        let current = state.checkpoint.clone().unwrap_or_else(|| {
            RetentionWorkerCheckpoint::initial(commit.expected_checkpoint().community_id())
        });
        if current == *commit.next_checkpoint() {
            return Ok(RetentionCommitOutcome::AlreadyCommitted);
        }
        if current != *commit.expected_checkpoint() {
            return Err(RetentionBackendError::StaleCheckpoint);
        }
        let mut next_actions = state.actions.clone();
        for evaluation in commit.evaluations() {
            match next_actions.entry(evaluation.position().sequence()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(evaluation.action());
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if *entry.get() == evaluation.action() => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(RetentionBackendError::InvalidData);
                }
            }
        }
        state.actions = next_actions;
        state.checkpoint = Some(commit.next_checkpoint().clone());
        if state.failure_mode == FailureMode::AfterCommitOnce {
            state.failure_mode = FailureMode::None;
            return Err(RetentionBackendError::OutcomeUnknown);
        }
        Ok(RetentionCommitOutcome::Committed)
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn record(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "retention-worker")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn event_kind() -> RetentionEventKind {
    RetentionEventKind::from_registry(1, RetentionPersistenceClass::Durable).expect("event kind")
}

fn policy(community_id: CommunityId) -> CommunityRetentionPolicy {
    CommunityRetentionPolicy::from_record(CommunityRetentionPolicyFields {
        community_id,
        schema_version: RetentionPolicySchemaVersion::new(1).expect("schema"),
        version: AggregateVersion::FIRST,
        default_ttl: Some(RetentionTtl::from_millis(100).expect("ttl")),
        archive_rule: ArchiveRetentionRule::FollowCommunityPolicy,
        kind_rules: Vec::new(),
    })
    .expect("policy")
}

fn authority(community_id: CommunityId) -> RetentionAuthoritySnapshot {
    RetentionAuthoritySnapshot {
        policy: RetentionSnapshot::Current(policy(community_id)),
        legal_hold: RetentionSnapshot::Absent,
        community_archive: RetentionSnapshot::Absent,
    }
}

fn item(
    community_id: CommunityId,
    sequence: u64,
    authority: RetentionAuthoritySnapshot,
) -> RetentionWorkItem {
    let token_byte = u8::try_from(sequence).expect("test sequence");
    RetentionWorkItem::new(
        RetentionSourcePosition::new(sequence, [token_byte; 32]).expect("position"),
        RetentionRecord {
            community_id,
            record_id: record(100 + u128::from(sequence)),
            event_kind: event_kind(),
            retention_started_at_millis: 100,
        },
        authority,
    )
}

#[tokio::test]
async fn interrupted_batch_resumes_from_the_exact_committed_prefix() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![
        item(community_id, 1, authority(community_id)),
        item(community_id, 2, authority(community_id)),
        item(community_id, 3, authority(community_id)),
    ]);
    backend.set_failure_mode(FailureMode::BeforeCommitOnce);
    let worker = RetentionWorker::new(backend.clone());
    assert!(matches!(
        worker.run_batch(&tenant(community_id), 1_000, 2).await,
        Err(RetentionWorkerError::Backend(
            RetentionBackendError::Unavailable
        ))
    ));
    assert!(backend.checkpoint().is_none());
    assert!(backend.actions().is_empty());

    let first = worker
        .run_batch(&tenant(community_id), 1_000, 2)
        .await
        .expect("first committed prefix");
    assert_eq!(first.checkpoint().cursor().sequence(), 2);
    assert!(!first.completed_sweep());
    let second = worker
        .run_batch(&tenant(community_id), 1_000, 2)
        .await
        .expect("resumed suffix");
    assert!(second.completed_sweep());
    assert_eq!(second.checkpoint().cursor().sequence(), 0);
    assert_eq!(second.checkpoint().completed_sweeps(), 1);
    assert_eq!(backend.actions().len(), 3);
    assert_eq!(backend.checkpoint(), Some(second.checkpoint().clone()));
}

#[tokio::test]
async fn unknown_commit_outcome_recovers_without_reapplying_authority_actions() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![
        item(community_id, 1, authority(community_id)),
        item(community_id, 2, authority(community_id)),
        item(community_id, 3, authority(community_id)),
    ]);
    backend.set_failure_mode(FailureMode::AfterCommitOnce);
    let worker = RetentionWorker::new(backend.clone());
    assert!(matches!(
        worker.run_batch(&tenant(community_id), 1_000, 2).await,
        Err(RetentionWorkerError::Backend(
            RetentionBackendError::OutcomeUnknown
        ))
    ));
    assert_eq!(
        backend
            .checkpoint()
            .expect("committed checkpoint")
            .cursor()
            .sequence(),
        2
    );
    assert_eq!(backend.actions().len(), 2);

    let resumed = worker
        .run_batch(&tenant(community_id), 1_000, 2)
        .await
        .expect("resume after uncertain outcome");
    assert!(resumed.completed_sweep());
    assert_eq!(backend.actions().len(), 3);
    assert_eq!(resumed.checkpoint().counts().deleted, 3);
}

#[tokio::test]
async fn active_legal_hold_preserves_authority_and_archive_visibility() {
    let community_id = community(1);
    let held_record_id = record(101);
    let held_authority = RetentionAuthoritySnapshot {
        policy: RetentionSnapshot::Current(policy(community_id)),
        legal_hold: RetentionSnapshot::Current(LegalHoldSnapshot {
            community_id,
            scope: LegalHoldScope::Record(held_record_id),
            state: LegalHoldState::Active,
            version: AggregateVersion::FIRST,
        }),
        community_archive: RetentionSnapshot::Current(
            collaboration_domain::RetentionArchiveSnapshot {
                archive: collaboration_domain::CommunityArchiveSnapshot {
                    community_id,
                    state: collaboration_domain::CommunityArchivePolicyState::Archived,
                    version: AggregateVersion::FIRST,
                },
                archived_at_millis: Some(500),
            },
        ),
    };
    let backend = RecordingBackend::new(vec![item(community_id, 1, held_authority)]);
    RetentionWorker::new(backend.clone())
        .run_batch(&tenant(community_id), 1_000, 10)
        .await
        .expect("held batch");
    assert_eq!(
        backend.actions().get(&1),
        Some(&RetentionAuthorityAction::SetVisibility(
            RetentionVisibility::ArchiveOnly
        ))
    );
    assert!(!matches!(
        backend.actions().get(&1),
        Some(RetentionAuthorityAction::Delete(_))
    ));
}

#[tokio::test]
async fn unavailable_authority_commits_only_the_safe_prefix_then_resumes() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![
        item(community_id, 1, authority(community_id)),
        item(
            community_id,
            2,
            RetentionAuthoritySnapshot {
                policy: RetentionSnapshot::Unavailable,
                legal_hold: RetentionSnapshot::Absent,
                community_archive: RetentionSnapshot::Absent,
            },
        ),
        item(community_id, 3, authority(community_id)),
    ]);
    let worker = RetentionWorker::new(backend.clone());
    let partial = worker
        .run_batch(&tenant(community_id), 1_000, 3)
        .await
        .expect("safe prefix");
    assert_eq!(partial.checkpoint().cursor().sequence(), 1);
    assert_eq!(partial.halt().expect("halt").position.sequence(), 2);
    assert_eq!(
        backend.actions().keys().copied().collect::<Vec<_>>(),
        vec![1]
    );

    backend.replace_authority(2, authority(community_id));
    let resumed = worker
        .run_batch(&tenant(community_id), 1_000, 3)
        .await
        .expect("resumed partial batch");
    assert!(resumed.completed_sweep());
    assert_eq!(
        backend.actions().keys().copied().collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(resumed.checkpoint().counts().deleted, 3);
}

#[tokio::test]
async fn bounds_and_foreign_batches_fail_before_authority_commit() {
    let community_id = community(1);
    let backend = RecordingBackend::new(vec![item(community(2), 1, authority(community(2)))]);
    let worker = RetentionWorker::new(backend.clone());
    assert!(matches!(
        worker.run_batch(&tenant(community_id), 1_000, 1).await,
        Err(RetentionWorkerError::InvalidBatch)
    ));
    assert!(backend.actions().is_empty());
    assert_eq!(backend.calls(), (1, 0));
    assert!(matches!(
        worker.run_batch(&tenant(community_id), 1_000, 0).await,
        Err(RetentionWorkerError::InvalidInput)
    ));
    assert!(matches!(
        worker
            .run_batch(&tenant(community_id), 1_000, MAX_RETENTION_BATCH_SIZE + 1,)
            .await,
        Err(RetentionWorkerError::InvalidInput)
    ));
    assert_eq!(backend.calls(), (1, 0));
    assert!(matches!(
        RetentionWorkerCheckpoint::from_record(RetentionWorkerCheckpointFields {
            community_id,
            checkpoint_version: 1,
            sweep_generation: 1,
            completed_sweeps: 0,
            cursor: RetentionSourcePosition::new(1, [1; 32]).expect("position"),
            counts: RetentionWorkerCounts::default(),
        }),
        Err(RetentionWorkerError::InvalidCheckpoint)
    ));
    let hydrated = RetentionWorkerCheckpoint::from_record(RetentionWorkerCheckpointFields {
        community_id,
        checkpoint_version: 2,
        sweep_generation: 2,
        completed_sweeps: 1,
        cursor: RetentionSourcePosition::initial(),
        counts: RetentionWorkerCounts {
            scanned: 2,
            retained_live: 0,
            retained_archive_only: 0,
            deleted: 2,
        },
    })
    .expect("completed checkpoint");
    assert_eq!(hydrated.completed_sweeps(), 1);
    assert_eq!(hydrated.cursor(), RetentionSourcePosition::initial());
}
