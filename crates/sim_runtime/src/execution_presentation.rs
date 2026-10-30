use crate::{
    ATTEMPT_HISTORY_SCHEMA_VERSION, AttemptEvent, AttemptEventKind, AttemptRecord, AttemptState,
    AttemptTransitionError, CompiledPlan, ExecutionAttemptPersistence, ExecutionFailure,
    ExecutionFailureOrigin, ExecutionHistory, ExecutionOutput, ExecutionPreview, ExecutionQueue,
    ExecutionQueueError, ExecutionRecoveryInterruptionReason, HistoryError,
    PersistedExecutionAttempt, PersistedExecutionProfile, QueuedPrompt, RetryPromptSource,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sim_tensor::DeviceId;
use sim_types::{AttemptId, CancellationToken, NodeId, ProfileId, PromptId, RequestId};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex, MutexGuard},
};
use thiserror::Error;
use uuid::Uuid;

pub const COMPLETED_REQUEST_CAPACITY: usize = 4_096;
pub const RECENT_COMMAND_RESULT_CAPACITY: usize = COMPLETED_REQUEST_CAPACITY;
pub const PENDING_REQUEST_CAPACITY: usize = 1_024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDataSource {
    Live,
    Persisted,
    Recovery,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExecutionSnapshotStatus {
    Loading,
    Ready,
    Stale {
        source_revision: Option<u64>,
        failure: ExecutionFailure,
    },
    Partial {
        failure: ExecutionFailure,
    },
    Unavailable {
        failure: ExecutionFailure,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeProgress {
    pub node_id: Option<NodeId>,
    pub completed: u64,
    pub total: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EffectiveNativeBackendState {
    pub device: DeviceId,
    pub device_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_memory_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocation_limit_bytes: Option<u64>,
    pub memory_limit_bytes: u64,
    pub memory_in_use_bytes: u64,
    pub memory_policy: crate::MemoryPolicy,
    pub supported_operation_rows: usize,
    pub deterministic_operation_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptPresentation {
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub retry_of: Option<AttemptId>,
    pub retry_source: Option<RetryPromptSource>,
    pub source_projection: Option<crate::AttemptSourceProjection>,
    pub state: AttemptState,
    pub last_sequence: Option<u64>,
    pub progress: Option<NodeProgress>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_backend: Option<EffectiveNativeBackendState>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub node_progress: BTreeMap<NodeId, NodeProgress>,
    pub preview: Option<ExecutionPreview>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub previews: Vec<ExecutionPreview>,
    pub outputs: Vec<ExecutionOutput>,
    pub failure: Option<ExecutionFailure>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupted_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_interruption_reason: Option<ExecutionRecoveryInterruptionReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acknowledged_state: Option<AttemptState>,
    pub canonical_event_count: usize,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl AttemptPresentation {
    fn queued(record: &AttemptRecord) -> Self {
        Self {
            profile_id: record.profile_id,
            prompt_id: record.prompt_id,
            attempt_id: record.attempt_id,
            retry_of: record.retry_of,
            retry_source: record.retry_source.clone(),
            source_projection: record.source_projection.clone(),
            state: AttemptState::Queued,
            last_sequence: None,
            progress: None,
            effective_backend: None,
            node_progress: BTreeMap::new(),
            preview: None,
            previews: Vec::new(),
            outputs: Vec::new(),
            failure: None,
            interrupted_reason: None,
            recovery_interruption_reason: None,
            acknowledged_state: None,
            canonical_event_count: 0,
            created_at: record.created_at,
            finished_at: None,
        }
    }

    fn apply_canonical(&mut self, event: &AttemptEvent, record: &AttemptRecord) {
        self.state = self
            .acknowledged_state
            .filter(|_| !record.state.is_terminal())
            .unwrap_or(record.state);
        self.last_sequence = record.last_sequence;
        self.finished_at = record.finished_at;
        self.canonical_event_count = canonical_event_count(record);
        match &event.kind {
            AttemptEventKind::Progress { completed, total } => {
                let progress = NodeProgress {
                    node_id: event.node_id.clone(),
                    completed: *completed,
                    total: *total,
                };
                if let Some(node_id) = &event.node_id {
                    self.node_progress.insert(node_id.clone(), progress.clone());
                }
                self.progress = Some(progress);
            }
            AttemptEventKind::Preview { preview } => {
                if let Some(existing) = self.previews.iter_mut().find(|current| {
                    current.node_id == preview.node_id
                        && current.frame_index == preview.frame_index
                        && current.output_index == preview.output_index
                }) {
                    *existing = preview.clone();
                } else {
                    self.previews.push(preview.clone());
                    self.previews.sort_by(|left, right| {
                        left.node_id
                            .0
                            .cmp(&right.node_id.0)
                            .then(left.frame_index.cmp(&right.frame_index))
                            .then(left.output_index.cmp(&right.output_index))
                    });
                }
                self.preview = Some(preview.clone());
            }
            AttemptEventKind::OutputAvailable { output } => {
                if let Some(existing) = self
                    .outputs
                    .iter_mut()
                    .find(|candidate| candidate.output_id == output.output_id)
                {
                    *existing = output.clone();
                } else {
                    self.outputs.push(output.clone());
                    self.outputs.sort_by(|left, right| {
                        left.node_id
                            .0
                            .cmp(&right.node_id.0)
                            .then(left.output_index.cmp(&right.output_index))
                            .then(left.output_id.cmp(&right.output_id))
                    });
                }
            }
            AttemptEventKind::Failed { failure } => {
                self.failure = Some(failure.clone());
            }
            AttemptEventKind::Interrupted { reason } => {
                self.interrupted_reason = Some(reason.clone());
            }
            AttemptEventKind::RecoveryInterrupted { reason } => {
                self.interrupted_reason = Some(reason.summary().to_owned());
                self.recovery_interruption_reason = Some(*reason);
            }
            AttemptEventKind::Started => {
                if let Some(data) = event.data.as_ref()
                    && let Some(value) = data.get("effective_native_backend")
                    && let Ok(effective_backend) = serde_json::from_value(value.clone())
                {
                    self.effective_backend = Some(effective_backend);
                }
            }
            AttemptEventKind::CacheHit
            | AttemptEventKind::OutputPrepared { .. }
            | AttemptEventKind::CancelRequested { .. }
            | AttemptEventKind::Succeeded
            | AttemptEventKind::Cancelled => {}
        }
        self.apply_output_availability_overrides(record);
        if record.state.is_terminal() {
            self.acknowledged_state = None;
            self.clear_transient_projection();
        }
    }

    fn apply_output_availability_overrides(&mut self, record: &AttemptRecord) {
        for output in &mut self.outputs {
            if let Some(availability) = record.output_availability_overrides.get(&output.output_id)
            {
                output.availability = availability.clone();
            }
        }
    }

    fn acknowledge_cancelled(&mut self, record: &AttemptRecord) {
        self.acknowledged_state = None;
        self.state = AttemptState::Cancelled;
        self.finished_at = record.finished_at;
        self.clear_transient_projection();
    }

    fn clear_transient_projection(&mut self) {
        self.progress = None;
        self.node_progress.clear();
        self.preview = None;
        self.previews.clear();
    }

    pub fn recovery_eligibility(&self) -> crate::OperationEligibility {
        match self.state {
            AttemptState::Failed => match &self.failure {
                Some(failure) if failure.retryable => crate::OperationEligibility::Allowed,
                Some(_) => crate::OperationEligibility::Unavailable {
                    reason: "the recorded failure is not recoverable".to_owned(),
                },
                None => crate::OperationEligibility::Unavailable {
                    reason: "the failed attempt has no recovery evidence".to_owned(),
                },
            },
            AttemptState::Interrupted => crate::OperationEligibility::Allowed,
            state => crate::OperationEligibility::Unavailable {
                reason: format!("attempts in state {state:?} cannot be recovered"),
            },
        }
    }

    pub fn retry_eligibility(&self) -> crate::OperationEligibility {
        match self.state {
            AttemptState::Failed => match &self.failure {
                Some(failure) if failure.retryable => crate::OperationEligibility::Allowed,
                Some(_) => crate::OperationEligibility::Unavailable {
                    reason: "the recorded failure is not retryable".to_owned(),
                },
                None => crate::OperationEligibility::Unavailable {
                    reason: "the failed attempt has no retry evidence".to_owned(),
                },
            },
            state if state.is_terminal() => crate::OperationEligibility::Allowed,
            _ => crate::OperationEligibility::Unavailable {
                reason: "only terminal attempts can be retried".to_owned(),
            },
        }
    }

    pub fn removal_eligibility(&self) -> crate::OperationEligibility {
        if self.state.is_terminal() {
            crate::OperationEligibility::Allowed
        } else {
            crate::OperationEligibility::Unavailable {
                reason: "only terminal attempts can be removed from history".to_owned(),
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum ExecutionControlCommandKind {
    Queue {
        plan: CompiledPlan,
        priority: i32,
        front: bool,
    },
    Reorder {
        attempt_id: AttemptId,
        position: usize,
    },
    Cancel {
        attempt_id: AttemptId,
        reason: String,
    },
    Interrupt {
        attempt_id: AttemptId,
        reason: String,
    },
    ClearPending {
        reason: String,
    },
    ClearHistory,
    Retry {
        attempt_id: AttemptId,
        source: RetryPromptSource,
        replacement_plan: Option<CompiledPlan>,
    },
    RemoveHistory {
        attempt_id: AttemptId,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionControlCommand {
    pub request_id: RequestId,
    pub profile_id: ProfileId,
    pub expected_revision: Option<u64>,
    pub kind: ExecutionControlCommandKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ExecutionCommandOutcome {
    Accepted {
        assigned_attempt_id: Option<AttemptId>,
    },
    Rejected {
        failure: ExecutionFailure,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionCommandAck {
    pub request_id: RequestId,
    pub profile_id: ProfileId,
    pub outcome: ExecutionCommandOutcome,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionCommandReceiptState {
    Completed(ExecutionCommandAck),
    Pending,
    NotApplied,
    ReceiptUnavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingExecutionCommand {
    pub command: ExecutionControlCommand,
    pub submitted_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionSnapshot {
    pub profile_id: ProfileId,
    pub revision: u64,
    pub source_revision: Option<u64>,
    pub source: ExecutionDataSource,
    pub status: ExecutionSnapshotStatus,
    pub queue: Vec<QueuedPrompt>,
    pub attempts: Vec<AttemptPresentation>,
    pub pending_commands: Vec<PendingExecutionCommand>,
    pub recent_command_results: Vec<ExecutionCommandAck>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptPlanSnapshot {
    pub attempt_id: AttemptId,
    pub plan: CompiledPlan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReconciliation {
    pub profile_id: ProfileId,
    pub source_revision: u64,
    pub source: ExecutionDataSource,
    pub status: ExecutionSnapshotStatus,
    pub queue: Vec<QueuedPrompt>,
    pub records: Vec<AttemptRecord>,
    pub plans: Vec<AttemptPlanSnapshot>,
    #[serde(default)]
    pub acknowledged_requests: Vec<RequestId>,
}

pub trait ExecutionController: Send + Sync {
    fn prepare<'a>(
        &'a self,
        command: &ExecutionControlCommand,
        assigned_attempt_id: Option<AttemptId>,
    ) -> Result<Box<dyn PreparedExecutionActivation + 'a>, ExecutionFailure> {
        self.accept(command, assigned_attempt_id)?;
        Ok(Box::new(ImmediateExecutionActivation))
    }

    fn accept(
        &self,
        command: &ExecutionControlCommand,
        assigned_attempt_id: Option<AttemptId>,
    ) -> Result<(), ExecutionFailure>;

    fn shutdown(&self) -> Result<(), ExecutionFailure> {
        Ok(())
    }
}

pub trait PreparedExecutionActivation: Send {
    fn commit(self: Box<Self>);
}

struct ImmediateExecutionActivation;

impl PreparedExecutionActivation for ImmediateExecutionActivation {
    fn commit(self: Box<Self>) {}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DisconnectedExecutionController;

impl DisconnectedExecutionController {
    fn runtime_unavailable() -> ExecutionFailure {
        ExecutionFailure::new(
            "runtime_controller_unavailable",
            "The native execution runtime is not connected to this profile yet.",
        )
        .with_origin(ExecutionFailureOrigin::Transport)
    }
}

impl ExecutionController for DisconnectedExecutionController {
    fn accept(
        &self,
        command: &ExecutionControlCommand,
        _assigned_attempt_id: Option<AttemptId>,
    ) -> Result<(), ExecutionFailure> {
        match &command.kind {
            ExecutionControlCommandKind::ClearHistory
            | ExecutionControlCommandKind::RemoveHistory { .. } => Ok(()),
            ExecutionControlCommandKind::Queue { .. }
            | ExecutionControlCommandKind::Reorder { .. }
            | ExecutionControlCommandKind::Cancel { .. }
            | ExecutionControlCommandKind::Interrupt { .. }
            | ExecutionControlCommandKind::ClearPending { .. }
            | ExecutionControlCommandKind::Retry { .. } => Err(Self::runtime_unavailable()),
        }
    }
}

#[derive(Clone)]
struct ProfileExecutionState {
    queue: ExecutionQueue,
    history: ExecutionHistory,
    presentations: HashMap<AttemptId, AttemptPresentation>,
    plans: HashMap<AttemptId, CompiledPlan>,
    revision: u64,
    source_revision: Option<u64>,
    source: ExecutionDataSource,
    status: ExecutionSnapshotStatus,
    recent_command_results: VecDeque<ExecutionCommandAck>,
}

impl ProfileExecutionState {
    fn new(history_capacity: usize) -> Result<Self, HistoryError> {
        Ok(Self {
            queue: ExecutionQueue::default(),
            history: ExecutionHistory::new(history_capacity)?,
            presentations: HashMap::new(),
            plans: HashMap::new(),
            revision: 0,
            source_revision: None,
            source: ExecutionDataSource::Live,
            status: ExecutionSnapshotStatus::Ready,
            recent_command_results: VecDeque::new(),
        })
    }

    fn next_revision(&mut self) -> Result<(), ExecutionPresentationError> {
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or(ExecutionPresentationError::RevisionExhausted)?;
        Ok(())
    }

    fn synchronize_retained_attempts(&mut self) {
        let retained = self
            .history
            .records()
            .iter()
            .map(|record| record.attempt_id)
            .collect::<HashSet<_>>();
        self.presentations
            .retain(|attempt_id, _| retained.contains(attempt_id));
        self.plans
            .retain(|attempt_id, _| retained.contains(attempt_id));
    }

    fn record_command_result(&mut self, ack: ExecutionCommandAck) {
        self.recent_command_results.push_back(ack);
        while self.recent_command_results.len() > RECENT_COMMAND_RESULT_CAPACITY {
            self.recent_command_results.pop_front();
        }
    }
}

pub type SharedExecutionPresentationService = Arc<ExecutionPresentationOwner>;

#[derive(Clone, Debug)]
pub struct ExecutionAttemptLease {
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub plan: CompiledPlan,
    pub cancellation: CancellationToken,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionActuatorEventInput {
    pub node_id: Option<NodeId>,
    pub kind: AttemptEventKind,
    pub data: Option<serde_json::Value>,
    pub at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionTerminationIntent {
    Cancel,
    Interrupt { reason: String },
}

pub struct ExecutionPresentationService {
    history_capacity: usize,
    profiles: HashMap<ProfileId, ProfileExecutionState>,
    pending_commands: HashMap<RequestId, PendingExecutionCommand>,
    reserved_attempts: HashMap<RequestId, AttemptId>,
    cancellation_tokens: HashMap<(ProfileId, AttemptId), CancellationToken>,
    completed_requests: VecDeque<RequestId>,
    next_attempt: Option<u128>,
}

pub struct ExecutionPresentationOwner {
    service: Mutex<ExecutionPresentationService>,
    mutation_gate: smol::lock::Mutex<()>,
    persistence: Option<Arc<dyn ExecutionAttemptPersistence>>,
}

pub(crate) struct ExecutionActuatorBatchValidator {
    staged: ExecutionPresentationService,
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
}

impl ExecutionActuatorBatchValidator {
    pub(crate) fn validate(
        &mut self,
        inputs: &[ExecutionActuatorEventInput],
    ) -> Result<(), ExecutionPresentationError> {
        self.staged
            .validate_actuator_event_batch(self.profile_id, self.prompt_id, self.attempt_id, inputs)
            .map(|_| ())
    }
}

pub(crate) enum ExecutionActuatorTransactionError<E> {
    Presentation(ExecutionPresentationError),
    Operation(E),
}

impl ExecutionPresentationOwner {
    pub fn ephemeral(service: ExecutionPresentationService) -> Arc<Self> {
        Arc::new(Self {
            service: Mutex::new(service),
            mutation_gate: smol::lock::Mutex::new(()),
            persistence: None,
        })
    }

    pub fn persistent(
        service: ExecutionPresentationService,
        persistence: Arc<dyn ExecutionAttemptPersistence>,
    ) -> Arc<Self> {
        Arc::new(Self {
            service: Mutex::new(service),
            mutation_gate: smol::lock::Mutex::new(()),
            persistence: Some(persistence),
        })
    }

    fn service(
        &self,
    ) -> Result<MutexGuard<'_, ExecutionPresentationService>, ExecutionPresentationError> {
        self.service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))
    }

    pub fn snapshot(
        &self,
        profile_id: ProfileId,
    ) -> Result<ExecutionSnapshot, ExecutionPresentationError> {
        self.service()?.snapshot(profile_id)
    }

    pub fn command_receipt_state(
        &self,
        profile_id: ProfileId,
        request_id: RequestId,
    ) -> Result<ExecutionCommandReceiptState, ExecutionPresentationError> {
        self.service()?
            .command_receipt_state(profile_id, request_id)
    }

    pub fn snapshot_with_persisted_attempts(
        &self,
        profile_id: ProfileId,
    ) -> Result<(ExecutionSnapshot, Vec<PersistedExecutionAttempt>), ExecutionPresentationError>
    {
        let service = self.service()?;
        Ok((
            service.snapshot(profile_id)?,
            service.persisted_attempts(profile_id)?,
        ))
    }

    pub fn persisted_attempts(
        &self,
        profile_id: ProfileId,
    ) -> Result<Vec<PersistedExecutionAttempt>, ExecutionPresentationError> {
        self.service()?.persisted_attempts(profile_id)
    }

    pub fn cancellation_token(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<CancellationToken, ExecutionPresentationError> {
        self.service()?.cancellation_token(profile_id, attempt_id)
    }

    pub fn termination_intent(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<Option<ExecutionTerminationIntent>, ExecutionPresentationError> {
        self.service()?.termination_intent(profile_id, attempt_id)
    }

    pub fn latest_termination_request_event(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<Option<AttemptEvent>, ExecutionPresentationError> {
        self.service()?
            .latest_termination_request_event(profile_id, attempt_id)
    }

    pub fn next_queued_attempt(
        &self,
        profile_id: ProfileId,
    ) -> Result<Option<ExecutionAttemptLease>, ExecutionPresentationError> {
        self.service()?.next_queued_attempt(profile_id)
    }

    pub fn output_operation_eligibility(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output_id: Uuid,
        action: crate::ExecutionOutputOperationAction,
    ) -> Result<crate::OperationEligibility, ExecutionPresentationError> {
        self.service()?
            .output_operation_eligibility(profile_id, attempt_id, output_id, action)
    }

    pub fn contains_canonical_event(
        &self,
        event: &AttemptEvent,
    ) -> Result<bool, ExecutionPresentationError> {
        self.service()?.contains_canonical_event(event)
    }

    pub fn validate_unapplied_event(
        &self,
        event: &AttemptEvent,
    ) -> Result<(), ExecutionPresentationError> {
        self.service()?.validate_unapplied_event(event)
    }

    pub async fn apply_event_durable(
        &self,
        event: AttemptEvent,
    ) -> Result<(), ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let profile_id = event.profile_id;
        let mut staged = self.service()?.stage_clone();
        staged.apply_event(event)?;
        self.persist_staged_profile(profile_id, &staged).await?;
        *self.service()? = staged;
        Ok(())
    }

    pub async fn dispatch_durable(
        &self,
        command: ExecutionControlCommand,
        controller: &dyn ExecutionController,
    ) -> Result<ExecutionCommandAck, ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let base = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        let mut staged = base.stage_clone();
        let (acknowledgement, accepted_command) = staged.reduce_command(command.clone())?;
        let activation = match &accepted_command {
            Some((command, assigned_attempt_id)) => {
                match controller.prepare(command, *assigned_attempt_id) {
                    Ok(activation) => Some(activation),
                    Err(failure) => {
                        let mut rejected = base;
                        rejected.submit(command.clone())?;
                        let acknowledgement = ExecutionCommandAck {
                            request_id: command.request_id,
                            profile_id: command.profile_id,
                            outcome: ExecutionCommandOutcome::Rejected { failure },
                        };
                        rejected.apply_ack_unpersisted(acknowledgement.clone())?;
                        self.persist_staged_profile(command.profile_id, &rejected)
                            .await?;
                        *self.service.lock().map_err(|error| {
                            ExecutionPresentationError::StateUnavailable(error.to_string())
                        })? = rejected;
                        return Ok(acknowledgement);
                    }
                }
            }
            None => None,
        };
        self.persist_staged_profile(acknowledgement.profile_id, &staged)
            .await?;
        if let Some((command, _)) = &accepted_command {
            staged.apply_cancellation_effects(command);
            staged.synchronize_cancellation_tokens(command.profile_id)?;
        }
        *self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))? =
            staged;
        if let Some(activation) = activation {
            activation.commit();
        }
        Ok(acknowledgement)
    }

    pub async fn apply_actuator_event_batch_durable(
        &self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        inputs: &[ExecutionActuatorEventInput],
    ) -> Result<Vec<AttemptEvent>, ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        self.apply_actuator_event_batch_while_locked(profile_id, prompt_id, attempt_id, inputs)
            .await
    }

    pub async fn initialize_profile_durable(
        &self,
        profile_id: ProfileId,
        source: ExecutionDataSource,
        status: ExecutionSnapshotStatus,
    ) -> Result<(), ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let mut staged = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        staged.initialize_profile(profile_id, source, status)?;
        self.persist_staged_profile(profile_id, &staged).await?;
        *self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))? =
            staged;
        Ok(())
    }

    pub async fn set_snapshot_status_durable(
        &self,
        profile_id: ProfileId,
        source: ExecutionDataSource,
        status: ExecutionSnapshotStatus,
    ) -> Result<(), ExecutionPresentationError> {
        self.initialize_profile_durable(profile_id, source, status)
            .await
    }

    pub async fn apply_ack_durable(
        &self,
        acknowledgement: ExecutionCommandAck,
    ) -> Result<(), ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let profile_id = acknowledgement.profile_id;
        let mut staged = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        let command = staged.apply_ack_unpersisted(acknowledgement)?;
        self.persist_staged_profile(profile_id, &staged).await?;
        staged.apply_cancellation_effects(&command);
        staged.synchronize_cancellation_tokens(profile_id)?;
        *self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))? =
            staged;
        Ok(())
    }

    pub async fn reconcile_durable(
        &self,
        reconciliation: ExecutionReconciliation,
    ) -> Result<bool, ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let profile_id = reconciliation.profile_id;
        let mut staged = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        let changed = staged.reconcile_with_change(reconciliation)?;
        self.persist_staged_profile(profile_id, &staged).await?;
        *self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))? =
            staged;
        Ok(changed)
    }

    pub async fn apply_output_operation_durable(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output_id: Uuid,
        action: crate::ExecutionOutputOperationAction,
        availability: crate::ExecutionOutputAvailability,
    ) -> Result<(), ExecutionPresentationError> {
        self.apply_output_operation_transaction_durable(
            profile_id,
            attempt_id,
            output_id,
            action,
            availability,
            || Ok(()),
            || Ok(()),
        )
        .await
    }

    pub async fn apply_output_operation_transaction_durable(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output_id: Uuid,
        action: crate::ExecutionOutputOperationAction,
        availability: crate::ExecutionOutputAvailability,
        commit: impl FnOnce() -> Result<(), String>,
        rollback: impl FnOnce() -> Result<(), String>,
    ) -> Result<(), ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let base = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        let mut staged = base.stage_clone();
        if let Err(primary) =
            staged.apply_output_operation(profile_id, attempt_id, output_id, action, availability)
        {
            return match rollback() {
                Ok(()) => Err(primary),
                Err(rollback) => Err(ExecutionPresentationError::OutputOperationRollback {
                    primary: primary.to_string(),
                    rollback,
                }),
            };
        }
        if let Err(primary) = self.persist_staged_profile(profile_id, &staged).await {
            return match rollback() {
                Ok(()) => Err(primary),
                Err(rollback) => Err(ExecutionPresentationError::OutputOperationRollback {
                    primary: primary.to_string(),
                    rollback,
                }),
            };
        }
        if let Err(primary) = commit() {
            return match self.persist_staged_profile(profile_id, &base).await {
                Ok(()) => {
                    *self.service.lock().map_err(|error| {
                        ExecutionPresentationError::StateUnavailable(error.to_string())
                    })? = base;
                    match rollback() {
                        Ok(()) => Err(ExecutionPresentationError::OutputOperationCommit(primary)),
                        Err(rollback) => Err(ExecutionPresentationError::OutputOperationRollback {
                            primary,
                            rollback,
                        }),
                    }
                }
                Err(compensation) => {
                    *self.service.lock().map_err(|error| {
                        ExecutionPresentationError::StateUnavailable(error.to_string())
                    })? = staged;
                    Err(
                        ExecutionPresentationError::OutputOperationRecoveryRequired {
                            primary,
                            compensation: compensation.to_string(),
                        },
                    )
                }
            };
        }
        *self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))? =
            staged;
        Ok(())
    }

    pub(crate) async fn apply_actuator_event_transaction_durable<T, E>(
        &self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        operation: impl FnOnce(
            Option<ExecutionTerminationIntent>,
            &mut ExecutionActuatorBatchValidator,
        ) -> Result<(Vec<ExecutionActuatorEventInput>, T), E>,
    ) -> Result<(Vec<AttemptEvent>, T), ExecutionActuatorTransactionError<E>> {
        let _mutation = self.mutation_gate.lock().await;
        let (inputs, result) = {
            let service = self
                .service()
                .map_err(ExecutionActuatorTransactionError::Presentation)?;
            let termination_intent = service
                .termination_intent(profile_id, attempt_id)
                .map_err(ExecutionActuatorTransactionError::Presentation)?;
            let mut validator = ExecutionActuatorBatchValidator {
                staged: service.stage_clone(),
                profile_id,
                prompt_id,
                attempt_id,
            };
            operation(termination_intent, &mut validator)
                .map_err(ExecutionActuatorTransactionError::Operation)?
        };
        let events = self
            .apply_actuator_event_batch_while_locked(profile_id, prompt_id, attempt_id, &inputs)
            .await
            .map_err(ExecutionActuatorTransactionError::Presentation)?;
        Ok((events, result))
    }

    async fn apply_actuator_event_batch_while_locked(
        &self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        inputs: &[ExecutionActuatorEventInput],
    ) -> Result<Vec<AttemptEvent>, ExecutionPresentationError> {
        let mut staged = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        let events =
            staged.apply_actuator_event_batch(profile_id, prompt_id, attempt_id, inputs)?;
        self.persist_staged_profile(profile_id, &staged).await?;
        *self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))? =
            staged;
        Ok(events)
    }

    pub async fn restore_profile(
        &self,
        profile_id: ProfileId,
    ) -> Result<bool, ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let Some(persistence) = &self.persistence else {
            return Ok(false);
        };
        let (profile, attempts) = persistence
            .load_execution_state(profile_id)
            .map_err(|error| ExecutionPresentationError::Persistence(error.to_string()))?;
        let mut staged = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        let metadata_changed = match profile {
            Some(profile) => {
                staged.restore_persisted_profile(profile)?;
                true
            }
            None => false,
        };
        let attempts_changed = staged.restore_persisted_attempts(profile_id, attempts)?;
        self.persist_staged_profile(profile_id, &staged).await?;
        *self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))? =
            staged;
        Ok(metadata_changed || attempts_changed)
    }

    pub async fn persist_profile_durable(
        &self,
        profile_id: ProfileId,
    ) -> Result<(), ExecutionPresentationError> {
        let _mutation = self.mutation_gate.lock().await;
        let staged = self
            .service
            .lock()
            .map_err(|error| ExecutionPresentationError::StateUnavailable(error.to_string()))?
            .stage_clone();
        self.persist_staged_profile(profile_id, &staged).await
    }

    async fn persist_staged_profile(
        &self,
        profile_id: ProfileId,
        staged: &ExecutionPresentationService,
    ) -> Result<(), ExecutionPresentationError> {
        let Some(persistence) = &self.persistence else {
            return Ok(());
        };
        persistence
            .replace_execution_state(
                staged.persisted_profile(profile_id)?,
                staged.persisted_attempts(profile_id)?,
            )
            .await
            .map_err(|error| ExecutionPresentationError::Persistence(error.to_string()))
    }
}

struct ExecutionPresentationCheckpoint {
    profiles: HashMap<ProfileId, ProfileExecutionState>,
    pending_commands: HashMap<RequestId, PendingExecutionCommand>,
    reserved_attempts: HashMap<RequestId, AttemptId>,
    cancellation_tokens: HashMap<(ProfileId, AttemptId), CancellationToken>,
    completed_requests: VecDeque<RequestId>,
    next_attempt: Option<u128>,
}

impl ExecutionPresentationService {
    pub fn new(history_capacity: usize) -> Result<Self, ExecutionPresentationError> {
        Self::new_with_first_attempt_id(history_capacity, AttemptId(Uuid::new_v4()))
    }

    pub fn new_with_first_attempt_id(
        history_capacity: usize,
        first_attempt_id: AttemptId,
    ) -> Result<Self, ExecutionPresentationError> {
        ExecutionHistory::new(history_capacity)?;
        Ok(Self {
            history_capacity,
            profiles: HashMap::new(),
            pending_commands: HashMap::new(),
            reserved_attempts: HashMap::new(),
            cancellation_tokens: HashMap::new(),
            completed_requests: VecDeque::new(),
            next_attempt: Some(first_attempt_id.0.as_u128()),
        })
    }

    pub fn initialize_profile(
        &mut self,
        profile_id: ProfileId,
        source: ExecutionDataSource,
        status: ExecutionSnapshotStatus,
    ) -> Result<(), ExecutionPresentationError> {
        self.ensure_profile(profile_id)?;
        let profile = self.profile_mut(profile_id)?;
        profile.source = source;
        profile.status = status;
        profile.next_revision()
    }

    pub fn set_snapshot_status(
        &mut self,
        profile_id: ProfileId,
        source: ExecutionDataSource,
        status: ExecutionSnapshotStatus,
    ) -> Result<(), ExecutionPresentationError> {
        self.initialize_profile(profile_id, source, status)
    }

    pub fn snapshot(
        &self,
        profile_id: ProfileId,
    ) -> Result<ExecutionSnapshot, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let mut attempts = profile.presentations.values().cloned().collect::<Vec<_>>();
        attempts.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then(left.attempt_id.0.cmp(&right.attempt_id.0))
        });
        let mut pending_commands = self
            .pending_commands
            .values()
            .filter(|pending| pending.command.profile_id == profile_id)
            .cloned()
            .collect::<Vec<_>>();
        pending_commands.sort_by(|left, right| {
            left.submitted_at
                .cmp(&right.submitted_at)
                .then(left.command.request_id.0.cmp(&right.command.request_id.0))
        });
        Ok(ExecutionSnapshot {
            profile_id,
            revision: profile.revision,
            source_revision: profile.source_revision,
            source: profile.source,
            status: profile.status.clone(),
            queue: profile.queue.items().to_vec(),
            attempts,
            pending_commands,
            recent_command_results: profile.recent_command_results.iter().cloned().collect(),
        })
    }

    pub fn command_receipt_state(
        &self,
        profile_id: ProfileId,
        request_id: RequestId,
    ) -> Result<ExecutionCommandReceiptState, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        if let Some(acknowledgement) = profile
            .recent_command_results
            .iter()
            .find(|acknowledgement| acknowledgement.request_id == request_id)
        {
            return Ok(ExecutionCommandReceiptState::Completed(
                acknowledgement.clone(),
            ));
        }
        if self
            .pending_commands
            .get(&request_id)
            .is_some_and(|pending| pending.command.profile_id == profile_id)
        {
            return Ok(ExecutionCommandReceiptState::Pending);
        }
        if self.completed_requests.contains(&request_id) {
            return Ok(ExecutionCommandReceiptState::ReceiptUnavailable);
        }
        Ok(ExecutionCommandReceiptState::NotApplied)
    }

    pub fn cancellation_token(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<CancellationToken, ExecutionPresentationError> {
        if let Some(token) = self.cancellation_tokens.get(&(profile_id, attempt_id)) {
            return Ok(token.clone());
        }
        let actual_profile = self
            .profiles
            .iter()
            .find_map(|(candidate_profile, profile)| {
                profile
                    .history
                    .record(*candidate_profile, attempt_id)
                    .map(|_| *candidate_profile)
            });
        if let Some(actual) = actual_profile
            && actual != profile_id
        {
            return Err(ExecutionPresentationError::CrossProfileAttempt {
                expected: profile_id,
                actual,
                attempt_id,
            });
        }
        Err(ExecutionPresentationError::MissingCancellationToken(
            attempt_id,
        ))
    }

    pub fn termination_intent(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<Option<ExecutionTerminationIntent>, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let record = profile
            .history
            .record(profile_id, attempt_id)
            .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
        Ok(record
            .events
            .iter()
            .rev()
            .find_map(|event| match &event.kind {
                AttemptEventKind::CancelRequested {
                    reason,
                    interrupt: true,
                } => Some(ExecutionTerminationIntent::Interrupt {
                    reason: reason.clone(),
                }),
                AttemptEventKind::CancelRequested {
                    interrupt: false, ..
                } => Some(ExecutionTerminationIntent::Cancel),
                _ => None,
            }))
    }

    pub fn latest_termination_request_event(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<Option<AttemptEvent>, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let record = profile
            .history
            .record(profile_id, attempt_id)
            .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
        Ok(record
            .events
            .iter()
            .rev()
            .find(|event| matches!(event.kind, AttemptEventKind::CancelRequested { .. }))
            .cloned())
    }

    pub fn next_queued_attempt(
        &self,
        profile_id: ProfileId,
    ) -> Result<Option<ExecutionAttemptLease>, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        profile
            .queue
            .profile_items(profile_id)
            .next()
            .map(|queued| {
                Ok(ExecutionAttemptLease {
                    profile_id,
                    prompt_id: queued.prompt_id,
                    attempt_id: queued.attempt_id,
                    plan: queued.plan.clone(),
                    cancellation: self.cancellation_token(profile_id, queued.attempt_id)?,
                })
            })
            .transpose()
    }

    pub fn apply_actuator_event(
        &mut self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: Option<NodeId>,
        kind: AttemptEventKind,
        data: Option<serde_json::Value>,
        at: DateTime<Utc>,
    ) -> Result<AttemptEvent, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let record = profile
            .history
            .record(profile_id, attempt_id)
            .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
        if record.prompt_id != prompt_id {
            return Err(ExecutionPresentationError::AttemptPromptMismatch {
                attempt_id,
                expected: record.prompt_id,
                actual: prompt_id,
            });
        }
        let sequence = record.last_sequence.map_or(Ok(0), |sequence| {
            sequence
                .checked_add(1)
                .ok_or(AttemptTransitionError::SequenceExhausted)
        })?;
        let event = AttemptEvent {
            profile_id,
            prompt_id,
            attempt_id,
            sequence,
            node_id,
            at,
            kind,
            data,
        };
        self.apply_event(event.clone())?;
        Ok(event)
    }

    pub fn validate_actuator_event_batch(
        &mut self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        inputs: &[ExecutionActuatorEventInput],
    ) -> Result<Vec<AttemptEvent>, ExecutionPresentationError> {
        let previous_profile = self
            .profiles
            .get(&profile_id)
            .cloned()
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let previous_cancellation_tokens = self.cancellation_tokens.clone();
        let result = self.apply_actuator_event_batch(profile_id, prompt_id, attempt_id, inputs);
        self.profiles.insert(profile_id, previous_profile);
        self.cancellation_tokens = previous_cancellation_tokens;
        result
    }

    pub fn apply_actuator_event_batch(
        &mut self,
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        inputs: &[ExecutionActuatorEventInput],
    ) -> Result<Vec<AttemptEvent>, ExecutionPresentationError> {
        let previous_profile = self
            .profiles
            .get(&profile_id)
            .cloned()
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let previous_cancellation_tokens = self.cancellation_tokens.clone();
        let mut applied = Vec::with_capacity(inputs.len());
        for input in inputs {
            match self.apply_actuator_event(
                profile_id,
                prompt_id,
                attempt_id,
                input.node_id.clone(),
                input.kind.clone(),
                input.data.clone(),
                input.at,
            ) {
                Ok(event) => applied.push(event),
                Err(error) => {
                    self.profiles.insert(profile_id, previous_profile);
                    self.cancellation_tokens = previous_cancellation_tokens;
                    return Err(error);
                }
            }
        }
        Ok(applied)
    }

    pub fn contains_canonical_event(
        &self,
        event: &AttemptEvent,
    ) -> Result<bool, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&event.profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(event.profile_id))?;
        let record = match profile.history.record(event.profile_id, event.attempt_id) {
            Some(record) => record,
            None => {
                if let Some(actual) = self.profiles.iter().find_map(|(profile_id, profile)| {
                    profile
                        .history
                        .record(*profile_id, event.attempt_id)
                        .map(|_| *profile_id)
                }) {
                    return Err(ExecutionPresentationError::CrossProfileAttempt {
                        expected: actual,
                        actual: event.profile_id,
                        attempt_id: event.attempt_id,
                    });
                }
                return Err(HistoryError::UnknownAttempt(event.attempt_id).into());
            }
        };
        if record.prompt_id != event.prompt_id {
            return Err(ExecutionPresentationError::AttemptPromptMismatch {
                attempt_id: event.attempt_id,
                expected: record.prompt_id,
                actual: event.prompt_id,
            });
        }
        Ok(record.events.iter().any(|candidate| candidate == event))
    }

    pub fn validate_unapplied_event(
        &self,
        event: &AttemptEvent,
    ) -> Result<(), ExecutionPresentationError> {
        let belongs_to_event_profile = self
            .profiles
            .get(&event.profile_id)
            .and_then(|profile| profile.history.record(event.profile_id, event.attempt_id))
            .is_some();
        if !belongs_to_event_profile
            && let Some(owning_profile) = self.profiles.iter().find_map(|(profile_id, profile)| {
                profile
                    .history
                    .record(*profile_id, event.attempt_id)
                    .map(|_| *profile_id)
            })
        {
            return Err(ExecutionPresentationError::CrossProfileAttempt {
                expected: owning_profile,
                actual: event.profile_id,
                attempt_id: event.attempt_id,
            });
        }
        let profile = self
            .profiles
            .get(&event.profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(event.profile_id))?;
        let mut history = profile.history.clone();
        history.apply(event.clone())?;
        Ok(())
    }

    pub fn persisted_attempts(
        &self,
        profile_id: ProfileId,
    ) -> Result<Vec<PersistedExecutionAttempt>, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        profile
            .history
            .records()
            .iter()
            .map(|record| {
                let queue = profile
                    .queue
                    .items()
                    .iter()
                    .enumerate()
                    .find(|(_, queued)| queued.attempt_id == record.attempt_id)
                    .map(|(position, queued)| crate::PersistedQueueMetadata {
                        position,
                        priority: queued.priority,
                        front: queued.front,
                        enqueue_sequence: queued.enqueue_sequence,
                        queued_at: queued.queued_at,
                    });
                PersistedExecutionAttempt::new_with_queue(
                    record.clone(),
                    profile.plans.get(&record.attempt_id).cloned(),
                    profile.source,
                    queue,
                )
                .map_err(|error| ExecutionPresentationError::Persistence(error.to_string()))
            })
            .collect()
    }

    pub fn persisted_profile(
        &self,
        profile_id: ProfileId,
    ) -> Result<PersistedExecutionProfile, ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let persisted = PersistedExecutionProfile {
            schema_version: crate::PERSISTED_EXECUTION_PROFILE_SCHEMA_VERSION,
            profile_id,
            revision: profile.revision,
            source_revision: profile.source_revision,
            source: profile.source,
            status: profile.status.clone(),
            next_attempt: self
                .next_attempt
                .map(|next_attempt| next_attempt.to_string()),
            completed_requests: self.completed_requests.iter().copied().collect(),
            recent_command_results: profile.recent_command_results.iter().cloned().collect(),
        };
        persisted
            .validate()
            .map_err(|error| ExecutionPresentationError::Persistence(error.to_string()))?;
        Ok(persisted)
    }

    pub fn restore_persisted_profile(
        &mut self,
        persisted: PersistedExecutionProfile,
    ) -> Result<(), ExecutionPresentationError> {
        persisted
            .validate()
            .map_err(|error| ExecutionPresentationError::Persistence(error.to_string()))?;
        self.ensure_profile(persisted.profile_id)?;
        let profile = self.profile_mut(persisted.profile_id)?;
        profile.revision = persisted.revision;
        profile.source_revision = persisted.source_revision;
        profile.source = persisted.source;
        profile.status = persisted.status;
        profile.recent_command_results = persisted.recent_command_results.into_iter().collect();
        for request_id in persisted.completed_requests {
            self.record_completed_request(request_id);
        }
        if let Some(next_attempt) = persisted.next_attempt {
            let next_attempt = next_attempt.parse::<u128>().map_err(|error| {
                ExecutionPresentationError::Persistence(format!(
                    "invalid next attempt cursor: {error}"
                ))
            })?;
            self.next_attempt = Some(
                self.next_attempt
                    .map_or(next_attempt, |current| current.max(next_attempt)),
            );
        }
        Ok(())
    }

    pub fn restore_persisted_attempts(
        &mut self,
        profile_id: ProfileId,
        attempts: Vec<PersistedExecutionAttempt>,
    ) -> Result<bool, ExecutionPresentationError> {
        self.ensure_profile(profile_id)?;
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let mut queue = profile.queue.items().to_vec();
        let mut records = Vec::with_capacity(attempts.len());
        let mut plans = Vec::new();
        let mut restored_queue = Vec::new();
        let mut seen_attempts = HashSet::new();
        for mut attempt in attempts {
            attempt
                .validate()
                .map_err(|error| ExecutionPresentationError::Persistence(error.to_string()))?;
            if attempt.record.profile_id != profile_id {
                return Err(ExecutionPresentationError::CrossProfileAttempt {
                    expected: profile_id,
                    actual: attempt.record.profile_id,
                    attempt_id: attempt.record.attempt_id,
                });
            }
            if !seen_attempts.insert(attempt.record.attempt_id) {
                return Err(ExecutionPresentationError::ReconciliationIdentityConflict(
                    attempt.record.attempt_id,
                ));
            }
            if matches!(
                attempt.record.state,
                AttemptState::Running | AttemptState::Cancelling
            ) {
                let sequence = match attempt.record.last_sequence {
                    Some(sequence) => sequence
                        .checked_add(1)
                        .ok_or(AttemptTransitionError::SequenceExhausted)?,
                    None => 0,
                };
                attempt.record.apply(AttemptEvent {
                    profile_id,
                    prompt_id: attempt.record.prompt_id,
                    attempt_id: attempt.record.attempt_id,
                    sequence,
                    node_id: None,
                    at: Utc::now(),
                    kind: AttemptEventKind::RecoveryInterrupted {
                        reason: ExecutionRecoveryInterruptionReason::RuntimeRestart,
                    },
                    data: None,
                })?;
            }
            validate_record_integrity(&attempt.record)?;
            if attempt.record.state == AttemptState::Queued
                && !queue
                    .iter()
                    .any(|item| item.attempt_id == attempt.record.attempt_id)
            {
                let plan = attempt
                    .plan
                    .as_ref()
                    .ok_or(ExecutionPresentationError::MissingPlan(
                        attempt.record.attempt_id,
                    ))?;
                let legacy_position = restored_queue.len();
                let metadata = attempt
                    .queue
                    .clone()
                    .unwrap_or(crate::PersistedQueueMetadata {
                        position: legacy_position,
                        priority: 0,
                        front: false,
                        enqueue_sequence: u64::try_from(legacy_position)
                            .map_err(|_| ExecutionQueueError::SequenceExhausted)?,
                        queued_at: attempt.record.created_at,
                    });
                restored_queue.push((
                    metadata.position,
                    QueuedPrompt {
                        profile_id,
                        prompt_id: attempt.record.prompt_id,
                        attempt_id: attempt.record.attempt_id,
                        plan: plan.clone(),
                        priority: metadata.priority,
                        front: metadata.front,
                        enqueue_sequence: metadata.enqueue_sequence,
                        queued_at: metadata.queued_at,
                    },
                ));
            }
            if let Some(plan) = attempt.plan {
                plans.push(AttemptPlanSnapshot {
                    attempt_id: attempt.record.attempt_id,
                    plan,
                });
            }
            records.push(attempt.record);
        }
        restored_queue.sort_by_key(|(position, _)| *position);
        if !restored_queue
            .iter()
            .enumerate()
            .all(|(expected, (position, _))| expected == *position)
        {
            return Err(ExecutionPresentationError::Persistence(
                "persisted execution queue positions are not contiguous".to_owned(),
            ));
        }
        let restored_queue = ExecutionQueue::from_ordered(
            restored_queue
                .into_iter()
                .map(|(_, queued)| queued)
                .collect(),
        )?;
        queue.extend(restored_queue.items().iter().cloned());
        let source_revision = profile.source_revision.map_or(Ok(0), |revision| {
            revision
                .checked_add(1)
                .ok_or(ExecutionPresentationError::RevisionExhausted)
        })?;
        self.reconcile_with_change(ExecutionReconciliation {
            profile_id,
            source_revision,
            source: ExecutionDataSource::Recovery,
            status: profile.status.clone(),
            queue,
            records,
            plans,
            acknowledged_requests: Vec::new(),
        })
    }

    pub fn submit(
        &mut self,
        command: ExecutionControlCommand,
    ) -> Result<(), ExecutionPresentationError> {
        self.ensure_profile(command.profile_id)?;
        if self.pending_commands.contains_key(&command.request_id)
            || self.completed_requests.contains(&command.request_id)
        {
            return Err(ExecutionPresentationError::DuplicateRequest(
                command.request_id,
            ));
        }
        if self.pending_commands.len() >= PENDING_REQUEST_CAPACITY {
            return Err(ExecutionPresentationError::PendingRequestCapacity {
                maximum: PENDING_REQUEST_CAPACITY,
            });
        }
        let profile = self.profile_mut(command.profile_id)?;
        if let Some(expected_revision) = command.expected_revision
            && expected_revision != profile.revision
        {
            return Err(ExecutionPresentationError::RevisionMismatch {
                expected: expected_revision,
                actual: profile.revision,
            });
        }
        let reserved_attempt = if command_requires_attempt_identity(&command.kind) {
            Some(self.allocate_attempt_id()?)
        } else {
            None
        };
        let profile = self.profile_mut(command.profile_id)?;
        profile.next_revision()?;
        if let Some(attempt_id) = reserved_attempt {
            self.reserved_attempts
                .insert(command.request_id, attempt_id);
        }
        self.pending_commands.insert(
            command.request_id,
            PendingExecutionCommand {
                command,
                submitted_at: Utc::now(),
            },
        );
        Ok(())
    }

    pub fn dispatch(
        &mut self,
        command: ExecutionControlCommand,
        controller: &dyn ExecutionController,
    ) -> Result<ExecutionCommandAck, ExecutionPresentationError> {
        let checkpoint = self.checkpoint();
        let (acknowledgement, accepted_command) = match self.reduce_command(command.clone()) {
            Ok(result) => result,
            Err(error) => {
                self.restore_checkpoint(checkpoint);
                return Err(error);
            }
        };
        let Some((accepted_command, assigned_attempt_id)) = accepted_command else {
            return Ok(acknowledgement);
        };
        if let Err(failure) = controller.accept(&accepted_command, assigned_attempt_id) {
            self.restore_checkpoint(checkpoint);
            self.submit(command.clone())?;
            let rejected = ExecutionCommandAck {
                request_id: command.request_id,
                profile_id: command.profile_id,
                outcome: ExecutionCommandOutcome::Rejected { failure },
            };
            self.apply_ack_unpersisted(rejected.clone())?;
            return Ok(rejected);
        }
        self.apply_cancellation_effects(&accepted_command);
        self.synchronize_cancellation_tokens(accepted_command.profile_id)?;
        Ok(acknowledgement)
    }

    pub fn apply_ack(
        &mut self,
        ack: ExecutionCommandAck,
    ) -> Result<(), ExecutionPresentationError> {
        let command = self.apply_ack_unpersisted(ack)?;
        self.apply_cancellation_effects(&command);
        self.synchronize_cancellation_tokens(command.profile_id)?;
        Ok(())
    }

    fn reduce_command(
        &mut self,
        command: ExecutionControlCommand,
    ) -> Result<
        (
            ExecutionCommandAck,
            Option<(ExecutionControlCommand, Option<AttemptId>)>,
        ),
        ExecutionPresentationError,
    > {
        self.ensure_profile(command.profile_id)?;
        self.submit(command.clone())?;
        let assigned_attempt_id = self.reserved_attempts.get(&command.request_id).copied();
        if let Some(failure) = self.command_validation_failure(&command)? {
            let acknowledgement = ExecutionCommandAck {
                request_id: command.request_id,
                profile_id: command.profile_id,
                outcome: ExecutionCommandOutcome::Rejected { failure },
            };
            self.apply_ack_unpersisted(acknowledgement.clone())?;
            return Ok((acknowledgement, None));
        }
        let acknowledgement = ExecutionCommandAck {
            request_id: command.request_id,
            profile_id: command.profile_id,
            outcome: ExecutionCommandOutcome::Accepted {
                assigned_attempt_id,
            },
        };
        self.apply_ack_unpersisted(acknowledgement.clone())?;
        Ok((acknowledgement, Some((command, assigned_attempt_id))))
    }

    fn apply_ack_unpersisted(
        &mut self,
        ack: ExecutionCommandAck,
    ) -> Result<ExecutionControlCommand, ExecutionPresentationError> {
        let pending = self
            .pending_commands
            .get(&ack.request_id)
            .cloned()
            .ok_or(ExecutionPresentationError::UnknownRequest(ack.request_id))?;
        if ack.profile_id != pending.command.profile_id {
            return Err(ExecutionPresentationError::AckProfileMismatch {
                expected: pending.command.profile_id,
                actual: ack.profile_id,
            });
        }
        let previous_profile = self
            .profiles
            .get(&ack.profile_id)
            .cloned()
            .ok_or(ExecutionPresentationError::UnknownProfile(ack.profile_id))?;
        let canonical_ack = match ack.outcome {
            ExecutionCommandOutcome::Accepted { .. } => ExecutionCommandAck {
                request_id: ack.request_id,
                profile_id: ack.profile_id,
                outcome: ExecutionCommandOutcome::Accepted {
                    assigned_attempt_id: self.reserved_attempts.get(&ack.request_id).copied(),
                },
            },
            ExecutionCommandOutcome::Rejected { failure } => ExecutionCommandAck {
                request_id: ack.request_id,
                profile_id: ack.profile_id,
                outcome: ExecutionCommandOutcome::Rejected { failure },
            },
        };
        let mutation_result = match &canonical_ack.outcome {
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id,
            } => self.apply_accepted_command(&pending.command, *assigned_attempt_id),
            ExecutionCommandOutcome::Rejected { .. } => Ok(()),
        };
        if mutation_result.is_err() {
            self.profiles.insert(ack.profile_id, previous_profile);
        }
        self.pending_commands.remove(&ack.request_id);
        self.reserved_attempts.remove(&ack.request_id);
        self.record_completed_request(ack.request_id);
        let profile = self.profile_mut(ack.profile_id)?;
        profile.record_command_result(canonical_ack);
        profile.next_revision()?;
        mutation_result.map(|()| pending.command)
    }

    pub fn apply_event(&mut self, event: AttemptEvent) -> Result<(), ExecutionPresentationError> {
        let belongs_to_event_profile = self
            .profiles
            .get(&event.profile_id)
            .and_then(|profile| profile.history.record(event.profile_id, event.attempt_id))
            .is_some();
        if !belongs_to_event_profile
            && let Some(owning_profile) = self.profiles.iter().find_map(|(profile_id, profile)| {
                profile
                    .history
                    .record(*profile_id, event.attempt_id)
                    .map(|_| *profile_id)
            })
        {
            return Err(ExecutionPresentationError::CrossProfileAttempt {
                expected: owning_profile,
                actual: event.profile_id,
                attempt_id: event.attempt_id,
            });
        }
        let profile = self.profile_mut(event.profile_id)?;
        profile.history.apply(event.clone())?;
        if matches!(&event.kind, AttemptEventKind::Started) {
            profile
                .queue
                .cancel_queued(event.profile_id, event.attempt_id);
        }
        let record = profile
            .history
            .record(event.profile_id, event.attempt_id)
            .ok_or(HistoryError::UnknownAttempt(event.attempt_id))?;
        let presentation = profile
            .presentations
            .entry(event.attempt_id)
            .or_insert_with(|| AttemptPresentation::queued(record));
        presentation.apply_canonical(&event, record);
        profile.next_revision()?;
        self.synchronize_cancellation_tokens(event.profile_id)
    }

    pub fn apply_output_operation(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output_id: Uuid,
        action: crate::ExecutionOutputOperationAction,
        availability: crate::ExecutionOutputAvailability,
    ) -> Result<(), ExecutionPresentationError> {
        let belongs_to_profile = self
            .profiles
            .get(&profile_id)
            .and_then(|profile| profile.history.record(profile_id, attempt_id))
            .is_some();
        if !belongs_to_profile
            && let Some(owning_profile) = self.profiles.iter().find_map(|(candidate, profile)| {
                profile
                    .history
                    .record(*candidate, attempt_id)
                    .map(|_| *candidate)
            })
        {
            return Err(ExecutionPresentationError::CrossProfileAttempt {
                expected: owning_profile,
                actual: profile_id,
                attempt_id,
            });
        }
        let previous_profile = self
            .profiles
            .get(&profile_id)
            .cloned()
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let mutation_result = (|| {
            let profile = self.profile_mut(profile_id)?;
            let presentation_has_output = profile
                .presentations
                .get(&attempt_id)
                .and_then(|presentation| {
                    presentation
                        .outputs
                        .iter()
                        .find(|output| output.output_id == output_id)
                })
                .is_some();
            if !presentation_has_output {
                return Err(ExecutionPresentationError::InconsistentRecord(attempt_id));
            }
            let record = profile
                .history
                .record_mut(profile_id, attempt_id)
                .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
            record.apply_output_operation(output_id, action, availability.clone())?;
            let presentation = profile
                .presentations
                .get_mut(&attempt_id)
                .ok_or(ExecutionPresentationError::InconsistentRecord(attempt_id))?;
            let output = presentation
                .outputs
                .iter_mut()
                .find(|output| output.output_id == output_id)
                .ok_or(ExecutionPresentationError::InconsistentRecord(attempt_id))?;
            output.availability = availability;
            profile.next_revision()
        })();
        if mutation_result.is_err() {
            self.profiles.insert(profile_id, previous_profile);
        }
        mutation_result
    }

    pub fn output_operation_eligibility(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output_id: Uuid,
        action: crate::ExecutionOutputOperationAction,
    ) -> Result<crate::OperationEligibility, ExecutionPresentationError> {
        let profile = match self.profiles.get(&profile_id) {
            Some(profile) => profile,
            None => {
                if let Some(owning_profile) =
                    self.profiles.iter().find_map(|(candidate, profile)| {
                        profile
                            .history
                            .record(*candidate, attempt_id)
                            .map(|_| *candidate)
                    })
                {
                    return Err(ExecutionPresentationError::CrossProfileAttempt {
                        expected: owning_profile,
                        actual: profile_id,
                        attempt_id,
                    });
                }
                return Err(ExecutionPresentationError::UnknownProfile(profile_id));
            }
        };
        let record = match profile.history.record(profile_id, attempt_id) {
            Some(record) => record,
            None => {
                if let Some(owning_profile) =
                    self.profiles.iter().find_map(|(candidate, profile)| {
                        profile
                            .history
                            .record(*candidate, attempt_id)
                            .map(|_| *candidate)
                    })
                {
                    return Err(ExecutionPresentationError::CrossProfileAttempt {
                        expected: owning_profile,
                        actual: profile_id,
                        attempt_id,
                    });
                }
                return Err(HistoryError::UnknownAttempt(attempt_id).into());
            }
        };
        record
            .output_operation_eligibility(output_id, action)
            .map_err(Into::into)
    }

    pub fn reconcile(
        &mut self,
        reconciliation: ExecutionReconciliation,
    ) -> Result<(), ExecutionPresentationError> {
        self.reconcile_with_change(reconciliation).map(|_| ())
    }

    pub fn reconcile_with_change(
        &mut self,
        reconciliation: ExecutionReconciliation,
    ) -> Result<bool, ExecutionPresentationError> {
        if let Some(profile) = self.profiles.get(&reconciliation.profile_id)
            && let Some(current_revision) = profile.source_revision
            && reconciliation.source_revision <= current_revision
        {
            return Err(ExecutionPresentationError::StaleReconciliation {
                current: current_revision,
                received: reconciliation.source_revision,
            });
        }
        let ExecutionReconciliation {
            profile_id,
            source_revision,
            source,
            status,
            queue: queue_items,
            records,
            plans,
            acknowledged_requests,
        } = reconciliation;
        if let Some(item) = queue_items
            .iter()
            .find(|item| item.profile_id != profile_id)
        {
            return Err(ExecutionPresentationError::CrossProfileAttempt {
                expected: profile_id,
                actual: item.profile_id,
                attempt_id: item.attempt_id,
            });
        }
        ExecutionQueue::from_ordered(queue_items.clone())?;
        self.ensure_profile(profile_id)?;
        let previous = self
            .profiles
            .get(&profile_id)
            .cloned()
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;

        let mut merged_records = previous
            .history
            .records()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for record in records {
            if record.profile_id != profile_id {
                return Err(ExecutionPresentationError::CrossProfileAttempt {
                    expected: profile_id,
                    actual: record.profile_id,
                    attempt_id: record.attempt_id,
                });
            }
            validate_record_integrity(&record)?;
            if let Some(existing) = merged_records
                .iter_mut()
                .find(|existing| existing.attempt_id == record.attempt_id)
            {
                if existing.prompt_id != record.prompt_id {
                    return Err(ExecutionPresentationError::ReconciliationIdentityConflict(
                        record.attempt_id,
                    ));
                }
                match record.last_sequence.cmp(&existing.last_sequence) {
                    std::cmp::Ordering::Greater => *existing = record,
                    std::cmp::Ordering::Equal => {
                        if !records_share_canonical_projection(existing, &record) {
                            return Err(
                                ExecutionPresentationError::ReconciliationIdentityConflict(
                                    record.attempt_id,
                                ),
                            );
                        }
                        let source_projection = record
                            .source_projection
                            .clone()
                            .or_else(|| existing.source_projection.clone());
                        if record.events.len() > existing.events.len() {
                            *existing = record;
                        }
                        existing.source_projection = source_projection;
                    }
                    std::cmp::Ordering::Less => {}
                }
            } else {
                merged_records.push(record);
            }
        }
        let mut merged_plans = previous.plans.clone();
        let mut reconciled_plan_ids = HashSet::new();
        for snapshot in plans {
            if !reconciled_plan_ids.insert(snapshot.attempt_id) {
                return Err(ExecutionPresentationError::DuplicatePlan(
                    snapshot.attempt_id,
                ));
            }
            let record = merged_records
                .iter()
                .find(|record| record.attempt_id == snapshot.attempt_id)
                .ok_or(HistoryError::UnknownAttempt(snapshot.attempt_id))?;
            if snapshot.plan.prompt_id != record.prompt_id {
                return Err(ExecutionPresentationError::PlanPromptMismatch {
                    attempt_id: snapshot.attempt_id,
                });
            }
            if let Some(existing) = merged_plans.get(&snapshot.attempt_id) {
                if existing.prompt_id != snapshot.plan.prompt_id {
                    return Err(ExecutionPresentationError::PlanPromptMismatch {
                        attempt_id: snapshot.attempt_id,
                    });
                }
                if existing != &snapshot.plan {
                    return Err(ExecutionPresentationError::ReconciliationIdentityConflict(
                        snapshot.attempt_id,
                    ));
                }
            }
            merged_plans.insert(snapshot.attempt_id, snapshot.plan);
        }
        let mut merged_queue_items = queue_items;
        for existing in previous.queue.items() {
            let already_present = merged_queue_items
                .iter()
                .any(|item| item.attempt_id == existing.attempt_id);
            let prompt_replaced = merged_queue_items
                .iter()
                .any(|item| item.prompt_id == existing.prompt_id);
            let still_queued = merged_records.iter().any(|record| {
                record.attempt_id == existing.attempt_id && record.state == AttemptState::Queued
            });
            if !already_present && !prompt_replaced && still_queued {
                merged_queue_items.push(existing.clone());
            }
        }
        let queue = ExecutionQueue::from_ordered(merged_queue_items)?;
        for item in queue.items() {
            let record = merged_records
                .iter()
                .find(|record| record.attempt_id == item.attempt_id)
                .ok_or(HistoryError::UnknownAttempt(item.attempt_id))?;
            if record.state != AttemptState::Queued || item.prompt_id != record.prompt_id {
                return Err(ExecutionPresentationError::InvalidQueuedRecord(
                    item.attempt_id,
                ));
            }
            match merged_plans.get(&item.attempt_id) {
                Some(plan) if plan == &item.plan => {}
                _ => {
                    return Err(ExecutionPresentationError::MissingPlan(item.attempt_id));
                }
            }
        }
        let mut history = ExecutionHistory::new(self.history_capacity)?;
        for record in merged_records {
            history.insert(record)?;
        }
        let mut presentations = HashMap::new();
        for record in history.records() {
            let mut presentation = reduce_record(record);
            if let Some(previous_presentation) = previous.presentations.get(&record.attempt_id)
                && previous_presentation.acknowledged_state.is_some()
                && !record.state.is_terminal()
            {
                presentation.acknowledged_state = previous_presentation.acknowledged_state;
                presentation.state = previous_presentation.state;
            }
            presentations.insert(record.attempt_id, presentation);
        }
        let visible_changed = previous.queue.items() != queue.items()
            || previous.presentations != presentations
            || previous.source != source
            || previous.status != status;
        let mut profile = ProfileExecutionState {
            queue,
            history,
            presentations,
            plans: merged_plans,
            revision: previous.revision,
            source_revision: Some(source_revision),
            source,
            status,
            recent_command_results: previous.recent_command_results,
        };
        let mut request_changed = false;
        for request_id in acknowledged_requests {
            if self
                .pending_commands
                .get(&request_id)
                .is_some_and(|pending| pending.command.profile_id == profile_id)
            {
                self.pending_commands.remove(&request_id);
                self.reserved_attempts.remove(&request_id);
                self.record_completed_request(request_id);
                request_changed = true;
            }
        }
        if visible_changed || request_changed {
            profile.next_revision()?;
        }
        self.profiles.insert(profile_id, profile);
        self.synchronize_cancellation_tokens(profile_id)?;
        Ok(visible_changed || request_changed)
    }

    fn allocate_attempt_id(&mut self) -> Result<AttemptId, ExecutionPresentationError> {
        let occupied = self
            .profiles
            .values()
            .flat_map(|profile| {
                profile
                    .history
                    .records()
                    .iter()
                    .map(|record| record.attempt_id)
            })
            .chain(self.reserved_attempts.values().copied())
            .collect::<HashSet<_>>();
        for _ in 0..=occupied.len() {
            let candidate = self
                .next_attempt
                .map(|value| AttemptId(Uuid::from_u128(value)))
                .ok_or(ExecutionPresentationError::AttemptIdentityExhausted)?;
            self.next_attempt = candidate.0.as_u128().checked_add(1);
            if !occupied.contains(&candidate) {
                return Ok(candidate);
            }
        }
        Err(ExecutionPresentationError::AttemptIdentityExhausted)
    }

    fn command_validation_failure(
        &self,
        command: &ExecutionControlCommand,
    ) -> Result<Option<ExecutionFailure>, ExecutionPresentationError> {
        let snapshot = self.snapshot(command.profile_id)?;
        let invalid = |code: &str, message: String| {
            Some(
                ExecutionFailure::new(code, message)
                    .with_origin(ExecutionFailureOrigin::Validation),
            )
        };
        let failure = match &command.kind {
            ExecutionControlCommandKind::Queue { .. }
            | ExecutionControlCommandKind::ClearPending { .. }
            | ExecutionControlCommandKind::ClearHistory => None,
            ExecutionControlCommandKind::Reorder {
                attempt_id,
                position,
            } => {
                let queued = snapshot
                    .queue
                    .iter()
                    .any(|item| item.attempt_id == *attempt_id);
                if !queued || *position >= snapshot.queue.len() {
                    invalid(
                        "invalid_queue_position",
                        "the queued attempt or destination position is no longer available"
                            .to_owned(),
                    )
                } else {
                    None
                }
            }
            ExecutionControlCommandKind::Cancel { attempt_id, .. } => {
                let cancellable = snapshot.attempts.iter().any(|attempt| {
                    attempt.attempt_id == *attempt_id
                        && matches!(attempt.state, AttemptState::Queued | AttemptState::Running)
                });
                if cancellable {
                    None
                } else {
                    invalid(
                        "attempt_not_cancellable",
                        "the attempt is no longer cancellable".to_owned(),
                    )
                }
            }
            ExecutionControlCommandKind::Interrupt { attempt_id, .. } => {
                let interruptible = snapshot.attempts.iter().any(|attempt| {
                    attempt.attempt_id == *attempt_id
                        && matches!(attempt.state, AttemptState::Queued | AttemptState::Running)
                });
                if interruptible {
                    None
                } else {
                    invalid(
                        "attempt_not_interruptible",
                        "the attempt is no longer interruptible".to_owned(),
                    )
                }
            }
            ExecutionControlCommandKind::Retry { attempt_id, .. } => {
                let eligibility = snapshot
                    .attempts
                    .iter()
                    .find(|attempt| attempt.attempt_id == *attempt_id)
                    .map(AttemptPresentation::retry_eligibility)
                    .unwrap_or_else(|| crate::OperationEligibility::Unavailable {
                        reason: "the attempt does not exist".to_owned(),
                    });
                match eligibility {
                    crate::OperationEligibility::Allowed => None,
                    crate::OperationEligibility::Unavailable { reason } => {
                        invalid("attempt_not_retryable", reason)
                    }
                }
            }
            ExecutionControlCommandKind::RemoveHistory { attempt_id } => {
                let removable = snapshot.attempts.iter().any(|attempt| {
                    attempt.attempt_id == *attempt_id && attempt.removal_eligibility().is_allowed()
                });
                if removable {
                    None
                } else {
                    invalid(
                        "attempt_not_removable",
                        "only terminal attempts can be removed".to_owned(),
                    )
                }
            }
        };
        Ok(failure)
    }

    fn apply_cancellation_effects(&self, command: &ExecutionControlCommand) {
        match &command.kind {
            ExecutionControlCommandKind::Cancel { attempt_id, .. }
            | ExecutionControlCommandKind::Interrupt { attempt_id, .. } => {
                if let Some(token) = self
                    .cancellation_tokens
                    .get(&(command.profile_id, *attempt_id))
                {
                    token.cancel();
                }
            }
            ExecutionControlCommandKind::ClearPending { .. } => {
                let profile = self.profiles.get(&command.profile_id);
                for ((profile_id, attempt_id), token) in &self.cancellation_tokens {
                    let was_cancelled = profile
                        .and_then(|profile| profile.history.record(command.profile_id, *attempt_id))
                        .is_some_and(|record| record.state == AttemptState::Cancelled);
                    if *profile_id == command.profile_id && was_cancelled {
                        token.cancel();
                    }
                }
            }
            _ => {}
        }
    }

    fn synchronize_cancellation_tokens(
        &mut self,
        profile_id: ProfileId,
    ) -> Result<(), ExecutionPresentationError> {
        let profile = self
            .profiles
            .get(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))?;
        let live_attempts = profile
            .history
            .records()
            .iter()
            .filter(|record| !record.state.is_terminal())
            .map(|record| record.attempt_id)
            .collect::<HashSet<_>>();
        self.cancellation_tokens
            .retain(|(candidate_profile, attempt_id), _| {
                *candidate_profile != profile_id || live_attempts.contains(attempt_id)
            });
        for attempt_id in live_attempts {
            self.cancellation_tokens
                .entry((profile_id, attempt_id))
                .or_default();
        }
        Ok(())
    }

    fn checkpoint(&self) -> ExecutionPresentationCheckpoint {
        ExecutionPresentationCheckpoint {
            profiles: self.profiles.clone(),
            pending_commands: self.pending_commands.clone(),
            reserved_attempts: self.reserved_attempts.clone(),
            cancellation_tokens: self.cancellation_tokens.clone(),
            completed_requests: self.completed_requests.clone(),
            next_attempt: self.next_attempt,
        }
    }

    fn restore_checkpoint(&mut self, checkpoint: ExecutionPresentationCheckpoint) {
        self.profiles = checkpoint.profiles;
        self.pending_commands = checkpoint.pending_commands;
        self.reserved_attempts = checkpoint.reserved_attempts;
        self.cancellation_tokens = checkpoint.cancellation_tokens;
        self.completed_requests = checkpoint.completed_requests;
        self.next_attempt = checkpoint.next_attempt;
    }

    fn stage_clone(&self) -> Self {
        Self {
            history_capacity: self.history_capacity,
            profiles: self.profiles.clone(),
            pending_commands: self.pending_commands.clone(),
            reserved_attempts: self.reserved_attempts.clone(),
            cancellation_tokens: self.cancellation_tokens.clone(),
            completed_requests: self.completed_requests.clone(),
            next_attempt: self.next_attempt,
        }
    }

    fn ensure_profile(&mut self, profile_id: ProfileId) -> Result<(), ExecutionPresentationError> {
        if !self.profiles.contains_key(&profile_id) {
            self.profiles.insert(
                profile_id,
                ProfileExecutionState::new(self.history_capacity)?,
            );
        }
        Ok(())
    }

    fn record_completed_request(&mut self, request_id: RequestId) {
        if self.completed_requests.contains(&request_id) {
            return;
        }
        self.completed_requests.push_back(request_id);
        while self.completed_requests.len() > COMPLETED_REQUEST_CAPACITY {
            self.completed_requests.pop_front();
        }
    }

    fn profile_mut(
        &mut self,
        profile_id: ProfileId,
    ) -> Result<&mut ProfileExecutionState, ExecutionPresentationError> {
        self.profiles
            .get_mut(&profile_id)
            .ok_or(ExecutionPresentationError::UnknownProfile(profile_id))
    }

    fn apply_accepted_command(
        &mut self,
        command: &ExecutionControlCommand,
        assigned_attempt_id: Option<AttemptId>,
    ) -> Result<(), ExecutionPresentationError> {
        let profile = self.profile_mut(command.profile_id)?;
        match &command.kind {
            ExecutionControlCommandKind::Queue {
                plan,
                priority,
                front,
            } => {
                let attempt_id = assigned_attempt_id
                    .ok_or(ExecutionPresentationError::MissingAssignedAttempt)?;
                profile.queue.enqueue(
                    command.profile_id,
                    plan.clone(),
                    attempt_id,
                    *priority,
                    *front,
                )?;
                let record = AttemptRecord::queued(command.profile_id, plan.prompt_id, attempt_id);
                profile.history.insert(record.clone())?;
                profile
                    .presentations
                    .insert(attempt_id, AttemptPresentation::queued(&record));
                profile.plans.insert(attempt_id, plan.clone());
                profile.synchronize_retained_attempts();
            }
            ExecutionControlCommandKind::Reorder {
                attempt_id,
                position,
            } => profile
                .queue
                .reorder(command.profile_id, *attempt_id, *position)?,
            ExecutionControlCommandKind::Cancel { attempt_id, reason } => {
                if profile
                    .queue
                    .cancel_queued(command.profile_id, *attempt_id)
                    .is_some()
                {
                    acknowledge_queued_cancellation(profile, command.profile_id, *attempt_id)?;
                } else {
                    acknowledge_running_cancellation(
                        profile,
                        command.profile_id,
                        *attempt_id,
                        reason,
                        false,
                    )?;
                }
            }
            ExecutionControlCommandKind::Interrupt { attempt_id, reason } => {
                if profile
                    .queue
                    .cancel_queued(command.profile_id, *attempt_id)
                    .is_some()
                {
                    acknowledge_queued_cancellation(profile, command.profile_id, *attempt_id)?;
                } else {
                    acknowledge_running_cancellation(
                        profile,
                        command.profile_id,
                        *attempt_id,
                        reason,
                        true,
                    )?;
                }
            }
            ExecutionControlCommandKind::ClearPending { reason: _ } => {
                let removed = profile.queue.clear_profile(command.profile_id);
                for item in removed {
                    acknowledge_queued_cancellation(profile, command.profile_id, item.attempt_id)?;
                }
            }
            ExecutionControlCommandKind::ClearHistory => {
                let removed = profile.history.clear_terminal(command.profile_id);
                for record in removed {
                    profile.presentations.remove(&record.attempt_id);
                    profile.plans.remove(&record.attempt_id);
                }
            }
            ExecutionControlCommandKind::Retry {
                attempt_id,
                source,
                replacement_plan,
            } => {
                let retry_attempt_id = assigned_attempt_id
                    .ok_or(ExecutionPresentationError::MissingAssignedAttempt)?;
                if retry_attempt_id == *attempt_id {
                    return Err(ExecutionPresentationError::ReusedAttemptIdentity(
                        retry_attempt_id,
                    ));
                }
                let retry_eligibility = profile
                    .presentations
                    .get(attempt_id)
                    .map(AttemptPresentation::retry_eligibility)
                    .ok_or(HistoryError::UnknownAttempt(*attempt_id))?;
                if let crate::OperationEligibility::Unavailable { reason } = retry_eligibility {
                    return Err(ExecutionPresentationError::InvalidRetrySource(reason));
                }
                let previous = profile
                    .history
                    .record(command.profile_id, *attempt_id)
                    .ok_or(HistoryError::UnknownAttempt(*attempt_id))?;
                if !previous.state.is_terminal() {
                    return Err(HistoryError::RetryNonterminal(*attempt_id).into());
                }
                let plan = match source {
                    RetryPromptSource::OriginalPrompt => profile
                        .plans
                        .get(attempt_id)
                        .cloned()
                        .ok_or(ExecutionPresentationError::MissingPlan(*attempt_id))?,
                    RetryPromptSource::CurrentWorkflow { revision } => {
                        if revision.trim().is_empty() {
                            return Err(ExecutionPresentationError::InvalidRetrySource(
                                "the current workflow revision is empty".to_owned(),
                            ));
                        }
                        replacement_plan.clone().ok_or_else(|| {
                            ExecutionPresentationError::InvalidRetrySource(
                                "current-workflow retry requires a replacement plan".to_owned(),
                            )
                        })?
                    }
                    RetryPromptSource::ProviderResume { operation_id: _ } => {
                        replacement_plan.clone().ok_or_else(|| {
                            ExecutionPresentationError::InvalidRetrySource(
                                "provider-resume retry requires a resolved native plan".to_owned(),
                            )
                        })?
                    }
                };
                profile.queue.enqueue(
                    command.profile_id,
                    plan.clone(),
                    retry_attempt_id,
                    0,
                    true,
                )?;
                let mut record =
                    AttemptRecord::queued(command.profile_id, plan.prompt_id, retry_attempt_id);
                record.retry_of = Some(*attempt_id);
                record.retry_source = Some(source.clone());
                profile.history.insert(record.clone())?;
                profile
                    .presentations
                    .insert(retry_attempt_id, AttemptPresentation::queued(&record));
                profile.plans.insert(retry_attempt_id, plan);
                profile.synchronize_retained_attempts();
            }
            ExecutionControlCommandKind::RemoveHistory { attempt_id } => {
                profile.history.remove(command.profile_id, *attempt_id)?;
                profile.presentations.remove(attempt_id);
                profile.plans.remove(attempt_id);
            }
        }
        profile.next_revision()
    }
}

fn acknowledge_queued_cancellation(
    profile: &mut ProfileExecutionState,
    profile_id: ProfileId,
    attempt_id: AttemptId,
) -> Result<(), ExecutionPresentationError> {
    let record = profile
        .history
        .record_mut(profile_id, attempt_id)
        .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
    record.acknowledge_queued_cancellation(Utc::now())?;
    let record = profile
        .history
        .record(profile_id, attempt_id)
        .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
    let presentation = profile
        .presentations
        .entry(attempt_id)
        .or_insert_with(|| AttemptPresentation::queued(record));
    presentation.acknowledge_cancelled(record);
    Ok(())
}

fn acknowledge_running_cancellation(
    profile: &mut ProfileExecutionState,
    profile_id: ProfileId,
    attempt_id: AttemptId,
    reason: &str,
    interrupt: bool,
) -> Result<(), ExecutionPresentationError> {
    let record = profile
        .history
        .record(profile_id, attempt_id)
        .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
    if !matches!(
        record.state,
        AttemptState::Running | AttemptState::Cancelling
    ) {
        return Err(AttemptTransitionError::Illegal {
            state: record.state,
            event: "acknowledged_cancellation",
        }
        .into());
    }
    let sequence = record.last_sequence.map_or(Ok(0), |sequence| {
        sequence
            .checked_add(1)
            .ok_or(AttemptTransitionError::SequenceExhausted)
    })?;
    let event = AttemptEvent {
        profile_id,
        prompt_id: record.prompt_id,
        attempt_id,
        sequence,
        node_id: None,
        at: Utc::now(),
        kind: AttemptEventKind::CancelRequested {
            reason: reason.to_owned(),
            interrupt,
        },
        data: None,
    };
    profile.history.apply(event.clone())?;
    let record = profile
        .history
        .record(profile_id, attempt_id)
        .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
    profile
        .presentations
        .entry(attempt_id)
        .or_insert_with(|| AttemptPresentation::queued(record))
        .apply_canonical(&event, record);
    Ok(())
}

fn command_requires_attempt_identity(kind: &ExecutionControlCommandKind) -> bool {
    matches!(
        kind,
        ExecutionControlCommandKind::Queue { .. } | ExecutionControlCommandKind::Retry { .. }
    )
}

fn canonical_event_count(record: &AttemptRecord) -> usize {
    usize::try_from(record.canonical_event_count()).unwrap_or(usize::MAX)
}

fn records_share_canonical_projection(current: &AttemptRecord, incoming: &AttemptRecord) -> bool {
    let mut current_projection = reduce_record(current);
    let mut incoming_projection = reduce_record(incoming);
    current_projection.source_projection = None;
    incoming_projection.source_projection = None;
    current_projection == incoming_projection
        && current.events.iter().all(|current_event| {
            incoming
                .events
                .iter()
                .find(|incoming_event| incoming_event.sequence == current_event.sequence)
                .is_none_or(|incoming_event| incoming_event == current_event)
        })
        && incoming.events.iter().all(|incoming_event| {
            current
                .events
                .iter()
                .find(|current_event| current_event.sequence == incoming_event.sequence)
                .is_none_or(|current_event| current_event == incoming_event)
        })
}

pub(crate) fn validate_record_integrity(
    record: &AttemptRecord,
) -> Result<(), ExecutionPresentationError> {
    if record.schema_version != ATTEMPT_HISTORY_SCHEMA_VERSION {
        return Err(HistoryError::UnsupportedSchema(record.schema_version).into());
    }
    if record
        .source_projection
        .as_ref()
        .is_some_and(|projection| !projection.is_valid())
    {
        return Err(ExecutionPresentationError::InvalidSourceProjection(
            record.attempt_id,
        ));
    }
    let canonical_event_count = record.canonical_event_count();
    let expected_event_count = record
        .last_sequence
        .map_or(0, |sequence| sequence.saturating_add(1));
    if canonical_event_count != expected_event_count
        || canonical_event_count < record.events.len() as u64
    {
        return Err(ExecutionPresentationError::InconsistentRecord(
            record.attempt_id,
        ));
    }
    let mut previous_sequence = None;
    for event in &record.events {
        if previous_sequence.is_some_and(|sequence| event.sequence <= sequence) {
            return Err(ExecutionPresentationError::InconsistentRecord(
                record.attempt_id,
            ));
        }
        previous_sequence = Some(event.sequence);
    }
    for (output_id, availability) in &record.output_availability_overrides {
        let output = record
            .events
            .iter()
            .rev()
            .find_map(|event| match &event.kind {
                AttemptEventKind::OutputAvailable { output } if output.output_id == *output_id => {
                    Some(output)
                }
                _ => None,
            })
            .ok_or(ExecutionPresentationError::InconsistentRecord(
                record.attempt_id,
            ))?;
        let valid = match availability {
            crate::ExecutionOutputAvailability::Ready {
                reference,
                byte_length,
            } => {
                output.recovery_eligibility().is_allowed()
                    && !reference.trim().is_empty()
                    && *byte_length > 0
            }
            crate::ExecutionOutputAvailability::Removed { reason } => {
                output.removal_eligibility().is_allowed() && !reason.trim().is_empty()
            }
            crate::ExecutionOutputAvailability::Missing { .. }
            | crate::ExecutionOutputAvailability::Expired { .. }
            | crate::ExecutionOutputAvailability::ExternallyDeleted { .. }
            | crate::ExecutionOutputAvailability::Forbidden { .. }
            | crate::ExecutionOutputAvailability::Unsupported { .. }
            | crate::ExecutionOutputAvailability::Corrupt { .. } => false,
        };
        if !valid {
            return Err(ExecutionPresentationError::InconsistentRecord(
                record.attempt_id,
            ));
        }
    }
    if record.events.is_empty() && record.last_sequence.is_none() {
        let valid_without_events = match record.state {
            AttemptState::Queued => record.finished_at.is_none(),
            AttemptState::Cancelled => record.finished_at.is_some(),
            _ => false,
        };
        if valid_without_events {
            return Ok(());
        }
        return Err(ExecutionPresentationError::InconsistentRecord(
            record.attempt_id,
        ));
    }
    let mut reduced = AttemptRecord::queued(record.profile_id, record.prompt_id, record.attempt_id);
    reduced.retry_of = record.retry_of;
    reduced.retry_source = record.retry_source.clone();
    reduced.source_projection = record.source_projection.clone();
    reduced.created_at = record.created_at;
    for (sequence, event) in record.events.iter().enumerate() {
        let mut event = event.clone();
        event.sequence = sequence as u64;
        reduced.apply(event)?;
    }
    if reduced.state != record.state || reduced.finished_at != record.finished_at {
        return Err(ExecutionPresentationError::InconsistentRecord(
            record.attempt_id,
        ));
    }
    Ok(())
}

fn reduce_record(record: &AttemptRecord) -> AttemptPresentation {
    let mut presentation = AttemptPresentation::queued(record);
    for event in &record.events {
        presentation.apply_canonical(event, record);
    }
    presentation.apply_output_availability_overrides(record);
    presentation.state = record.state;
    presentation.last_sequence = record.last_sequence;
    presentation.finished_at = record.finished_at;
    presentation.canonical_event_count = canonical_event_count(record);
    if record.state.is_terminal() {
        presentation.clear_transient_projection();
    }
    presentation
}

#[derive(Debug, Error)]
pub enum ExecutionPresentationError {
    #[error("execution profile {0:?} is not initialized")]
    UnknownProfile(ProfileId),
    #[error("execution request {0:?} has already been submitted")]
    DuplicateRequest(RequestId),
    #[error("execution request {0:?} is not pending")]
    UnknownRequest(RequestId),
    #[error("execution has reached its {maximum}-request pending limit")]
    PendingRequestCapacity { maximum: usize },
    #[error("execution command expected revision {expected}, found {actual}")]
    RevisionMismatch { expected: u64, actual: u64 },
    #[error("execution revision is exhausted")]
    RevisionExhausted,
    #[error("execution attempt identity allocation is exhausted")]
    AttemptIdentityExhausted,
    #[error("execution acknowledgement profile expected {expected:?}, received {actual:?}")]
    AckProfileMismatch {
        expected: ProfileId,
        actual: ProfileId,
    },
    #[error("accepted queue or retry command did not assign an attempt identity")]
    MissingAssignedAttempt,
    #[error("attempt {0:?} has no live canonical cancellation token")]
    MissingCancellationToken(AttemptId),
    #[error("retry reused attempt identity {0:?}")]
    ReusedAttemptIdentity(AttemptId),
    #[error("attempt {attempt_id:?} belongs to profile {actual:?}, not {expected:?}")]
    CrossProfileAttempt {
        expected: ProfileId,
        actual: ProfileId,
        attempt_id: AttemptId,
    },
    #[error("reconciliation revision {received} is not newer than {current}")]
    StaleReconciliation { current: u64, received: u64 },
    #[error("attempt {0:?} has no retained native execution plan")]
    MissingPlan(AttemptId),
    #[error("attempt {0:?} has more than one retained plan")]
    DuplicatePlan(AttemptId),
    #[error("attempt {0:?} reconciliation conflicts with retained attempt identity")]
    ReconciliationIdentityConflict(AttemptId),
    #[error("attempt {attempt_id:?} plan does not match its prompt identity")]
    PlanPromptMismatch { attempt_id: AttemptId },
    #[error("attempt {attempt_id:?} prompt expected {expected:?}, received {actual:?}")]
    AttemptPromptMismatch {
        attempt_id: AttemptId,
        expected: PromptId,
        actual: PromptId,
    },
    #[error("queued attempt {0:?} does not have a matching queued history record")]
    InvalidQueuedRecord(AttemptId),
    #[error("attempt {0:?} persisted state does not match its canonical event reduction")]
    InconsistentRecord(AttemptId),
    #[error("attempt {0:?} has an invalid provider or unknown source projection")]
    InvalidSourceProjection(AttemptId),
    #[error("retry source is invalid: {0}")]
    InvalidRetrySource(String),
    #[error("execution presentation state is unavailable: {0}")]
    StateUnavailable(String),
    #[error("execution persistence is invalid: {0}")]
    Persistence(String),
    #[error("execution output operation commit failed: {0}")]
    OutputOperationCommit(String),
    #[error("execution output operation failed ({primary}) and rollback failed ({rollback})")]
    OutputOperationRollback { primary: String, rollback: String },
    #[error(
        "execution output operation failed ({primary}) and durable compensation requires recovery ({compensation})"
    )]
    OutputOperationRecoveryRequired {
        primary: String,
        compensation: String,
    },
    #[error(transparent)]
    Queue(#[from] ExecutionQueueError),
    #[error(transparent)]
    History(#[from] HistoryError),
    #[error(transparent)]
    Transition(#[from] AttemptTransitionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompiledNode, NativeCachePolicy, NativeEffectClass, NativeNodeDescriptor,
        NativeOutputDescriptor, NativeValueType,
    };
    use sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
    use std::{
        collections::{BTreeMap, BTreeSet},
        sync::atomic::{AtomicBool, Ordering},
    };

    struct RecordingPersistence {
        fail: bool,
        steps: Arc<Mutex<Vec<&'static str>>>,
    }

    impl ExecutionAttemptPersistence for RecordingPersistence {
        fn replace_execution_state(
            &self,
            _profile: PersistedExecutionProfile,
            _attempts: Vec<PersistedExecutionAttempt>,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<()>> {
            let fail = self.fail;
            let steps = self.steps.clone();
            Box::pin(async move {
                steps
                    .lock()
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?
                    .push("persist");
                anyhow::ensure!(!fail, "injected execution persistence failure");
                Ok(())
            })
        }

        fn load_execution_state(
            &self,
            _profile_id: ProfileId,
        ) -> anyhow::Result<(
            Option<PersistedExecutionProfile>,
            Vec<PersistedExecutionAttempt>,
        )> {
            Ok((None, Vec::new()))
        }
    }

    struct RecordingActivation {
        steps: Arc<Mutex<Vec<&'static str>>>,
    }

    impl PreparedExecutionActivation for RecordingActivation {
        fn commit(self: Box<Self>) {
            match self.steps.lock() {
                Ok(mut steps) => steps.push("commit"),
                Err(error) => eprintln!("recording activation state is unavailable: {error}"),
            }
        }
    }

    struct RecordingPreparedController {
        steps: Arc<Mutex<Vec<&'static str>>>,
    }

    impl ExecutionController for RecordingPreparedController {
        fn prepare<'a>(
            &'a self,
            _command: &ExecutionControlCommand,
            _assigned_attempt_id: Option<AttemptId>,
        ) -> Result<Box<dyn PreparedExecutionActivation + 'a>, ExecutionFailure> {
            self.steps
                .lock()
                .map_err(|error| {
                    ExecutionFailure::new("recording_state_unavailable", error.to_string())
                })?
                .push("prepare");
            Ok(Box::new(RecordingActivation {
                steps: self.steps.clone(),
            }))
        }

        fn accept(
            &self,
            _command: &ExecutionControlCommand,
            _assigned_attempt_id: Option<AttemptId>,
        ) -> Result<(), ExecutionFailure> {
            Err(ExecutionFailure::new(
                "unexpected_direct_accept",
                "durable dispatch bypassed prepared activation",
            ))
        }
    }

    struct AcceptingExecutionController;

    impl ExecutionController for AcceptingExecutionController {
        fn accept(
            &self,
            _command: &ExecutionControlCommand,
            _assigned_attempt_id: Option<AttemptId>,
        ) -> Result<(), ExecutionFailure> {
            Ok(())
        }
    }

    fn profile(value: u128) -> ProfileId {
        ProfileId(Uuid::from_u128(value))
    }

    fn plan(value: u128) -> CompiledPlan {
        let prompt_id = PromptId(Uuid::from_u128(value));
        let node_id = NodeId::from("output");
        CompiledPlan {
            prompt_id,
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([(
                node_id.clone(),
                CompiledNode {
                    id: node_id.clone(),
                    class_type: "Output".to_owned(),
                    descriptor: NativeNodeDescriptor {
                        schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                        class_type: "Output".to_owned(),
                        implementation_version: "1".to_owned(),
                        source_schema: Some(sim_nodes::NativeDescriptorSchemaMetadata::synthetic(
                            std::iter::empty(),
                            std::iter::empty(),
                            ["value".to_owned()],
                        )),
                        inputs: Vec::new(),
                        dynamic_inputs: Vec::new(),
                        outputs: vec![NativeOutputDescriptor {
                            name: "value".to_owned(),
                            produced_type: NativeValueType::Any,
                            is_list: false,
                        }],
                        output_node: true,
                        effect: NativeEffectClass::Pure,
                        cache: NativeCachePolicy::Never,
                    },
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            )]),
            topological_order: vec![node_id.clone()],
            static_required_nodes: BTreeSet::from([node_id.clone()]),
            output_nodes: vec![node_id],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        }
    }

    fn command(
        request: u128,
        profile_id: ProfileId,
        kind: ExecutionControlCommandKind,
    ) -> ExecutionControlCommand {
        ExecutionControlCommand {
            request_id: RequestId(Uuid::from_u128(request)),
            profile_id,
            expected_revision: None,
            kind,
        }
    }

    #[test]
    fn durable_owner_persists_before_committing_prepared_activation()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(0x701);
        let mut service = ExecutionPresentationService::new_with_first_attempt_id(
            16,
            AttemptId(Uuid::from_u128(0x702)),
        )?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let steps = Arc::new(Mutex::new(Vec::new()));
        let owner = ExecutionPresentationOwner::persistent(
            service,
            Arc::new(RecordingPersistence {
                fail: false,
                steps: steps.clone(),
            }),
        );
        let controller = RecordingPreparedController {
            steps: steps.clone(),
        };

        let acknowledgement = smol::block_on(owner.dispatch_durable(
            command(
                0x703,
                profile_id,
                ExecutionControlCommandKind::Queue {
                    plan: plan(0x704),
                    priority: 0,
                    front: false,
                },
            ),
            &controller,
        ))?;

        assert!(matches!(
            acknowledgement.outcome,
            ExecutionCommandOutcome::Accepted { .. }
        ));
        assert_eq!(
            *steps
                .lock()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            vec!["prepare", "persist", "commit"]
        );
        assert_eq!(owner.snapshot(profile_id)?.queue.len(), 1);
        Ok(())
    }

    #[test]
    fn durable_owner_aborts_activation_and_state_when_persistence_fails()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(0x711);
        let mut service = ExecutionPresentationService::new_with_first_attempt_id(
            16,
            AttemptId(Uuid::from_u128(0x712)),
        )?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let original = service.snapshot(profile_id)?;
        let steps = Arc::new(Mutex::new(Vec::new()));
        let owner = ExecutionPresentationOwner::persistent(
            service,
            Arc::new(RecordingPersistence {
                fail: true,
                steps: steps.clone(),
            }),
        );
        let controller = RecordingPreparedController {
            steps: steps.clone(),
        };

        let result = smol::block_on(owner.dispatch_durable(
            command(
                0x713,
                profile_id,
                ExecutionControlCommandKind::Queue {
                    plan: plan(0x714),
                    priority: 0,
                    front: false,
                },
            ),
            &controller,
        ));

        assert!(matches!(
            result,
            Err(ExecutionPresentationError::Persistence(_))
        ));
        assert_eq!(
            *steps
                .lock()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            vec!["prepare", "persist"]
        );
        assert_eq!(owner.snapshot(profile_id)?, original);
        Ok(())
    }

    #[test]
    fn durable_owner_keeps_status_ack_and_reconciliation_atomic_on_persistence_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(0x715);
        let mut service = ExecutionPresentationService::new_with_first_attempt_id(
            16,
            AttemptId(Uuid::from_u128(0x716)),
        )?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let pending_command = command(0x717, profile_id, ExecutionControlCommandKind::ClearHistory);
        service.submit(pending_command.clone())?;
        let original = service.snapshot(profile_id)?;
        let owner = ExecutionPresentationOwner::persistent(
            service,
            Arc::new(RecordingPersistence {
                fail: true,
                steps: Arc::new(Mutex::new(Vec::new())),
            }),
        );

        assert!(matches!(
            smol::block_on(owner.set_snapshot_status_durable(
                profile_id,
                ExecutionDataSource::Recovery,
                ExecutionSnapshotStatus::Ready,
            )),
            Err(ExecutionPresentationError::Persistence(_))
        ));
        assert_eq!(owner.snapshot(profile_id)?, original);

        assert!(matches!(
            smol::block_on(owner.apply_ack_durable(ExecutionCommandAck {
                request_id: pending_command.request_id,
                profile_id,
                outcome: ExecutionCommandOutcome::Accepted {
                    assigned_attempt_id: None,
                },
            })),
            Err(ExecutionPresentationError::Persistence(_))
        ));
        assert_eq!(owner.snapshot(profile_id)?, original);

        assert!(matches!(
            smol::block_on(owner.reconcile_durable(ExecutionReconciliation {
                profile_id,
                source_revision: 1,
                source: ExecutionDataSource::Recovery,
                status: ExecutionSnapshotStatus::Ready,
                queue: Vec::new(),
                records: Vec::new(),
                plans: Vec::new(),
                acknowledged_requests: Vec::new(),
            })),
            Err(ExecutionPresentationError::Persistence(_))
        ));
        assert_eq!(owner.snapshot(profile_id)?, original);
        Ok(())
    }

    #[test]
    fn durable_owner_restores_queue_cursor_and_command_receipts()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let profile_id = profile(0x721);
            let request_id = RequestId(Uuid::from_u128(0x722));
            let database =
                crate::SimRuntimeDb::open_test_db("execution_presentation_durable_owner_restore")
                    .await;
            let mut service = ExecutionPresentationService::new_with_first_attempt_id(
                16,
                AttemptId(Uuid::from_u128(0x723)),
            )?;
            service.initialize_profile(
                profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )?;
            let owner = ExecutionPresentationOwner::persistent(service, Arc::new(database.clone()));
            let acknowledgement = owner
                .dispatch_durable(
                    ExecutionControlCommand {
                        request_id,
                        profile_id,
                        expected_revision: None,
                        kind: ExecutionControlCommandKind::Queue {
                            plan: plan(0x724),
                            priority: 0,
                            front: false,
                        },
                    },
                    &AcceptingExecutionController,
                )
                .await?;
            let assigned_attempt_id = match &acknowledgement.outcome {
                ExecutionCommandOutcome::Accepted {
                    assigned_attempt_id: Some(attempt_id),
                } => *attempt_id,
                outcome => return Err(format!("unexpected acknowledgement: {outcome:?}").into()),
            };

            let restored = ExecutionPresentationOwner::persistent(
                ExecutionPresentationService::new_with_first_attempt_id(
                    16,
                    AttemptId(Uuid::from_u128(0x700)),
                )?,
                Arc::new(database),
            );
            assert!(restored.restore_profile(profile_id).await?);
            let snapshot = restored.snapshot(profile_id)?;
            assert_eq!(snapshot.queue[0].attempt_id, assigned_attempt_id);
            assert_eq!(snapshot.recent_command_results[0].request_id, request_id);
            drop(snapshot);
            assert_eq!(
                restored.command_receipt_state(profile_id, request_id)?,
                ExecutionCommandReceiptState::Completed(acknowledgement.clone())
            );
            assert_eq!(
                restored.command_receipt_state(profile_id, RequestId(Uuid::from_u128(0x725)),)?,
                ExecutionCommandReceiptState::NotApplied
            );

            let duplicate = restored
                .dispatch_durable(
                    ExecutionControlCommand {
                        request_id,
                        profile_id,
                        expected_revision: None,
                        kind: ExecutionControlCommandKind::ClearHistory,
                    },
                    &AcceptingExecutionController,
                )
                .await;
            assert!(matches!(
                duplicate,
                Err(ExecutionPresentationError::DuplicateRequest(id)) if id == request_id
            ));
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    fn event(
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        sequence: u64,
        node_id: Option<NodeId>,
        kind: AttemptEventKind,
    ) -> AttemptEvent {
        AttemptEvent {
            profile_id,
            prompt_id,
            attempt_id,
            sequence,
            node_id,
            at: DateTime::<Utc>::from_timestamp(1_700_000_000 + sequence as i64, 0)
                .unwrap_or(DateTime::<Utc>::UNIX_EPOCH),
            kind,
            data: None,
        }
    }

    #[test]
    fn started_event_projects_effective_backend_without_owning_capability_validation()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = ProfileId(Uuid::from_u128(0x730));
        let prompt_id = PromptId(Uuid::from_u128(0x731));
        let attempt_id = AttemptId(Uuid::from_u128(0x732));
        let effective_backend = EffectiveNativeBackendState {
            device: DeviceId::new(sim_types::DeviceKind::Mlu, 3),
            device_name: "Cambricon MLU".to_owned(),
            architecture: Some("Neuware 1.20".to_owned()),
            total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
            allocation_limit_bytes: Some(20 * 1024 * 1024 * 1024),
            memory_limit_bytes: 16 * 1024 * 1024 * 1024,
            memory_in_use_bytes: 0,
            memory_policy: crate::MemoryPolicy::Balanced,
            supported_operation_rows: 7,
            deterministic_operation_rows: 6,
        };
        let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
        let mut presentation = AttemptPresentation::queued(&record);
        let mut started = event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        );
        started.data = Some(serde_json::json!({
            "effective_native_backend": effective_backend
        }));
        record.apply(started.clone())?;
        presentation.apply_canonical(&started, &record);

        assert_eq!(presentation.effective_backend, Some(effective_backend));
        let restored: AttemptPresentation =
            serde_json::from_slice(&serde_json::to_vec(&presentation)?)?;
        assert_eq!(restored, presentation);
        Ok(())
    }

    #[test]
    fn started_event_preserves_directml_physical_allocation_and_effective_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = ProfileId(Uuid::from_u128(0x733));
        let prompt_id = PromptId(Uuid::from_u128(0x734));
        let attempt_id = AttemptId(Uuid::from_u128(0x735));
        let effective_backend = EffectiveNativeBackendState {
            device: DeviceId::new(sim_types::DeviceKind::DirectMl, 0),
            device_name: "DirectML certified adapter".to_owned(),
            architecture: Some("DXGI adapter LUID 0x1122334455667788".to_owned()),
            total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
            allocation_limit_bytes: Some(18 * 1024 * 1024 * 1024),
            memory_limit_bytes: 12 * 1024 * 1024 * 1024,
            memory_in_use_bytes: 0,
            memory_policy: crate::MemoryPolicy::Balanced,
            supported_operation_rows: 7,
            deterministic_operation_rows: 7,
        };
        let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
        let mut presentation = AttemptPresentation::queued(&record);
        let mut started = event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        );
        started.data = Some(serde_json::json!({
            "effective_native_backend": effective_backend
        }));
        record.apply(started.clone())?;
        presentation.apply_canonical(&started, &record);

        let projected = presentation
            .effective_backend
            .as_ref()
            .ok_or("DirectML effective backend projection is absent")?;
        assert_eq!(projected.device.kind(), sim_types::DeviceKind::DirectMl);
        assert_eq!(projected.total_memory_bytes, Some(24 * 1024 * 1024 * 1024));
        assert_eq!(
            projected.allocation_limit_bytes,
            Some(18 * 1024 * 1024 * 1024)
        );
        assert_eq!(projected.memory_limit_bytes, 12 * 1024 * 1024 * 1024);
        let restored: AttemptPresentation =
            serde_json::from_slice(&serde_json::to_vec(&presentation)?)?;
        assert_eq!(restored, presentation);
        Ok(())
    }

    #[test]
    fn started_event_preserves_xpu_physical_allocation_and_effective_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = ProfileId(Uuid::from_u128(0x736));
        let prompt_id = PromptId(Uuid::from_u128(0x737));
        let attempt_id = AttemptId(Uuid::from_u128(0x738));
        let effective_backend = EffectiveNativeBackendState {
            device: DeviceId::new(sim_types::DeviceKind::Xpu, 3),
            device_name: "Intel XPU certified device".to_owned(),
            architecture: Some("Intel 0x8086:0x56a0; oneDNN 3.5.0".to_owned()),
            total_memory_bytes: Some(16 * 1024 * 1024 * 1024),
            allocation_limit_bytes: Some(6 * 1024 * 1024 * 1024),
            memory_limit_bytes: 6 * 1024 * 1024 * 1024,
            memory_in_use_bytes: 0,
            memory_policy: crate::MemoryPolicy::Balanced,
            supported_operation_rows: 7,
            deterministic_operation_rows: 7,
        };
        let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
        let mut presentation = AttemptPresentation::queued(&record);
        let mut started = event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        );
        started.data = Some(serde_json::json!({
            "effective_native_backend": effective_backend
        }));
        record.apply(started.clone())?;
        presentation.apply_canonical(&started, &record);

        let projected = presentation
            .effective_backend
            .as_ref()
            .ok_or("XPU effective backend projection is absent")?;
        assert_eq!(projected.device.kind(), sim_types::DeviceKind::Xpu);
        assert_eq!(projected.total_memory_bytes, Some(16 * 1024 * 1024 * 1024));
        assert_eq!(
            projected.allocation_limit_bytes,
            Some(6 * 1024 * 1024 * 1024)
        );
        assert_eq!(projected.memory_limit_bytes, 6 * 1024 * 1024 * 1024);
        let restored: AttemptPresentation =
            serde_json::from_slice(&serde_json::to_vec(&presentation)?)?;
        assert_eq!(restored, presentation);
        Ok(())
    }

    #[test]
    fn started_event_preserves_cuda_physical_allocation_and_effective_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = ProfileId(Uuid::from_u128(0x739));
        let prompt_id = PromptId(Uuid::from_u128(0x73a));
        let attempt_id = AttemptId(Uuid::from_u128(0x73b));
        let effective_backend = EffectiveNativeBackendState {
            device: DeviceId::new(sim_types::DeviceKind::Cuda, 3),
            device_name: "NVIDIA CUDA certified device".to_owned(),
            architecture: Some("CUDA driver 12080; NVRTC 12.8".to_owned()),
            total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
            allocation_limit_bytes: Some(18 * 1024 * 1024 * 1024),
            memory_limit_bytes: 12 * 1024 * 1024 * 1024,
            memory_in_use_bytes: 0,
            memory_policy: crate::MemoryPolicy::Balanced,
            supported_operation_rows: 7,
            deterministic_operation_rows: 7,
        };
        let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
        let mut presentation = AttemptPresentation::queued(&record);
        let mut started = event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        );
        started.data = Some(serde_json::json!({
            "effective_native_backend": effective_backend
        }));
        record.apply(started.clone())?;
        presentation.apply_canonical(&started, &record);

        let projected = presentation
            .effective_backend
            .as_ref()
            .ok_or("CUDA effective backend projection is absent")?;
        assert_eq!(projected.device.kind(), sim_types::DeviceKind::Cuda);
        assert_eq!(projected.total_memory_bytes, Some(24 * 1024 * 1024 * 1024));
        assert_eq!(
            projected.allocation_limit_bytes,
            Some(18 * 1024 * 1024 * 1024)
        );
        assert_eq!(projected.memory_limit_bytes, 12 * 1024 * 1024 * 1024);
        let restored: AttemptPresentation =
            serde_json::from_slice(&serde_json::to_vec(&presentation)?)?;
        assert_eq!(restored, presentation);
        Ok(())
    }

    fn queue_attempt(
        service: &mut ExecutionPresentationService,
        controller: &AcceptingExecutionController,
        profile_id: ProfileId,
        request: u128,
        prompt: u128,
    ) -> Result<AttemptId, Box<dyn std::error::Error>> {
        let ack = service.dispatch(
            command(
                request,
                profile_id,
                ExecutionControlCommandKind::Queue {
                    plan: plan(prompt),
                    priority: 0,
                    front: false,
                },
            ),
            controller,
        )?;
        match ack.outcome {
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            } => Ok(attempt_id),
            outcome => Err(format!("queue was not accepted with an identity: {outcome:?}").into()),
        }
    }

    #[test]
    fn disconnected_controller_rejects_runtime_commands_without_mutating_execution()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let attempt_id = AttemptId(Uuid::from_u128(30));
        let controller = DisconnectedExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let commands = [
            (
                ExecutionControlCommandKind::Queue {
                    plan: plan(20),
                    priority: 0,
                    front: false,
                },
                ExecutionFailureOrigin::Transport,
            ),
            (
                ExecutionControlCommandKind::Reorder {
                    attempt_id,
                    position: 0,
                },
                ExecutionFailureOrigin::Validation,
            ),
            (
                ExecutionControlCommandKind::Cancel {
                    attempt_id,
                    reason: "test cancellation".to_owned(),
                },
                ExecutionFailureOrigin::Validation,
            ),
            (
                ExecutionControlCommandKind::Interrupt {
                    attempt_id,
                    reason: "test interruption".to_owned(),
                },
                ExecutionFailureOrigin::Validation,
            ),
            (
                ExecutionControlCommandKind::ClearPending {
                    reason: "test clear".to_owned(),
                },
                ExecutionFailureOrigin::Transport,
            ),
            (
                ExecutionControlCommandKind::Retry {
                    attempt_id,
                    source: RetryPromptSource::OriginalPrompt,
                    replacement_plan: None,
                },
                ExecutionFailureOrigin::Validation,
            ),
        ];

        for (index, (kind, expected_origin)) in commands.into_iter().enumerate() {
            let acknowledgement =
                service.dispatch(command(100 + index as u128, profile_id, kind), &controller)?;
            let ExecutionCommandOutcome::Rejected { failure } = acknowledgement.outcome else {
                return Err("disconnected command was accepted".into());
            };
            assert_eq!(failure.origin, expected_origin);
        }

        let snapshot = service.snapshot(profile_id)?;
        assert!(snapshot.queue.is_empty());
        assert!(snapshot.attempts.is_empty());
        assert!(snapshot.pending_commands.is_empty());
        assert_eq!(snapshot.recent_command_results.len(), 6);
        Ok(())
    }

    #[test]
    fn disconnected_controller_preserves_local_history_commands()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let controller = DisconnectedExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Partial {
                failure: ExecutionFailure::new("runtime_not_connected", "runtime unavailable")
                    .with_origin(ExecutionFailureOrigin::Transport),
            },
        )?;

        let acknowledgement = service.dispatch(
            command(1, profile_id, ExecutionControlCommandKind::ClearHistory),
            &controller,
        )?;
        assert!(matches!(
            acknowledgement.outcome,
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: None
            }
        ));

        let acknowledgement = service.dispatch(
            command(
                2,
                profile_id,
                ExecutionControlCommandKind::RemoveHistory {
                    attempt_id: AttemptId(Uuid::from_u128(30)),
                },
            ),
            &controller,
        )?;
        assert!(matches!(
            acknowledgement.outcome,
            ExecutionCommandOutcome::Rejected {
                failure: ExecutionFailure {
                    origin: ExecutionFailureOrigin::Validation,
                    ..
                }
            }
        ));
        Ok(())
    }

    #[test]
    fn commands_are_pending_until_ack_and_reject_cross_profile_ack()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let other_profile = profile(2);
        let mut service = ExecutionPresentationService::new(16)?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let queue = command(
            10,
            profile_id,
            ExecutionControlCommandKind::Queue {
                plan: plan(20),
                priority: 0,
                front: false,
            },
        );
        service.submit(queue.clone())?;
        let pending = service.snapshot(profile_id)?;
        assert!(pending.queue.is_empty());
        assert_eq!(pending.pending_commands.len(), 1);
        assert!(matches!(
            service.apply_ack(ExecutionCommandAck {
                request_id: queue.request_id,
                profile_id: other_profile,
                outcome: ExecutionCommandOutcome::Accepted {
                    assigned_attempt_id: Some(AttemptId(Uuid::from_u128(30))),
                },
            }),
            Err(ExecutionPresentationError::AckProfileMismatch { .. })
        ));
        assert_eq!(service.snapshot(profile_id)?.pending_commands.len(), 1);
        service.apply_ack(ExecutionCommandAck {
            request_id: queue.request_id,
            profile_id,
            outcome: ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(AttemptId(Uuid::from_u128(30))),
            },
        })?;
        let applied = service.snapshot(profile_id)?;
        assert_eq!(applied.queue.len(), 1);
        assert!(applied.pending_commands.is_empty());
        assert_eq!(
            applied.attempts.first().map(|attempt| attempt.state),
            Some(AttemptState::Queued)
        );
        Ok(())
    }

    #[test]
    fn canonical_event_lookup_requires_exact_identity_and_payload()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let other_profile = profile(2);
        let prompt_id = PromptId(Uuid::from_u128(20));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 20)?;
        service.initialize_profile(
            other_profile,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let applied = service.apply_actuator_event(
            profile_id,
            prompt_id,
            attempt_id,
            Some(NodeId::from("output")),
            AttemptEventKind::Started,
            Some(serde_json::json!({"source": "native-worker"})),
            DateTime::<Utc>::from_timestamp(1_700_000_000, 0).ok_or("invalid fixture timestamp")?,
        )?;
        assert!(service.contains_canonical_event(&applied)?);

        let mut different_payload = applied.clone();
        different_payload.data = Some(serde_json::json!({"source": "protocol-bridge"}));
        assert!(!service.contains_canonical_event(&different_payload)?);

        let mut different_prompt = applied.clone();
        different_prompt.prompt_id = PromptId(Uuid::from_u128(21));
        assert!(matches!(
            service.contains_canonical_event(&different_prompt),
            Err(ExecutionPresentationError::AttemptPromptMismatch { attempt_id: actual, .. })
                if actual == attempt_id
        ));

        let mut different_profile = applied;
        different_profile.profile_id = other_profile;
        assert!(matches!(
            service.contains_canonical_event(&different_profile),
            Err(ExecutionPresentationError::CrossProfileAttempt {
                expected,
                actual,
                attempt_id: actual_attempt,
            }) if expected == profile_id
                && actual == other_profile
                && actual_attempt == attempt_id
        ));
        Ok(())
    }

    #[test]
    fn actuator_event_batches_prevalidate_and_apply_atomically()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(20));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 20)?;
        let at =
            DateTime::<Utc>::from_timestamp(1_700_000_000, 0).ok_or("invalid fixture timestamp")?;
        let valid = vec![
            ExecutionActuatorEventInput {
                node_id: None,
                kind: AttemptEventKind::Started,
                data: None,
                at,
            },
            ExecutionActuatorEventInput {
                node_id: None,
                kind: AttemptEventKind::Succeeded,
                data: None,
                at,
            },
        ];
        let before = service.snapshot(profile_id)?;
        let validated =
            service.validate_actuator_event_batch(profile_id, prompt_id, attempt_id, &valid)?;
        assert_eq!(
            validated
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(service.snapshot(profile_id)?, before);

        let mut invalid = valid.clone();
        invalid.push(ExecutionActuatorEventInput {
            node_id: None,
            kind: AttemptEventKind::Progress {
                completed: 1,
                total: 1,
            },
            data: None,
            at,
        });
        assert!(
            service
                .apply_actuator_event_batch(profile_id, prompt_id, attempt_id, &invalid)
                .is_err()
        );
        assert_eq!(service.snapshot(profile_id)?, before);

        let applied =
            service.apply_actuator_event_batch(profile_id, prompt_id, attempt_id, &valid)?;
        assert_eq!(applied, validated);
        let snapshot = service.snapshot(profile_id)?;
        assert_eq!(
            snapshot
                .attempts
                .first()
                .ok_or("missing applied attempt")?
                .state,
            AttemptState::Succeeded
        );
        Ok(())
    }

    #[test]
    fn unapplied_event_validation_matches_canonical_rejections_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let other_profile = profile(2);
        let prompt_id = PromptId(Uuid::from_u128(20));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 20)?;
        service.initialize_profile(
            other_profile,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;

        let started = event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        );
        let before_valid = service.snapshot(profile_id)?;
        service.validate_unapplied_event(&started)?;
        assert_eq!(service.snapshot(profile_id)?, before_valid);
        assert!(!service.contains_canonical_event(&started)?);

        let out_of_sequence = event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            None,
            AttemptEventKind::Started,
        );
        assert!(matches!(
            service.validate_unapplied_event(&out_of_sequence),
            Err(ExecutionPresentationError::History(
                HistoryError::Transition(AttemptTransitionError::Sequence {
                    expected: 0,
                    actual: 1,
                })
            ))
        ));
        assert_eq!(service.snapshot(profile_id)?, before_valid);

        service.apply_event(started)?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            None,
            AttemptEventKind::Succeeded,
        ))?;
        let before_rejections = service.snapshot(profile_id)?;
        let late_terminal = event(
            profile_id,
            prompt_id,
            attempt_id,
            2,
            None,
            AttemptEventKind::Succeeded,
        );
        assert!(matches!(
            service.validate_unapplied_event(&late_terminal),
            Err(ExecutionPresentationError::History(
                HistoryError::Transition(AttemptTransitionError::Terminal(AttemptState::Succeeded))
            ))
        ));
        assert_eq!(service.snapshot(profile_id)?, before_rejections);

        let mut cross_profile = late_terminal;
        cross_profile.profile_id = other_profile;
        let other_before = service.snapshot(other_profile)?;
        assert!(matches!(
            service.validate_unapplied_event(&cross_profile),
            Err(ExecutionPresentationError::CrossProfileAttempt {
                expected,
                actual,
                attempt_id: actual_attempt,
            }) if expected == profile_id
                && actual == other_profile
                && actual_attempt == attempt_id
        ));
        assert_eq!(service.snapshot(profile_id)?, before_rejections);
        assert_eq!(service.snapshot(other_profile)?, other_before);
        Ok(())
    }

    #[test]
    fn canonical_reducer_retains_ten_thousand_updates_before_presentation_coalescing()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(20));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 20)?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        for completed in 1..=10_000_u64 {
            service.apply_event(event(
                profile_id,
                prompt_id,
                attempt_id,
                completed,
                Some(NodeId::from("output")),
                AttemptEventKind::Progress {
                    completed,
                    total: 10_000,
                },
            ))?;
        }
        let snapshot = service.snapshot(profile_id)?;
        let attempt = snapshot
            .attempts
            .first()
            .ok_or("missing attempt presentation")?;
        assert_eq!(attempt.canonical_event_count, 10_001);
        assert_eq!(attempt.last_sequence, Some(10_000));
        assert_eq!(
            attempt.progress,
            Some(NodeProgress {
                node_id: Some(NodeId::from("output")),
                completed: 10_000,
                total: 10_000,
            })
        );
        assert!(matches!(
            service.apply_event(event(
                profile_id,
                prompt_id,
                attempt_id,
                9_999,
                Some(NodeId::from("output")),
                AttemptEventKind::Progress {
                    completed: 9_999,
                    total: 10_000,
                },
            )),
            Err(ExecutionPresentationError::History(
                HistoryError::Transition(AttemptTransitionError::Sequence { .. })
            ))
        ));
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            10_001,
            None,
            AttemptEventKind::Succeeded,
        ))?;
        assert!(matches!(
            service.apply_event(event(
                profile_id,
                prompt_id,
                attempt_id,
                10_002,
                None,
                AttemptEventKind::Succeeded,
            )),
            Err(ExecutionPresentationError::History(
                HistoryError::Transition(AttemptTransitionError::Terminal(AttemptState::Succeeded))
            ))
        ));
        Ok(())
    }

    #[test]
    fn interleaved_profiles_are_isolated_and_cross_profile_events_are_rejected()
    -> Result<(), Box<dyn std::error::Error>> {
        let first_profile = profile(1);
        let second_profile = profile(2);
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let first = queue_attempt(&mut service, &controller, first_profile, 1, 21)?;
        let second = queue_attempt(&mut service, &controller, second_profile, 2, 22)?;
        service.apply_event(event(
            first_profile,
            PromptId(Uuid::from_u128(21)),
            first,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        service.apply_event(event(
            second_profile,
            PromptId(Uuid::from_u128(22)),
            second,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        assert!(matches!(
            service.apply_event(event(
                second_profile,
                PromptId(Uuid::from_u128(21)),
                first,
                1,
                None,
                AttemptEventKind::Succeeded,
            )),
            Err(ExecutionPresentationError::CrossProfileAttempt { .. })
        ));
        assert_eq!(service.snapshot(first_profile)?.attempts.len(), 1);
        assert_eq!(service.snapshot(second_profile)?.attempts.len(), 1);
        Ok(())
    }

    #[test]
    fn interrupt_ack_waits_for_worker_terminal_event() -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(21));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        service.dispatch(
            command(
                2,
                profile_id,
                ExecutionControlCommandKind::Interrupt {
                    attempt_id,
                    reason: "user interrupt".to_owned(),
                },
            ),
            &controller,
        )?;
        assert_eq!(
            service
                .snapshot(profile_id)?
                .attempts
                .first()
                .map(|attempt| attempt.state),
            Some(AttemptState::Cancelling)
        );
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            2,
            None,
            AttemptEventKind::Interrupted {
                reason: "worker stopped".to_owned(),
            },
        ))?;
        assert_eq!(
            service
                .snapshot(profile_id)?
                .attempts
                .first()
                .map(|attempt| attempt.state),
            Some(AttemptState::Interrupted)
        );
        assert_eq!(
            service
                .snapshot(profile_id)?
                .attempts
                .first()
                .and_then(|attempt| attempt.interrupted_reason.as_deref()),
            Some("worker stopped")
        );
        Ok(())
    }

    #[test]
    fn queue_controls_and_retry_preserve_explicit_lineage() -> Result<(), Box<dyn std::error::Error>>
    {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(21));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let first = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        let second = queue_attempt(&mut service, &controller, profile_id, 2, 22)?;
        service.dispatch(
            command(
                3,
                profile_id,
                ExecutionControlCommandKind::Reorder {
                    attempt_id: second,
                    position: 0,
                },
            ),
            &controller,
        )?;
        assert_eq!(
            service
                .snapshot(profile_id)?
                .queue
                .first()
                .map(|item| item.attempt_id),
            Some(second)
        );
        service.apply_event(event(
            profile_id,
            prompt_id,
            first,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            first,
            1,
            None,
            AttemptEventKind::Failed {
                failure: ExecutionFailure::new("fixture_failure", "fixture failed").retryable(true),
            },
        ))?;
        let retry_ack = service.dispatch(
            command(
                4,
                profile_id,
                ExecutionControlCommandKind::Retry {
                    attempt_id: first,
                    source: RetryPromptSource::OriginalPrompt,
                    replacement_plan: None,
                },
            ),
            &controller,
        )?;
        let retry_attempt = match retry_ack.outcome {
            ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            } => attempt_id,
            _ => return Err("retry was rejected".into()),
        };
        assert_ne!(retry_attempt, first);
        let snapshot = service.snapshot(profile_id)?;
        let retry = snapshot
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == retry_attempt)
            .ok_or("missing retry attempt")?;
        assert_eq!(retry.retry_of, Some(first));
        assert_eq!(retry.retry_source, Some(RetryPromptSource::OriginalPrompt));
        service.dispatch(
            command(
                5,
                profile_id,
                ExecutionControlCommandKind::Cancel {
                    attempt_id: second,
                    reason: "user".to_owned(),
                },
            ),
            &controller,
        )?;
        assert!(
            service
                .snapshot(profile_id)?
                .attempts
                .iter()
                .any(|attempt| attempt.attempt_id == second
                    && attempt.state == AttemptState::Cancelled)
        );
        service.dispatch(
            command(
                6,
                profile_id,
                ExecutionControlCommandKind::Interrupt {
                    attempt_id: retry_attempt,
                    reason: "worker stopped".to_owned(),
                },
            ),
            &controller,
        )?;
        let third = queue_attempt(&mut service, &controller, profile_id, 7, 23)?;
        let clear_pending = command(
            8,
            profile_id,
            ExecutionControlCommandKind::ClearPending {
                reason: "user cleared the queue".to_owned(),
            },
        );
        service.submit(clear_pending.clone())?;
        assert!(
            service
                .snapshot(profile_id)?
                .queue
                .iter()
                .any(|item| item.attempt_id == third)
        );
        service.apply_ack(ExecutionCommandAck {
            request_id: clear_pending.request_id,
            profile_id,
            outcome: ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: None,
            },
        })?;
        assert!(service.snapshot(profile_id)?.queue.is_empty());
        let clear_history = command(9, profile_id, ExecutionControlCommandKind::ClearHistory);
        service.submit(clear_history.clone())?;
        assert!(!service.snapshot(profile_id)?.attempts.is_empty());
        service.apply_ack(ExecutionCommandAck {
            request_id: clear_history.request_id,
            profile_id,
            outcome: ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: None,
            },
        })?;
        assert!(service.snapshot(profile_id)?.attempts.is_empty());
        Ok(())
    }

    #[test]
    fn typed_preview_output_and_failure_reduce_without_losing_payload_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(21));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        let node_id = NodeId::from("output");
        let preview = ExecutionPreview {
            preview_id: Uuid::from_u128(200),
            node_id: node_id.clone(),
            revision: 1,
            frame_index: Some(0),
            output_index: Some(0),
            media_kind: crate::OutputMediaKind::Image,
            media_type: "image/png".to_owned(),
            width: Some(64),
            height: Some(64),
            encoded_bytes: vec![1, 2, 3],
        };
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            Some(node_id.clone()),
            AttemptEventKind::Preview {
                preview: preview.clone(),
            },
        ))?;
        let output = ExecutionOutput {
            output_id: Uuid::from_u128(201),
            node_id: node_id.clone(),
            output_index: 0,
            name: "result".to_owned(),
            media_kind: crate::OutputMediaKind::Image,
            media_type: "image/png".to_owned(),
            subfolder: Some("fixtures".to_owned()),
            storage_type: Some("output".to_owned()),
            metadata: BTreeMap::new(),
            view_reference: Some("native://result?preview=1".to_owned()),
            download_reference: Some("native://result?download=1".to_owned()),
            availability: crate::ExecutionOutputAvailability::Ready {
                reference: "native://result".to_owned(),
                byte_length: 3,
            },
            created_at: DateTime::<Utc>::UNIX_EPOCH,
        };
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            2,
            Some(node_id.clone()),
            AttemptEventKind::OutputAvailable {
                output: output.clone(),
            },
        ))?;
        let running_snapshot = service.snapshot(profile_id)?;
        let running_attempt = running_snapshot.attempts.first().ok_or("missing attempt")?;
        assert_eq!(running_attempt.preview, Some(preview.clone()));
        assert_eq!(
            running_attempt.previews.as_slice(),
            std::slice::from_ref(&preview)
        );
        let failure = ExecutionFailure::new("node_failed", "fixture failure")
            .at_node(node_id.clone())
            .retryable(true);
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            3,
            Some(node_id),
            AttemptEventKind::Failed {
                failure: failure.clone(),
            },
        ))?;
        let snapshot = service.snapshot(profile_id)?;
        let attempt = snapshot.attempts.first().ok_or("missing attempt")?;
        assert_eq!(attempt.preview, None);
        assert!(attempt.previews.is_empty());
        assert_eq!(attempt.outputs, [output]);
        assert_eq!(attempt.failure, Some(failure));
        assert_eq!(attempt.state, AttemptState::Failed);
        Ok(())
    }

    #[test]
    fn reconciliation_acknowledges_pending_commands_and_rejects_stale_snapshots()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        let pending = command(
            2,
            profile_id,
            ExecutionControlCommandKind::Reorder {
                attempt_id,
                position: 0,
            },
        );
        service.submit(pending.clone())?;
        let state = service.profiles.get(&profile_id).ok_or("missing profile")?;
        let mut records = state.history.records().iter().cloned().collect::<Vec<_>>();
        if let Some(record) = records.first_mut() {
            record.source_projection = Some(crate::AttemptSourceProjection::Provider {
                provider_id: "native-provider-fixture".to_owned(),
                state: crate::ProviderAttemptState::Unknown {
                    raw_state: "waiting_for_remote_capacity".to_owned(),
                },
            });
        }
        let unknown_attempt_id = AttemptId(Uuid::from_u128(999));
        let mut unknown_record = AttemptRecord::queued(
            profile_id,
            PromptId(Uuid::from_u128(998)),
            unknown_attempt_id,
        );
        unknown_record.source_projection = Some(crate::AttemptSourceProjection::Unknown {
            source_id: Some("legacy-provider".to_owned()),
            raw_state: "orphaned_by_provider".to_owned(),
        });
        records.push(unknown_record);
        let reconciliation = ExecutionReconciliation {
            profile_id,
            source_revision: 7,
            source: ExecutionDataSource::Persisted,
            status: ExecutionSnapshotStatus::Ready,
            queue: state.queue.items().to_vec(),
            records,
            plans: state
                .plans
                .iter()
                .map(|(attempt_id, plan)| AttemptPlanSnapshot {
                    attempt_id: *attempt_id,
                    plan: plan.clone(),
                })
                .collect(),
            acknowledged_requests: vec![pending.request_id],
        };
        service.reconcile(reconciliation.clone())?;
        let snapshot = service.snapshot(profile_id)?;
        assert_eq!(snapshot.source_revision, Some(7));
        assert_eq!(snapshot.source, ExecutionDataSource::Persisted);
        assert!(snapshot.pending_commands.is_empty());
        assert!(snapshot.attempts.iter().any(|attempt| {
            matches!(
                &attempt.source_projection,
                Some(crate::AttemptSourceProjection::Provider {
                    provider_id,
                    state: crate::ProviderAttemptState::Unknown { raw_state },
                }) if provider_id == "native-provider-fixture"
                    && raw_state == "waiting_for_remote_capacity"
            )
        }));
        assert!(snapshot.attempts.iter().any(|attempt| {
            attempt.attempt_id == unknown_attempt_id
                && matches!(
                    &attempt.source_projection,
                    Some(crate::AttemptSourceProjection::Unknown {
                        source_id: Some(source_id),
                        raw_state,
                    }) if source_id == "legacy-provider" && raw_state == "orphaned_by_provider"
                )
        }));
        assert!(matches!(
            service.reconcile(reconciliation),
            Err(ExecutionPresentationError::StaleReconciliation {
                current: 7,
                received: 7
            })
        ));
        Ok(())
    }

    #[test]
    fn failed_ack_mutation_is_completed_without_partial_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        let duplicate_prompt = command(
            2,
            profile_id,
            ExecutionControlCommandKind::Queue {
                plan: plan(21),
                priority: 0,
                front: false,
            },
        );
        service.submit(duplicate_prompt.clone())?;
        assert!(matches!(
            service.apply_ack(ExecutionCommandAck {
                request_id: duplicate_prompt.request_id,
                profile_id,
                outcome: ExecutionCommandOutcome::Accepted {
                    assigned_attempt_id: Some(AttemptId(Uuid::from_u128(101))),
                },
            }),
            Err(ExecutionPresentationError::Queue(
                ExecutionQueueError::DuplicatePrompt(_)
            ))
        ));
        let snapshot = service.snapshot(profile_id)?;
        assert_eq!(snapshot.queue.len(), 1);
        assert_eq!(snapshot.attempts.len(), 1);
        assert!(snapshot.pending_commands.is_empty());
        assert_eq!(
            snapshot
                .recent_command_results
                .last()
                .map(|ack| ack.request_id),
            Some(duplicate_prompt.request_id)
        );
        assert!(matches!(
            service.submit(duplicate_prompt.clone()),
            Err(ExecutionPresentationError::DuplicateRequest(request_id))
                if request_id == duplicate_prompt.request_id
        ));
        Ok(())
    }

    #[test]
    fn pending_and_completed_request_retention_is_bounded() -> Result<(), Box<dyn std::error::Error>>
    {
        let profile_id = profile(1);
        let mut pending_service = ExecutionPresentationService::new(16)?;
        pending_service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        for request in 1..=PENDING_REQUEST_CAPACITY as u128 {
            pending_service.submit(command(
                request,
                profile_id,
                ExecutionControlCommandKind::ClearHistory,
            ))?;
        }
        assert!(matches!(
            pending_service.submit(command(
                PENDING_REQUEST_CAPACITY as u128 + 1,
                profile_id,
                ExecutionControlCommandKind::ClearHistory,
            )),
            Err(ExecutionPresentationError::PendingRequestCapacity {
                maximum: PENDING_REQUEST_CAPACITY
            })
        ));

        let controller = AcceptingExecutionController;
        let mut completed_service = ExecutionPresentationService::new(16)?;
        completed_service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        for request in 1..=COMPLETED_REQUEST_CAPACITY as u128 + 1 {
            completed_service.dispatch(
                command(
                    request,
                    profile_id,
                    ExecutionControlCommandKind::ClearHistory,
                ),
                &controller,
            )?;
        }
        assert_eq!(
            completed_service.completed_requests.len(),
            COMPLETED_REQUEST_CAPACITY
        );
        let completed_snapshot = completed_service.snapshot(profile_id)?;
        assert_eq!(
            completed_snapshot.recent_command_results.len(),
            RECENT_COMMAND_RESULT_CAPACITY
        );
        assert_eq!(
            completed_snapshot
                .recent_command_results
                .first()
                .map(|acknowledgement| acknowledgement.request_id),
            Some(RequestId(Uuid::from_u128(2)))
        );
        assert!(
            !completed_service
                .completed_requests
                .contains(&RequestId(Uuid::from_u128(1)))
        );
        assert!(
            completed_service
                .completed_requests
                .contains(&RequestId(Uuid::from_u128(
                    COMPLETED_REQUEST_CAPACITY as u128 + 1
                )))
        );
        Ok(())
    }

    #[test]
    fn reconciliation_unions_partial_state_and_identical_success_is_idempotent()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let local_attempt = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        let remote_attempt = AttemptId(Uuid::from_u128(500));
        let remote_plan = plan(22);
        let mut remote_record =
            AttemptRecord::queued(profile_id, remote_plan.prompt_id, remote_attempt);
        remote_record.apply(event(
            profile_id,
            remote_plan.prompt_id,
            remote_attempt,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        remote_record.apply(event(
            profile_id,
            remote_plan.prompt_id,
            remote_attempt,
            1,
            None,
            AttemptEventKind::Succeeded,
        ))?;
        let reconciliation = |source_revision, queue| ExecutionReconciliation {
            profile_id,
            source_revision,
            source: ExecutionDataSource::Live,
            status: ExecutionSnapshotStatus::Ready,
            queue,
            records: vec![remote_record.clone()],
            plans: vec![AttemptPlanSnapshot {
                attempt_id: remote_attempt,
                plan: remote_plan.clone(),
            }],
            acknowledged_requests: Vec::new(),
        };
        let original_queue = service.snapshot(profile_id)?.queue;
        assert!(service.reconcile_with_change(reconciliation(1, original_queue))?);
        let first = service.snapshot(profile_id)?;
        assert!(
            first
                .attempts
                .iter()
                .any(|attempt| attempt.attempt_id == local_attempt)
        );
        assert!(
            first
                .attempts
                .iter()
                .any(|attempt| attempt.attempt_id == remote_attempt)
        );
        let presentation_revision = first.revision;
        assert!(!service.reconcile_with_change(reconciliation(2, first.queue))?);
        let second = service.snapshot(profile_id)?;
        assert_eq!(second.revision, presentation_revision);
        assert_eq!(second.source_revision, Some(2));
        Ok(())
    }

    #[test]
    fn progress_and_preview_projection_is_per_node_frame_and_terminally_cleared()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(21));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        for (sequence, node, completed) in [(1, "first", 1), (2, "second", 2)] {
            service.apply_event(event(
                profile_id,
                prompt_id,
                attempt_id,
                sequence,
                Some(NodeId::from(node)),
                AttemptEventKind::Progress {
                    completed,
                    total: 4,
                },
            ))?;
        }
        for (sequence, frame_index, revision) in [(3, 0, 1), (4, 1, 1), (5, 0, 2)] {
            let node_id = NodeId::from("preview");
            service.apply_event(event(
                profile_id,
                prompt_id,
                attempt_id,
                sequence,
                Some(node_id.clone()),
                AttemptEventKind::Preview {
                    preview: ExecutionPreview {
                        preview_id: Uuid::from_u128(sequence as u128),
                        node_id,
                        revision,
                        frame_index: Some(frame_index),
                        output_index: Some(0),
                        media_kind: crate::OutputMediaKind::Image,
                        media_type: "image/png".to_owned(),
                        width: Some(1),
                        height: Some(1),
                        encoded_bytes: vec![0],
                    },
                },
            ))?;
        }
        let running = service.snapshot(profile_id)?;
        let attempt = running.attempts.first().ok_or("missing attempt")?;
        assert_eq!(attempt.node_progress.len(), 2);
        assert_eq!(attempt.previews.len(), 2);
        assert_eq!(
            attempt
                .previews
                .iter()
                .find(|preview| preview.frame_index == Some(0))
                .map(|preview| preview.revision),
            Some(2)
        );
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            6,
            None,
            AttemptEventKind::Interrupted {
                reason: "fixture interruption".to_owned(),
            },
        ))?;
        let terminal = service.snapshot(profile_id)?;
        let attempt = terminal.attempts.first().ok_or("missing attempt")?;
        assert!(attempt.node_progress.is_empty());
        assert!(attempt.previews.is_empty());
        assert_eq!(
            attempt.interrupted_reason.as_deref(),
            Some("fixture interruption")
        );
        Ok(())
    }

    #[test]
    fn failed_attempt_recovery_requires_retryable_evidence()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(21));
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            None,
            AttemptEventKind::Failed {
                failure: ExecutionFailure::new("invalid", "fixture")
                    .with_origin(crate::ExecutionFailureOrigin::Validation),
            },
        ))?;
        let snapshot = service.snapshot(profile_id)?;
        let attempt = snapshot.attempts.first().ok_or("missing attempt")?;
        assert_eq!(
            attempt.failure.as_ref().map(|failure| failure.origin),
            Some(crate::ExecutionFailureOrigin::Validation)
        );
        assert!(matches!(
            attempt.retry_eligibility(),
            crate::OperationEligibility::Unavailable { reason }
                if reason.contains("not retryable")
        ));
        assert!(!attempt.recovery_eligibility().is_allowed());
        let acknowledgement = service.dispatch(
            command(
                2,
                profile_id,
                ExecutionControlCommandKind::Retry {
                    attempt_id,
                    source: RetryPromptSource::OriginalPrompt,
                    replacement_plan: None,
                },
            ),
            &controller,
        )?;
        assert!(matches!(
            acknowledgement.outcome,
            ExecutionCommandOutcome::Rejected { failure }
                if failure.message.contains("not retryable")
        ));
        Ok(())
    }

    #[test]
    fn persisted_attempts_restore_with_identity_and_plan() -> Result<(), Box<dyn std::error::Error>>
    {
        let profile_id = profile(1);
        let prompt_id = PromptId(Uuid::from_u128(21));
        let controller = AcceptingExecutionController;
        let mut original = ExecutionPresentationService::new(16)?;
        let attempt_id = queue_attempt(&mut original, &controller, profile_id, 1, 21)?;
        original.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ))?;
        original.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            None,
            AttemptEventKind::Succeeded,
        ))?;
        let persisted = original.persisted_attempts(profile_id)?;

        let mut restored = ExecutionPresentationService::new(16)?;
        restored.initialize_profile(
            profile_id,
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Partial {
                failure: ExecutionFailure::new("runtime_not_connected", "fixture"),
            },
        )?;
        assert!(restored.restore_persisted_attempts(profile_id, persisted)?);
        let snapshot = restored.snapshot(profile_id)?;
        assert_eq!(snapshot.source, ExecutionDataSource::Recovery);
        assert!(matches!(
            snapshot.status,
            ExecutionSnapshotStatus::Partial { .. }
        ));
        assert_eq!(snapshot.attempts.len(), 1);
        assert_eq!(snapshot.attempts[0].attempt_id, attempt_id);
        assert_eq!(snapshot.attempts[0].state, AttemptState::Succeeded);
        assert!(restored.persisted_attempts(profile_id)?[0].plan.is_some());
        Ok(())
    }

    #[test]
    fn service_allocates_attempts_and_owns_their_cancellation_tokens()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let controller = AcceptingExecutionController;
        let mut service = ExecutionPresentationService::new_with_first_attempt_id(
            16,
            AttemptId(Uuid::from_u128(100)),
        )?;
        let first = queue_attempt(&mut service, &controller, profile_id, 1, 21)?;
        let second = queue_attempt(&mut service, &controller, profile_id, 2, 22)?;
        assert_eq!(first, AttemptId(Uuid::from_u128(100)));
        assert_eq!(second, AttemptId(Uuid::from_u128(101)));

        let lease = service
            .next_queued_attempt(profile_id)?
            .ok_or("missing canonical execution lease")?;
        assert_eq!(lease.attempt_id, first);
        let cancellation = lease.cancellation;
        let acknowledgement = service.dispatch(
            command(
                3,
                profile_id,
                ExecutionControlCommandKind::Cancel {
                    attempt_id: first,
                    reason: "fixture cancellation".to_owned(),
                },
            ),
            &controller,
        )?;
        assert!(matches!(
            acknowledgement.outcome,
            ExecutionCommandOutcome::Accepted { .. }
        ));
        assert!(cancellation.is_cancelled());
        assert!(matches!(
            service.cancellation_token(profile_id, first),
            Err(ExecutionPresentationError::MissingCancellationToken(attempt_id))
                if attempt_id == first
        ));
        assert!(
            !service
                .cancellation_token(profile_id, second)?
                .is_cancelled()
        );
        Ok(())
    }

    #[test]
    fn persisted_queue_projection_restores_exact_ordering_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let controller = AcceptingExecutionController;
        let mut original = ExecutionPresentationService::new_with_first_attempt_id(
            16,
            AttemptId(Uuid::from_u128(200)),
        )?;
        original.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let mut attempts = Vec::new();
        for (request, prompt, priority, front) in
            [(1, 21, 0, false), (2, 22, 5, false), (3, 23, 5, true)]
        {
            let acknowledgement = original.dispatch(
                command(
                    request,
                    profile_id,
                    ExecutionControlCommandKind::Queue {
                        plan: plan(prompt),
                        priority,
                        front,
                    },
                ),
                &controller,
            )?;
            let ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            } = acknowledgement.outcome
            else {
                return Err("queue fixture was rejected".into());
            };
            attempts.push(attempt_id);
        }
        let first_attempt = attempts
            .first()
            .copied()
            .ok_or("missing first queued attempt")?;
        original.dispatch(
            command(
                4,
                profile_id,
                ExecutionControlCommandKind::Reorder {
                    attempt_id: first_attempt,
                    position: 1,
                },
            ),
            &controller,
        )?;
        let expected = original.snapshot(profile_id)?.queue;
        let persisted = original.persisted_attempts(profile_id)?;
        assert!(persisted.iter().all(|attempt| attempt.queue.is_some()));

        let mut restored = ExecutionPresentationService::new(16)?;
        restored.initialize_profile(
            profile_id,
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Ready,
        )?;
        assert!(restored.restore_persisted_attempts(profile_id, persisted)?);
        assert_eq!(restored.snapshot(profile_id)?.queue, expected);
        Ok(())
    }

    #[test]
    fn persisted_queued_attempts_resume_and_in_flight_attempts_are_interrupted_after_restart()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(1);
        let mut persisted = Vec::new();
        for (offset, state) in [
            (0_u128, AttemptState::Queued),
            (1, AttemptState::Running),
            (2, AttemptState::Cancelling),
        ] {
            let plan = plan(30 + offset);
            let attempt_id = AttemptId(Uuid::from_u128(300 + offset));
            let mut record = AttemptRecord::queued(profile_id, plan.prompt_id, attempt_id);
            if matches!(state, AttemptState::Running | AttemptState::Cancelling) {
                record.apply(event(
                    profile_id,
                    plan.prompt_id,
                    attempt_id,
                    0,
                    None,
                    AttemptEventKind::Started,
                ))?;
            }
            if state == AttemptState::Cancelling {
                record.apply(event(
                    profile_id,
                    plan.prompt_id,
                    attempt_id,
                    1,
                    None,
                    AttemptEventKind::CancelRequested {
                        reason: "fixture cancellation".to_owned(),
                        interrupt: false,
                    },
                ))?;
            }
            persisted.push(PersistedExecutionAttempt::new(
                record,
                Some(plan),
                ExecutionDataSource::Persisted,
            )?);
        }
        let mut service = ExecutionPresentationService::new(16)?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Partial {
                failure: ExecutionFailure::new("runtime_not_connected", "fixture"),
            },
        )?;
        assert!(service.restore_persisted_attempts(profile_id, persisted)?);
        let snapshot = service.snapshot(profile_id)?;
        assert_eq!(snapshot.queue.len(), 1);
        assert_eq!(
            snapshot.queue[0].attempt_id,
            AttemptId(Uuid::from_u128(300))
        );
        assert_eq!(snapshot.attempts.len(), 3);
        let queued = snapshot
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == AttemptId(Uuid::from_u128(300)))
            .ok_or("missing recovered queued attempt")?;
        assert_eq!(queued.state, AttemptState::Queued);
        assert_eq!(queued.last_sequence, None);
        for attempt_id in [301_u128, 302] {
            let attempt = snapshot
                .attempts
                .iter()
                .find(|attempt| attempt.attempt_id == AttemptId(Uuid::from_u128(attempt_id)))
                .ok_or("missing interrupted in-flight attempt")?;
            assert_eq!(attempt.state, AttemptState::Interrupted);
            assert_eq!(
                attempt.recovery_interruption_reason,
                Some(ExecutionRecoveryInterruptionReason::RuntimeRestart)
            );
            assert_eq!(
                attempt.interrupted_reason.as_deref(),
                Some(ExecutionRecoveryInterruptionReason::RuntimeRestart.summary())
            );
        }
        let persisted = service.persisted_attempts(profile_id)?;
        assert!(persisted.iter().any(|attempt| {
            attempt.record.attempt_id == AttemptId(Uuid::from_u128(300))
                && attempt.record.state == AttemptState::Queued
                && attempt.record.events.is_empty()
        }));
        assert!(
            persisted
                .iter()
                .filter(|attempt| { attempt.record.attempt_id != AttemptId(Uuid::from_u128(300)) })
                .all(|attempt| {
                    attempt.record.state == AttemptState::Interrupted
                        && matches!(
                            attempt.record.events.last().map(|event| &event.kind),
                            Some(AttemptEventKind::RecoveryInterrupted {
                                reason: ExecutionRecoveryInterruptionReason::RuntimeRestart
                            })
                        )
                })
        );
        Ok(())
    }

    #[test]
    fn output_availability_has_explicit_recovery_and_removal_eligibility() {
        let output = |availability| ExecutionOutput {
            output_id: Uuid::from_u128(1),
            node_id: NodeId::from("output"),
            output_index: 0,
            name: "preview".to_owned(),
            media_kind: crate::OutputMediaKind::Image,
            media_type: "image/png".to_owned(),
            subfolder: None,
            storage_type: Some("output".to_owned()),
            metadata: BTreeMap::new(),
            view_reference: None,
            download_reference: None,
            availability,
            created_at: DateTime::<Utc>::UNIX_EPOCH,
        };
        let missing = output(crate::ExecutionOutputAvailability::Missing {
            reference: Some("native://missing".to_owned()),
            reason: "moved".to_owned(),
        });
        assert!(missing.recovery_eligibility().is_allowed());
        assert!(missing.removal_eligibility().is_allowed());
        let expired = output(crate::ExecutionOutputAvailability::Expired {
            reference: Some("native://expired".to_owned()),
            expired_at: DateTime::<Utc>::UNIX_EPOCH,
            reason: "retention window elapsed".to_owned(),
        });
        assert!(expired.recovery_eligibility().is_allowed());
        assert!(expired.removal_eligibility().is_allowed());
        let externally_deleted = output(crate::ExecutionOutputAvailability::ExternallyDeleted {
            reference: "native://deleted".to_owned(),
            detected_at: DateTime::<Utc>::UNIX_EPOCH,
        });
        assert!(externally_deleted.recovery_eligibility().is_allowed());
        assert!(externally_deleted.removal_eligibility().is_allowed());
        let forbidden = output(crate::ExecutionOutputAvailability::Forbidden {
            reason: "capability denied".to_owned(),
        });
        assert!(!forbidden.recovery_eligibility().is_allowed());
        assert!(!forbidden.removal_eligibility().is_allowed());
        let unsupported = output(crate::ExecutionOutputAvailability::Unsupported {
            media_type: "application/x-fixture".to_owned(),
            reason: "no native decoder".to_owned(),
        });
        assert!(!unsupported.recovery_eligibility().is_allowed());
        assert!(!unsupported.removal_eligibility().is_allowed());
        let corrupt = output(crate::ExecutionOutputAvailability::Corrupt {
            reference: Some("native://corrupt".to_owned()),
            reason: "checksum mismatch".to_owned(),
        });
        assert!(corrupt.recovery_eligibility().is_allowed());
        assert!(corrupt.removal_eligibility().is_allowed());
        let ready = output(crate::ExecutionOutputAvailability::Ready {
            reference: "native://ready".to_owned(),
            byte_length: 42,
        });
        assert!(!ready.recovery_eligibility().is_allowed());
        assert!(ready.removal_eligibility().is_allowed());
        let removed = output(crate::ExecutionOutputAvailability::Removed {
            reason: "removed by the user".to_owned(),
        });
        assert!(!removed.recovery_eligibility().is_allowed());
        assert!(!removed.removal_eligibility().is_allowed());
    }

    #[test]
    fn output_operations_update_and_restore_one_canonical_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(0x1800);
        let prompt_id = PromptId(Uuid::from_u128(0x1801));
        let output_id = Uuid::from_u128(0x1802);
        let mut service = ExecutionPresentationService::new_with_first_attempt_id(
            8,
            AttemptId(Uuid::from_u128(0x1803)),
        )?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let attempt_id = queue_attempt(
            &mut service,
            &AcceptingExecutionController,
            profile_id,
            0x1804,
            prompt_id.0.as_u128(),
        )?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            Some(NodeId::from("output")),
            AttemptEventKind::Started,
        ))?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            Some(NodeId::from("output")),
            AttemptEventKind::OutputAvailable {
                output: ExecutionOutput {
                    output_id,
                    node_id: NodeId::from("output"),
                    output_index: 0,
                    name: "native-image.png".to_owned(),
                    media_kind: crate::OutputMediaKind::Image,
                    media_type: "image/png".to_owned(),
                    subfolder: None,
                    storage_type: Some("output".to_owned()),
                    metadata: BTreeMap::new(),
                    view_reference: Some("zed-asset://output/native-image.png".to_owned()),
                    download_reference: Some("zed-asset://output/native-image.png".to_owned()),
                    availability: crate::ExecutionOutputAvailability::Missing {
                        reference: Some("zed-asset://output/native-image.png".to_owned()),
                        reason: "fixture missing".to_owned(),
                    },
                    created_at: DateTime::<Utc>::UNIX_EPOCH,
                },
            },
        ))?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            2,
            None,
            AttemptEventKind::Succeeded,
        ))?;
        service.apply_output_operation(
            profile_id,
            attempt_id,
            output_id,
            crate::ExecutionOutputOperationAction::Recover,
            crate::ExecutionOutputAvailability::Ready {
                reference: "zed-asset://output/native-image.png".to_owned(),
                byte_length: 128,
            },
        )?;
        let snapshot = service.snapshot(profile_id)?;
        let recovered_output = snapshot
            .attempts
            .first()
            .and_then(|attempt| attempt.outputs.first())
            .ok_or("recovered output missing from canonical projection")?;
        assert!(matches!(
            recovered_output.availability,
            crate::ExecutionOutputAvailability::Ready {
                byte_length: 128,
                ..
            }
        ));
        let recovered_revision = snapshot.revision;
        service.profile_mut(profile_id)?.revision = u64::MAX;
        assert!(matches!(
            service.apply_output_operation(
                profile_id,
                attempt_id,
                output_id,
                crate::ExecutionOutputOperationAction::Remove,
                crate::ExecutionOutputAvailability::Removed {
                    reason: "failure-injected removal".to_owned(),
                },
            ),
            Err(ExecutionPresentationError::RevisionExhausted)
        ));
        let rolled_back_snapshot = service.snapshot(profile_id)?;
        let rolled_back_output = rolled_back_snapshot
            .attempts
            .first()
            .and_then(|attempt| attempt.outputs.first())
            .ok_or("rollback removed output from canonical projection")?;
        assert!(matches!(
            rolled_back_output.availability,
            crate::ExecutionOutputAvailability::Ready { .. }
        ));
        service.profile_mut(profile_id)?.revision = recovered_revision;
        service.apply_output_operation(
            profile_id,
            attempt_id,
            output_id,
            crate::ExecutionOutputOperationAction::Remove,
            crate::ExecutionOutputAvailability::Removed {
                reason: "confirmed fixture removal".to_owned(),
            },
        )?;
        let persisted_profile = service.persisted_profile(profile_id)?;
        let persisted_attempts = service.persisted_attempts(profile_id)?;
        let mut restored = ExecutionPresentationService::new(8)?;
        restored.restore_persisted_profile(persisted_profile)?;
        restored.restore_persisted_attempts(profile_id, persisted_attempts)?;
        let restored_snapshot = restored.snapshot(profile_id)?;
        let removed_output = restored_snapshot
            .attempts
            .first()
            .and_then(|attempt| attempt.outputs.first())
            .ok_or("removed output missing from restored canonical projection")?;
        assert!(matches!(
            removed_output.availability,
            crate::ExecutionOutputAvailability::Removed { .. }
        ));
        assert!(matches!(
            restored.output_operation_eligibility(
                profile_id,
                attempt_id,
                output_id,
                crate::ExecutionOutputOperationAction::Remove,
            )?,
            crate::OperationEligibility::Unavailable { .. }
        ));
        Ok(())
    }

    #[test]
    fn durable_output_transaction_rolls_back_prepared_effect_when_persistence_fails()
    -> Result<(), Box<dyn std::error::Error>> {
        let profile_id = profile(0x1810);
        let prompt_id = PromptId(Uuid::from_u128(0x1811));
        let output_id = Uuid::from_u128(0x1812);
        let mut service = ExecutionPresentationService::new_with_first_attempt_id(
            8,
            AttemptId(Uuid::from_u128(0x1813)),
        )?;
        service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let attempt_id = queue_attempt(
            &mut service,
            &AcceptingExecutionController,
            profile_id,
            0x1814,
            prompt_id.0.as_u128(),
        )?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            Some(NodeId::from("output")),
            AttemptEventKind::Started,
        ))?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            Some(NodeId::from("output")),
            AttemptEventKind::OutputAvailable {
                output: ExecutionOutput {
                    output_id,
                    node_id: NodeId::from("output"),
                    output_index: 0,
                    name: "native-image.png".to_owned(),
                    media_kind: crate::OutputMediaKind::Image,
                    media_type: "image/png".to_owned(),
                    subfolder: None,
                    storage_type: Some("output".to_owned()),
                    metadata: BTreeMap::new(),
                    view_reference: Some("zed-asset://output/native-image.png".to_owned()),
                    download_reference: Some("zed-asset://output/native-image.png".to_owned()),
                    availability: crate::ExecutionOutputAvailability::Ready {
                        reference: "zed-asset://output/native-image.png".to_owned(),
                        byte_length: 128,
                    },
                    created_at: DateTime::<Utc>::UNIX_EPOCH,
                },
            },
        ))?;
        service.apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            2,
            None,
            AttemptEventKind::Succeeded,
        ))?;
        let steps = Arc::new(Mutex::new(Vec::new()));
        let owner = ExecutionPresentationOwner::persistent(
            service,
            Arc::new(RecordingPersistence { fail: true, steps }),
        );
        let committed = Arc::new(AtomicBool::new(false));
        let rolled_back = Arc::new(AtomicBool::new(false));
        let result = smol::block_on(owner.apply_output_operation_transaction_durable(
            profile_id,
            attempt_id,
            output_id,
            crate::ExecutionOutputOperationAction::Remove,
            crate::ExecutionOutputAvailability::Removed {
                reason: "confirmed fixture removal".to_owned(),
            },
            {
                let committed = committed.clone();
                move || {
                    committed.store(true, Ordering::SeqCst);
                    Ok(())
                }
            },
            {
                let rolled_back = rolled_back.clone();
                move || {
                    rolled_back.store(true, Ordering::SeqCst);
                    Ok(())
                }
            },
        ));
        assert!(matches!(
            result,
            Err(ExecutionPresentationError::Persistence(_))
        ));
        assert!(!committed.load(Ordering::SeqCst));
        assert!(rolled_back.load(Ordering::SeqCst));
        assert!(matches!(
            owner
                .snapshot(profile_id)?
                .attempts
                .first()
                .and_then(|attempt| attempt.outputs.first())
                .map(|output| &output.availability),
            Some(crate::ExecutionOutputAvailability::Ready { .. })
        ));
        Ok(())
    }
}
