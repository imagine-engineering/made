use crate::CompiledPlan;
use async_channel::{Receiver, Sender, TrySendError};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sim_types::{AttemptId, NodeId, ProfileId, PromptId};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashSet, VecDeque},
    sync::Arc,
};
use thiserror::Error;
use uuid::Uuid;

pub const ATTEMPT_HISTORY_SCHEMA_VERSION: u16 = 2;
pub const MAX_RETAINED_ATTEMPT_EVENTS: usize = 16_384;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueuedPrompt {
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub plan: CompiledPlan,
    pub priority: i32,
    pub front: bool,
    pub enqueue_sequence: u64,
    pub queued_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ExecutionQueueError {
    #[error("prompt {0:?} is already queued")]
    DuplicatePrompt(PromptId),
    #[error("attempt {0:?} is already queued")]
    DuplicateAttempt(AttemptId),
    #[error("queue sequence is exhausted")]
    SequenceExhausted,
    #[error("attempt {0:?} is not queued")]
    UnknownAttempt(AttemptId),
    #[error("queue position {position} is outside a queue of length {length}")]
    InvalidPosition { position: usize, length: usize },
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionQueue {
    items: Vec<QueuedPrompt>,
    next_sequence: u64,
}

impl ExecutionQueue {
    pub fn from_ordered(items: Vec<QueuedPrompt>) -> Result<Self, ExecutionQueueError> {
        let mut prompts = HashSet::new();
        let mut attempts = HashSet::new();
        let mut next_sequence = 0_u64;
        for item in &items {
            if !prompts.insert((item.profile_id, item.prompt_id)) {
                return Err(ExecutionQueueError::DuplicatePrompt(item.prompt_id));
            }
            if !attempts.insert((item.profile_id, item.attempt_id)) {
                return Err(ExecutionQueueError::DuplicateAttempt(item.attempt_id));
            }
            next_sequence = next_sequence.max(
                item.enqueue_sequence
                    .checked_add(1)
                    .ok_or(ExecutionQueueError::SequenceExhausted)?,
            );
        }
        Ok(Self {
            items,
            next_sequence,
        })
    }

    pub fn enqueue(
        &mut self,
        profile_id: ProfileId,
        plan: CompiledPlan,
        attempt_id: AttemptId,
        priority: i32,
        front: bool,
    ) -> Result<(), ExecutionQueueError> {
        if self
            .items
            .iter()
            .any(|item| item.profile_id == profile_id && item.prompt_id == plan.prompt_id)
        {
            return Err(ExecutionQueueError::DuplicatePrompt(plan.prompt_id));
        }
        if self
            .items
            .iter()
            .any(|item| item.profile_id == profile_id && item.attempt_id == attempt_id)
        {
            return Err(ExecutionQueueError::DuplicateAttempt(attempt_id));
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(ExecutionQueueError::SequenceExhausted)?;
        let item = QueuedPrompt {
            profile_id,
            prompt_id: plan.prompt_id,
            attempt_id,
            plan,
            priority,
            front,
            enqueue_sequence: sequence,
            queued_at: Utc::now(),
        };
        let insertion_index = self
            .items
            .iter()
            .position(|queued| queue_order(&item, queued) == Ordering::Less)
            .unwrap_or(self.items.len());
        self.items.insert(insertion_index, item);
        Ok(())
    }

    pub fn pop_next(&mut self) -> Option<QueuedPrompt> {
        (!self.items.is_empty()).then(|| self.items.remove(0))
    }

    pub fn cancel_queued(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Option<QueuedPrompt> {
        let index = self
            .items
            .iter()
            .position(|item| item.profile_id == profile_id && item.attempt_id == attempt_id)?;
        Some(self.items.remove(index))
    }

    pub fn position(&self, profile_id: ProfileId, attempt_id: AttemptId) -> Option<usize> {
        self.items
            .iter()
            .filter(|item| item.profile_id == profile_id)
            .position(|item| item.attempt_id == attempt_id)
    }

    pub fn reorder(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        position: usize,
    ) -> Result<(), ExecutionQueueError> {
        let current_index = self
            .items
            .iter()
            .position(|item| item.profile_id == profile_id && item.attempt_id == attempt_id)
            .ok_or(ExecutionQueueError::UnknownAttempt(attempt_id))?;
        let profile_length = self
            .items
            .iter()
            .filter(|item| item.profile_id == profile_id)
            .count();
        if position >= profile_length {
            return Err(ExecutionQueueError::InvalidPosition {
                position,
                length: profile_length,
            });
        }
        let item = self.items.remove(current_index);
        let insertion_index = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.profile_id == profile_id)
            .nth(position)
            .map_or_else(
                || {
                    self.items
                        .iter()
                        .rposition(|item| item.profile_id == profile_id)
                        .map_or(self.items.len(), |index| index + 1)
                },
                |(index, _)| index,
            );
        self.items.insert(insertion_index, item);
        Ok(())
    }

    pub fn clear_profile(&mut self, profile_id: ProfileId) -> Vec<QueuedPrompt> {
        let mut removed = Vec::new();
        self.items.retain(|item| {
            if item.profile_id == profile_id {
                removed.push(item.clone());
                false
            } else {
                true
            }
        });
        removed
    }

    pub fn profile_items(&self, profile_id: ProfileId) -> impl Iterator<Item = &QueuedPrompt> {
        self.items
            .iter()
            .filter(move |item| item.profile_id == profile_id)
    }

    pub fn items(&self) -> &[QueuedPrompt] {
        &self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

fn queue_order(left: &QueuedPrompt, right: &QueuedPrompt) -> Ordering {
    right
        .priority
        .cmp(&left.priority)
        .then_with(|| right.front.cmp(&left.front))
        .then_with(|| {
            if left.front && right.front {
                right.enqueue_sequence.cmp(&left.enqueue_sequence)
            } else {
                left.enqueue_sequence.cmp(&right.enqueue_sequence)
            }
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptState {
    Queued,
    Running,
    Cancelling,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

impl AttemptState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Interrupted
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMediaKind {
    Image,
    Animation,
    Video,
    Audio,
    ThreeD,
    Text,
    Json,
    Binary,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ExecutionOutputAvailability {
    Ready {
        reference: String,
        byte_length: u64,
    },
    Missing {
        reference: Option<String>,
        reason: String,
    },
    Expired {
        reference: Option<String>,
        expired_at: DateTime<Utc>,
        reason: String,
    },
    ExternallyDeleted {
        reference: String,
        detected_at: DateTime<Utc>,
    },
    Forbidden {
        reason: String,
    },
    Unsupported {
        media_type: String,
        reason: String,
    },
    Corrupt {
        reference: Option<String>,
        reason: String,
    },
    Removed {
        reason: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum OperationEligibility {
    Allowed,
    Unavailable { reason: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOutputOperationAction {
    Recover,
    Remove,
}

impl ExecutionOutputOperationAction {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Recover => "recovery",
            Self::Remove => "removal",
        }
    }
}

impl OperationEligibility {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionOutput {
    pub output_id: Uuid,
    pub node_id: NodeId,
    pub output_index: usize,
    pub name: String,
    pub media_kind: OutputMediaKind,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subfolder: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_type: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download_reference: Option<String>,
    pub availability: ExecutionOutputAvailability,
    pub created_at: DateTime<Utc>,
}

impl ExecutionOutput {
    pub fn recovery_eligibility(&self) -> OperationEligibility {
        match &self.availability {
            ExecutionOutputAvailability::Missing { .. }
            | ExecutionOutputAvailability::Expired { .. }
            | ExecutionOutputAvailability::ExternallyDeleted { .. }
            | ExecutionOutputAvailability::Corrupt { .. } => OperationEligibility::Allowed,
            ExecutionOutputAvailability::Ready { .. } => OperationEligibility::Unavailable {
                reason: "the output is already available".to_owned(),
            },
            ExecutionOutputAvailability::Forbidden { reason } => {
                OperationEligibility::Unavailable {
                    reason: format!("access to the output is forbidden: {reason}"),
                }
            }
            ExecutionOutputAvailability::Unsupported { reason, .. } => {
                OperationEligibility::Unavailable {
                    reason: format!("the output type is unsupported: {reason}"),
                }
            }
            ExecutionOutputAvailability::Removed { reason } => OperationEligibility::Unavailable {
                reason: format!("the output reference was removed: {reason}"),
            },
        }
    }

    pub fn removal_eligibility(&self) -> OperationEligibility {
        match &self.availability {
            ExecutionOutputAvailability::Ready { .. }
            | ExecutionOutputAvailability::Missing { .. }
            | ExecutionOutputAvailability::Expired { .. }
            | ExecutionOutputAvailability::ExternallyDeleted { .. }
            | ExecutionOutputAvailability::Corrupt { .. } => OperationEligibility::Allowed,
            ExecutionOutputAvailability::Forbidden { reason } => {
                OperationEligibility::Unavailable {
                    reason: format!("access to the output is forbidden: {reason}"),
                }
            }
            ExecutionOutputAvailability::Unsupported { reason, .. } => {
                OperationEligibility::Unavailable {
                    reason: format!("the output cannot be safely removed: {reason}"),
                }
            }
            ExecutionOutputAvailability::Removed { reason } => OperationEligibility::Unavailable {
                reason: format!("the output reference was already removed: {reason}"),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPreview {
    pub preview_id: Uuid,
    pub node_id: NodeId,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_index: Option<usize>,
    pub media_kind: OutputMediaKind,
    pub media_type: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub encoded_bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionFailureOrigin {
    Validation,
    Node,
    Transport,
    Provider,
    Decoding,
    Filesystem,
    Permission,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionFailure {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub origin: ExecutionFailureOrigin,
    pub node_id: Option<NodeId>,
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionRecoveryInterruptionReason {
    RuntimeRestart,
}

impl ExecutionRecoveryInterruptionReason {
    pub fn summary(self) -> &'static str {
        match self {
            Self::RuntimeRestart => "native runtime restarted before the attempt completed",
        }
    }
}

impl ExecutionFailure {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            origin: ExecutionFailureOrigin::Unknown,
            node_id: None,
            retryable: false,
            details: BTreeMap::new(),
        }
    }

    pub fn at_node(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }

    pub fn with_origin(mut self, origin: ExecutionFailureOrigin) -> Self {
        self.origin = origin;
        self
    }

    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttemptEventKind {
    Started,
    Progress {
        completed: u64,
        total: u64,
    },
    Preview {
        preview: ExecutionPreview,
    },
    CacheHit,
    OutputPrepared {
        transaction_id: Uuid,
    },
    OutputAvailable {
        output: ExecutionOutput,
    },
    CancelRequested {
        reason: String,
        #[serde(default)]
        interrupt: bool,
    },
    Succeeded,
    Failed {
        failure: ExecutionFailure,
    },
    Cancelled,
    Interrupted {
        reason: String,
    },
    RecoveryInterrupted {
        reason: ExecutionRecoveryInterruptionReason,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptEvent {
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub sequence: u64,
    pub node_id: Option<NodeId>,
    pub at: DateTime<Utc>,
    pub kind: AttemptEventKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryPromptSource {
    OriginalPrompt,
    CurrentWorkflow { revision: String },
    ProviderResume { operation_id: Uuid },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProviderAttemptState {
    Queued,
    Running,
    Cancelling,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
    Unknown { raw_state: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum AttemptSourceProjection {
    Provider {
        provider_id: String,
        state: ProviderAttemptState,
    },
    Unknown {
        source_id: Option<String>,
        raw_state: String,
    },
}

impl AttemptSourceProjection {
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Provider { provider_id, state } => {
                !provider_id.trim().is_empty()
                    && match state {
                        ProviderAttemptState::Unknown { raw_state } => !raw_state.trim().is_empty(),
                        ProviderAttemptState::Queued
                        | ProviderAttemptState::Running
                        | ProviderAttemptState::Cancelling
                        | ProviderAttemptState::Succeeded
                        | ProviderAttemptState::Failed
                        | ProviderAttemptState::Cancelled
                        | ProviderAttemptState::Interrupted => true,
                    }
            }
            Self::Unknown {
                source_id,
                raw_state,
            } => {
                source_id
                    .as_ref()
                    .is_none_or(|source_id| !source_id.trim().is_empty())
                    && !raw_state.trim().is_empty()
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub schema_version: u16,
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub retry_of: Option<AttemptId>,
    pub retry_source: Option<RetryPromptSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_projection: Option<AttemptSourceProjection>,
    pub state: AttemptState,
    pub last_sequence: Option<u64>,
    #[serde(default)]
    pub canonical_event_count: u64,
    pub events: Vec<AttemptEvent>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub output_availability_overrides: BTreeMap<Uuid, ExecutionOutputAvailability>,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(default, flatten)]
    pub persistence_unknown_fields: BTreeMap<String, Value>,
}

impl AttemptRecord {
    pub fn queued(profile_id: ProfileId, prompt_id: PromptId, attempt_id: AttemptId) -> Self {
        Self {
            schema_version: ATTEMPT_HISTORY_SCHEMA_VERSION,
            profile_id,
            prompt_id,
            attempt_id,
            retry_of: None,
            retry_source: None,
            source_projection: None,
            state: AttemptState::Queued,
            last_sequence: None,
            canonical_event_count: 0,
            events: Vec::new(),
            output_availability_overrides: BTreeMap::new(),
            created_at: Utc::now(),
            finished_at: None,
            persistence_unknown_fields: BTreeMap::new(),
        }
    }

    pub fn apply(&mut self, event: AttemptEvent) -> Result<(), AttemptTransitionError> {
        if event.profile_id != self.profile_id {
            return Err(AttemptTransitionError::ProfileMismatch {
                expected: self.profile_id,
                actual: event.profile_id,
            });
        }
        if event.prompt_id != self.prompt_id || event.attempt_id != self.attempt_id {
            return Err(AttemptTransitionError::IdentityMismatch);
        }
        let expected = match self.last_sequence {
            Some(sequence) => sequence
                .checked_add(1)
                .ok_or(AttemptTransitionError::SequenceExhausted)?,
            None => 0,
        };
        if event.sequence != expected {
            return Err(AttemptTransitionError::Sequence {
                expected,
                actual: event.sequence,
            });
        }
        if self.state.is_terminal() {
            return Err(AttemptTransitionError::Terminal(self.state));
        }
        validate_event_payload(&event)?;
        if let AttemptEventKind::Preview { preview } = &event.kind
            && let Some(current_revision) = self.latest_preview_revision(preview)
            && preview.revision <= current_revision
        {
            return Err(AttemptTransitionError::StalePreviewRevision {
                current: current_revision,
                received: preview.revision,
            });
        }
        let next = next_state(self.state, &event.kind)?;
        if let AttemptEventKind::OutputAvailable { output } = &event.kind {
            self.output_availability_overrides.remove(&output.output_id);
        }
        let canonical_event_count = self
            .canonical_event_count
            .checked_add(1)
            .ok_or(AttemptTransitionError::SequenceExhausted)?;
        self.retain_event(event.clone())?;
        self.state = next;
        self.last_sequence = Some(event.sequence);
        self.canonical_event_count = canonical_event_count;
        if next.is_terminal() {
            self.finished_at = Some(event.at);
        }
        Ok(())
    }

    pub fn apply_output_operation(
        &mut self,
        output_id: Uuid,
        action: ExecutionOutputOperationAction,
        availability: ExecutionOutputAvailability,
    ) -> Result<(), AttemptTransitionError> {
        if let OperationEligibility::Unavailable { reason } =
            self.output_operation_eligibility(output_id, action)?
        {
            return Err(AttemptTransitionError::OutputOperationUnavailable { action, reason });
        }
        let valid_result = match (&action, &availability) {
            (
                ExecutionOutputOperationAction::Recover,
                ExecutionOutputAvailability::Ready {
                    reference,
                    byte_length,
                },
            ) => !reference.trim().is_empty() && *byte_length > 0,
            (
                ExecutionOutputOperationAction::Remove,
                ExecutionOutputAvailability::Removed { reason },
            ) => !reason.trim().is_empty(),
            _ => false,
        };
        if !valid_result {
            return Err(AttemptTransitionError::InvalidOutputOperationResult { action });
        }
        self.output_availability_overrides
            .insert(output_id, availability);
        Ok(())
    }

    pub fn output_operation_eligibility(
        &self,
        output_id: Uuid,
        action: ExecutionOutputOperationAction,
    ) -> Result<OperationEligibility, AttemptTransitionError> {
        let output = self
            .events
            .iter()
            .rev()
            .find_map(|event| match &event.kind {
                AttemptEventKind::OutputAvailable { output } if output.output_id == output_id => {
                    Some(output)
                }
                _ => None,
            })
            .ok_or(AttemptTransitionError::UnknownOutput(output_id))?;
        let current_availability = self
            .output_availability_overrides
            .get(&output_id)
            .unwrap_or(&output.availability);
        let mut current_output = output.clone();
        current_output.availability = current_availability.clone();
        Ok(match action {
            ExecutionOutputOperationAction::Recover => current_output.recovery_eligibility(),
            ExecutionOutputOperationAction::Remove => current_output.removal_eligibility(),
        })
    }

    pub fn canonical_event_count(&self) -> u64 {
        self.canonical_event_count.max(self.events.len() as u64)
    }

    pub fn acknowledge_queued_cancellation(
        &mut self,
        at: DateTime<Utc>,
    ) -> Result<(), AttemptTransitionError> {
        if self.state != AttemptState::Queued {
            return Err(AttemptTransitionError::Illegal {
                state: self.state,
                event: "acknowledged_queued_cancellation",
            });
        }
        if self.last_sequence.is_some()
            || self.canonical_event_count != 0
            || !self.events.is_empty()
            || self.finished_at.is_some()
        {
            return Err(AttemptTransitionError::InconsistentQueuedState);
        }
        self.state = AttemptState::Cancelled;
        self.finished_at = Some(at);
        Ok(())
    }

    fn latest_preview_revision(&self, preview: &ExecutionPreview) -> Option<u64> {
        self.events
            .iter()
            .rev()
            .find_map(|event| match &event.kind {
                AttemptEventKind::Preview { preview: current }
                    if preview_identity(current) == preview_identity(preview) =>
                {
                    Some(current.revision)
                }
                _ => None,
            })
    }

    fn retain_event(&mut self, event: AttemptEvent) -> Result<(), AttemptTransitionError> {
        if let Some(index) = self
            .events
            .iter()
            .position(|current| events_share_projection(current, &event))
        {
            self.events.remove(index);
        }
        if self.events.len() >= MAX_RETAINED_ATTEMPT_EVENTS {
            let compactable = self.events.iter().position(|current| {
                matches!(
                    &current.kind,
                    AttemptEventKind::Progress { .. } | AttemptEventKind::CacheHit
                )
            });
            if let Some(index) = compactable {
                self.events.remove(index);
            } else {
                return Err(AttemptTransitionError::EventRetentionCapacity {
                    maximum: MAX_RETAINED_ATTEMPT_EVENTS,
                });
            }
        }
        self.events.push(event);
        Ok(())
    }
}

fn preview_identity(preview: &ExecutionPreview) -> (&NodeId, Option<u64>, Option<usize>) {
    (&preview.node_id, preview.frame_index, preview.output_index)
}

fn events_share_projection(current: &AttemptEvent, incoming: &AttemptEvent) -> bool {
    match (&current.kind, &incoming.kind) {
        (AttemptEventKind::Progress { .. }, AttemptEventKind::Progress { .. }) => {
            current.node_id == incoming.node_id
        }
        (
            AttemptEventKind::Preview { preview: current },
            AttemptEventKind::Preview { preview: incoming },
        ) => preview_identity(current) == preview_identity(incoming),
        (
            AttemptEventKind::OutputAvailable { output: current },
            AttemptEventKind::OutputAvailable { output: incoming },
        ) => current.output_id == incoming.output_id,
        (AttemptEventKind::CacheHit, AttemptEventKind::CacheHit) => true,
        _ => false,
    }
}

fn validate_event_payload(event: &AttemptEvent) -> Result<(), AttemptTransitionError> {
    match &event.kind {
        AttemptEventKind::Progress { completed, total } => {
            if *total == 0 || completed > total {
                return Err(AttemptTransitionError::InvalidProgress {
                    completed: *completed,
                    total: *total,
                });
            }
        }
        AttemptEventKind::Preview { preview } => {
            if event.node_id.as_ref() != Some(&preview.node_id) {
                return Err(AttemptTransitionError::NodeMismatch);
            }
            if preview.encoded_bytes.len() > sim_types::MAX_ENCODED_PREVIEW_BYTES {
                return Err(AttemptTransitionError::OversizedPreview {
                    actual: preview.encoded_bytes.len(),
                    maximum: sim_types::MAX_ENCODED_PREVIEW_BYTES,
                });
            }
        }
        AttemptEventKind::OutputAvailable { output } => {
            if event.node_id.as_ref() != Some(&output.node_id) {
                return Err(AttemptTransitionError::NodeMismatch);
            }
        }
        AttemptEventKind::Failed { failure } => {
            if failure.code.trim().is_empty() || failure.message.trim().is_empty() {
                return Err(AttemptTransitionError::InvalidFailure);
            }
            if event.node_id != failure.node_id {
                return Err(AttemptTransitionError::NodeMismatch);
            }
        }
        AttemptEventKind::Started
        | AttemptEventKind::CacheHit
        | AttemptEventKind::OutputPrepared { .. }
        | AttemptEventKind::CancelRequested { .. }
        | AttemptEventKind::Succeeded
        | AttemptEventKind::Cancelled
        | AttemptEventKind::Interrupted { .. }
        | AttemptEventKind::RecoveryInterrupted { .. } => {}
    }
    Ok(())
}

fn next_state(
    current: AttemptState,
    event: &AttemptEventKind,
) -> Result<AttemptState, AttemptTransitionError> {
    let next = match (current, event) {
        (AttemptState::Queued, AttemptEventKind::Started) => AttemptState::Running,
        (AttemptState::Queued, AttemptEventKind::Cancelled) => AttemptState::Cancelled,
        (AttemptState::Queued, AttemptEventKind::Interrupted { .. }) => AttemptState::Interrupted,
        (AttemptState::Queued, AttemptEventKind::RecoveryInterrupted { .. }) => {
            AttemptState::Interrupted
        }
        (AttemptState::Running, AttemptEventKind::Progress { .. })
        | (AttemptState::Running, AttemptEventKind::Preview { .. })
        | (AttemptState::Running, AttemptEventKind::CacheHit)
        | (AttemptState::Running, AttemptEventKind::OutputPrepared { .. })
        | (AttemptState::Running, AttemptEventKind::OutputAvailable { .. }) => {
            AttemptState::Running
        }
        (AttemptState::Running, AttemptEventKind::CancelRequested { .. }) => {
            AttemptState::Cancelling
        }
        (AttemptState::Running, AttemptEventKind::Succeeded) => AttemptState::Succeeded,
        (AttemptState::Running, AttemptEventKind::Failed { .. }) => AttemptState::Failed,
        (AttemptState::Running, AttemptEventKind::Cancelled) => AttemptState::Cancelled,
        (AttemptState::Running, AttemptEventKind::Interrupted { .. }) => AttemptState::Interrupted,
        (AttemptState::Running, AttemptEventKind::RecoveryInterrupted { .. }) => {
            AttemptState::Interrupted
        }
        (AttemptState::Cancelling, AttemptEventKind::Progress { .. })
        | (AttemptState::Cancelling, AttemptEventKind::Preview { .. }) => AttemptState::Cancelling,
        (AttemptState::Cancelling, AttemptEventKind::Cancelled) => AttemptState::Cancelled,
        (AttemptState::Cancelling, AttemptEventKind::Failed { .. }) => AttemptState::Failed,
        (AttemptState::Cancelling, AttemptEventKind::Interrupted { .. }) => {
            AttemptState::Interrupted
        }
        (AttemptState::Cancelling, AttemptEventKind::RecoveryInterrupted { .. }) => {
            AttemptState::Interrupted
        }
        _ => {
            return Err(AttemptTransitionError::Illegal {
                state: current,
                event: event_name(event),
            });
        }
    };
    Ok(next)
}

fn event_name(event: &AttemptEventKind) -> &'static str {
    match event {
        AttemptEventKind::Started => "started",
        AttemptEventKind::Progress { .. } => "progress",
        AttemptEventKind::Preview { .. } => "preview",
        AttemptEventKind::CacheHit => "cache_hit",
        AttemptEventKind::OutputPrepared { .. } => "output_prepared",
        AttemptEventKind::OutputAvailable { .. } => "output_available",
        AttemptEventKind::CancelRequested { .. } => "cancel_requested",
        AttemptEventKind::Succeeded => "succeeded",
        AttemptEventKind::Failed { .. } => "failed",
        AttemptEventKind::Cancelled => "cancelled",
        AttemptEventKind::Interrupted { .. } => "interrupted",
        AttemptEventKind::RecoveryInterrupted { .. } => "recovery_interrupted",
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AttemptTransitionError {
    #[error("attempt event profile expected {expected:?}, received {actual:?}")]
    ProfileMismatch {
        expected: ProfileId,
        actual: ProfileId,
    },
    #[error("attempt event identity does not match its record")]
    IdentityMismatch,
    #[error("attempt event sequence expected {expected}, received {actual}")]
    Sequence { expected: u64, actual: u64 },
    #[error("attempt event sequence is exhausted")]
    SequenceExhausted,
    #[error("attempt is terminal in state {0:?}")]
    Terminal(AttemptState),
    #[error("attempt progress {completed}/{total} is invalid")]
    InvalidProgress { completed: u64, total: u64 },
    #[error("attempt event node does not match its typed payload")]
    NodeMismatch,
    #[error("attempt preview is {actual} bytes, above the {maximum}-byte limit")]
    OversizedPreview { actual: usize, maximum: usize },
    #[error("attempt preview revision {received} is not newer than {current}")]
    StalePreviewRevision { current: u64, received: u64 },
    #[error("attempt event retention reached its {maximum}-event limit")]
    EventRetentionCapacity { maximum: usize },
    #[error("attempt failure must have a non-empty code and message")]
    InvalidFailure,
    #[error("queued attempt contains canonical event or terminal state data")]
    InconsistentQueuedState,
    #[error("attempt output {0} does not exist")]
    UnknownOutput(Uuid),
    #[error("attempt output {action:?} is unavailable: {reason}")]
    OutputOperationUnavailable {
        action: ExecutionOutputOperationAction,
        reason: String,
    },
    #[error("attempt output {action:?} returned an invalid canonical availability")]
    InvalidOutputOperationResult {
        action: ExecutionOutputOperationAction,
    },
    #[error("attempt event {event} is illegal in state {state:?}")]
    Illegal {
        state: AttemptState,
        event: &'static str,
    },
}

#[derive(Clone, Debug)]
pub struct ExecutionHistory {
    maximum_records: usize,
    records: VecDeque<AttemptRecord>,
}

impl ExecutionHistory {
    pub fn new(maximum_records: usize) -> Result<Self, HistoryError> {
        if maximum_records == 0 {
            return Err(HistoryError::ZeroCapacity);
        }
        Ok(Self {
            maximum_records,
            records: VecDeque::new(),
        })
    }

    pub fn insert(&mut self, record: AttemptRecord) -> Result<(), HistoryError> {
        if record.schema_version != ATTEMPT_HISTORY_SCHEMA_VERSION {
            return Err(HistoryError::UnsupportedSchema(record.schema_version));
        }
        if self.records.iter().any(|candidate| {
            candidate.profile_id == record.profile_id && candidate.attempt_id == record.attempt_id
        }) {
            return Err(HistoryError::DuplicateAttempt(record.attempt_id));
        }
        if self.records.len() >= self.maximum_records {
            let terminal_index = self
                .records
                .iter()
                .position(|record| record.state.is_terminal())
                .ok_or(HistoryError::LiveCapacityExhausted {
                    maximum: self.maximum_records,
                })?;
            self.records.remove(terminal_index);
        }
        self.records.push_back(record);
        Ok(())
    }

    pub fn apply(&mut self, event: AttemptEvent) -> Result<(), HistoryError> {
        let attempt_id = event.attempt_id;
        let profile_id = event.profile_id;
        let record_index = self
            .records
            .iter()
            .position(|record| record.profile_id == profile_id && record.attempt_id == attempt_id);
        let record_index = match record_index {
            Some(index) => index,
            None if self
                .records
                .iter()
                .any(|record| record.attempt_id == attempt_id) =>
            {
                return Err(HistoryError::ProfileMismatch {
                    profile_id,
                    attempt_id,
                });
            }
            None => return Err(HistoryError::UnknownAttempt(attempt_id)),
        };
        self.records
            .get_mut(record_index)
            .ok_or(HistoryError::UnknownAttempt(attempt_id))?
            .apply(event)
            .map_err(HistoryError::Transition)
    }

    #[cfg(test)]
    fn retry_with_attempt_id(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        source: RetryPromptSource,
        retry_attempt_id: AttemptId,
    ) -> Result<AttemptRecord, HistoryError> {
        let previous = self
            .records
            .iter()
            .find(|record| record.profile_id == profile_id && record.attempt_id == attempt_id)
            .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
        if !previous.state.is_terminal() {
            return Err(HistoryError::RetryNonterminal(attempt_id));
        }
        let mut retry = AttemptRecord::queued(profile_id, previous.prompt_id, retry_attempt_id);
        retry.retry_of = Some(attempt_id);
        retry.retry_source = Some(source);
        self.insert(retry.clone())?;
        Ok(retry)
    }

    pub fn record(&self, profile_id: ProfileId, attempt_id: AttemptId) -> Option<&AttemptRecord> {
        self.records
            .iter()
            .find(|record| record.profile_id == profile_id && record.attempt_id == attempt_id)
    }

    pub fn record_mut(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Option<&mut AttemptRecord> {
        self.records
            .iter_mut()
            .find(|record| record.profile_id == profile_id && record.attempt_id == attempt_id)
    }

    pub fn record_by_attempt(&self, attempt_id: AttemptId) -> Option<&AttemptRecord> {
        self.records
            .iter()
            .find(|record| record.attempt_id == attempt_id)
    }

    pub fn remove(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<AttemptRecord, HistoryError> {
        let index = self
            .records
            .iter()
            .position(|record| record.profile_id == profile_id && record.attempt_id == attempt_id)
            .ok_or(HistoryError::UnknownAttempt(attempt_id))?;
        let is_terminal = self
            .records
            .get(index)
            .is_some_and(|record| record.state.is_terminal());
        if !is_terminal {
            return Err(HistoryError::RemoveNonterminal(attempt_id));
        }
        self.records
            .remove(index)
            .ok_or(HistoryError::UnknownAttempt(attempt_id))
    }

    pub fn clear_terminal(&mut self, profile_id: ProfileId) -> Vec<AttemptRecord> {
        let mut removed = Vec::new();
        self.records.retain(|record| {
            if record.profile_id == profile_id && record.state.is_terminal() {
                removed.push(record.clone());
                false
            } else {
                true
            }
        });
        removed
    }

    pub fn profile_records(&self, profile_id: ProfileId) -> impl Iterator<Item = &AttemptRecord> {
        self.records
            .iter()
            .filter(move |record| record.profile_id == profile_id)
    }

    pub fn records(&self) -> &VecDeque<AttemptRecord> {
        &self.records
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum HistoryError {
    #[error("history capacity must be non-zero")]
    ZeroCapacity,
    #[error("history reached its {maximum}-record limit with only live attempts")]
    LiveCapacityExhausted { maximum: usize },
    #[error("attempt {0:?} already exists in history")]
    DuplicateAttempt(AttemptId),
    #[error("attempt {0:?} is not present in history")]
    UnknownAttempt(AttemptId),
    #[error("attempt {attempt_id:?} belongs to a different profile than {profile_id:?}")]
    ProfileMismatch {
        profile_id: ProfileId,
        attempt_id: AttemptId,
    },
    #[error("attempt {0:?} is not terminal and cannot be retried")]
    RetryNonterminal(AttemptId),
    #[error("attempt {0:?} is not terminal and cannot be removed")]
    RemoveNonterminal(AttemptId),
    #[error("attempt history schema {0} is unsupported")]
    UnsupportedSchema(u16),
    #[error(transparent)]
    Transition(AttemptTransitionError),
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum EventBusError {
    #[error("event bus capacity must be non-zero")]
    ZeroCapacity,
    #[error("one or more execution event subscribers are backpressured")]
    Backpressure,
}

#[derive(Clone)]
pub struct ExecutionEventBus {
    capacity: usize,
    subscribers: Arc<Mutex<Vec<ExecutionEventSubscriber>>>,
}

struct ExecutionEventSubscriber {
    sender: Sender<AttemptEvent>,
    receiver: Receiver<AttemptEvent>,
}

impl ExecutionEventBus {
    pub fn new(capacity: usize) -> Result<Self, EventBusError> {
        if capacity == 0 {
            return Err(EventBusError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            subscribers: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn subscribe(&self) -> Receiver<AttemptEvent> {
        let (sender, receiver) = async_channel::bounded(self.capacity);
        self.subscribers.lock().push(ExecutionEventSubscriber {
            sender,
            receiver: receiver.clone(),
        });
        receiver
    }

    pub fn publish(&self, event: AttemptEvent) -> Result<(), EventBusError> {
        let mut subscribers = self.subscribers.lock();
        subscribers.retain(|subscriber| subscriber.sender.receiver_count() > 1);
        let mut backpressured = false;
        for subscriber in subscribers.iter() {
            match subscriber.sender.try_send(event.clone()) {
                Ok(()) => {}
                Err(TrySendError::Full(event)) => {
                    backpressured = true;
                    match subscriber.receiver.try_recv() {
                        Ok(_) | Err(async_channel::TryRecvError::Empty) => {
                            match subscriber.sender.try_send(event) {
                                Ok(()) | Err(TrySendError::Closed(_)) => {}
                                Err(TrySendError::Full(_)) => {
                                    return Err(EventBusError::Backpressure);
                                }
                            }
                        }
                        Err(async_channel::TryRecvError::Closed) => {}
                    }
                }
                Err(TrySendError::Closed(_)) => {}
            }
        }
        subscribers.retain(|subscriber| subscriber.sender.receiver_count() > 1);
        if backpressured {
            Err(EventBusError::Backpressure)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CacheEntry, CacheKey, NativeCache, NativeCachePolicy, NativeEffectClass,
        NativeNodeDescriptor, NativeOutputDescriptor, NativeValue, NativeValueType,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use sim_nodes::NATIVE_NODE_CONTRACT_SCHEMA_VERSION;
    use std::collections::{BTreeMap, BTreeSet};

    fn plan(prompt: u128) -> CompiledPlan {
        let prompt_id = PromptId(Uuid::from_u128(prompt));
        let node_id = NodeId::from("out");
        CompiledPlan {
            prompt_id,
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([(
                node_id.clone(),
                crate::CompiledNode {
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

    fn event(
        prompt_id: PromptId,
        attempt_id: AttemptId,
        sequence: u64,
        kind: AttemptEventKind,
    ) -> AttemptEvent {
        AttemptEvent {
            profile_id: ProfileId(Uuid::nil()),
            prompt_id,
            attempt_id,
            sequence,
            node_id: None,
            at: Utc::now(),
            kind,
            data: None,
        }
    }

    #[test]
    fn queue_honors_priority_front_and_stable_order() -> Result<(), ExecutionQueueError> {
        let mut queue = ExecutionQueue::default();
        let profile_id = ProfileId(Uuid::nil());
        queue.enqueue(
            profile_id,
            plan(1),
            AttemptId(Uuid::from_u128(11)),
            0,
            false,
        )?;
        queue.enqueue(profile_id, plan(2), AttemptId(Uuid::from_u128(12)), 0, true)?;
        queue.enqueue(
            profile_id,
            plan(3),
            AttemptId(Uuid::from_u128(13)),
            5,
            false,
        )?;
        assert_eq!(
            queue.pop_next().map(|item| item.prompt_id),
            Some(PromptId(Uuid::from_u128(3)))
        );
        assert_eq!(
            queue.pop_next().map(|item| item.prompt_id),
            Some(PromptId(Uuid::from_u128(2)))
        );
        assert_eq!(
            queue.pop_next().map(|item| item.prompt_id),
            Some(PromptId(Uuid::from_u128(1)))
        );
        Ok(())
    }

    #[test]
    fn acknowledged_reorder_survives_later_enqueue() -> Result<(), ExecutionQueueError> {
        let profile_id = ProfileId(Uuid::nil());
        let mut queue = ExecutionQueue::default();
        for value in 1..=3 {
            queue.enqueue(
                profile_id,
                plan(value),
                AttemptId(Uuid::from_u128(value + 10)),
                0,
                false,
            )?;
        }
        queue.reorder(profile_id, AttemptId(Uuid::from_u128(13)), 0)?;
        queue.enqueue(
            profile_id,
            plan(4),
            AttemptId(Uuid::from_u128(14)),
            0,
            false,
        )?;
        assert_eq!(
            queue
                .profile_items(profile_id)
                .map(|item| item.attempt_id)
                .collect::<Vec<_>>(),
            [13_u128, 11, 12, 14].map(|value| AttemptId(Uuid::from_u128(value)))
        );
        Ok(())
    }

    #[test]
    fn terminal_attempt_rejects_late_or_duplicate_events() -> Result<(), AttemptTransitionError> {
        let prompt_id = PromptId(Uuid::from_u128(1));
        let attempt_id = AttemptId(Uuid::from_u128(2));
        let mut record = AttemptRecord::queued(ProfileId(Uuid::nil()), prompt_id, attempt_id);
        record.apply(event(prompt_id, attempt_id, 0, AttemptEventKind::Started))?;
        record.apply(event(prompt_id, attempt_id, 1, AttemptEventKind::Succeeded))?;
        assert_eq!(
            record.apply(event(
                prompt_id,
                attempt_id,
                2,
                AttemptEventKind::Progress {
                    completed: 1,
                    total: 1
                }
            )),
            Err(AttemptTransitionError::Terminal(AttemptState::Succeeded))
        );
        assert_eq!(record.events.len(), 2);
        Ok(())
    }

    #[test]
    fn cancelling_attempt_rejects_new_output_receipts() -> Result<(), AttemptTransitionError> {
        let prompt_id = PromptId(Uuid::from_u128(1));
        let attempt_id = AttemptId(Uuid::from_u128(2));
        let mut record = AttemptRecord::queued(ProfileId(Uuid::nil()), prompt_id, attempt_id);
        record.apply(event(prompt_id, attempt_id, 0, AttemptEventKind::Started))?;
        record.apply(event(
            prompt_id,
            attempt_id,
            1,
            AttemptEventKind::CancelRequested {
                reason: "user".to_owned(),
                interrupt: false,
            },
        ))?;
        assert_eq!(
            record.apply(event(
                prompt_id,
                attempt_id,
                2,
                AttemptEventKind::OutputPrepared {
                    transaction_id: Uuid::from_u128(3),
                },
            )),
            Err(AttemptTransitionError::Illegal {
                state: AttemptState::Cancelling,
                event: "output_prepared",
            })
        );
        assert_eq!(record.last_sequence, Some(1));
        assert_eq!(record.events.len(), 2);
        Ok(())
    }

    #[test]
    fn bounded_history_never_evicts_a_live_attempt() -> Result<(), HistoryError> {
        let profile_id = ProfileId(Uuid::nil());
        let mut history = ExecutionHistory::new(2)?;
        let first = AttemptId(Uuid::from_u128(1));
        let second = AttemptId(Uuid::from_u128(2));
        let third = AttemptId(Uuid::from_u128(3));
        history.insert(AttemptRecord::queued(
            profile_id,
            PromptId(Uuid::from_u128(11)),
            first,
        ))?;
        history.insert(AttemptRecord::queued(
            profile_id,
            PromptId(Uuid::from_u128(12)),
            second,
        ))?;
        assert_eq!(
            history.insert(AttemptRecord::queued(
                profile_id,
                PromptId(Uuid::from_u128(13)),
                third,
            )),
            Err(HistoryError::LiveCapacityExhausted { maximum: 2 })
        );
        assert!(history.record(profile_id, first).is_some());
        assert!(history.record(profile_id, second).is_some());

        history
            .record_mut(profile_id, first)
            .ok_or(HistoryError::UnknownAttempt(first))?
            .acknowledge_queued_cancellation(Utc::now())
            .map_err(HistoryError::Transition)?;
        history.insert(AttemptRecord::queued(
            profile_id,
            PromptId(Uuid::from_u128(13)),
            third,
        ))?;
        assert!(history.record(profile_id, first).is_none());
        assert!(history.record(profile_id, second).is_some());
        assert!(history.record(profile_id, third).is_some());
        Ok(())
    }

    #[test]
    fn failure_origin_is_backward_compatible_and_typed() -> Result<(), serde_json::Error> {
        let legacy = serde_json::json!({
            "code": "decode_failed",
            "message": "fixture",
            "node_id": null,
            "retryable": true,
            "details": {}
        });
        let failure: ExecutionFailure = serde_json::from_value(legacy)?;
        assert_eq!(failure.origin, ExecutionFailureOrigin::Unknown);
        assert_eq!(
            ExecutionFailure::new("permission_denied", "fixture")
                .with_origin(ExecutionFailureOrigin::Permission)
                .origin,
            ExecutionFailureOrigin::Permission
        );
        Ok(())
    }

    #[test]
    fn preview_revisions_are_monotonic_per_frame_output() -> Result<(), AttemptTransitionError> {
        let profile_id = ProfileId(Uuid::nil());
        let prompt_id = PromptId(Uuid::from_u128(1));
        let attempt_id = AttemptId(Uuid::from_u128(2));
        let node_id = NodeId::from("preview");
        let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
        record.apply(event(prompt_id, attempt_id, 0, AttemptEventKind::Started))?;
        let preview = |revision| ExecutionPreview {
            preview_id: Uuid::from_u128(revision as u128 + 10),
            node_id: node_id.clone(),
            revision,
            frame_index: Some(3),
            output_index: Some(1),
            media_kind: OutputMediaKind::Image,
            media_type: "image/png".to_owned(),
            width: Some(1),
            height: Some(1),
            encoded_bytes: vec![0],
        };
        let mut first = event(
            prompt_id,
            attempt_id,
            1,
            AttemptEventKind::Preview {
                preview: preview(2),
            },
        );
        first.node_id = Some(node_id.clone());
        record.apply(first)?;
        let mut stale = event(
            prompt_id,
            attempt_id,
            2,
            AttemptEventKind::Preview {
                preview: preview(1),
            },
        );
        stale.node_id = Some(node_id);
        assert_eq!(
            record.apply(stale),
            Err(AttemptTransitionError::StalePreviewRevision {
                current: 2,
                received: 1,
            })
        );
        assert_eq!(record.last_sequence, Some(1));
        assert_eq!(record.canonical_event_count(), 2);
        Ok(())
    }

    #[test]
    fn high_frequency_progress_is_retained_as_a_bounded_projection()
    -> Result<(), AttemptTransitionError> {
        let prompt_id = PromptId(Uuid::from_u128(1));
        let attempt_id = AttemptId(Uuid::from_u128(2));
        let mut record = AttemptRecord::queued(ProfileId(Uuid::nil()), prompt_id, attempt_id);
        record.apply(event(prompt_id, attempt_id, 0, AttemptEventKind::Started))?;
        for sequence in 1..=20_000 {
            record.apply(event(
                prompt_id,
                attempt_id,
                sequence,
                AttemptEventKind::Progress {
                    completed: sequence,
                    total: 20_000,
                },
            ))?;
        }
        assert_eq!(record.canonical_event_count(), 20_001);
        assert_eq!(record.events.len(), 2);
        assert_eq!(
            record.events.last().map(|event| event.sequence),
            Some(20_000)
        );
        Ok(())
    }

    #[test]
    fn retry_records_explicit_source_and_new_attempt_identity() -> Result<(), HistoryError> {
        let prompt_id = PromptId(Uuid::from_u128(1));
        let attempt_id = AttemptId(Uuid::from_u128(2));
        let profile_id = ProfileId(Uuid::nil());
        let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
        record
            .apply(event(prompt_id, attempt_id, 0, AttemptEventKind::Started))
            .map_err(HistoryError::Transition)?;
        record
            .apply(event(
                prompt_id,
                attempt_id,
                1,
                AttemptEventKind::Failed {
                    failure: ExecutionFailure::new("fixture", "fixture"),
                },
            ))
            .map_err(HistoryError::Transition)?;
        let mut history = ExecutionHistory::new(8)?;
        history.insert(record)?;
        let retry = history.retry_with_attempt_id(
            profile_id,
            attempt_id,
            RetryPromptSource::OriginalPrompt,
            AttemptId(Uuid::from_u128(3)),
        )?;
        assert_ne!(retry.attempt_id, attempt_id);
        assert_eq!(retry.prompt_id, prompt_id);
        assert_eq!(retry.retry_of, Some(attempt_id));
        assert_eq!(retry.retry_source, Some(RetryPromptSource::OriginalPrompt));
        Ok(())
    }

    #[test]
    fn event_bus_compacts_backpressure_to_the_latest_terminal_event() -> Result<(), EventBusError> {
        let bus = ExecutionEventBus::new(1)?;
        let slow_receiver = bus.subscribe();
        let healthy_receiver = bus.subscribe();
        let prompt_id = PromptId(Uuid::nil());
        let attempt_id = AttemptId(Uuid::nil());
        bus.publish(event(prompt_id, attempt_id, 0, AttemptEventKind::Started))?;
        assert_eq!(
            healthy_receiver.try_recv().map(|event| event.sequence),
            Ok(0)
        );
        assert_eq!(
            bus.publish(event(prompt_id, attempt_id, 1, AttemptEventKind::Succeeded)),
            Err(EventBusError::Backpressure)
        );
        assert_eq!(slow_receiver.try_recv().map(|event| event.sequence), Ok(1));
        assert_eq!(
            healthy_receiver.try_recv().map(|event| event.sequence),
            Ok(1)
        );
        Ok(())
    }

    #[test]
    fn val_domain_004() -> Result<(), Box<dyn std::error::Error>> {
        let mut case_results = BTreeMap::new();
        let prompt_id = PromptId(Uuid::from_u128(101));
        let attempt_id = AttemptId(Uuid::from_u128(201));
        let second_attempt = AttemptId(Uuid::from_u128(202));

        let profile_id = ProfileId(Uuid::nil());
        let prompt_plan = plan(101);
        case_results.insert(
            "prompt_identity".to_owned(),
            prompt_plan.prompt_id == prompt_id
                && prompt_plan.topological_order == [NodeId::from("out")],
        );

        let mut queue = ExecutionQueue::default();
        queue.enqueue(profile_id, plan(101), attempt_id, 0, false)?;
        queue.enqueue(profile_id, plan(102), second_attempt, 5, false)?;
        let second_position = queue.position(profile_id, second_attempt);
        let popped_attempt = queue.pop_next().map(|item| item.attempt_id);
        case_results.insert(
            "queue_priority_front".to_owned(),
            second_position == Some(0) && popped_attempt == Some(second_attempt),
        );
        let reordered_attempts = [401_u128, 402, 403];
        let mut reordered = ExecutionQueue::default();
        for (index, attempt) in reordered_attempts.into_iter().enumerate() {
            reordered.enqueue(
                profile_id,
                plan(401 + index as u128),
                AttemptId(Uuid::from_u128(attempt)),
                0,
                false,
            )?;
        }
        reordered.reorder(profile_id, AttemptId(Uuid::from_u128(403)), 0)?;
        reordered.enqueue(
            profile_id,
            plan(404),
            AttemptId(Uuid::from_u128(404)),
            0,
            false,
        )?;
        let reordered_attempts = reordered
            .profile_items(profile_id)
            .map(|item| item.attempt_id)
            .collect::<Vec<_>>();
        case_results.insert(
            "queue_reorder_stability".to_owned(),
            reordered_attempts
                == [403_u128, 401, 402, 404].map(|value| AttemptId(Uuid::from_u128(value))),
        );

        let mut first = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
        let mut second =
            AttemptRecord::queued(profile_id, PromptId(Uuid::from_u128(102)), second_attempt);
        first.apply(event(prompt_id, attempt_id, 0, AttemptEventKind::Started))?;
        second.apply(event(
            PromptId(Uuid::from_u128(102)),
            second_attempt,
            0,
            AttemptEventKind::Started,
        ))?;
        first.apply(event(
            prompt_id,
            attempt_id,
            1,
            AttemptEventKind::Progress {
                completed: 1,
                total: 2,
            },
        ))?;
        let sequence_gap = first.apply(event(
            prompt_id,
            attempt_id,
            3,
            AttemptEventKind::Progress {
                completed: 2,
                total: 2,
            },
        ));
        case_results.insert(
            "history_strict_sequence".to_owned(),
            sequence_gap
                == Err(AttemptTransitionError::Sequence {
                    expected: 2,
                    actual: 3,
                }),
        );
        second.apply(event(
            PromptId(Uuid::from_u128(102)),
            second_attempt,
            1,
            AttemptEventKind::Interrupted {
                reason: "worker lost".to_owned(),
            },
        ))?;
        first.apply(event(
            prompt_id,
            attempt_id,
            2,
            AttemptEventKind::CancelRequested {
                reason: "user".to_owned(),
                interrupt: false,
            },
        ))?;
        let post_cancel_output = first.apply(event(
            prompt_id,
            attempt_id,
            3,
            AttemptEventKind::OutputPrepared {
                transaction_id: Uuid::from_u128(203),
            },
        ));
        case_results.insert(
            "cancellation_fence".to_owned(),
            post_cancel_output
                == Err(AttemptTransitionError::Illegal {
                    state: AttemptState::Cancelling,
                    event: "output_prepared",
                })
                && first.last_sequence == Some(2),
        );
        first.apply(event(prompt_id, attempt_id, 3, AttemptEventKind::Cancelled))?;
        let late_result = first.apply(event(prompt_id, attempt_id, 4, AttemptEventKind::Succeeded));
        case_results.insert(
            "late_terminal_rejection".to_owned(),
            matches!(
                late_result,
                Err(AttemptTransitionError::Terminal(AttemptState::Cancelled))
            ) && first.last_sequence == Some(3),
        );
        case_results.insert(
            "attempt_interleaving".to_owned(),
            second.state == AttemptState::Interrupted && first.state == AttemptState::Cancelled,
        );

        let mut history = ExecutionHistory::new(8)?;
        history.insert(first)?;
        history.insert(second)?;
        let retry = history.retry_with_attempt_id(
            profile_id,
            attempt_id,
            RetryPromptSource::OriginalPrompt,
            AttemptId(Uuid::from_u128(3)),
        )?;
        case_results.insert(
            "retry_lineage".to_owned(),
            retry.attempt_id != attempt_id
                && retry.retry_of == Some(attempt_id)
                && retry.retry_source == Some(RetryPromptSource::OriginalPrompt),
        );

        let mut bounded_history = ExecutionHistory::new(1)?;
        let retained_attempt = AttemptId(Uuid::from_u128(501));
        bounded_history.insert(AttemptRecord::queued(
            profile_id,
            PromptId(Uuid::from_u128(501)),
            retained_attempt,
        ))?;
        let capacity_result = bounded_history.insert(AttemptRecord::queued(
            profile_id,
            PromptId(Uuid::from_u128(502)),
            AttemptId(Uuid::from_u128(502)),
        ));
        case_results.insert(
            "history_live_retention".to_owned(),
            capacity_result == Err(HistoryError::LiveCapacityExhausted { maximum: 1 })
                && bounded_history
                    .record(profile_id, retained_attempt)
                    .is_some(),
        );

        let key = CacheKey::from_inputs(
            "Fixture",
            "1",
            &BTreeMap::from([(
                "value".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "zed.json@1".to_owned(),
                    value: json!({"b": 2, "a": 1}),
                },
            )]),
            BTreeMap::new(),
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )?;
        let mut cache = NativeCache::new(4)?;
        cache.insert(
            key.clone(),
            CacheEntry {
                outputs: vec![NativeValue::Primitive {
                    value: sim_nodes::NativePrimitive::Integer(3),
                }],
                ui: None,
            },
        );
        let cached_outputs = cache.get(&key).map(|entry| entry.outputs);
        let invalidated = cache.invalidate_registry("registry-v2");
        case_results.insert(
            "cache_identity".to_owned(),
            cached_outputs
                == Some(vec![NativeValue::Primitive {
                    value: sim_nodes::NativePrimitive::Integer(3),
                }])
                && invalidated == 1
                && cache.get(&key).is_none(),
        );

        let bus = ExecutionEventBus::new(1)?;
        let receiver = bus.subscribe();
        bus.publish(event(
            prompt_id,
            retry.attempt_id,
            0,
            AttemptEventKind::Started,
        ))?;
        let backpressure = bus.publish(event(
            prompt_id,
            retry.attempt_id,
            1,
            AttemptEventKind::Succeeded,
        ));
        let received = receiver.try_recv();
        case_results.insert(
            "event_backpressure".to_owned(),
            backpressure == Err(EventBusError::Backpressure) && received.is_ok(),
        );

        let recovering = AttemptId(Uuid::from_u128(301));
        let mut recovered_record = AttemptRecord::queued(profile_id, prompt_id, recovering);
        recovered_record.apply(event(prompt_id, recovering, 0, AttemptEventKind::Started))?;
        recovered_record.apply(event(
            prompt_id,
            recovering,
            1,
            AttemptEventKind::RecoveryInterrupted {
                reason: ExecutionRecoveryInterruptionReason::RuntimeRestart,
            },
        ))?;
        let repeated_recovery = recovered_record.apply(event(
            prompt_id,
            recovering,
            2,
            AttemptEventKind::RecoveryInterrupted {
                reason: ExecutionRecoveryInterruptionReason::RuntimeRestart,
            },
        ));
        case_results.insert(
            "restart_reconciliation".to_owned(),
            recovered_record.state == AttemptState::Interrupted
                && matches!(
                    recovered_record.events.last().map(|event| &event.kind),
                    Some(AttemptEventKind::RecoveryInterrupted {
                        reason: ExecutionRecoveryInterruptionReason::RuntimeRestart
                    })
                )
                && matches!(
                    repeated_recovery,
                    Err(AttemptTransitionError::Terminal(AttemptState::Interrupted))
                ),
        );

        let runtime_root = include_str!("sim_runtime.rs");
        let executor_source = include_str!("executor.rs");
        let test_support_root = include_str!("../../sim_test_support/src/sim_test_support.rs");
        let native_image_recovery_source =
            include_str!("../../sim_test_support/tests/native_image_recovery.rs");
        case_results.insert(
            "owner_absence".to_owned(),
            ["pub struct ExecutionRequest", "pub struct NativeQueue"]
                .iter()
                .all(|prohibited| !runtime_root.contains(prohibited))
                && ["JournaledEffectCoordinator", "EffectCoordinatorState"]
                    .iter()
                    .all(|prohibited| !executor_source.contains(prohibited))
                && ["fn empty_request", "struct OutputTransaction"]
                    .iter()
                    .all(|prohibited| !test_support_root.contains(prohibited))
                && ["struct OutputTransaction", "struct TestOutputTransaction"]
                    .iter()
                    .all(|prohibited| !native_image_recovery_source.contains(prohibited)),
        );
        for (identifier, passed) in crate::executor::tests::val_domain_004_executor_case_results()?
        {
            if case_results.insert(identifier.to_owned(), passed).is_some() {
                return Err(
                    format!("duplicate VAL-DOMAIN-004 case identifier: {identifier}").into(),
                );
            }
        }
        for (identifier, passed) in
            crate::prompt_compiler::tests::val_domain_004_prompt_case_results()?
        {
            if case_results.insert(identifier.to_owned(), passed).is_some() {
                return Err(
                    format!("duplicate VAL-DOMAIN-004 case identifier: {identifier}").into(),
                );
            }
        }
        for (identifier, passed) in crate::cache::tests::val_domain_004_cache_case_results()? {
            if case_results.insert(identifier.to_owned(), passed).is_some() {
                return Err(
                    format!("duplicate VAL-DOMAIN-004 case identifier: {identifier}").into(),
                );
            }
        }
        case_results.insert(
            "worker_ui_wire_adapter".to_owned(),
            crate::native_execution_controller::val_domain_004_worker_ui_wire_adapter_case()?,
        );

        let failed_cases = case_results
            .iter()
            .filter_map(|(identifier, passed)| (!passed).then_some(identifier.clone()))
            .collect::<Vec<_>>();
        let passed_count = case_results.len().saturating_sub(failed_cases.len());
        let fixture_definition = format!(
            "sim-runtime-native-execution-reducers-v3\ncases={}\n",
            case_results.keys().cloned().collect::<Vec<_>>().join(",")
        );
        let cases = case_results
            .iter()
            .map(|(identifier, passed)| json!({"id": identifier, "passed": passed}))
            .collect::<Vec<_>>();
        let artifact = json!({
            "schema_version": 1,
            "validation_id": "VAL-DOMAIN-004",
            "fixture": &fixture_definition,
            "fixture_sha256": format!("{:x}", Sha256::digest(fixture_definition.as_bytes())),
            "environment": {
                "backend": "cpu",
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
            },
            "cases": cases,
            "summary": {
                "total": case_results.len(),
                "passed": passed_count,
                "failed": failed_cases.len(),
                "skipped": 0,
            },
            "failures": &failed_cases,
            "skipped": [],
        });
        let artifact_bytes = serde_json::to_vec_pretty(&artifact)?;
        assert_eq!(artifact_bytes, serde_json::to_vec_pretty(&artifact)?);
        let target_directory = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target")
            })
            .join("sim-parity");
        std::fs::create_dir_all(&target_directory)?;
        std::fs::write(target_directory.join("val-domain-004.json"), artifact_bytes)?;
        if !failed_cases.is_empty() {
            return Err(format!("VAL-DOMAIN-004 failed cases: {}", failed_cases.join(", ")).into());
        }
        Ok(())
    }
}
