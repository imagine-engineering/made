use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use collaboration_domain::{
    AggregateId, AuthenticatedPrincipalKind, AuthorizationAction, AuthorizationDecision,
    AuthorizationRequest, AuthorizationResourceKind, CommunityId, PrincipalId, TenantContext,
    authorize,
};
use collaboration_workflow::definition::{
    ConditionExpression, Schedule, WorkflowDefinition, WorkflowTrigger,
};
use evalexpr::{
    ContextWithMutableFunctions, ContextWithMutableVariables, Function, HashMapContext, Value,
    eval_boolean_with_context,
};
use serde_json::{Value as JsonValue, json};
use tokio::sync::Semaphore;
use uuid::Uuid;

use super::repository::{
    StoredWorkflowDefinition, WorkflowIdentity, WorkflowLifecycle, WorkflowProvenance,
    WorkflowRepositoryError, WorkflowRunIdentity, WorkflowRunRequest, WorkflowStoreOutcome,
    WorkflowTriggerKind,
};
use super::scheduler::{WorkflowQueueAdmission, WorkflowScheduler};

pub const EVENT_TRIGGER_SCOPE: &str = "messages:write";
pub const WORKFLOW_RUN_SCOPE: &str = "workflows:run";
pub const MAX_CONDITION_INPUT_BYTES: usize = 4_096;
pub const CONDITION_TIMEOUT: Duration = Duration::from_millis(100);
pub const MAX_SCHEDULE_CATCH_UP_MILLIS: u64 = 24 * 60 * 60 * 1_000;
pub const MAX_SCHEDULE_CATCH_UP_RUNS: usize = 16;
pub const MAX_OBSERVED_WORKER_CLOCK_SKEW_MILLIS: u64 = 2 * 60 * 1_000;

const MAX_EVENT_TEXT_BYTES: usize = 4_096;
const MAX_EMOJI_BYTES: usize = 128;
const MAX_SCHEDULE_SCAN_OCCURRENCES: usize = 1_441;
const RUN_NAMESPACE: Uuid = Uuid::from_u128(0xe4be805c_26b7_5e3a_b085_60fe96dcf5e9);
const TRIGGER_NAMESPACE: Uuid = Uuid::from_u128(0x648f2fc6_1f7e_502d_8498_be58617971af);
const STEP_NAMESPACE: Uuid = Uuid::from_u128(0xc35db43d_7ce2_5ae1_963f_66c2e3cf1652);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationEventTriggerKind {
    MessagePosted,
    ReactionAdded,
    DiffPosted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationEventTrigger {
    community_id: CommunityId,
    event_id: [u8; 32],
    kind: CollaborationEventTriggerKind,
    channel_id: AggregateId,
    author_principal_id: PrincipalId,
    author_public_key: [u8; 32],
    created_at_millis: u64,
    verified_at_millis: u64,
    text: Option<String>,
    emoji: Option<String>,
    message_id: Option<[u8; 32]>,
}

impl CollaborationEventTrigger {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        community_id: CommunityId,
        event_id: [u8; 32],
        kind: CollaborationEventTriggerKind,
        channel_id: AggregateId,
        author_principal_id: PrincipalId,
        author_public_key: [u8; 32],
        created_at_millis: u64,
        verified_at_millis: u64,
        text: Option<String>,
        emoji: Option<String>,
        message_id: Option<[u8; 32]>,
    ) -> Result<Self, WorkflowTriggerAdmissionError> {
        if event_id == [0; 32]
            || author_public_key == [0; 32]
            || created_at_millis == 0
            || verified_at_millis == 0
            || text
                .as_ref()
                .is_some_and(|text| text.len() > MAX_EVENT_TEXT_BYTES || text.contains('\0'))
            || emoji.as_ref().is_some_and(|emoji| {
                emoji.trim().is_empty()
                    || emoji.len() > MAX_EMOJI_BYTES
                    || emoji.chars().any(char::is_control)
            })
        {
            return Err(WorkflowTriggerAdmissionError::InvalidEvent);
        }
        match kind {
            CollaborationEventTriggerKind::MessagePosted
            | CollaborationEventTriggerKind::DiffPosted
                if emoji.is_some() || message_id.is_some() =>
            {
                return Err(WorkflowTriggerAdmissionError::InvalidEvent);
            }
            CollaborationEventTriggerKind::ReactionAdded
                if emoji.is_none() || message_id.is_none() =>
            {
                return Err(WorkflowTriggerAdmissionError::InvalidEvent);
            }
            _ => {}
        }
        let event = Self {
            community_id,
            event_id,
            kind,
            channel_id,
            author_principal_id,
            author_public_key,
            created_at_millis,
            verified_at_millis,
            text,
            emoji,
            message_id,
        };
        if encoded_context_bytes(&event.trigger_context())? > MAX_CONDITION_INPUT_BYTES {
            return Err(WorkflowTriggerAdmissionError::ConditionInputTooLarge);
        }
        Ok(event)
    }

    fn source_id(&self) -> String {
        format!("event:{}", hex::encode(self.event_id))
    }

    fn trigger_context(&self) -> JsonValue {
        let message_id = self.message_id.unwrap_or(self.event_id);
        json!({
            "author": hex::encode(self.author_public_key),
            "channel_id": self.channel_id.to_string(),
            "emoji": self.emoji.as_deref().unwrap_or_default(),
            "message_id": hex::encode(message_id),
            "text": self.text.as_deref().unwrap_or_default(),
            "timestamp": self.created_at_millis.to_string(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduleClock {
    database_now_millis: u64,
    worker_now_millis: u64,
}

impl ScheduleClock {
    pub fn new(
        database_now_millis: u64,
        worker_now_millis: u64,
    ) -> Result<Self, WorkflowTriggerAdmissionError> {
        if database_now_millis == 0 || worker_now_millis == 0 {
            return Err(WorkflowTriggerAdmissionError::InvalidScheduleWindow);
        }
        Ok(Self {
            database_now_millis,
            worker_now_millis,
        })
    }

    pub const fn database_now_millis(self) -> u64 {
        self.database_now_millis
    }

    pub fn worker_clock_skew_millis(self) -> u64 {
        self.database_now_millis.abs_diff(self.worker_now_millis)
    }

    pub fn worker_clock_skewed(self) -> bool {
        self.worker_clock_skew_millis() > MAX_OBSERVED_WORKER_CLOCK_SKEW_MILLIS
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduledWorkflowFire {
    scheduled_for_millis: u64,
}

impl ScheduledWorkflowFire {
    pub const fn scheduled_for_millis(self) -> u64 {
        self.scheduled_for_millis
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScheduleEvaluation {
    fires: Vec<ScheduledWorkflowFire>,
    skipped_fires: u64,
    backlog_before_window: bool,
    evaluated_through_millis: u64,
    worker_clock_skew_millis: u64,
}

impl ScheduleEvaluation {
    pub fn fires(&self) -> &[ScheduledWorkflowFire] {
        &self.fires
    }

    pub const fn skipped_fires(&self) -> u64 {
        self.skipped_fires
    }

    pub const fn backlog_before_window(&self) -> bool {
        self.backlog_before_window
    }

    pub const fn evaluated_through_millis(&self) -> u64 {
        self.evaluated_through_millis
    }

    pub const fn worker_clock_skew_millis(&self) -> u64 {
        self.worker_clock_skew_millis
    }
}

pub fn evaluate_schedule(
    definition: &WorkflowDefinition,
    previous_evaluated_through_millis: Option<u64>,
    clock: ScheduleClock,
) -> Result<ScheduleEvaluation, WorkflowTriggerAdmissionError> {
    let WorkflowTrigger::Schedule { schedule } = definition.trigger() else {
        return Err(WorkflowTriggerAdmissionError::TriggerMismatch);
    };
    if !definition.enabled() {
        return Err(WorkflowTriggerAdmissionError::InactiveDefinition);
    }
    let now = clock.database_now_millis();
    let Some(previous) = previous_evaluated_through_millis else {
        return Ok(ScheduleEvaluation {
            fires: Vec::new(),
            skipped_fires: 0,
            backlog_before_window: false,
            evaluated_through_millis: now,
            worker_clock_skew_millis: clock.worker_clock_skew_millis(),
        });
    };
    if previous > now {
        return Err(WorkflowTriggerAdmissionError::InvalidScheduleWindow);
    }
    let earliest = now.saturating_sub(MAX_SCHEDULE_CATCH_UP_MILLIS);
    let backlog_before_window = previous < earliest;
    let start = previous.max(earliest);
    let (fires, skipped_fires) = match schedule {
        Schedule::IntervalSeconds(seconds) => interval_fires(*seconds, start, now)?,
        Schedule::Cron(expression) => cron_fires(expression, start, now)?,
    };
    Ok(ScheduleEvaluation {
        fires,
        skipped_fires,
        backlog_before_window,
        evaluated_through_millis: now,
        worker_clock_skew_millis: clock.worker_clock_skew_millis(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowTriggerAdmissionStatus {
    Claimed,
    Duplicate,
    Filtered,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkflowTriggerAdmissionOutcome {
    pub status: WorkflowTriggerAdmissionStatus,
    pub run_identity: Option<WorkflowRunIdentity>,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkflowTriggerAdmissionError {
    #[error("workflow trigger source is invalid")]
    InvalidEvent,
    #[error("workflow trigger source is unauthorized")]
    UnauthorizedSource,
    #[error("workflow owner is not currently authorized to run the workflow")]
    UnauthorizedOwner,
    #[error("workflow trigger does not match the immutable definition")]
    TriggerMismatch,
    #[error("workflow definition is not the current active version")]
    InactiveDefinition,
    #[error("workflow trigger condition input exceeds its bound")]
    ConditionInputTooLarge,
    #[error("workflow trigger condition could not be evaluated")]
    ConditionEvaluation,
    #[error("workflow trigger condition evaluator is at capacity")]
    ConditionCapacityUnavailable,
    #[error("workflow trigger condition evaluation timed out")]
    ConditionTimeout,
    #[error("workflow schedule window is invalid")]
    InvalidScheduleWindow,
    #[error("workflow schedule is invalid")]
    InvalidSchedule,
    #[error(transparent)]
    Repository(#[from] WorkflowRepositoryError),
}

#[async_trait]
pub trait WorkflowRunClaimer: Send + Sync {
    async fn claim_run(
        &self,
        tenant: &TenantContext,
        request: &WorkflowRunRequest,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError>;
}

#[async_trait]
impl WorkflowRunClaimer for WorkflowScheduler {
    async fn claim_run(
        &self,
        tenant: &TenantContext,
        request: &WorkflowRunRequest,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        match self.queue_run(tenant, request).await? {
            WorkflowQueueAdmission::Queued => Ok(WorkflowStoreOutcome::Applied),
            WorkflowQueueAdmission::Duplicate => Ok(WorkflowStoreOutcome::Duplicate),
        }
    }
}

#[derive(Clone)]
pub struct WorkflowConditionEvaluator {
    permits: Arc<Semaphore>,
}

impl Default for WorkflowConditionEvaluator {
    fn default() -> Self {
        let parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self::with_parallelism(parallelism.saturating_mul(2).clamp(2, 32))
    }
}

impl WorkflowConditionEvaluator {
    pub fn with_parallelism(parallelism: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(parallelism.clamp(2, 32))),
        }
    }

    pub async fn evaluate(
        &self,
        condition: &ConditionExpression,
        trigger_context: &JsonValue,
    ) -> Result<bool, WorkflowTriggerAdmissionError> {
        if condition.as_str().len() > MAX_CONDITION_INPUT_BYTES
            || encoded_context_bytes(trigger_context)? > MAX_CONDITION_INPUT_BYTES
        {
            return Err(WorkflowTriggerAdmissionError::ConditionInputTooLarge);
        }
        let permit = Arc::clone(&self.permits)
            .try_acquire_owned()
            .map_err(|_| WorkflowTriggerAdmissionError::ConditionCapacityUnavailable)?;
        let expression = condition.as_str().to_owned();
        let context = condition_context(trigger_context)?;
        let task = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            eval_boolean_with_context(&expression, &context)
        });
        tokio::time::timeout(CONDITION_TIMEOUT, task)
            .await
            .map_err(|_| WorkflowTriggerAdmissionError::ConditionTimeout)?
            .map_err(|_| WorkflowTriggerAdmissionError::ConditionEvaluation)?
            .map_err(|_| WorkflowTriggerAdmissionError::ConditionEvaluation)
    }
}

pub struct WorkflowTriggerAdmission<R> {
    run_claimer: R,
    condition_evaluator: WorkflowConditionEvaluator,
}

impl<R> WorkflowTriggerAdmission<R>
where
    R: WorkflowRunClaimer,
{
    pub fn new(run_claimer: R) -> Self {
        Self {
            run_claimer,
            condition_evaluator: WorkflowConditionEvaluator::default(),
        }
    }

    pub fn with_condition_evaluator(
        run_claimer: R,
        condition_evaluator: WorkflowConditionEvaluator,
    ) -> Self {
        Self {
            run_claimer,
            condition_evaluator,
        }
    }

    pub async fn admit_event(
        &self,
        tenant: &TenantContext,
        definition: &StoredWorkflowDefinition,
        event: &CollaborationEventTrigger,
        source_authorization: &AuthorizationRequest<'_>,
        owner_authorization: &AuthorizationRequest<'_>,
    ) -> Result<WorkflowTriggerAdmissionOutcome, WorkflowTriggerAdmissionError> {
        validate_current_definition(tenant, definition)?;
        validate_event_authorization(
            tenant,
            definition,
            event,
            source_authorization,
            owner_authorization,
        )?;
        let trigger_context = event.trigger_context();
        if !self
            .event_matches(definition.definition.trigger(), event, &trigger_context)
            .await?
        {
            return Ok(WorkflowTriggerAdmissionOutcome {
                status: WorkflowTriggerAdmissionStatus::Filtered,
                run_identity: None,
            });
        }
        self.claim(
            tenant,
            definition,
            WorkflowTriggerKind::Event,
            &event.source_id(),
            &trigger_context,
            event.verified_at_millis,
        )
        .await
    }

    pub async fn admit_schedule_fire(
        &self,
        tenant: &TenantContext,
        definition: &StoredWorkflowDefinition,
        fire: ScheduledWorkflowFire,
        owner_authorization: &AuthorizationRequest<'_>,
    ) -> Result<WorkflowTriggerAdmissionOutcome, WorkflowTriggerAdmissionError> {
        validate_current_definition(tenant, definition)?;
        validate_owner_authorization(tenant, definition, owner_authorization)?;
        if !matches!(
            definition.definition.trigger(),
            WorkflowTrigger::Schedule { .. }
        ) {
            return Err(WorkflowTriggerAdmissionError::TriggerMismatch);
        }
        let source_id = format!("schedule:{}", fire.scheduled_for_millis);
        let trigger_context = json!({"scheduled_for_millis": fire.scheduled_for_millis});
        self.claim(
            tenant,
            definition,
            WorkflowTriggerKind::Schedule,
            &source_id,
            &trigger_context,
            fire.scheduled_for_millis,
        )
        .await
    }

    async fn event_matches(
        &self,
        trigger: &WorkflowTrigger,
        event: &CollaborationEventTrigger,
        trigger_context: &JsonValue,
    ) -> Result<bool, WorkflowTriggerAdmissionError> {
        match (trigger, event.kind) {
            (
                WorkflowTrigger::MessagePosted { condition },
                CollaborationEventTriggerKind::MessagePosted,
            )
            | (
                WorkflowTrigger::DiffPosted { condition },
                CollaborationEventTriggerKind::DiffPosted,
            ) => match condition {
                Some(condition) => {
                    self.condition_evaluator
                        .evaluate(condition, trigger_context)
                        .await
                }
                None => Ok(true),
            },
            (
                WorkflowTrigger::ReactionAdded { emoji },
                CollaborationEventTriggerKind::ReactionAdded,
            ) => Ok(emoji
                .as_ref()
                .is_none_or(|expected| event.emoji.as_deref() == Some(expected))),
            _ => Err(WorkflowTriggerAdmissionError::TriggerMismatch),
        }
    }

    async fn claim(
        &self,
        tenant: &TenantContext,
        definition: &StoredWorkflowDefinition,
        trigger_kind: WorkflowTriggerKind,
        source_id: &str,
        trigger_context: &JsonValue,
        source_observed_at_millis: u64,
    ) -> Result<WorkflowTriggerAdmissionOutcome, WorkflowTriggerAdmissionError> {
        let request = run_request(
            definition,
            trigger_kind,
            source_id,
            trigger_context.clone(),
            source_observed_at_millis,
        )?;
        let identity = request.identity;
        let outcome = self.run_claimer.claim_run(tenant, &request).await?;
        Ok(WorkflowTriggerAdmissionOutcome {
            status: match outcome {
                WorkflowStoreOutcome::Applied => WorkflowTriggerAdmissionStatus::Claimed,
                WorkflowStoreOutcome::Duplicate => WorkflowTriggerAdmissionStatus::Duplicate,
            },
            run_identity: Some(identity),
        })
    }
}

pub(super) fn validate_current_definition(
    tenant: &TenantContext,
    definition: &StoredWorkflowDefinition,
) -> Result<(), WorkflowTriggerAdmissionError> {
    if definition.identity.community_id() != tenant.community_id()
        || definition.lifecycle != WorkflowLifecycle::Active
        || !definition.definition.enabled()
        || definition.definition_version != definition.current_definition_version
    {
        return Err(WorkflowTriggerAdmissionError::InactiveDefinition);
    }
    Ok(())
}

fn validate_event_authorization(
    tenant: &TenantContext,
    definition: &StoredWorkflowDefinition,
    event: &CollaborationEventTrigger,
    source: &AuthorizationRequest<'_>,
    owner: &AuthorizationRequest<'_>,
) -> Result<(), WorkflowTriggerAdmissionError> {
    if event.community_id != tenant.community_id()
        || source.tenant.community_id() != tenant.community_id()
        || source.required_scope.as_str() != EVENT_TRIGGER_SCOPE
        || source.action != AuthorizationAction::Write
        || source.resource.community_id != tenant.community_id()
        || source.resource.kind != AuthorizationResourceKind::Channel
        || source.resource.resource_id != event.channel_id
        || source.resource.channel_id != Some(event.channel_id)
        || authorization_subject(source) != event.author_principal_id
        || authorize(source) != AuthorizationDecision::Allowed
    {
        return Err(WorkflowTriggerAdmissionError::UnauthorizedSource);
    }
    validate_owner_authorization(tenant, definition, owner)
}

pub(super) fn validate_owner_authorization(
    tenant: &TenantContext,
    definition: &StoredWorkflowDefinition,
    owner: &AuthorizationRequest<'_>,
) -> Result<(), WorkflowTriggerAdmissionError> {
    if owner.tenant.community_id() != tenant.community_id()
        || owner.required_scope.as_str() != WORKFLOW_RUN_SCOPE
        || owner.action != AuthorizationAction::Write
        || owner.resource.community_id != tenant.community_id()
        || owner.resource.kind != AuthorizationResourceKind::Workflow
        || owner.resource.resource_id.as_uuid() != definition.identity.workflow_id()
        || owner.resource.owner_principal_id != Some(definition.creator_principal_id)
        || owner.resource.channel_id.is_some()
        || authorization_subject(owner) != definition.creator_principal_id
        || authorize(owner) != AuthorizationDecision::Allowed
    {
        return Err(WorkflowTriggerAdmissionError::UnauthorizedOwner);
    }
    Ok(())
}

fn authorization_subject(request: &AuthorizationRequest<'_>) -> PrincipalId {
    match request.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => request.principal.principal_id(),
    }
}

pub(super) fn run_request(
    definition: &StoredWorkflowDefinition,
    trigger_kind: WorkflowTriggerKind,
    source_id: &str,
    trigger_context: JsonValue,
    source_observed_at_millis: u64,
) -> Result<WorkflowRunRequest, WorkflowTriggerAdmissionError> {
    let identity_bytes = trigger_identity_bytes(definition.identity, source_id);
    let trigger_operation_id = Uuid::new_v5(&TRIGGER_NAMESPACE, &identity_bytes);
    let run_id = Uuid::new_v5(&RUN_NAMESPACE, trigger_operation_id.as_bytes());
    let identity = WorkflowRunIdentity::new(definition.identity.community_id(), run_id)?;
    let step_operation_ids = definition
        .definition
        .steps()
        .iter()
        .map(|step| {
            let mut bytes = run_id.as_bytes().to_vec();
            bytes.extend_from_slice(step.id().as_bytes());
            Uuid::new_v5(&STEP_NAMESPACE, &bytes)
        })
        .collect();
    let provenance = WorkflowProvenance::new(
        match trigger_kind {
            WorkflowTriggerKind::Event => "collaboration_event",
            WorkflowTriggerKind::Schedule => "workflow_scheduler",
            WorkflowTriggerKind::Webhook => "workflow_webhook",
            WorkflowTriggerKind::Manual => {
                return Err(WorkflowTriggerAdmissionError::TriggerMismatch);
            }
        },
        source_id,
        "1",
        source_observed_at_millis,
        None,
    )?;
    Ok(WorkflowRunRequest {
        identity,
        workflow: definition.identity,
        definition_version: definition.definition_version,
        trigger_operation_id,
        trigger_kind,
        trigger_source_id: source_id.to_owned(),
        trigger_context,
        step_operation_ids,
        provenance,
        created_at_millis: source_observed_at_millis,
    })
}

fn trigger_identity_bytes(identity: WorkflowIdentity, source_id: &str) -> Vec<u8> {
    let mut bytes = identity.community_id().as_uuid().as_bytes().to_vec();
    bytes.extend_from_slice(identity.workflow_id().as_bytes());
    bytes.extend_from_slice(source_id.as_bytes());
    bytes
}

fn interval_fires(
    interval_seconds: u64,
    start_millis: u64,
    now_millis: u64,
) -> Result<(Vec<ScheduledWorkflowFire>, u64), WorkflowTriggerAdmissionError> {
    let interval_millis = interval_seconds
        .checked_mul(1_000)
        .filter(|interval| *interval >= 60_000)
        .ok_or(WorkflowTriggerAdmissionError::InvalidSchedule)?;
    let first_bucket = start_millis
        .checked_div(interval_millis)
        .and_then(|bucket| bucket.checked_add(1))
        .and_then(|bucket| bucket.checked_mul(interval_millis))
        .ok_or(WorkflowTriggerAdmissionError::InvalidScheduleWindow)?;
    if first_bucket > now_millis {
        return Ok((Vec::new(), 0));
    }
    let total = now_millis
        .saturating_sub(first_bucket)
        .checked_div(interval_millis)
        .and_then(|count| count.checked_add(1))
        .ok_or(WorkflowTriggerAdmissionError::InvalidScheduleWindow)?;
    let retained = total.min(MAX_SCHEDULE_CATCH_UP_RUNS as u64);
    let skipped = total.saturating_sub(retained);
    let first_retained = first_bucket
        .checked_add(
            skipped
                .checked_mul(interval_millis)
                .ok_or(WorkflowTriggerAdmissionError::InvalidScheduleWindow)?,
        )
        .ok_or(WorkflowTriggerAdmissionError::InvalidScheduleWindow)?;
    let mut fires = Vec::with_capacity(retained as usize);
    for index in 0..retained {
        fires.push(ScheduledWorkflowFire {
            scheduled_for_millis: first_retained
                .checked_add(index * interval_millis)
                .ok_or(WorkflowTriggerAdmissionError::InvalidScheduleWindow)?,
        });
    }
    Ok((fires, skipped))
}

fn cron_fires(
    expression: &str,
    start_millis: u64,
    now_millis: u64,
) -> Result<(Vec<ScheduledWorkflowFire>, u64), WorkflowTriggerAdmissionError> {
    let normalized = normalize_cron(expression)?;
    let schedule = normalized
        .parse::<cron::Schedule>()
        .map_err(|_| WorkflowTriggerAdmissionError::InvalidSchedule)?;
    let start = date_time(start_millis)?;
    let now = date_time(now_millis)?;
    let mut probe = schedule.after(&start);
    if let (Some(first), Some(second)) = (probe.next(), probe.next())
        && second
            .timestamp_millis()
            .saturating_sub(first.timestamp_millis())
            < 60_000
    {
        return Err(WorkflowTriggerAdmissionError::InvalidSchedule);
    }
    let mut fires = VecDeque::with_capacity(MAX_SCHEDULE_CATCH_UP_RUNS);
    let mut skipped = 0_u64;
    let mut scanned = 0_usize;
    for instant in schedule.after(&start) {
        if instant > now {
            break;
        }
        scanned = scanned.saturating_add(1);
        if scanned > MAX_SCHEDULE_SCAN_OCCURRENCES {
            return Err(WorkflowTriggerAdmissionError::InvalidSchedule);
        }
        push_bounded_fire(&mut fires, &mut skipped, instant.timestamp_millis())?;
    }
    Ok((fires.into_iter().collect(), skipped))
}

fn push_bounded_fire(
    fires: &mut VecDeque<ScheduledWorkflowFire>,
    skipped: &mut u64,
    timestamp_millis: i64,
) -> Result<(), WorkflowTriggerAdmissionError> {
    let scheduled_for_millis = u64::try_from(timestamp_millis)
        .map_err(|_| WorkflowTriggerAdmissionError::InvalidScheduleWindow)?;
    if fires.len() == MAX_SCHEDULE_CATCH_UP_RUNS {
        fires.pop_front();
        *skipped = skipped.saturating_add(1);
    }
    fires.push_back(ScheduledWorkflowFire {
        scheduled_for_millis,
    });
    Ok(())
}

fn normalize_cron(expression: &str) -> Result<String, WorkflowTriggerAdmissionError> {
    match expression.split_whitespace().count() {
        5 => Ok(format!("0 {expression} *")),
        6 => Ok(format!("{expression} *")),
        7 => Ok(expression.to_owned()),
        _ => Err(WorkflowTriggerAdmissionError::InvalidSchedule),
    }
}

fn date_time(timestamp_millis: u64) -> Result<DateTime<Utc>, WorkflowTriggerAdmissionError> {
    let timestamp_millis = i64::try_from(timestamp_millis)
        .map_err(|_| WorkflowTriggerAdmissionError::InvalidScheduleWindow)?;
    DateTime::from_timestamp_millis(timestamp_millis)
        .ok_or(WorkflowTriggerAdmissionError::InvalidScheduleWindow)
}

fn encoded_context_bytes(context: &JsonValue) -> Result<usize, WorkflowTriggerAdmissionError> {
    serde_json::to_vec(context)
        .map(|encoded| encoded.len())
        .map_err(|_| WorkflowTriggerAdmissionError::ConditionEvaluation)
}

fn condition_context(
    trigger_context: &JsonValue,
) -> Result<HashMapContext, WorkflowTriggerAdmissionError> {
    let object = trigger_context
        .as_object()
        .ok_or(WorkflowTriggerAdmissionError::ConditionEvaluation)?;
    let mut context = HashMapContext::new();
    install_string_functions(&mut context)?;
    for field in [
        "author",
        "channel_id",
        "emoji",
        "message_id",
        "text",
        "timestamp",
    ] {
        let value = object
            .get(field)
            .and_then(JsonValue::as_str)
            .ok_or(WorkflowTriggerAdmissionError::ConditionEvaluation)?;
        context
            .set_value(format!("trigger_{field}"), Value::String(value.to_owned()))
            .map_err(|_| WorkflowTriggerAdmissionError::ConditionEvaluation)?;
    }
    Ok(context)
}

fn install_string_functions(
    context: &mut HashMapContext,
) -> Result<(), WorkflowTriggerAdmissionError> {
    let functions: BTreeMap<&str, Function> = BTreeMap::from([
        (
            "str_contains",
            Function::new(|arguments| {
                let arguments = arguments.as_fixed_len_tuple(2)?;
                let haystack = arguments[0].as_string()?;
                let needle = arguments[1].as_string()?;
                Ok(Value::Boolean(haystack.contains(needle.as_str())))
            }),
        ),
        (
            "str_starts_with",
            Function::new(|arguments| {
                let arguments = arguments.as_fixed_len_tuple(2)?;
                let value = arguments[0].as_string()?;
                let prefix = arguments[1].as_string()?;
                Ok(Value::Boolean(value.starts_with(prefix.as_str())))
            }),
        ),
        (
            "str_ends_with",
            Function::new(|arguments| {
                let arguments = arguments.as_fixed_len_tuple(2)?;
                let value = arguments[0].as_string()?;
                let suffix = arguments[1].as_string()?;
                Ok(Value::Boolean(value.ends_with(suffix.as_str())))
            }),
        ),
        (
            "str_len",
            Function::new(|argument| Ok(Value::Int(argument.as_string()?.len() as i64))),
        ),
    ]);
    for (name, function) in functions {
        context
            .set_function(name.into(), function)
            .map_err(|_| WorkflowTriggerAdmissionError::ConditionEvaluation)?;
    }
    Ok(())
}
