use std::{error::Error, fmt, sync::Arc};

use gpui::{App, IntoElement, RenderOnce, Role, Window};
use ui::{Button, ButtonStyle, Color, Label, LabelSize, TintColor, prelude::*};

const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_LABEL_BYTES: usize = 512;
const MAX_APPROVAL_MESSAGE_BYTES: usize = 16 * 1024;
const MAX_STEPS: usize = 256;
const MAX_APPROVALS: usize = 256;
const MAX_FAILURE_CODE_BYTES: usize = 64;

pub type WorkflowTaskActionHandler = Arc<dyn Fn(WorkflowTaskAction, &mut Window, &mut App)>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowDefinitionLifecycle {
    Draft,
    Active,
    Disabled,
    Archived,
}

impl WorkflowDefinitionLifecycle {
    const fn label(self) -> &'static str {
        match self {
            Self::Draft => "Draft",
            Self::Active => "Active",
            Self::Disabled => "Disabled",
            Self::Archived => "Archived",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowRunPresentationState {
    Claimed,
    Queued,
    Running,
    WaitingApproval,
    RetryScheduled,
    RepairRequired,
    Completed,
    Failed,
    Cancelled,
}

impl WorkflowRunPresentationState {
    const fn label(self) -> &'static str {
        match self {
            Self::Claimed => "Claimed",
            Self::Queued => "Queued",
            Self::Running => "Running",
            Self::WaitingApproval => "Waiting for approval",
            Self::RetryScheduled => "Retry scheduled",
            Self::RepairRequired => "Repair required",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }

    const fn tone(self) -> WorkflowTaskTone {
        match self {
            Self::Claimed | Self::Queued | Self::Running | Self::RetryScheduled => {
                WorkflowTaskTone::Progress
            }
            Self::WaitingApproval | Self::RepairRequired => WorkflowTaskTone::Attention,
            Self::Completed => WorkflowTaskTone::Success,
            Self::Failed => WorkflowTaskTone::Error,
            Self::Cancelled => WorkflowTaskTone::Muted,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowStepPresentationState {
    Pending,
    Running,
    WaitingApproval,
    RetryScheduled,
    RepairRequired,
    Completed,
    Skipped,
    Failed,
    Cancelled,
}

impl WorkflowStepPresentationState {
    const fn label(self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Running => "Running",
            Self::WaitingApproval => "Waiting for approval",
            Self::RetryScheduled => "Retry scheduled",
            Self::RepairRequired => "Repair required",
            Self::Completed => "Completed",
            Self::Skipped => "Skipped",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }

    const fn tone(self) -> WorkflowTaskTone {
        match self {
            Self::Pending | Self::Skipped | Self::Cancelled => WorkflowTaskTone::Muted,
            Self::Running | Self::RetryScheduled => WorkflowTaskTone::Progress,
            Self::WaitingApproval | Self::RepairRequired => WorkflowTaskTone::Attention,
            Self::Completed => WorkflowTaskTone::Success,
            Self::Failed => WorkflowTaskTone::Error,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowApprovalPresentationState {
    Pending,
    Granted,
    Denied,
    Expired,
    Cancelled,
}

impl WorkflowApprovalPresentationState {
    const fn label(self) -> &'static str {
        match self {
            Self::Pending => "Approval required",
            Self::Granted => "Granted",
            Self::Denied => "Denied",
            Self::Expired => "Expired",
            Self::Cancelled => "Cancelled",
        }
    }

    const fn tone(self) -> WorkflowTaskTone {
        match self {
            Self::Pending => WorkflowTaskTone::Attention,
            Self::Granted => WorkflowTaskTone::Success,
            Self::Denied => WorkflowTaskTone::Error,
            Self::Expired | Self::Cancelled => WorkflowTaskTone::Muted,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowTaskTone {
    Progress,
    Success,
    Attention,
    Error,
    Muted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowDefinitionPresentation {
    community_id: String,
    workflow_id: String,
    name: String,
    definition_version: u64,
    head_revision: u64,
    lifecycle: WorkflowDefinitionLifecycle,
    scope_label: String,
}

impl WorkflowDefinitionPresentation {
    pub fn new(
        community_id: impl Into<String>,
        workflow_id: impl Into<String>,
        name: impl Into<String>,
        definition_version: u64,
        head_revision: u64,
        lifecycle: WorkflowDefinitionLifecycle,
        scope_label: impl Into<String>,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        let definition = Self {
            community_id: community_id.into(),
            workflow_id: workflow_id.into(),
            name: name.into(),
            definition_version,
            head_revision,
            lifecycle,
            scope_label: scope_label.into(),
        };
        validate_identifier(&definition.community_id)?;
        validate_identifier(&definition.workflow_id)?;
        validate_label(&definition.name)?;
        validate_label(&definition.scope_label)?;
        if definition_version == 0 || head_revision == 0 {
            return Err(WorkflowTaskPresentationError::InvalidVersion);
        }
        Ok(definition)
    }

    pub fn workflow_id(&self) -> &str {
        &self.workflow_id
    }

    pub fn community_id(&self) -> &str {
        &self.community_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowRetryPresentation {
    attempt_number: u16,
    due_at_millis: u64,
}

impl WorkflowRetryPresentation {
    pub fn new(
        attempt_number: u16,
        due_at_millis: u64,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        if attempt_number < 2 || due_at_millis == 0 {
            return Err(WorkflowTaskPresentationError::InvalidRetry);
        }
        Ok(Self {
            attempt_number,
            due_at_millis,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowFailureScope {
    Definition,
    Run,
    Step(u16),
    Approval(u16),
    Service,
}

impl WorkflowFailureScope {
    fn label(self) -> String {
        match self {
            Self::Definition => "workflow definition".into(),
            Self::Run => "workflow run".into(),
            Self::Step(index) => format!("step {}", u32::from(index) + 1),
            Self::Approval(index) => {
                format!("approval for step {}", u32::from(index) + 1)
            }
            Self::Service => "workflow service".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowFailureKind {
    PermissionDenied,
    TemporaryUnavailable,
    TimedOut,
    RateLimited,
    AmbiguousDelivery,
    Permanent,
    Unknown,
}

impl WorkflowFailureKind {
    const fn summary(self) -> &'static str {
        match self {
            Self::PermissionDenied => "Permission was denied. This run will not retry.",
            Self::TemporaryUnavailable => "A required service is temporarily unavailable.",
            Self::TimedOut => "The workflow action timed out.",
            Self::RateLimited => "The workflow action was rate limited.",
            Self::AmbiguousDelivery => {
                "Delivery could not be confirmed. Review is required before continuing."
            }
            Self::Permanent => "The workflow action failed and will not retry.",
            Self::Unknown => "The workflow run failed. Private error details were hidden.",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowTrustworthyCheckpoint {
    label: String,
    version: u64,
}

impl WorkflowTrustworthyCheckpoint {
    pub fn new(
        label: impl Into<String>,
        version: u64,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        let checkpoint = Self {
            label: label.into(),
            version,
        };
        validate_label(&checkpoint.label)?;
        if version == 0 {
            return Err(WorkflowTaskPresentationError::InvalidVersion);
        }
        Ok(checkpoint)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowFailurePresentation {
    kind: WorkflowFailureKind,
    code: String,
    scope: WorkflowFailureScope,
    last_trustworthy: WorkflowTrustworthyCheckpoint,
    retry_projection: bool,
}

impl WorkflowFailurePresentation {
    pub fn from_service_error(
        code: impl AsRef<str>,
        _untrusted_message: impl AsRef<str>,
        scope: WorkflowFailureScope,
        last_trustworthy: WorkflowTrustworthyCheckpoint,
        retry_projection: bool,
    ) -> Self {
        // Service messages can contain credentials or private event content, so the native
        // surface classifies only a bounded stable code and never retains the message.
        let code = redact_failure_code(code.as_ref());
        let kind = failure_kind(&code);
        Self {
            kind,
            code,
            scope,
            last_trustworthy,
            retry_projection,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowStepPresentation {
    index: u16,
    step_id: String,
    operation_id: String,
    state: WorkflowStepPresentationState,
    attempt_count: u16,
    retry: Option<WorkflowRetryPresentation>,
    failure: Option<WorkflowFailurePresentation>,
}

impl WorkflowStepPresentation {
    pub fn new(
        index: u16,
        step_id: impl Into<String>,
        operation_id: impl Into<String>,
        state: WorkflowStepPresentationState,
        attempt_count: u16,
        retry: Option<WorkflowRetryPresentation>,
        failure: Option<WorkflowFailurePresentation>,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        let step = Self {
            index,
            step_id: step_id.into(),
            operation_id: operation_id.into(),
            state,
            attempt_count,
            retry,
            failure,
        };
        validate_identifier(&step.step_id)?;
        validate_identifier(&step.operation_id)?;
        if step.retry.is_some() != (state == WorkflowStepPresentationState::RetryScheduled) {
            return Err(WorkflowTaskPresentationError::InvalidRetry);
        }
        if step.failure.is_some()
            && !matches!(
                state,
                WorkflowStepPresentationState::RetryScheduled
                    | WorkflowStepPresentationState::RepairRequired
                    | WorkflowStepPresentationState::Failed
            )
        {
            return Err(WorkflowTaskPresentationError::InvalidFailure);
        }
        Ok(step)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowApprovalPresentation {
    approval_id: String,
    step_index: u16,
    request_message: String,
    eligibility_label: String,
    state: WorkflowApprovalPresentationState,
    expires_at_millis: u64,
    viewer_may_decide: bool,
}

impl WorkflowApprovalPresentation {
    pub fn new(
        approval_id: impl Into<String>,
        step_index: u16,
        request_message: impl Into<String>,
        eligibility_label: impl Into<String>,
        state: WorkflowApprovalPresentationState,
        expires_at_millis: u64,
        viewer_may_decide: bool,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        let approval = Self {
            approval_id: approval_id.into(),
            step_index,
            request_message: request_message.into(),
            eligibility_label: eligibility_label.into(),
            state,
            expires_at_millis,
            viewer_may_decide,
        };
        validate_identifier(&approval.approval_id)?;
        validate_bounded_text(&approval.request_message, MAX_APPROVAL_MESSAGE_BYTES)?;
        validate_label(&approval.eligibility_label)?;
        if expires_at_millis == 0
            || (viewer_may_decide && state != WorkflowApprovalPresentationState::Pending)
        {
            return Err(WorkflowTaskPresentationError::InvalidApproval);
        }
        Ok(approval)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowRunPresentation {
    run_id: String,
    definition_version: u64,
    run_version: u64,
    state: WorkflowRunPresentationState,
    current_step_index: u16,
    steps: Vec<WorkflowStepPresentation>,
    approvals: Vec<WorkflowApprovalPresentation>,
    failure: Option<WorkflowFailurePresentation>,
    updated_at_millis: u64,
}

impl WorkflowRunPresentation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: impl Into<String>,
        definition_version: u64,
        run_version: u64,
        state: WorkflowRunPresentationState,
        current_step_index: u16,
        steps: Vec<WorkflowStepPresentation>,
        approvals: Vec<WorkflowApprovalPresentation>,
        failure: Option<WorkflowFailurePresentation>,
        updated_at_millis: u64,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        let run = Self {
            run_id: run_id.into(),
            definition_version,
            run_version,
            state,
            current_step_index,
            steps,
            approvals,
            failure,
            updated_at_millis,
        };
        run.validate()?;
        Ok(run)
    }

    fn validate(&self) -> Result<(), WorkflowTaskPresentationError> {
        validate_identifier(&self.run_id)?;
        if self.definition_version == 0 || self.run_version == 0 || self.updated_at_millis == 0 {
            return Err(WorkflowTaskPresentationError::InvalidVersion);
        }
        if self.steps.is_empty()
            || self.steps.len() > MAX_STEPS
            || usize::from(self.current_step_index) >= self.steps.len()
            || self
                .steps
                .iter()
                .enumerate()
                .any(|(index, step)| usize::from(step.index) != index)
            || self.steps.iter().enumerate().any(|(index, step)| {
                self.steps[index + 1..].iter().any(|other| {
                    step.step_id == other.step_id || step.operation_id == other.operation_id
                })
            })
        {
            return Err(WorkflowTaskPresentationError::InvalidSteps);
        }
        if self.approvals.len() > MAX_APPROVALS
            || self
                .approvals
                .iter()
                .any(|approval| usize::from(approval.step_index) >= self.steps.len())
            || self.approvals.iter().enumerate().any(|(index, approval)| {
                self.approvals[index + 1..]
                    .iter()
                    .any(|other| approval.approval_id == other.approval_id)
            })
        {
            return Err(WorkflowTaskPresentationError::InvalidApproval);
        }
        let pending_approvals = self
            .approvals
            .iter()
            .filter(|approval| approval.state == WorkflowApprovalPresentationState::Pending)
            .collect::<Vec<_>>();
        if self.state == WorkflowRunPresentationState::WaitingApproval {
            if pending_approvals.len() != 1
                || pending_approvals.iter().any(|approval| {
                    approval.step_index != self.current_step_index
                        || self.steps[usize::from(approval.step_index)].state
                            != WorkflowStepPresentationState::WaitingApproval
                })
            {
                return Err(WorkflowTaskPresentationError::InvalidApproval);
            }
        } else if !pending_approvals.is_empty() {
            return Err(WorkflowTaskPresentationError::InvalidApproval);
        }
        if self.failure.is_some()
            && !matches!(
                self.state,
                WorkflowRunPresentationState::RetryScheduled
                    | WorkflowRunPresentationState::RepairRequired
                    | WorkflowRunPresentationState::Failed
            )
        {
            return Err(WorkflowTaskPresentationError::InvalidFailure);
        }
        if matches!(
            self.state,
            WorkflowRunPresentationState::RepairRequired | WorkflowRunPresentationState::Failed
        ) && self.failure.is_none()
        {
            return Err(WorkflowTaskPresentationError::InvalidFailure);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowServiceUnavailablePresentation {
    community_id: Option<String>,
    scope: WorkflowFailureScope,
    last_trustworthy: Option<WorkflowTrustworthyCheckpoint>,
}

impl WorkflowServiceUnavailablePresentation {
    pub fn new(
        community_id: Option<String>,
        scope: WorkflowFailureScope,
        last_trustworthy: Option<WorkflowTrustworthyCheckpoint>,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        if let Some(community_id) = community_id.as_deref() {
            validate_identifier(community_id)?;
        }
        Ok(Self {
            community_id,
            scope,
            last_trustworthy,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkflowTaskProjection {
    Available {
        definition: WorkflowDefinitionPresentation,
        run: WorkflowRunPresentation,
    },
    Unavailable(WorkflowServiceUnavailablePresentation),
}

impl WorkflowTaskProjection {
    pub fn available(
        definition: WorkflowDefinitionPresentation,
        run: WorkflowRunPresentation,
    ) -> Result<Self, WorkflowTaskPresentationError> {
        if definition.definition_version != run.definition_version {
            return Err(WorkflowTaskPresentationError::DefinitionVersionMismatch);
        }
        Ok(Self::Available { definition, run })
    }

    pub fn unavailable(unavailable: WorkflowServiceUnavailablePresentation) -> Self {
        Self::Unavailable(unavailable)
    }

    pub fn presentation(&self) -> WorkflowTaskSurfacePresentation {
        match self {
            Self::Available { definition, run } => WorkflowTaskSurfacePresentation {
                headline: definition.name.clone(),
                definition_summary: format!(
                    "Definition v{} · head {} · {} · {}",
                    definition.definition_version,
                    definition.head_revision,
                    definition.lifecycle.label(),
                    definition.scope_label
                ),
                state_label: run.state.label(),
                tone: run.state.tone(),
                run_summary: Some(format!(
                    "Run {} · checkpoint {} · updated {}",
                    run.run_id, run.run_version, run.updated_at_millis
                )),
                steps: run
                    .steps
                    .iter()
                    .map(|step| WorkflowStepSurfacePresentation {
                        index: step.index,
                        label: step.step_id.clone(),
                        state_label: step.state.label(),
                        tone: step.state.tone(),
                        attempt_label: format!("{} attempt(s)", step.attempt_count),
                        retry_label: step.retry.as_ref().map(|retry| {
                            format!(
                                "Retry attempt {} scheduled for {}",
                                retry.attempt_number, retry.due_at_millis
                            )
                        }),
                        failure: step.failure.as_ref().map(failure_presentation),
                    })
                    .collect(),
                approvals: run
                    .approvals
                    .iter()
                    .map(|approval| WorkflowApprovalSurfacePresentation {
                        approval_id: approval.approval_id.clone(),
                        step_index: approval.step_index,
                        state_label: approval.state.label(),
                        tone: approval.state.tone(),
                        request_message: approval.request_message.clone(),
                        eligibility_label: approval.eligibility_label.clone(),
                        expires_at_millis: approval.expires_at_millis,
                    })
                    .collect(),
                failure: run.failure.as_ref().map(failure_presentation),
                last_trustworthy: None,
            },
            Self::Unavailable(unavailable) => WorkflowTaskSurfacePresentation {
                headline: "Workflows unavailable".into(),
                definition_summary: "The native task surface cannot refresh workflow state.".into(),
                state_label: "Unavailable",
                tone: WorkflowTaskTone::Error,
                run_summary: None,
                steps: Vec::new(),
                approvals: Vec::new(),
                failure: Some(WorkflowFailureSurfacePresentation {
                    code: "service_unavailable".into(),
                    summary: "The workflow service is temporarily unavailable.",
                    scope_label: unavailable.scope.label(),
                    last_trustworthy_label: unavailable
                        .last_trustworthy
                        .as_ref()
                        .map(|checkpoint| checkpoint.label.clone()),
                }),
                last_trustworthy: unavailable
                    .last_trustworthy
                    .as_ref()
                    .map(|checkpoint| checkpoint.label.clone()),
            },
        }
    }

    pub fn action_requests(&self) -> Vec<WorkflowTaskAction> {
        match self {
            Self::Available { definition, run } => {
                let mut actions = run
                    .approvals
                    .iter()
                    .filter(|approval| {
                        approval.state == WorkflowApprovalPresentationState::Pending
                            && approval.viewer_may_decide
                    })
                    .flat_map(|approval| {
                        [
                            WorkflowTaskAction::GrantApproval {
                                community_id: definition.community_id.clone(),
                                workflow_id: definition.workflow_id.clone(),
                                run_id: run.run_id.clone(),
                                approval_id: approval.approval_id.clone(),
                                expected_run_version: run.run_version,
                            },
                            WorkflowTaskAction::DenyApproval {
                                community_id: definition.community_id.clone(),
                                workflow_id: definition.workflow_id.clone(),
                                run_id: run.run_id.clone(),
                                approval_id: approval.approval_id.clone(),
                                expected_run_version: run.run_version,
                            },
                        ]
                    })
                    .collect::<Vec<_>>();
                if run
                    .failure
                    .as_ref()
                    .is_some_and(|failure| failure.retry_projection)
                {
                    actions.push(WorkflowTaskAction::RetryProjection {
                        community_id: Some(definition.community_id.clone()),
                        workflow_id: Some(definition.workflow_id.clone()),
                        run_id: Some(run.run_id.clone()),
                        scope: run
                            .failure
                            .as_ref()
                            .map_or(WorkflowFailureScope::Run, |failure| failure.scope),
                        last_trustworthy_version: Some(run.run_version),
                    });
                }
                actions
            }
            Self::Unavailable(unavailable) => vec![WorkflowTaskAction::RetryProjection {
                community_id: unavailable.community_id.clone(),
                workflow_id: None,
                run_id: None,
                scope: unavailable.scope,
                last_trustworthy_version: unavailable
                    .last_trustworthy
                    .as_ref()
                    .map(|checkpoint| checkpoint.version),
            }],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowTaskSurfacePresentation {
    pub headline: String,
    pub definition_summary: String,
    pub state_label: &'static str,
    pub tone: WorkflowTaskTone,
    pub run_summary: Option<String>,
    pub steps: Vec<WorkflowStepSurfacePresentation>,
    pub approvals: Vec<WorkflowApprovalSurfacePresentation>,
    pub failure: Option<WorkflowFailureSurfacePresentation>,
    pub last_trustworthy: Option<String>,
}

impl WorkflowTaskSurfacePresentation {
    pub fn accessibility_label(&self) -> String {
        let mut label = format!(
            "{}. {}. {}",
            self.headline, self.definition_summary, self.state_label
        );
        if !self.approvals.is_empty() {
            label.push_str(&format!(". {} approval request(s)", self.approvals.len()));
        }
        if let Some(failure) = &self.failure {
            label.push_str(&format!(
                ". Affected scope: {}. {}",
                failure.scope_label, failure.summary
            ));
        }
        label
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowStepSurfacePresentation {
    pub index: u16,
    pub label: String,
    pub state_label: &'static str,
    pub tone: WorkflowTaskTone,
    pub attempt_label: String,
    pub retry_label: Option<String>,
    pub failure: Option<WorkflowFailureSurfacePresentation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowApprovalSurfacePresentation {
    pub approval_id: String,
    pub step_index: u16,
    pub state_label: &'static str,
    pub tone: WorkflowTaskTone,
    pub request_message: String,
    pub eligibility_label: String,
    pub expires_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowFailureSurfacePresentation {
    pub code: String,
    pub summary: &'static str,
    pub scope_label: String,
    pub last_trustworthy_label: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowTaskActionKind {
    GrantApproval,
    DenyApproval,
    RetryProjection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkflowTaskAction {
    GrantApproval {
        community_id: String,
        workflow_id: String,
        run_id: String,
        approval_id: String,
        expected_run_version: u64,
    },
    DenyApproval {
        community_id: String,
        workflow_id: String,
        run_id: String,
        approval_id: String,
        expected_run_version: u64,
    },
    RetryProjection {
        community_id: Option<String>,
        workflow_id: Option<String>,
        run_id: Option<String>,
        scope: WorkflowFailureScope,
        last_trustworthy_version: Option<u64>,
    },
}

impl WorkflowTaskAction {
    pub const fn kind(&self) -> WorkflowTaskActionKind {
        match self {
            Self::GrantApproval { .. } => WorkflowTaskActionKind::GrantApproval,
            Self::DenyApproval { .. } => WorkflowTaskActionKind::DenyApproval,
            Self::RetryProjection { .. } => WorkflowTaskActionKind::RetryProjection,
        }
    }

    const fn label(&self) -> &'static str {
        match self {
            Self::GrantApproval { .. } => "Grant",
            Self::DenyApproval { .. } => "Deny",
            Self::RetryProjection { .. } => "Retry",
        }
    }
}

#[derive(IntoElement)]
pub struct WorkflowTaskSurface {
    index: usize,
    projection: WorkflowTaskProjection,
    on_action: Option<WorkflowTaskActionHandler>,
}

impl WorkflowTaskSurface {
    pub fn new(index: usize, projection: WorkflowTaskProjection) -> Self {
        Self {
            index,
            projection,
            on_action: None,
        }
    }

    pub fn on_action(mut self, handler: WorkflowTaskActionHandler) -> Self {
        self.on_action = Some(handler);
        self
    }
}

impl RenderOnce for WorkflowTaskSurface {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let presentation = self.projection.presentation();
        let action_handler = self.on_action;
        let actions = action_handler
            .is_some()
            .then(|| self.projection.action_requests())
            .unwrap_or_default();
        let background = match presentation.tone {
            WorkflowTaskTone::Attention => cx.theme().status().warning_background,
            WorkflowTaskTone::Error => cx.theme().status().error_background,
            _ => cx.theme().colors().editor_background,
        };

        v_flex()
            .id(("workflow-task-surface", self.index))
            .debug_selector(|| "WORKFLOW-TASK-SURFACE".to_owned())
            .role(Role::ListItem)
            .aria_label(presentation.accessibility_label())
            .w_full()
            .p_3()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(cx.theme().colors().border)
            .bg(background)
            .child(
                h_flex()
                    .w_full()
                    .items_start()
                    .justify_between()
                    .gap_3()
                    .child(
                        v_flex()
                            .min_w_0()
                            .gap_0p5()
                            .child(Label::new(presentation.headline).size(LabelSize::Default))
                            .child(
                                Label::new(presentation.definition_summary)
                                    .size(LabelSize::XSmall)
                                    .color(Color::Muted),
                            ),
                    )
                    .child(
                        Label::new(presentation.state_label)
                            .size(LabelSize::Small)
                            .color(tone_color(presentation.tone)),
                    ),
            )
            .when_some(presentation.run_summary, |this, summary| {
                this.child(
                    Label::new(summary)
                        .size(LabelSize::XSmall)
                        .color(Color::Muted),
                )
            })
            .when(!presentation.steps.is_empty(), |this| {
                this.child(
                    v_flex()
                        .gap_1()
                        .child(
                            Label::new("Steps")
                                .size(LabelSize::XSmall)
                                .color(Color::Muted),
                        )
                        .children(presentation.steps.into_iter().map(|step| {
                            v_flex()
                                .p_2()
                                .gap_0p5()
                                .rounded_sm()
                                .border_1()
                                .border_color(cx.theme().colors().border_variant)
                                .child(
                                    h_flex()
                                        .justify_between()
                                        .gap_2()
                                        .child(
                                            Label::new(format!(
                                                "{}. {}",
                                                u32::from(step.index) + 1,
                                                step.label
                                            ))
                                            .size(LabelSize::Small),
                                        )
                                        .child(
                                            Label::new(step.state_label)
                                                .size(LabelSize::XSmall)
                                                .color(tone_color(step.tone)),
                                        ),
                                )
                                .child(
                                    Label::new(step.attempt_label)
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                                .when_some(step.retry_label, |this, retry| {
                                    this.child(
                                        Label::new(retry)
                                            .size(LabelSize::XSmall)
                                            .color(Color::Info),
                                    )
                                })
                                .when_some(step.failure, |this, failure| {
                                    this.child(failure_element(failure, cx))
                                })
                        })),
                )
            })
            .when(!presentation.approvals.is_empty(), |this| {
                this.child(
                    v_flex()
                        .gap_1()
                        .child(
                            Label::new("Approval requests")
                                .size(LabelSize::XSmall)
                                .color(Color::Muted),
                        )
                        .children(presentation.approvals.into_iter().map(|approval| {
                            v_flex()
                                .p_2()
                                .gap_0p5()
                                .rounded_sm()
                                .border_1()
                                .border_color(cx.theme().status().warning_border)
                                .child(
                                    h_flex()
                                        .justify_between()
                                        .gap_2()
                                        .child(
                                            Label::new(format!(
                                                "Step {} · {}",
                                                u32::from(approval.step_index) + 1,
                                                approval.request_message
                                            ))
                                            .size(LabelSize::Small),
                                        )
                                        .child(
                                            Label::new(approval.state_label)
                                                .size(LabelSize::XSmall)
                                                .color(tone_color(approval.tone)),
                                        ),
                                )
                                .child(
                                    Label::new(format!(
                                        "Eligible: {} · expires {}",
                                        approval.eligibility_label, approval.expires_at_millis
                                    ))
                                    .size(LabelSize::XSmall)
                                    .color(Color::Muted),
                                )
                        })),
                )
            })
            .when_some(presentation.failure, |this, failure| {
                this.child(failure_element(failure, cx))
            })
            .when_some(presentation.last_trustworthy, |this, checkpoint| {
                this.child(
                    Label::new(format!("Last trustworthy state: {checkpoint}"))
                        .size(LabelSize::XSmall)
                        .color(Color::Muted),
                )
            })
            .when(!actions.is_empty(), |this| {
                this.child(
                    h_flex()
                        .gap_1()
                        .children(
                            actions
                                .into_iter()
                                .enumerate()
                                .map(|(action_index, action)| {
                                    let handler = action_handler.clone();
                                    let style = match action.kind() {
                                        WorkflowTaskActionKind::GrantApproval => {
                                            ButtonStyle::Tinted(TintColor::Success)
                                        }
                                        WorkflowTaskActionKind::DenyApproval => {
                                            ButtonStyle::Tinted(TintColor::Error)
                                        }
                                        WorkflowTaskActionKind::RetryProjection => {
                                            ButtonStyle::Outlined
                                        }
                                    };
                                    Button::new(
                                        format!(
                                            "workflow-task-action-{}-{action_index}",
                                            self.index
                                        ),
                                        action.label(),
                                    )
                                    .style(style)
                                    .label_size(LabelSize::Small)
                                    .when_some(
                                        handler,
                                        |this, handler| {
                                            this.on_click(move |_, window, cx| {
                                                handler(action.clone(), window, cx)
                                            })
                                        },
                                    )
                                }),
                        ),
                )
            })
    }
}

fn failure_element(failure: WorkflowFailureSurfacePresentation, cx: &App) -> impl IntoElement {
    v_flex()
        .p_2()
        .gap_0p5()
        .rounded_sm()
        .border_1()
        .border_color(cx.theme().status().error_border)
        .child(
            Label::new(format!(
                "Affected scope: {} · failure code {}",
                failure.scope_label, failure.code
            ))
            .size(LabelSize::XSmall)
            .color(Color::Error),
        )
        .child(Label::new(failure.summary).size(LabelSize::Small))
        .when_some(failure.last_trustworthy_label, |this, checkpoint| {
            this.child(
                Label::new(format!("Last trustworthy state: {checkpoint}"))
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
        })
}

fn failure_presentation(
    failure: &WorkflowFailurePresentation,
) -> WorkflowFailureSurfacePresentation {
    WorkflowFailureSurfacePresentation {
        code: failure.code.clone(),
        summary: failure.kind.summary(),
        scope_label: failure.scope.label(),
        last_trustworthy_label: Some(failure.last_trustworthy.label.clone()),
    }
}

const fn tone_color(tone: WorkflowTaskTone) -> Color {
    match tone {
        WorkflowTaskTone::Progress => Color::Info,
        WorkflowTaskTone::Success => Color::Success,
        WorkflowTaskTone::Attention => Color::Warning,
        WorkflowTaskTone::Error => Color::Error,
        WorkflowTaskTone::Muted => Color::Muted,
    }
}

fn failure_kind(code: &str) -> WorkflowFailureKind {
    match code {
        "permission_denied" | "approval_denied" => WorkflowFailureKind::PermissionDenied,
        "temporary_unavailable" | "transport" => WorkflowFailureKind::TemporaryUnavailable,
        "timeout" => WorkflowFailureKind::TimedOut,
        "rate_limited" => WorkflowFailureKind::RateLimited,
        "ambiguous_delivery" | "repair_required" => WorkflowFailureKind::AmbiguousDelivery,
        "permanent_action" | "invalid_action" => WorkflowFailureKind::Permanent,
        _ => WorkflowFailureKind::Unknown,
    }
}

fn redact_failure_code(code: &str) -> String {
    if code.is_empty()
        || code.len() > MAX_FAILURE_CODE_BYTES
        || !code.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
    {
        "workflow_failure".into()
    } else {
        code.into()
    }
}

fn validate_identifier(value: &str) -> Result<(), WorkflowTaskPresentationError> {
    validate_bounded_text(value, MAX_IDENTIFIER_BYTES)
}

fn validate_label(value: &str) -> Result<(), WorkflowTaskPresentationError> {
    validate_bounded_text(value, MAX_LABEL_BYTES)
}

fn validate_bounded_text(
    value: &str,
    maximum_bytes: usize,
) -> Result<(), WorkflowTaskPresentationError> {
    if value.trim().is_empty() || value.len() > maximum_bytes || value.contains('\0') {
        Err(WorkflowTaskPresentationError::InvalidText)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowTaskPresentationError {
    InvalidText,
    InvalidVersion,
    DefinitionVersionMismatch,
    InvalidSteps,
    InvalidApproval,
    InvalidRetry,
    InvalidFailure,
}

impl fmt::Display for WorkflowTaskPresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidText => "workflow presentation text is invalid",
            Self::InvalidVersion => "workflow presentation version is invalid",
            Self::DefinitionVersionMismatch => {
                "workflow run does not match the displayed definition version"
            }
            Self::InvalidSteps => "workflow presentation steps are invalid",
            Self::InvalidApproval => "workflow presentation approval is invalid",
            Self::InvalidRetry => "workflow presentation retry is invalid",
            Self::InvalidFailure => "workflow presentation failure is invalid",
        })
    }
}

impl Error for WorkflowTaskPresentationError {}

#[cfg(test)]
mod tests {
    use super::*;

    const UPDATED_AT: u64 = 1_900_000_000_000;

    fn definition() -> WorkflowDefinitionPresentation {
        WorkflowDefinitionPresentation::new(
            "community-1",
            "workflow-1",
            "Release workspace",
            3,
            7,
            WorkflowDefinitionLifecycle::Active,
            "Project imagineerings/zed",
        )
        .expect("valid definition")
    }

    fn checkpoint(label: &str, version: u64) -> WorkflowTrustworthyCheckpoint {
        WorkflowTrustworthyCheckpoint::new(label, version).expect("valid checkpoint")
    }

    fn step(index: u16, state: WorkflowStepPresentationState) -> WorkflowStepPresentation {
        WorkflowStepPresentation::new(
            index,
            format!("step-{index}"),
            format!("operation-{index}"),
            state,
            u16::from(state != WorkflowStepPresentationState::Pending),
            None,
            None,
        )
        .expect("valid step")
    }

    fn make_projection(
        state: WorkflowRunPresentationState,
        steps: Vec<WorkflowStepPresentation>,
        approvals: Vec<WorkflowApprovalPresentation>,
        failure: Option<WorkflowFailurePresentation>,
    ) -> WorkflowTaskProjection {
        let run = WorkflowRunPresentation::new(
            "run-1", 3, 11, state, 0, steps, approvals, failure, UPDATED_AT,
        )
        .expect("valid run");
        WorkflowTaskProjection::available(definition(), run).expect("matching definition")
    }

    fn approval(
        state: WorkflowApprovalPresentationState,
        viewer_may_decide: bool,
    ) -> WorkflowApprovalPresentation {
        WorkflowApprovalPresentation::new(
            "approval-1",
            0,
            "Deploy the signed release?",
            "Owners or administrators",
            state,
            UPDATED_AT + 60_000,
            viewer_may_decide,
        )
        .expect("valid approval")
    }

    #[gpui::test]
    fn workflow_task_surface_renders_running_definition_run_and_steps() {
        let projection = make_projection(
            WorkflowRunPresentationState::Running,
            vec![step(0, WorkflowStepPresentationState::Running)],
            Vec::new(),
            None,
        );
        let presentation = projection.presentation();
        assert_eq!(presentation.headline, "Release workspace");
        assert!(presentation.definition_summary.contains("Definition v3"));
        assert_eq!(presentation.state_label, "Running");
        assert_eq!(presentation.steps[0].state_label, "Running");
        assert_eq!(presentation.tone, WorkflowTaskTone::Progress);
        assert!(projection.action_requests().is_empty());
    }

    #[gpui::test]
    fn workflow_task_surface_exposes_waiting_approval_actions_only_to_eligible_viewer() {
        let projection = make_projection(
            WorkflowRunPresentationState::WaitingApproval,
            vec![step(0, WorkflowStepPresentationState::WaitingApproval)],
            vec![approval(WorkflowApprovalPresentationState::Pending, true)],
            None,
        );
        let presentation = projection.presentation();
        assert_eq!(presentation.state_label, "Waiting for approval");
        assert_eq!(presentation.approvals[0].state_label, "Approval required");
        let actions = projection.action_requests();
        assert_eq!(
            actions
                .iter()
                .map(WorkflowTaskAction::kind)
                .collect::<Vec<_>>(),
            vec![
                WorkflowTaskActionKind::GrantApproval,
                WorkflowTaskActionKind::DenyApproval
            ]
        );
        assert!(actions.first().is_some_and(|action| matches!(
            action,
            WorkflowTaskAction::GrantApproval {
                community_id,
                workflow_id,
                run_id,
                approval_id,
                expected_run_version: 11,
            } if community_id == "community-1"
                && workflow_id == "workflow-1"
                && run_id == "run-1"
                && approval_id == "approval-1"
        )));

        let hidden = make_projection(
            WorkflowRunPresentationState::WaitingApproval,
            vec![step(0, WorkflowStepPresentationState::WaitingApproval)],
            vec![approval(WorkflowApprovalPresentationState::Pending, false)],
            None,
        );
        assert!(hidden.action_requests().is_empty());
    }

    #[gpui::test]
    fn workflow_task_surface_renders_granted_approval_as_terminal_history() {
        let projection = make_projection(
            WorkflowRunPresentationState::Queued,
            vec![step(0, WorkflowStepPresentationState::Completed)],
            vec![approval(WorkflowApprovalPresentationState::Granted, false)],
            None,
        );
        assert_eq!(
            projection.presentation().approvals[0].state_label,
            "Granted"
        );
        assert_eq!(
            projection.presentation().approvals[0].tone,
            WorkflowTaskTone::Success
        );
        assert!(projection.action_requests().is_empty());
    }

    #[gpui::test]
    fn workflow_task_surface_renders_denied_approval_without_mutable_actions() {
        let projection = make_projection(
            WorkflowRunPresentationState::Cancelled,
            vec![step(0, WorkflowStepPresentationState::Cancelled)],
            vec![approval(WorkflowApprovalPresentationState::Denied, false)],
            None,
        );
        assert_eq!(projection.presentation().approvals[0].state_label, "Denied");
        assert_eq!(
            projection.presentation().approvals[0].tone,
            WorkflowTaskTone::Error
        );
        assert!(projection.action_requests().is_empty());
    }

    #[gpui::test]
    fn workflow_task_surface_renders_scheduled_retry_attempt() {
        let failure = WorkflowFailurePresentation::from_service_error(
            "timeout",
            "upstream hostname and token must not render",
            WorkflowFailureScope::Step(0),
            checkpoint("Step 1 running at checkpoint 10", 10),
            false,
        );
        let retry = WorkflowRetryPresentation::new(2, UPDATED_AT + 30_000).expect("valid retry");
        let retry_step = WorkflowStepPresentation::new(
            0,
            "step-0",
            "operation-0",
            WorkflowStepPresentationState::RetryScheduled,
            1,
            Some(retry),
            Some(failure.clone()),
        )
        .expect("valid retry step");
        let projection = make_projection(
            WorkflowRunPresentationState::RetryScheduled,
            vec![retry_step],
            Vec::new(),
            Some(failure),
        );
        let presentation = projection.presentation();
        assert_eq!(presentation.state_label, "Retry scheduled");
        assert!(
            presentation.steps[0]
                .retry_label
                .as_deref()
                .is_some_and(|label| label.contains("attempt 2"))
        );
    }

    #[gpui::test]
    fn workflow_task_surface_redacts_service_failure_messages_and_scopes_recovery() {
        let secret = "Bearer secret-token for private.example";
        let failure = WorkflowFailurePresentation::from_service_error(
            "temporary_unavailable",
            secret,
            WorkflowFailureScope::Step(0),
            checkpoint("Step 1 completed at checkpoint 10", 10),
            true,
        );
        let projection = make_projection(
            WorkflowRunPresentationState::Failed,
            vec![
                WorkflowStepPresentation::new(
                    0,
                    "step-0",
                    "operation-0",
                    WorkflowStepPresentationState::Failed,
                    1,
                    None,
                    Some(failure.clone()),
                )
                .expect("valid failed step"),
            ],
            Vec::new(),
            Some(failure),
        );
        let presentation = projection.presentation();
        let visible = format!("{presentation:?}");
        assert!(!visible.contains(secret));
        assert_eq!(
            presentation
                .failure
                .as_ref()
                .map(|failure| failure.code.as_str()),
            Some("temporary_unavailable")
        );
        assert_eq!(
            presentation
                .failure
                .as_ref()
                .map(|failure| failure.scope_label.as_str()),
            Some("step 1")
        );
        assert_eq!(
            projection.action_requests()[0].kind(),
            WorkflowTaskActionKind::RetryProjection
        );
    }

    #[gpui::test]
    fn workflow_task_surface_preserves_last_trustworthy_state_when_service_is_unavailable() {
        let projection = WorkflowTaskProjection::unavailable(
            WorkflowServiceUnavailablePresentation::new(
                Some("community-1".into()),
                WorkflowFailureScope::Service,
                Some(checkpoint(
                    "Run 1 waiting for approval at checkpoint 11",
                    11,
                )),
            )
            .expect("valid unavailable projection"),
        );
        let presentation = projection.presentation();
        assert_eq!(presentation.state_label, "Unavailable");
        assert_eq!(presentation.tone, WorkflowTaskTone::Error);
        assert_eq!(
            presentation.last_trustworthy.as_deref(),
            Some("Run 1 waiting for approval at checkpoint 11")
        );
        assert_eq!(
            projection.action_requests()[0].kind(),
            WorkflowTaskActionKind::RetryProjection
        );
    }
}
