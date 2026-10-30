use gpui::{App, AppContext as _, Context, Entity, EventEmitter, Global, Task};
#[cfg(all(test, feature = "test-support"))]
use sim_runtime::compile_generated_native_prompt;
use sim_runtime::{
    AssetAvailability, AssetIdentity, AssetNamespace, AssetService, AttemptEvent, AttemptId,
    AttemptState, AuthorizedCapabilities, CompiledPlan, DEFAULT_NATIVE_PROFILE_ID,
    DisconnectedExecutionController, ExecutionCommandAck, ExecutionControlCommand,
    ExecutionControlCommandKind, ExecutionController, ExecutionDataSource, ExecutionEventBus,
    ExecutionFailure, ExecutionFailureOrigin, ExecutionOutput, ExecutionOutputAvailability,
    ExecutionPresentationError, ExecutionPresentationOwner, ExecutionPresentationService,
    ExecutionReconciliation, ExecutionSnapshot, ExecutionSnapshotStatus, NativeExecutionController,
    NativeExecutionControllerConfig, NativeExecutionRegistryBundle,
    NativeProviderWorkerBridgeAttachment, OperationEligibility, ProfileId, RequestId,
    RetryPromptSource, SharedAssetService, SharedExecutionPresentationService, SimRuntimeDb,
    WorkflowFormatDocument, authorize_native_output_ui, generated_native_frontend_contracts,
    graph_to_prompt,
};
use sim_tensor::CancellationToken;
use sim_types::NodeId;
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::Arc,
};
use thiserror::Error;
use uuid::Uuid;

pub const EXECUTION_HISTORY_CAPACITY: usize = 10_000;
pub const EXECUTION_DIAGNOSTIC_CAPACITY: usize = 256;
pub const EXECUTION_EVENT_INGESTION_BATCH_CAPACITY: usize = 1_024;
pub const EXECUTION_EVENT_BATCHES_PER_NOTIFICATION: usize = 16;
pub const EXECUTION_EVENT_BUSES_PER_PROFILE: usize = 2;
pub const LOCAL_EXECUTION_PROFILE_ID: ProfileId = ProfileId(DEFAULT_NATIVE_PROFILE_ID);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionRunMode {
    #[default]
    Manual,
    OnChange,
    InstantIdle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlanRequest {
    pub profile_id: ProfileId,
    pub document_identity: String,
    pub workflow_bytes: Vec<u8>,
    pub selected_output_nodes: BTreeSet<NodeId>,
}

pub trait ExecutionPlanProvider: Send + Sync {
    fn compile(&self, request: &ExecutionPlanRequest) -> Result<CompiledPlan, ExecutionFailure>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionOutputReferenceAction {
    View,
    Download,
}

pub trait ExecutionOutputReferenceHandler: Send + Sync {
    fn handle(
        &self,
        profile_id: ProfileId,
        action: ExecutionOutputReferenceAction,
        reference: &str,
    ) -> Result<(), ExecutionFailure>;
}

pub use sim_runtime::ExecutionOutputOperationAction;

pub trait ExecutionOutputOperationHandler: Send + Sync {
    fn handle(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output: &ExecutionOutput,
        action: ExecutionOutputOperationAction,
        presentation: &SharedExecutionPresentationService,
        cancellation: &CancellationToken,
    ) -> Result<ExecutionOutputAvailability, ExecutionFailure>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionDiagnosticKind {
    Duplicate,
    Stale,
    Gap,
    CrossProfile,
    Terminal,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionDiagnostic {
    pub kind: ExecutionDiagnosticKind,
    pub profile_id: Option<ProfileId>,
    pub attempt_id: Option<AttemptId>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionUiEvent {
    Changed {
        profiles: Vec<ProfileId>,
    },
    CommandSubmitted {
        profile_id: ProfileId,
        request_id: RequestId,
        queue_batch_count: Option<usize>,
    },
    CommandAcknowledged {
        profile_id: ProfileId,
        request_id: RequestId,
        queue_batch_count: Option<usize>,
        outcome: sim_runtime::ExecutionCommandOutcome,
    },
    Error(ExecutionDiagnostic),
}

pub struct ExecutionUiModel {
    service: SharedExecutionPresentationService,
    controller: Arc<dyn ExecutionController>,
    runtime_controller_available: bool,
    plan_provider: Option<Arc<dyn ExecutionPlanProvider>>,
    output_reference_handler: Option<Arc<dyn ExecutionOutputReferenceHandler>>,
    output_operation_handler: Option<Arc<dyn ExecutionOutputOperationHandler>>,
    output_operation_tasks: HashMap<(ProfileId, AttemptId, Uuid), PendingOutputOperation>,
    active_profile_id: Option<ProfileId>,
    diagnostics: VecDeque<ExecutionDiagnostic>,
    event_bus_subscriptions: HashMap<ProfileId, usize>,
    event_ingestion_tasks: HashMap<ProfileId, VecDeque<Task<()>>>,
    pending_changed_profiles: Vec<ProfileId>,
    pending_diagnostic_notification: bool,
    notification_batches: u64,
}

struct PendingOutputOperation {
    cancellation: CancellationToken,
    _task: Task<()>,
}

impl ExecutionUiModel {
    pub fn new(
        service: ExecutionPresentationService,
        controller: Arc<dyn ExecutionController>,
    ) -> Self {
        Self::new_shared(
            sim_runtime::ExecutionPresentationOwner::ephemeral(service),
            controller,
        )
    }

    pub fn new_without_runtime_controller(service: ExecutionPresentationService) -> Self {
        Self::new_shared_without_runtime_controller(
            sim_runtime::ExecutionPresentationOwner::ephemeral(service),
        )
    }

    pub fn new_shared(
        service: SharedExecutionPresentationService,
        controller: Arc<dyn ExecutionController>,
    ) -> Self {
        Self::with_controller(service, controller, true)
    }

    pub fn new_shared_without_runtime_controller(
        service: SharedExecutionPresentationService,
    ) -> Self {
        Self::with_controller(service, Arc::new(DisconnectedExecutionController), false)
    }

    fn with_controller(
        service: SharedExecutionPresentationService,
        controller: Arc<dyn ExecutionController>,
        runtime_controller_available: bool,
    ) -> Self {
        Self {
            service,
            controller,
            runtime_controller_available,
            plan_provider: None,
            output_reference_handler: None,
            output_operation_handler: None,
            output_operation_tasks: HashMap::new(),
            active_profile_id: None,
            diagnostics: VecDeque::new(),
            event_bus_subscriptions: HashMap::new(),
            event_ingestion_tasks: HashMap::new(),
            pending_changed_profiles: Vec::new(),
            pending_diagnostic_notification: false,
            notification_batches: 0,
        }
    }

    pub fn shared_service(&self) -> SharedExecutionPresentationService {
        self.service.clone()
    }

    pub fn register_runtime_controller(
        &mut self,
        controller: Arc<dyn ExecutionController>,
        cx: &mut Context<Self>,
    ) {
        let previous = std::mem::replace(&mut self.controller, controller);
        if let Err(failure) = previous.shutdown() {
            self.record_error(
                ExecutionDiagnostic {
                    kind: ExecutionDiagnosticKind::Invalid,
                    profile_id: self.active_profile_id,
                    attempt_id: None,
                    message: failure.message,
                },
                cx,
            );
        }
        self.runtime_controller_available = true;
        self.emit_capabilities_changed(cx);
    }

    pub fn clear_runtime_controller(&mut self, cx: &mut Context<Self>) {
        let previous = std::mem::replace(
            &mut self.controller,
            Arc::new(DisconnectedExecutionController),
        );
        if let Err(failure) = previous.shutdown() {
            self.record_error(
                ExecutionDiagnostic {
                    kind: ExecutionDiagnosticKind::Invalid,
                    profile_id: self.active_profile_id,
                    attempt_id: None,
                    message: failure.message,
                },
                cx,
            );
        }
        self.runtime_controller_available = false;
        self.emit_capabilities_changed(cx);
    }

    pub fn runtime_controller_available(&self) -> bool {
        self.runtime_controller_available
    }

    pub fn initialize_profile(
        &mut self,
        profile_id: ProfileId,
        source: ExecutionDataSource,
        status: ExecutionSnapshotStatus,
        cx: &mut Context<Self>,
    ) -> Result<(), ExecutionUiModelError> {
        smol::block_on(
            self.service
                .initialize_profile_durable(profile_id, source, status),
        )?;
        if self.active_profile_id.is_none() {
            self.active_profile_id = Some(profile_id);
        }
        self.emit_changed([profile_id], cx);
        Ok(())
    }

    pub fn set_snapshot_status(
        &mut self,
        profile_id: ProfileId,
        source: ExecutionDataSource,
        status: ExecutionSnapshotStatus,
        cx: &mut Context<Self>,
    ) -> Result<(), ExecutionUiModelError> {
        smol::block_on(
            self.service
                .set_snapshot_status_durable(profile_id, source, status),
        )?;
        self.emit_changed([profile_id], cx);
        Ok(())
    }

    pub fn set_active_profile(
        &mut self,
        profile_id: ProfileId,
        cx: &mut Context<Self>,
    ) -> Result<(), ExecutionUiModelError> {
        self.service.snapshot(profile_id)?;
        if self.active_profile_id != Some(profile_id) {
            for (key, operation) in &self.output_operation_tasks {
                if key.0 != profile_id {
                    operation.cancellation.cancel();
                }
            }
        }
        self.active_profile_id = Some(profile_id);
        self.emit_changed([profile_id], cx);
        Ok(())
    }

    pub fn active_profile_id(&self) -> Option<ProfileId> {
        self.active_profile_id
    }

    pub fn active_snapshot(&self) -> Result<ExecutionSnapshot, ExecutionUiModelError> {
        let profile_id = self
            .active_profile_id
            .ok_or(ExecutionUiModelError::NoActiveProfile)?;
        self.snapshot(profile_id)
    }

    pub fn snapshot(
        &self,
        profile_id: ProfileId,
    ) -> Result<ExecutionSnapshot, ExecutionUiModelError> {
        Ok(self.service.snapshot(profile_id)?)
    }

    pub fn register_plan_provider(
        &mut self,
        provider: Arc<dyn ExecutionPlanProvider>,
        cx: &mut Context<Self>,
    ) {
        self.plan_provider = Some(provider);
        self.emit_capabilities_changed(cx);
    }

    pub fn plan_provider_available(&self) -> bool {
        self.plan_provider.is_some()
    }

    pub fn clear_plan_provider(&mut self, cx: &mut Context<Self>) {
        self.plan_provider = None;
        self.emit_capabilities_changed(cx);
    }

    pub fn register_output_reference_handler(
        &mut self,
        handler: Arc<dyn ExecutionOutputReferenceHandler>,
        cx: &mut Context<Self>,
    ) {
        self.output_reference_handler = Some(handler);
        self.emit_capabilities_changed(cx);
    }

    pub fn clear_output_reference_handler(&mut self, cx: &mut Context<Self>) {
        self.output_reference_handler = None;
        self.emit_capabilities_changed(cx);
    }

    pub fn output_reference_actions_available(&self) -> bool {
        self.output_reference_handler.is_some()
    }

    pub fn handle_output_reference(
        &self,
        profile_id: ProfileId,
        action: ExecutionOutputReferenceAction,
        reference: &str,
    ) -> Result<(), ExecutionUiModelError> {
        if reference.trim().is_empty() {
            return Err(ExecutionUiModelError::InvalidOutputReference);
        }
        let handler = self
            .output_reference_handler
            .as_ref()
            .ok_or(ExecutionUiModelError::OutputReferenceHandlerUnavailable)?;
        handler
            .handle(profile_id, action, reference)
            .map_err(|failure| ExecutionUiModelError::OutputReference {
                code: failure.code,
                message: failure.message,
            })
    }

    pub fn register_output_operation_handler(
        &mut self,
        handler: Arc<dyn ExecutionOutputOperationHandler>,
        cx: &mut Context<Self>,
    ) {
        self.output_operation_handler = Some(handler);
        self.emit_capabilities_changed(cx);
    }

    pub fn clear_output_operation_handler(&mut self, cx: &mut Context<Self>) {
        for operation in self.output_operation_tasks.values() {
            operation.cancellation.cancel();
        }
        self.output_operation_tasks.clear();
        self.output_operation_handler = None;
        self.emit_capabilities_changed(cx);
    }

    pub fn output_operations_available(&self) -> bool {
        self.output_operation_handler.is_some()
    }

    pub(crate) fn handle_output_operation(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output_id: Uuid,
        action: ExecutionOutputOperationAction,
        cx: &mut Context<Self>,
    ) -> Result<(), ExecutionUiModelError> {
        let snapshot = self.service.snapshot(profile_id)?;
        let attempt = snapshot
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == attempt_id)
            .ok_or(ExecutionUiModelError::OutputAttemptNotFound(attempt_id))?;
        let output = attempt
            .outputs
            .iter()
            .find(|output| output.output_id == output_id)
            .cloned()
            .ok_or(ExecutionUiModelError::OutputNotFound(output_id))?;
        let eligibility = self
            .service
            .output_operation_eligibility(profile_id, attempt_id, output_id, action)?;
        if let OperationEligibility::Unavailable { reason } = eligibility {
            return Err(ExecutionUiModelError::OutputOperationNotAllowed {
                operation: action.label(),
                reason,
            });
        }
        let handler = self
            .output_operation_handler
            .as_ref()
            .cloned()
            .ok_or(ExecutionUiModelError::OutputOperationHandlerUnavailable)?;
        let key = (profile_id, attempt_id, output_id);
        if self.output_operation_tasks.contains_key(&key) {
            return Err(ExecutionUiModelError::OutputOperationPending(output_id));
        }
        let cancellation = CancellationToken::default();
        let background = cx.background_spawn({
            let cancellation = cancellation.clone();
            let presentation = self.service.clone();
            async move {
                handler.handle(
                    profile_id,
                    attempt_id,
                    &output,
                    action,
                    &presentation,
                    &cancellation,
                )
            }
        });
        let task = cx.spawn(async move |this, cx| {
            let result = background.await;
            if let Err(error) = this.update(cx, |this, cx| {
                this.output_operation_tasks.remove(&key);
                match result {
                    Ok(_) => this.emit_changed([profile_id], cx),
                    Err(failure) => this.record_error(
                        ExecutionDiagnostic {
                            kind: ExecutionDiagnosticKind::Invalid,
                            profile_id: Some(profile_id),
                            attempt_id: Some(attempt_id),
                            message: format!(
                                "{} failed ({}): {}",
                                action.label(),
                                failure.code,
                                failure.message
                            ),
                        },
                        cx,
                    ),
                }
            }) {
                log::debug!("execution output operation stopped after model teardown: {error}");
            }
        });
        self.output_operation_tasks.insert(
            key,
            PendingOutputOperation {
                cancellation,
                _task: task,
            },
        );
        Ok(())
    }

    pub fn queue(
        &mut self,
        request: ExecutionPlanRequest,
        priority: i32,
        front: bool,
        cx: &mut Context<Self>,
    ) -> Result<ExecutionCommandAck, ExecutionUiModelError> {
        let provider = self
            .plan_provider
            .as_ref()
            .ok_or(ExecutionUiModelError::PlanProviderUnavailable)?;
        let plan = provider.compile(&request).map_err(|failure| {
            ExecutionUiModelError::PlanCompilation {
                code: failure.code,
                message: failure.message,
            }
        })?;
        self.dispatch(
            request.profile_id,
            ExecutionControlCommandKind::Queue {
                plan,
                priority,
                front,
            },
            cx,
        )
    }

    pub fn dispatch(
        &mut self,
        profile_id: ProfileId,
        kind: ExecutionControlCommandKind,
        cx: &mut Context<Self>,
    ) -> Result<ExecutionCommandAck, ExecutionUiModelError> {
        let snapshot = self.service.snapshot(profile_id)?;
        let command = ExecutionControlCommand {
            request_id: RequestId(Uuid::new_v4()),
            profile_id,
            expected_revision: Some(snapshot.revision),
            kind,
        };
        let request_id = command.request_id;
        let queue_batch_count = command_queue_batch_count(&command.kind);
        cx.emit(ExecutionUiEvent::CommandSubmitted {
            profile_id,
            request_id,
            queue_batch_count,
        });
        let dispatch_result = smol::block_on(
            self.service
                .dispatch_durable(command, self.controller.as_ref()),
        );
        let acknowledgement = match dispatch_result {
            Ok(acknowledgement) => acknowledgement,
            Err(error) => {
                self.emit_changed([profile_id], cx);
                return Err(error.into());
            }
        };
        cx.emit(ExecutionUiEvent::CommandAcknowledged {
            profile_id,
            request_id,
            queue_batch_count,
            outcome: acknowledgement.outcome.clone(),
        });
        self.emit_changed([profile_id], cx);
        Ok(acknowledgement)
    }

    pub fn apply_ack(
        &mut self,
        acknowledgement: ExecutionCommandAck,
        cx: &mut Context<Self>,
    ) -> Result<(), ExecutionUiModelError> {
        let profile_id = acknowledgement.profile_id;
        let queue_batch_count = self.service.snapshot(profile_id).ok().and_then(|snapshot| {
            snapshot
                .pending_commands
                .iter()
                .find(|pending| pending.command.request_id == acknowledgement.request_id)
                .and_then(|pending| command_queue_batch_count(&pending.command.kind))
        });
        let request_id = acknowledgement.request_id;
        let outcome = acknowledgement.outcome.clone();
        let result = smol::block_on(self.service.apply_ack_durable(acknowledgement));
        if result.is_ok() {
            cx.emit(ExecutionUiEvent::CommandAcknowledged {
                profile_id,
                request_id,
                queue_batch_count,
                outcome,
            });
        }
        self.emit_changed([profile_id], cx);
        result.map_err(Into::into)
    }

    pub fn reconcile(
        &mut self,
        reconciliation: ExecutionReconciliation,
        cx: &mut Context<Self>,
    ) -> Result<(), ExecutionUiModelError> {
        let profile_id = reconciliation.profile_id;
        let reconciliation_result = smol::block_on(self.service.reconcile_durable(reconciliation));
        match reconciliation_result {
            Ok(changed) => {
                if changed {
                    self.emit_changed([profile_id], cx);
                }
                Ok(())
            }
            Err(error) => {
                if matches!(
                    error,
                    ExecutionPresentationError::StaleReconciliation { .. }
                ) {
                    let snapshot_result = self.service.snapshot(profile_id);
                    match snapshot_result {
                        Ok(snapshot) => {
                            let stale_failure =
                                ExecutionFailure::new("stale_reconciliation", error.to_string());
                            let status_result =
                                smol::block_on(self.service.set_snapshot_status_durable(
                                    profile_id,
                                    snapshot.source,
                                    ExecutionSnapshotStatus::Stale {
                                        source_revision: snapshot.source_revision,
                                        failure: stale_failure,
                                    },
                                ));
                            if let Err(status_error) = status_result {
                                self.record_error(
                                    classify_error(&status_error, Some(profile_id), None),
                                    cx,
                                );
                            } else {
                                self.emit_changed([profile_id], cx);
                            }
                        }
                        Err(snapshot_error) => self.record_error(
                            classify_error(&snapshot_error, Some(profile_id), None),
                            cx,
                        ),
                    }
                }
                self.record_error(classify_error(&error, Some(profile_id), None), cx);
                Err(error.into())
            }
        }
    }

    pub fn attach_event_bus(&mut self, bus: ExecutionEventBus, cx: &mut Context<Self>) -> bool {
        let Some(profile_id) = self.active_profile_id else {
            return false;
        };
        if self
            .event_bus_subscriptions
            .get(&profile_id)
            .is_some_and(|count| *count > 0)
        {
            return false;
        }
        self.attach_profile_event_bus(profile_id, bus, cx);
        true
    }

    pub fn attach_profile_event_bus(
        &mut self,
        profile_id: ProfileId,
        bus: ExecutionEventBus,
        cx: &mut Context<Self>,
    ) {
        self.event_bus_subscriptions
            .entry(profile_id)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
        let receiver = bus.subscribe();
        let task = cx.spawn(async move |this, cx| {
            let mut batches_since_notification = 0_usize;
            while let Ok(first_event) = receiver.recv().await {
                let mut events = Vec::with_capacity(EXECUTION_EVENT_INGESTION_BATCH_CAPACITY);
                events.push(first_event);
                while events.len() < EXECUTION_EVENT_INGESTION_BATCH_CAPACITY {
                    match receiver.try_recv() {
                        Ok(event) => events.push(event),
                        Err(_) => break,
                    }
                }
                batches_since_notification = batches_since_notification.saturating_add(1);
                let flush = receiver.is_empty()
                    || batches_since_notification >= EXECUTION_EVENT_BATCHES_PER_NOTIFICATION;
                if this
                    .update(cx, |this, cx| {
                        this.ingest_event_batch_coalesced(events, flush, cx)
                    })
                    .is_err()
                {
                    break;
                }
                if flush {
                    batches_since_notification = 0;
                }
            }
            if let Err(error) = this.update(cx, |this, _| {
                let remove = if let Some(count) = this.event_bus_subscriptions.get_mut(&profile_id)
                {
                    *count = count.saturating_sub(1);
                    *count == 0
                } else {
                    false
                };
                if remove {
                    this.event_bus_subscriptions.remove(&profile_id);
                }
            }) {
                log::debug!("execution event ingestion stopped after model teardown: {error}");
            }
        });
        let tasks = self.event_ingestion_tasks.entry(profile_id).or_default();
        tasks.push_back(task);
        while tasks.len() > EXECUTION_EVENT_BUSES_PER_PROFILE {
            tasks.pop_front();
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn ingest_event_batch(&mut self, events: Vec<AttemptEvent>, cx: &mut Context<Self>) {
        let mut applied = Vec::with_capacity(events.len());
        let mut diagnostics = Vec::new();
        for event in events {
            match smol::block_on(self.service.apply_event_durable(event.clone())) {
                Ok(()) => applied.push(event),
                Err(error) => diagnostics.push(classify_error(
                    &error,
                    Some(event.profile_id),
                    Some(event.attempt_id),
                )),
            }
        }
        for diagnostic in diagnostics {
            self.push_diagnostic(diagnostic.clone());
            cx.emit(ExecutionUiEvent::Error(diagnostic));
        }
        self.ingest_event_batch_coalesced(applied, true, cx);
    }

    fn ingest_event_batch_coalesced(
        &mut self,
        events: Vec<AttemptEvent>,
        flush: bool,
        cx: &mut Context<Self>,
    ) {
        if events.is_empty() {
            return;
        }
        let mut diagnostics = Vec::new();
        for event in events {
            let profile_id = event.profile_id;
            let attempt_id = event.attempt_id;
            let event_result = self.service.contains_canonical_event(&event);
            match event_result {
                Ok(true) => {
                    if !self.pending_changed_profiles.contains(&profile_id) {
                        self.pending_changed_profiles.push(profile_id);
                    }
                }
                Ok(false) => {
                    let validation_result = self.service.validate_unapplied_event(&event);
                    match validation_result {
                        Ok(()) => diagnostics.push(ExecutionDiagnostic {
                            kind: ExecutionDiagnosticKind::Invalid,
                            profile_id: Some(profile_id),
                            attempt_id: Some(attempt_id),
                            message: "execution event bus delivered an event before the canonical actuator applied it".to_owned(),
                        }),
                        Err(error) => diagnostics.push(classify_error(
                            &error,
                            Some(profile_id),
                            Some(attempt_id),
                        )),
                    }
                }
                Err(error) => {
                    diagnostics.push(classify_error(&error, Some(profile_id), Some(attempt_id)))
                }
            }
        }
        let had_diagnostics = !diagnostics.is_empty();
        self.pending_diagnostic_notification |= had_diagnostics;
        for diagnostic in diagnostics {
            self.push_diagnostic(diagnostic.clone());
            cx.emit(ExecutionUiEvent::Error(diagnostic));
        }
        if flush && !self.pending_changed_profiles.is_empty() {
            let changed_profiles = std::mem::take(&mut self.pending_changed_profiles);
            self.emit_changed(changed_profiles, cx);
        } else if flush && self.pending_diagnostic_notification {
            cx.notify();
        }
        if flush {
            self.pending_diagnostic_notification = false;
        }
    }

    pub fn diagnostics(&self) -> impl DoubleEndedIterator<Item = &ExecutionDiagnostic> {
        self.diagnostics.iter()
    }

    pub fn notification_batches(&self) -> u64 {
        self.notification_batches
    }

    pub fn attempt(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
    ) -> Result<Option<sim_runtime::AttemptPresentation>, ExecutionUiModelError> {
        Ok(self
            .service
            .snapshot(profile_id)?
            .attempts
            .into_iter()
            .find(|attempt| attempt.attempt_id == attempt_id))
    }

    pub fn retry(
        &mut self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        cx: &mut Context<Self>,
    ) -> Result<ExecutionCommandAck, ExecutionUiModelError> {
        self.dispatch(
            profile_id,
            ExecutionControlCommandKind::Retry {
                attempt_id,
                source: RetryPromptSource::OriginalPrompt,
                replacement_plan: None,
            },
            cx,
        )
    }

    pub fn interrupt_active(
        &mut self,
        profile_id: ProfileId,
        cx: &mut Context<Self>,
    ) -> Result<ExecutionCommandAck, ExecutionUiModelError> {
        let attempt_id = self
            .service
            .snapshot(profile_id)?
            .attempts
            .into_iter()
            .find(|attempt| matches!(attempt.state, AttemptState::Running | AttemptState::Queued))
            .map(|attempt| attempt.attempt_id)
            .ok_or(ExecutionUiModelError::NoInterruptibleAttempt)?;
        self.dispatch(
            profile_id,
            ExecutionControlCommandKind::Interrupt {
                attempt_id,
                reason: "interrupted from native execution UI".to_owned(),
            },
            cx,
        )
    }

    fn emit_changed(
        &mut self,
        profiles: impl IntoIterator<Item = ProfileId>,
        cx: &mut Context<Self>,
    ) {
        let mut unique_profiles = Vec::new();
        for profile_id in profiles.into_iter() {
            if !unique_profiles.contains(&profile_id) {
                unique_profiles.push(profile_id);
            }
        }
        self.notification_batches = self.notification_batches.saturating_add(1);
        cx.emit(ExecutionUiEvent::Changed {
            profiles: unique_profiles,
        });
        cx.notify();
    }

    fn emit_capabilities_changed(&self, cx: &mut Context<Self>) {
        cx.emit(ExecutionUiEvent::Changed {
            profiles: self.active_profile_id.into_iter().collect(),
        });
        cx.notify();
    }

    fn record_error(&mut self, diagnostic: ExecutionDiagnostic, cx: &mut Context<Self>) {
        self.push_diagnostic(diagnostic.clone());
        cx.emit(ExecutionUiEvent::Error(diagnostic));
        cx.notify();
    }

    fn push_diagnostic(&mut self, diagnostic: ExecutionDiagnostic) {
        self.diagnostics.push_back(diagnostic);
        while self.diagnostics.len() > EXECUTION_DIAGNOSTIC_CAPACITY {
            self.diagnostics.pop_front();
        }
    }
}

impl EventEmitter<ExecutionUiEvent> for ExecutionUiModel {}

impl Drop for ExecutionUiModel {
    fn drop(&mut self) {
        for operation in self.output_operation_tasks.values() {
            operation.cancellation.cancel();
        }
    }
}

fn command_queue_batch_count(kind: &ExecutionControlCommandKind) -> Option<usize> {
    let ExecutionControlCommandKind::Queue { plan, .. } = kind else {
        return None;
    };
    Some(
        ["batch_count", "batchCount"]
            .into_iter()
            .find_map(|key| plan.extra_data.get(key).and_then(serde_json::Value::as_u64))
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| *count > 0)
            .unwrap_or(1),
    )
}

struct NativeGeneratedPlanProvider {
    bundle: Arc<NativeExecutionRegistryBundle>,
}

impl ExecutionPlanProvider for NativeGeneratedPlanProvider {
    fn compile(&self, request: &ExecutionPlanRequest) -> Result<CompiledPlan, ExecutionFailure> {
        if request.profile_id != self.bundle.profile_id() {
            return Err(ExecutionFailure::new(
                "native_profile_unavailable",
                "the workflow profile is not registered with the local native runtime",
            )
            .with_origin(ExecutionFailureOrigin::Validation));
        }
        if request.document_identity.trim().is_empty() || request.workflow_bytes.is_empty() {
            return Err(ExecutionFailure::new(
                "invalid_native_workflow",
                "native execution requires a document identity and serialized workflow",
            )
            .with_origin(ExecutionFailureOrigin::Validation));
        }
        compile_generated_native_workflow_with_bundle(
            &request.workflow_bytes,
            &request.selected_output_nodes,
            &self.bundle,
        )
    }
}

#[cfg(all(test, feature = "test-support"))]
pub(crate) fn compile_generated_native_workflow(
    workflow_bytes: &[u8],
    selected_output_nodes: &BTreeSet<NodeId>,
) -> Result<CompiledPlan, ExecutionFailure> {
    let workflow = WorkflowFormatDocument::parse(workflow_bytes).map_err(|error| {
        ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
            .with_origin(ExecutionFailureOrigin::Validation)
    })?;
    let contracts = generated_native_frontend_contracts(None).map_err(|error| {
        ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
            .with_origin(ExecutionFailureOrigin::Validation)
    })?;
    let descriptors = contracts
        .into_iter()
        .map(|(class_type, contract)| (class_type, contract.graph))
        .collect();
    let submission =
        graph_to_prompt(&workflow, &descriptors, "zed-native-generated-v1").map_err(|error| {
            ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
                .with_origin(ExecutionFailureOrigin::Validation)
        })?;
    let mut plan = compile_generated_native_prompt(submission, None).map_err(|error| {
        ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
            .with_origin(ExecutionFailureOrigin::Validation)
    })?;
    if !selected_output_nodes.is_empty() {
        for node_id in selected_output_nodes {
            let node = plan.nodes.get(node_id).ok_or_else(|| {
                ExecutionFailure::new(
                    "native_plan_compilation_failed",
                    format!("selected output node {node_id:?} is not in the compiled plan"),
                )
                .with_origin(ExecutionFailureOrigin::Validation)
            })?;
            if !node.descriptor.output_node {
                return Err(ExecutionFailure::new(
                    "native_plan_compilation_failed",
                    format!("selected node {node_id:?} is not an output node"),
                )
                .with_origin(ExecutionFailureOrigin::Validation));
            }
        }
        plan.output_nodes = selected_output_nodes.iter().cloned().collect();
    }
    Ok(plan)
}

fn compile_generated_native_workflow_with_bundle(
    workflow_bytes: &[u8],
    selected_output_nodes: &BTreeSet<NodeId>,
    bundle: &NativeExecutionRegistryBundle,
) -> Result<CompiledPlan, ExecutionFailure> {
    let workflow = WorkflowFormatDocument::parse(workflow_bytes).map_err(|error| {
        ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
            .with_origin(ExecutionFailureOrigin::Validation)
    })?;
    let contracts = generated_native_frontend_contracts(None).map_err(|error| {
        ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
            .with_origin(ExecutionFailureOrigin::Validation)
    })?;
    let descriptors = contracts
        .into_iter()
        .map(|(class_type, contract)| (class_type, contract.graph))
        .collect();
    let submission =
        graph_to_prompt(&workflow, &descriptors, bundle.identity_sha256()).map_err(|error| {
            ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
                .with_origin(ExecutionFailureOrigin::Validation)
        })?;
    let mut plan = bundle.compile(submission).map_err(|error| {
        ExecutionFailure::new("native_plan_compilation_failed", error.to_string())
            .with_origin(ExecutionFailureOrigin::Validation)
    })?;
    apply_selected_output_nodes(&mut plan, selected_output_nodes)?;
    Ok(plan)
}

fn apply_selected_output_nodes(
    plan: &mut CompiledPlan,
    selected_output_nodes: &BTreeSet<NodeId>,
) -> Result<(), ExecutionFailure> {
    if selected_output_nodes.is_empty() {
        return Ok(());
    }
    for node_id in selected_output_nodes {
        let node = plan.nodes.get(node_id).ok_or_else(|| {
            ExecutionFailure::new(
                "native_plan_compilation_failed",
                format!("selected output node {node_id:?} is not in the compiled plan"),
            )
            .with_origin(ExecutionFailureOrigin::Validation)
        })?;
        if !node.descriptor.output_node {
            return Err(ExecutionFailure::new(
                "native_plan_compilation_failed",
                format!("selected node {node_id:?} is not an output node"),
            )
            .with_origin(ExecutionFailureOrigin::Validation));
        }
    }
    plan.output_nodes = selected_output_nodes.iter().cloned().collect();
    Ok(())
}

struct NativeOutputService {
    profile_id: ProfileId,
    assets: SharedAssetService,
    output_committer: sim_runtime::SharedOutputCommitter,
    authorization: AuthorizedCapabilities,
}

impl NativeOutputService {
    fn new(config: &NativeExecutionControllerConfig) -> Result<Self, NativeExecutionServiceError> {
        let profile_identity = config
            .assets
            .lock()
            .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?
            .roots()
            .profile_id
            .clone();
        let service = Self {
            profile_id: config.worker.profile_id,
            assets: config.assets.clone(),
            output_committer: config.output_committer.clone(),
            authorization: authorize_native_output_ui(profile_identity)
                .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?,
        };
        service.reconcile_pending_removals(&config.presentation)?;
        Ok(service)
    }

    fn identity(
        &self,
        profile_id: ProfileId,
        reference: &str,
    ) -> Result<AssetIdentity, ExecutionFailure> {
        if profile_id != self.profile_id {
            return Err(ExecutionFailure::new(
                "native_output_profile_mismatch",
                "the output reference belongs to a different execution profile",
            )
            .with_origin(ExecutionFailureOrigin::Validation));
        }
        self.lock_assets()?
            .roots()
            .identity_from_reference(reference)
            .map_err(|error| {
                ExecutionFailure::new("invalid_native_output_reference", error.to_string())
                    .with_origin(ExecutionFailureOrigin::Validation)
            })
    }

    fn lock_assets(&self) -> Result<std::sync::MutexGuard<'_, AssetService>, ExecutionFailure> {
        self.assets.lock().map_err(|error| {
            ExecutionFailure::new(
                "native_output_service_poisoned",
                format!("native output state is unavailable: {error}"),
            )
            .with_origin(ExecutionFailureOrigin::Filesystem)
            .retryable(true)
        })
    }

    fn reconcile_pending_removals(
        &self,
        presentation: &SharedExecutionPresentationService,
    ) -> Result<(), NativeExecutionServiceError> {
        let pending = self
            .output_committer
            .lock()
            .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?
            .pending_removals();
        if pending.is_empty() {
            return Ok(());
        }
        let snapshot = presentation
            .snapshot(self.profile_id)
            .map_err(|error| NativeExecutionServiceError::Runtime(error.to_string()))?;
        for operation in pending {
            let reference = operation
                .identity
                .to_reference()
                .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?;
            let projection_removed = snapshot.attempts.iter().any(|attempt| {
                attempt.outputs.iter().any(|output| {
                    output_reference(output) == Some(reference.as_str())
                        && matches!(
                            output.availability,
                            ExecutionOutputAvailability::Removed { .. }
                        )
                })
            });
            if projection_removed {
                let cancellation = CancellationToken::default();
                {
                    let mut committer = self
                        .output_committer
                        .lock()
                        .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?;
                    let mut assets = self
                        .assets
                        .lock()
                        .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?;
                    committer
                        .commit_removal_and_register(
                            operation.operation_id,
                            &mut assets,
                            &self.authorization,
                            &cancellation,
                        )
                        .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?;
                }
                self.output_committer
                    .lock()
                    .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?
                    .cleanup_committed_removal(operation.operation_id)
                    .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?;
            } else {
                self.output_committer
                    .lock()
                    .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?
                    .rollback_removal(operation.operation_id, &self.authorization)
                    .map_err(|error| NativeExecutionServiceError::Asset(error.to_string()))?;
            }
        }
        Ok(())
    }
}

impl ExecutionOutputOperationHandler for NativeOutputService {
    fn handle(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output: &ExecutionOutput,
        action: ExecutionOutputOperationAction,
        presentation: &SharedExecutionPresentationService,
        cancellation: &CancellationToken,
    ) -> Result<ExecutionOutputAvailability, ExecutionFailure> {
        let reference = output_reference(output).ok_or_else(|| {
            ExecutionFailure::new(
                "native_output_reference_missing",
                "the output has no typed native asset reference",
            )
            .with_origin(ExecutionFailureOrigin::Filesystem)
        })?;
        let identity = self.identity(profile_id, reference)?;
        let mut assets = self.lock_assets()?;
        assets
            .scan_namespaces(
                &[AssetNamespace::Output, AssetNamespace::Temporary],
                &self.authorization,
                &cancellation,
            )
            .map_err(|error| {
                ExecutionFailure::new("native_output_scan_failed", error.to_string())
                    .with_origin(ExecutionFailureOrigin::Filesystem)
                    .retryable(true)
            })?;
        match action {
            ExecutionOutputOperationAction::Recover => {
                let recovered = assets
                    .record(&identity)
                    .filter(|record| record.availability == AssetAvailability::Present);
                let Some(recovered) = recovered else {
                    return Err(ExecutionFailure::new(
                        "native_output_not_recovered",
                        "the output file is still absent or invalid after reconciliation",
                    )
                    .with_origin(ExecutionFailureOrigin::Filesystem)
                    .retryable(true));
                };
                let availability = ExecutionOutputAvailability::Ready {
                    reference: reference.to_owned(),
                    byte_length: recovered.byte_size,
                };
                drop(assets);
                smol::block_on(presentation.apply_output_operation_durable(
                    profile_id,
                    attempt_id,
                    output.output_id,
                    action,
                    availability.clone(),
                ))
                .map_err(output_presentation_failure)?;
                Ok(availability)
            }
            ExecutionOutputOperationAction::Remove => {
                drop(assets);
                let availability = ExecutionOutputAvailability::Removed {
                    reason: "removed from the native execution output record".to_owned(),
                };
                if !matches!(
                    output.availability,
                    ExecutionOutputAvailability::Ready { .. }
                ) {
                    smol::block_on(presentation.apply_output_operation_durable(
                        profile_id,
                        attempt_id,
                        output.output_id,
                        action,
                        availability.clone(),
                    ))
                    .map_err(output_presentation_failure)?;
                    return Ok(availability);
                }
                let prepared = self
                    .output_committer
                    .lock()
                    .map_err(output_committer_lock_failure)?
                    .prepare_removal(&identity, &self.authorization, cancellation)
                    .map_err(output_filesystem_failure)?;
                let operation_id = prepared.operation_id;
                let commit_committer = self.output_committer.clone();
                let rollback_committer = self.output_committer.clone();
                let commit_assets = self.assets.clone();
                let commit_authorization = self.authorization.clone();
                let rollback_authorization = self.authorization.clone();
                let commit_cancellation = cancellation.clone();
                smol::block_on(presentation.apply_output_operation_transaction_durable(
                    profile_id,
                    attempt_id,
                    output.output_id,
                    action,
                    availability.clone(),
                    move || {
                        let mut committer =
                            commit_committer.lock().map_err(|error| error.to_string())?;
                        let mut assets = commit_assets.lock().map_err(|error| error.to_string())?;
                        committer
                            .commit_removal_and_register(
                                operation_id,
                                &mut assets,
                                &commit_authorization,
                                &commit_cancellation,
                            )
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    },
                    move || {
                        rollback_committer
                            .lock()
                            .map_err(|error| error.to_string())?
                            .rollback_removal(operation_id, &rollback_authorization)
                            .map(|_| ())
                            .map_err(|error| error.to_string())
                    },
                ))
                .map_err(output_presentation_failure)?;
                if let Err(error) = self
                    .output_committer
                    .lock()
                    .map_err(output_committer_lock_failure)?
                    .cleanup_committed_removal(operation_id)
                {
                    log::warn!(
                        "native output removal committed with deferred staging cleanup: {error}"
                    );
                }
                Ok(availability)
            }
        }
    }
}

fn output_reference(output: &ExecutionOutput) -> Option<&str> {
    output
        .view_reference
        .as_deref()
        .or_else(|| match &output.availability {
            ExecutionOutputAvailability::Ready { reference, .. }
            | ExecutionOutputAvailability::ExternallyDeleted { reference, .. } => {
                Some(reference.as_str())
            }
            ExecutionOutputAvailability::Missing {
                reference: Some(reference),
                ..
            }
            | ExecutionOutputAvailability::Expired {
                reference: Some(reference),
                ..
            }
            | ExecutionOutputAvailability::Corrupt {
                reference: Some(reference),
                ..
            } => Some(reference.as_str()),
            ExecutionOutputAvailability::Removed { .. }
            | ExecutionOutputAvailability::Forbidden { .. }
            | ExecutionOutputAvailability::Unsupported { .. }
            | ExecutionOutputAvailability::Missing {
                reference: None, ..
            }
            | ExecutionOutputAvailability::Expired {
                reference: None, ..
            }
            | ExecutionOutputAvailability::Corrupt {
                reference: None, ..
            } => None,
        })
}

fn output_filesystem_failure(error: impl ToString) -> ExecutionFailure {
    ExecutionFailure::new("native_output_remove_failed", error.to_string())
        .with_origin(ExecutionFailureOrigin::Filesystem)
        .retryable(true)
}

fn output_presentation_failure(error: impl ToString) -> ExecutionFailure {
    ExecutionFailure::new("native_output_persistence_failed", error.to_string())
        .with_origin(ExecutionFailureOrigin::Unknown)
        .retryable(true)
}

fn output_committer_lock_failure(
    error: std::sync::PoisonError<std::sync::MutexGuard<'_, sim_runtime::OutputCommitter>>,
) -> ExecutionFailure {
    output_filesystem_failure(error)
}

#[derive(Debug, Error)]
pub enum NativeExecutionServiceError {
    #[error("native execution runtime initialization failed: {0}")]
    Runtime(String),
    #[error("native execution asset initialization failed: {0}")]
    Asset(String),
    #[error("native execution UI model is unavailable")]
    ModelUnavailable,
    #[error(
        "native execution controller and GPUI model do not share the canonical presentation service"
    )]
    PresentationOwnerMismatch,
    #[error("native execution UI registration failed: {0}")]
    Model(String),
}

pub fn register_native_execution_services(
    config: NativeExecutionControllerConfig,
    bundle: Arc<NativeExecutionRegistryBundle>,
    cx: &mut App,
) -> Result<NativeProviderWorkerBridgeAttachment, NativeExecutionServiceError> {
    let profile_id = config.worker.profile_id;
    if bundle.profile_id() != profile_id
        || config.provider_registry.as_ref() != bundle.provider_registry()
    {
        return Err(NativeExecutionServiceError::Runtime(
            "native execution controller and plan provider do not share one registry bundle"
                .to_owned(),
        ));
    }
    let model = execution_ui_model(cx).ok_or(NativeExecutionServiceError::ModelUnavailable)?;
    if !Arc::ptr_eq(&model.read(cx).shared_service(), &config.presentation) {
        return Err(NativeExecutionServiceError::PresentationOwnerMismatch);
    }
    let event_bus = ExecutionEventBus::new(EXECUTION_HISTORY_CAPACITY)
        .map_err(|error| NativeExecutionServiceError::Runtime(error.to_string()))?;
    let (controller, provider_worker_bridge) =
        NativeExecutionController::start_with_provider_worker_bridge(
            config.clone(),
            event_bus.clone(),
        )
        .map_err(|error| NativeExecutionServiceError::Runtime(error.to_string()))?
        .into_parts();
    let outputs = Arc::new(NativeOutputService::new(&config)?);
    let registration = model.update(cx, |model, cx| {
        model.register_runtime_controller(controller, cx);
        model.register_plan_provider(Arc::new(NativeGeneratedPlanProvider { bundle }), cx);
        model.register_output_operation_handler(outputs, cx);
        model.attach_profile_event_bus(profile_id, event_bus, cx);
        model.set_snapshot_status(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
            cx,
        )
    });
    if let Err(error) = registration {
        model.update(cx, |model, cx| {
            model.clear_runtime_controller(cx);
            model.clear_plan_provider(cx);
            model.clear_output_reference_handler(cx);
            model.clear_output_operation_handler(cx);
        });
        return Err(NativeExecutionServiceError::Model(error.to_string()));
    }
    Ok(provider_worker_bridge)
}

pub fn clear_native_execution_services(cx: &mut App) -> Result<(), NativeExecutionServiceError> {
    let model = execution_ui_model(cx).ok_or(NativeExecutionServiceError::ModelUnavailable)?;
    model.update(cx, |model, cx| {
        model.clear_runtime_controller(cx);
        model.clear_plan_provider(cx);
        model.clear_output_reference_handler(cx);
        model.clear_output_operation_handler(cx);
    });
    Ok(())
}

pub struct GlobalExecutionUiModel(pub Entity<ExecutionUiModel>);

impl Global for GlobalExecutionUiModel {}

pub fn execution_ui_model(cx: &App) -> Option<Entity<ExecutionUiModel>> {
    cx.try_global::<GlobalExecutionUiModel>()
        .map(|global| global.0.clone())
}

pub fn init_execution_ui_model(
    cx: &mut App,
) -> Result<Entity<ExecutionUiModel>, ExecutionUiModelError> {
    init_execution_ui_model_for_profile(LOCAL_EXECUTION_PROFILE_ID, cx)
}

pub fn init_execution_ui_model_for_profile(
    profile_id: ProfileId,
    cx: &mut App,
) -> Result<Entity<ExecutionUiModel>, ExecutionUiModelError> {
    if let Some(model) = execution_ui_model(cx) {
        match model.read(cx).snapshot(profile_id) {
            Ok(_) => {
                model.update(cx, |model, cx| model.set_active_profile(profile_id, cx))?;
            }
            Err(ExecutionUiModelError::Presentation(
                ExecutionPresentationError::UnknownProfile(_),
            )) => {
                let service = model.read(cx).shared_service();
                smol::block_on(service.restore_profile(profile_id))?;
                let has_recovered_attempts = !service.snapshot(profile_id)?.attempts.is_empty();
                let (source, status) = disconnected_snapshot_state(has_recovered_attempts, None);
                smol::block_on(service.set_snapshot_status_durable(profile_id, source, status))?;
                model.update(cx, |model, cx| {
                    model.active_profile_id = Some(profile_id);
                    model.emit_changed([profile_id], cx);
                    Ok::<(), ExecutionUiModelError>(())
                })?;
            }
            Err(error) => return Err(error),
        }
        return Ok(model);
    }
    let database = SimRuntimeDb::global(cx);
    let service = ExecutionPresentationOwner::persistent(
        ExecutionPresentationService::new(EXECUTION_HISTORY_CAPACITY)?,
        Arc::new(database),
    );
    smol::block_on(service.restore_profile(profile_id))?;
    let has_recovered_attempts = !service.snapshot(profile_id)?.attempts.is_empty();
    let (source, status) = disconnected_snapshot_state(has_recovered_attempts, None);
    smol::block_on(service.set_snapshot_status_durable(profile_id, source, status))?;
    let model = cx.new(|_cx| {
        let mut model = ExecutionUiModel::new_shared_without_runtime_controller(service);
        model.active_profile_id = Some(profile_id);
        model
    });
    cx.set_global(GlobalExecutionUiModel(model.clone()));
    Ok(model)
}

fn disconnected_snapshot_state(
    has_recovered_attempts: bool,
    recovery_failure: Option<&str>,
) -> (ExecutionDataSource, ExecutionSnapshotStatus) {
    if let Some(failure) = recovery_failure {
        return (
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Stale {
                source_revision: None,
                failure: ExecutionFailure::new(
                    "persistence_recovery_failed",
                    format!("Execution history recovery failed: {failure}"),
                )
                .with_origin(ExecutionFailureOrigin::Filesystem),
            },
        );
    }
    let failure = ExecutionFailure::new(
        "runtime_not_connected",
        "The native execution runtime is not connected to this profile yet.",
    )
    .with_origin(ExecutionFailureOrigin::Transport);
    if has_recovered_attempts {
        (
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Partial { failure },
        )
    } else {
        (
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Unavailable { failure },
        )
    }
}

fn classify_error(
    error: &ExecutionPresentationError,
    profile_id: Option<ProfileId>,
    attempt_id: Option<AttemptId>,
) -> ExecutionDiagnostic {
    let kind = match error {
        ExecutionPresentationError::DuplicateRequest(_)
        | ExecutionPresentationError::History(sim_runtime::HistoryError::DuplicateAttempt(_)) => {
            ExecutionDiagnosticKind::Duplicate
        }
        ExecutionPresentationError::RevisionMismatch { .. }
        | ExecutionPresentationError::StaleReconciliation { .. }
        | ExecutionPresentationError::History(sim_runtime::HistoryError::Transition(
            sim_runtime::AttemptTransitionError::StalePreviewRevision { .. },
        ))
        | ExecutionPresentationError::Transition(
            sim_runtime::AttemptTransitionError::StalePreviewRevision { .. },
        ) => ExecutionDiagnosticKind::Stale,
        ExecutionPresentationError::CrossProfileAttempt { .. }
        | ExecutionPresentationError::AckProfileMismatch { .. }
        | ExecutionPresentationError::History(sim_runtime::HistoryError::ProfileMismatch {
            ..
        }) => ExecutionDiagnosticKind::CrossProfile,
        ExecutionPresentationError::History(sim_runtime::HistoryError::Transition(
            sim_runtime::AttemptTransitionError::Sequence { expected, actual },
        ))
        | ExecutionPresentationError::Transition(sim_runtime::AttemptTransitionError::Sequence {
            expected,
            actual,
        }) if actual < expected => {
            if actual.saturating_add(1) == *expected {
                ExecutionDiagnosticKind::Duplicate
            } else {
                ExecutionDiagnosticKind::Stale
            }
        }
        ExecutionPresentationError::History(sim_runtime::HistoryError::Transition(
            sim_runtime::AttemptTransitionError::Sequence { expected, actual },
        ))
        | ExecutionPresentationError::Transition(sim_runtime::AttemptTransitionError::Sequence {
            expected,
            actual,
        }) if actual > expected => ExecutionDiagnosticKind::Gap,
        ExecutionPresentationError::History(sim_runtime::HistoryError::Transition(
            sim_runtime::AttemptTransitionError::Terminal(_),
        ))
        | ExecutionPresentationError::Transition(sim_runtime::AttemptTransitionError::Terminal(
            _,
        )) => ExecutionDiagnosticKind::Terminal,
        _ => ExecutionDiagnosticKind::Invalid,
    };
    ExecutionDiagnostic {
        kind,
        profile_id,
        attempt_id,
        message: error.to_string(),
    }
}

#[derive(Debug, Error)]
pub enum ExecutionUiModelError {
    #[error("the canonical execution presentation service is unavailable")]
    PresentationServiceUnavailable,
    #[error("no execution profile is active")]
    NoActiveProfile,
    #[error("no native execution plan provider is registered")]
    PlanProviderUnavailable,
    #[error("native execution plan compilation failed ({code}): {message}")]
    PlanCompilation { code: String, message: String },
    #[error("there is no queued or running attempt to interrupt")]
    NoInterruptibleAttempt,
    #[error("no execution attempt is selected")]
    NoSelectedAttempt,
    #[error("no native output view/download handler is registered")]
    OutputReferenceHandlerUnavailable,
    #[error("native output reference is empty")]
    InvalidOutputReference,
    #[error("native output reference action failed ({code}): {message}")]
    OutputReference { code: String, message: String },
    #[error("no native output recovery/removal handler is registered")]
    OutputOperationHandlerUnavailable,
    #[error("execution attempt {0:?} does not exist in the active profile")]
    OutputAttemptNotFound(AttemptId),
    #[error("execution output {0} does not exist in the selected attempt")]
    OutputNotFound(Uuid),
    #[error("execution output {0} already has an operation in progress")]
    OutputOperationPending(Uuid),
    #[error("native output {operation} is unavailable: {reason}")]
    OutputOperationNotAllowed {
        operation: &'static str,
        reason: String,
    },
    #[error("native output {operation} failed ({code}): {message}")]
    OutputOperation {
        operation: &'static str,
        code: String,
        message: String,
    },
    #[error(transparent)]
    Presentation(#[from] ExecutionPresentationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "test-support")]
    use gpui::TestAppContext;
    #[cfg(feature = "test-support")]
    use sim_runtime::ExecutionCommandOutcome;
    use std::{fs, path::PathBuf};

    #[test]
    fn native_provider_worker_bridge_registration_returns_only_after_controller_registration() {
        let source = include_str!("execution_model.rs");
        let registration_start = source
            .find("pub fn register_native_execution_services(")
            .expect("native execution service registration");
        let registration_end = source[registration_start..]
            .find("pub fn clear_native_execution_services(")
            .map(|offset| registration_start + offset)
            .expect("native execution service clear boundary");
        let registration = &source[registration_start..registration_end];
        let start = registration
            .find("NativeExecutionController::start_with_provider_worker_bridge")
            .expect("controller-owned bridge startup");
        let register = registration[start..]
            .find("model.register_runtime_controller(controller, cx)")
            .map(|offset| start + offset)
            .expect("runtime controller registration");
        let returned = registration[register..]
            .find("Ok(provider_worker_bridge)")
            .map(|offset| register + offset)
            .expect("registered bridge return");
        assert!(start < register && register < returned);
        assert!(!registration.contains("provider_runtime_stream_service"));
        assert!(!registration.contains("provider_runtime_activation_grants"));
    }

    #[cfg(feature = "test-support")]
    struct AcceptingExecutionActuator;

    #[cfg(feature = "test-support")]
    impl ExecutionController for AcceptingExecutionActuator {
        fn accept(
            &self,
            _command: &ExecutionControlCommand,
            _assigned_attempt_id: Option<AttemptId>,
        ) -> Result<(), ExecutionFailure> {
            Ok(())
        }
    }

    #[test]
    fn recovered_history_remains_visible_while_runtime_is_disconnected() {
        let (source, status) = disconnected_snapshot_state(true, None);
        assert_eq!(source, ExecutionDataSource::Recovery);
        assert!(matches!(
            status,
            ExecutionSnapshotStatus::Partial {
                failure: ExecutionFailure {
                    origin: ExecutionFailureOrigin::Transport,
                    ..
                }
            }
        ));

        let (source, status) = disconnected_snapshot_state(false, None);
        assert_eq!(source, ExecutionDataSource::Live);
        assert!(matches!(
            status,
            ExecutionSnapshotStatus::Unavailable { .. }
        ));
    }

    #[test]
    fn native_output_adapter_rolls_back_staging_when_projection_rejects_removal()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let profile_id = ProfileId(Uuid::from_u128(0x5a11));
        let assets = sim_runtime::open_native_profile_asset_service(
            profile_id.0.to_string(),
            directory.path(),
            &[],
        )?;
        let identity = assets
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .roots()
            .identity(AssetNamespace::Output, "native-output.png")?;
        let path = assets
            .lock()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .roots()
            .test_root_path(AssetNamespace::Output)?
            .join(&identity.relative_path);
        fs::write(&path, b"native-output")?;
        let mut presentation_service = ExecutionPresentationService::new(8)?;
        presentation_service.initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )?;
        let presentation = ExecutionPresentationOwner::ephemeral(presentation_service);
        let config = NativeExecutionControllerConfig::new(
            assets,
            presentation.clone(),
            sim_runtime::WorkerLaunchConfig::new(
                PathBuf::from("unused-native-output-worker"),
                profile_id,
                sim_types::WorkerId(Uuid::from_u128(0x5a12)),
                sim_runtime::NATIVE_IMAGE_REGISTRY_VERSION,
                1024,
            ),
            true,
        )?;
        let handler = NativeOutputService::new(&config)?;
        let reference = identity.to_reference()?;
        let output = ExecutionOutput {
            output_id: Uuid::from_u128(0x5a13),
            node_id: NodeId::from("output"),
            output_index: 0,
            name: "native-output.png".to_owned(),
            media_kind: sim_runtime::OutputMediaKind::Image,
            media_type: "image/png".to_owned(),
            subfolder: None,
            storage_type: Some("output".to_owned()),
            metadata: Default::default(),
            view_reference: Some(reference.clone()),
            download_reference: Some(reference.clone()),
            availability: ExecutionOutputAvailability::Ready {
                reference,
                byte_length: 13,
            },
            created_at: sim_runtime::AttemptRecord::queued(
                profile_id,
                sim_types::PromptId(Uuid::from_u128(0x5a15)),
                AttemptId(Uuid::from_u128(0x5a14)),
            )
            .created_at,
        };
        let result = handler.handle(
            profile_id,
            AttemptId(Uuid::from_u128(0x5a14)),
            &output,
            ExecutionOutputOperationAction::Remove,
            &presentation,
            &CancellationToken::default(),
        );
        assert!(result.is_err());
        assert_eq!(fs::read(path)?, b"native-output");
        assert!(
            config
                .output_committer
                .lock()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .pending_removals()
                .is_empty()
        );
        Ok(())
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    fn production_initialization_installs_a_fail_closed_controller(cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let model = init_execution_ui_model(cx).expect("initialize production execution model");
            assert!(!model.read(cx).runtime_controller_available());

            let acknowledgement = model
                .update(cx, |model, cx| {
                    model.dispatch(
                        LOCAL_EXECUTION_PROFILE_ID,
                        ExecutionControlCommandKind::ClearPending {
                            reason: "production fail-closed test".to_owned(),
                        },
                        cx,
                    )
                })
                .expect("disconnected controller returns a typed acknowledgement");
            assert!(matches!(
                acknowledgement.outcome,
                ExecutionCommandOutcome::Rejected {
                    failure: ExecutionFailure {
                        origin: ExecutionFailureOrigin::Transport,
                        ..
                    }
                }
            ));

            let snapshot = model
                .read(cx)
                .snapshot(LOCAL_EXECUTION_PROFILE_ID)
                .expect("read production execution snapshot");
            assert!(snapshot.queue.is_empty());
            assert!(snapshot.attempts.is_empty());
            assert!(snapshot.pending_commands.is_empty());
        });
    }

    #[cfg(feature = "test-support")]
    #[gpui::test(seed = 16017)]
    fn initialization_adds_missing_profiles_and_switches_without_resetting_existing_state(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            cx.set_global(db::AppDatabase::test_new());
            let first_profile_id = ProfileId(Uuid::from_u128(16_017));
            let second_profile_id = ProfileId(Uuid::from_u128(16_018));
            let model = init_execution_ui_model_for_profile(first_profile_id, cx)
                .expect("initialize first execution profile");
            model
                .update(cx, |model, cx| {
                    model.set_snapshot_status(
                        first_profile_id,
                        ExecutionDataSource::Live,
                        ExecutionSnapshotStatus::Ready,
                        cx,
                    )
                })
                .expect("mark first execution profile ready");

            let same_model = init_execution_ui_model_for_profile(second_profile_id, cx)
                .expect("initialize missing second execution profile");
            assert_eq!(same_model.entity_id(), model.entity_id());
            assert_eq!(model.read(cx).active_profile_id(), Some(second_profile_id));
            assert!(matches!(
                model
                    .read(cx)
                    .snapshot(second_profile_id)
                    .expect("second profile snapshot")
                    .status,
                ExecutionSnapshotStatus::Unavailable { .. }
            ));
            assert_eq!(
                model
                    .read(cx)
                    .snapshot(first_profile_id)
                    .expect("first profile snapshot")
                    .status,
                ExecutionSnapshotStatus::Ready
            );

            init_execution_ui_model_for_profile(first_profile_id, cx)
                .expect("switch back to existing execution profile");
            assert_eq!(model.read(cx).active_profile_id(), Some(first_profile_id));
            assert_eq!(
                model
                    .read(cx)
                    .snapshot(first_profile_id)
                    .expect("preserved first profile snapshot")
                    .status,
                ExecutionSnapshotStatus::Ready
            );
        });
    }

    #[cfg(feature = "test-support")]
    #[gpui::test]
    fn registered_test_controller_is_available_and_accepts_commands(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let mut service = ExecutionPresentationService::new(16)
                .expect("create execution presentation service");
            service
                .initialize_profile(
                    LOCAL_EXECUTION_PROFILE_ID,
                    ExecutionDataSource::Live,
                    ExecutionSnapshotStatus::Ready,
                )
                .expect("initialize local execution profile");
            let model = cx.new(|_| ExecutionUiModel::new_without_runtime_controller(service));

            model.update(cx, |model, cx| {
                assert!(!model.runtime_controller_available());
                model.register_runtime_controller(Arc::new(AcceptingExecutionActuator), cx);
                assert!(model.runtime_controller_available());
                let acknowledgement = model
                    .dispatch(
                        LOCAL_EXECUTION_PROFILE_ID,
                        ExecutionControlCommandKind::ClearPending {
                            reason: "accepted controller test".to_owned(),
                        },
                        cx,
                    )
                    .expect("registered test controller accepts command");
                assert!(matches!(
                    acknowledgement.outcome,
                    ExecutionCommandOutcome::Accepted {
                        assigned_attempt_id: None
                    }
                ));

                model.clear_runtime_controller(cx);
                assert!(!model.runtime_controller_available());
            });
        });
    }
}
