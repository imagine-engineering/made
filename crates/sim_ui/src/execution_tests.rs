use crate::execution_surfaces::{
    ErrorOverlaySurface, ExecutionJobTab, ExecutionNotificationKind, ExecutionProgressToastPhase,
    ExecutionSortMode, ExecutionSurfaceAction, ExecutionSurfaceActionHandler,
    ExecutionWorkflowFilter, effective_device_label_for_test, effective_memory_label_for_test,
    error_matches_query,
};
use crate::generated_menu_catalog::{GENERATED_COMPONENT_CATALOG, GENERATED_MENU_CATALOG};
use crate::graph_render::node_has_execution_failure;
use crate::*;
use db::kvp::KeyValueStore;
use gpui::{
    AppContext as _, Context, FocusHandle, Focusable as _, IntoElement, Modifiers, MouseButton,
    ParentElement as _, Render, ScrollDelta, ScrollWheelEvent, Styled as _, TestAppContext,
    TouchPhase, WeakEntity, Window, div, point, px,
};
use project::{FakeFs, Project};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_runtime::{
    AttemptEvent, AttemptEventKind, AttemptRecord, AttemptSourceProjection, AttemptState,
    CompiledPlan, EffectiveNativeBackendState, ExecutionCommandOutcome, ExecutionControlCommand,
    ExecutionControlCommandKind, ExecutionController, ExecutionDataSource, ExecutionEventBus,
    ExecutionFailure, ExecutionFailureOrigin, ExecutionOutput, ExecutionOutputAvailability,
    ExecutionPresentationService, ExecutionReconciliation, ExecutionSnapshotStatus,
    ExternalNavigationPolicy, GraphCommand, GraphIdentifier, GraphLevel, GraphNode, GraphPoint,
    MemoryPolicy, OutputMediaKind, ProviderAttemptState, RetryPromptSource,
    SharedExecutionPresentationService, SubgraphDefinition,
};
use sim_tensor::{CancellationToken, DeviceId};
use sim_types::{AttemptId, NodeId, ProfileId, PromptId, RequestId};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use uuid::Uuid;
use workspace::MultiWorkspace;

const QUEUE_CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/sim_test_support/parity/catalogs/frontend-features.csv"
));
const COMMAND_CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/sim_test_support/parity/catalogs/frontend-commands.csv"
));
const MENU_CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/sim_test_support/parity/catalogs/frontend-menus.csv"
));
const COMPONENT_CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/sim_test_support/parity/catalogs/frontend-component-surfaces.csv"
));
const PARITY_MATRIX: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/sim_test_support/parity/parity-matrix.md"
));
const EXECUTION_PANEL_SOURCE: &str = include_str!("execution_panel.rs");
const EXECUTION_MODEL_SOURCE: &str = include_str!("execution_model.rs");
const EXECUTION_CATALOG_SOURCE: &str = include_str!("execution_catalog.rs");
const EXECUTION_SURFACES_SOURCE: &str = include_str!("execution_surfaces.rs");
const EXECUTION_TEST_SOURCE: &str = include_str!("execution_tests.rs");
const ACTIONS_SOURCE: &str = include_str!("actions.rs");
const SHELL_SOURCE: &str = include_str!("shell.rs");
const QUEUE_PANEL_SOURCE: &str = include_str!("queue_panel.rs");
const HISTORY_PANEL_SOURCE: &str = include_str!("history_panel.rs");
const OUTPUT_VIEW_SOURCE: &str = include_str!("output_view.rs");
const GRAPH_RENDER_SOURCE: &str = include_str!("graph_render.rs");
const WORKFLOW_ITEM_SOURCE: &str = include_str!("workflow_item.rs");
const ZED_SOURCE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../zed/src/zed.rs"));
const RUNTIME_PRESENTATION_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_runtime/src/execution_presentation.rs"
));

#[test]
fn effective_backend_memory_label_includes_ceiling_and_physical_capacity() {
    let backend = EffectiveNativeBackendState {
        device: DeviceId::new(sim_types::DeviceKind::Mlu, 2),
        device_name: "Cambricon MLU fixture".to_owned(),
        architecture: Some("Neuware 1.20".to_owned()),
        total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
        allocation_limit_bytes: Some(20 * 1024 * 1024 * 1024),
        memory_limit_bytes: 16 * 1024 * 1024 * 1024,
        memory_in_use_bytes: 0,
        memory_policy: MemoryPolicy::Balanced,
        supported_operation_rows: 1,
        deterministic_operation_rows: 1,
    };
    assert_eq!(
        effective_memory_label_for_test(&backend),
        "0 / 17179869184 bytes · 21474836480 bytes device ceiling · 25769803776 bytes physical · balanced"
    );
}

#[test]
fn directml_job_details_preserve_device_physical_ceiling_and_effective_labels() {
    let backend = EffectiveNativeBackendState {
        device: DeviceId::new(sim_types::DeviceKind::DirectMl, 0),
        device_name: "DirectML certified adapter".to_owned(),
        architecture: Some("DXGI adapter LUID 0x1122334455667788".to_owned()),
        total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
        allocation_limit_bytes: Some(18 * 1024 * 1024 * 1024),
        memory_limit_bytes: 12 * 1024 * 1024 * 1024,
        memory_in_use_bytes: 4 * 1024 * 1024,
        memory_policy: MemoryPolicy::Balanced,
        supported_operation_rows: 7,
        deterministic_operation_rows: 7,
    };
    assert_eq!(
        effective_device_label_for_test(&backend),
        "DirectML certified adapter · DXGI adapter LUID 0x1122334455667788"
    );
    assert_eq!(
        effective_memory_label_for_test(&backend),
        "4194304 / 12884901888 bytes · 19327352832 bytes device ceiling · 25769803776 bytes physical · balanced"
    );
    assert!(EXECUTION_SURFACES_SOURCE.contains("\"sim-job-effective-device\""));
    assert!(EXECUTION_SURFACES_SOURCE.contains("\"sim-job-effective-memory\""));
    assert!(EXECUTION_SURFACES_SOURCE.contains(".role(Role::Status)"));
    assert!(EXECUTION_SURFACES_SOURCE.contains(".aria_label(format!(\"{label}: {value}\"))"));
}

#[test]
fn npu_job_details_preserve_device_physical_ceiling_and_effective_labels() {
    let backend = EffectiveNativeBackendState {
        device: DeviceId::new(sim_types::DeviceKind::Npu, 2),
        device_name: "Huawei Ascend certified device".to_owned(),
        architecture: Some("AscendCL 8.0.0".to_owned()),
        total_memory_bytes: Some(32 * 1024 * 1024 * 1024),
        allocation_limit_bytes: Some(24 * 1024 * 1024 * 1024),
        memory_limit_bytes: 12 * 1024 * 1024 * 1024,
        memory_in_use_bytes: 8 * 1024 * 1024,
        memory_policy: MemoryPolicy::Balanced,
        supported_operation_rows: 7,
        deterministic_operation_rows: 7,
    };
    assert_eq!(
        effective_device_label_for_test(&backend),
        "Huawei Ascend certified device · AscendCL 8.0.0"
    );
    assert_eq!(
        effective_memory_label_for_test(&backend),
        "8388608 / 12884901888 bytes · 25769803776 bytes device ceiling · 34359738368 bytes physical · balanced"
    );
}

#[test]
fn xpu_job_details_preserve_device_physical_ceiling_and_effective_labels() {
    let backend = EffectiveNativeBackendState {
        device: DeviceId::new(sim_types::DeviceKind::Xpu, 3),
        device_name: "Intel XPU certified device".to_owned(),
        architecture: Some("Intel 0x8086:0x56a0; oneDNN 3.5.0".to_owned()),
        total_memory_bytes: Some(16 * 1024 * 1024 * 1024),
        allocation_limit_bytes: Some(6 * 1024 * 1024 * 1024),
        memory_limit_bytes: 6 * 1024 * 1024 * 1024,
        memory_in_use_bytes: 8 * 1024 * 1024,
        memory_policy: MemoryPolicy::Balanced,
        supported_operation_rows: 7,
        deterministic_operation_rows: 7,
    };
    assert_eq!(
        effective_device_label_for_test(&backend),
        "Intel XPU certified device · Intel 0x8086:0x56a0; oneDNN 3.5.0"
    );
    assert_eq!(
        effective_memory_label_for_test(&backend),
        "8388608 / 6442450944 bytes · 6442450944 bytes device ceiling · 17179869184 bytes physical · balanced"
    );
}

#[test]
fn cuda_job_details_preserve_device_physical_ceiling_and_effective_labels() {
    let backend = EffectiveNativeBackendState {
        device: DeviceId::new(sim_types::DeviceKind::Cuda, 3),
        device_name: "NVIDIA CUDA certified device".to_owned(),
        architecture: Some("CUDA driver 12080; NVRTC 12.8".to_owned()),
        total_memory_bytes: Some(24 * 1024 * 1024 * 1024),
        allocation_limit_bytes: Some(18 * 1024 * 1024 * 1024),
        memory_limit_bytes: 12 * 1024 * 1024 * 1024,
        memory_in_use_bytes: 8 * 1024 * 1024,
        memory_policy: MemoryPolicy::Balanced,
        supported_operation_rows: 7,
        deterministic_operation_rows: 7,
    };
    assert_eq!(
        effective_device_label_for_test(&backend),
        "NVIDIA CUDA certified device · CUDA driver 12080; NVRTC 12.8"
    );
    assert_eq!(
        effective_memory_label_for_test(&backend),
        "8388608 / 12884901888 bytes · 19327352832 bytes device ceiling · 25769803776 bytes physical · balanced"
    );
}
const RUNTIME_QUEUE_HISTORY_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_runtime/src/queue_history.rs"
));
const RUNTIME_PERSISTENCE_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_runtime/src/persistence.rs"
));
const RUNTIME_CRATE_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_runtime/src/sim_runtime.rs"
));
const DEFAULT_SIM_KEYMAP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/keymaps/default-sim.json"
));
const EXECUTION_LEDGER_GENERATOR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/sim_test_support/parity/regenerate_native_zed_evidence.py"
));

struct ErrorOverlayProbe {
    attempt_id: AttemptId,
    view_focus_handle: FocusHandle,
    dismiss_focus_handle: FocusHandle,
    actions: Arc<Mutex<Vec<ExecutionSurfaceAction>>>,
}

impl ErrorOverlayProbe {
    fn new(
        attempt_id: AttemptId,
        actions: Arc<Mutex<Vec<ExecutionSurfaceAction>>>,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            attempt_id,
            view_focus_handle: cx.focus_handle(),
            dismiss_focus_handle: cx.focus_handle(),
            actions,
        }
    }
}

impl Render for ErrorOverlayProbe {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let actions = self.actions.clone();
        let on_action: ExecutionSurfaceActionHandler =
            Arc::new(move |action, _, _| match actions.lock() {
                Ok(mut actions) => actions.push(action),
                Err(error) => error.into_inner().push(action),
            });

        div().size_full().child(ErrorOverlaySurface {
            attempt_id: self.attempt_id,
            failure_count: 1,
            view_focus_handle: self.view_focus_handle.clone(),
            dismiss_focus_handle: self.dismiss_focus_handle.clone(),
            on_action,
        })
    }
}

fn recorded_overlay_probe_actions(
    actions: &Arc<Mutex<Vec<ExecutionSurfaceAction>>>,
) -> Vec<ExecutionSurfaceAction> {
    match actions.lock() {
        Ok(actions) => actions.clone(),
        Err(error) => error.into_inner().clone(),
    }
}

fn task_18_commands() -> Vec<&'static str> {
    command_registry()
        .filter(|registration| {
            registration.placement == NativePlacement::ExecutionDock
                && registration.owner == EXECUTION_UI_OWNER
                && registration.status == CommandNativeStatus::Executable
        })
        .map(|registration| registration.command_id)
        .collect()
}

fn task_27_commands() -> Vec<&'static str> {
    command_registry()
        .filter(|registration| {
            registration.placement == NativePlacement::ExecutionDock
                && registration.owner == NATIVE_MEMORY_OWNER
        })
        .map(|registration| registration.command_id)
        .collect()
}

fn is_job_or_run_menu(surface: &str) -> bool {
    matches!(
        surface,
        "job context" | "job history actions" | "run-mode menu"
    )
}

fn job_and_run_menu_ids() -> Vec<&'static str> {
    GENERATED_MENU_CATALOG
        .iter()
        .filter(|row| is_job_or_run_menu(row.menu_surface))
        .map(|row| row.feature_id)
        .collect()
}

fn task_18_menu_ids() -> Vec<&'static str> {
    GENERATED_MENU_CATALOG
        .iter()
        .filter(|row| is_job_or_run_menu(row.menu_surface) && row.owner == EXECUTION_UI_OWNER)
        .map(|row| row.feature_id)
        .collect()
}

fn execution_component_ids() -> Vec<&'static str> {
    GENERATED_COMPONENT_CATALOG
        .iter()
        .filter(|row| row.owner == EXECUTION_UI_OWNER || row.domain == "queue-execution-ui")
        .map(|row| row.feature_id)
        .collect()
}

#[derive(Clone)]
struct DeterministicPlanProvider {
    plan: CompiledPlan,
}

impl ExecutionPlanProvider for DeterministicPlanProvider {
    fn compile(&self, request: &ExecutionPlanRequest) -> Result<CompiledPlan, ExecutionFailure> {
        if request.document_identity.trim().is_empty() {
            return Err(ExecutionFailure::new(
                "missing_document_identity",
                "the execution fixture requires a document identity",
            ));
        }
        let mut plan = self.plan.clone();
        if !request.selected_output_nodes.is_empty() {
            plan.output_nodes = request.selected_output_nodes.iter().cloned().collect();
            plan.prompt_id = PromptId(Uuid::from_u128(
                plan.prompt_id.0.as_u128().saturating_add(1),
            ));
        }
        Ok(plan)
    }
}

#[derive(Clone, Default)]
struct NativeGeneratedPlanProviderProbe {
    compiled_plans: Arc<Mutex<Vec<CompiledPlan>>>,
}

impl ExecutionPlanProvider for NativeGeneratedPlanProviderProbe {
    fn compile(&self, request: &ExecutionPlanRequest) -> Result<CompiledPlan, ExecutionFailure> {
        let plan = compile_generated_native_workflow(
            &request.workflow_bytes,
            &request.selected_output_nodes,
        )?;
        match self.compiled_plans.lock() {
            Ok(mut compiled_plans) => compiled_plans.push(plan.clone()),
            Err(error) => error.into_inner().push(plan.clone()),
        }
        Ok(plan)
    }
}

#[derive(Clone)]
struct NativeUiControllerProbe {
    commands: Arc<Mutex<Vec<ExecutionControlCommand>>>,
}

impl NativeUiControllerProbe {
    fn new() -> Self {
        Self {
            commands: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn commands(&self) -> Vec<ExecutionControlCommand> {
        match self.commands.lock() {
            Ok(commands) => commands.clone(),
            Err(error) => error.into_inner().clone(),
        }
    }
}

impl ExecutionController for NativeUiControllerProbe {
    fn accept(
        &self,
        command: &ExecutionControlCommand,
        _assigned_attempt_id: Option<AttemptId>,
    ) -> Result<(), ExecutionFailure> {
        match self.commands.lock() {
            Ok(mut commands) => commands.push(command.clone()),
            Err(error) => error.into_inner().push(command.clone()),
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default)]
struct AcceptingExecutionActuator;

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
fn generated_ui_compiler_tracks_the_comprehensive_frontend_projection() -> Result<(), Box<dyn Error>>
{
    let registry = sim_runtime::generated_native_node_registry_projection(None)?;
    registry.validate_comprehensive_bindings()?;
    let frontend = sim_runtime::generated_native_frontend_descriptors(None)?;
    let contracts = sim_runtime::generated_native_frontend_contracts(None)?;
    assert_eq!(frontend.len(), registry.descriptor_len());
    assert_eq!(contracts.len(), registry.descriptor_len());
    for (class_type, contract) in &contracts {
        contract.runtime.validate_exact_schema_v2()?;
        assert_eq!(contract.graph, frontend[class_type]);
        assert_eq!(
            contract.runtime.source_schema.as_ref().map(|schema| {
                schema
                    .inputs
                    .iter()
                    .map(|input| input.name.as_str())
                    .collect::<Vec<_>>()
            }),
            Some(
                contract
                    .graph
                    .inputs
                    .iter()
                    .map(|input| input.name.as_str())
                    .collect::<Vec<_>>()
            )
        );
    }

    let model = native_image_graph_fixture(LOCAL_EXECUTION_PROFILE_ID)?;
    let WorkflowOpenState::Editable(engine) = &model.open_state else {
        return Err("native UI fixture opened read-only".into());
    };
    let bytes = engine.document.to_workflow_bytes()?;
    let plan = compile_generated_native_workflow(&bytes, &BTreeSet::new()).map_err(|error| {
        format!(
            "generated native workflow compilation failed: {}",
            error.message
        )
    })?;
    assert_eq!(plan.nodes.len(), 5);
    assert!(
        plan.nodes
            .values()
            .all(|node| frontend.contains_key(&node.class_type))
    );

    let error = compile_generated_native_workflow(
        &bytes,
        &BTreeSet::from([NodeId("missing-output".to_owned())]),
    )
    .expect_err("missing selected output must fail before queue mutation");
    assert_eq!(error.origin, ExecutionFailureOrigin::Validation);
    assert!(error.message.contains("not in the compiled plan"));
    Ok(())
}

#[derive(Default)]
struct RecordingReferenceHandler {
    actions: Mutex<Vec<(ProfileId, ExecutionOutputReferenceAction, String)>>,
}

#[derive(Default)]
struct RecordingOperationHandler {
    actions: Mutex<Vec<(ProfileId, AttemptId, Uuid, ExecutionOutputOperationAction)>>,
}

#[derive(Clone, Copy, Default)]
struct CancellationObservation {
    started: bool,
    cancelled: bool,
}

#[derive(Default)]
struct CancellationObservingOperationHandler {
    observation: Mutex<CancellationObservation>,
}

impl CancellationObservingOperationHandler {
    fn observation(&self) -> Result<CancellationObservation, String> {
        self.observation
            .lock()
            .map(|observation| *observation)
            .map_err(|_| "cancellation observation lock poisoned".to_owned())
    }
}

impl ExecutionOutputOperationHandler for CancellationObservingOperationHandler {
    fn handle(
        &self,
        _profile_id: ProfileId,
        _attempt_id: AttemptId,
        _output: &ExecutionOutput,
        _action: ExecutionOutputOperationAction,
        _presentation: &SharedExecutionPresentationService,
        cancellation: &CancellationToken,
    ) -> Result<ExecutionOutputAvailability, ExecutionFailure> {
        let mut observation = self.observation.lock().map_err(|_| {
            ExecutionFailure::new(
                "cancellation_observation_lock",
                "cancellation observation lock poisoned",
            )
        })?;
        observation.started = true;
        if !cancellation.is_cancelled() {
            return Err(ExecutionFailure::new(
                "cancellation_observation_not_cancelled",
                "output operation started without profile-switch cancellation",
            ));
        }
        observation.cancelled = true;
        Err(ExecutionFailure::new(
            "output_operation_cancelled",
            "output operation was cancelled after the active profile changed",
        ))
    }
}

impl ExecutionOutputOperationHandler for RecordingOperationHandler {
    fn handle(
        &self,
        profile_id: ProfileId,
        attempt_id: AttemptId,
        output: &ExecutionOutput,
        action: ExecutionOutputOperationAction,
        presentation: &SharedExecutionPresentationService,
        _cancellation: &CancellationToken,
    ) -> Result<ExecutionOutputAvailability, ExecutionFailure> {
        self.actions
            .lock()
            .map_err(|_| ExecutionFailure::new("recording_lock", "operation recorder poisoned"))?
            .push((profile_id, attempt_id, output.output_id, action));
        let availability = match action {
            ExecutionOutputOperationAction::Recover => ExecutionOutputAvailability::Ready {
                reference: output
                    .view_reference
                    .clone()
                    .unwrap_or_else(|| "zed-asset://output/recovered-output".to_owned()),
                byte_length: 1,
            },
            ExecutionOutputOperationAction::Remove => ExecutionOutputAvailability::Removed {
                reason: "removed by the deterministic operation fixture".to_owned(),
            },
        };
        smol::block_on(presentation.apply_output_operation_durable(
            profile_id,
            attempt_id,
            output.output_id,
            action,
            availability.clone(),
        ))
        .map_err(|error| ExecutionFailure::new("recording_persistence", error.to_string()))?;
        Ok(availability)
    }
}

impl ExecutionOutputReferenceHandler for RecordingReferenceHandler {
    fn handle(
        &self,
        profile_id: ProfileId,
        action: ExecutionOutputReferenceAction,
        reference: &str,
    ) -> Result<(), ExecutionFailure> {
        self.actions
            .lock()
            .map_err(|_| ExecutionFailure::new("recording_lock", "reference recorder poisoned"))?
            .push((profile_id, action, reference.to_owned()));
        Ok(())
    }
}

fn identifier(value: u128) -> Uuid {
    Uuid::from_u128(0x1805_0000_0000_0000_0000_0000_0000_0000 | value)
}

fn profile(value: u128) -> ProfileId {
    ProfileId(identifier(value))
}

fn attempt(value: u128) -> AttemptId {
    AttemptId(identifier(0x1000 + value))
}

fn prompt(value: u128) -> PromptId {
    PromptId(identifier(0x2000 + value))
}

fn request(value: u128) -> RequestId {
    RequestId(identifier(0x3000 + value))
}

fn plan(prompt_id: PromptId) -> CompiledPlan {
    CompiledPlan {
        prompt_id,
        client_id: Some("val-gpui-005".to_owned()),
        prompt_number: Some(18.0),
        extra_data: BTreeMap::from([("profile".to_owned(), json!("native"))]),
        unknown: BTreeMap::from([("forward_compatible".to_owned(), json!(true))]),
        nodes: BTreeMap::new(),
        topological_order: Vec::new(),
        static_required_nodes: BTreeSet::new(),
        output_nodes: Vec::new(),
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    }
}

fn event(
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    sequence: u64,
    node_id: Option<NodeId>,
    kind: AttemptEventKind,
) -> AttemptEvent {
    let at = AttemptRecord::queued(profile_id, prompt_id, attempt_id).created_at;
    AttemptEvent {
        profile_id,
        prompt_id,
        attempt_id,
        sequence,
        node_id,
        at,
        kind,
        data: Some(json!({"fixture_sequence": sequence})),
    }
}

fn acknowledge_queue(
    service: &mut ExecutionPresentationService,
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    request_id: RequestId,
) -> Result<(), String> {
    let revision = service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?
        .revision;
    let command = ExecutionControlCommand {
        request_id,
        profile_id,
        expected_revision: Some(revision),
        kind: ExecutionControlCommandKind::Queue {
            plan: plan(prompt_id),
            priority: 0,
            front: false,
        },
    };
    let acknowledgement = service
        .dispatch(command, &AcceptingExecutionActuator)
        .map_err(|error| error.to_string())?;
    match acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(assigned_attempt_id),
        } if assigned_attempt_id == attempt_id => Ok(()),
        outcome => Err(format!(
            "canonical presentation assigned {outcome:?} instead of {attempt_id:?}"
        )),
    }
}

fn record_with_events(
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    events: Vec<(Option<NodeId>, AttemptEventKind)>,
) -> Result<AttemptRecord, String> {
    let mut record = AttemptRecord::queued(profile_id, prompt_id, attempt_id);
    for (sequence, (node_id, kind)) in events.into_iter().enumerate() {
        record
            .apply(event(
                profile_id,
                prompt_id,
                attempt_id,
                sequence as u64,
                node_id,
                kind,
            ))
            .map_err(|error| error.to_string())?;
    }
    Ok(record)
}

fn lifecycle_projection_records(profile_id: ProfileId) -> Result<Vec<AttemptRecord>, String> {
    let failure = ExecutionFailure::new("projection_failure", "projected node failure")
        .at_node(NodeId("error-node".to_owned()));
    let event_sets = vec![
        Vec::new(),
        vec![
            (None, AttemptEventKind::Started),
            (
                Some(NodeId("error-node".to_owned())),
                AttemptEventKind::Progress {
                    completed: 2,
                    total: 5,
                },
            ),
        ],
        vec![
            (None, AttemptEventKind::Started),
            (
                None,
                AttemptEventKind::CancelRequested {
                    reason: "projection cancelling".to_owned(),
                    interrupt: false,
                },
            ),
        ],
        vec![
            (None, AttemptEventKind::Started),
            (None, AttemptEventKind::Succeeded),
        ],
        vec![
            (None, AttemptEventKind::Started),
            (
                Some(NodeId("error-node".to_owned())),
                AttemptEventKind::Failed { failure },
            ),
        ],
        vec![(None, AttemptEventKind::Cancelled)],
        vec![(
            None,
            AttemptEventKind::Interrupted {
                reason: "projection interrupted".to_owned(),
            },
        )],
    ];
    let provider_states = [
        ProviderAttemptState::Queued,
        ProviderAttemptState::Running,
        ProviderAttemptState::Cancelling,
        ProviderAttemptState::Succeeded,
        ProviderAttemptState::Failed,
        ProviderAttemptState::Cancelled,
        ProviderAttemptState::Interrupted,
    ];
    let mut records = Vec::new();
    for (index, (events, provider_state)) in event_sets.into_iter().zip(provider_states).enumerate()
    {
        let mut record = record_with_events(
            profile_id,
            prompt(0x800 + index as u128),
            attempt(0x800 + index as u128),
            events,
        )?;
        record.source_projection = Some(AttemptSourceProjection::Provider {
            provider_id: "projection-provider".to_owned(),
            state: provider_state,
        });
        records.push(record);
    }
    let mut unknown_provider = AttemptRecord::queued(profile_id, prompt(0x807), attempt(0x807));
    unknown_provider.source_projection = Some(AttemptSourceProjection::Provider {
        provider_id: "projection-provider".to_owned(),
        state: ProviderAttemptState::Unknown {
            raw_state: "provider-future-state".to_owned(),
        },
    });
    records.push(unknown_provider);
    let mut unknown_source = AttemptRecord::queued(profile_id, prompt(0x808), attempt(0x808));
    unknown_source.source_projection = Some(AttemptSourceProjection::Unknown {
        source_id: Some("projection-future-source".to_owned()),
        raw_state: "future-state".to_owned(),
    });
    records.push(unknown_source);
    Ok(records)
}

fn digest(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}

fn parse_csv(source: &str) -> Result<Vec<BTreeMap<String, String>>, String> {
    let mut records = Vec::<Vec<String>>::new();
    let mut record = Vec::<String>::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = source.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                field.push('"');
                characters.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => record.push(std::mem::take(&mut field)),
            '\n' if !quoted => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            '\r' if !quoted => {}
            other => field.push(other),
        }
    }
    if quoted {
        return Err("unterminated quoted CSV field".to_owned());
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    let headers = records
        .first()
        .cloned()
        .ok_or_else(|| "CSV has no header".to_owned())?;
    records
        .into_iter()
        .skip(1)
        .filter(|record| !record.iter().all(String::is_empty))
        .map(|record| {
            if record.len() != headers.len() {
                return Err(format!(
                    "CSV row has {} fields instead of {}",
                    record.len(),
                    headers.len()
                ));
            }
            Ok(headers.iter().cloned().zip(record).collect())
        })
        .collect()
}

fn field<'a>(row: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str, String> {
    row.get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("CSV row has no `{name}` field"))
}

fn parity_decision(feature_id: &str) -> Result<&str, String> {
    let prefix = format!("| `{feature_id}` |");
    let rows = PARITY_MATRIX
        .lines()
        .filter(|line| line.starts_with(&prefix))
        .collect::<Vec<_>>();
    if rows.len() != 1 {
        return Err(format!(
            "component `{feature_id}` has {} parity rows",
            rows.len()
        ));
    }
    rows[0]
        .rsplit('|')
        .nth(1)
        .map(str::trim)
        .ok_or_else(|| format!("component `{feature_id}` has no decision"))
}

fn record_native_behavior(
    results: &mut BTreeMap<String, Value>,
    feature_id: &str,
    runtime_assertion: &str,
) -> Result<(), String> {
    if execution_feature_disposition(feature_id) != Some(ExecutionFeatureDisposition::Native) {
        return Err(format!(
            "runtime evidence was recorded for non-native feature `{feature_id}`"
        ));
    }
    if results
        .insert(
            feature_id.to_owned(),
            json!({
                "feature_id": feature_id,
                "passed": true,
                "runtime_assertion": runtime_assertion,
            }),
        )
        .is_some()
    {
        return Err(format!(
            "runtime evidence for native feature `{feature_id}` was recorded twice"
        ));
    }
    Ok(())
}

fn record_component_behavior(
    results: &mut BTreeMap<&'static str, Value>,
    feature_id: &'static str,
    selector: &'static str,
    runtime_assertion: &'static str,
) -> Result<(), String> {
    if results
        .insert(
            feature_id,
            json!({
                "feature_id": feature_id,
                "selector": selector,
                "rendered": true,
                "event_state_assertion": runtime_assertion,
            }),
        )
        .is_some()
    {
        return Err(format!(
            "runtime component evidence for `{feature_id}` was recorded twice"
        ));
    }
    Ok(())
}

fn component_behavior_evidence(feature_id: &str) -> Option<(&'static str, &'static str)> {
    Some(match feature_id {
        "COMFY-FRONTEND-SURFACE-922B12C3CA3D" => {
            ("sim-error-overlay", "dismissible structured-error overlay")
        }
        "COMFY-FRONTEND-SURFACE-F7223A6667BB" => (
            "sim-execute-button",
            "capability-gated graph execute control",
        ),
        "COMFY-FRONTEND-SURFACE-19BAB3FC51C6" => {
            ("sim-queue-inline-progress-", "per-attempt inline progress")
        }
        "COMFY-FRONTEND-SURFACE-BA68BC33A2AB" => (
            "sim-queue-inline-progress-summary-",
            "accessible progress summary",
        ),
        "COMFY-FRONTEND-SURFACE-67797BF57062" => (
            "sim-queue-notification-banner",
            "request-correlated queue notification",
        ),
        "COMFY-FRONTEND-SURFACE-63A4ABE54AC4" => (
            "sim-queue-notification-banner-host",
            "sequential notification host",
        ),
        "COMFY-FRONTEND-SURFACE-F494BDB6FD2E" => (
            "sim-execution-dock-queue-overlay-active",
            "active queue group",
        ),
        "COMFY-FRONTEND-SURFACE-052D51C10184" => (
            "sim-execution-dock-queue-overlay-expanded",
            "expanded queue group",
        ),
        "COMFY-FRONTEND-SURFACE-0C01631C3DFA" => (
            "sim-execution-dock-queue-overlay-header",
            "queue overlay header",
        ),
        "COMFY-FRONTEND-SURFACE-BE5BE58D2FDE" => (
            "sim-execution-dock-queue-progress-overlay",
            "queue progress overlay",
        ),
        "COMFY-FRONTEND-SURFACE-26F40752861E" => (
            "Clear native execution history?",
            "clear-history confirmation lifecycle",
        ),
        "COMFY-FRONTEND-SURFACE-6085B98C498A" => ("sim-job-context-menu", "typed job action menu"),
        "COMFY-FRONTEND-SURFACE-F6FF6DAE75BF" => (
            "sim-job-details-hover-popover",
            "selected-job hover/details controls",
        ),
        "COMFY-FRONTEND-SURFACE-F3428874E71D" => ("sim-job-details-popover", "bounded job details"),
        "COMFY-FRONTEND-SURFACE-9F0D36286AB9" => {
            ("sim-job-filter-actions", "search/workflow/sort actions")
        }
        "COMFY-FRONTEND-SURFACE-FB9FC24AF7FA" => {
            ("sim-job-filter-tabs", "all/active/completed tabs")
        }
        "COMFY-FRONTEND-SURFACE-BC42336531A9" => {
            ("sim-job-filters-bar", "persistent job filter bar")
        }
        "COMFY-FRONTEND-SURFACE-A14F4CA91E43" => {
            ("sim-error-card-section-", "structured error detail section")
        }
        "COMFY-FRONTEND-SURFACE-E721F4A4F9B9" => ("sim-error-group-list", "filtered error group"),
        "COMFY-FRONTEND-SURFACE-97D04E89D68E" => ("sim-error-node-card-", "node-scoped error card"),
        "COMFY-FRONTEND-SURFACE-F69CDE266EDA" => ("sim-tab-errors", "searchable errors tab"),
        "COMFY-FRONTEND-SURFACE-92B3D7C9D258" => {
            ("sim-progress-toast-item", "running/terminal progress toast")
        }
        "COMFY-FRONTEND-SURFACE-228A24CC9226" => {
            ("sim-linear-progress-bar", "numeric linear progress")
        }
        "COMFY-FRONTEND-SURFACE-1F516FA6CD5A" => (
            "sim-output-history-active-queue-item",
            "active queue item outside output history scrolling",
        ),
        _ => return None,
    })
}

fn catalog_case() -> Result<Value, String> {
    let queue_rows = parse_csv(QUEUE_CATALOG)?
        .into_iter()
        .filter(|row| {
            row.get("feature_id")
                .is_some_and(|feature_id| feature_id.starts_with("COMFY-QUEUE-"))
        })
        .collect::<Vec<_>>();
    if queue_rows.len() != 119 {
        return Err(format!("queue catalog has {} rows", queue_rows.len()));
    }
    let mut queue_ids = BTreeSet::new();
    let mut current_count = 0_usize;
    let mut later_count = 0_usize;
    let mut native_count = 0_usize;
    let mut shared_count = 0_usize;
    let mut foundation_count = 0_usize;
    let mut dispositions = Vec::with_capacity(queue_rows.len());
    for row in &queue_rows {
        let feature_id = field(row, "feature_id")?;
        if !queue_ids.insert(feature_id.to_owned()) {
            return Err(format!("duplicate queue feature `{feature_id}`"));
        }
        let disposition = execution_feature_disposition(feature_id)
            .ok_or_else(|| format!("queue feature `{feature_id}` has no disposition"))?;
        if disposition.current_owner().is_some() {
            current_count += 1;
        } else {
            later_count += 1;
        }
        match disposition {
            ExecutionFeatureDisposition::Native => native_count += 1,
            ExecutionFeatureDisposition::SharedClosure { .. } => shared_count += 1,
            ExecutionFeatureDisposition::Foundation { .. } => foundation_count += 1,
            ExecutionFeatureDisposition::LaterOwned { .. } => {}
        }
        if disposition.closure_owner().trim().is_empty() {
            return Err(format!("queue feature `{feature_id}` has no closure owner"));
        }
        dispositions.push(json!({
            "feature_id": feature_id,
            "disposition": match disposition {
                ExecutionFeatureDisposition::Native => "native",
                ExecutionFeatureDisposition::SharedClosure { .. } => "shared_closure",
                ExecutionFeatureDisposition::Foundation { .. } => "foundation",
                ExecutionFeatureDisposition::LaterOwned { .. } => "later_owned",
            },
            "current_owner": disposition.current_owner(),
            "closure_owner": disposition.closure_owner(),
            "runtime_evidence_case": matches!(disposition, ExecutionFeatureDisposition::Native)
                .then_some("registered-controller-gpui-panel-graph-focus-progress-error-copy-and-navigation"),
        }));
    }
    for number in 1..=119 {
        let feature_id = format!("COMFY-QUEUE-{number:03}");
        if !queue_ids.contains(&feature_id) {
            return Err(format!("queue catalog is missing `{feature_id}`"));
        }
    }
    let command_rows = parse_csv(COMMAND_CATALOG)?;
    if command_rows.len() != 118 {
        return Err(format!("command catalog has {} rows", command_rows.len()));
    }
    let execution_commands = command_registry()
        .filter(|registration| registration.placement == NativePlacement::ExecutionDock)
        .collect::<Vec<_>>();
    if execution_commands.len() != 9 {
        return Err(format!(
            "command registry has {} execution commands",
            execution_commands.len()
        ));
    }
    let task_18_commands = task_18_commands();
    let task_27_commands = task_27_commands();
    for command_id in &task_18_commands {
        let registration = execution_commands
            .iter()
            .find(|registration| registration.command_id == *command_id)
            .ok_or_else(|| format!("missing Task 18 command `{command_id}`"))?;
        if registration.status != CommandNativeStatus::Executable
            || registration.gpui_action.is_none()
        {
            return Err(format!("Task 18 command `{command_id}` is not executable"));
        }
    }
    for command_id in &task_27_commands {
        let registration = execution_commands
            .iter()
            .find(|registration| registration.command_id == *command_id)
            .ok_or_else(|| format!("missing Task 27 command `{command_id}`"))?;
        if !matches!(registration.status, CommandNativeStatus::LaterOwned { owner } if owner == registration.owner)
        {
            return Err(format!("memory command `{command_id}` has the wrong owner"));
        }
    }

    let menu_rows = parse_csv(MENU_CATALOG)?;
    let menus_by_id = menu_rows
        .iter()
        .map(|row| Ok((field(row, "feature_id")?.to_owned(), row)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let job_and_run_menu_ids = job_and_run_menu_ids();
    let task_18_menu_ids = task_18_menu_ids();
    for feature_id in &job_and_run_menu_ids {
        menus_by_id
            .get(*feature_id)
            .ok_or_else(|| format!("missing job/run menu `{feature_id}`"))?;
        let reconciled = menu_registration(feature_id)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("native menu registry is missing `{feature_id}`"))?;
        let generated = GENERATED_MENU_CATALOG
            .iter()
            .find(|row| row.feature_id == *feature_id)
            .ok_or_else(|| format!("generated menu registry is missing `{feature_id}`"))?;
        if reconciled.owner != generated.owner {
            return Err(format!(
                "menu `{feature_id}` owner is `{}`, expected `{}`",
                reconciled.owner, generated.owner
            ));
        }
        if task_18_menu_ids.contains(feature_id)
            && reconciled.placement != NativePlacement::ExecutionDock
        {
            return Err(format!("menu `{feature_id}` is not in the execution dock"));
        }
    }

    let component_rows = parse_csv(COMPONENT_CATALOG)?;
    let component_ids = component_rows
        .iter()
        .map(|row| field(row, "feature_id").map(str::to_owned))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let component_sources = [
        EXECUTION_PANEL_SOURCE,
        EXECUTION_SURFACES_SOURCE,
        QUEUE_PANEL_SOURCE,
        HISTORY_PANEL_SOURCE,
        OUTPUT_VIEW_SOURCE,
        GRAPH_RENDER_SOURCE,
    ];
    let mut component_evidence = Vec::new();
    let execution_component_ids = execution_component_ids();
    for feature_id in &execution_component_ids {
        if !component_ids.contains(*feature_id) {
            return Err(format!("missing execution component `{feature_id}`"));
        }
        let generated = GENERATED_COMPONENT_CATALOG
            .iter()
            .find(|row| row.feature_id == *feature_id)
            .ok_or_else(|| format!("generated component registry is missing `{feature_id}`"))?;
        let disposition = match generated.disposition {
            crate::GeneratedComponentDisposition::Place => "place",
            crate::GeneratedComponentDisposition::Defer => "defer",
        };
        let expected = format!(
            "{disposition}:{:?};owner:{}",
            generated.placement, generated.owner
        );
        let actual = parity_decision(feature_id)?;
        if actual != expected {
            return Err(format!(
                "component `{feature_id}` decision is `{actual}`, expected `{expected}`"
            ));
        }
        if *feature_id != "COMFY-FRONTEND-SURFACE-6F5EE356A779" {
            let (marker, behavior) = component_behavior_evidence(feature_id).ok_or_else(|| {
                format!("Task 18 component `{feature_id}` has no behavioral evidence mapping")
            })?;
            if !component_sources
                .iter()
                .any(|source| source.contains(marker))
            {
                return Err(format!(
                    "Task 18 component `{feature_id}` lacks its implementation marker `{marker}`"
                ));
            }
            component_evidence.push(json!({
                "feature_id": feature_id,
                "implementation_marker": marker,
                "behavior": behavior,
            }));
        }
    }

    Ok(json!({
        "name": "exact-execution-catalog-reconciliation",
        "passed": true,
        "queue_features": queue_rows.len(),
        "queue_current": current_count,
        "queue_later": later_count,
        "queue_native": native_count,
        "queue_shared_closure": shared_count,
        "queue_foundation": foundation_count,
        "execution_commands": execution_commands.len(),
        "task_18_commands": task_18_commands.len(),
        "task_27_commands": task_27_commands.len(),
        "job_and_run_menus": job_and_run_menu_ids.len(),
        "execution_components": execution_component_ids.len(),
        "task_18_components": component_evidence.len(),
        "queue_dispositions": dispositions,
        "task_18_component_evidence": component_evidence,
    }))
}

fn acknowledgement_and_projection_case() -> Result<Value, String> {
    let profile_id = profile(1);
    let prompt_id = prompt(1);
    let attempt_id = attempt(1);
    let mut service = ExecutionPresentationService::new_with_first_attempt_id(128, attempt_id)
        .map_err(|error| error.to_string())?;
    service
        .initialize_profile(
            profile_id,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )
        .map_err(|error| error.to_string())?;
    let revision = service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?
        .revision;
    let command = ExecutionControlCommand {
        request_id: request(1),
        profile_id,
        expected_revision: Some(revision),
        kind: ExecutionControlCommandKind::Queue {
            plan: plan(prompt_id),
            priority: 7,
            front: true,
        },
    };
    service
        .submit(command.clone())
        .map_err(|error| error.to_string())?;
    let pending = service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?;
    if !pending.queue.is_empty()
        || !pending.attempts.is_empty()
        || pending.pending_commands.len() != 1
    {
        return Err("queue state mutated before acknowledgement".to_owned());
    }
    service
        .apply_ack(sim_runtime::ExecutionCommandAck {
            request_id: command.request_id,
            profile_id,
            outcome: ExecutionCommandOutcome::Accepted {
                assigned_attempt_id: Some(attempt_id),
            },
        })
        .map_err(|error| error.to_string())?;
    let queued = service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?;
    if queued.queue.len() != 1
        || queued.attempts.len() != 1
        || queued.attempts[0].state != AttemptState::Queued
        || !queued.pending_commands.is_empty()
    {
        return Err("acknowledged queue state was not projected".to_owned());
    }

    service
        .apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ))
        .map_err(|error| error.to_string())?;

    let terminal_specs = [
        (attempt(2), prompt(2), AttemptEventKind::Succeeded),
        (
            attempt(3),
            prompt(3),
            AttemptEventKind::Failed {
                failure: ExecutionFailure::new("terminal_failure", "terminal failed fixture"),
            },
        ),
        (
            attempt(4),
            prompt(4),
            AttemptEventKind::Interrupted {
                reason: "terminal interrupted fixture".to_owned(),
            },
        ),
    ];
    for (index, (terminal_attempt, terminal_prompt, terminal_event)) in
        terminal_specs.into_iter().enumerate()
    {
        acknowledge_queue(
            &mut service,
            profile_id,
            terminal_prompt,
            terminal_attempt,
            request(0x10 + index as u128),
        )?;
        service
            .apply_event(event(
                profile_id,
                terminal_prompt,
                terminal_attempt,
                0,
                None,
                AttemptEventKind::Started,
            ))
            .map_err(|error| error.to_string())?;
        service
            .apply_event(event(
                profile_id,
                terminal_prompt,
                terminal_attempt,
                1,
                None,
                terminal_event,
            ))
            .map_err(|error| error.to_string())?;
    }
    service
        .apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            1,
            Some(NodeId("error-node".to_owned())),
            AttemptEventKind::Progress {
                completed: 2,
                total: 4,
            },
        ))
        .map_err(|error| error.to_string())?;
    service
        .apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            2,
            None,
            AttemptEventKind::CancelRequested {
                reason: "validation cancellation".to_owned(),
                interrupt: false,
            },
        ))
        .map_err(|error| error.to_string())?;
    if service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?
        .attempts[0]
        .state
        != AttemptState::Cancelling
    {
        return Err("cancel request did not project the cancelling state".to_owned());
    }
    service
        .apply_event(event(
            profile_id,
            prompt_id,
            attempt_id,
            3,
            None,
            AttemptEventKind::Cancelled,
        ))
        .map_err(|error| error.to_string())?;

    let lifecycle_states = service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?
        .attempts
        .into_iter()
        .map(|attempt| attempt.state)
        .collect::<Vec<_>>();
    for state in [
        AttemptState::Cancelled,
        AttemptState::Succeeded,
        AttemptState::Failed,
        AttemptState::Interrupted,
    ] {
        if !lifecycle_states.contains(&state) {
            return Err(format!("lifecycle fixture did not project `{state:?}`"));
        }
    }

    let retry_revision = service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?
        .revision;
    let retry_command = ExecutionControlCommand {
        request_id: request(2),
        profile_id,
        expected_revision: Some(retry_revision),
        kind: ExecutionControlCommandKind::Retry {
            attempt_id,
            source: RetryPromptSource::OriginalPrompt,
            replacement_plan: None,
        },
    };
    let retry_ack = service
        .dispatch(retry_command, &AcceptingExecutionActuator)
        .map_err(|error| error.to_string())?;
    let retry_attempt_id = match retry_ack.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(attempt_id),
        } => attempt_id,
        ref outcome => return Err(format!("retry acknowledgement was `{outcome:?}`")),
    };
    let retried = service
        .snapshot(profile_id)
        .map_err(|error| error.to_string())?
        .attempts
        .into_iter()
        .find(|candidate| candidate.attempt_id == retry_attempt_id)
        .ok_or_else(|| "retry attempt was not projected".to_owned())?;
    if retried.retry_of != Some(attempt_id)
        || retried.prompt_id != prompt_id
        || retried.retry_source != Some(RetryPromptSource::OriginalPrompt)
    {
        return Err("retry identity or original prompt lineage was lost".to_owned());
    }

    let statuses = [
        ExecutionSnapshotStatus::Loading,
        ExecutionSnapshotStatus::Ready,
        ExecutionSnapshotStatus::Partial {
            failure: ExecutionFailure::new("partial_fixture", "partial profile data"),
        },
        ExecutionSnapshotStatus::Stale {
            source_revision: Some(17),
            failure: ExecutionFailure::new("stale_fixture", "stale profile data"),
        },
        ExecutionSnapshotStatus::Unavailable {
            failure: ExecutionFailure::new("unavailable_fixture", "profile unavailable"),
        },
    ];
    let sources = [
        ExecutionDataSource::Live,
        ExecutionDataSource::Persisted,
        ExecutionDataSource::Recovery,
    ];
    for (index, status) in statuses.into_iter().enumerate() {
        let status_profile = profile(0x100 + index as u128);
        let source = sources[index % sources.len()];
        service
            .initialize_profile(status_profile, source, status.clone())
            .map_err(|error| error.to_string())?;
        let snapshot = service
            .snapshot(status_profile)
            .map_err(|error| error.to_string())?;
        if snapshot.source != source || snapshot.status != status {
            return Err(format!("profile status projection drift at index {index}"));
        }
    }

    let projection_profile = profile(0x200);
    let provider_states = [
        ProviderAttemptState::Queued,
        ProviderAttemptState::Running,
        ProviderAttemptState::Cancelling,
        ProviderAttemptState::Succeeded,
        ProviderAttemptState::Failed,
        ProviderAttemptState::Cancelled,
        ProviderAttemptState::Interrupted,
        ProviderAttemptState::Unknown {
            raw_state: "provider-warming".to_owned(),
        },
    ];
    let mut records = provider_states
        .into_iter()
        .enumerate()
        .map(|(index, state)| {
            let mut record = AttemptRecord::queued(
                projection_profile,
                prompt(0x200 + index as u128),
                attempt(0x200 + index as u128),
            );
            record.source_projection = Some(AttemptSourceProjection::Provider {
                provider_id: "native-provider".to_owned(),
                state,
            });
            record
        })
        .collect::<Vec<_>>();
    let mut unknown_record =
        AttemptRecord::queued(projection_profile, prompt(0x210), attempt(0x210));
    unknown_record.source_projection = Some(AttemptSourceProjection::Unknown {
        source_id: Some("future-source".to_owned()),
        raw_state: "future-state".to_owned(),
    });
    records.push(unknown_record);
    service
        .reconcile(ExecutionReconciliation {
            profile_id: projection_profile,
            source_revision: 1,
            source: ExecutionDataSource::Recovery,
            status: ExecutionSnapshotStatus::Ready,
            queue: Vec::new(),
            records,
            plans: Vec::new(),
            acknowledged_requests: Vec::new(),
        })
        .map_err(|error| error.to_string())?;
    let projected = service
        .snapshot(projection_profile)
        .map_err(|error| error.to_string())?;
    if projected.attempts.len() != 9
        || projected
            .attempts
            .iter()
            .filter(|attempt| {
                matches!(
                    attempt.source_projection,
                    Some(AttemptSourceProjection::Provider { .. })
                )
            })
            .count()
            != 8
        || !projected.attempts.iter().any(|attempt| {
            matches!(
                attempt.source_projection,
                Some(AttemptSourceProjection::Unknown { .. })
            )
        })
    {
        return Err("provider or unknown projection was lost".to_owned());
    }

    Ok(json!({
        "name": "acknowledgement-lifecycle-source-provider-and-retry-projection",
        "passed": true,
        "pending_before_ack": pending.pending_commands.len(),
        "attempts_after_ack": queued.attempts.len(),
        "retry_identity_preserved": true,
        "snapshot_statuses": 5,
        "data_sources": 3,
        "provider_state_projections": 8,
        "unknown_source_projections": 1,
        "terminal_lifecycle_states": 4,
    }))
}

fn output_availability_case(cx: &mut TestAppContext) -> Result<Value, String> {
    let profile_id = profile(3);
    let created_at = AttemptRecord::queued(profile_id, prompt(3), attempt(3)).created_at;
    let variants = vec![
        ExecutionOutputAvailability::Ready {
            reference: "native://ready".to_owned(),
            byte_length: 64,
        },
        ExecutionOutputAvailability::Missing {
            reference: Some("native://missing".to_owned()),
            reason: "file missing".to_owned(),
        },
        ExecutionOutputAvailability::Expired {
            reference: Some("native://expired".to_owned()),
            expired_at: created_at,
            reason: "lease expired".to_owned(),
        },
        ExecutionOutputAvailability::ExternallyDeleted {
            reference: "native://deleted".to_owned(),
            detected_at: created_at,
        },
        ExecutionOutputAvailability::Forbidden {
            reason: "capability denied".to_owned(),
        },
        ExecutionOutputAvailability::Unsupported {
            media_type: "application/x-future".to_owned(),
            reason: "unknown viewer".to_owned(),
        },
        ExecutionOutputAvailability::Corrupt {
            reference: Some("native://corrupt".to_owned()),
            reason: "checksum mismatch".to_owned(),
        },
    ];
    let outputs = variants
        .into_iter()
        .enumerate()
        .map(|(index, availability)| ExecutionOutput {
            output_id: identifier(0x5000 + index as u128),
            node_id: NodeId("error-node".to_owned()),
            output_index: index,
            name: format!("output-{index}"),
            media_kind: match index {
                0 => OutputMediaKind::ThreeD,
                1 => OutputMediaKind::Unknown,
                _ => OutputMediaKind::Image,
            },
            media_type: "image/png".to_owned(),
            subfolder: Some("validation".to_owned()),
            storage_type: Some("native".to_owned()),
            metadata: BTreeMap::from([("fixture".to_owned(), json!(index))]),
            view_reference: Some(format!("native://view/{index}")),
            download_reference: Some(format!("native://download/{index}")),
            availability,
            created_at,
        })
        .collect::<Vec<_>>();
    let recoverable = outputs
        .iter()
        .filter(|output| output.recovery_eligibility().is_allowed())
        .count();
    let removable = outputs
        .iter()
        .filter(|output| output.removal_eligibility().is_allowed())
        .count();
    if recoverable != 4 || removable != 5 {
        return Err(format!(
            "output eligibility drift: recoverable={recoverable}, removable={removable}"
        ));
    }
    if !outputs
        .iter()
        .any(|output| output.media_kind == OutputMediaKind::ThreeD)
        || !outputs
            .iter()
            .any(|output| output.media_kind == OutputMediaKind::Unknown)
    {
        return Err("3D or unknown output media projection is missing".to_owned());
    }

    let projection_profile = profile(0x301);
    let projection_prompt = prompt(0x301);
    let projection_attempt = attempt(0x301);
    let mut service =
        ExecutionPresentationService::new_with_first_attempt_id(16, projection_attempt)
            .map_err(|error| error.to_string())?;
    service
        .initialize_profile(
            projection_profile,
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Ready,
        )
        .map_err(|error| error.to_string())?;
    acknowledge_queue(
        &mut service,
        projection_profile,
        projection_prompt,
        projection_attempt,
        request(0x301),
    )?;
    let first_output = ExecutionOutput {
        output_id: identifier(0x7101),
        node_id: NodeId("z-node".to_owned()),
        output_index: 1,
        name: "z-output".to_owned(),
        media_kind: OutputMediaKind::ThreeD,
        media_type: "model/gltf-binary".to_owned(),
        subfolder: Some("models".to_owned()),
        storage_type: Some("native-artifact".to_owned()),
        metadata: BTreeMap::from([("revision".to_owned(), json!(1))]),
        view_reference: Some("native://view/z".to_owned()),
        download_reference: Some("native://download/z".to_owned()),
        availability: ExecutionOutputAvailability::Ready {
            reference: "native://ready/z".to_owned(),
            byte_length: 128,
        },
        created_at,
    };
    let second_output = ExecutionOutput {
        output_id: identifier(0x7102),
        node_id: NodeId("a-node".to_owned()),
        output_index: 0,
        name: "a-output".to_owned(),
        media_kind: OutputMediaKind::Unknown,
        media_type: "application/x-native-future".to_owned(),
        subfolder: Some("future".to_owned()),
        storage_type: Some("provider".to_owned()),
        metadata: BTreeMap::from([("opaque".to_owned(), json!({"kept": true}))]),
        view_reference: Some("native://view/a".to_owned()),
        download_reference: Some("native://download/a".to_owned()),
        availability: ExecutionOutputAvailability::Ready {
            reference: "native://ready/a".to_owned(),
            byte_length: 256,
        },
        created_at,
    };
    let updated_first_output = ExecutionOutput {
        metadata: BTreeMap::from([("revision".to_owned(), json!(2))]),
        ..first_output.clone()
    };
    let missing_output_id = identifier(0x7103);
    let missing_output = ExecutionOutput {
        output_id: missing_output_id,
        node_id: NodeId("m-node".to_owned()),
        output_index: 0,
        name: "missing-output".to_owned(),
        media_kind: OutputMediaKind::Image,
        media_type: "image/png".to_owned(),
        subfolder: Some("recovery".to_owned()),
        storage_type: Some("native-artifact".to_owned()),
        metadata: BTreeMap::from([("recoverable".to_owned(), json!(true))]),
        view_reference: None,
        download_reference: None,
        availability: ExecutionOutputAvailability::Missing {
            reference: Some("native://missing/committed".to_owned()),
            reason: "fixture artifact was evicted".to_owned(),
        },
        created_at,
    };
    let canonical_events = [
        event(
            projection_profile,
            projection_prompt,
            projection_attempt,
            0,
            None,
            AttemptEventKind::Started,
        ),
        event(
            projection_profile,
            projection_prompt,
            projection_attempt,
            1,
            Some(first_output.node_id.clone()),
            AttemptEventKind::OutputAvailable {
                output: first_output,
            },
        ),
        event(
            projection_profile,
            projection_prompt,
            projection_attempt,
            2,
            Some(second_output.node_id.clone()),
            AttemptEventKind::OutputAvailable {
                output: second_output,
            },
        ),
        event(
            projection_profile,
            projection_prompt,
            projection_attempt,
            3,
            Some(updated_first_output.node_id.clone()),
            AttemptEventKind::OutputAvailable {
                output: updated_first_output,
            },
        ),
        event(
            projection_profile,
            projection_prompt,
            projection_attempt,
            4,
            Some(missing_output.node_id.clone()),
            AttemptEventKind::OutputAvailable {
                output: missing_output,
            },
        ),
        event(
            projection_profile,
            projection_prompt,
            projection_attempt,
            5,
            None,
            AttemptEventKind::Succeeded,
        ),
    ];
    for event in canonical_events.iter().cloned() {
        let sequence = event.sequence;
        service
            .apply_event(event)
            .map_err(|error| format!("live output event {sequence}: {error}"))?;
    }
    let live_attempt = service
        .snapshot(projection_profile)
        .map_err(|error| error.to_string())?
        .attempts
        .into_iter()
        .find(|attempt| attempt.attempt_id == projection_attempt)
        .ok_or_else(|| "successful output attempt was not projected".to_owned())?;
    if live_attempt.state != AttemptState::Succeeded
        || live_attempt.outputs.len() != 3
        || live_attempt.outputs[0].node_id.0 != "a-node"
        || live_attempt.outputs[2].metadata.get("revision") != Some(&json!(2))
    {
        return Err(
            "live output identity, order, metadata, or success projection drifted".to_owned(),
        );
    }
    let record = service
        .persisted_attempts(projection_profile)
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|attempt| attempt.record.attempt_id == projection_attempt)
        .map(|attempt| attempt.record)
        .ok_or_else(|| "live output attempt had no retained canonical record".to_owned())?;
    service
        .reconcile(ExecutionReconciliation {
            profile_id: projection_profile,
            source_revision: 1,
            source: ExecutionDataSource::Recovery,
            status: ExecutionSnapshotStatus::Ready,
            queue: Vec::new(),
            records: vec![record],
            plans: vec![sim_runtime::AttemptPlanSnapshot {
                attempt_id: projection_attempt,
                plan: plan(projection_prompt),
            }],
            acknowledged_requests: Vec::new(),
        })
        .map_err(|error| error.to_string())?;
    let reconciled_attempt = service
        .snapshot(projection_profile)
        .map_err(|error| error.to_string())?
        .attempts
        .into_iter()
        .find(|attempt| attempt.attempt_id == projection_attempt)
        .ok_or_else(|| "reconciled successful attempt disappeared".to_owned())?;
    if reconciled_attempt.outputs.len() != 3
        || reconciled_attempt.canonical_event_count != 6
        || reconciled_attempt.outputs[2].metadata.get("revision") != Some(&json!(2))
    {
        return Err("success plus reconciliation duplicated or lost output state".to_owned());
    }

    let reconciled_output_count = reconciled_attempt.outputs.len();
    let reconciled_event_count = reconciled_attempt.canonical_event_count;
    let reference_handler = Arc::new(RecordingReferenceHandler::default());
    let operation_handler = Arc::new(RecordingOperationHandler::default());
    let model = cx.update(|cx| {
        cx.new(|_| ExecutionUiModel::new(service, Arc::new(AcceptingExecutionActuator)))
    });
    model
        .update(cx, |model, cx| {
            model.register_output_reference_handler(reference_handler.clone(), cx);
            model.register_output_operation_handler(operation_handler.clone(), cx);
            model.handle_output_reference(
                projection_profile,
                ExecutionOutputReferenceAction::View,
                "native://view/0",
            )?;
            model.handle_output_reference(
                projection_profile,
                ExecutionOutputReferenceAction::Download,
                "native://download/0",
            )?;
            model.handle_output_operation(
                projection_profile,
                projection_attempt,
                missing_output_id,
                ExecutionOutputOperationAction::Recover,
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    cx.run_until_parked();
    model
        .update(cx, |model, cx| {
            model.handle_output_operation(
                projection_profile,
                projection_attempt,
                missing_output_id,
                ExecutionOutputOperationAction::Remove,
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    cx.run_until_parked();
    let canonical_operation_availability = model
        .read_with(cx, |model, _| model.snapshot(projection_profile))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == projection_attempt)
        .and_then(|attempt| {
            attempt
                .outputs
                .iter()
                .find(|output| output.output_id == missing_output_id)
        })
        .map(|output| output.availability.clone());
    if !matches!(
        canonical_operation_availability,
        Some(ExecutionOutputAvailability::Removed { .. })
    ) {
        return Err(format!(
            "typed recovery/removal did not update canonical presentation state: {canonical_operation_availability:?}"
        ));
    }
    let recorded_references = reference_handler
        .actions
        .lock()
        .map_err(|_| "reference recorder poisoned".to_owned())?;
    if recorded_references.len() != 2
        || recorded_references[0].1 != ExecutionOutputReferenceAction::View
        || recorded_references[1].1 != ExecutionOutputReferenceAction::Download
    {
        return Err("typed view/download actions were not preserved".to_owned());
    }
    let recorded_operations = operation_handler
        .actions
        .lock()
        .map_err(|_| "operation recorder poisoned".to_owned())?;
    if recorded_operations.len() != 2
        || recorded_operations[0].3 != ExecutionOutputOperationAction::Recover
        || recorded_operations[1].3 != ExecutionOutputOperationAction::Remove
        || recorded_operations
            .iter()
            .any(|operation| operation.2 != missing_output_id)
    {
        return Err("typed output recovery/removal actions were not preserved".to_owned());
    }

    Ok(json!({
        "name": "typed-output-availability-recovery-removal-and-reference-actions",
        "passed": true,
        "availability_variants": outputs.len(),
        "recoverable": recoverable,
        "removable": removable,
        "typed_reference_actions": recorded_references.len(),
        "typed_recovery_removal_actions": recorded_operations.len(),
        "canonical_post_operation_state": "removed",
        "media_kinds_include_3d_and_unknown": true,
        "committed_output_count": reconciled_output_count,
        "canonical_output_events": reconciled_event_count,
        "success_reconciliation_duplicates": 0,
    }))
}

fn event_reduction_case(cx: &mut TestAppContext) -> Result<Value, String> {
    let profile_id = profile(4);
    let other_profile_id = profile(5);
    let prompt_id = prompt(4);
    let plan = plan(prompt_id);
    let model = cx.update(|cx| {
        let service = ExecutionPresentationService::new(16).expect("valid presentation capacity");
        cx.new(|_| ExecutionUiModel::new(service, Arc::new(AcceptingExecutionActuator)))
    });
    let acknowledgement = model
        .update(cx, |model, cx| {
            model.initialize_profile(
                profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
                cx,
            )?;
            model.initialize_profile(
                other_profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
                cx,
            )?;
            model.register_plan_provider(Arc::new(DeterministicPlanProvider { plan }), cx);
            model.queue(
                ExecutionPlanRequest {
                    profile_id,
                    document_identity: "val-gpui-005-large-stream".to_owned(),
                    workflow_bytes: br#"{"version":0.4,"nodes":[],"links":[]}"#.to_vec(),
                    selected_output_nodes: BTreeSet::new(),
                },
                0,
                false,
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    let attempt_id = match acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(attempt_id),
        } => attempt_id,
        outcome => return Err(format!("large-stream queue was not accepted: {outcome:?}")),
    };
    let mut events = Vec::with_capacity(10_001);
    events.push(event(
        profile_id,
        prompt_id,
        attempt_id,
        0,
        None,
        AttemptEventKind::Started,
    ));
    for sequence in 1..=10_000_u64 {
        let node_id = NodeId("stream-node".to_owned());
        let kind = if sequence < 10_000 && sequence % 1_000 == 0 {
            AttemptEventKind::Preview {
                preview: sim_runtime::ExecutionPreview {
                    preview_id: identifier(0x6000 + sequence as u128),
                    node_id: node_id.clone(),
                    revision: sequence,
                    frame_index: Some(sequence / 1_000),
                    output_index: Some(0),
                    media_kind: OutputMediaKind::Image,
                    media_type: "image/png".to_owned(),
                    width: Some(8),
                    height: Some(8),
                    encoded_bytes: vec![sequence as u8; 8],
                },
            }
        } else {
            AttemptEventKind::Progress {
                completed: sequence,
                total: 10_000,
            }
        };
        events.push(event(
            profile_id,
            prompt_id,
            attempt_id,
            sequence,
            Some(node_id),
            kind,
        ));
    }
    let notifications_before = model.read_with(cx, |model, _| model.notification_batches());
    let event_bus =
        sim_runtime::ExecutionEventBus::new(events.len()).map_err(|error| error.to_string())?;
    let attached = model.update(cx, |model, cx| {
        model.attach_event_bus(event_bus.clone(), cx)
    });
    if !attached
        || model.update(cx, |model, cx| {
            model.attach_event_bus(event_bus.clone(), cx)
        })
    {
        return Err("execution event bus did not enforce one model subscription".to_owned());
    }
    let presentation = model.read_with(cx, |model, _| model.shared_service());
    for event in &events {
        smol::block_on(presentation.apply_event_durable(event.clone()))
            .map_err(|error| error.to_string())?;
    }
    for event in events {
        event_bus
            .publish(event)
            .map_err(|error| error.to_string())?;
    }
    drop(event_bus);
    cx.run_until_parked();
    let (notifications_after, projected) = model.read_with(cx, |model, _| {
        (
            model.notification_batches(),
            model.attempt(profile_id, attempt_id),
        )
    });
    let projected = projected
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "large-stream attempt disappeared".to_owned())?;
    if notifications_after != notifications_before + 1
        || projected.canonical_event_count != 10_001
        || projected.last_sequence != Some(10_000)
        || projected
            .progress
            .as_ref()
            .is_none_or(|progress| progress.completed != 10_000 || progress.total != 10_000)
        || projected
            .preview
            .as_ref()
            .is_none_or(|preview| preview.revision != 9_000)
    {
        return Err(
            "10,000-event canonical reduction lost final state or batched notification".to_owned(),
        );
    }
    let invalid_events = vec![
        event(
            other_profile_id,
            prompt_id,
            attempt_id,
            10_001,
            None,
            AttemptEventKind::Succeeded,
        ),
        event(
            profile_id,
            prompt_id,
            attempt_id,
            10_000,
            None,
            AttemptEventKind::Succeeded,
        ),
        event(
            profile_id,
            prompt_id,
            attempt_id,
            10_002,
            None,
            AttemptEventKind::Succeeded,
        ),
    ];
    model.update(cx, |model, cx| model.ingest_event_batch(invalid_events, cx));
    let diagnostic_kinds = model.read_with(cx, |model, _| {
        model
            .diagnostics()
            .map(|diagnostic| diagnostic.kind)
            .collect::<Vec<_>>()
    });
    if !diagnostic_kinds.contains(&ExecutionDiagnosticKind::CrossProfile)
        || !diagnostic_kinds.contains(&ExecutionDiagnosticKind::Duplicate)
        || !diagnostic_kinds.contains(&ExecutionDiagnosticKind::Gap)
    {
        return Err(format!(
            "cross-profile/stale/gap diagnostics incomplete: {diagnostic_kinds:?}"
        ));
    }

    let reconciliation_profile = profile(6);
    let stale_result = model.update(cx, |model, cx| {
        model.initialize_profile(
            reconciliation_profile,
            ExecutionDataSource::Persisted,
            ExecutionSnapshotStatus::Ready,
            cx,
        )?;
        model.reconcile(
            ExecutionReconciliation {
                profile_id: reconciliation_profile,
                source_revision: 9,
                source: ExecutionDataSource::Persisted,
                status: ExecutionSnapshotStatus::Ready,
                queue: Vec::new(),
                records: Vec::new(),
                plans: Vec::new(),
                acknowledged_requests: Vec::new(),
            },
            cx,
        )?;
        model.reconcile(
            ExecutionReconciliation {
                profile_id: reconciliation_profile,
                source_revision: 9,
                source: ExecutionDataSource::Recovery,
                status: ExecutionSnapshotStatus::Ready,
                queue: Vec::new(),
                records: Vec::new(),
                plans: Vec::new(),
                acknowledged_requests: Vec::new(),
            },
            cx,
        )
    });
    if stale_result.is_ok() {
        return Err("stale reconciliation was accepted".to_owned());
    }
    let stale_snapshot = model
        .read_with(cx, |model, _| model.snapshot(reconciliation_profile))
        .map_err(|error| error.to_string())?;
    if !matches!(stale_snapshot.status, ExecutionSnapshotStatus::Stale { .. }) {
        return Err("stale reconciliation did not expose a stale snapshot".to_owned());
    }

    Ok(json!({
        "name": "cross-profile-stale-rejection-and-large-canonical-event-reduction",
        "passed": true,
        "canonical_events": projected.canonical_event_count,
        "notification_batches_for_stream": notifications_after - notifications_before,
        "final_progress": projected.progress,
        "final_preview_revision": projected.preview.map(|preview| preview.revision),
        "diagnostic_kinds": diagnostic_kinds.into_iter().map(|kind| format!("{kind:?}")).collect::<Vec<_>>(),
        "stale_reconciliation_visible": true,
    }))
}

fn production_controller_boundary_case(cx: &mut TestAppContext) -> Result<Value, String> {
    let production_model = cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
        execution_ui_model(cx).ok_or_else(|| {
            "production initialization did not install the native execution model".to_owned()
        })
    })?;
    if production_model.read_with(cx, |model, _| model.runtime_controller_available()) {
        return Err(
            "production initialization incorrectly installed a synthetic runtime controller"
                .to_owned(),
        );
    }
    if production_model.read_with(cx, |model, _| model.output_reference_actions_available())
        || EXECUTION_MODEL_SOURCE.contains("model.register_output_reference_handler(outputs")
    {
        return Err(
            "production exposed view/download actions without a real native viewer or save adapter"
                .to_owned(),
        );
    }
    let before = production_model
        .read_with(cx, |model, _| model.snapshot(LOCAL_EXECUTION_PROFILE_ID))
        .map_err(|error| error.to_string())?;
    let rejection = production_model
        .update(cx, |model, cx| {
            model.dispatch(
                LOCAL_EXECUTION_PROFILE_ID,
                ExecutionControlCommandKind::ClearPending {
                    reason: "VAL-GPUI-005 production fail-closed boundary".to_owned(),
                },
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    let ExecutionCommandOutcome::Rejected { failure } = rejection.outcome else {
        return Err(
            "production controller accepted a runtime command while disconnected".to_owned(),
        );
    };
    if failure.origin != ExecutionFailureOrigin::Transport
        || failure.code != "runtime_controller_unavailable"
    {
        return Err(format!(
            "production controller returned the wrong typed rejection: {failure:?}"
        ));
    }
    let after = production_model
        .read_with(cx, |model, _| model.snapshot(LOCAL_EXECUTION_PROFILE_ID))
        .map_err(|error| error.to_string())?;
    if before.queue != after.queue
        || before.attempts != after.attempts
        || !after.pending_commands.is_empty()
    {
        return Err("production fail-closed rejection mutated execution state".to_owned());
    }

    let registered_controller_outcome = cx.update(|cx| {
        let mut service =
            ExecutionPresentationService::new(16).map_err(|error| error.to_string())?;
        service
            .initialize_profile(
                profile(0x6f0),
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .map_err(|error| error.to_string())?;
        let registered_model =
            cx.new(|_| ExecutionUiModel::new_without_runtime_controller(service));
        registered_model.update(cx, |model, cx| {
            model.register_runtime_controller(Arc::new(AcceptingExecutionActuator), cx);
            if !model.runtime_controller_available() {
                return Err(
                    "explicit controller registration did not enable runtime controls".to_owned(),
                );
            }
            model
                .dispatch(
                    profile(0x6f0),
                    ExecutionControlCommandKind::ClearPending {
                        reason: "VAL-GPUI-005 explicitly registered test controller".to_owned(),
                    },
                    cx,
                )
                .map_err(|error| error.to_string())
        })
    })?;
    if !matches!(
        registered_controller_outcome.outcome,
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: None
        }
    ) {
        return Err(format!(
            "explicitly registered test controller did not acknowledge the command: {:?}",
            registered_controller_outcome.outcome
        ));
    }

    let (production_panel, production_window) =
        cx.add_window_view(|_, cx| ExecutionPanel::test_new(production_model.clone(), cx));
    let rejected_queue = production_model
        .update(production_window, |model, cx| {
            model.dispatch(
                LOCAL_EXECUTION_PROFILE_ID,
                ExecutionControlCommandKind::Queue {
                    plan: plan(prompt(0x6f1)),
                    priority: 0,
                    front: false,
                },
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    if !matches!(
        rejected_queue.outcome,
        ExecutionCommandOutcome::Rejected { .. }
    ) {
        return Err("production disconnected queue fixture was not rejected".to_owned());
    }
    production_window.run_until_parked();
    let failure_banner =
        production_panel.read_with(production_window, |panel, _| panel.surface_state_for_test());
    let failure_banner_identity =
        failure_banner
            .current_notification_identity
            .ok_or_else(|| {
                "production queue rejection did not create a request-correlated banner".to_owned()
            })?;
    if failure_banner.notification_count != 1
        || !matches!(
            failure_banner.current_notification,
            Some((ExecutionNotificationKind::Failure, Some(request_id), 1, ref message))
                if request_id == rejected_queue.request_id
                    && message.contains("runtime_controller_unavailable")
        )
    {
        return Err("production queue rejection did not upgrade its banner to failure".to_owned());
    }
    production_panel.update_in(production_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::DismissNotification(failure_banner_identity),
            window,
            cx,
        );
    });

    Ok(json!({
        "name": "production-fail-closed-and-explicit-controller-registration",
        "passed": true,
        "production_runtime_controller_available": false,
        "production_view_download_visibly_later_owned": true,
        "production_rejection_origin": "Transport",
        "production_state_unchanged": true,
        "explicit_test_controller_registered": true,
        "explicit_test_controller_acknowledged": true,
        "production_queue_failure_banner": true,
    }))
}

fn graph_fixture(profile_id: ProfileId) -> Result<GraphWorkspaceModel, Box<dyn Error>> {
    let mut model = GraphWorkspaceModel::create("VAL-GPUI-005 execution graph")?;
    if let WorkflowOpenState::Editable(engine) = &mut model.open_state {
        engine.document.profile_identity = Some(profile_id.0);
    }
    model.apply(GraphCommand::AddNode {
        node: GraphNode::new(
            GraphIdentifier::from("error-node"),
            "NativeExecutionFixture",
            "Error node",
            GraphPoint { x: 420.0, y: 220.0 },
        ),
        source: sim_runtime::NodeCreationSource::Library,
    })?;
    model.apply(GraphCommand::AddSubgraphDefinition {
        definition: SubgraphDefinition {
            identifier: GraphIdentifier::from("validation-subgraph"),
            name: "Validation subgraph".to_owned(),
            graph: Box::new(GraphLevel::default()),
            inputs: Vec::new(),
            outputs: Vec::new(),
            published: false,
            description: "VAL-GPUI-005 navigation boundary".to_owned(),
            search_aliases: Vec::new(),
            exposed_widgets: Vec::new(),
            graph_inline: false,
            unknown: BTreeMap::new(),
        },
    })?;
    Ok(model)
}

fn native_image_graph_fixture(
    profile_id: ProfileId,
) -> Result<GraphWorkspaceModel, Box<dyn Error>> {
    let workflow = serde_json::to_vec(&json!({
        "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
        "2": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "nearest-exact",
                "width": 4,
                "height": 0,
                "crop": "disabled"
            }
        },
        "3": {"class_type": "ImageInvert", "inputs": {"image": ["2", 0]}},
        "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
        "5": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "task19-native"}
        }
    }))?;
    let mut model = GraphWorkspaceModel::open(
        "Task 19 native image interaction",
        "task19-native-image-interaction",
        sim_runtime::WorkflowStorageProvider::Draft,
        workflow,
    )?;
    let WorkflowOpenState::Editable(engine) = &mut model.open_state else {
        return Err("native image graph fixture opened read-only".into());
    };
    engine.document.profile_identity = Some(profile_id.0);
    for node in engine.document.root.nodes.values_mut() {
        let widget_names = node
            .widgets
            .iter()
            .map(|widget| Value::String(widget.identifier.clone()))
            .collect::<Vec<_>>();
        node.source_fields.insert(
            "properties".to_owned(),
            json!({"widget_input_names": widget_names}),
        );
    }
    engine
        .document
        .root
        .selection
        .nodes
        .insert(GraphIdentifier::from("5"));
    Ok(model)
}

fn native_image_output(
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    output_id: Uuid,
    name: &str,
    reference: &str,
) -> ExecutionOutput {
    ExecutionOutput {
        output_id,
        node_id: NodeId("5".to_owned()),
        output_index: 0,
        name: name.to_owned(),
        media_kind: OutputMediaKind::Image,
        media_type: "image/png".to_owned(),
        subfolder: Some("task19".to_owned()),
        storage_type: Some("native-asset".to_owned()),
        metadata: BTreeMap::from([
            ("native".to_owned(), json!(true)),
            ("prompt_id".to_owned(), json!(prompt_id.0)),
        ]),
        view_reference: Some(reference.to_owned()),
        download_reference: Some(reference.to_owned()),
        availability: ExecutionOutputAvailability::Ready {
            reference: reference.to_owned(),
            byte_length: 128,
        },
        created_at: AttemptRecord::queued(profile_id, prompt_id, attempt_id).created_at,
    }
}

fn profile_bound_confirmation_fixture(
    cx: &mut TestAppContext,
    fixture_id: u128,
) -> Result<
    (
        gpui::Entity<ExecutionUiModel>,
        Arc<RecordingOperationHandler>,
        ProfileId,
        ProfileId,
        AttemptId,
        Uuid,
    ),
    String,
> {
    let originating_profile_id = profile(fixture_id);
    let switched_profile_id = profile(fixture_id.saturating_add(1));
    let originating_prompt_id = prompt(fixture_id);
    let originating_attempt_id = attempt(fixture_id);
    let output_id = identifier(fixture_id.saturating_add(0x100));
    let originating_record = record_with_events(
        originating_profile_id,
        originating_prompt_id,
        originating_attempt_id,
        vec![
            (None, AttemptEventKind::Started),
            (
                Some(NodeId("5".to_owned())),
                AttemptEventKind::OutputAvailable {
                    output: native_image_output(
                        originating_profile_id,
                        originating_prompt_id,
                        originating_attempt_id,
                        output_id,
                        "profile-bound-output.png",
                        "zed-asset://output/profile-bound-output.png",
                    ),
                },
            ),
            (None, AttemptEventKind::Succeeded),
        ],
    )?;
    let switched_record = record_with_events(
        switched_profile_id,
        prompt(fixture_id.saturating_add(1)),
        attempt(fixture_id.saturating_add(1)),
        vec![
            (None, AttemptEventKind::Started),
            (None, AttemptEventKind::Succeeded),
        ],
    )?;
    let operation_handler = Arc::new(RecordingOperationHandler::default());
    let model = cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
        let service = ExecutionPresentationService::new(16).expect("valid presentation capacity");
        cx.new(|_| ExecutionUiModel::new(service, Arc::new(AcceptingExecutionActuator)))
    });
    model
        .update(cx, |model, cx| {
            model.initialize_profile(
                originating_profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
                cx,
            )?;
            model.initialize_profile(
                switched_profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
                cx,
            )?;
            model.reconcile(
                ExecutionReconciliation {
                    profile_id: originating_profile_id,
                    source_revision: 1,
                    source: ExecutionDataSource::Live,
                    status: ExecutionSnapshotStatus::Ready,
                    queue: Vec::new(),
                    records: vec![originating_record],
                    plans: Vec::new(),
                    acknowledged_requests: Vec::new(),
                },
                cx,
            )?;
            model.reconcile(
                ExecutionReconciliation {
                    profile_id: switched_profile_id,
                    source_revision: 1,
                    source: ExecutionDataSource::Live,
                    status: ExecutionSnapshotStatus::Ready,
                    queue: Vec::new(),
                    records: vec![switched_record],
                    plans: Vec::new(),
                    acknowledged_requests: Vec::new(),
                },
                cx,
            )?;
            model.set_active_profile(originating_profile_id, cx)?;
            model.register_output_operation_handler(operation_handler.clone(), cx);
            Ok::<_, ExecutionUiModelError>(())
        })
        .map_err(|error| error.to_string())?;
    Ok((
        model,
        operation_handler,
        originating_profile_id,
        switched_profile_id,
        originating_attempt_id,
        output_id,
    ))
}

fn parent_subgraph_error_projection_case() -> Result<Value, String> {
    let mut parent = GraphNode::new(
        GraphIdentifier::from("parent-subgraph"),
        "SimSubgraph",
        "Parent subgraph",
        GraphPoint { x: 40.0, y: 40.0 },
    );
    parent.subgraph_definition = Some(GraphIdentifier::from("validation-subgraph"));
    let interior_failure =
        ExecutionFailure::new("interior_failure", "an interior subgraph node failed")
            .at_node(NodeId("parent-subgraph::interior-node".to_owned()));
    let unrelated_failure = ExecutionFailure::new("unrelated_failure", "different node")
        .at_node(NodeId("different-parent::interior-node".to_owned()));
    let root_node_failure = ExecutionFailure::new("root_failure", "the root node failed")
        .at_node(NodeId("parent-subgraph".to_owned()));
    if !node_has_execution_failure(&parent, &interior_failure)
        || !node_has_execution_failure(&parent, &root_node_failure)
        || node_has_execution_failure(&parent, &unrelated_failure)
    {
        return Err("parent subgraph failure-ring projection did not preserve scope".to_owned());
    }

    Ok(json!({
        "name": "parent-subgraph-error-ring-projection",
        "passed": true,
        "exact_node_failure": true,
        "namespaced_interior_failure": true,
        "unrelated_interior_failure_rejected": true,
        "native_feature_results": [{
            "feature_id": "COMFY-QUEUE-116",
            "passed": true,
            "runtime_assertion": "a namespaced interior-node failure projected onto its parent subgraph node and rejected an unrelated parent namespace",
        }],
    }))
}

fn gpui_interaction_case(cx: &mut TestAppContext) -> Result<Value, String> {
    let profile_id = profile(7);
    let prompt_id = prompt(7);
    let mut rendered_component_evidence = BTreeMap::<&'static str, Value>::new();
    let mut native_feature_results = BTreeMap::<String, Value>::new();
    let operation_handler = Arc::new(RecordingOperationHandler::default());
    let model = cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
        let service = ExecutionPresentationService::new(128).expect("valid presentation capacity");
        let model =
            cx.new(|_| ExecutionUiModel::new(service, Arc::new(AcceptingExecutionActuator)));
        cx.set_global(GlobalExecutionUiModel(model.clone()));
        model
    });
    model
        .update(cx, |model, cx| {
            model.initialize_profile(
                profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
                cx,
            )?;
            model.set_active_profile(profile_id, cx)?;
            model.register_plan_provider(
                Arc::new(DeterministicPlanProvider {
                    plan: plan(prompt_id),
                }),
                cx,
            );
            model.register_output_operation_handler(operation_handler.clone(), cx);
            Ok::<_, ExecutionUiModelError>(())
        })
        .map_err(|error| error.to_string())?;

    let (item, window) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            graph_fixture(profile_id).expect("valid execution graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    let mut command_outcomes = item.update_in(window, |item, window, cx| {
        item.focus_graph(window, cx);
        if !item.focus_handle(cx).is_focused(window) {
            return Err("execution graph did not receive focus".to_owned());
        }
        let outcomes = ["Sim.QueuePrompt", "Sim.Queue.ToggleOverlay"]
            .into_iter()
            .map(|command_id| {
                let outcome = item.dispatch_shell_command(command_id, cx);
                (
                    command_id.to_owned(),
                    outcome.is_executed(),
                    format!("{outcome:?}"),
                )
            })
            .collect::<Vec<_>>();
        if outcomes.iter().any(|(_, executed, _)| !executed) {
            return Err(format!(
                "registered test controller did not execute initial commands: {outcomes:?}"
            ));
        }
        Ok(outcomes)
    })?;
    window.run_until_parked();
    if window.debug_bounds("SIM-EXECUTION-ACTIONBAR").is_none()
        || window.debug_bounds("SIM-EXECUTE-BUTTON").is_none()
        || window.debug_bounds("SIM-QUEUE-OVERLAY").is_none()
        || window
            .debug_bounds("SIM-EXECUTION-RUN-MODE-TRIGGER")
            .is_none()
    {
        return Err("registered-controller graph execution selectors did not render".to_owned());
    }
    record_component_behavior(
        &mut rendered_component_evidence,
        "COMFY-FRONTEND-SURFACE-F7223A6667BB",
        "SIM-EXECUTE-BUTTON",
        "mouseenter and keyboard focus rendered selected output-node feedback, mouseleave cleared it, and both focused Enter and pointer click dispatched selected-output queue commands",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-035",
        "dispatching Sim.Queue.ToggleOverlay rendered the graph queue overlay",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-068",
        "the native Run control remained rendered in the graph execution action bar",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-094",
        "the native Run control rendered in the graph top bar",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-095",
        "the queue-mode trigger rendered with real GPUI bounds",
    )?;
    item.update(window, |item, cx| {
        item.apply_graph_command(GraphCommand::SelectAll, cx)
    });
    let execute_bounds = window
        .debug_bounds("SIM-EXECUTE-BUTTON")
        .ok_or_else(|| "Execute control has no GPUI bounds".to_owned())?;
    let stable_execution_bounds = [
        "SIM-EXECUTION-ACTIONBAR",
        "SIM-EXECUTE-BUTTON",
        "SIM-EXECUTION-RUN-MODE-TRIGGER",
    ]
    .into_iter()
    .map(|selector| {
        window
            .debug_bounds(selector)
            .map(|bounds| (selector, bounds))
            .ok_or_else(|| format!("Execution control {selector} has no GPUI bounds"))
    })
    .collect::<Result<Vec<_>, _>>()?;
    let check_stable_execution_bounds = |window: &mut gpui::VisualTestContext| {
        for (selector, expected) in &stable_execution_bounds {
            let actual = window
                .debug_bounds(selector)
                .ok_or_else(|| format!("Execution control {selector} lost GPUI bounds"))?;
            if actual != *expected {
                return Err(format!(
                    "Execute feedback moved {selector}: {expected:?} -> {actual:?}"
                ));
            }
        }
        Ok::<_, String>(())
    };
    let check_execution_feedback_bounds = |window: &mut gpui::VisualTestContext| {
        let toolbar = window
            .debug_bounds("SIM-EXECUTION-ACTIONBAR")
            .ok_or_else(|| "Execution toolbar lost GPUI bounds".to_owned())?;
        let feedback = window
            .debug_bounds("SIM-EXECUTE-OUTPUT-FEEDBACK")
            .ok_or_else(|| "Execute feedback has no GPUI bounds".to_owned())?;
        let execute = window
            .debug_bounds("SIM-EXECUTE-BUTTON")
            .ok_or_else(|| "Execute control lost GPUI bounds".to_owned())?;
        let mode = window
            .debug_bounds("SIM-EXECUTION-RUN-MODE-TRIGGER")
            .ok_or_else(|| "Execution mode trigger lost GPUI bounds".to_owned())?;
        if !execute.contains(&execute_bounds.center())
            || feedback.size.width <= px(0.0)
            || feedback.size.height <= px(0.0)
            || feedback.origin.y < toolbar.bottom_right().y
            || feedback.intersects(&execute)
            || feedback.intersects(&mode)
        {
            return Err("Execute feedback obscured or moved its execution controls".to_owned());
        }
        Ok::<_, String>(())
    };
    window.simulate_mouse_move(
        execute_bounds.center(),
        None::<MouseButton>,
        Modifiers::default(),
    );
    window.run_until_parked();
    check_stable_execution_bounds(window)?;
    check_execution_feedback_bounds(window)?;
    if !item.read_with(window, |item, _| item.execute_output_feedback_hovered())
        || window.debug_bounds("SIM-EXECUTE-OUTPUT-FEEDBACK").is_none()
        || window.debug_bounds("SIM-EXECUTE-OUTPUT-TARGET").is_none()
    {
        return Err(
            "Execute mouseenter did not expose selected output-node render feedback".to_owned(),
        );
    }
    window.simulate_mouse_move(
        point(px(1.0), px(1.0)),
        None::<MouseButton>,
        Modifiers::default(),
    );
    window.run_until_parked();
    check_stable_execution_bounds(window)?;
    if item.read_with(window, |item, _| item.execute_output_feedback_hovered())
        || window.debug_bounds("SIM-EXECUTE-OUTPUT-FEEDBACK").is_some()
        || window.debug_bounds("SIM-EXECUTE-OUTPUT-TARGET").is_some()
    {
        return Err("Execute mouseleave did not clear output-node feedback".to_owned());
    }
    let attempts_before_keyboard_execute = model
        .read_with(window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .len();
    item.update_in(window, |item, window, cx| {
        let execute_focus_handle = item.control_focus_handle("execution:execute-button", cx);
        window.focus(&execute_focus_handle, cx);
    });
    window.run_until_parked();
    check_stable_execution_bounds(window)?;
    check_execution_feedback_bounds(window)?;
    if window.debug_bounds("SIM-EXECUTE-OUTPUT-FEEDBACK").is_none()
        || window.debug_bounds("SIM-EXECUTE-OUTPUT-TARGET").is_none()
    {
        return Err(
            "keyboard focus on Execute did not expose the same selected-output feedback".to_owned(),
        );
    }
    window.simulate_keystrokes("enter");
    let attempts_after_keyboard_execute = model
        .read_with(window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .len();
    if attempts_after_keyboard_execute != attempts_before_keyboard_execute.saturating_add(1)
        || item
            .read_with(window, |item, _| {
                item.shell_dispatch_trace_for_test().to_vec()
            })
            .last()
            .is_none_or(|command| command != "Sim.QueueSelectedOutputNodes")
    {
        return Err(
            "focused Enter did not dispatch QueueSelectedOutputNodes exactly once".to_owned(),
        );
    }
    let attempts_before_pointer_execute = attempts_after_keyboard_execute;
    let trace_before_pointer_execute = item
        .read_with(window, |item, _| {
            item.shell_dispatch_trace_for_test().to_vec()
        })
        .len();
    let pointer_execute_bounds = window
        .debug_bounds("SIM-EXECUTE-BUTTON")
        .ok_or_else(|| "Execute control disappeared after keyboard activation".to_owned())?;
    window.simulate_click(pointer_execute_bounds.center(), Modifiers::default());
    window.run_until_parked();
    let attempts_after_pointer_execute = model
        .read_with(window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .len();
    let trace_after_pointer_execute = item
        .read_with(window, |item, _| {
            item.shell_dispatch_trace_for_test().to_vec()
        })
        .len();
    if trace_after_pointer_execute != trace_before_pointer_execute.saturating_add(1) {
        let trace = item.read_with(window, |item, _| {
            item.shell_dispatch_trace_for_test().to_vec()
        });
        let reason = item.read_with(window, |item, cx| {
            item.execution_queue_unavailable_reason(cx)
        });
        return Err(format!(
            "clicking Execute did not dispatch one selected-output queue command: attempts_before={attempts_before_pointer_execute}, attempts_after={attempts_after_pointer_execute}, trace_before={trace_before_pointer_execute}, trace_after={trace_after_pointer_execute}, trace={trace:?}, unavailable={reason:?}"
        ));
    }
    let initially_queued_attempt_ids = model
        .read_with(window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .into_iter()
        .map(|attempt| attempt.attempt_id)
        .collect::<Vec<_>>();
    command_outcomes.push((
        "Sim.QueueSelectedOutputNodes".to_owned(),
        true,
        "Executed through focused Enter and pointer click".to_owned(),
    ));
    let mode_trigger = window
        .debug_bounds("SIM-EXECUTION-RUN-MODE-TRIGGER")
        .ok_or_else(|| "execution mode trigger did not expose debug bounds".to_owned())?;
    window.simulate_mouse_move(
        mode_trigger.center(),
        None::<MouseButton>,
        Modifiers::default(),
    );
    window.run_until_parked();
    let mode_trigger = window
        .debug_bounds("SIM-EXECUTION-RUN-MODE-TRIGGER")
        .ok_or_else(|| "execution mode trigger disappeared after pointer movement".to_owned())?;
    window.simulate_click(mode_trigger.center(), Modifiers::default());
    window.run_until_parked();
    for selector in [
        "SIM-EXECUTION-RUN-MODE-MENU",
        "SIM-EXECUTION-RUN-MODE-MANUAL",
        "SIM-EXECUTION-RUN-MODE-ON-CHANGE",
        "SIM-EXECUTION-RUN-MODE-INSTANT-IDLE",
    ] {
        if window.debug_bounds(selector).is_none() {
            return Err(format!(
                "execution mode interaction did not render `{selector}`"
            ));
        }
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-096",
        "clicking the queue-mode trigger opened the native mode menu",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-097",
        "the opened queue-mode menu rendered Manual, On change, and Instant idle choices",
    )?;
    let on_change_mode = window
        .debug_bounds("SIM-EXECUTION-RUN-MODE-ON-CHANGE")
        .ok_or_else(|| "On change execution mode has no bounds".to_owned())?;
    window.simulate_click(on_change_mode.center(), Modifiers::default());
    window.run_until_parked();
    if item.read_with(window, |item, _| item.execution_run_mode()) != ExecutionRunMode::OnChange
        || item.read_with(window, |item, _| item.execution_mode_menu_open())
        || window
            .debug_bounds("SIM-EXECUTE-BUTTON-LABEL-ON-CHANGE")
            .is_none()
        || window
            .debug_bounds("SIM-EXECUTE-BUTTON-LABEL-MANUAL")
            .is_some()
    {
        return Err(
            "selecting On change did not update the exact Execute button label and close the mode menu"
                .to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-098",
        "clicking On change changed the exact Run button label from `Execute` to `Execute · On change`",
    )?;
    item.update(window, |item, cx| {
        item.choose_execution_run_mode(ExecutionRunMode::Manual, cx)
    });
    window.run_until_parked();
    if window
        .debug_bounds("SIM-EXECUTE-BUTTON-LABEL-MANUAL")
        .is_none()
        || window
            .debug_bounds("SIM-EXECUTE-BUTTON-LABEL-ON-CHANGE")
            .is_some()
    {
        return Err("restoring Manual did not restore the exact Execute label".to_owned());
    }
    let associated_attempt = item
        .read_with(window, |item, cx| item.active_execution_presentation(cx))
        .ok_or_else(|| "graph queue command created no associated attempt".to_owned())?;
    let attempt_id = associated_attempt.attempt_id;
    let attempt_prompt_id = associated_attempt.prompt_id;
    let unavailable_output_id = identifier(0x7801);
    model.update(window, |model, cx| {
        model.ingest_event_batch(
            vec![
                event(
                    profile_id,
                    attempt_prompt_id,
                    attempt_id,
                    0,
                    None,
                    AttemptEventKind::Started,
                ),
                event(
                    profile_id,
                    attempt_prompt_id,
                    attempt_id,
                    1,
                    Some(NodeId("error-node".to_owned())),
                    AttemptEventKind::Progress {
                        completed: 3,
                        total: 8,
                    },
                ),
                event(
                    profile_id,
                    attempt_prompt_id,
                    attempt_id,
                    2,
                    Some(NodeId("error-node".to_owned())),
                    AttemptEventKind::OutputAvailable {
                        output: ExecutionOutput {
                            output_id: unavailable_output_id,
                            node_id: NodeId("error-node".to_owned()),
                            output_index: 0,
                            name: "missing-preview.png".to_owned(),
                            media_kind: OutputMediaKind::Image,
                            media_type: "image/png".to_owned(),
                            subfolder: Some("validation".to_owned()),
                            storage_type: Some("native-artifact".to_owned()),
                            metadata: BTreeMap::from([(
                                "prompt_id".to_owned(),
                                json!(attempt_prompt_id.0),
                            )]),
                            view_reference: Some("native://view/missing-preview".to_owned()),
                            download_reference: Some(
                                "native://download/missing-preview".to_owned(),
                            ),
                            availability: ExecutionOutputAvailability::Missing {
                                reference: Some("native://missing-preview".to_owned()),
                                reason: "artifact evicted for recovery validation".to_owned(),
                            },
                            created_at: AttemptRecord::queued(
                                profile_id,
                                attempt_prompt_id,
                                attempt_id,
                            )
                            .created_at,
                        },
                    },
                ),
                event(
                    profile_id,
                    attempt_prompt_id,
                    attempt_id,
                    3,
                    Some(NodeId("error-node".to_owned())),
                    AttemptEventKind::OutputAvailable {
                        output: ExecutionOutput {
                            output_id: identifier(0x7802),
                            node_id: NodeId("error-node".to_owned()),
                            output_index: 1,
                            name: "available-preview.png".to_owned(),
                            media_kind: OutputMediaKind::Image,
                            media_type: "image/png".to_owned(),
                            subfolder: Some("validation".to_owned()),
                            storage_type: Some("native-artifact".to_owned()),
                            metadata: BTreeMap::from([(
                                "prompt_id".to_owned(),
                                json!(attempt_prompt_id.0),
                            )]),
                            view_reference: Some("native://view/available-preview".to_owned()),
                            download_reference: Some(
                                "native://download/available-preview".to_owned(),
                            ),
                            availability: ExecutionOutputAvailability::Ready {
                                reference: "native://view/available-preview".to_owned(),
                                byte_length: 256,
                            },
                            created_at: AttemptRecord::queued(
                                profile_id,
                                attempt_prompt_id,
                                attempt_id,
                            )
                            .created_at,
                        },
                    },
                ),
            ],
            cx,
        )
    });
    window.run_until_parked();
    let graph_source_has_numeric_progress = GRAPH_RENDER_SOURCE.contains("aria_numeric_value")
        && GRAPH_RENDER_SOURCE.contains("aria_min_numeric_value")
        && GRAPH_RENDER_SOURCE.contains("aria_max_numeric_value");
    if !graph_source_has_numeric_progress {
        return Err("graph progress lacks numeric accessibility values".to_owned());
    }
    let projected_before_navigation = item
        .read_with(window, |item, cx| item.active_execution_presentation(cx))
        .ok_or_else(|| "associated execution presentation disappeared".to_owned())?;
    if projected_before_navigation.progress.is_none()
        || projected_before_navigation.node_progress.is_empty()
        || projected_before_navigation.outputs.len() != 2
    {
        return Err(
            "progress and multiple-output fixtures were not projected before subgraph navigation"
                .to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-073",
        "one native attempt projected two distinct output artifacts",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-080",
        "one progress event updated both overall and node-scoped native progress",
    )?;
    let navigation_changed = item.update(window, |item, cx| {
        item.apply_graph_command(
            GraphCommand::OpenSubgraph {
                definition_identifier: GraphIdentifier::from("validation-subgraph"),
            },
            cx,
        )
    });
    if !navigation_changed {
        return Err("validation fixture could not enter its subgraph".to_owned());
    }
    let inside_subgraph = item
        .read_with(window, |item, cx| item.active_execution_presentation(cx))
        .ok_or_else(|| "execution association was lost inside the subgraph".to_owned())?;
    if inside_subgraph.progress.is_some()
        || !inside_subgraph.node_progress.is_empty()
        || inside_subgraph.preview.is_some()
        || !inside_subgraph.previews.is_empty()
    {
        return Err("stale execution projection leaked into a different graph scope".to_owned());
    }
    let navigation_restored = item.update(window, |item, cx| {
        item.execute_catalog_action(
            sim_runtime::CatalogGraphAction::ExitSubgraph,
            GraphActionInput::None,
            cx,
        )
    });
    if !navigation_restored {
        return Err("validation fixture could not return from its subgraph".to_owned());
    }
    let after_navigation_return = item
        .read_with(window, |item, cx| item.active_execution_presentation(cx))
        .ok_or_else(|| "execution association was lost after subgraph return".to_owned())?;
    if after_navigation_return.progress.is_some()
        || !after_navigation_return.node_progress.is_empty()
        || after_navigation_return.preview.is_some()
        || !after_navigation_return.previews.is_empty()
    {
        return Err(
            "stale execution progress reappeared after returning from a subgraph".to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-111",
        "exiting a subgraph invalidated stale node and overall progress until explicit reassociation",
    )?;
    item.update(window, |item, cx| {
        item.associate_execution_attempt(attempt_id, cx)
    });
    if item
        .read_with(window, |item, cx| item.active_execution_presentation(cx))
        .is_none_or(|attempt| attempt.progress.is_none())
    {
        return Err(
            "explicit reassociation did not restore the current execution projection".to_owned(),
        );
    }

    let failure = ExecutionFailure::new("fixture_failure", "deterministic node failure")
        .at_node(NodeId("error-node".to_owned()))
        .retryable(true);
    model.update(window, |model, cx| {
        model.ingest_event_batch(
            vec![event(
                profile_id,
                attempt_prompt_id,
                attempt_id,
                4,
                Some(NodeId("error-node".to_owned())),
                AttemptEventKind::Failed { failure },
            )],
            cx,
        )
    });
    window.run_until_parked();
    if window.debug_bounds("SIM-EXECUTION-ERROR-OVERLAY").is_none()
        || window.debug_bounds("SIM-NODE-error-node").is_none()
    {
        return Err("structured execution error overlay or node status did not render".to_owned());
    }
    let overflow_attempt_ids = model.update(window, |model, cx| {
        (0_u128..16)
            .map(|index| {
                let acknowledgement = model
                    .dispatch(
                        profile_id,
                        ExecutionControlCommandKind::Queue {
                            plan: plan(prompt(0x900 + index)),
                            priority: 0,
                            front: false,
                        },
                        cx,
                    )
                    .map_err(|error| error.to_string())?;
                match acknowledgement.outcome {
                    ExecutionCommandOutcome::Accepted {
                        assigned_attempt_id: Some(attempt_id),
                    } => Ok(attempt_id),
                    outcome => Err(format!(
                        "queue overlay overflow fixture was not accepted: {outcome:?}"
                    )),
                }
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    window.run_until_parked();
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-056",
        "a failed native prompt rendered its structured execution error dialog",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-057",
        "a failed native attempt rendered the graph execution error overlay",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-115",
        "a node-scoped execution failure projected failed state onto the exact graph node",
    )?;
    for selector in [
        "SIM-QUEUE-OVERLAY-FILTER-TABS",
        "SIM-QUEUE-OVERLAY-TAB-ALL",
        "SIM-QUEUE-OVERLAY-TAB-COMPLETED",
        "SIM-QUEUE-OVERLAY-TAB-FAILED",
        "SIM-QUEUE-OVERLAY-DOCKED-HISTORY",
    ] {
        if window.debug_bounds(selector).is_none() {
            return Err(format!(
                "expanded queue overlay did not render `{selector}`"
            ));
        }
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-086",
        "the graph queue toggle opened the expanded queue overlay",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-087",
        "the expanded overlay rendered interactive All and Completed filter tabs",
    )?;
    let completed_tab = window
        .debug_bounds("SIM-QUEUE-OVERLAY-TAB-COMPLETED")
        .ok_or_else(|| "completed queue overlay tab has no bounds".to_owned())?;
    window.simulate_click(completed_tab.center(), Modifiers::default());
    window.run_until_parked();
    let completed_attempt_ids = item.read_with(window, |item, cx| {
        item.queue_overlay_attempt_ids_for_test(cx)
    });
    if item.read_with(window, |item, _| item.queue_overlay_tab) != QueueOverlayTab::Completed
        || window.debug_bounds("SIM-QUEUE-OVERLAY-ATTEMPT").is_none()
        || completed_attempt_ids.as_slice() != [attempt_id]
        || initially_queued_attempt_ids
            .iter()
            .chain(overflow_attempt_ids.iter())
            .filter(|queued_attempt_id| **queued_attempt_id != attempt_id)
            .any(|queued_attempt_id| completed_attempt_ids.contains(queued_attempt_id))
    {
        return Err(
            "Completed filtering did not render only terminal queue-overlay attempts".to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-089",
        "clicking Completed selected the Completed tab, retained the failed terminal attempt, and removed every queued non-terminal attempt row",
    )?;
    let failed_tab = window
        .debug_bounds("SIM-QUEUE-OVERLAY-TAB-FAILED")
        .ok_or_else(|| "failed queue overlay tab has no bounds".to_owned())?;
    window.simulate_click(failed_tab.center(), Modifiers::default());
    window.run_until_parked();
    if item.read_with(window, |item, _| item.queue_overlay_tab) != QueueOverlayTab::Failed {
        return Err("clicking the Failed filter did not select the Failed overlay tab".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-088",
        "a failed native job caused the Failed filter to render and clicking it selected Failed",
    )?;
    let bottom_attempt_id = overflow_attempt_ids
        .last()
        .copied()
        .ok_or_else(|| "queue-overlay overflow fixture created no bottom attempt".to_owned())?;
    model.update(window, |model, cx| {
        model.ingest_event_batch(
            vec![
                event(
                    profile_id,
                    prompt(0x90f),
                    bottom_attempt_id,
                    0,
                    None,
                    AttemptEventKind::Started,
                ),
                event(
                    profile_id,
                    prompt(0x90f),
                    bottom_attempt_id,
                    1,
                    None,
                    AttemptEventKind::Succeeded,
                ),
            ],
            cx,
        )
    });
    window.run_until_parked();
    let completed_tab = window
        .debug_bounds("SIM-QUEUE-OVERLAY-TAB-COMPLETED")
        .ok_or_else(|| "Completed tab disappeared before bottom-row interaction".to_owned())?;
    window.simulate_click(completed_tab.center(), Modifiers::default());
    window.run_until_parked();
    let completed_ids = item.read_with(window, |item, cx| {
        item.queue_overlay_attempt_ids_for_test(cx)
    });
    if completed_ids.as_slice() != [attempt_id, bottom_attempt_id] {
        return Err(format!(
            "bottom-row fixture did not sort after the original completed attempt: {completed_ids:?}"
        ));
    }
    let overlay_bounds = window
        .debug_bounds("SIM-QUEUE-OVERLAY")
        .ok_or_else(|| "queue overlay disappeared before bottom-row scrolling".to_owned())?;
    let viewport_size = item.update_in(window, |_, window, _| window.viewport_size());
    let mut trigger_bounds = window
        .debug_bounds("SIM-QUEUE-OVERLAY-DETAILS-TRIGGER")
        .ok_or_else(|| "queue overlay details trigger has no bounds".to_owned())?;
    for _ in 0..16 {
        if trigger_bounds.origin.y >= px(0.0) && trigger_bounds.bottom() <= viewport_size.height {
            break;
        }
        let vertical_delta = if trigger_bounds.bottom() > viewport_size.height {
            px(-128.0)
        } else {
            px(128.0)
        };
        window.simulate_event(ScrollWheelEvent {
            position: overlay_bounds.center(),
            delta: ScrollDelta::Pixels(point(px(0.0), vertical_delta)),
            modifiers: Modifiers::default(),
            touch_phase: TouchPhase::Moved,
        });
        window.run_until_parked();
        trigger_bounds = window
            .debug_bounds("SIM-QUEUE-OVERLAY-DETAILS-TRIGGER")
            .ok_or_else(|| "bottom details trigger disappeared while scrolling".to_owned())?;
    }
    if trigger_bounds.origin.y < px(0.0) || trigger_bounds.bottom() > viewport_size.height {
        return Err(format!(
            "bottom details trigger remained outside the viewport: {trigger_bounds:?} versus {viewport_size:?}"
        ));
    }
    window.simulate_click(trigger_bounds.center(), Modifiers::default());
    window.run_until_parked();
    if item.read_with(window, |item, _| item.queue_details_attempt) != Some(bottom_attempt_id) {
        return Err("clicking the bottom row did not open that exact job's details".to_owned());
    }
    let details_bounds = window
        .debug_bounds("SIM-QUEUE-OVERLAY-DETAILS-CONTENT")
        .ok_or_else(|| "bottom-row details content did not render".to_owned())?;
    if details_bounds.origin.x < px(0.0)
        || details_bounds.origin.y < px(0.0)
        || details_bounds.right() > viewport_size.width
        || details_bounds.bottom() > viewport_size.height
    {
        return Err(format!(
            "bottom-row details content escaped the viewport: {details_bounds:?} versus {viewport_size:?}"
        ));
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-091",
        "after scrolling a 19-row overlay, clicking the final completed row opened that exact AttemptId and its measured details bounds remained inside the GPUI viewport",
    )?;
    item.update(window, |item, cx| {
        item.copy_queue_attempt_id(bottom_attempt_id, cx);
    });
    let copied_attempt_id = window
        .read_from_clipboard()
        .and_then(|item| item.text())
        .ok_or_else(|| "copy job ID interaction wrote no clipboard text".to_owned())?;
    if copied_attempt_id != bottom_attempt_id.0.to_string() {
        return Err("copy job ID interaction wrote the wrong attempt identity".to_owned());
    }
    let close_overlay_outcome = item.update(window, |item, cx| {
        item.dispatch_shell_command("Sim.Queue.ToggleOverlay", cx)
    });
    window.run_until_parked();
    if !close_overlay_outcome.is_executed()
        || item.read_with(window, |item, _| item.queue_overlay_visible)
        || window.debug_bounds("SIM-QUEUE-OVERLAY").is_some()
    {
        return Err(
            "dispatching Queue.ToggleOverlay a second time did not close the overlay".to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-090",
        "dispatching Sim.Queue.ToggleOverlay a second time closed the overlay without changing the QPOV2 surface setting",
    )?;
    item.update(window, |item, cx| {
        item.dispatch_shell_command("Sim.Queue.ToggleOverlay", cx)
    });
    window.run_until_parked();
    let docked_history_bounds = window
        .debug_bounds("SIM-QUEUE-OVERLAY-DOCKED-HISTORY")
        .ok_or_else(|| "docked History action did not render with bounds".to_owned())?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-064",
        "the expanded queue overlay rendered its focusable docked History action with real GPUI bounds and visible text",
    )?;
    window.simulate_click(docked_history_bounds.center(), Modifiers::default());
    window.run_until_parked();
    let docked_history_click_closed_overlay = !item
        .read_with(window, |item, _| item.queue_overlay_visible)
        && window.debug_bounds("SIM-QUEUE-OVERLAY").is_none();
    if !docked_history_click_closed_overlay {
        return Err("clicking docked History did not close the graph overlay".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-066",
        "pointer-clicking the docked History action closed the queue overlay popover",
    )?;
    let before_navigation = item.read_with(window, |item, _| {
        item.model()
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| (graph.selection.clone(), graph.viewport.clone()))
    });
    item.update(window, |item, cx| {
        item.copy_execution_error(cx);
        item.locate_active_execution_error(cx);
    });
    let clipboard = window
        .read_from_clipboard()
        .and_then(|item| item.text())
        .ok_or_else(|| "structured error copy wrote no clipboard text".to_owned())?;
    if !clipboard.contains("fixture_failure") || !clipboard.contains("error-node") {
        return Err("structured error copy omitted code or node identity".to_owned());
    }
    if !item.read_with(window, |item, _| item.can_restore_execution_navigation()) {
        return Err("error navigation did not preserve prior graph state".to_owned());
    }
    let restored = item.update(window, |item, cx| {
        item.restore_execution_navigation(cx)
            .map_err(|error| error.to_string())
    })?;
    if !restored {
        return Err("error navigation had no state to restore".to_owned());
    }
    let after_restore = item.read_with(window, |item, _| {
        item.model()
            .document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| (graph.selection.clone(), graph.viewport.clone()))
    });
    if before_navigation != after_restore {
        return Err("error navigation did not restore selection and viewport".to_owned());
    }

    let pre_management_snapshot = model
        .read_with(window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let pre_management_attempt_count = pre_management_snapshot.attempts.len();
    let pre_management_nonterminal_count = pre_management_snapshot
        .attempts
        .iter()
        .filter(|attempt| !attempt.state.is_terminal())
        .count();
    model.update(window, |model, cx| {
        model.register_plan_provider(
            Arc::new(DeterministicPlanProvider {
                plan: plan(prompt(0xb00)),
            }),
            cx,
        );
    });
    let front_outcome = item.update(window, |item, cx| {
        let outcome = item.dispatch_shell_command("Sim.QueuePromptFront", cx);
        (
            "Sim.QueuePromptFront".to_owned(),
            outcome.is_executed(),
            format!("{outcome:?}"),
        )
    });
    if !front_outcome.1 {
        return Err(format!(
            "registered-controller front queue command did not execute: {front_outcome:?}"
        ));
    }
    let interrupt_target = item
        .read_with(window, |item, cx| item.active_execution_presentation(cx))
        .ok_or_else(|| "front queue did not associate its assigned attempt".to_owned())?;
    model.update(window, |model, cx| {
        model.ingest_event_batch(
            vec![event(
                profile_id,
                interrupt_target.prompt_id,
                interrupt_target.attempt_id,
                0,
                None,
                AttemptEventKind::Started,
            )],
            cx,
        )
    });
    let remaining_execution_commands = item.update(window, |item, cx| {
        item.apply_graph_command(GraphCommand::SelectAll, cx);
        [
            (
                "Sim.Interrupt",
                item.dispatch_shell_command("Sim.Interrupt", cx),
            ),
            (
                "Sim.ClearPendingTasks",
                item.dispatch_shell_command("Sim.ClearPendingTasks", cx),
            ),
            (
                "Sim.ToggleQPOV2",
                item.dispatch_shell_command("Sim.ToggleQPOV2", cx),
            ),
        ]
        .map(|(command_id, outcome)| {
            (
                command_id.to_owned(),
                outcome.is_executed(),
                format!("{outcome:?}"),
            )
        })
    });
    if !front_outcome.1
        || remaining_execution_commands
            .iter()
            .any(|(_, executed, _)| !executed)
    {
        return Err(format!(
            "one or more registered-controller execution commands did not execute: front={front_outcome:?}, remaining={remaining_execution_commands:?}"
        ));
    }
    command_outcomes.push(front_outcome);
    command_outcomes.extend(remaining_execution_commands);
    let actual_command_ids = command_outcomes
        .iter()
        .map(|(command_id, _, _)| command_id.as_str())
        .collect::<BTreeSet<_>>();
    let expected_command_ids = task_18_commands().into_iter().collect::<BTreeSet<_>>();
    if actual_command_ids != expected_command_ids {
        return Err(format!(
            "behaviorally exercised command IDs differ from Task 18 registrations: actual={actual_command_ids:?}, expected={expected_command_ids:?}"
        ));
    }
    let post_command_snapshot = model
        .read_with(window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    if !post_command_snapshot.queue.is_empty()
        || post_command_snapshot.attempts.len() != pre_management_attempt_count.saturating_add(1)
        || post_command_snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.state == AttemptState::Cancelled)
            .count()
            != pre_management_nonterminal_count
        || post_command_snapshot
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == interrupt_target.attempt_id)
            .is_none_or(|attempt| attempt.state != AttemptState::Cancelling)
    {
        let cancelled_count = post_command_snapshot
            .attempts
            .iter()
            .filter(|attempt| attempt.state == AttemptState::Cancelled)
            .count();
        let states = post_command_snapshot
            .attempts
            .iter()
            .map(|attempt| (attempt.attempt_id, attempt.state))
            .collect::<Vec<_>>();
        return Err(format!(
            "interrupt or clear-pending acknowledgement did not mutate queue state: queue={}, attempts={} expected_attempts={}, cancelled={cancelled_count} expected_cancelled={}, states={states:?}",
            post_command_snapshot.queue.len(),
            post_command_snapshot.attempts.len(),
            pre_management_attempt_count.saturating_add(1),
            pre_management_nonterminal_count,
        ));
    }
    window.run_until_parked();
    if window.debug_bounds("SIM-EXECUTION-ACTIONBAR").is_some()
        || window.debug_bounds("SIM-QUEUE-OVERLAY").is_some()
    {
        return Err("QPOV2 toggle did not materially hide graph execution surfaces".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-039",
        "dispatching Sim.ToggleQPOV2 materially hid the graph execution action bar and overlay",
    )?;
    item.update(window, |graph, cx| {
        graph.associate_execution_attempt(attempt_id, cx)
    });
    let entered_subgraph_before_workflow_switch = item.update(window, |graph, cx| {
        graph.apply_graph_command(
            GraphCommand::OpenSubgraph {
                definition_identifier: GraphIdentifier::from("validation-subgraph"),
            },
            cx,
        )
    });
    if !entered_subgraph_before_workflow_switch
        || item
            .read_with(window, |graph, cx| graph.active_execution_presentation(cx))
            .is_none_or(|attempt| {
                attempt.progress.is_some()
                    || !attempt.node_progress.is_empty()
                    || !attempt.previews.is_empty()
            })
    {
        return Err(
            "the owning workflow retained stale scoped progress before workflow switch".to_owned(),
        );
    }

    let (unassociated_graph, unassociated_window) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            graph_fixture(profile_id).expect("valid unassociated graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    unassociated_window.run_until_parked();
    if unassociated_graph
        .read_with(unassociated_window, |graph, cx| {
            graph.active_execution_presentation(cx)
        })
        .is_some()
        || unassociated_window
            .debug_bounds("SIM-EXECUTION-ERROR-OVERLAY")
            .is_some()
    {
        return Err(
            "unassociated workflow inherited another workflow's execution attempt".to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-112",
        "switching to a second workflow while the owning workflow was inside a subgraph left both workflows free of the old scoped node progress and previews",
    )?;
    let snapshot_before_unassociated_interrupt = model
        .read_with(unassociated_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let unassociated_interrupt = unassociated_graph.update(unassociated_window, |graph, cx| {
        graph.dispatch_shell_command("Sim.Interrupt", cx)
    });
    if !matches!(
        unassociated_interrupt,
        CommandDispatchOutcome::Rejected { .. }
    ) {
        return Err(format!(
            "unassociated workflow was allowed to interrupt another attempt: {unassociated_interrupt:?}"
        ));
    }
    let snapshot_after_unassociated_interrupt = model
        .read_with(unassociated_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    if snapshot_after_unassociated_interrupt.queue != snapshot_before_unassociated_interrupt.queue
        || snapshot_after_unassociated_interrupt.attempts
            != snapshot_before_unassociated_interrupt.attempts
        || snapshot_after_unassociated_interrupt.pending_commands
            != snapshot_before_unassociated_interrupt.pending_commands
    {
        return Err("unassociated interrupt mutated another workflow's attempt".to_owned());
    }
    let mut stale_fixture = graph_fixture(profile_id).map_err(|error| error.to_string())?;
    stale_fixture.execution_association = Some(identifier(0x7fff).to_string());
    let (stale_graph, stale_window) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(stale_fixture, WeakEntity::new_invalid(), cx)
    });
    stale_window.run_until_parked();
    if stale_graph
        .read_with(stale_window, |graph, cx| {
            graph.active_execution_presentation(cx)
        })
        .is_some()
        || stale_window
            .debug_bounds("SIM-EXECUTION-ERROR-OVERLAY")
            .is_some()
    {
        return Err("stale workflow association inherited the latest profile attempt".to_owned());
    }

    let retry_acknowledgement = model
        .update(stale_window, |model, cx| {
            model.retry(profile_id, attempt_id, cx)
        })
        .map_err(|error| error.to_string())?;
    let retry_attempt_id = match retry_acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(retry_attempt_id),
        } => retry_attempt_id,
        outcome => return Err(format!("D23 retry was not accepted: {outcome:?}")),
    };
    let original_owned = item.read_with(stale_window, |graph, _| {
        graph.is_associated_with_execution(attempt_id)
    });
    let unrelated_owned = unassociated_graph.read_with(stale_window, |graph, _| {
        graph.is_associated_with_execution(attempt_id)
    });
    let stale_owned = stale_graph.read_with(stale_window, |graph, _| {
        graph.is_associated_with_execution(attempt_id)
    });
    if original_owned {
        item.update(stale_window, |graph, cx| {
            graph.associate_execution_attempt(retry_attempt_id, cx)
        });
    }
    if unrelated_owned {
        unassociated_graph.update(stale_window, |graph, cx| {
            graph.associate_execution_attempt(retry_attempt_id, cx)
        });
    }
    if stale_owned {
        stale_graph.update(stale_window, |graph, cx| {
            graph.associate_execution_attempt(retry_attempt_id, cx)
        });
    }
    if !item.read_with(stale_window, |graph, _| {
        graph.is_associated_with_execution(retry_attempt_id)
    }) || unassociated_graph.read_with(stale_window, |graph, _| {
        graph.is_associated_with_execution(retry_attempt_id)
    }) || stale_graph.read_with(stale_window, |graph, _| {
        graph.is_associated_with_execution(retry_attempt_id)
    }) {
        return Err("retry association was not isolated to the owning graph".to_owned());
    }

    let isolated_overlay_actions = Arc::new(Mutex::new(Vec::new()));
    let (isolated_overlay, isolated_overlay_window) = cx.add_window_view({
        let isolated_overlay_actions = isolated_overlay_actions.clone();
        move |_, cx| ErrorOverlayProbe::new(attempt_id, isolated_overlay_actions, cx)
    });
    isolated_overlay_window.run_until_parked();
    let isolated_view_bounds = isolated_overlay_window
        .debug_bounds("SIM-SURFACE-ERROR-OVERLAY-VIEW")
        .ok_or_else(|| "isolated error overlay View Errors has no bounds".to_owned())?;
    isolated_overlay_window.simulate_click(isolated_view_bounds.center(), Modifiers::default());
    isolated_overlay_window.run_until_parked();
    if recorded_overlay_probe_actions(&isolated_overlay_actions)
        != [ExecutionSurfaceAction::ViewErrors]
    {
        return Err(format!(
            "isolated error overlay pointer View Errors emitted the wrong actions: {:?}",
            recorded_overlay_probe_actions(&isolated_overlay_actions)
        ));
    }
    let isolated_dismiss_bounds = isolated_overlay_window
        .debug_bounds("SIM-SURFACE-ERROR-OVERLAY-DISMISS")
        .ok_or_else(|| "isolated error overlay Dismiss has no bounds".to_owned())?;
    isolated_overlay_window.simulate_click(isolated_dismiss_bounds.center(), Modifiers::default());
    isolated_overlay_window.run_until_parked();
    if recorded_overlay_probe_actions(&isolated_overlay_actions)
        != [
            ExecutionSurfaceAction::ViewErrors,
            ExecutionSurfaceAction::DismissErrorOverlay(attempt_id),
        ]
    {
        return Err(format!(
            "isolated error overlay pointer Dismiss emitted the wrong actions: {:?}",
            recorded_overlay_probe_actions(&isolated_overlay_actions)
        ));
    }
    isolated_overlay.update_in(isolated_overlay_window, |overlay, window, cx| {
        overlay.view_focus_handle.focus(window, cx)
    });
    isolated_overlay_window.run_until_parked();
    isolated_overlay_window.simulate_keystrokes("enter");
    if recorded_overlay_probe_actions(&isolated_overlay_actions)
        != [
            ExecutionSurfaceAction::ViewErrors,
            ExecutionSurfaceAction::DismissErrorOverlay(attempt_id),
            ExecutionSurfaceAction::ViewErrors,
        ]
    {
        return Err(format!(
            "isolated error overlay focused Enter View Errors emitted the wrong actions: {:?}",
            recorded_overlay_probe_actions(&isolated_overlay_actions)
        ));
    }
    isolated_overlay.update_in(isolated_overlay_window, |overlay, window, cx| {
        overlay.dismiss_focus_handle.focus(window, cx)
    });
    isolated_overlay_window.run_until_parked();
    isolated_overlay_window.simulate_keystrokes("enter");
    if recorded_overlay_probe_actions(&isolated_overlay_actions)
        != [
            ExecutionSurfaceAction::ViewErrors,
            ExecutionSurfaceAction::DismissErrorOverlay(attempt_id),
            ExecutionSurfaceAction::ViewErrors,
            ExecutionSurfaceAction::DismissErrorOverlay(attempt_id),
        ]
    {
        return Err(format!(
            "isolated error overlay focused Enter Dismiss emitted the wrong actions: {:?}",
            recorded_overlay_probe_actions(&isolated_overlay_actions)
        ));
    }

    let (panel, panel_window) =
        cx.add_window_view(|_, cx| ExecutionPanel::test_new(model.clone(), cx));
    if !docked_history_click_closed_overlay {
        return Err("docked History pointer interaction evidence was lost".to_owned());
    }
    panel_window.update(|window, cx| {
        crate::execution_panel::open_docked_execution_history_for_test(
            item.clone(),
            panel.clone(),
            window,
            cx,
        )
    });
    panel_window.run_until_parked();
    let first_docked_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    let first_docked_focused =
        panel_window.update(|window, cx| panel.read(cx).is_focused_for_test(window));
    if !first_docked_state.active
        || !first_docked_state.docked_history
        || first_docked_state.selected_tab != ExecutionPanelTab::History
        || !first_docked_focused
    {
        return Err(
            "the production docked-History core did not open, select, and focus the History dock"
                .to_owned(),
        );
    }
    panel_window.update(|window, cx| {
        crate::execution_panel::open_docked_execution_history_for_test(
            item.clone(),
            panel.clone(),
            window,
            cx,
        )
    });
    panel_window.run_until_parked();
    let repeated_docked_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    let repeated_docked_focused =
        panel_window.update(|window, cx| panel.read(cx).is_focused_for_test(window));
    if !repeated_docked_state.active
        || !repeated_docked_state.docked_history
        || repeated_docked_state.selected_tab != ExecutionPanelTab::History
        || !repeated_docked_focused
    {
        return Err("reopening docked History was not idempotent and focused".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-108",
        "the queue-overlay History interaction closed the graph overlay and the shared production action core opened, selected, and focused docked History idempotently",
    )?;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::OpenErrorHelp(attempt_id),
            window,
            cx,
        );
    });
    let allowed_navigation = panel_window.opened_url();
    if allowed_navigation.as_deref() != Some("https://docs.comfy.org/troubleshooting/overview") {
        return Err(format!(
            "canonical external-navigation policy did not allow the trusted help URL: {allowed_navigation:?}"
        ));
    }
    let denying_policy = ExternalNavigationPolicy::new(["mailto".to_owned()], true)
        .map_err(|error| error.to_string())?;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.set_external_navigation_policy_for_test(denying_policy);
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::OpenErrorGitHub(attempt_id),
            window,
            cx,
        );
    });
    if panel_window.opened_url() != allowed_navigation
        || panel
            .read_with(panel_window, |panel, _| {
                panel.status_message_for_test().map(str::to_owned)
            })
            .is_none_or(|message| !message.contains("Blocked unsafe execution navigation"))
    {
        return Err(
            "denied execution navigation reached the platform or lacked visible feedback"
                .to_owned(),
        );
    }
    let current_persisted_state = panel.update_in(panel_window, |panel, window, cx| {
        let focus_handle = panel.focus_handle(cx);
        window.focus(&focus_handle, cx);
        if !focus_handle.is_focused(window) {
            return Err("execution panel did not receive focus".to_owned());
        }
        panel.set_test_state(
            ExecutionPanelTab::Errors,
            Some(attempt_id),
            "active",
            "failed",
            0,
            cx,
        );
        let round_trip = panel
            .persisted_state_round_trip_for_test()
            .map_err(|error| error.to_string())?;
        if !round_trip.contains("\"selected_tab\":\"errors\"")
            || !round_trip.contains("\"queue_filter\":\"active\"")
            || !round_trip.contains("\"history_filter\":\"failed\"")
        {
            return Err("execution panel state did not round-trip".to_owned());
        }
        panel.handle_output_action_for_test(
            OutputViewAction::RecoverOutput(unavailable_output_id),
            window,
            cx,
        );
        panel.handle_output_action_for_test(
            OutputViewAction::RemoveOutput(unavailable_output_id),
            window,
            cx,
        );
        Ok(round_trip)
    })?;
    if !panel_window.has_pending_prompt() {
        return Err("output removal did not open a GPUI confirmation prompt".to_owned());
    }
    panel_window.simulate_prompt_answer("Cancel");
    panel_window.run_until_parked();
    let availability_after_cancel = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == attempt_id)
        .and_then(|attempt| {
            attempt
                .outputs
                .iter()
                .find(|output| output.output_id == unavailable_output_id)
        })
        .map(|output| output.availability.clone());
    if !matches!(
        availability_after_cancel,
        Some(ExecutionOutputAvailability::Ready { .. })
    ) || operation_handler
        .actions
        .lock()
        .map_err(|_| "panel operation recorder poisoned".to_owned())?
        .len()
        != 1
    {
        return Err(
            "cancelling output removal changed canonical state or invoked the handler".to_owned(),
        );
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_output_action_for_test(
            OutputViewAction::RemoveOutput(unavailable_output_id),
            window,
            cx,
        );
    });
    if !panel_window.has_pending_prompt() {
        return Err("confirmed output removal did not reopen its GPUI prompt".to_owned());
    }
    panel_window.simulate_prompt_answer("Remove Output");
    panel_window.run_until_parked();
    let availability_after_remove = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == attempt_id)
        .and_then(|attempt| {
            attempt
                .outputs
                .iter()
                .find(|output| output.output_id == unavailable_output_id)
        })
        .map(|output| output.availability.clone());
    if !matches!(
        availability_after_remove,
        Some(ExecutionOutputAvailability::Removed { .. })
    ) {
        return Err(format!(
            "confirmed output removal did not persist a removed projection: {availability_after_remove:?}"
        ));
    }

    let recover_persisted_fixture =
        |panel_window: &mut gpui::VisualTestContext,
         suffix: &str,
         serialized: String|
         -> Result<(u16, ExecutionPanelTab, Option<String>, Option<String>), String> {
            let serialization_key = format!("val-gpui-005-execution-panel-{suffix}");
            let key_value_store = panel_window.update(|_, cx| KeyValueStore::global(cx));
            panel_window
                .foreground_executor
                .block_test(key_value_store.write_kvp(serialization_key.clone(), serialized))
                .map_err(|error| error.to_string())?;
            let recovery_task = panel_window.update(|_, cx| {
                ExecutionPanel::recover_persisted_state_for_test(serialization_key, cx)
            });
            let recovery = panel_window.foreground_executor.block_test(recovery_task);
            Ok((
                recovery.schema_version,
                recovery.selected_tab,
                recovery.persistence_error,
                recovery.serialized_after_recovery,
            ))
        };
    let current_recovery =
        recover_persisted_fixture(panel_window, "current", current_persisted_state.clone())?;
    if current_recovery.0 != 2
        || current_recovery.1 != ExecutionPanelTab::Errors
        || current_recovery.2.is_some()
        || current_recovery
            .3
            .as_deref()
            .is_none_or(|serialized| !serialized.contains("\"schema_version\":2"))
    {
        return Err(format!(
            "current-schema KVP state did not recover exactly: {current_recovery:?}"
        ));
    }
    let schema_one_recovery = recover_persisted_fixture(
        panel_window,
        "schema-one",
        current_persisted_state.replacen("\"schema_version\":2", "\"schema_version\":1", 1),
    )?;
    if schema_one_recovery.0 != 2
        || schema_one_recovery.1 != ExecutionPanelTab::Errors
        || schema_one_recovery
            .3
            .as_deref()
            .is_none_or(|serialized| !serialized.contains("\"schema_version\":2"))
    {
        return Err(format!(
            "schema-one KVP state did not migrate and rewrite safely: {schema_one_recovery:?}"
        ));
    }
    for (suffix, serialized) in [
        ("malformed", "{".to_owned()),
        (
            "future",
            current_persisted_state.replacen("\"schema_version\":2", "\"schema_version\":999", 1),
        ),
    ] {
        let recovery = recover_persisted_fixture(panel_window, suffix, serialized)?;
        if recovery.0 != 2
            || recovery.1 != ExecutionPanelTab::Queue
            || recovery.2.as_deref().is_none_or(str::is_empty)
            || recovery
                .3
                .as_deref()
                .is_none_or(|serialized| !serialized.contains("\"schema_version\":2"))
        {
            return Err(format!(
                "{suffix} KVP state did not recover with diagnostics and current-schema defaults: {recovery:?}"
            ));
        }
    }
    let live_panel_after_recovery =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if live_panel_after_recovery.schema_version != 2
        || live_panel_after_recovery.persistence_error.is_some()
        || !model.read_with(panel_window, |model, _| {
            model.runtime_controller_available() && model.plan_provider_available()
        })
        || panel_window
            .debug_bounds("SIM-SURFACE-TAB-ERRORS")
            .is_none()
    {
        return Err(
            "KVP corruption recovery disturbed live panel defaults or capability registration"
                .to_owned(),
        );
    }

    panel.update(panel_window, |panel, cx| {
        panel.set_test_state(
            ExecutionPanelTab::Queue,
            Some(attempt_id),
            "all",
            "all",
            0,
            cx,
        )
    });
    panel_window.run_until_parked();
    let history_tab_bounds = panel_window
        .debug_bounds("SIM-EXECUTION-TAB-HISTORY")
        .ok_or_else(|| "top-level History tab has no pointer bounds".to_owned())?;
    panel_window.simulate_click(history_tab_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    if panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().selected_tab
    }) != ExecutionPanelTab::History
    {
        return Err(format!(
            "standalone ExecutionPanel did not route top-level History tab pointer input at {history_tab_bounds:?}"
        ));
    }
    let queue_tab_bounds = panel_window
        .debug_bounds("SIM-EXECUTION-TAB-QUEUE")
        .ok_or_else(|| "top-level Queue tab has no pointer bounds".to_owned())?;
    panel_window.simulate_click(queue_tab_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    if panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().selected_tab
    }) != ExecutionPanelTab::Queue
    {
        return Err(format!(
            "standalone ExecutionPanel did not route top-level Queue tab pointer input at {queue_tab_bounds:?}"
        ));
    }
    let initial_overlay_attempt_id = panel
        .read_with(panel_window, |panel, cx| {
            panel.current_error_overlay_attempt_id_for_test(cx)
        })
        .ok_or_else(|| "the initial failed attempt produced no durable panel overlay".to_owned())?;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.focus_error_overlay_view_for_test(window, cx)
    });
    panel_window.run_until_parked();
    let overlay_view_is_focused = panel_window.update(|window, cx| {
        panel
            .read(cx)
            .error_overlay_view_is_focused_for_test(window)
    });
    if !overlay_view_is_focused {
        return Err("overlay View Errors could not receive keyboard focus".to_owned());
    }
    let overlay_trace_before_keyboard_view = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    panel_window.simulate_keystrokes("enter");
    let overlay_trace_after_keyboard_view = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    let overlay_keyboard_view_activated = panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().selected_tab
    }) == ExecutionPanelTab::Errors
        && overlay_trace_after_keyboard_view.len()
            == overlay_trace_before_keyboard_view.len().saturating_add(1)
        && matches!(
            overlay_trace_after_keyboard_view.last(),
            Some(ExecutionSurfaceAction::ViewErrors)
        );
    panel.update(panel_window, |panel, cx| {
        panel.set_test_state(
            ExecutionPanelTab::Queue,
            Some(attempt_id),
            "all",
            "all",
            0,
            cx,
        )
    });
    panel_window.run_until_parked();
    let view_overlay_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-OVERLAY-VIEW")
        .ok_or_else(|| "durable error overlay View Errors disappeared after Enter".to_owned())?;
    let overlay_parent_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-OVERLAY")
        .ok_or_else(|| "durable error overlay parent disappeared after Enter".to_owned())?;
    let overlay_trace_before = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    let view_pointer_points = [
        view_overlay_bounds.center(),
        point(
            view_overlay_bounds.origin.x + px(2.0),
            view_overlay_bounds.origin.y + px(2.0),
        ),
        point(
            view_overlay_bounds.origin.x + view_overlay_bounds.size.width - px(2.0),
            view_overlay_bounds.origin.y + px(2.0),
        ),
        point(
            view_overlay_bounds.origin.x + px(2.0),
            view_overlay_bounds.origin.y + view_overlay_bounds.size.height - px(2.0),
        ),
        point(
            view_overlay_bounds.origin.x + view_overlay_bounds.size.width - px(2.0),
            view_overlay_bounds.origin.y + view_overlay_bounds.size.height - px(2.0),
        ),
    ];
    for pointer_position in view_pointer_points {
        panel_window.simulate_mouse_move(
            pointer_position,
            None::<MouseButton>,
            Modifiers::default(),
        );
        panel_window.simulate_click(pointer_position, Modifiers::default());
        panel_window.run_until_parked();
        if panel
            .read_with(panel_window, |panel, _| {
                panel.surface_action_trace_for_test()
            })
            .len()
            > overlay_trace_before.len()
        {
            break;
        }
    }
    let overlay_trace_after_view = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    if panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().selected_tab
    }) != ExecutionPanelTab::Errors
    {
        let viewport = panel.update_in(panel_window, |_, window, _| window.viewport_size());
        return Err(format!(
            "adjacent overlay View Errors button did not receive pointer input: keyboard_activated={overlay_keyboard_view_activated}, keyboard_trace_before={overlay_trace_before_keyboard_view:?}, keyboard_trace_after={overlay_trace_after_keyboard_view:?}, parent_bounds={overlay_parent_bounds:?}, child_bounds={view_overlay_bounds:?}, points={view_pointer_points:?}, viewport={viewport:?}, trace_before={overlay_trace_before:?}, trace_after={overlay_trace_after_view:?}"
        ));
    }
    if overlay_trace_after_view.len() != overlay_trace_before.len().saturating_add(1)
        || !matches!(
            overlay_trace_after_view.last(),
            Some(ExecutionSurfaceAction::ViewErrors)
        )
    {
        return Err("overlay View Errors changed state without one traced action".to_owned());
    }
    if !overlay_keyboard_view_activated {
        return Err(format!(
            "focused Enter did not activate overlay View Errors exactly once: before={overlay_trace_before_keyboard_view:?}, after={overlay_trace_after_keyboard_view:?}"
        ));
    }
    panel.update(panel_window, |panel, cx| {
        panel.set_test_state(
            ExecutionPanelTab::Queue,
            Some(attempt_id),
            "all",
            "all",
            0,
            cx,
        )
    });
    panel_window.run_until_parked();
    let dismiss_overlay_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-OVERLAY-DISMISS")
        .ok_or_else(|| "durable error overlay dismiss action has no bounds".to_owned())?;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.focus_error_overlay_dismiss_for_test(window, cx)
    });
    panel_window.run_until_parked();
    let overlay_dismiss_is_focused = panel_window.update(|window, cx| {
        panel
            .read(cx)
            .error_overlay_dismiss_is_focused_for_test(window)
    });
    if !overlay_dismiss_is_focused {
        return Err("overlay Dismiss could not receive keyboard focus".to_owned());
    }
    panel_window.simulate_mouse_move(
        dismiss_overlay_bounds.center(),
        None::<MouseButton>,
        Modifiers::default(),
    );
    panel_window.run_until_parked();
    let dismiss_overlay_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-OVERLAY-DISMISS")
        .ok_or_else(|| "durable error overlay dismiss action disappeared after hover".to_owned())?;
    let dismiss_overlay_viewport =
        panel.update_in(panel_window, |_, window, _| window.viewport_size());
    panel_window.simulate_mouse_down(
        dismiss_overlay_bounds.center(),
        MouseButton::Left,
        Modifiers::default(),
    );
    panel_window.simulate_mouse_up(
        dismiss_overlay_bounds.center(),
        MouseButton::Left,
        Modifiers::default(),
    );
    panel_window.run_until_parked();
    let dismissed_overlay_state = panel.read_with(panel_window, |panel, cx| {
        (
            panel.surface_state_for_test(),
            panel.current_error_overlay_attempt_id_for_test(cx),
        )
    });
    let overlay_trace_after_dismiss = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    if overlay_trace_after_dismiss.len() != overlay_trace_after_view.len().saturating_add(1)
        || !matches!(
            overlay_trace_after_dismiss.last(),
            Some(ExecutionSurfaceAction::DismissErrorOverlay(attempt_id))
                if *attempt_id == initial_overlay_attempt_id
        )
    {
        return Err(format!(
            "overlay Dismiss pointer path emitted the wrong action trace: before={overlay_trace_after_view:?}, after={overlay_trace_after_dismiss:?}"
        ));
    }
    let initial_failure_remains_canonical = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .any(|attempt| {
            attempt.attempt_id == initial_overlay_attempt_id && attempt.failure.is_some()
        });
    if dismissed_overlay_state.1 == Some(initial_overlay_attempt_id)
        || !dismissed_overlay_state
            .0
            .dismissed_error_overlay_attempts
            .contains(&initial_overlay_attempt_id)
        || !initial_failure_remains_canonical
    {
        return Err(format!(
            "dismissing the old failure did not hide only that AttemptId while retaining canonical history: dismissed={:?}, current={:?}, canonical={initial_failure_remains_canonical}, bounds={dismiss_overlay_bounds:?}, viewport={dismiss_overlay_viewport:?}",
            dismissed_overlay_state.0.dismissed_error_overlay_attempts, dismissed_overlay_state.1,
        ));
    }
    let revision_before_unrelated_status = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .revision;
    model
        .update(panel_window, |model, cx| {
            model.set_snapshot_status(
                profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    panel_window.run_until_parked();
    let revision_after_unrelated_status = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .revision;
    if revision_after_unrelated_status <= revision_before_unrelated_status
        || panel.read_with(panel_window, |panel, cx| {
            panel.current_error_overlay_attempt_id_for_test(cx)
        }) == Some(initial_overlay_attempt_id)
    {
        return Err(
            "an unrelated status/revision update resurrected a dismissed error overlay".to_owned(),
        );
    }
    let distinct_failure_prompt_id = prompt(0xc00);
    let distinct_failure_acknowledgement = model
        .update(panel_window, |model, cx| {
            model.dispatch(
                profile_id,
                ExecutionControlCommandKind::Queue {
                    plan: plan(distinct_failure_prompt_id),
                    priority: 0,
                    front: false,
                },
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    let distinct_failure_attempt_id = match distinct_failure_acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(attempt_id),
        } => attempt_id,
        outcome => {
            return Err(format!(
                "distinct failure fixture was not queued: {outcome:?}"
            ));
        }
    };
    let distinct_failure =
        ExecutionFailure::new("distinct_failure", "second deterministic node failure")
            .at_node(NodeId("error-node".to_owned()))
            .retryable(true);
    model.update(panel_window, |model, cx| {
        model.ingest_event_batch(
            vec![
                event(
                    profile_id,
                    distinct_failure_prompt_id,
                    distinct_failure_attempt_id,
                    0,
                    None,
                    AttemptEventKind::Started,
                ),
                event(
                    profile_id,
                    distinct_failure_prompt_id,
                    distinct_failure_attempt_id,
                    1,
                    Some(NodeId("error-node".to_owned())),
                    AttemptEventKind::Failed {
                        failure: distinct_failure,
                    },
                ),
            ],
            cx,
        )
    });
    panel_window.run_until_parked();
    let failure_attempt_ids = panel.read_with(panel_window, |panel, cx| {
        panel.filtered_failure_attempt_ids_for_test(cx)
    });
    if panel.read_with(panel_window, |panel, cx| {
        panel.current_error_overlay_attempt_id_for_test(cx)
    }) != Some(distinct_failure_attempt_id)
        || !failure_attempt_ids.contains(&initial_overlay_attempt_id)
        || !failure_attempt_ids.contains(&distinct_failure_attempt_id)
        || panel_window
            .debug_bounds("SIM-SURFACE-ERROR-OVERLAY")
            .is_none()
    {
        return Err(
            "a distinct failed attempt did not render a new overlay while retaining old Errors history"
                .to_owned(),
        );
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(ExecutionSurfaceAction::ViewErrors, window, cx);
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::SetErrorSearch(String::new()),
            window,
            cx,
        );
    });
    panel_window.run_until_parked();
    let collapse_all_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-COLLAPSE-ALL")
        .ok_or_else(|| "Collapse All has no GPUI bounds".to_owned())?;
    let collapse_trace_before = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    let collapse_invocation_before = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_invocation_trace_for_test()
    });
    let collapse_pointer_points = [
        collapse_all_bounds.center(),
        point(
            collapse_all_bounds.origin.x + px(2.0),
            collapse_all_bounds.origin.y + px(2.0),
        ),
        point(
            collapse_all_bounds.origin.x + collapse_all_bounds.size.width - px(2.0),
            collapse_all_bounds.origin.y + px(2.0),
        ),
        point(
            collapse_all_bounds.origin.x + px(2.0),
            collapse_all_bounds.origin.y + collapse_all_bounds.size.height - px(2.0),
        ),
        point(
            collapse_all_bounds.origin.x + collapse_all_bounds.size.width - px(2.0),
            collapse_all_bounds.origin.y + collapse_all_bounds.size.height - px(2.0),
        ),
    ];
    for pointer_position in collapse_pointer_points {
        panel_window.simulate_mouse_move(
            pointer_position,
            None::<MouseButton>,
            Modifiers::default(),
        );
        panel_window.simulate_click(pointer_position, Modifiers::default());
        panel_window.run_until_parked();
        if panel
            .read_with(panel_window, |panel, _| {
                panel.surface_action_invocation_trace_for_test()
            })
            .len()
            > collapse_invocation_before.len()
        {
            break;
        }
    }
    let fully_collapsed_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    let collapse_trace_after = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    let collapse_invocation_trace = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_invocation_trace_for_test()
    });
    let collapse_update_error = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_update_error_for_test()
    });
    if !fully_collapsed_state.errors_all_collapsed
        || failure_attempt_ids.iter().any(|attempt_id| {
            !fully_collapsed_state
                .collapsed_error_attempts
                .contains(attempt_id)
        })
        || panel_window
            .debug_bounds("SIM-SURFACE-ERROR-DETAILS-CONTENT")
            .is_some()
    {
        return Err(format!(
            "Collapse All did not collapse every structured failure: all_collapsed={}, expected_ids={failure_attempt_ids:?}, collapsed_ids={:?}, details_visible={}, bounds={collapse_all_bounds:?}, points={collapse_pointer_points:?}, reducer_trace_before={collapse_trace_before:?}, reducer_trace_after={collapse_trace_after:?}, invocation_trace_before={collapse_invocation_before:?}, invocation_trace_after={collapse_invocation_trace:?}, update_error={collapse_update_error:?}",
            fully_collapsed_state.errors_all_collapsed,
            fully_collapsed_state.collapsed_error_attempts,
            panel_window
                .debug_bounds("SIM-SURFACE-ERROR-DETAILS-CONTENT")
                .is_some(),
        ));
    }
    panel_window.update(|window, _| window.refresh());
    panel_window.run_until_parked();
    let errors_root_bounds = panel_window.debug_bounds("SIM-SURFACE-TAB-ERRORS");
    let error_group_list_bounds = panel_window.debug_bounds("SIM-SURFACE-ERROR-GROUP-LIST");
    let first_error_card_bounds = panel_window.debug_bounds("SIM-SURFACE-ERROR-CARD-SECTION-FIRST");
    let error_node_card_bounds = panel_window.debug_bounds("SIM-SURFACE-ERROR-NODE-CARD");
    let execution_main_region_bounds = panel_window.debug_bounds("SIM-EXECUTION-MAIN-REGION");
    let execution_main_content_bounds = panel_window.debug_bounds("SIM-EXECUTION-MAIN-CONTENT");
    let error_surface_viewport = panel_window.update(|window, _| window.viewport_size());
    let error_details_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-DETAILS-TRIGGER-FIRST")
        .ok_or_else(|| "collapsed error details trigger has no bounds".to_owned())?;
    let expand_trace_before = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    let expand_invocation_before = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_invocation_trace_for_test()
    });
    let expand_pointer_points = [
        error_details_bounds.center(),
        point(
            error_details_bounds.origin.x + px(2.0),
            error_details_bounds.origin.y + px(2.0),
        ),
        point(
            error_details_bounds.origin.x + error_details_bounds.size.width - px(2.0),
            error_details_bounds.origin.y + px(2.0),
        ),
        point(
            error_details_bounds.origin.x + px(2.0),
            error_details_bounds.origin.y + error_details_bounds.size.height - px(2.0),
        ),
        point(
            error_details_bounds.origin.x + error_details_bounds.size.width - px(2.0),
            error_details_bounds.origin.y + error_details_bounds.size.height - px(2.0),
        ),
    ];
    for pointer_position in expand_pointer_points {
        panel_window.simulate_mouse_move(
            pointer_position,
            None::<MouseButton>,
            Modifiers::default(),
        );
        panel_window.simulate_click(pointer_position, Modifiers::default());
        panel_window.run_until_parked();
        if panel
            .read_with(panel_window, |panel, _| {
                panel.surface_action_invocation_trace_for_test()
            })
            .len()
            > expand_invocation_before.len()
        {
            break;
        }
    }
    let one_expanded_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    let expanded_attempts = failure_attempt_ids
        .iter()
        .filter(|attempt_id| {
            !one_expanded_state
                .collapsed_error_attempts
                .contains(attempt_id)
        })
        .copied()
        .collect::<Vec<_>>();
    if expanded_attempts.len() != 1
        || failure_attempt_ids
            .iter()
            .filter(|attempt_id| !expanded_attempts.contains(attempt_id))
            .any(|attempt_id| {
                !one_expanded_state
                    .collapsed_error_attempts
                    .contains(attempt_id)
            })
        || panel_window
            .debug_bounds("SIM-SURFACE-ERROR-DETAILS-CONTENT")
            .is_none()
    {
        let expand_trace_after = panel.read_with(panel_window, |panel, _| {
            panel.surface_action_trace_for_test()
        });
        let expand_invocation_after = panel.read_with(panel_window, |panel, _| {
            panel.surface_action_invocation_trace_for_test()
        });
        return Err(format!(
            "expanding one collapsed failure changed another failure's collapse state: expected_ids={failure_attempt_ids:?}, collapsed_before={:?}, collapsed_after={:?}, computed_expanded={expanded_attempts:?}, details_visible={}, trigger_bounds={error_details_bounds:?}, points={expand_pointer_points:?}, errors_root={errors_root_bounds:?}, group_list={error_group_list_bounds:?}, first_card={first_error_card_bounds:?}, node_card={error_node_card_bounds:?}, main_region={execution_main_region_bounds:?}, main_content={execution_main_content_bounds:?}, viewport={error_surface_viewport:?}, reducer_trace_before={expand_trace_before:?}, reducer_trace_after={expand_trace_after:?}, invocation_trace_before={expand_invocation_before:?}, invocation_trace_after={expand_invocation_after:?}",
            fully_collapsed_state.collapsed_error_attempts,
            one_expanded_state.collapsed_error_attempts,
            panel_window
                .debug_bounds("SIM-SURFACE-ERROR-DETAILS-CONTENT")
                .is_some(),
        ));
    }
    let error_details_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-DETAILS-TRIGGER-FIRST")
        .ok_or_else(|| "expanded error details trigger disappeared".to_owned())?;
    panel_window.simulate_click(error_details_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    let recollapsed_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if failure_attempt_ids.iter().any(|attempt_id| {
        !recollapsed_state
            .collapsed_error_attempts
            .contains(attempt_id)
    }) || panel_window
        .debug_bounds("SIM-SURFACE-ERROR-DETAILS-CONTENT")
        .is_some()
    {
        return Err("toggling the same failure did not collapse it again".to_owned());
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_output_action_for_test(
            OutputViewAction::SelectAttempt(distinct_failure_attempt_id),
            window,
            cx,
        );
        panel.set_test_state(
            ExecutionPanelTab::Queue,
            Some(distinct_failure_attempt_id),
            "all",
            "all",
            0,
            cx,
        )
    });
    panel_window.run_until_parked();
    let context_state_before =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if context_state_before.selected_attempt_id != Some(distinct_failure_attempt_id) {
        return Err(format!(
            "failed-job context fixture selected the wrong attempt before pointer input: expected={distinct_failure_attempt_id:?}, selected={:?}",
            context_state_before.selected_attempt_id
        ));
    }
    let context_trigger_bounds = panel_window
        .debug_bounds("SIM-SURFACE-JOB-CONTEXT-TRIGGER")
        .ok_or_else(|| "failed-job context trigger has no bounds".to_owned())?;
    let context_trace_before = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_trace_for_test()
    });
    let context_invocation_before = panel.read_with(panel_window, |panel, _| {
        panel.surface_action_invocation_trace_for_test()
    });
    panel_window.simulate_click(context_trigger_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    let context_state_after =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if context_state_after.job_context_attempt_id != Some(distinct_failure_attempt_id) {
        let context_trace_after = panel.read_with(panel_window, |panel, _| {
            panel.surface_action_trace_for_test()
        });
        let context_invocation_after = panel.read_with(panel_window, |panel, _| {
            panel.surface_action_invocation_trace_for_test()
        });
        return Err(format!(
            "clicking the context trigger targeted the wrong failed attempt: expected={distinct_failure_attempt_id:?}, selected_before={:?}, selected_after={:?}, context_before={:?}, context_after={:?}, bounds={context_trigger_bounds:?}, reducer_trace_before={context_trace_before:?}, reducer_trace_after={context_trace_after:?}, invocation_trace_before={context_invocation_before:?}, invocation_trace_after={context_invocation_after:?}",
            context_state_before.selected_attempt_id,
            context_state_after.selected_attempt_id,
            context_state_before.job_context_attempt_id,
            context_state_after.job_context_attempt_id,
        ));
    }
    let copy_error_bounds = panel_window
        .debug_bounds("SIM-SURFACE-JOB-CONTEXT-COPY-ERROR")
        .ok_or_else(|| "failed-job context menu rendered no Copy Error item".to_owned())?;
    panel_window.simulate_click(copy_error_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    let copied_context_error = panel_window
        .read_from_clipboard()
        .and_then(|item| item.text())
        .ok_or_else(|| "context-menu Copy Error wrote no clipboard text".to_owned())?;
    let expected_context_error = "origin: Unknown\ndistinct_failure: second deterministic node failure\nretryable: true\nnode: error-node";
    if copied_context_error != expected_context_error {
        return Err(format!(
            "context-menu Copy Error clipboard mismatch: {copied_context_error:?}"
        ));
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(ExecutionSurfaceAction::ViewErrors, window, cx);
    });
    panel_window.run_until_parked();
    for selector in ["SIM-SURFACE-ERROR-COPY", "SIM-SURFACE-ERROR-GITHUB"] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!("structured error card did not render `{selector}`"));
        }
    }
    let error_copy_bounds = panel_window
        .debug_bounds("SIM-SURFACE-ERROR-COPY-FIRST")
        .ok_or_else(|| "structured error Copy action has no bounds".to_owned())?;
    panel_window.simulate_click(error_copy_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    if panel_window
        .read_from_clipboard()
        .and_then(|item| item.text())
        .as_deref()
        != Some(expected_context_error)
    {
        return Err("error-card Copy did not copy the exact structured failure".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-084",
        "the error card rendered measured Find on GitHub and Copy controls, and clicking Copy wrote the exact structured failure text",
    )?;

    for _ in 0..64 {
        let current_notification_identity = panel.read_with(panel_window, |panel, _| {
            panel.surface_state_for_test().current_notification_identity
        });
        let Some(current_notification_identity) = current_notification_identity else {
            break;
        };
        panel.update_in(panel_window, |panel, window, cx| {
            panel.handle_surface_action_for_test(
                ExecutionSurfaceAction::DismissNotification(current_notification_identity),
                window,
                cx,
            );
        });
        panel_window.run_until_parked();
    }
    let notification_setup_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if notification_setup_state.notification_count != 0
        || notification_setup_state
            .current_notification_identity
            .is_some()
    {
        return Err(format!(
            "notification conformance fixture could not drain prior execution activity: count={}, current={:?}, bounded={:?}",
            notification_setup_state.notification_count,
            notification_setup_state.current_notification_identity,
            notification_setup_state.bounded_counts,
        ));
    }

    let queueing_request = request(0x701);
    let mismatched_request = request(0x702);
    panel.update(panel_window, |panel, cx| {
        panel.observe_prompt_queueing_for_test(queueing_request, 1, cx);
    });
    panel_window.run_until_parked();
    for (feature_id, selector, assertion) in [
        (
            "COMFY-FRONTEND-SURFACE-67797BF57062",
            "SIM-SURFACE-QUEUE-NOTIFICATION-BANNER",
            "promptQueueing rendered a request-correlated current banner",
        ),
        (
            "COMFY-FRONTEND-SURFACE-63A4ABE54AC4",
            "SIM-SURFACE-QUEUE-NOTIFICATION-BANNER-HOST",
            "the notification host retained FIFO state while rendering one current banner",
        ),
    ] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "runtime component `{feature_id}` did not render `{selector}`"
            ));
        }
        record_component_behavior(
            &mut rendered_component_evidence,
            feature_id,
            selector,
            assertion,
        )?;
    }
    let queueing_state = panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    let queueing_identity = queueing_state
        .current_notification_identity
        .ok_or_else(|| "promptQueueing did not create a notification".to_owned())?;
    if queueing_state.notification_count != 1
        || !matches!(
            queueing_state.current_notification,
            Some((ExecutionNotificationKind::Queueing, Some(id), 1, _)) if id == queueing_request
        )
    {
        return Err("promptQueueing did not create the expected single-job banner".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-100",
        "a promptQueueing event rendered one request-correlated queueing banner",
    )?;
    panel.update(panel_window, |panel, cx| {
        panel.observe_prompt_queued_for_test(queueing_request, 3, attempt_id, cx);
    });
    let upgraded_state = panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if upgraded_state.current_notification_identity != Some(queueing_identity)
        || !matches!(
            upgraded_state.current_notification,
            Some((ExecutionNotificationKind::Queued, Some(id), 3, ref message))
                if id == queueing_request && message.contains("3 jobs")
        )
    {
        return Err(
            "promptQueued did not upgrade its request-correlated banner with plural text"
                .to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-101",
        "promptQueued upgraded the same request banner identity from queueing to queued",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-102",
        "a three-job promptQueued acknowledgement rendered plural queue text",
    )?;
    panel.update(panel_window, |panel, cx| {
        panel.observe_prompt_queued_for_test(mismatched_request, 1, attempt(0x702), cx);
    });
    let queued_state = panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if queued_state.notification_count != 2
        || queued_state.current_notification_identity != Some(queueing_identity)
    {
        return Err("mismatched promptQueued request did not enter the FIFO".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-103",
        "a mismatched promptQueued request entered a distinct FIFO banner behind the current request",
    )?;
    panel_window
        .background_executor
        .advance_clock(Duration::from_secs(4));
    panel_window.run_until_parked();
    let after_first_timeout =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if after_first_timeout.notification_count != 1
        || !matches!(
            after_first_timeout.current_notification,
            Some((ExecutionNotificationKind::Queued, Some(id), 1, _)) if id == mismatched_request
        )
    {
        return Err("notification timeout did not reveal the next FIFO banner".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-104",
        "advancing the GPUI clock by the banner timeout dismissed the current notification",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-105",
        "the second FIFO notification became current after the first timed out",
    )?;
    panel_window
        .background_executor
        .advance_clock(Duration::from_secs(4));
    panel_window.run_until_parked();
    if panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().notification_count
    }) != 0
    {
        return Err("second queue notification did not auto-dismiss".to_owned());
    }
    let direct_queued_request = request(0x703);
    panel.update(panel_window, |panel, cx| {
        panel.observe_prompt_queued_for_test(direct_queued_request, 0, attempt(0x703), cx);
    });
    panel_window.run_until_parked();
    let direct_state = panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    let direct_identity = direct_state
        .current_notification_identity
        .ok_or_else(|| "direct promptQueued did not create a banner".to_owned())?;
    if !matches!(
        direct_state.current_notification,
        Some((ExecutionNotificationKind::Queued, Some(id), 1, _)) if id == direct_queued_request
    ) {
        return Err("direct promptQueued did not sanitize and render its batch count".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-106",
        "promptQueued without prior queueing directly rendered a queued banner with a sanitized batch count",
    )?;
    let selected_job_attempt_id = direct_state
        .selected_attempt_id
        .ok_or_else(|| "job details surface has no selected auto-follow attempt".to_owned())?;
    let details_hover_trigger = panel_window
        .debug_bounds("SIM-SURFACE-JOB-DETAILS-HOVER-TRIGGER")
        .ok_or_else(|| "job details hover trigger has no GPUI bounds".to_owned())?;
    panel_window.simulate_mouse_move(
        details_hover_trigger.center(),
        None::<MouseButton>,
        Modifiers::default(),
    );
    panel_window.run_until_parked();
    let trigger_hover_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    let hovered_job_attempt_id = trigger_hover_state
        .job_details_trigger_hovered
        .ok_or_else(|| "mouseenter did not identify a hovered job".to_owned())?;
    if trigger_hover_state.job_details_hover_attempt_id != Some(hovered_job_attempt_id) {
        return Err(format!(
            "mouseenter on the job details trigger did not open exact hovered details: trigger={:?}, hover={:?}",
            trigger_hover_state.job_details_trigger_hovered,
            trigger_hover_state.job_details_hover_attempt_id,
        ));
    }
    let hovered_details = panel_window
        .debug_bounds("SIM-SURFACE-JOB-DETAILS-POPOVER")
        .ok_or_else(|| "job details hover content did not render after mouseenter".to_owned())?;
    panel_window.simulate_mouse_move(
        hovered_details.center(),
        None::<MouseButton>,
        Modifiers::default(),
    );
    panel_window.run_until_parked();
    let content_hover_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if content_hover_state.job_details_content_hovered != Some(hovered_job_attempt_id)
        || content_hover_state.job_details_hover_attempt_id != Some(hovered_job_attempt_id)
    {
        return Err(
            "moving from the details trigger into its content did not retain the hover surface"
                .to_owned(),
        );
    }
    panel_window.simulate_mouse_move(
        point(px(1.0), px(1.0)),
        None::<MouseButton>,
        Modifiers::default(),
    );
    panel_window
        .background_executor
        .advance_clock(Duration::from_secs(1));
    panel_window.run_until_parked();
    let closed_hover_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if closed_hover_state.job_details_trigger_hovered.is_some()
        || closed_hover_state.job_details_content_hovered.is_some()
        || closed_hover_state.job_details_hover_attempt_id.is_some()
        || panel_window
            .debug_bounds("SIM-SURFACE-JOB-DETAILS-POPOVER")
            .is_some()
    {
        return Err(
            "mouseleave from both job-details trigger and content did not close hover details"
                .to_owned(),
        );
    }
    panel.update_in(panel_window, |panel, window, cx| {
        for action in [
            ExecutionSurfaceAction::SelectJobTab(ExecutionJobTab::Active),
            ExecutionSurfaceAction::SetJobSearch("fixture failure".to_owned()),
            ExecutionSurfaceAction::ToggleWorkflowFilter,
            ExecutionSurfaceAction::CycleSortMode,
            ExecutionSurfaceAction::ToggleShowProgress,
            ExecutionSurfaceAction::SetErrorSearch("fixture_failure".to_owned()),
            ExecutionSurfaceAction::ToggleJobContextMenu(selected_job_attempt_id),
        ] {
            panel.handle_surface_action_for_test(action, window, cx);
        }
    });
    panel_window.run_until_parked();
    let context_menu_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if context_menu_state.job_context_attempt_id != Some(selected_job_attempt_id)
        || context_menu_state.job_details_attempt_id.is_some()
    {
        return Err(format!(
            "job context-menu action did not target the selected attempt: expected={selected_job_attempt_id:?}, selected={:?}, context={:?}, pinned_details={:?}, hover_details={:?}, trigger_hovered={:?}, content_hovered={:?}, selected_tab={:?}, selected_job_tab={:?}",
            context_menu_state.selected_attempt_id,
            context_menu_state.job_context_attempt_id,
            context_menu_state.job_details_attempt_id,
            context_menu_state.job_details_hover_attempt_id,
            context_menu_state.job_details_trigger_hovered,
            context_menu_state.job_details_content_hovered,
            context_menu_state.selected_tab,
            context_menu_state.selected_job_tab,
        ));
    }
    for (feature_id, selector, assertion) in [
        (
            "COMFY-FRONTEND-SURFACE-6085B98C498A",
            "SIM-SURFACE-JOB-CONTEXT-MENU",
            "opening actions targeted the selected attempt and rendered its typed menu",
        ),
        (
            "COMFY-FRONTEND-SURFACE-F6FF6DAE75BF",
            "SIM-SURFACE-JOB-DETAILS-HOVER-POPOVER",
            "trigger mouseenter opened exact-attempt details, content mouseenter retained them, and leaving both closed them after the GPUI delay",
        ),
    ] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "runtime component `{feature_id}` did not render `{selector}`"
            ));
        }
        record_component_behavior(
            &mut rendered_component_evidence,
            feature_id,
            selector,
            assertion,
        )?;
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::ToggleJobContextMenu(selected_job_attempt_id),
            window,
            cx,
        );
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::ToggleJobDetails(selected_job_attempt_id),
            window,
            cx,
        );
    });
    panel_window.run_until_parked();
    let details_state = panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if details_state.job_details_attempt_id != Some(selected_job_attempt_id)
        || details_state.job_context_attempt_id.is_some()
    {
        return Err("job details action did not replace the context menu".to_owned());
    }
    if panel_window
        .debug_bounds("SIM-SURFACE-JOB-DETAILS-POPOVER")
        .is_none()
    {
        return Err(
            "runtime component `COMFY-FRONTEND-SURFACE-F3428874E71D` did not render job details"
                .to_owned(),
        );
    }
    for selector in ["SIM-JOB-DEVICE", "SIM-JOB-MEMORY"] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "job details omitted effective backend projection `{selector}`"
            ));
        }
    }
    record_component_behavior(
        &mut rendered_component_evidence,
        "COMFY-FRONTEND-SURFACE-F3428874E71D",
        "SIM-SURFACE-JOB-DETAILS-POPOVER",
        "opening details replaced the context menu and rendered bounded details for the selected attempt",
    )?;
    panel.update_in(panel_window, |panel, window, cx| {
        for action in [
            ExecutionSurfaceAction::CopyAttemptId(attempt_id),
            ExecutionSurfaceAction::CopyError(attempt_id),
            ExecutionSurfaceAction::DismissNotification(direct_identity),
            ExecutionSurfaceAction::ViewErrors,
        ] {
            panel.handle_surface_action_for_test(action, window, cx);
        }
    });
    panel_window.run_until_parked();
    for (feature_id, selector, assertion) in [
        (
            "COMFY-FRONTEND-SURFACE-A14F4CA91E43",
            "SIM-SURFACE-ERROR-CARD-SECTION-FIRST",
            "the searched failure rendered its structured details section",
        ),
        (
            "COMFY-FRONTEND-SURFACE-E721F4A4F9B9",
            "SIM-SURFACE-ERROR-GROUP-LIST",
            "the Errors tab grouped the filtered structured failure",
        ),
        (
            "COMFY-FRONTEND-SURFACE-97D04E89D68E",
            "SIM-SURFACE-ERROR-NODE-CARD",
            "the node-scoped runtime failure rendered as an error card",
        ),
        (
            "COMFY-FRONTEND-SURFACE-F69CDE266EDA",
            "SIM-SURFACE-TAB-ERRORS",
            "View Errors selected the searchable native Errors tab",
        ),
    ] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "runtime component `{feature_id}` did not render `{selector}`"
            ));
        }
        record_component_behavior(
            &mut rendered_component_evidence,
            feature_id,
            selector,
            assertion,
        )?;
    }
    let exercised_surface_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if exercised_surface_state.selected_job_tab != ExecutionJobTab::Active
        || exercised_surface_state.job_search_query != "fixture failure"
        || exercised_surface_state.workflow_filter != ExecutionWorkflowFilter::Selected
        || exercised_surface_state.sort_mode != ExecutionSortMode::Oldest
        || exercised_surface_state.show_progress
        || exercised_surface_state.error_search_query != "fixture_failure"
        || !exercised_surface_state.errors_all_collapsed
        || exercised_surface_state.notification_count != 0
    {
        return Err("execution surface actions did not update their durable state".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-067",
        "activating Show Run Progress toggled the durable progress setting off",
    )?;
    let copied_surface_error = panel_window
        .read_from_clipboard()
        .and_then(|item| item.text())
        .ok_or_else(|| "surface error copy wrote no clipboard text".to_owned())?;
    if !copied_surface_error.contains("fixture_failure")
        || !error_matches_query(
            &model
                .read_with(panel_window, |model, _| model.snapshot(profile_id))
                .map_err(|error| error.to_string())?
                .attempts
                .into_iter()
                .find(|attempt| attempt.attempt_id == attempt_id)
                .ok_or_else(|| "failed attempt disappeared before error search".to_owned())?,
            &ExecutionFailure::new("fixture_failure", "deterministic node failure")
                .at_node(NodeId("error-node".to_owned())),
            "error-node",
        )
    {
        return Err("error search or copy did not retain structured runtime details".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-083",
        "the Errors-tab search retained the exact node-scoped structured failure",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-085",
        "the execution error group rendered the native runtime failure code, message, and node",
    )?;
    panel.update_in(panel_window, |panel, window, cx| {
        for action in [
            ExecutionSurfaceAction::ToggleShowProgress,
            ExecutionSurfaceAction::SetJobSearch(String::new()),
            ExecutionSurfaceAction::ToggleWorkflowFilter,
        ] {
            panel.handle_surface_action_for_test(action, window, cx);
        }
    });
    let reset_surface_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if !reset_surface_state.show_progress
        || !reset_surface_state.job_search_query.is_empty()
        || reset_surface_state.workflow_filter != ExecutionWorkflowFilter::All
    {
        return Err("progress/filter reset did not restore the full queue surface".to_owned());
    }

    panel.update(panel_window, |panel, cx| {
        for index in 0_u128..300 {
            panel.observe_prompt_queueing_for_test(request(0x1000 + index), 1, cx);
        }
    });
    panel_window.run_until_parked();
    let overflow_counts = panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().bounded_counts
    });
    if overflow_counts.observed_attempt_states
        > crate::execution_surfaces::EXECUTION_ATTEMPT_TRACKING_CAPACITY
        || overflow_counts.observed_queueing_requests
            > crate::execution_surfaces::EXECUTION_REQUEST_TRACKING_CAPACITY
        || overflow_counts.observed_queued_requests
            > crate::execution_surfaces::EXECUTION_REQUEST_TRACKING_CAPACITY
        || overflow_counts.queue_request_batch_counts
            > crate::execution_surfaces::EXECUTION_REQUEST_TRACKING_CAPACITY
        || overflow_counts.pending_notifications
            > crate::execution_surfaces::EXECUTION_NOTIFICATION_FIFO_CAPACITY
        || overflow_counts.current_notification > 1
        || overflow_counts.coalesced_notifications == 0
    {
        return Err(format!(
            "high-volume queueing state exceeded a declared bound: {overflow_counts:?}"
        ));
    }
    for _ in 0..40 {
        panel_window
            .background_executor
            .advance_clock(Duration::from_secs(4));
        panel_window.run_until_parked();
    }
    let newest_retained_request = request(0x1000 + 299);
    panel.update(panel_window, |panel, cx| {
        panel.observe_prompt_queued_for_test(
            newest_retained_request,
            2,
            distinct_failure_attempt_id,
            cx,
        );
    });
    panel_window.run_until_parked();
    let newest_queued_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if !matches!(
        newest_queued_state.current_notification,
        Some((ExecutionNotificationKind::Queued, Some(request_id), 2, _))
            if request_id == newest_retained_request
    ) {
        return Err(
            "the newest retained RequestId did not correlate queueing to queued after eviction"
                .to_owned(),
        );
    }
    let queued_identity = newest_queued_state
        .current_notification_identity
        .ok_or_else(|| "newest correlated queued banner has no identity".to_owned())?;
    let newest_failure_request = request(0x5000);
    let overflow_failure =
        ExecutionFailure::new("bounded_failure", "bounded terminal notification fixture");
    panel.update(panel_window, |panel, cx| {
        panel.observe_prompt_queueing_for_test(newest_failure_request, 1, cx);
        panel.observe_prompt_queue_failed_for_test(
            newest_failure_request,
            1,
            &overflow_failure,
            cx,
        );
    });
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::DismissNotification(queued_identity),
            window,
            cx,
        );
    });
    panel_window.run_until_parked();
    let correlated_failure_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if !matches!(
        correlated_failure_state.current_notification,
        Some((ExecutionNotificationKind::Failure, Some(request_id), 1, ref message))
            if request_id == newest_failure_request && message.contains("bounded_failure")
    ) {
        return Err(
            "the newest retained RequestId did not correlate queueing to failure after eviction"
                .to_owned(),
        );
    }
    let terminal_identity = correlated_failure_state
        .current_notification_identity
        .ok_or_else(|| "correlated failure banner has no identity".to_owned())?;
    panel.update(panel_window, |panel, cx| {
        for index in 0_u128..40 {
            panel.observe_prompt_queue_failed_for_test(
                request(0x5100 + index),
                1,
                &overflow_failure,
                cx,
            );
        }
    });
    panel_window.run_until_parked();
    let terminal_overflow_counts = panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().bounded_counts
    });
    if terminal_overflow_counts.observed_attempt_states
        > crate::execution_surfaces::EXECUTION_ATTEMPT_TRACKING_CAPACITY
        || terminal_overflow_counts.observed_queueing_requests
            > crate::execution_surfaces::EXECUTION_REQUEST_TRACKING_CAPACITY
        || terminal_overflow_counts.observed_queued_requests
            > crate::execution_surfaces::EXECUTION_REQUEST_TRACKING_CAPACITY
        || terminal_overflow_counts.queue_request_batch_counts
            > crate::execution_surfaces::EXECUTION_REQUEST_TRACKING_CAPACITY
        || terminal_overflow_counts.pending_notifications
            > crate::execution_surfaces::EXECUTION_NOTIFICATION_FIFO_CAPACITY
        || terminal_overflow_counts.current_notification > 1
        || terminal_overflow_counts.coalesced_failures == 0
    {
        return Err(format!(
            "terminal notification overflow exceeded a declared bound: {terminal_overflow_counts:?}"
        ));
    }
    panel_window
        .background_executor
        .advance_clock(Duration::from_secs(4));
    panel_window.run_until_parked();
    if panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().current_notification_identity
    }) == Some(terminal_identity)
    {
        return Err("the notification timer did not advance the bounded terminal FIFO".to_owned());
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(ExecutionSurfaceAction::ViewErrors, window, cx);
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::SetErrorSearch(String::new()),
            window,
            cx,
        );
    });
    panel_window.run_until_parked();
    let durable_failure_ids = panel.read_with(panel_window, |panel, cx| {
        panel.filtered_failure_attempt_ids_for_test(cx)
    });
    if !durable_failure_ids.contains(&initial_overlay_attempt_id)
        || !durable_failure_ids.contains(&distinct_failure_attempt_id)
        || panel_window
            .debug_bounds("SIM-SURFACE-ERROR-GROUP-LIST")
            .is_none()
    {
        return Err(
            "bounded notification eviction removed durable failures from the Errors tab".to_owned(),
        );
    }

    let projection_records = lifecycle_projection_records(profile_id)?;
    model
        .update(panel_window, |model, cx| {
            model.reconcile(
                ExecutionReconciliation {
                    profile_id,
                    source_revision: 1,
                    source: ExecutionDataSource::Recovery,
                    status: ExecutionSnapshotStatus::Ready,
                    queue: Vec::new(),
                    records: projection_records,
                    plans: Vec::new(),
                    acknowledged_requests: Vec::new(),
                },
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    panel.update(panel_window, |panel, cx| {
        panel.set_test_state(ExecutionPanelTab::Queue, None, "all", "all", 0, cx)
    });
    panel_window.run_until_parked();
    for (feature_id, selector, assertion) in [
        (
            "COMFY-FRONTEND-SURFACE-19BAB3FC51C6",
            "SIM-SURFACE-QUEUE-INLINE-PROGRESS",
            "a projected running attempt rendered per-attempt inline progress",
        ),
        (
            "COMFY-FRONTEND-SURFACE-BA68BC33A2AB",
            "SIM-SURFACE-QUEUE-INLINE-PROGRESS-SUMMARY",
            "the running attempt exposed its numeric accessible progress summary",
        ),
        (
            "COMFY-FRONTEND-SURFACE-F494BDB6FD2E",
            "SIM-SURFACE-QUEUE-OVERLAY-ACTIVE",
            "the execution dock grouped non-terminal attempts in its Active section",
        ),
        (
            "COMFY-FRONTEND-SURFACE-052D51C10184",
            "SIM-SURFACE-QUEUE-OVERLAY-EXPANDED",
            "opening the dock queue surface rendered its expanded job group",
        ),
        (
            "COMFY-FRONTEND-SURFACE-0C01631C3DFA",
            "SIM-SURFACE-QUEUE-OVERLAY-HEADER",
            "the dock queue overlay header reflected the reconciled snapshot",
        ),
        (
            "COMFY-FRONTEND-SURFACE-BE5BE58D2FDE",
            "SIM-SURFACE-QUEUE-PROGRESS-OVERLAY",
            "the dock queue overlay projected the running attempt's progress",
        ),
        (
            "COMFY-FRONTEND-SURFACE-9F0D36286AB9",
            "SIM-SURFACE-JOB-FILTER-ACTIONS",
            "search, workflow, sort, and progress actions mutated durable filter state",
        ),
        (
            "COMFY-FRONTEND-SURFACE-FB9FC24AF7FA",
            "SIM-SURFACE-JOB-FILTER-TABS",
            "selecting a job filter tab updated the active panel view",
        ),
        (
            "COMFY-FRONTEND-SURFACE-BC42336531A9",
            "SIM-SURFACE-JOB-FILTERS-BAR",
            "the persistent filter bar survived snapshot reconciliation",
        ),
        (
            "COMFY-FRONTEND-SURFACE-922B12C3CA3D",
            "SIM-SURFACE-ERROR-OVERLAY",
            "a failed projection rendered a dismissible dock error overlay while retaining history",
        ),
    ] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "runtime component `{feature_id}` did not render `{selector}`"
            ));
        }
        record_component_behavior(
            &mut rendered_component_evidence,
            feature_id,
            selector,
            assertion,
        )?;
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-065",
        "the execution dock rendered the Show Run Progress action with real GPUI bounds",
    )?;
    let projected_snapshot = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let projected_states = projected_snapshot
        .attempts
        .iter()
        .map(|attempt| format!("{:?}", attempt.state))
        .collect::<BTreeSet<_>>();
    let expected_states = BTreeSet::from([
        "Queued".to_owned(),
        "Running".to_owned(),
        "Cancelling".to_owned(),
        "Succeeded".to_owned(),
        "Failed".to_owned(),
        "Cancelled".to_owned(),
        "Interrupted".to_owned(),
    ]);
    if projected_states != expected_states
        || projected_snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                matches!(
                    attempt.source_projection,
                    Some(AttemptSourceProjection::Provider { .. })
                )
            })
            .count()
            != 8
        || projected_snapshot
            .attempts
            .iter()
            .filter(|attempt| {
                matches!(
                    attempt.source_projection,
                    Some(AttemptSourceProjection::Unknown { .. })
                )
            })
            .count()
            != 1
    {
        return Err("rendered lifecycle/provider/unknown projections are incomplete".to_owned());
    }
    let running_attempt = projected_snapshot
        .attempts
        .iter()
        .find(|attempt| attempt.state == AttemptState::Running && attempt.progress.is_some())
        .cloned()
        .ok_or_else(|| "running progress-toast fixture was not projected".to_owned())?;
    panel.update(panel_window, |panel, cx| {
        panel.set_test_state(
            ExecutionPanelTab::Output,
            Some(running_attempt.attempt_id),
            "all",
            "all",
            0,
            cx,
        );
        panel.set_output_auto_follow_for_test(true, cx);
    });
    panel_window.run_until_parked();
    for selector in [
        "SIM-SURFACE-OUTPUT-HISTORY-ACTIVE-QUEUE-ITEM",
        "SIM-OUTPUT-HISTORY-SKELETON",
    ] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "in-progress output surface did not render `{selector}`"
            ));
        }
    }
    record_component_behavior(
        &mut rendered_component_evidence,
        "COMFY-FRONTEND-SURFACE-1F516FA6CD5A",
        "SIM-SURFACE-OUTPUT-HISTORY-ACTIVE-QUEUE-ITEM",
        "selecting Output with a running attempt rendered the active item outside the history scroll region and its skeleton",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-072",
        "starting a native attempt rendered its output skeleton",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-078",
        "the active output item rendered outside the output-history scrolling region",
    )?;
    let initially_following =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if !initially_following.output_auto_follow
        || initially_following.selected_attempt_id.is_none()
        || !initially_following
            .in_progress_attempt_ids
            .contains(&running_attempt.attempt_id)
    {
        return Err(format!(
            "output history did not auto-follow an in-progress attempt: expected_running={:?}, output_auto_follow={}, selected={:?}, in_progress={:?}, projected_running={:?}",
            running_attempt.attempt_id,
            initially_following.output_auto_follow,
            initially_following.selected_attempt_id,
            initially_following.in_progress_attempt_ids,
            projected_snapshot
                .attempts
                .iter()
                .filter(|attempt| !attempt.state.is_terminal())
                .map(|attempt| (attempt.attempt_id, attempt.state))
                .collect::<Vec<_>>(),
        ));
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_output_action_for_test(
            OutputViewAction::SelectAttempt(running_attempt.attempt_id),
            window,
            cx,
        );
    });
    let broken_follow = panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if broken_follow.output_auto_follow
        || broken_follow.selected_attempt_id != Some(running_attempt.attempt_id)
    {
        return Err("clicking an output-history item did not break auto-follow".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-077",
        "clicking an in-progress output-history item disabled automatic following and retained that exact attempt",
    )?;
    for _ in 0..64 {
        let current_notification_identity = panel.read_with(panel_window, |panel, _| {
            panel.surface_state_for_test().current_notification_identity
        });
        let Some(current_notification_identity) = current_notification_identity else {
            break;
        };
        panel.update_in(panel_window, |panel, window, cx| {
            panel.handle_surface_action_for_test(
                ExecutionSurfaceAction::DismissNotification(current_notification_identity),
                window,
                cx,
            );
        });
        panel_window.run_until_parked();
    }
    let model_event_notification_setup =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if model_event_notification_setup.notification_count != 0
        || model_event_notification_setup
            .current_notification_identity
            .is_some()
    {
        return Err(format!(
            "model-event notification fixture could not drain prior bounded failures: count={}, current={:?}, bounded={:?}",
            model_event_notification_setup.notification_count,
            model_event_notification_setup.current_notification_identity,
            model_event_notification_setup.bounded_counts,
        ));
    }
    let new_attempt_acknowledgement = model
        .update(panel_window, |model, cx| {
            model.dispatch(
                profile_id,
                ExecutionControlCommandKind::Queue {
                    plan: plan(prompt(0x8f0)),
                    priority: 0,
                    front: false,
                },
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    let new_attempt_id = match new_attempt_acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(attempt_id),
        } => attempt_id,
        outcome => {
            return Err(format!(
                "new auto-follow fixture was not accepted: {outcome:?}"
            ));
        }
    };
    panel_window.run_until_parked();
    let restarted_follow = panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if !restarted_follow.output_auto_follow
        || restarted_follow.selected_attempt_id != Some(new_attempt_id)
    {
        return Err("a new execution did not reset auto-follow to the latest attempt".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-076",
        "a newly queued native attempt reset auto-follow to the newest in-progress item",
    )?;
    let lifecycle_notification_identity = restarted_follow
        .current_notification_identity
        .ok_or_else(|| "model queue dispatch emitted no queue lifecycle banner".to_owned())?;
    if !matches!(
        restarted_follow.current_notification,
        Some((ExecutionNotificationKind::Queued, Some(request_id), 1, _))
            if request_id == new_attempt_acknowledgement.request_id
    ) {
        return Err(format!(
            "model command submission/acknowledgement did not drive a correlated queued banner: expected_request={:?}, current={:?}, identity={:?}, count={}, bounded={:?}",
            new_attempt_acknowledgement.request_id,
            restarted_follow.current_notification,
            restarted_follow.current_notification_identity,
            restarted_follow.notification_count,
            restarted_follow.bounded_counts,
        ));
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(
            ExecutionSurfaceAction::DismissNotification(lifecycle_notification_identity),
            window,
            cx,
        );
    });
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_output_action_for_test(
            OutputViewAction::CancelAttempt(new_attempt_id),
            window,
            cx,
        );
    });
    panel_window.run_until_parked();
    if model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == new_attempt_id)
        .is_none_or(|attempt| attempt.state != AttemptState::Cancelled)
    {
        return Err("output skeleton cancel did not cancel its exact queued attempt".to_owned());
    }
    let running_toast = panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().progress_toast
    });
    let Some((running_toast_identity, ExecutionProgressToastPhase::Running)) = running_toast else {
        return Err(format!(
            "running progress toast did not appear: {running_toast:?}"
        ));
    };
    for (feature_id, selector, assertion) in [
        (
            "COMFY-FRONTEND-SURFACE-92B3D7C9D258",
            "SIM-SURFACE-PROGRESS-TOAST",
            "a running attempt created a progress toast that later transitioned to a new terminal identity",
        ),
        (
            "COMFY-FRONTEND-SURFACE-228A24CC9226",
            "SIM-SURFACE-LINEAR-PROGRESS-BAR",
            "the toast rendered bounded numeric completed and total progress",
        ),
    ] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "runtime component `{feature_id}` did not render `{selector}`"
            ));
        }
        record_component_behavior(
            &mut rendered_component_evidence,
            feature_id,
            selector,
            assertion,
        )?;
    }
    model.update(panel_window, |model, cx| {
        model.ingest_event_batch(
            vec![event(
                profile_id,
                running_attempt.prompt_id,
                running_attempt.attempt_id,
                2,
                None,
                AttemptEventKind::Succeeded,
            )],
            cx,
        )
    });
    panel_window.run_until_parked();
    let terminal_toast = panel.read_with(panel_window, |panel, _| {
        panel.surface_state_for_test().progress_toast
    });
    if !matches!(
        terminal_toast,
        Some((identity, ExecutionProgressToastPhase::Succeeded)) if identity != running_toast_identity
    ) {
        return Err(format!(
            "progress toast did not transition to a terminal phase: {terminal_toast:?}"
        ));
    }
    let succeeded_surface_state =
        panel.read_with(panel_window, |panel, _| panel.surface_state_for_test());
    if succeeded_surface_state
        .in_progress_attempt_ids
        .contains(&running_attempt.attempt_id)
        || model
            .read_with(panel_window, |model, _| model.snapshot(profile_id))
            .map_err(|error| error.to_string())?
            .attempts
            .iter()
            .find(|attempt| attempt.attempt_id == running_attempt.attempt_id)
            .is_none_or(|attempt| attempt.state != AttemptState::Succeeded)
    {
        return Err(
            "successful terminal transition did not remove the exact attempt from Output in-progress items"
                .to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-075",
        "the exact running Output attempt rendered in-progress, then its Succeeded event removed its AttemptId from the production active-item set while retaining terminal history",
    )?;
    panel_window
        .background_executor
        .advance_clock(Duration::from_secs(5));
    panel_window.run_until_parked();
    if panel
        .read_with(panel_window, |panel, _| {
            panel.surface_state_for_test().progress_toast
        })
        .is_some()
    {
        return Err("terminal progress toast did not auto-dismiss".to_owned());
    }
    let failed_cleanup_prompt_id = prompt(0xd00);
    let failed_cleanup_acknowledgement = model
        .update(panel_window, |model, cx| {
            model.dispatch(
                profile_id,
                ExecutionControlCommandKind::Queue {
                    plan: plan(failed_cleanup_prompt_id),
                    priority: 0,
                    front: false,
                },
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    let failed_cleanup_attempt_id = match failed_cleanup_acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(attempt_id),
        } => attempt_id,
        outcome => {
            return Err(format!(
                "failed Output cleanup fixture was not queued: {outcome:?}"
            ));
        }
    };
    model.update(panel_window, |model, cx| {
        model.ingest_event_batch(
            vec![event(
                profile_id,
                failed_cleanup_prompt_id,
                failed_cleanup_attempt_id,
                0,
                None,
                AttemptEventKind::Started,
            )],
            cx,
        )
    });
    panel_window.run_until_parked();
    if !panel.read_with(panel_window, |panel, _| {
        panel
            .surface_state_for_test()
            .in_progress_attempt_ids
            .contains(&failed_cleanup_attempt_id)
    }) {
        return Err("failed cleanup fixture never entered Output in-progress items".to_owned());
    }
    model.update(panel_window, |model, cx| {
        model.ingest_event_batch(
            vec![event(
                profile_id,
                failed_cleanup_prompt_id,
                failed_cleanup_attempt_id,
                1,
                Some(NodeId("error-node".to_owned())),
                AttemptEventKind::Failed {
                    failure: ExecutionFailure::new(
                        "output_cleanup_failure",
                        "terminal failure cleanup fixture",
                    )
                    .at_node(NodeId("error-node".to_owned())),
                },
            )],
            cx,
        )
    });
    panel_window.run_until_parked();
    if panel.read_with(panel_window, |panel, _| {
        panel
            .surface_state_for_test()
            .in_progress_attempt_ids
            .contains(&failed_cleanup_attempt_id)
    }) || model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == failed_cleanup_attempt_id)
        .is_none_or(|attempt| attempt.state != AttemptState::Failed)
    {
        return Err(
            "failed terminal transition did not remove the exact attempt from Output in-progress items"
                .to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-079",
        "the exact running Output attempt entered the active set, then its Failed event removed that AttemptId from in-progress items while retaining failed history",
    )?;
    let interrupt_fixture_acknowledgement = model
        .update(panel_window, |model, cx| {
            model.dispatch(
                profile_id,
                ExecutionControlCommandKind::Queue {
                    plan: plan(prompt(0x8f1)),
                    priority: 0,
                    front: false,
                },
                cx,
            )
        })
        .map_err(|error| error.to_string())?;
    let interrupt_fixture_id = match interrupt_fixture_acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(attempt_id),
        } => attempt_id,
        outcome => {
            return Err(format!(
                "output interrupt fixture was not queued: {outcome:?}"
            ));
        }
    };
    model.update(panel_window, |model, cx| {
        model.ingest_event_batch(
            vec![event(
                profile_id,
                prompt(0x8f1),
                interrupt_fixture_id,
                0,
                None,
                AttemptEventKind::Started,
            )],
            cx,
        )
    });
    panel_window.run_until_parked();
    let interrupt_fixture_state_before = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == interrupt_fixture_id)
        .map(|attempt| attempt.state);
    if interrupt_fixture_state_before != Some(AttemptState::Running) {
        return Err(format!(
            "output interrupt fixture was not running before the action: {interrupt_fixture_state_before:?}"
        ));
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_output_action_for_test(
            OutputViewAction::InterruptAttempt(interrupt_fixture_id),
            window,
            cx,
        );
    });
    panel_window.run_until_parked();
    let interrupt_fixture_state_after = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == interrupt_fixture_id)
        .map(|attempt| attempt.state);
    if interrupt_fixture_state_after != Some(AttemptState::Cancelling) {
        let panel_status = panel.read_with(panel_window, |panel, _| {
            panel.status_message_for_test().map(str::to_owned)
        });
        return Err(format!(
            "output running cancel route did not request interruption for its exact attempt: state={interrupt_fixture_state_after:?}, panel_status={panel_status:?}"
        ));
    }
    for (source, status) in [
        (ExecutionDataSource::Live, ExecutionSnapshotStatus::Loading),
        (
            ExecutionDataSource::Persisted,
            ExecutionSnapshotStatus::Partial {
                failure: ExecutionFailure::new("partial_render", "partial render fixture"),
            },
        ),
        (
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Stale {
                source_revision: Some(1),
                failure: ExecutionFailure::new("stale_render", "stale render fixture"),
            },
        ),
        (
            ExecutionDataSource::Live,
            ExecutionSnapshotStatus::Unavailable {
                failure: ExecutionFailure::new("unavailable_render", "unavailable render fixture"),
            },
        ),
        (
            ExecutionDataSource::Recovery,
            ExecutionSnapshotStatus::Ready,
        ),
    ] {
        model
            .update(panel_window, |model, cx| {
                model.set_snapshot_status(profile_id, source, status, cx)
            })
            .map_err(|error| error.to_string())?;
        panel_window.run_until_parked();
    }
    let panel_operations = operation_handler
        .actions
        .lock()
        .map_err(|_| "panel operation recorder poisoned".to_owned())?;
    if panel_operations.len() != 2
        || panel_operations[0].3 != ExecutionOutputOperationAction::Recover
        || panel_operations[1].3 != ExecutionOutputOperationAction::Remove
        || panel_operations
            .iter()
            .any(|operation| operation.2 != unavailable_output_id)
    {
        return Err(
            "execution panel did not dispatch typed recovery/removal operations".to_owned(),
        );
    }

    let terminal_attempts_before_clear = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .filter(|attempt| attempt.state.is_terminal())
        .count();
    if terminal_attempts_before_clear == 0 {
        return Err("clear-history fixture contains no terminal attempts".to_owned());
    }
    panel.update_in(panel_window, |panel, window, cx| {
        panel.confirm_clear_history_for_test(window, cx)
    });
    if !panel_window.has_pending_prompt() {
        return Err("clear-history component did not open a GPUI confirmation prompt".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-051",
        "invoking the queue history action opened the native GPUI confirmation dialog",
    )?;
    record_component_behavior(
        &mut rendered_component_evidence,
        "COMFY-FRONTEND-SURFACE-26F40752861E",
        "GPUI-PROMPT-CANCEL-CLOSE-CONFIRM",
        "the real GPUI prompt preserved history for Cancel and close, reset on reopen, and cleared only on confirmation",
    )?;
    panel_window.simulate_prompt_answer("Cancel");
    panel_window.run_until_parked();
    let terminal_attempts_after_cancel = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .filter(|attempt| attempt.state.is_terminal())
        .count();
    if terminal_attempts_after_cancel != terminal_attempts_before_clear {
        return Err("clear-history Cancel removed terminal attempts".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-052",
        "Cancel closed the confirmation without removing any terminal attempt",
    )?;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.confirm_clear_history_for_test(window, cx)
    });
    panel_window.simulate_prompt_answer("×");
    panel_window.run_until_parked();
    let terminal_attempts_after_close = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        .attempts
        .iter()
        .filter(|attempt| attempt.state.is_terminal())
        .count();
    if terminal_attempts_after_close != terminal_attempts_before_clear {
        return Err("clear-history Close (×) removed terminal attempts".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-053",
        "the dialog close control closed confirmation without removing terminal attempts",
    )?;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.confirm_clear_history_for_test(window, cx)
    });
    panel_window.simulate_prompt_answer("Clear History");
    panel_window.run_until_parked();
    let after_confirm_clear = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    if after_confirm_clear
        .attempts
        .iter()
        .any(|attempt| attempt.state.is_terminal())
        || !panel
            .read_with(panel_window, |panel, _| {
                panel.status_message_for_test().map(str::to_owned)
            })
            .as_deref()
            .is_some_and(|message| message.contains("acknowledged"))
    {
        return Err("clear-history confirmation did not clear terminal attempts".to_owned());
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-054",
        "confirming Clear History removed all terminal attempts and acknowledged the command",
    )?;
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-055",
        "the confirmation reopened after both Cancel and close, proving its transient state reset",
    )?;

    panel.update_in(panel_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(ExecutionSurfaceAction::ClearPending, window, cx);
    });
    panel_window.run_until_parked();
    let empty_queue_fixture = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    if !empty_queue_fixture.queue.is_empty() {
        return Err(format!(
            "acknowledged ClearPending did not establish the empty-queue fixture: remaining={:?}",
            empty_queue_fixture
                .queue
                .iter()
                .map(|queued| queued.attempt_id)
                .collect::<Vec<_>>()
        ));
    }
    panel.update(panel_window, |panel, cx| {
        panel.set_test_state(ExecutionPanelTab::Queue, None, "all", "all", 0, cx)
    });
    panel_window.run_until_parked();
    for selector in [
        "SIM-SURFACE-CLEAR-QUEUE",
        "SIM-SURFACE-CLEAR-QUEUE-UNAVAILABLE-REASON",
    ] {
        if panel_window.debug_bounds(selector).is_none() {
            return Err(format!("empty Clear Queue did not render `{selector}`"));
        }
    }
    let empty_clear_snapshot_before = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.focus_clear_queue_for_test(window, cx)
    });
    let empty_clear_is_focused =
        panel_window.update(|window, cx| panel.read(cx).clear_queue_is_focused_for_test(window));
    if !empty_clear_is_focused {
        return Err("empty Clear Queue could not receive keyboard focus".to_owned());
    }
    panel_window.simulate_keystrokes("enter");
    panel_window.simulate_keystrokes("space");
    let empty_clear_bounds = panel_window
        .debug_bounds("SIM-SURFACE-CLEAR-QUEUE")
        .ok_or_else(|| "empty Clear Queue lost its bounds".to_owned())?;
    panel_window.simulate_click(empty_clear_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    let empty_clear_snapshot_after = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    if empty_clear_snapshot_after != empty_clear_snapshot_before {
        return Err(
            "empty Clear Queue keyboard/click activation mutated execution state".to_owned(),
        );
    }

    item.update(panel_window, |graph, cx| {
        graph.set_execution_run_mode(ExecutionRunMode::OnChange, cx)
    });
    if item.read_with(panel_window, |graph, _| graph.execution_run_mode())
        != ExecutionRunMode::OnChange
    {
        return Err("controller-loss fixture did not enter On change mode".to_owned());
    }
    let snapshot_before_controller_loss = empty_clear_snapshot_after;
    model.update(panel_window, |model, cx| model.clear_runtime_controller(cx));
    panel_window.run_until_parked();
    let snapshot_after_controller_loss = model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let controller_loss_reason = item.read_with(panel_window, |graph, cx| {
        graph.execution_queue_unavailable_reason(cx)
    });
    if item.read_with(panel_window, |graph, _| graph.execution_run_mode())
        != ExecutionRunMode::Manual
        || controller_loss_reason
            .as_deref()
            .is_none_or(|reason| !reason.contains("controller"))
        || snapshot_after_controller_loss != snapshot_before_controller_loss
    {
        return Err(format!(
            "controller loss did not immediately reset automatic mode without model mutation: mode={:?}, reason={controller_loss_reason:?}",
            item.read_with(panel_window, |graph, _| graph.execution_run_mode())
        ));
    }
    panel_window.run_until_parked();
    if panel_window
        .debug_bounds("SIM-SURFACE-CLEAR-QUEUE-UNAVAILABLE-REASON")
        .is_none()
    {
        return Err("disconnected Clear Queue rendered no typed unavailable reason".to_owned());
    }
    let disconnected_clear_snapshot_before = snapshot_after_controller_loss;
    panel.update_in(panel_window, |panel, window, cx| {
        panel.focus_clear_queue_for_test(window, cx)
    });
    let disconnected_clear_is_focused =
        panel_window.update(|window, cx| panel.read(cx).clear_queue_is_focused_for_test(window));
    if !disconnected_clear_is_focused {
        return Err("disconnected Clear Queue could not receive keyboard focus".to_owned());
    }
    panel_window.simulate_keystrokes("enter");
    panel_window.simulate_keystrokes("space");
    let disconnected_clear_bounds = panel_window
        .debug_bounds("SIM-SURFACE-CLEAR-QUEUE")
        .ok_or_else(|| "disconnected Clear Queue lost its bounds".to_owned())?;
    panel_window.simulate_click(disconnected_clear_bounds.center(), Modifiers::default());
    panel_window.run_until_parked();
    if model
        .read_with(panel_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?
        != disconnected_clear_snapshot_before
    {
        return Err(
            "disconnected Clear Queue keyboard/click activation mutated execution state".to_owned(),
        );
    }
    record_native_behavior(
        &mut native_feature_results,
        "COMFY-QUEUE-110",
        "Clear Queue remained focusable with an explicit reason when empty and disconnected; Enter, Space, and pointer click were all no-ops with byte-for-byte equivalent snapshots",
    )?;

    let (unavailable_graph, unavailable_window) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            graph_fixture(profile_id).expect("valid unavailable-provider graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    unavailable_window.run_until_parked();
    if unavailable_graph.read_with(unavailable_window, |graph, cx| {
        graph.execution_queue_available(cx)
    }) {
        return Err(
            "execution queue remained available with a plan provider but no runtime controller"
                .to_owned(),
        );
    }
    for selector in ["SIM-EXECUTE-BUTTON", "SIM-EXECUTE-UNAVAILABLE-REASON"] {
        if unavailable_window.debug_bounds(selector).is_none() {
            return Err(format!(
                "controller-unavailable Execute did not render `{selector}`"
            ));
        }
    }
    let unavailable_execute_snapshot_before = model
        .read_with(unavailable_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let unavailable_execute_trace_before = unavailable_graph
        .read_with(unavailable_window, |graph, _| {
            graph.shell_dispatch_trace_for_test().to_vec()
        })
        .len();
    unavailable_graph.update_in(unavailable_window, |graph, window, cx| {
        let execute_focus_handle = graph.control_focus_handle("execution:execute-button", cx);
        window.focus(&execute_focus_handle, cx);
    });
    unavailable_window.run_until_parked();
    if unavailable_window
        .debug_bounds("SIM-EXECUTE-OUTPUT-FEEDBACK")
        .is_none()
    {
        return Err("controller-unavailable Execute could not receive focus".to_owned());
    }
    unavailable_window.simulate_keystrokes("enter");
    unavailable_window.simulate_keystrokes("space");
    let unavailable_execute_bounds = unavailable_window
        .debug_bounds("SIM-EXECUTE-BUTTON")
        .ok_or_else(|| "controller-unavailable Execute lost its bounds".to_owned())?;
    unavailable_window.simulate_click(unavailable_execute_bounds.center(), Modifiers::default());
    unavailable_window.run_until_parked();
    let unavailable_execute_snapshot_after = model
        .read_with(unavailable_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let unavailable_execute_trace_after = unavailable_graph
        .read_with(unavailable_window, |graph, _| {
            graph.shell_dispatch_trace_for_test().to_vec()
        })
        .len();
    if unavailable_execute_snapshot_after != unavailable_execute_snapshot_before
        || unavailable_execute_trace_after != unavailable_execute_trace_before
    {
        return Err(
            "controller-unavailable Execute activation dispatched or mutated execution state"
                .to_owned(),
        );
    }
    model.update(unavailable_window, |model, cx| {
        model.register_runtime_controller(Arc::new(AcceptingExecutionActuator), cx);
        model.clear_plan_provider(cx);
    });
    unavailable_window.run_until_parked();
    let providerless_snapshot_before = model
        .read_with(unavailable_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let providerless_outcome = unavailable_graph.update(unavailable_window, |graph, cx| {
        graph.dispatch_shell_command("Sim.QueuePrompt", cx)
    });
    let CommandDispatchOutcome::Rejected { ref error, .. } = providerless_outcome else {
        return Err(format!(
            "execution queue without a plan provider was not rejected: {providerless_outcome:?}"
        ));
    };
    unavailable_window.run_until_parked();
    if !error.contains("plan provider")
        || model
            .read_with(unavailable_window, |model, _| model.snapshot(profile_id))
            .map_err(|error| error.to_string())?
            != providerless_snapshot_before
        || !unavailable_graph
            .read_with(unavailable_window, |graph, _| {
                graph.model().last_error.clone()
            })
            .as_deref()
            .is_some_and(|error| error.contains("plan provider"))
    {
        return Err(
            "providerless queue rejection omitted its reason or mutated execution state".to_owned(),
        );
    }

    let switched_profile_id = profile(0x18ff);
    model
        .update(unavailable_window, |model, cx| {
            model.initialize_profile(
                switched_profile_id,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
                cx,
            )?;
            model.set_active_profile(switched_profile_id, cx)
        })
        .map_err(|error| error.to_string())?;
    unavailable_window.run_until_parked();
    let switched_panel_state = panel.read_with(unavailable_window, |panel, _| {
        panel.surface_state_for_test()
    });
    if switched_panel_state.selected_profile_id != Some(switched_profile_id)
        || switched_panel_state.selected_attempt_id.is_some()
    {
        return Err(format!(
            "execution panel retained stale profile/attempt state after the model switch: profile={:?}, attempt={:?}",
            switched_panel_state.selected_profile_id, switched_panel_state.selected_attempt_id
        ));
    }
    let old_profile_before_switched_action = model
        .read_with(unavailable_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let switched_profile_before_action = model
        .read_with(unavailable_window, |model, _| {
            model.snapshot(switched_profile_id)
        })
        .map_err(|error| error.to_string())?;
    panel.update_in(unavailable_window, |panel, window, cx| {
        panel.handle_surface_action_for_test(ExecutionSurfaceAction::ClearPending, window, cx)
    });
    unavailable_window.run_until_parked();
    let old_profile_after_switched_action = model
        .read_with(unavailable_window, |model, _| model.snapshot(profile_id))
        .map_err(|error| error.to_string())?;
    let switched_profile_after_action = model
        .read_with(unavailable_window, |model, _| {
            model.snapshot(switched_profile_id)
        })
        .map_err(|error| error.to_string())?;
    if old_profile_after_switched_action != old_profile_before_switched_action
        || switched_profile_after_action.revision <= switched_profile_before_action.revision
    {
        return Err(
            "execution panel dispatched through a stale persisted profile after profile switching"
                .to_owned(),
        );
    }

    let accessibility_contracts = [
        (EXECUTION_PANEL_SOURCE, "Role::TabList"),
        (EXECUTION_PANEL_SOURCE, "Role::Tab"),
        (QUEUE_PANEL_SOURCE, "Role::ProgressIndicator"),
        (HISTORY_PANEL_SOURCE, "Role::ListItem"),
        (OUTPUT_VIEW_SOURCE, "aria_numeric_value"),
        (OUTPUT_VIEW_SOURCE, "Role::Alert"),
        (OUTPUT_VIEW_SOURCE, "with_capabilities"),
        (OUTPUT_VIEW_SOURCE, "sim-output-history-skeleton-"),
        (OUTPUT_VIEW_SOURCE, "sim-output-history-active-queue-item"),
        (OUTPUT_VIEW_SOURCE, "OutputViewAction::SelectAttempt"),
        (OUTPUT_VIEW_SOURCE, "OutputViewAction::InterruptAttempt"),
        (EXECUTION_SURFACES_SOURCE, "Role::SearchInput"),
        (EXECUTION_SURFACES_SOURCE, "sim-job-filter-tabs"),
        (EXECUTION_SURFACES_SOURCE, "sim-error-group-list"),
        (EXECUTION_SURFACES_SOURCE, "sim-error-github-"),
        (EXECUTION_SURFACES_SOURCE, "sim-error-copy-"),
        (EXECUTION_SURFACES_SOURCE, "sim-progress-toast-item"),
        (EXECUTION_SURFACES_SOURCE, "aria_numeric_value"),
        (EXECUTION_PANEL_SOURCE, "model.handle_output_operation"),
        (
            EXECUTION_PANEL_SOURCE,
            "ExecutionOutputOperationAction::Recover",
        ),
        (
            EXECUTION_PANEL_SOURCE,
            "ExecutionOutputOperationAction::Remove",
        ),
        (
            EXECUTION_PANEL_SOURCE,
            "is_associated_with_execution(original_attempt_id)",
        ),
        (WORKFLOW_ITEM_SOURCE, "execution_queue_available"),
        (WORKFLOW_ITEM_SOURCE, "plan_provider_available"),
        (
            EXECUTION_PANEL_SOURCE,
            "https://github.com/Comfy-Org/ComfyUI_frontend/issues",
        ),
    ];
    for (source, contract) in accessibility_contracts {
        if !source.contains(contract) {
            return Err(format!("production execution surface lacks `{contract}`"));
        }
    }
    let keyboard_element_ids = [
        "sim-execute-button",
        "sim-copy-execution-error",
        "sim-locate-execution-error",
        "sim-dismiss-execution-error",
    ];
    for element_id in keyboard_element_ids {
        let source_after_id = GRAPH_RENDER_SOURCE
            .split_once(element_id)
            .map(|(_, source)| source)
            .ok_or_else(|| format!("production graph lacks `{element_id}`"))?;
        if !source_after_id
            .lines()
            .take(45)
            .any(|line| line.contains(".on_key_down"))
        {
            return Err(format!(
                "production graph `{element_id}` lacks keyboard activation"
            ));
        }
    }
    if !GRAPH_RENDER_SOURCE.contains("sim-node-execution-")
        || !GRAPH_RENDER_SOURCE.contains("Node {} execution attempt {}")
    {
        return Err("graph node execution status projection is missing".to_owned());
    }
    if !ZED_SOURCE.contains("sim_ui::ExecutionPanel::load")
        || !ZED_SOURCE.contains("workspace.add_panel(panel, window, cx)")
    {
        return Err("Zed does not load and register the production execution panel".to_owned());
    }
    let expected_runtime_components = execution_component_ids()
        .into_iter()
        .filter(|feature_id| *feature_id != "COMFY-FRONTEND-SURFACE-6F5EE356A779")
        .collect::<BTreeSet<_>>();
    let rendered_runtime_components = rendered_component_evidence
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    if rendered_runtime_components != expected_runtime_components {
        return Err(format!(
            "runtime component coverage is incomplete: rendered={rendered_runtime_components:?}, expected={expected_runtime_components:?}"
        ));
    }

    Ok(json!({
        "name": "registered-controller-gpui-panel-graph-focus-progress-error-copy-and-navigation",
        "passed": true,
        "rendered_selectors": [
            "SIM-EXECUTION-ACTIONBAR",
            "SIM-EXECUTE-BUTTON",
            "SIM-QUEUE-OVERLAY",
            "SIM-EXECUTION-ERROR-OVERLAY",
            "SIM-NODE-error-node"
        ],
        "graph_focus": true,
        "panel_focus": true,
        "numeric_progress_accessibility": true,
        "structured_error_clipboard_digest": digest(&clipboard),
        "selection_and_viewport_restored": true,
        "panel_persistence_round_trip": true,
        "execution_command_outcomes": command_outcomes
            .iter()
            .map(|(command_id, executed, dispatch)| json!({
                "command_id": command_id,
                "executed": executed,
                "dispatch": dispatch,
            }))
            .collect::<Vec<_>>(),
        "execution_commands_exercised": command_outcomes.len(),
        "qpov2_hidden_surfaces_verified": true,
        "d23_unassociated_graph_isolation": true,
        "d23_stale_association_isolation": true,
        "d23_retry_reassociated_only_owner": true,
        "native_plan_provider_capability_gate": true,
        "unavailable_execute_visible_non_operable": true,
        "unavailable_automatic_mode_remains_manual": true,
        "unavailable_queue_state_unchanged": true,
        "typed_panel_output_operations": panel_operations.len(),
        "keyboard_activation_paths": keyboard_element_ids.len(),
        "rendered_lifecycle_states": projected_states.len(),
        "rendered_snapshot_statuses": 5,
        "rendered_data_sources": 3,
        "rendered_provider_states": 8,
        "rendered_unknown_states": 1,
        "zed_production_panel_load_registered": true,
        "runtime_component_evidence": rendered_component_evidence
            .values()
            .cloned()
            .collect::<Vec<_>>(),
        "runtime_components_rendered": rendered_component_evidence.len(),
        "native_feature_results": native_feature_results.values().cloned().collect::<Vec<_>>(),
        "queue_notification_fifo_and_fake_time": true,
        "model_driven_queue_notification_lifecycle": true,
        "progress_toast_running_terminal_and_fake_time": true,
        "output_skeleton_and_outside_scroll_active_item": true,
        "registered_controller_output_interrupt_exact_attempt": true,
        "output_auto_follow_break_and_reset": true,
        "clear_history_cancel_close_confirm": true,
        "sticky_subgraph_progress_invalidation": true,
        "parent_subgraph_failure_ring": true,
    }))
}

fn write_artifact(cases: Vec<Value>) -> Result<(), Box<dyn Error>> {
    if cases.iter().any(|case| case["passed"] != true) {
        return Err("VAL-GPUI-005 contains a failing case".into());
    }
    let expected_native_feature_ids = (1..=119)
        .map(|number| format!("COMFY-QUEUE-{number:03}"))
        .filter(|feature_id| {
            execution_feature_disposition(feature_id) == Some(ExecutionFeatureDisposition::Native)
        })
        .collect::<BTreeSet<_>>();
    let mut native_feature_results = BTreeMap::<String, Value>::new();
    for result in cases
        .iter()
        .filter_map(|case| case.get("native_feature_results").and_then(Value::as_array))
        .flatten()
    {
        let feature_id = result
            .get("feature_id")
            .and_then(Value::as_str)
            .ok_or("native feature result has no feature_id")?;
        if result.get("passed") != Some(&Value::Bool(true))
            || result
                .get("runtime_assertion")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Err(format!(
                "native feature result `{feature_id}` lacks passing runtime evidence"
            )
            .into());
        }
        if native_feature_results
            .insert(feature_id.to_owned(), result.clone())
            .is_some()
        {
            return Err(
                format!("native feature result `{feature_id}` appears more than once").into(),
            );
        }
    }
    let actual_native_feature_ids = native_feature_results
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual_native_feature_ids != expected_native_feature_ids {
        return Err(format!(
            "native runtime evidence IDs differ from the catalog disposition: actual={actual_native_feature_ids:?}, expected={expected_native_feature_ids:?}"
        )
        .into());
    }
    let artifact = json!({
        "validation_id": "VAL-GPUI-005",
        "environment": {
            "backend": "gpui-test",
            "platform": "mock-window",
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "feature": "test-support",
            "scheduler_seed": 18005,
            "iterations": std::env::var("ITERATIONS").unwrap_or_else(|_| "1".to_owned()),
        },
        "fixture_digests": {
            "queue_features": digest(QUEUE_CATALOG),
            "commands": digest(COMMAND_CATALOG),
            "menus": digest(MENU_CATALOG),
            "component_surfaces": digest(COMPONENT_CATALOG),
            "parity_matrix": digest(PARITY_MATRIX),
            "execution_panel": digest(EXECUTION_PANEL_SOURCE),
            "execution_model": digest(EXECUTION_MODEL_SOURCE),
            "execution_catalog": digest(EXECUTION_CATALOG_SOURCE),
            "execution_surfaces": digest(EXECUTION_SURFACES_SOURCE),
            "execution_tests": digest(EXECUTION_TEST_SOURCE),
            "actions": digest(ACTIONS_SOURCE),
            "shell": digest(SHELL_SOURCE),
            "queue_panel": digest(QUEUE_PANEL_SOURCE),
            "history_panel": digest(HISTORY_PANEL_SOURCE),
            "output_view": digest(OUTPUT_VIEW_SOURCE),
            "graph_render": digest(GRAPH_RENDER_SOURCE),
            "workflow_item": digest(WORKFLOW_ITEM_SOURCE),
            "zed": digest(ZED_SOURCE),
            "runtime_execution_presentation": digest(RUNTIME_PRESENTATION_SOURCE),
            "runtime_queue_history": digest(RUNTIME_QUEUE_HISTORY_SOURCE),
            "runtime_persistence": digest(RUNTIME_PERSISTENCE_SOURCE),
            "runtime_crate_root": digest(RUNTIME_CRATE_SOURCE),
            "default_sim_keymap": digest(DEFAULT_SIM_KEYMAP),
            "execution_ledger_generator": digest(EXECUTION_LEDGER_GENERATOR),
        },
        "catalog_counts": {
            "queue_features": 119,
            "execution_commands": 9,
            "job_and_run_menus": 17,
            "execution_components": 25,
        },
        "native_feature_results": native_feature_results.values().cloned().collect::<Vec<_>>(),
        "cases": cases,
        "skipped": [],
    });
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"))
        .join("comfy-parity");
    fs::create_dir_all(&target)?;
    fs::write(
        target.join("val-gpui-005.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}

#[gpui::test(seed = 18005)]
fn val_gpui_005(cx: &mut TestAppContext) {
    let cases = vec![
        {
            eprintln!("VAL-GPUI-005 phase: catalog");
            catalog_case().expect("reconcile exact execution catalogs")
        },
        {
            eprintln!("VAL-GPUI-005 phase: acknowledgement-and-projection");
            acknowledgement_and_projection_case()
                .expect("validate acknowledged lifecycle and retry projections")
        },
        {
            eprintln!("VAL-GPUI-005 phase: output-availability");
            output_availability_case(cx).expect("validate output recovery and reference actions")
        },
        {
            eprintln!("VAL-GPUI-005 phase: event-reduction-10001");
            event_reduction_case(cx).expect("validate large canonical event reduction")
        },
        {
            eprintln!("VAL-GPUI-005 phase: production-controller-boundary");
            production_controller_boundary_case(cx)
                .expect("validate production fail-closed controller boundary")
        },
        {
            eprintln!("VAL-GPUI-005 phase: parent-subgraph-error-projection");
            parent_subgraph_error_projection_case()
                .expect("validate parent subgraph failure-ring projection")
        },
        {
            eprintln!("VAL-GPUI-005 phase: gpui-interaction");
            gpui_interaction_case(cx)
                .expect("validate GPUI execution surfaces with a registered test controller")
        },
        {
            eprintln!("VAL-GPUI-005 phase: output-operation-cancellation");
            output_operation_cancellation_case(cx)
                .expect("validate profile-bound output-operation cancellation")
        },
    ];
    eprintln!("VAL-GPUI-005 phase: write-artifact");
    write_artifact(cases).expect("write VAL-GPUI-005 artifact");
}

#[gpui::test(seed = 18018)]
fn clear_history_confirmation_does_not_follow_profile_switch(cx: &mut TestAppContext) {
    let (
        model,
        _operation_handler,
        originating_profile_id,
        switched_profile_id,
        _attempt_id,
        _output_id,
    ) = profile_bound_confirmation_fixture(cx, 0x1818)
        .expect("create clear-history profile-binding fixture");
    let (panel, window) = cx.add_window_view(|_, cx| ExecutionPanel::test_new(model.clone(), cx));
    let originating_snapshot_before = model
        .read_with(window, |model, _| model.snapshot(originating_profile_id))
        .expect("read originating profile before confirmation");
    let switched_snapshot_before = model
        .read_with(window, |model, _| model.snapshot(switched_profile_id))
        .expect("read switched profile before confirmation");

    panel.update_in(window, |panel, window, cx| {
        panel.confirm_clear_history_for_test(window, cx)
    });
    assert!(window.has_pending_prompt());
    model
        .update(window, |model, cx| {
            model.set_active_profile(switched_profile_id, cx)
        })
        .expect("switch execution profile while clear-history prompt is open");
    window.run_until_parked();
    window.simulate_prompt_answer("Clear History");
    window.run_until_parked();

    assert_eq!(
        model
            .read_with(window, |model, _| model.snapshot(originating_profile_id))
            .expect("read originating profile after confirmation"),
        originating_snapshot_before,
        "clear-history confirmation mutated its originating profile after a profile switch"
    );
    assert_eq!(
        model
            .read_with(window, |model, _| model.snapshot(switched_profile_id))
            .expect("read switched profile after confirmation"),
        switched_snapshot_before,
        "clear-history confirmation followed the newly active profile"
    );
    assert!(
        panel
            .read_with(window, |panel, _| {
                panel.status_message_for_test().map(str::to_owned)
            })
            .is_some_and(|message| message.contains("active execution profile changed")),
        "profile-bound cancellation did not remain visible"
    );
}

#[gpui::test(seed = 18019)]
fn output_removal_confirmation_does_not_follow_restored_profile(cx: &mut TestAppContext) {
    let (
        model,
        operation_handler,
        originating_profile_id,
        switched_profile_id,
        attempt_id,
        output_id,
    ) = profile_bound_confirmation_fixture(cx, 0x1819)
        .expect("create output-removal profile-binding fixture");
    let (panel, window) = cx.add_window_view(|_, cx| ExecutionPanel::test_new(model.clone(), cx));
    let originating_snapshot_before = model
        .read_with(window, |model, _| model.snapshot(originating_profile_id))
        .expect("read originating profile before confirmation");
    let switched_snapshot_before = model
        .read_with(window, |model, _| model.snapshot(switched_profile_id))
        .expect("read switched profile before confirmation");

    panel.update_in(window, |panel, window, cx| {
        panel.set_test_state(ExecutionPanelTab::Output, Some(attempt_id), "", "", 0, cx);
        panel.handle_output_action_for_test(OutputViewAction::RemoveOutput(output_id), window, cx);
    });
    assert!(window.has_pending_prompt());
    model
        .update(window, |model, cx| {
            model.set_active_profile(switched_profile_id, cx)
        })
        .expect("switch away while output-removal prompt is open");
    window.run_until_parked();
    model
        .update(window, |model, cx| {
            model.set_active_profile(originating_profile_id, cx)
        })
        .expect("restore originating profile while output-removal prompt is open");
    window.run_until_parked();
    window.simulate_prompt_answer("Remove Output");
    window.run_until_parked();

    assert_eq!(
        model
            .read_with(window, |model, _| model.snapshot(originating_profile_id))
            .expect("read originating profile after confirmation"),
        originating_snapshot_before,
        "output-removal confirmation mutated the restored originating profile"
    );
    assert_eq!(
        model
            .read_with(window, |model, _| model.snapshot(switched_profile_id))
            .expect("read switched profile after confirmation"),
        switched_snapshot_before,
        "output-removal confirmation mutated the intermediate profile"
    );
    assert!(
        operation_handler
            .actions
            .lock()
            .expect("operation recorder lock")
            .is_empty(),
        "invalidated output-removal confirmation reached the operation owner"
    );
    assert!(
        panel
            .read_with(window, |panel, _| {
                panel.status_message_for_test().map(str::to_owned)
            })
            .is_some_and(|message| message.contains("active execution profile changed")),
        "profile-bound cancellation did not remain visible"
    );
}

fn output_operation_cancellation_case(cx: &mut TestAppContext) -> Result<Value, String> {
    let (
        model,
        _operation_handler,
        originating_profile_id,
        switched_profile_id,
        attempt_id,
        output_id,
    ) = profile_bound_confirmation_fixture(cx, 0x1820)
        .map_err(|error| format!("create output-operation cancellation fixture: {error}"))?;
    let cancellation_handler = Arc::new(CancellationObservingOperationHandler::default());
    model
        .update(cx, |model, cx| {
            model.register_output_operation_handler(cancellation_handler.clone(), cx);
            model.handle_output_operation(
                originating_profile_id,
                attempt_id,
                output_id,
                ExecutionOutputOperationAction::Remove,
                cx,
            )
        })
        .map_err(|error| format!("start output operation for the originating profile: {error}"))?;

    model
        .update(cx, |model, cx| {
            model.set_active_profile(switched_profile_id, cx)
        })
        .map_err(|error| {
            format!("switch active profile while the output operation is pending: {error}")
        })?;
    cx.run_until_parked();

    let diagnostic = model
        .read_with(cx, |model, _| model.diagnostics().next_back().cloned())
        .ok_or_else(|| "cancelled operation did not emit a visible diagnostic".to_owned())?;
    let observation = cancellation_handler.observation()?;
    if !observation.started || !observation.cancelled {
        return Err(format!(
            "output operation did not observe canonical cancellation: started={}, cancelled={}",
            observation.started, observation.cancelled
        ));
    }
    if diagnostic.profile_id != Some(originating_profile_id)
        || diagnostic.attempt_id != Some(attempt_id)
        || !diagnostic.message.contains("output_operation_cancelled")
    {
        return Err(format!(
            "cancelled operation diagnostic lost its originating scope: {diagnostic:?}"
        ));
    }
    if model.read_with(cx, |model, _| model.active_profile_id()) != Some(switched_profile_id) {
        return Err("cancelled output operation changed the active profile".to_owned());
    }
    Ok(json!({
        "name": "profile-switch cancels the originating output operation",
        "passed": true,
        "canonical_token_observed": true,
        "diagnostic_profile_id": originating_profile_id.0.to_string(),
        "diagnostic_attempt_id": attempt_id.0.to_string(),
    }))
}

#[gpui::test(seed = 18020)]
fn active_profile_switch_cancels_the_originating_output_operation(cx: &mut TestAppContext) {
    output_operation_cancellation_case(cx)
        .expect("validate profile-bound output-operation cancellation");
}

#[gpui::test]
async fn execution_panel_first_workspace_toggle_opens_before_taking_focus(cx: &mut TestAppContext) {
    use gpui::UpdateGlobal as _;

    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
        settings::SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.workspace.close_panel_on_toggle = Some(true);
            });
        });
    });
    let project = Project::test(FakeFs::new(cx.executor()), [], cx).await;
    let (multi_workspace, workspace_window) =
        cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
    let workspace = multi_workspace.read_with(workspace_window, |multi_workspace, _| {
        multi_workspace.workspace().clone()
    });
    let panel = workspace.update_in(workspace_window, |workspace, window, cx| {
        let workspace_handle = cx.entity().downgrade();
        let graph = cx.new(|cx| {
            GraphWorkspaceItem::new(
                crate::graph_tests::fixture_model().expect("editable graph"),
                workspace_handle,
                cx,
            )
        });
        workspace.add_item_to_active_pane(Box::new(graph), None, true, window, cx);
        let model = execution_ui_model(cx).expect("canonical execution model");
        let panel = cx.new(|cx| ExecutionPanel::test_new(model, cx));
        workspace.add_panel(panel.clone(), window, cx);
        workspace.focus_center_pane(window, cx);
        assert!(!workspace.bottom_dock().read(cx).is_open());
        panel
    });
    workspace_window.run_until_parked();
    for expected_open in [true, false, true] {
        workspace_window.dispatch_action(ToggleExecutionPanel);
        workspace_window.run_until_parked();
        workspace.update_in(workspace_window, |workspace, window, cx| {
            assert_eq!(
                workspace.bottom_dock().read(cx).is_open(),
                expected_open,
                "the canonical toggle must open on its first activation, close, and reopen"
            );
            assert_eq!(
                panel.focus_handle(cx).contains_focused(window, cx),
                expected_open,
                "Workspace must retain ownership of panel focus after activation"
            );
        });
    }
}

#[gpui::test(seed = 16017)]
async fn execution_menu_actions_have_one_context_owner_and_visible_missing_target(
    cx: &mut TestAppContext,
) {
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
    });
    let file_system = FakeFs::new(cx.executor());
    let project = Project::test(file_system, [], cx).await;
    let (multi_workspace, workspace_window) =
        cx.add_window_view(|window, cx| MultiWorkspace::test_new(project, window, cx));
    let workspace = multi_workspace.read_with(workspace_window, |multi_workspace, _| {
        multi_workspace.workspace().clone()
    });
    workspace.update_in(workspace_window, |workspace, window, cx| {
        workspace.focus_center_pane(window, cx);
        cx.notify();
    });
    workspace_window.run_until_parked();

    for graph_owned_action in [
        "ExecutionRunManual",
        "ExecutionRunOnChange",
        "ExecutionRunInstantIdle",
    ] {
        assert!(!EXECUTION_PANEL_SOURCE.contains(graph_owned_action));
    }
    let direct_native_actions = menu_registry()
        .map(|registration| registration.expect("valid generated menu registration"))
        .filter(|registration| {
            registration.command_id.is_none()
                && registration.status == CommandNativeStatus::Executable
                && registration.native_action.is_some()
        })
        .map(|registration| registration.native_action.expect("checked native action"))
        .collect::<Vec<_>>();
    let graph_run_actions = [
        NativeAction::ExecutionRunManual,
        NativeAction::ExecutionRunOnChange,
    ];
    let workspace_actions = direct_native_actions
        .iter()
        .copied()
        .filter(|native_action| !graph_run_actions.contains(native_action))
        .collect::<Vec<_>>();
    assert_eq!(direct_native_actions.len(), 9);
    assert_eq!(
        direct_native_actions
            .iter()
            .map(|native_action| native_action.name())
            .collect::<BTreeSet<_>>()
            .len(),
        9
    );
    assert_eq!(workspace_actions.len(), 7);

    let unavailable_notification = workspace::notifications::NotificationId::named(
        EXECUTION_ACTION_UNAVAILABLE_NOTIFICATION_ID.into(),
    );
    for native_action in workspace_actions {
        workspace.update(workspace_window, |workspace, cx| {
            workspace.dismiss_notification(&unavailable_notification, cx);
        });
        assert!(
            workspace
                .read_with(workspace_window, |workspace, _| workspace
                    .notification_ids())
                .is_empty()
        );
        workspace_window.update(|window, cx| {
            let action = cx
                .build_action(native_action.name(), None)
                .expect("registered direct native menu action");
            assert!(window.is_action_available(action.as_ref(), cx));
            window.dispatch_action(action, cx);
        });
        workspace_window.run_until_parked();
        let notification_ids = workspace.read_with(workspace_window, |workspace, _| {
            workspace.notification_ids()
        });
        assert_eq!(
            notification_ids.as_slice(),
            std::slice::from_ref(&unavailable_notification),
            "{} did not use the canonical visible unavailable adapter",
            native_action.name(),
        );
    }

    let graph_profile = profile(16);
    cx.update(|cx| {
        let mut service =
            ExecutionPresentationService::new(16).expect("valid scoped-action presentation");
        service
            .initialize_profile(
                graph_profile,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .expect("initialize scoped-action profile");
        let model =
            cx.new(|_| ExecutionUiModel::new(service, Arc::new(AcceptingExecutionActuator)));
        model.update(cx, |model, cx| {
            model
                .set_active_profile(graph_profile, cx)
                .expect("activate scoped-action profile");
            model.register_plan_provider(
                Arc::new(DeterministicPlanProvider {
                    plan: plan(prompt(16)),
                }),
                cx,
            );
        });
        cx.set_global(GlobalExecutionUiModel(model));
    });
    let (graph, graph_window) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            graph_fixture(graph_profile).expect("valid scoped-action graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    graph.update_in(graph_window, |graph, window, cx| {
        graph.focus_graph(window, cx);
    });
    graph_window.run_until_parked();
    for (native_action, expected_mode) in [
        (
            NativeAction::ExecutionRunOnChange,
            ExecutionRunMode::OnChange,
        ),
        (NativeAction::ExecutionRunManual, ExecutionRunMode::Manual),
    ] {
        graph_window.update(|window, cx| {
            let action = cx
                .build_action(native_action.name(), None)
                .expect("registered graph run-mode action");
            assert!(window.is_action_available(action.as_ref(), cx));
            window.dispatch_action(action, cx);
        });
        graph_window.run_until_parked();
        assert_eq!(
            graph.read_with(graph_window, |graph, _| graph.execution_run_mode()),
            expected_mode,
            "{} did not reach the focused graph owner",
            native_action.name(),
        );
    }
}

#[gpui::test(seed = 19019)]
async fn native_image_ui_queues_projects_outputs_and_rejects_late_cancelled_output(
    cx: &mut TestAppContext,
) {
    let event_bus = ExecutionEventBus::new(64).expect("valid native UI event bus");
    let controller = Arc::new(NativeUiControllerProbe::new());
    let plan_provider = Arc::new(NativeGeneratedPlanProviderProbe::default());
    let reference_handler = Arc::new(RecordingReferenceHandler::default());
    let model = cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
        let mut service =
            ExecutionPresentationService::new(32).expect("valid native UI history capacity");
        service
            .initialize_profile(
                LOCAL_EXECUTION_PROFILE_ID,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .expect("initialize native UI profile");
        let model = cx.new(|_| ExecutionUiModel::new(service, controller.clone()));
        model.update(cx, |model, cx| {
            model
                .set_active_profile(LOCAL_EXECUTION_PROFILE_ID, cx)
                .expect("activate native UI profile");
            model.register_plan_provider(plan_provider.clone(), cx);
            model.register_output_reference_handler(reference_handler.clone(), cx);
            assert!(model.attach_event_bus(event_bus.clone(), cx));
        });
        cx.set_global(GlobalExecutionUiModel(model.clone()));
        model
    });

    let (item, graph_window) = cx.add_window_view(|_, cx| {
        GraphWorkspaceItem::new(
            native_image_graph_fixture(LOCAL_EXECUTION_PROFILE_ID)
                .expect("create native image graph fixture"),
            WeakEntity::new_invalid(),
            cx,
        )
    });
    let execute_bounds = graph_window
        .debug_bounds("SIM-EXECUTE-BUTTON")
        .expect("native image graph Execute control");
    item.update_in(graph_window, |item, window, cx| {
        item.focus_graph(window, cx);
        let execute_focus_handle = item.control_focus_handle("execution:execute-button", cx);
        window.focus(&execute_focus_handle, cx);
        assert!(execute_focus_handle.is_focused(window));
    });
    graph_window.simulate_keystrokes("enter");
    graph_window.run_until_parked();
    assert!(execute_bounds.size.width > px(0.0));

    let first_attempt = item
        .read_with(graph_window, |item, cx| {
            item.active_execution_presentation(cx).ok_or_else(|| {
                format!(
                    "keyboard queue associated no native image attempt: announcement={:?}, last_error={:?}, dispatch_trace={:?}",
                    item.model.announcement,
                    item.model.last_error,
                    item.shell_dispatch_trace_for_test()
                )
            })
        })
        .expect("keyboard queue associated the native image attempt");
    let first_plan = match controller.commands().first().map(|command| &command.kind) {
        Some(ExecutionControlCommandKind::Queue { plan, .. }) => plan.clone(),
        other => panic!("keyboard Execute did not submit a queue command: {other:?}"),
    };
    assert_eq!(first_plan.nodes.len(), 5);
    assert_eq!(
        first_plan
            .topological_order
            .iter()
            .filter_map(|node_id| first_plan.nodes.get(node_id))
            .map(|node| node.class_type.as_str())
            .collect::<Vec<_>>(),
        vec![
            "LoadImage",
            "ImageScale",
            "ImageInvert",
            "PreviewImage",
            "SaveImage"
        ]
    );
    assert_eq!(
        match plan_provider.compiled_plans.lock() {
            Ok(compiled_plans) => compiled_plans.len(),
            Err(error) => error.into_inner().len(),
        },
        1
    );

    let first_output_id = identifier(0x1901);
    let first_reference = "zed-asset://output/task19/native-image.png";
    graph_window
        .background_executor
        .timer(Duration::from_millis(10))
        .await;
    for attempt_event in [
        event(
            LOCAL_EXECUTION_PROFILE_ID,
            first_attempt.prompt_id,
            first_attempt.attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ),
        event(
            LOCAL_EXECUTION_PROFILE_ID,
            first_attempt.prompt_id,
            first_attempt.attempt_id,
            1,
            Some(NodeId("3".to_owned())),
            AttemptEventKind::Progress {
                completed: 3,
                total: 5,
            },
        ),
        event(
            LOCAL_EXECUTION_PROFILE_ID,
            first_attempt.prompt_id,
            first_attempt.attempt_id,
            2,
            Some(NodeId("4".to_owned())),
            AttemptEventKind::Preview {
                preview: sim_runtime::ExecutionPreview {
                    preview_id: identifier(0x1902),
                    node_id: NodeId("4".to_owned()),
                    revision: 1,
                    frame_index: Some(0),
                    output_index: Some(0),
                    media_kind: OutputMediaKind::Image,
                    media_type: "image/png".to_owned(),
                    width: Some(4),
                    height: Some(2),
                    encoded_bytes: vec![137, 80, 78, 71],
                },
            },
        ),
        event(
            LOCAL_EXECUTION_PROFILE_ID,
            first_attempt.prompt_id,
            first_attempt.attempt_id,
            3,
            Some(NodeId("5".to_owned())),
            AttemptEventKind::OutputAvailable {
                output: native_image_output(
                    LOCAL_EXECUTION_PROFILE_ID,
                    first_attempt.prompt_id,
                    first_attempt.attempt_id,
                    first_output_id,
                    "native-image.png",
                    first_reference,
                ),
            },
        ),
    ] {
        smol::block_on(
            model
                .read_with(graph_window, |model, _| model.shared_service())
                .apply_event_durable(attempt_event.clone()),
        )
        .expect("apply canonical native UI attempt event");
        event_bus
            .publish(attempt_event)
            .expect("publish native UI attempt event");
    }
    graph_window.run_until_parked();
    let running_projection = model
        .read_with(graph_window, |model, _| {
            model.attempt(LOCAL_EXECUTION_PROFILE_ID, first_attempt.attempt_id)
        })
        .expect("read running native UI projection")
        .expect("running native UI attempt");
    assert_eq!(running_projection.state, AttemptState::Running);
    assert!(matches!(
        running_projection.progress,
        Some(sim_runtime::NodeProgress {
            completed: 3,
            total: 5,
            ..
        })
    ));
    assert!(running_projection.preview.is_some());
    assert_eq!(running_projection.outputs.len(), 1);

    let (panel, panel_window) =
        graph_window.add_window_view(|_, cx| ExecutionPanel::test_new(model.clone(), cx));
    let inspect_attempt = panel_window
        .debug_bounds("SIM-QUEUE-ACTIVE-INSPECT")
        .expect("active native attempt Inspect control");
    panel_window.simulate_click(inspect_attempt.center(), Modifiers::default());
    panel_window.run_until_parked();
    assert_eq!(
        panel.read_with(panel_window, |panel, _| {
            panel.surface_state_for_test().selected_tab
        }),
        ExecutionPanelTab::Output
    );
    let inspect_output = panel_window
        .debug_bounds("SIM-OUTPUT-INSPECT")
        .expect("native image output Inspect control");
    panel_window.simulate_click(inspect_output.center(), Modifiers::default());
    panel_window.run_until_parked();
    let view_output = panel_window
        .debug_bounds("SIM-OUTPUT-VIEW")
        .expect("native image View control");
    panel.update_in(panel_window, |panel, window, cx| {
        panel.focus_selected_output_view_for_test(window, cx)
    });
    panel_window.run_until_parked();
    assert!(panel_window.update(|window, cx| {
        panel
            .read(cx)
            .selected_output_view_is_focused_for_test(window)
    }));
    panel_window.simulate_keystrokes("enter");
    panel_window.run_until_parked();
    let keyboard_action_count = match reference_handler.actions.lock() {
        Ok(actions) => actions.len(),
        Err(error) => error.into_inner().len(),
    };
    assert_eq!(keyboard_action_count, 1);
    panel_window.simulate_click(view_output.center(), Modifiers::default());
    panel_window.run_until_parked();
    let reference_actions = match reference_handler.actions.lock() {
        Ok(actions) => actions.clone(),
        Err(error) => error.into_inner().clone(),
    };
    assert_eq!(reference_actions.len(), 2);
    assert!(
        reference_actions
            .iter()
            .all(|(profile_id, action, reference)| {
                *profile_id == LOCAL_EXECUTION_PROFILE_ID
                    && *action == ExecutionOutputReferenceAction::View
                    && reference == first_reference
            })
    );

    let succeeded = event(
        LOCAL_EXECUTION_PROFILE_ID,
        first_attempt.prompt_id,
        first_attempt.attempt_id,
        4,
        None,
        AttemptEventKind::Succeeded,
    );
    smol::block_on(
        model
            .read_with(panel_window, |model, _| model.shared_service())
            .apply_event_durable(succeeded.clone()),
    )
    .expect("apply canonical native UI success");
    event_bus
        .publish(succeeded)
        .expect("publish native UI success");
    panel_window.run_until_parked();

    let second_queue = item.update(panel_window, |item, cx| {
        item.dispatch_shell_command("Sim.QueuePrompt", cx)
    });
    assert!(
        second_queue.is_executed(),
        "second native image queue failed: {second_queue:?}"
    );
    panel_window.run_until_parked();
    let second_attempt = item
        .read_with(panel_window, |item, cx| {
            item.active_execution_presentation(cx)
        })
        .expect("second native image attempt association");
    assert_ne!(second_attempt.attempt_id, first_attempt.attempt_id);
    for attempt_event in [
        event(
            LOCAL_EXECUTION_PROFILE_ID,
            second_attempt.prompt_id,
            second_attempt.attempt_id,
            0,
            None,
            AttemptEventKind::Started,
        ),
        event(
            LOCAL_EXECUTION_PROFILE_ID,
            second_attempt.prompt_id,
            second_attempt.attempt_id,
            1,
            Some(NodeId("2".to_owned())),
            AttemptEventKind::Progress {
                completed: 2,
                total: 5,
            },
        ),
    ] {
        smol::block_on(
            model
                .read_with(panel_window, |model, _| model.shared_service())
                .apply_event_durable(attempt_event.clone()),
        )
        .expect("apply canonical second native UI event");
        event_bus
            .publish(attempt_event)
            .expect("publish second native UI event");
    }
    panel_window.run_until_parked();
    let interrupt = item.update(panel_window, |item, cx| {
        item.dispatch_shell_command("Sim.Interrupt", cx)
    });
    assert!(
        interrupt.is_executed(),
        "native image interrupt failed: {interrupt:?}"
    );
    panel_window.run_until_parked();
    assert_eq!(
        model
            .read_with(panel_window, |model, _| {
                model.attempt(LOCAL_EXECUTION_PROFILE_ID, second_attempt.attempt_id)
            })
            .expect("read cancelling native UI projection")
            .expect("cancelling native UI attempt")
            .state,
        AttemptState::Cancelling
    );
    assert!(matches!(
        controller.commands().last().map(|command| &command.kind),
        Some(ExecutionControlCommandKind::Interrupt { attempt_id, .. })
            if *attempt_id == second_attempt.attempt_id
    ));

    panel_window
        .background_executor
        .timer(Duration::from_millis(10))
        .await;
    let cancelled_event = event(
        LOCAL_EXECUTION_PROFILE_ID,
        second_attempt.prompt_id,
        second_attempt.attempt_id,
        3,
        None,
        AttemptEventKind::Cancelled,
    );
    smol::block_on(
        model
            .read_with(panel_window, |model, _| model.shared_service())
            .apply_event_durable(cancelled_event.clone()),
    )
    .expect("apply canonical native UI cancellation");
    event_bus
        .publish(cancelled_event)
        .expect("publish native UI cancellation");
    panel_window.run_until_parked();
    let cancelled = model
        .read_with(panel_window, |model, _| {
            model.attempt(LOCAL_EXECUTION_PROFILE_ID, second_attempt.attempt_id)
        })
        .expect("read cancelled native UI projection")
        .expect("cancelled native UI attempt");
    assert_eq!(cancelled.state, AttemptState::Cancelled);
    assert!(cancelled.outputs.is_empty());

    let late_output_id = identifier(0x1903);
    panel_window
        .background_executor
        .timer(Duration::from_millis(10))
        .await;
    event_bus
        .publish(event(
            LOCAL_EXECUTION_PROFILE_ID,
            second_attempt.prompt_id,
            second_attempt.attempt_id,
            4,
            Some(NodeId("5".to_owned())),
            AttemptEventKind::OutputAvailable {
                output: native_image_output(
                    LOCAL_EXECUTION_PROFILE_ID,
                    second_attempt.prompt_id,
                    second_attempt.attempt_id,
                    late_output_id,
                    "late-native-image.png",
                    "zed-asset://output/task19/late-native-image.png",
                ),
            },
        ))
        .expect("publish delayed post-cancellation output");
    panel_window.run_until_parked();
    let (cancelled_after_late_output, terminal_diagnostic) =
        model.read_with(panel_window, |model, _| {
            (
                model
                    .attempt(LOCAL_EXECUTION_PROFILE_ID, second_attempt.attempt_id)
                    .expect("read post-terminal native UI projection")
                    .expect("post-terminal native UI attempt"),
                model.diagnostics().next_back().cloned(),
            )
        });
    assert_eq!(cancelled_after_late_output.state, AttemptState::Cancelled);
    assert!(cancelled_after_late_output.outputs.is_empty());
    assert!(matches!(
        terminal_diagnostic,
        Some(ExecutionDiagnostic {
            kind: ExecutionDiagnosticKind::Terminal,
            attempt_id: Some(attempt_id),
            ..
        }) if attempt_id == second_attempt.attempt_id
    ));
}

fn native_preview_test_payload() -> Result<sim_runtime::ExecutionPreview, Box<dyn Error>> {
    let pixels = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let encoded_bytes = sim_media::encode_png_frame(
        &pixels,
        1,
        1,
        2,
        3,
        0,
        &BTreeMap::new(),
        crate::output_view::native_preview_limits(),
    )?;
    Ok(sim_runtime::ExecutionPreview {
        preview_id: identifier(0x181001),
        node_id: NodeId("image".to_owned()),
        revision: 1,
        frame_index: Some(0),
        output_index: Some(0),
        media_kind: OutputMediaKind::Image,
        media_type: "image/png".to_owned(),
        width: Some(2),
        height: Some(1),
        encoded_bytes,
    })
}

struct NativePreviewProbe {
    scope: crate::output_view::NativePreviewScope,
    preview: Option<sim_runtime::ExecutionPreview>,
    state: Option<WeakEntity<crate::output_view::NativePreviewState>>,
}

impl Render for NativePreviewProbe {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(preview) = self.preview.as_ref() else {
            return div().size_full().into_any_element();
        };
        let state = crate::output_view::native_preview_state(
            self.scope.clone(),
            Arc::new(preview.clone()),
            window,
            cx,
        );
        self.state = Some(state.downgrade());
        div()
            .size_full()
            .child(crate::output_view::render_native_preview(
                self.scope.profile_id,
                self.scope.prompt_id,
                self.scope.attempt_id,
                preview.clone(),
                window,
                cx,
            ))
            .into_any_element()
    }
}

#[gpui::test]
async fn task18_native_preview_pixels_cache_replacement_and_release(cx: &mut TestAppContext) {
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
    });
    let preview = native_preview_test_payload().expect("encode genuine two-pixel PNG");
    let scope = crate::output_view::NativePreviewScope {
        profile_id: ProfileId(identifier(0x181002)),
        prompt_id: PromptId(identifier(0x181003)),
        attempt_id: AttemptId(identifier(0x181004)),
    };
    let (view, window) = cx.add_window_view(|_, _| NativePreviewProbe {
        scope,
        preview: Some(preview),
        state: None,
    });
    window.run_until_parked();
    let state = view
        .read_with(window, |view, _| view.state.clone())
        .expect("preview state rendered");
    let (image, first_token) = state
        .read_with(window, |state, _| {
            (
                state
                    .result
                    .clone()
                    .expect("background preparation completed")
                    .expect("valid PNG"),
                state.cancellation_for_test(),
            )
        })
        .expect("live keyed state");
    let pixels = window
        .update(|_, cx| image.to_image_data(cx.svg_renderer()))
        .expect("GPUI decoded admitted PNG");
    assert_eq!(
        pixels.as_bytes(0),
        Some([0, 0, 255, 255, 0, 255, 0, 255].as_slice())
    );
    let bounds = window
        .debug_bounds("SIM-NATIVE-PREVIEW-IMAGE")
        .expect("rendered image element");
    assert!(bounds.size.width > px(0.0) && bounds.size.height > px(0.0));
    assert!(window.update(|_, cx| gpui::ImageSource::Image(image.clone()).is_asset_cached(cx)));
    view.update_in(window, |_view, _, cx| cx.notify());
    window.run_until_parked();
    let retained = state
        .read_with(window, |state, _| {
            state
                .result
                .clone()
                .expect("retained preparation")
                .expect("retained PNG")
        })
        .expect("live keyed state");
    assert!(Arc::ptr_eq(&image, &retained));
    view.update_in(window, |view, _, cx| {
        let preview = view.preview.as_mut().expect("current preview");
        preview.revision += 1;
        preview.width = Some(999);
        cx.notify();
    });
    window.run_until_parked();
    assert!(first_token.is_cancelled());
    assert!(!window.update(|_, cx| gpui::ImageSource::Image(image.clone()).is_asset_cached(cx)));
    let second_token = state
        .read_with(window, |state, _| {
            assert!(
                state
                    .result
                    .as_ref()
                    .expect("dimension failure retained")
                    .is_err()
            );
            state.cancellation_for_test()
        })
        .expect("replacement keyed state");
    view.update_in(window, |view, _, cx| {
        view.preview = None;
        cx.notify();
    });
    window.run_until_parked();
    window.update(|window, _| window.refresh());
    window.run_until_parked();
    assert!(second_token.is_cancelled());
}

#[gpui::test]
async fn task18_native_preview_rejects_malformed_unsupported_bombs_and_cancellation(
    cx: &mut TestAppContext,
) {
    let preview = native_preview_test_payload().expect("genuine PNG");
    let mut malformed = preview.clone();
    malformed.encoded_bytes.truncate(8);
    let mut unsupported = preview.clone();
    unsupported.media_type = "image/svg+xml".to_owned();
    let oversized = sim_media::encode_png_frame(
        &vec![0.0; 1_025 * 3],
        1,
        1,
        1_025,
        3,
        0,
        &BTreeMap::new(),
        sim_media::PngLimits::default(),
    )
    .expect("genuine larger PNG under media's ordinary limit");
    let mut bomb = preview.clone();
    bomb.width = Some(2);
    bomb.height = Some(1);
    bomb.encoded_bytes = oversized;
    for rejected in [malformed, unsupported, bomb] {
        let result = cx
            .background_executor
            .spawn(async move {
                crate::output_view::prepare_native_preview(&rejected, &CancellationToken::default())
            })
            .await;
        assert!(result.is_err());
    }
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let result = cx
        .background_executor
        .spawn(async move { crate::output_view::prepare_native_preview(&preview, &cancellation) })
        .await;
    assert!(result.is_err());
}

struct NativeSavedOutputProbe {
    snapshot: sim_runtime::ExecutionSnapshot,
    attempt_id: AttemptId,
    output_id: Uuid,
}

impl Render for NativeSavedOutputProbe {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        crate::output_view::OutputView::new(
            self.snapshot.clone(),
            Some(self.attempt_id),
            Some(self.output_id),
            Arc::new(|_, _, _| {}),
        )
    }
}

#[gpui::test]
async fn task18_native_saved_image_survives_success_and_reconciliation_and_suppresses_stale(
    cx: &mut TestAppContext,
) {
    use sim_runtime::{
        AssetNamespace, AssetRoots, AssetService, UploadRequest, authorize_native_output_committer,
    };
    let directory = tempfile::tempdir().expect("owned asset test directory");
    let profile_id = profile(0x182001);
    let prompt_id = prompt(0x182002);
    let attempt_id = attempt(0x182003);
    let output_id = identifier(0x182004);
    let mut roots = Vec::new();
    for (namespace, name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = directory.path().join(name);
        fs::create_dir(&path).expect("create canonical asset namespace");
        roots.push((namespace, path));
    }
    let assets = Arc::new(Mutex::new(
        AssetService::open(
            AssetRoots::new(profile_id.0.to_string(), roots).expect("canonical profile roots"),
        )
        .expect("open canonical asset service"),
    ));
    let preview = native_preview_test_payload().expect("genuine PNG");
    let byte_length = preview.encoded_bytes.len() as u64;
    let uploaded = assets
        .lock()
        .expect("asset service lock")
        .upload(
            UploadRequest {
                namespace: AssetNamespace::Output,
                filename: "saved.png".into(),
                subfolder: PathBuf::new(),
                bytes: preview.encoded_bytes.clone(),
                overwrite: false,
                tags: Default::default(),
            },
            &authorize_native_output_committer(profile_id.0.to_string())
                .expect("canonical output write grant"),
            &CancellationToken::default(),
        )
        .expect("canonical saved output bytes");
    let reference = uploaded
        .record
        .identity
        .to_reference()
        .expect("canonical asset reference");
    let mut output = native_image_output(
        profile_id,
        prompt_id,
        attempt_id,
        output_id,
        "saved.png",
        &reference,
    );
    output.availability = ExecutionOutputAvailability::Ready {
        reference,
        byte_length,
    };
    let record = record_with_events(
        profile_id,
        prompt_id,
        attempt_id,
        vec![
            (None, AttemptEventKind::Started),
            (
                Some(preview.node_id.clone()),
                AttemptEventKind::Preview { preview },
            ),
            (
                Some(output.node_id.clone()),
                AttemptEventKind::OutputAvailable { output },
            ),
            (None, AttemptEventKind::Succeeded),
        ],
    )
    .expect("canonical successful record");
    let mut presentation = ExecutionPresentationService::new(8).expect("bounded presentation");
    presentation
        .reconcile(ExecutionReconciliation {
            profile_id,
            source_revision: 1,
            source: ExecutionDataSource::Recovery,
            status: ExecutionSnapshotStatus::Ready,
            queue: Vec::new(),
            records: vec![record],
            plans: Vec::new(),
            acknowledged_requests: Vec::new(),
        })
        .expect("reconcile persisted successful attempt");
    let snapshot = presentation
        .snapshot(profile_id)
        .expect("successful persisted snapshot");
    let terminal = snapshot.attempts.first().expect("terminal attempt");
    assert_eq!(terminal.state, AttemptState::Succeeded);
    assert!(terminal.preview.is_none() && terminal.previews.is_empty());
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
        crate::register_native_asset_services(assets.clone(), cx)
            .expect("register real canonical reader");
    });
    let (view, window) = cx.add_window_view(|_, _| NativeSavedOutputProbe {
        snapshot,
        attempt_id,
        output_id,
    });
    window.run_until_parked();
    let bounds = window
        .debug_bounds("SIM-NATIVE-PREVIEW-IMAGE")
        .expect("saved output image rendered after terminal recovery");
    assert!(bounds.size.width > px(0.0) && bounds.size.height > px(0.0));
    view.update_in(window, |view, _, cx| {
        view.snapshot.status = ExecutionSnapshotStatus::Stale {
            source_revision: Some(1),
            failure: ExecutionFailure::new("test_stale", "bounded stale projection"),
        };
        cx.notify();
    });
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-STATUS").is_some());
    view.update_in(window, |view, _, cx| {
        view.snapshot.status = ExecutionSnapshotStatus::Ready;
        view.snapshot.revision += 1;
        cx.notify();
    });
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_some());
    let replacement_directory = tempfile::tempdir().expect("replacement profile directory");
    let mut replacement_roots = Vec::new();
    for (namespace, name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = replacement_directory.path().join(name);
        fs::create_dir(&path).expect("replacement namespace");
        replacement_roots.push((namespace, path));
    }
    let replacement_assets = Arc::new(Mutex::new(
        AssetService::open(
            AssetRoots::new(profile(0x182005).0.to_string(), replacement_roots)
                .expect("replacement roots"),
        )
        .expect("replacement asset service"),
    ));
    window.update(|_, cx| {
        crate::register_native_asset_services(replacement_assets, cx)
            .expect("replace profile without changing the execution snapshot");
    });
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-STATUS").is_some());
    window.update(|_, cx| {
        crate::register_native_asset_services(assets, cx).expect("restore original profile");
    });
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_some());
    view.update_in(window, |view, _, cx| {
        view.snapshot.profile_id = profile(0x182005);
        cx.notify();
    });
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    view.update_in(window, |view, _, cx| {
        view.snapshot.profile_id = profile_id;
        if let Some(attempt) = view.snapshot.attempts.first_mut()
            && let Some(output) = attempt.outputs.first_mut()
        {
            output.availability = ExecutionOutputAvailability::Removed {
                reason: "removed by canonical owner".into(),
            };
        }
        cx.notify();
    });
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-STATUS").is_some());
}

#[gpui::test]
async fn task18_native_saved_image_retry_recovers_verified_read_and_rejects_obsolete_binding(
    cx: &mut TestAppContext,
) {
    use sim_runtime::{
        AssetNamespace, AssetRoots, AssetService, UploadRequest, authorize_native_output_committer,
    };
    let directory = tempfile::tempdir().expect("owned retry asset directory");
    let profile_id = profile(0x183001);
    let prompt_id = prompt(0x183002);
    let attempt_id = attempt(0x183003);
    let output_id = identifier(0x183004);
    let mut roots = Vec::new();
    for (namespace, name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = directory.path().join(name);
        fs::create_dir(&path).expect("create canonical retry namespace");
        roots.push((namespace, path));
    }
    let assets = Arc::new(Mutex::new(
        AssetService::open(
            AssetRoots::new(profile_id.0.to_string(), roots).expect("canonical retry roots"),
        )
        .expect("canonical retry assets"),
    ));
    let bytes = native_preview_test_payload()
        .expect("genuine red and green PNG")
        .encoded_bytes;
    let image = Arc::new(gpui::Image::from_bytes(
        gpui::ImageFormat::Png,
        bytes.clone(),
    ));
    let uploaded = assets
        .lock()
        .expect("retry asset lock")
        .upload(
            UploadRequest {
                namespace: AssetNamespace::Output,
                filename: "retry.png".into(),
                subfolder: PathBuf::new(),
                bytes: bytes.clone(),
                overwrite: false,
                tags: Default::default(),
            },
            &authorize_native_output_committer(profile_id.0.to_string())
                .expect("canonical output grant"),
            &CancellationToken::default(),
        )
        .expect("real indexed saved PNG");
    let reference = uploaded
        .record
        .identity
        .to_reference()
        .expect("saved retry reference");
    let mut output = native_image_output(
        profile_id,
        prompt_id,
        attempt_id,
        output_id,
        "retry.png",
        &reference,
    );
    output.availability = ExecutionOutputAvailability::Ready {
        reference: reference.clone(),
        byte_length: bytes.len() as u64,
    };
    let record = record_with_events(
        profile_id,
        prompt_id,
        attempt_id,
        vec![
            (None, AttemptEventKind::Started),
            (
                Some(output.node_id.clone()),
                AttemptEventKind::OutputAvailable { output },
            ),
            (None, AttemptEventKind::Succeeded),
        ],
    )
    .expect("canonical retry attempt");
    let mut presentation =
        ExecutionPresentationService::new(8).expect("bounded retry presentation");
    presentation
        .reconcile(ExecutionReconciliation {
            profile_id,
            source_revision: 1,
            source: ExecutionDataSource::Recovery,
            status: ExecutionSnapshotStatus::Ready,
            queue: Vec::new(),
            records: vec![record],
            plans: Vec::new(),
            acknowledged_requests: Vec::new(),
        })
        .expect("reconcile retry attempt");
    let snapshot = presentation
        .snapshot(profile_id)
        .expect("ready retry snapshot");
    let original_revision = snapshot.revision;
    let saved = directory.path().join("output/retry.png");
    let parked = directory.path().join("retry-owned-parking.png");
    fs::rename(&saved, &parked)
        .expect("temporarily remove exact indexed file without changing its bytes");
    cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        crate::init(cx);
        crate::register_native_asset_services(assets.clone(), cx)
            .expect("register canonical retry reader");
    });
    let (view, window) = cx.add_window_view(|_, _| NativeSavedOutputProbe {
        snapshot,
        attempt_id,
        output_id,
    });
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    let retry = window
        .debug_bounds("SIM-NATIVE-IMAGE-RETRY")
        .expect("real Retry Image control after verified read fails");
    assert!(!window.update(|_, cx| image.is_asset_cached(cx)));
    fs::rename(&parked, &saved).expect("restore exact original indexed file and metadata");
    assert_eq!(fs::read(&saved).expect("restored saved bytes"), bytes);
    view.update_in(window, |_, _, cx| cx.notify());
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    assert!(window.debug_bounds("SIM-NATIVE-IMAGE-RETRY").is_some());
    window.simulate_click(retry.center(), Modifiers::default());
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-IMAGE-RETRY").is_none());
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_some());
    assert_eq!(
        view.read_with(window, |view, _| view.snapshot.revision),
        original_revision
    );
    let pixels = window
        .update(|window, cx| image.clone().get_render_image(window, cx))
        .expect("actual rendered GPUI cache entry, without fetching a decoder task");
    assert_eq!(
        pixels.as_bytes(0),
        Some([0, 0, 255, 255, 0, 255, 0, 255].as_slice())
    );
    view.update_in(window, |view, _, cx| {
        view.snapshot.status = ExecutionSnapshotStatus::Stale {
            source_revision: Some(1),
            failure: ExecutionFailure::new("retry_stale", "stale retry projection"),
        };
        cx.notify();
    });
    window.run_until_parked();
    window.refresh().expect("release hidden retry state");
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    assert!(window.debug_bounds("SIM-NATIVE-IMAGE-RETRY").is_none());
    assert!(!window.update(|_, cx| image.is_asset_cached(cx)));
    fs::rename(&saved, &parked).expect("prepare second genuine read failure");
    view.update_in(window, |view, _, cx| {
        view.snapshot.status = ExecutionSnapshotStatus::Ready;
        cx.notify();
    });
    window.run_until_parked();
    let obsolete_retry = window
        .debug_bounds("SIM-NATIVE-IMAGE-RETRY")
        .expect("second real retry control");
    let old_reader = window.update(|_, cx| {
        crate::native_asset_services(cx)
            .expect("old service")
            .image_output_reader()
            .expect("old reader")
    });
    let replacement_directory = tempfile::tempdir().expect("replacement retry profile");
    let mut replacement_roots = Vec::new();
    for (namespace, name) in [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ] {
        let path = replacement_directory.path().join(name);
        fs::create_dir(&path).expect("replacement retry namespace");
        replacement_roots.push((namespace, path));
    }
    let replacement = Arc::new(Mutex::new(
        AssetService::open(
            AssetRoots::new(profile(0x183005).0.to_string(), replacement_roots)
                .expect("replacement profile roots"),
        )
        .expect("replacement profile assets"),
    ));
    window.update(|_, cx| {
        crate::register_native_asset_services(replacement, cx)
            .expect("replace retry service/profile")
    });
    window.run_until_parked();
    fs::rename(&parked, &saved).expect("restore original bytes after service revocation");
    assert!(
        old_reader
            .read(&reference, &CancellationToken::default())
            .is_err()
    );
    assert!(window.debug_bounds("SIM-NATIVE-IMAGE-RETRY").is_none());
    window.simulate_click(obsolete_retry.center(), Modifiers::default());
    window.run_until_parked();
    assert!(window.debug_bounds("SIM-NATIVE-PREVIEW-IMAGE").is_none());
    assert!(!window.update(|_, cx| image.is_asset_cached(cx)));
}
