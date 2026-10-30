use gpui::{AppContext as _, TestAppContext, WeakEntity, px};
use serde_json::json;
use sim_runtime::{
    CanonicalClipCacheIdentities, CanonicalConditioningCacheIdentities,
    CanonicalNativeDiffusionCacheIdentities, CanonicalVaeCacheIdentities, CompiledPlan,
    ExecutionControlCommand, ExecutionControlCommandKind, ExecutionController, ExecutionDataSource,
    ExecutionFailure, ExecutionPresentationService, ExecutionSnapshotStatus, GraphCommand,
    NativeDiffusionBundle, NativeDiffusionProvider, NativeImageRuntimeError, ProfileId,
    WorkflowStorageProvider, compile_native_diffusion_workflow,
};
use sim_tensor::{CancellationToken, CpuBackend, ExecutionContext};
use sim_types::AttemptId;
use sim_ui::{
    ExecutionPlanProvider, ExecutionPlanRequest, ExecutionUiModel, GlobalExecutionUiModel,
    GraphWorkspaceItem, GraphWorkspaceModel, WorkflowOpenState,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use uuid::Uuid;

const WORKFLOW: &[u8] =
    include_bytes!("../../sim_test_support/fixtures/native_diffusion/workflow.json");
const PROFILE: ProfileId = ProfileId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3737));

struct CompileOnlyProvider;

impl NativeDiffusionProvider for CompileOnlyProvider {
    fn cache_identities(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<CanonicalNativeDiffusionCacheIdentities, NativeImageRuntimeError> {
        if cancellation.is_cancelled() {
            return Err(NativeImageRuntimeError::Cancelled);
        }
        let clip = CanonicalClipCacheIdentities::checked(
            "1".repeat(64),
            "2".repeat(64),
            "0".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64),
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
        let vae = CanonicalVaeCacheIdentities::checked(
            "6".repeat(64),
            "0".repeat(64),
            "7".repeat(64),
            "8".repeat(64),
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
        let conditioning = CanonicalConditioningCacheIdentities::checked(
            "9".repeat(64),
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64),
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))?;
        CanonicalNativeDiffusionCacheIdentities::checked(
            "0".repeat(64),
            "1".repeat(64),
            clip,
            vae,
            conditioning,
        )
        .map_err(|error| NativeImageRuntimeError::Registry(error.to_string()))
    }

    fn load(
        &self,
        _backend: Arc<CpuBackend>,
        _context: &ExecutionContext<'_>,
    ) -> Result<NativeDiffusionBundle, NativeImageRuntimeError> {
        Err(NativeImageRuntimeError::Execution(
            "the GPUI compile probe does not execute model kernels".to_owned(),
        ))
    }
}

struct PinnedPlanProvider {
    plan: CompiledPlan,
    compile_count: AtomicU64,
}

impl ExecutionPlanProvider for PinnedPlanProvider {
    fn compile(&self, request: &ExecutionPlanRequest) -> Result<CompiledPlan, ExecutionFailure> {
        if request.profile_id != PROFILE || request.workflow_bytes.is_empty() {
            return Err(ExecutionFailure::new(
                "invalid_diffusion_interaction_request",
                "the pinned diffusion interaction request is incomplete",
            ));
        }
        let mut plan = self.plan.clone();
        let compile_index = self.compile_count.fetch_add(1, Ordering::SeqCst);
        if compile_index > 0 {
            plan.prompt_id = sim_types::PromptId(Uuid::from_u128(
                0x5349_4d00_0000_0000_0000_0000_3737_0000 + u128::from(compile_index),
            ));
        }
        Ok(plan)
    }
}

#[derive(Default)]
struct RecordingController {
    commands: Mutex<Vec<ExecutionControlCommand>>,
}

impl ExecutionController for RecordingController {
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

#[gpui::test(seed = 37037)]
async fn native_diffusion_keyboard_and_pointer_queue_exact_plan(cx: &mut TestAppContext) {
    let provider: Arc<dyn NativeDiffusionProvider> = Arc::new(CompileOnlyProvider);
    let plan = compile_native_diffusion_workflow(WORKFLOW, &Default::default(), provider)
        .expect("compile pinned native diffusion workflow");
    let controller = Arc::new(RecordingController::default());
    let model = cx.update(|cx| {
        cx.set_global(db::AppDatabase::test_new());
        workspace::AppState::test(cx);
        sim_ui::init(cx);
        let mut service = ExecutionPresentationService::new(16).expect("presentation service");
        service
            .initialize_profile(
                PROFILE,
                ExecutionDataSource::Live,
                ExecutionSnapshotStatus::Ready,
            )
            .expect("initialize profile");
        let model = cx.new(|_| ExecutionUiModel::new(service, controller.clone()));
        model.update(cx, |model, cx| {
            model
                .set_active_profile(PROFILE, cx)
                .expect("activate profile");
            model.register_plan_provider(
                Arc::new(PinnedPlanProvider {
                    plan,
                    compile_count: AtomicU64::new(0),
                }),
                cx,
            );
        });
        cx.set_global(GlobalExecutionUiModel(model.clone()));
        model
    });
    let prompt = serde_json::to_vec(&json!({
        "1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"model.safetensors"}},
        "2":{"class_type":"CLIPTextEncode","inputs":{"text":"a test","clip":["1",1]}},
        "3":{"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["1",1]}},
        "4":{"class_type":"EmptyLatentImage","inputs":{"width":32,"height":32,"batch_size":1}},
        "5":{"class_type":"KSampler","inputs":{"model":["1",0],"seed":81985529216486895_u64,"steps":4,"cfg":7.0,"sampler_name":"euler","scheduler":"normal","positive":["2",0],"negative":["3",0],"latent_image":["4",0],"denoise":1.0}},
        "6":{"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}},
        "7":{"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":"native-diffusion"}}
    }))
    .expect("serialize interaction graph");
    let mut graph = GraphWorkspaceModel::open(
        "Native diffusion interaction",
        "native-diffusion-interaction",
        WorkflowStorageProvider::Draft,
        prompt,
    )
    .expect("open graph");
    graph.bind_profile_identity(PROFILE);
    let WorkflowOpenState::Editable(_) = &mut graph.open_state else {
        panic!("native diffusion graph opened read-only");
    };
    let (item, window) =
        cx.add_window_view(|_, cx| GraphWorkspaceItem::new(graph, WeakEntity::new_invalid(), cx));
    item.update(window, |item, cx| {
        assert!(item.apply_graph_command(GraphCommand::SelectAll, cx));
    });
    window.run_until_parked();
    let execute_bounds = window
        .debug_bounds("SIM-EXECUTE-BUTTON")
        .expect("accessible Execute button");
    assert!(execute_bounds.size.width > px(0.0));
    window.simulate_mouse_move(
        execute_bounds.center(),
        None::<gpui::MouseButton>,
        Default::default(),
    );
    window.run_until_parked();
    let execute_bounds = window
        .debug_bounds("SIM-EXECUTE-BUTTON")
        .expect("hovered Execute button");
    window.simulate_click(execute_bounds.center(), Default::default());
    window.run_until_parked();
    let first_attempt = item
        .read_with(window, |item, cx| {
            item.active_execution_presentation_for_test(cx)
                .ok_or_else(|| {
                    format!(
                        "pointer queue failed: error={:?} announcement={:?} trace={:?}",
                        item.model().last_error,
                        item.model().announcement,
                        item.shell_dispatch_trace_for_test()
                    )
                })
        })
        .expect("pointer queue associated an attempt");
    assert_eq!(first_attempt.profile_id, PROFILE);
    let commands = match controller.commands.lock() {
        Ok(commands) => commands.clone(),
        Err(error) => error.into_inner().clone(),
    };
    let first_command = commands.first().expect("pointer command recorded");
    let queued = match &first_command.kind {
        ExecutionControlCommandKind::Queue { plan, .. } => plan,
        other => panic!("expected pointer Queue command, got {other:?}"),
    };
    assert_eq!(queued.nodes.len(), 7);
    assert_eq!(
        queued
            .nodes
            .get(&sim_types::NodeId("5".to_owned()))
            .expect("compiled KSampler node")
            .class_type,
        "KSampler"
    );

    item.update(window, |item, cx| {
        assert!(item.apply_graph_command(GraphCommand::SelectAll, cx));
    });
    item.update_in(window, |item, window, cx| {
        item.focus_graph(window, cx);
        let execute = item.control_focus_handle_for_test("execution:execute-button", cx);
        window.focus(&execute, cx);
        assert!(execute.is_focused(window));
    });
    window.simulate_keystrokes("enter");
    window.run_until_parked();
    let command_count = match controller.commands.lock() {
        Ok(commands) => commands.len(),
        Err(error) => error.into_inner().len(),
    };
    assert_eq!(
        command_count,
        2,
        "keyboard activation did not queue after pointer activation: error={:?} announcement={:?} trace={:?}",
        item.read_with(window, |item, _| item.model().last_error.clone()),
        item.read_with(window, |item, _| item.model().announcement.clone()),
        item.read_with(window, |item, _| item
            .shell_dispatch_trace_for_test()
            .to_vec())
    );
    assert_eq!(
        model.read_with(window, |model, _| model.active_profile_id()),
        Some(PROFILE)
    );
}
