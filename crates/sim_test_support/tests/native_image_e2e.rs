use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, decode_png, encode_png_frame};
use sim_runtime::{
    AssetNamespace, AssetRoots, AssetService, AttemptState, AuthorizedCapabilities, InputBinding,
    NATIVE_IMAGE_REGISTRY_VERSION, NativeImageOutputProposal, NativeImageWorkerEvent,
    NativeImageWorkerPlan, NativeValue, OutputCommitReceipt, OutputCommitter, RuntimeSupervisor,
    RuntimeSupervisorError, SharedAssetService, SupervisorPolicy, WorkerHealth, WorkerLaunchConfig,
    authorize_native_input_reader, authorize_native_output_committer,
    compile_native_image_workflow,
};
use sim_tensor::CancellationToken;
use sim_types::{AttemptId, NodeId, ProfileId, PromptId, WorkerId, WorkerMessage};
use uuid::Uuid;

#[path = "support/accelerator_selection.rs"]
mod accelerator_selection;
#[path = "support/native_controller.rs"]
mod native_controller;

const INPUT_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/input.json");
const EXPECTED_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/expected.json");
const WORKFLOW_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/workflow.json");
const MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const PROFILE_UUID: Uuid = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1901);

#[derive(Deserialize)]
struct InputFixture {
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    pixels_bhwc: Vec<f32>,
}

#[derive(Deserialize)]
struct ExpectedFixture {
    width: u32,
    height: u32,
    rgb_u8: Vec<u8>,
}

struct NativeFixture {
    _directory: tempfile::TempDir,
    roots: AssetRoots,
    assets: SharedAssetService,
    input_authorization: AuthorizedCapabilities,
    worker_directory: PathBuf,
    input_path: PathBuf,
    input_bytes: Vec<u8>,
}

struct CompletedNativeExecution {
    result: sim_runtime::NativeImageWorkerResult,
    outputs: Vec<(NativeImageOutputProposal, OutputCommitReceipt)>,
}

impl NativeFixture {
    fn new(input: &InputFixture) -> Result<Self, Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        let worker_directory = directory.path().join("worker");
        fs::create_dir(&worker_directory)?;
        let mut typed_roots = Vec::new();
        for (namespace, name) in [
            (AssetNamespace::Input, "input"),
            (AssetNamespace::Output, "output"),
            (AssetNamespace::Temporary, "temporary"),
            (AssetNamespace::Model, "model"),
            (AssetNamespace::Plugin, "plugin"),
        ] {
            let path = directory.path().join(name);
            fs::create_dir(&path)?;
            typed_roots.push((namespace, path));
        }
        let roots = AssetRoots::new(PROFILE_UUID.to_string(), typed_roots)?;
        let input_bytes = encode_png_frame(
            &input.pixels_bhwc,
            input.batch,
            input.height,
            input.width,
            input.channels,
            0,
            &BTreeMap::new(),
            PngLimits::default(),
        )?;
        let input_path = roots
            .test_root_path(AssetNamespace::Input)?
            .join("fixture.png");
        fs::write(&input_path, &input_bytes)?;
        let assets = Arc::new(Mutex::new(AssetService::open(roots.clone())?));
        let input_authorization = authorize_native_input_reader(&roots.profile_id)?;
        Ok(Self {
            _directory: directory,
            roots,
            assets,
            input_authorization,
            worker_directory,
            input_path,
            input_bytes,
        })
    }

    fn launch_config(&self) -> WorkerLaunchConfig {
        let mut config = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
            ProfileId(PROFILE_UUID),
            WorkerId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1902)),
            NATIVE_IMAGE_REGISTRY_VERSION,
            MEMORY_LIMIT_BYTES,
        );
        config.working_directory = Some(self.worker_directory.clone());
        config.environment = vec![("PATH".to_owned(), String::new())];
        config.policy = SupervisorPolicy {
            heartbeat_interval: Duration::from_secs(30),
            missed_heartbeat_limit: 3,
            shutdown_timeout: Duration::from_secs(3),
            ready_timeout: Duration::from_secs(10),
            maximum_automatic_restarts: 1,
            restart_backoff: Duration::from_millis(1),
        };
        config
    }
}

#[test]
fn val_native_e2e_001() -> Result<(), Box<dyn Error>> {
    let input: InputFixture = serde_json::from_slice(INPUT_FIXTURE)?;
    let expected: ExpectedFixture = serde_json::from_slice(EXPECTED_FIXTURE)?;
    let fixture = NativeFixture::new(&input)?;
    let mut plan = compile_native_image_workflow(WORKFLOW_FIXTURE, &BTreeSet::new())?;
    plan.prompt_id = PromptId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1903));

    let mut cases = BTreeMap::new();
    cases.insert(
        "workflow_compiles_exact_five_nodes",
        plan.nodes.len() == 5
            && plan.nodes.keys().cloned().collect::<BTreeSet<_>>()
                == ["1", "2", "3", "4", "5"]
                    .into_iter()
                    .map(|identifier| NodeId(identifier.to_owned()))
                    .collect(),
    );

    let mut supervisor = smol::block_on(RuntimeSupervisor::start(fixture.launch_config()))?;
    cases.insert(
        "native_worker_ready_on_cpu",
        supervisor.snapshot().health == WorkerHealth::BackendReady
            && supervisor
                .accepted_backend()
                .is_some_and(|matrix| matrix.device() == sim_tensor::DeviceId::CPU),
    );
    cases.insert(
        "isolated_worker_has_no_path_or_source_tree",
        supervisor
            .snapshot()
            .launch
            .environment_names
            .iter()
            .any(|name| name == "PATH")
            && !fixture.worker_directory.join("projects/comfy").exists()
            && !fixture.worker_directory.join("ComfyUI").exists()
            && !fixture.worker_directory.join("ComfyUI-Frontend").exists(),
    );

    let first_result = execute_plan(
        &mut supervisor,
        &fixture,
        &plan,
        AttemptId(Uuid::from_u128(1)),
        0,
    )?;
    cases.insert(
        "first_execution_succeeded",
        first_result.result.report.state == AttemptState::Succeeded
            && first_result.result.report.error.is_none()
            && first_result.result.report.cache_hits == 0,
    );
    cases.insert(
        "all_five_native_nodes_executed",
        first_result.result.executed_node_count == 5,
    );
    cases.insert(
        "preview_and_save_committed_transactionally",
        first_result.outputs.len() == 2
            && first_result.outputs.iter().any(|(_, receipt)| {
                receipt.operation().identity.namespace == AssetNamespace::Temporary
            })
            && first_result.outputs.iter().any(|(_, receipt)| {
                receipt.operation().identity.namespace == AssetNamespace::Output
            }),
    );

    let (_, saved) = first_result
        .outputs
        .iter()
        .find(|(_, receipt)| receipt.operation().identity.namespace == AssetNamespace::Output)
        .ok_or("native image result omitted SaveImage output")?;
    let saved_bytes = fs::read(
        fixture
            .roots
            .test_resolve_existing(&saved.operation().identity)?,
    )?;
    let decoded = decode_png(&saved_bytes, PngLimits::default())?;
    let expected_pixels = expected
        .rgb_u8
        .iter()
        .map(|value| f32::from(*value) / 255.0)
        .collect::<Vec<_>>();
    cases.insert(
        "saved_pixels_match_deterministic_scale_and_invert",
        (decoded.width, decoded.height) == (expected.width, expected.height)
            && decoded.pixels_bhwc == expected_pixels,
    );
    cases.insert(
        "saved_png_contains_prompt_and_workflow_metadata",
        decoded.metadata.comfy_metadata().prompt
            == Some(serde_json::to_string(compiled_hidden_literal(
                &plan, "5", "prompt",
            )?)?)
            && decoded.metadata.comfy_metadata().workflow
                == Some(serde_json::to_string(
                    compiled_hidden_literal(&plan, "5", "extra_pnginfo")?
                        .get("workflow")
                        .ok_or("compiled extra_pnginfo omitted workflow")?,
                )?),
    );
    cases.insert(
        "committed_output_digest_matches_bytes",
        saved.operation().sha256 == format!("{:x}", Sha256::digest(&saved_bytes))
            && saved.operation().byte_size == u64::try_from(saved_bytes.len())?,
    );

    let warm_result = execute_plan(
        &mut supervisor,
        &fixture,
        &plan,
        AttemptId(Uuid::from_u128(2)),
        0,
    )?;
    cases.insert(
        "warm_execution_hits_three_pure_or_input_cache_entries",
        warm_result.result.report.state == AttemptState::Succeeded
            && warm_result.result.report.cache_hits == 3,
    );

    let changed_pixels = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let changed_input = encode_png_frame(
        &changed_pixels,
        1,
        1,
        2,
        3,
        0,
        &BTreeMap::new(),
        PngLimits::default(),
    )?;
    fs::write(&fixture.input_path, &changed_input)?;
    let invalidated_result = execute_plan(
        &mut supervisor,
        &fixture,
        &plan,
        AttemptId(Uuid::from_u128(3)),
        0,
    )?;
    cases.insert(
        "input_digest_change_invalidates_native_cache",
        invalidated_result.result.report.state == AttemptState::Succeeded
            && invalidated_result.result.report.cache_hits == 0,
    );

    let png_count_before_cancel = count_png_files(&fixture.roots)?;
    let cancellation_attempt = AttemptId(Uuid::from_u128(4));
    let cancellation_plan = NativeImageWorkerPlan::from_asset_service(
        plan.clone(),
        &fixture.assets,
        &fixture.input_authorization,
        &CancellationToken::default(),
        true,
        500,
    )?;
    smol::block_on(supervisor.execute(
        plan.prompt_id,
        cancellation_attempt,
        serde_json::to_vec(&cancellation_plan)?,
    ))?;
    let started = smol::block_on(supervisor.next_event(Duration::from_secs(5)))?;
    cases.insert(
        "delayed_execution_acknowledges_start_before_cancel",
        matches!(
            started.message,
            WorkerMessage::Lifecycle {
                event: sim_types::WorkerLifecycleEvent::ExecutionStarted
            }
        ),
    );
    smol::block_on(supervisor.cancel(
        plan.prompt_id,
        cancellation_attempt,
        "VAL-NATIVE-E2E-001 deterministic cancellation",
    ))?;
    let cancelled = smol::block_on(await_native_result(&supervisor, Duration::from_secs(5)))?;
    cases.insert(
        "delayed_cancellation_converges",
        matches!(
            cancelled,
            NativeImageWorkerEvent::Failed {
                cancelled: true,
                ..
            }
        ) && supervisor.snapshot().health == WorkerHealth::BackendReady,
    );
    let late_event = smol::block_on(supervisor.next_event(Duration::from_millis(200)));
    cases.insert(
        "cancelled_execution_has_no_late_output",
        count_png_files(&fixture.roots)? == png_count_before_cancel
            && matches!(late_event, Err(RuntimeSupervisorError::Timeout { .. })),
    );

    for (name, passed) in native_controller::run_native_controller_e2e()? {
        cases.insert(name, passed);
    }
    for (name, passed) in accelerator_selection::accelerator_selection_contract_cases() {
        cases.insert(name, passed);
    }

    let shutdown_status = smol::block_on(supervisor.shutdown())?;
    cases.insert("worker_shutdown_succeeds", shutdown_status.success());
    assert_all_cases(&cases);
    write_artifact(
        &fixture,
        &cases,
        json!({
            "input_declaration_sha256": format!("{:x}", Sha256::digest(INPUT_FIXTURE)),
            "input_png_sha256": format!("{:x}", Sha256::digest(&fixture.input_bytes)),
            "workflow_sha256": format!("{:x}", Sha256::digest(WORKFLOW_FIXTURE)),
            "expected_sha256": format!("{:x}", Sha256::digest(EXPECTED_FIXTURE)),
        }),
    )?;
    Ok(())
}

fn execute_plan(
    supervisor: &mut RuntimeSupervisor,
    fixture: &NativeFixture,
    plan: &sim_runtime::CompiledPlan,
    attempt_id: AttemptId,
    injected_delay_millis: u64,
) -> Result<CompletedNativeExecution, Box<dyn Error>> {
    let published_before_worker = count_png_files(&fixture.roots)?;
    let worker_plan = NativeImageWorkerPlan::from_asset_service(
        plan.clone(),
        &fixture.assets,
        &fixture.input_authorization,
        &CancellationToken::default(),
        true,
        injected_delay_millis,
    )?;
    smol::block_on(supervisor.execute(
        plan.prompt_id,
        attempt_id,
        serde_json::to_vec(&worker_plan)?,
    ))?;
    let (event, proposals) = smol::block_on(await_native_result_with_proposals(
        supervisor,
        Duration::from_secs(10),
    ))?;
    if count_png_files(&fixture.roots)? != published_before_worker {
        return Err("native worker published output before the host commit boundary".into());
    }
    completed((event, proposals), &fixture.roots)
}

async fn await_native_result_with_proposals(
    supervisor: &RuntimeSupervisor,
    timeout: Duration,
) -> Result<(NativeImageWorkerEvent, Vec<NativeImageOutputProposal>), RuntimeSupervisorError> {
    let deadline = Instant::now() + timeout;
    let mut proposals = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(RuntimeSupervisorError::Timeout {
                stage: "native image result",
            });
        }
        let envelope = supervisor.next_event(remaining).await?;
        match envelope.message {
            WorkerMessage::OutputProposal { proposal } => {
                proposals.push(
                    NativeImageOutputProposal::from_worker_proposal(proposal)
                        .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))?,
                );
            }
            WorkerMessage::Event { event } => {
                if let Ok(result) = postcard::from_bytes::<NativeImageWorkerEvent>(&event) {
                    match result {
                        NativeImageWorkerEvent::Progress { .. } => {}
                        NativeImageWorkerEvent::Completed { .. }
                        | NativeImageWorkerEvent::BackendUnavailable { .. }
                        | NativeImageWorkerEvent::Failed { .. } => {
                            return Ok((result, proposals));
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

async fn await_native_result(
    supervisor: &RuntimeSupervisor,
    timeout: Duration,
) -> Result<NativeImageWorkerEvent, RuntimeSupervisorError> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(RuntimeSupervisorError::Timeout {
                stage: "native image result",
            });
        }
        let envelope = supervisor.next_event(remaining).await?;
        if let WorkerMessage::Event { event } = envelope.message
            && let Ok(result) = postcard::from_bytes::<NativeImageWorkerEvent>(&event)
        {
            match result {
                NativeImageWorkerEvent::Progress { .. } => {}
                NativeImageWorkerEvent::Completed { .. }
                | NativeImageWorkerEvent::BackendUnavailable { .. }
                | NativeImageWorkerEvent::Failed { .. } => return Ok(result),
            }
        }
    }
}

fn completed(
    execution: (NativeImageWorkerEvent, Vec<NativeImageOutputProposal>),
    roots: &AssetRoots,
) -> Result<CompletedNativeExecution, Box<dyn Error>> {
    let (event, proposals) = execution;
    match event {
        NativeImageWorkerEvent::Completed { result } => {
            let proposal_ids = proposals
                .iter()
                .map(NativeImageOutputProposal::proposal_id)
                .collect::<Vec<_>>();
            if proposal_ids != result.output_proposal_ids {
                return Err(
                    "native image terminal result did not bind the exact proposal set".into(),
                );
            }
            let canonical = proposals
                .iter()
                .map(|proposal| proposal.output().clone())
                .collect::<Vec<_>>();
            let authorization = authorize_native_output_committer(&roots.profile_id)?;
            let mut committer = OutputCommitter::open(roots.clone())?;
            let receipts = committer.commit_proposal_batch_now(
                &canonical,
                &authorization,
                &CancellationToken::default(),
            )?;
            Ok(CompletedNativeExecution {
                result,
                outputs: proposals.into_iter().zip(receipts).collect(),
            })
        }
        NativeImageWorkerEvent::Failed { message, cancelled } => Err(format!(
            "native image worker failed unexpectedly (cancelled={cancelled}): {message}"
        )
        .into()),
        NativeImageWorkerEvent::BackendUnavailable { unavailable } => {
            Err(format!("native image backend was unavailable unexpectedly: {unavailable}").into())
        }
        NativeImageWorkerEvent::Progress { .. } => {
            Err("native image result helper received a nonterminal progress event".into())
        }
    }
}

fn compiled_hidden_literal<'a>(
    plan: &'a sim_runtime::CompiledPlan,
    node_id: &str,
    input_name: &str,
) -> Result<&'a Value, Box<dyn Error>> {
    let input = plan
        .nodes
        .get(&NodeId(node_id.to_owned()))
        .and_then(|node| node.inputs.get(input_name))
        .ok_or_else(|| format!("compiled node {node_id} omitted hidden input {input_name}"))?;
    match input {
        InputBinding::Literal {
            value: NativeValue::PreservedUnknown { value, .. },
        } => Ok(value),
        InputBinding::Literal { .. } => Err(format!(
            "compiled node {node_id} hidden input {input_name} was not preserved JSON"
        )
        .into()),
        InputBinding::Link { .. } => Err(format!(
            "compiled node {node_id} hidden input {input_name} was unexpectedly linked"
        )
        .into()),
    }
}

fn count_png_files(roots: &AssetRoots) -> Result<usize, Box<dyn Error>> {
    let mut count = 0;
    for namespace in [AssetNamespace::Output, AssetNamespace::Temporary] {
        for entry in fs::read_dir(roots.test_root_path(namespace)?)? {
            let entry = entry?;
            if entry.file_type()?.is_file()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "png")
            {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn assert_all_cases(cases: &BTreeMap<&str, bool>) {
    assert!(
        cases.values().all(|passed| *passed),
        "VAL-NATIVE-E2E-001 cases failed: {cases:#?}"
    );
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

fn target_directory() -> Result<PathBuf, Box<dyn Error>> {
    Ok(std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or(workspace_root()?.join("target")))
}

fn write_artifact(
    fixture: &NativeFixture,
    cases: &BTreeMap<&str, bool>,
    fixture_digests: Value,
) -> Result<(), Box<dyn Error>> {
    let directory = target_directory()?.join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let artifact = json!({
        "validation_id": "VAL-NATIVE-E2E-001",
        "validation": "VAL-NATIVE-E2E-001",
        "scope": "native-image-worker-controller-e2e",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "worker_binary": "sim_native_image_worker_fixture",
            "path_available_to_worker": false,
            "source_tree_available_to_worker": false,
            "python_or_javascript_runtime": false
        },
        "fixture_digests": fixture_digests,
        "profile_id": fixture.roots.profile_id,
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0
        },
        "cases": cases,
        "skipped": []
    });
    fs::write(
        directory.join("val-native-e2e-001.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}
