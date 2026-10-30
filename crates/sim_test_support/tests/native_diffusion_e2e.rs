use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, decode_png};
use sim_runtime::{
    AttemptState, NATIVE_DIFFUSION_REGISTRY_VERSION, NativeDiffusionProvider,
    NativeImageOutputProposal, NativeImageWorkerEvent, NativeImageWorkerPlan, OutputCommitter,
    RuntimeSupervisor, RuntimeSupervisorError, Sd15GuidanceAdapter, SupervisorPolicy, WorkerHealth,
    WorkerLaunchConfig, WorkerOperationStage, authorize_native_output_committer,
    compile_native_diffusion_workflow,
};
use sim_sampler::generated_native_diffusion::{
    NativeDiffusionSamplerError, checked_native_diffusion_plan, normal_sigmas, sample_euler,
    sd15_interpret_prediction,
};
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, StreamId, generated_native_diffusion::tensor_from_f32,
};
use sim_test_support::NativeDiffusionFixture;
use sim_types::{AttemptId, NodeId, ProfileId, PromptId, WorkerId, WorkerMessage};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use uuid::Uuid;

#[path = "support/accelerator_selection.rs"]
mod accelerator_selection;

const WORKFLOW: &[u8] = include_bytes!("../fixtures/native_diffusion/workflow.json");
const PROFILE: ProfileId = ProfileId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3701));
const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
// This debug-build deadlock watchdog is deliberately separate from the five-second
// release-profile performance gate in VAL-NATIVE-E2E-002.
const DEBUG_WORKER_RESULT_TIMEOUT: Duration = Duration::from_secs(600);

struct TestRoots {
    _directory: tempfile::TempDir,
    model_root: PathBuf,
    asset_roots: sim_runtime::AssetRoots,
    worker_root: PathBuf,
}

impl TestRoots {
    fn new() -> Result<Self, Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        let model_root = directory.path().join("model");
        let worker_root = directory.path().join("worker");
        fs::create_dir(&model_root)?;
        fs::create_dir(&worker_root)?;
        let source = NativeDiffusionFixture::checked_in();
        for entry in fs::read_dir(source.root())? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                fs::copy(entry.path(), model_root.join(entry.file_name()))?;
            }
        }
        let mut roots = Vec::new();
        for (namespace, name) in [
            (sim_runtime::AssetNamespace::Input, "input"),
            (sim_runtime::AssetNamespace::Output, "output"),
            (sim_runtime::AssetNamespace::Temporary, "temporary"),
            (sim_runtime::AssetNamespace::Model, "model"),
            (sim_runtime::AssetNamespace::Plugin, "plugin"),
        ] {
            let path = if namespace == sim_runtime::AssetNamespace::Model {
                model_root.clone()
            } else {
                let path = directory.path().join(name);
                fs::create_dir(&path)?;
                path
            };
            roots.push((namespace, path));
        }
        Ok(Self {
            _directory: directory,
            model_root,
            asset_roots: sim_runtime::AssetRoots::new(PROFILE.0.to_string(), roots)?,
            worker_root,
        })
    }

    fn launch(&self, worker: u128, memory_limit: u64) -> WorkerLaunchConfig {
        let mut config = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_native_diffusion_worker_fixture"),
            PROFILE,
            WorkerId(Uuid::from_u128(worker)),
            NATIVE_DIFFUSION_REGISTRY_VERSION,
            memory_limit,
        );
        config.arguments = vec![
            "--fixture-model-root".to_owned(),
            self.model_root.to_string_lossy().into_owned(),
        ];
        config.working_directory = Some(self.worker_root.clone());
        config.environment = vec![("PATH".to_owned(), String::new())];
        config.policy = SupervisorPolicy {
            heartbeat_interval: Duration::from_secs(30),
            missed_heartbeat_limit: 3,
            shutdown_timeout: Duration::from_secs(3),
            ready_timeout: Duration::from_secs(30),
            maximum_automatic_restarts: 1,
            restart_backoff: Duration::from_millis(1),
        };
        config
    }
}

#[test]
fn val_native_e2e_002() -> Result<(), Box<dyn Error>> {
    let roots = TestRoots::new()?;
    let fixture = Arc::new(NativeDiffusionFixture::at(roots.model_root.clone()));
    let source_reference = fixture.source_sampler_reference()?;
    let provider: Arc<dyn NativeDiffusionProvider> = fixture.clone();
    let mut plan = compile_native_diffusion_workflow(WORKFLOW, &BTreeSet::new(), provider)?;
    plan.prompt_id = PromptId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_3702));
    let mut cases = BTreeMap::new();
    let cancellation = CancellationToken::default();
    let cache_identities = fixture.cache_identities(&cancellation)?;
    let model_digest = cache_identities.model_digest();
    let vae_cache_identities = cache_identities.vae();
    let conditioning_cache_identities = cache_identities.conditioning();
    cases.insert(
        "canonical_vae_cache_binds_identity_artifact_patch_and_execution",
        vae_cache_identities.artifact() == model_digest
            && vae_cache_identities.identity().len() == 64
            && vae_cache_identities.patch().len() == 64
            && vae_cache_identities.execution().len() == 64
            && vae_cache_identities.identity() != vae_cache_identities.artifact()
            && vae_cache_identities.patch() != vae_cache_identities.artifact()
            && vae_cache_identities.execution() != vae_cache_identities.identity(),
    );
    cases.insert(
        "canonical_conditioning_cache_binds_patch_control_guidance_and_execution",
        conditioning_cache_identities.conditioning().len() == 64
            && conditioning_cache_identities.guidance().len() == 64
            && conditioning_cache_identities.model_patch().len() == 64
            && conditioning_cache_identities.model_execution().len() == 64
            && conditioning_cache_identities.control().len() == 64
            && conditioning_cache_identities.execution().len() == 64
            && conditioning_cache_identities.execution()
                != conditioning_cache_identities.guidance(),
    );
    cases.insert(
        "workflow_has_six_exact_node_types",
        plan.nodes.len() == 7
            && plan
                .nodes
                .values()
                .map(|node| node.class_type.as_str())
                .collect::<BTreeSet<_>>()
                == [
                    "CheckpointLoaderSimple",
                    "CLIPTextEncode",
                    "EmptyLatentImage",
                    "KSampler",
                    "VAEDecode",
                    "SaveImage",
                ]
                .into_iter()
                .collect(),
    );

    let mut supervisor =
        smol::block_on(RuntimeSupervisor::start(roots.launch(0x3703, MEMORY_LIMIT)))?;
    cases.insert(
        "native_worker_ready_on_cpu",
        supervisor.snapshot().health == WorkerHealth::BackendReady,
    );
    cases.insert(
        "worker_has_no_path_or_source_checkout",
        supervisor
            .snapshot()
            .launch
            .environment_names
            .iter()
            .any(|name| name == "PATH")
            && !roots.worker_root.join("projects").exists()
            && !roots.worker_root.join("ComfyUI").exists(),
    );

    let first = execute(&mut supervisor, &plan, AttemptId(Uuid::from_u128(1)))?;
    let (result, proposals) = completed(first)?;
    cases.insert(
        "seven_node_execution_succeeded",
        result.report.state == AttemptState::Succeeded
            && result.executed_node_count == 7
            && result.report.error.is_none(),
    );
    let ui = result.decode_ui_outputs()?;
    let expected_tokens: Value = serde_json::from_slice(&fixture.read("tokens.json")?)?;
    cases.insert(
        "positive_and_negative_tokens_match",
        ui.get(&NodeId("2".to_owned()))
            .and_then(|value| value.get("tokens"))
            == expected_tokens.get("positive")
            && ui
                .get(&NodeId("3".to_owned()))
                .and_then(|value| value.get("tokens"))
                == expected_tokens.get("negative"),
    );
    let sampler = ui.get(&NodeId("5".to_owned())).ok_or_else(|| {
        format!(
            "KSampler omitted checkpoint trace; execution error: {:?}",
            result.report.error
        )
    })?;
    cases.insert(
        "all_sampler_intermediate_hashes_match",
        sampler_hashes_match(sampler, &source_reference),
    );
    let receipts = OutputCommitter::open(roots.asset_roots.clone())?.commit_proposal_batch_now(
        &proposals
            .iter()
            .map(|proposal| proposal.output().clone())
            .collect::<Vec<_>>(),
        &authorize_native_output_committer(&roots.asset_roots.profile_id)?,
        &cancellation,
    )?;
    let receipt = receipts.first().ok_or("SaveImage produced no output")?;
    let actual_png = fs::read(
        roots
            .asset_roots
            .test_resolve_existing(&receipt.operation().identity)?,
    )?;
    let expected_png = fixture.read("output.png")?;
    let actual = decode_png(&actual_png, PngLimits::default())?;
    let expected = decode_png(&expected_png, PngLimits::default())?;
    cases.insert(
        "vae_pixels_and_transactional_png_match",
        actual.width == 32
            && actual.height == 32
            && actual.pixels_bhwc == expected.pixels_bhwc
            && receipt.operation().sha256 == format!("{:x}", Sha256::digest(&actual_png)),
    );
    let metadata = actual.metadata.sim_metadata();
    let prompt_metadata = metadata
        .prompt
        .as_deref()
        .map(serde_json::from_str::<Value>)
        .transpose()?;
    let workflow_metadata = metadata
        .workflow
        .as_deref()
        .map(serde_json::from_str::<Value>)
        .transpose()?;
    let frontend_version = metadata
        .unknown
        .get("frontendVersion")
        .map(|value| serde_json::from_str::<Value>(value))
        .transpose()?;
    let expected_workflow: Value = serde_json::from_slice(WORKFLOW)?;
    cases.insert(
        "prompt_workflow_and_frontend_metadata_match",
        prompt_metadata
            .as_ref()
            .and_then(Value::as_object)
            .is_some_and(|prompt| {
                prompt.len() == 7
                    && prompt
                        .get("5")
                        .and_then(|node| node.get("class_type"))
                        .and_then(Value::as_str)
                        == Some("KSampler")
                    && prompt
                        .get("7")
                        .and_then(|node| node.get("class_type"))
                        .and_then(Value::as_str)
                        == Some("SaveImage")
            })
            && workflow_metadata.as_ref() == Some(&expected_workflow)
            && frontend_version == Some(json!("zed-native-diffusion-v1")),
    );

    let warm = execute(&mut supervisor, &plan, AttemptId(Uuid::from_u128(2)))?;
    let (warm, _) = completed(warm)?;
    cases.insert(
        "warm_execution_uses_canonical_cache",
        warm.report.state == AttemptState::Succeeded && warm.report.cache_hits >= 6,
    );
    cases.insert(
        "cancellation_is_typed_at_every_denoiser_evaluation",
        cancellation_at_every_denoiser_evaluation(&fixture)?,
    );

    let cancelled_attempt = AttemptId(Uuid::from_u128(4));
    let cancelled_worker_plan =
        NativeImageWorkerPlan::new(plan.clone(), BTreeMap::new(), true, 100)?;
    smol::block_on(supervisor.execute(
        plan.prompt_id,
        cancelled_attempt,
        serde_json::to_vec(&cancelled_worker_plan)?,
    ))?;
    smol::block_on(supervisor.cancel(
        plan.prompt_id,
        cancelled_attempt,
        "VAL-NATIVE-E2E-002 cancellation probe",
    ))?;
    let (cancelled_event, cancelled_proposals) =
        smol::block_on(await_result(&supervisor, Duration::from_secs(30)))?;
    cases.insert(
        "worker_cancellation_emits_no_output_proposal",
        matches!(
            cancelled_event,
            NativeImageWorkerEvent::Failed {
                cancelled: true,
                ..
            }
        ) && cancelled_proposals.is_empty(),
    );

    let crash_status = smol::block_on(supervisor.terminate())?;
    cases.insert(
        "worker_crash_is_observed_as_an_unsuccessful_exit",
        !crash_status.success()
            && matches!(
                supervisor.snapshot().health,
                WorkerHealth::Exited { success: false, .. }
            ),
    );
    let mut recovered = smol::block_on(supervisor.recover())?;
    let recovery_snapshot = recovered.snapshot();
    let recovered_result = execute(&mut recovered, &plan, AttemptId(Uuid::from_u128(3)))?;
    let (recovered_result, recovered_proposals) = completed(recovered_result)?;
    let recovered_ui = recovered_result.decode_ui_outputs()?;
    let recovered_sampler_matches = match recovered_ui.get(&NodeId("5".to_owned())) {
        Some(sampler) => sampler_hashes_match(sampler, &source_reference),
        None => false,
    };
    cases.insert(
        "worker_crash_restart_replays_deterministically",
        recovery_snapshot.health == WorkerHealth::BackendReady
            && recovery_snapshot
                .operation
                .transitions
                .iter()
                .any(|transition| transition.stage == WorkerOperationStage::Recover)
            && recovered_result.report.state == AttemptState::Succeeded
            && recovered_sampler_matches
            && recovered_proposals.len() == 1
            && recovered_proposals[0].output().content() == actual_png,
    );
    cases.insert(
        "worker_shutdown_succeeds",
        smol::block_on(recovered.shutdown())?.success(),
    );

    let mut constrained = smol::block_on(RuntimeSupervisor::start(roots.launch(0x3705, 1024)))?;
    let (out_of_memory_event, out_of_memory_proposals) =
        execute(&mut constrained, &plan, AttemptId(Uuid::from_u128(5)))?;
    cases.insert(
        "out_of_memory_plan_is_rejected_before_dispatch",
        matches!(
            out_of_memory_event,
            NativeImageWorkerEvent::Failed {
                cancelled: false,
                ref message,
            } if message.contains("memory preflight")
        ) && out_of_memory_proposals.is_empty(),
    );
    cases.insert(
        "constrained_worker_shutdown_succeeds",
        smol::block_on(constrained.shutdown())?.success(),
    );

    for (name, passed) in accelerator_selection::accelerator_selection_contract_cases() {
        cases.insert(name, passed);
    }
    assert!(
        cases.values().all(|passed| *passed),
        "VAL-NATIVE-E2E-002 failures: {cases:#?}"
    );
    write_artifact(&cases, fixture.root(), &actual_png)?;
    Ok(())
}

fn cancellation_at_every_denoiser_evaluation(
    fixture: &NativeDiffusionFixture,
) -> Result<bool, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
    let loading_cancellation = CancellationToken::default();
    let loading_context =
        backend.execution_context(StreamId::DEFAULT, workspace.clone(), &loading_cancellation);
    let bundle = fixture.load_bundle_with_context(backend.clone(), &loading_context)?;
    let model = bundle.model();
    let (_, positive) = bundle.encode_text("a test", &loading_context)?;
    let (_, negative) = bundle.encode_text("", &loading_context)?;
    let plan = checked_native_diffusion_plan("euler", "normal", 0, 4, 7.0, 1.0)?;
    for cancellation_step in 0..4 {
        let cancellation = CancellationToken::default();
        let context =
            backend.execution_context(StreamId::DEFAULT, workspace.clone(), &cancellation);
        let sigmas = normal_sigmas(&backend, &context, 4, 1.0)?;
        let initial = tensor_from_f32(&backend, &[1, 4, 4, 4], &[0.0; 64], &context)?;
        let mut guidance =
            Sd15GuidanceAdapter::checked(model.as_ref(), &positive, &negative, &context)?;
        let mut reached_step = None;
        let result = sample_euler(
            &backend,
            initial,
            &sigmas,
            &context,
            |latent, sigma, step| {
                reached_step = Some(step);
                if step == cancellation_step {
                    cancellation.cancel();
                }
                let prediction = guidance
                    .execute(&backend, latent, sigma, &plan, &context)
                    .map_err(|error| error.to_string())?;
                sd15_interpret_prediction(&backend, prediction.guided(), latent, sigma, &context)
                    .map_err(|error| error.to_string())
            },
        );
        if !matches!(
            result,
            Err(NativeDiffusionSamplerError::Denoiser { ref reason, .. })
                if reason.contains("cancelled")
        ) || reached_step != Some(cancellation_step)
            || workspace.in_use_bytes() != 0
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn execute(
    supervisor: &mut RuntimeSupervisor,
    plan: &sim_runtime::CompiledPlan,
    attempt_id: AttemptId,
) -> Result<(NativeImageWorkerEvent, Vec<NativeImageOutputProposal>), Box<dyn Error>> {
    let worker_plan = NativeImageWorkerPlan::new(plan.clone(), BTreeMap::new(), true, 0)?;
    smol::block_on(supervisor.execute(
        plan.prompt_id,
        attempt_id,
        serde_json::to_vec(&worker_plan)?,
    ))?;
    Ok(smol::block_on(await_result(
        supervisor,
        DEBUG_WORKER_RESULT_TIMEOUT,
    ))?)
}

async fn await_result(
    supervisor: &RuntimeSupervisor,
    timeout: Duration,
) -> Result<(NativeImageWorkerEvent, Vec<NativeImageOutputProposal>), RuntimeSupervisorError> {
    let deadline = Instant::now() + timeout;
    let mut proposals = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(RuntimeSupervisorError::Timeout {
                stage: "native diffusion result",
            });
        }
        let envelope = supervisor.next_event(remaining).await?;
        match envelope.message {
            WorkerMessage::OutputProposal { proposal } => proposals.push(
                NativeImageOutputProposal::from_worker_proposal(proposal)
                    .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))?,
            ),
            WorkerMessage::Event { event } => {
                if let Ok(event) = postcard::from_bytes::<NativeImageWorkerEvent>(&event)
                    && matches!(
                        event,
                        NativeImageWorkerEvent::Completed { .. }
                            | NativeImageWorkerEvent::Failed { .. }
                            | NativeImageWorkerEvent::BackendUnavailable { .. }
                    )
                {
                    return Ok((event, proposals));
                }
            }
            _ => {}
        }
    }
}

fn completed(
    execution: (NativeImageWorkerEvent, Vec<NativeImageOutputProposal>),
) -> Result<
    (
        sim_runtime::NativeImageWorkerResult,
        Vec<NativeImageOutputProposal>,
    ),
    Box<dyn Error>,
> {
    match execution.0 {
        NativeImageWorkerEvent::Completed { result } => {
            let ids = execution
                .1
                .iter()
                .map(NativeImageOutputProposal::proposal_id)
                .collect::<Vec<_>>();
            if result.output_proposal_ids != ids {
                return Err("terminal result did not bind the exact proposal set".into());
            }
            Ok((result, execution.1))
        }
        NativeImageWorkerEvent::Failed { message, cancelled } => {
            Err(format!("native diffusion failed (cancelled={cancelled}): {message}").into())
        }
        NativeImageWorkerEvent::BackendUnavailable { unavailable } => {
            Err(format!("native diffusion backend is unavailable: {unavailable}").into())
        }
        NativeImageWorkerEvent::Progress { .. } => Err("nonterminal result".into()),
    }
}

fn sampler_hashes_match(value: &Value, source_reference: &Value) -> bool {
    ["noise_sha256", "denoiser_sha256", "latent_sha256"]
        .into_iter()
        .all(|name| value.get(name).is_some() && value.get(name) == source_reference.get(name))
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root missing")?
        .to_path_buf())
}

fn write_artifact(
    cases: &BTreeMap<&str, bool>,
    fixture_root: &Path,
    output: &[u8],
) -> Result<(), Box<dyn Error>> {
    let directory = workspace_root()?.join("target/comfy-parity");
    fs::create_dir_all(&directory)?;
    let artifact = json!({
        "validation_id": "VAL-NATIVE-E2E-002",
        "validation": "VAL-NATIVE-E2E-002",
        "scope": "native-sd15-tiny-worker-gpui-e2e",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "python_or_javascript_runtime": false
        },
        "fixture_digests": {
            "workflow": format!("{:x}", Sha256::digest(WORKFLOW)),
            "model": format!("{:x}", Sha256::digest(fs::read(fixture_root.join("model.safetensors"))?)),
            "expected_output": format!("{:x}", Sha256::digest(fs::read(fixture_root.join("output.png"))?)),
            "actual_output": format!("{:x}", Sha256::digest(output))
        },
        "summary": {"passed": cases.len(), "failed": 0, "skipped": 0},
        "cases": cases,
        "skipped": []
    });
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(directory.join("val-native-e2e-002.json"), bytes)?;
    Ok(())
}
