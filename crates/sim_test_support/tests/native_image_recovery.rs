use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, encode_png_frame};
use sim_runtime::{
    AssetNamespace, AssetRoots, AssetService, AttemptEventKind, AttemptState,
    AuthorizedCapabilities, ExecutionCommandAck, ExecutionCommandOutcome, ExecutionControlCommand,
    ExecutionControlCommandKind, ExecutionDataSource, ExecutionPresentationService,
    ExecutionRecoveryInterruptionReason, ExecutionSnapshotStatus, NATIVE_IMAGE_REGISTRY_VERSION,
    NativeImageOutputProposal, NativeImageWorkerEvent, NativeImageWorkerPlan,
    NativeImageWorkerProgress, NativeImageWorkerProgressKind, OutputCommitReceipt, OutputCommitter,
    OutputExecutionScope, RecoveryJournal, RetryPromptSource, RuntimeSupervisor,
    RuntimeSupervisorError, SharedAssetService, SupervisorPolicy, WorkerHealth, WorkerLaunchConfig,
    authorize_native_input_reader, authorize_native_output_committer,
    compile_native_image_workflow,
};
use sim_types::{AttemptId, ProfileId, PromptId, RequestId, WorkerId, WorkerMessage};
use uuid::Uuid;

#[path = "support/accelerator_selection.rs"]
mod accelerator_selection;

const INPUT_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/input.json");
const WORKFLOW_FIXTURE: &[u8] = include_bytes!("../fixtures/native_image/workflow.json");
const MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const PROFILE_UUID: Uuid = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1931);
const PROMPT_UUID: Uuid = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1932);
const CRASH_ATTEMPT_UUID: Uuid = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1933);

#[derive(Deserialize)]
struct InputFixture {
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    pixels_bhwc: Vec<f32>,
}

struct NativeRecoveryFixture {
    _directory: tempfile::TempDir,
    roots: AssetRoots,
    assets: SharedAssetService,
    input_authorization: AuthorizedCapabilities,
    worker_directory: PathBuf,
    input_png: Vec<u8>,
}

impl NativeRecoveryFixture {
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
        let input_png = encode_png_frame(
            &input.pixels_bhwc,
            input.batch,
            input.height,
            input.width,
            input.channels,
            0,
            &BTreeMap::new(),
            PngLimits::default(),
        )?;
        fs::write(
            roots
                .test_root_path(AssetNamespace::Input)?
                .join("fixture.png"),
            &input_png,
        )?;
        let assets = Arc::new(Mutex::new(AssetService::open(roots.clone())?));
        let input_authorization = authorize_native_input_reader(&roots.profile_id)?;
        Ok(Self {
            _directory: directory,
            roots,
            assets,
            input_authorization,
            worker_directory,
            input_png,
        })
    }

    fn launch_config(&self) -> WorkerLaunchConfig {
        let mut config = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
            ProfileId(PROFILE_UUID),
            WorkerId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1935)),
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

#[derive(Default)]
struct PhaseEvidence {
    started: bool,
    load_image_completed: bool,
    scale_tensor_completed: bool,
    invert_tensor_completed: bool,
    preview_prepared: bool,
    output_prepared: bool,
    prepared_proposal_ids: BTreeSet<Uuid>,
}

struct CompletedRetry {
    result: sim_runtime::NativeImageWorkerResult,
    proposals: Vec<NativeImageOutputProposal>,
    receipts: Vec<OutputCommitReceipt>,
}

#[test]
fn val_recovery_003() -> Result<(), Box<dyn Error>> {
    let input: InputFixture = serde_json::from_slice(INPUT_FIXTURE)?;
    let fixture = NativeRecoveryFixture::new(&input)?;
    let mut plan = compile_native_image_workflow(WORKFLOW_FIXTURE, &BTreeSet::new())?;
    plan.prompt_id = PromptId(PROMPT_UUID);
    let crash_attempt = AttemptId(CRASH_ATTEMPT_UUID);
    let profile_id = ProfileId(PROFILE_UUID);

    let mut presentation =
        ExecutionPresentationService::new_with_first_attempt_id(8, crash_attempt)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    queue_plan(&mut presentation, profile_id, 1, plan.clone())?;
    let crash_lease = presentation
        .next_queued_attempt(profile_id)?
        .ok_or("canonical presentation did not lease the crash attempt")?;
    assert_eq!(crash_lease.attempt_id, crash_attempt);
    let mut recovery_journal = RecoveryJournal::default();

    let mut supervisor = smol::block_on(RuntimeSupervisor::start(fixture.launch_config()))?;
    let original_worker_process_id = supervisor.worker_process_id();
    let worker_plan = NativeImageWorkerPlan::from_asset_service(
        crash_lease.plan.clone(),
        &fixture.assets,
        &fixture.input_authorization,
        &crash_lease.cancellation,
        true,
        0,
    )?;
    smol::block_on(supervisor.execute(
        plan.prompt_id,
        crash_attempt,
        serde_json::to_vec(&worker_plan)?,
    ))?;
    let active_snapshot = supervisor.snapshot();
    let private_started = smol::block_on(await_private_execution_started(
        &supervisor,
        Duration::from_secs(5),
    ))?;
    let mut phase_evidence = PhaseEvidence::default();
    smol::block_on(await_output_prepared(
        &supervisor,
        &mut presentation,
        &mut phase_evidence,
        Duration::from_secs(10),
    ))?;

    let crash_status = smol::block_on(supervisor.terminate())?;
    let crashed_snapshot = supervisor.snapshot();
    let published_after_crash = count_png_files(&fixture.roots)?;

    let persisted_attempts = presentation.persisted_attempts(profile_id)?;
    let mut presentation = ExecutionPresentationService::new(8)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Recovery,
        ExecutionSnapshotStatus::Ready,
    )?;
    presentation.restore_persisted_attempts(profile_id, persisted_attempts)?;
    let interrupted_attempt = presentation
        .snapshot(profile_id)?
        .attempts
        .into_iter()
        .find(|attempt| attempt.attempt_id == crash_attempt)
        .ok_or("recovered presentation omitted the crashed attempt")?;
    let interrupted = interrupted_attempt.state == AttemptState::Interrupted
        && interrupted_attempt.recovery_interruption_reason
            == Some(ExecutionRecoveryInterruptionReason::RuntimeRestart);

    let output_committer = OutputCommitter::open(fixture.roots.clone())?;
    let worker_owned_output_operations = output_committer.operations().len();
    drop(output_committer);

    let mut recovered = smol::block_on(supervisor.recover())?;
    let replacement_worker_process_id = recovered.worker_process_id();
    let recovered_snapshot = recovered.snapshot();
    let implicit_retry = smol::block_on(recovered.next_event(Duration::from_millis(200)));
    let published_before_explicit_retry = count_png_files(&fixture.roots)?;

    retry_attempt(&mut presentation, profile_id, 2, crash_attempt)?;
    let retry_record = presentation
        .snapshot(profile_id)?
        .attempts
        .into_iter()
        .find(|attempt| attempt.retry_of == Some(crash_attempt))
        .ok_or("canonical presentation did not allocate the retry attempt")?;
    let retry_attempt = retry_record.attempt_id;
    let retry_lease = presentation
        .next_queued_attempt(profile_id)?
        .ok_or("canonical presentation did not lease the retry attempt")?;
    assert_eq!(retry_lease.attempt_id, retry_attempt);
    let retry_worker_plan = NativeImageWorkerPlan::from_asset_service(
        retry_lease.plan,
        &fixture.assets,
        &fixture.input_authorization,
        &retry_lease.cancellation,
        true,
        0,
    )?;
    smol::block_on(recovered.execute(
        plan.prompt_id,
        retry_attempt,
        serde_json::to_vec(&retry_worker_plan)?,
    ))?;
    let retry_result = smol::block_on(await_completed(
        &recovered,
        &mut presentation,
        &mut recovery_journal,
        &fixture.assets,
        &fixture.roots,
        Duration::from_secs(10),
    ))?;
    let published_after_explicit_retry = count_png_files(&fixture.roots)?;
    let output_scope = OutputExecutionScope {
        profile_id,
        prompt_id: plan.prompt_id,
        attempt_id: retry_attempt,
    };
    let authoritative_receipts = OutputCommitter::open(fixture.roots.clone())?
        .committed_receipts_for_scope(&output_scope)?;
    let recovered_receipts = RecoveryJournal::decode(&recovery_journal.encode()?)?
        .receipts_for_attempt(profile_id, plan.prompt_id, retry_attempt)
        .cloned()
        .collect::<Vec<_>>();
    let replacement_backend_ready = recovered.accepted_backend().is_some();

    let second_crash_status = smol::block_on(recovered.terminate())?;
    let restart_loop_prevented = matches!(
        smol::block_on(recovered.recover()),
        Err(RuntimeSupervisorError::RecoveryBudgetExhausted { maximum: 1 })
    );

    let mut cases = BTreeMap::new();
    cases.insert(
        "active_attempt_identity_visible_before_process_tree_kill",
        active_snapshot.active_prompt_id == Some(plan.prompt_id)
            && active_snapshot.active_attempt_id == Some(crash_attempt),
    );
    cases.insert(
        "attempt_is_interrupted_through_public_recovery_apis",
        interrupted,
    );
    cases.insert(
        "bounded_recovery_runs_exactly_once",
        recovered_snapshot.health == WorkerHealth::BackendReady
            && restart_loop_prevented
            && !second_crash_status.success(),
    );
    cases.insert(
        "fresh_worker_revokes_process_local_tensor_and_cache_handles",
        original_worker_process_id.is_some()
            && replacement_worker_process_id.is_some()
            && original_worker_process_id != replacement_worker_process_id
            && replacement_backend_ready
            && retry_result.result.report.cache_hits == 0
            && retry_result.result.report.outputs.is_empty(),
    );
    cases.insert(
        "gpui_host_process_survives_owned_worker_loss",
        !crash_status.success()
            && matches!(
                crashed_snapshot.health,
                WorkerHealth::Exited { success: false, .. }
            ),
    );
    cases.insert(
        "load_image_node_phase_completed_before_crash",
        phase_evidence.load_image_completed,
    );
    cases.insert(
        "native_execution_started_before_kill",
        private_started && phase_evidence.started,
    );
    cases.insert(
        "no_implicit_retry_after_bounded_recovery",
        matches!(implicit_retry, Err(RuntimeSupervisorError::Timeout { .. }))
            && published_before_explicit_retry == 0,
    );
    cases.insert(
        "no_partial_preview_or_output_is_published",
        published_after_crash == 0 && worker_owned_output_operations == 0,
    );
    cases.insert(
        "output_phase_was_prepared_before_process_tree_kill",
        phase_evidence.output_prepared,
    );
    cases.insert(
        "prepared_worker_outputs_never_enter_host_commit_ownership",
        phase_evidence.prepared_proposal_ids.len() == 2
            && worker_owned_output_operations == 0
            && recovery_journal
                .receipts_for_attempt(profile_id, plan.prompt_id, crash_attempt)
                .next()
                .is_none(),
    );
    cases.insert(
        "preview_phase_was_prepared_before_process_tree_kill",
        phase_evidence.preview_prepared,
    );
    cases.insert(
        "retry_is_explicit_and_uses_a_new_auditable_attempt_identity",
        retry_record.retry_of == Some(crash_attempt)
            && retry_record.retry_source == Some(RetryPromptSource::OriginalPrompt)
            && retry_attempt != crash_attempt,
    );
    cases.insert(
        "explicit_retry_execution_succeeds",
        retry_result.result.report.state == AttemptState::Succeeded
            && retry_result.result.executed_node_count == 5
            && retry_result.proposals.len() == 2,
    );
    cases.insert(
        "explicit_retry_commits_one_receipt_per_proposal",
        retry_result.receipts.len() == 2 && published_after_explicit_retry == 2,
    );
    cases.insert(
        "output_committer_restart_recovers_exact_scoped_receipts",
        authoritative_receipts == retry_result.receipts,
    );
    cases.insert(
        "receipt_only_recovery_journal_round_trips_exact_commit_identities",
        recovered_receipts.len() == retry_result.receipts.len()
            && recovered_receipts.iter().all(|recorded| {
                retry_result.receipts.iter().any(|receipt| {
                    recorded.proposal_id() == receipt.proposal_id()
                        && recorded.operation_id() == receipt.operation().operation_id
                })
            }),
    );
    cases.insert(
        "tensor_scale_phase_completed_before_crash",
        phase_evidence.scale_tensor_completed,
    );
    cases.insert(
        "tensor_invert_phase_completed_before_crash",
        phase_evidence.invert_tensor_completed,
    );
    for (name, passed) in accelerator_selection::accelerator_selection_contract_cases() {
        cases.insert(name, passed);
    }
    assert!(
        cases.values().all(|passed| *passed),
        "VAL-RECOVERY-003 cases failed: {cases:#?}"
    );
    write_artifact(&fixture, &cases)?;
    Ok(())
}

fn queue_plan(
    presentation: &mut ExecutionPresentationService,
    profile_id: ProfileId,
    request_sequence: u128,
    plan: sim_runtime::CompiledPlan,
) -> Result<(), Box<dyn Error>> {
    accept_command(
        presentation,
        ExecutionControlCommand {
            request_id: request_id(request_sequence),
            profile_id,
            expected_revision: None,
            kind: ExecutionControlCommandKind::Queue {
                plan,
                priority: 0,
                front: false,
            },
        },
    )
}

fn retry_attempt(
    presentation: &mut ExecutionPresentationService,
    profile_id: ProfileId,
    request_sequence: u128,
    attempt_id: AttemptId,
) -> Result<(), Box<dyn Error>> {
    accept_command(
        presentation,
        ExecutionControlCommand {
            request_id: request_id(request_sequence),
            profile_id,
            expected_revision: None,
            kind: ExecutionControlCommandKind::Retry {
                attempt_id,
                source: RetryPromptSource::OriginalPrompt,
                replacement_plan: None,
            },
        },
    )
}

fn accept_command(
    presentation: &mut ExecutionPresentationService,
    command: ExecutionControlCommand,
) -> Result<(), Box<dyn Error>> {
    let request_id = command.request_id;
    let profile_id = command.profile_id;
    presentation.submit(command)?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: None,
        },
    })?;
    Ok(())
}

fn request_id(sequence: u128) -> RequestId {
    RequestId(Uuid::from_u128(
        0x5349_4d00_0000_0000_0000_0000_0001_9000 + sequence,
    ))
}

async fn await_private_execution_started(
    supervisor: &RuntimeSupervisor,
    timeout: Duration,
) -> Result<bool, RuntimeSupervisorError> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(RuntimeSupervisorError::Timeout {
                stage: "private execution start",
            });
        }
        let envelope = supervisor.next_event(remaining).await?;
        if matches!(
            envelope.message,
            WorkerMessage::Lifecycle {
                event: sim_types::WorkerLifecycleEvent::ExecutionStarted
            }
        ) {
            return Ok(true);
        }
    }
}

async fn await_output_prepared(
    supervisor: &RuntimeSupervisor,
    presentation: &mut ExecutionPresentationService,
    evidence: &mut PhaseEvidence,
    timeout: Duration,
) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("timed out waiting for SaveImage output preparation".into());
        }
        let envelope = supervisor.next_event(remaining).await?;
        let WorkerMessage::Event { event } = envelope.message else {
            continue;
        };
        let Ok(worker_event) = postcard::from_bytes::<NativeImageWorkerEvent>(&event) else {
            continue;
        };
        match worker_event {
            NativeImageWorkerEvent::Progress { progress } => {
                record_progress(presentation, evidence, progress)?;
                if evidence.output_prepared {
                    return Ok(());
                }
            }
            NativeImageWorkerEvent::Completed { .. } => {
                return Err("native execution completed before the crash failpoint".into());
            }
            NativeImageWorkerEvent::Failed { message, cancelled } => {
                return Err(format!(
                    "native execution failed before crash injection (cancelled={cancelled}): {message}"
                )
                .into());
            }
            NativeImageWorkerEvent::BackendUnavailable { unavailable } => {
                return Err(format!(
                    "native backend became unavailable before crash injection: {unavailable}"
                )
                .into());
            }
        }
    }
}

fn record_progress(
    presentation: &mut ExecutionPresentationService,
    evidence: &mut PhaseEvidence,
    progress: NativeImageWorkerProgress,
) -> Result<(), Box<dyn Error>> {
    let kind = match &progress.kind {
        NativeImageWorkerProgressKind::Started => {
            evidence.started = true;
            let state = presentation
                .snapshot(progress.profile_id)?
                .attempts
                .iter()
                .find(|attempt| attempt.attempt_id == progress.attempt_id)
                .map(|attempt| attempt.state)
                .ok_or("worker progress referenced an unknown attempt")?;
            (state == AttemptState::Queued).then_some(AttemptEventKind::Started)
        }
        NativeImageWorkerProgressKind::Progress { completed, total } => {
            if let Some(node_id) = &progress.node_id {
                match node_id.0.as_str() {
                    "1" => evidence.load_image_completed = true,
                    "2" => evidence.scale_tensor_completed = true,
                    "3" => evidence.invert_tensor_completed = true,
                    _ => {}
                }
            }
            Some(AttemptEventKind::Progress {
                completed: *completed,
                total: *total,
            })
        }
        NativeImageWorkerProgressKind::OutputPrepared { transaction_id } => {
            evidence.prepared_proposal_ids.insert(*transaction_id);
            match progress.node_id.as_ref().map(|node_id| node_id.0.as_str()) {
                Some("4") => evidence.preview_prepared = true,
                Some("5") => evidence.output_prepared = true,
                _ => {}
            }
            Some(AttemptEventKind::OutputPrepared {
                transaction_id: *transaction_id,
            })
        }
        NativeImageWorkerProgressKind::CacheHit => Some(AttemptEventKind::CacheHit),
    };
    if let Some(kind) = kind {
        apply_canonical_kind(
            presentation,
            progress.profile_id,
            progress.prompt_id,
            progress.attempt_id,
            progress.node_id,
            kind,
        )?;
    }
    Ok(())
}

fn apply_canonical_kind(
    presentation: &mut ExecutionPresentationService,
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: Option<sim_types::NodeId>,
    kind: AttemptEventKind,
) -> Result<(), Box<dyn Error>> {
    let at = presentation
        .snapshot(profile_id)?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == attempt_id)
        .map(|attempt| attempt.created_at)
        .ok_or("canonical presentation omitted the worker attempt")?;
    presentation
        .apply_actuator_event(profile_id, prompt_id, attempt_id, node_id, kind, None, at)?;
    Ok(())
}

async fn await_completed(
    supervisor: &RuntimeSupervisor,
    presentation: &mut ExecutionPresentationService,
    recovery_journal: &mut RecoveryJournal,
    assets: &SharedAssetService,
    roots: &AssetRoots,
    timeout: Duration,
) -> Result<CompletedRetry, Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    let mut evidence = PhaseEvidence::default();
    let mut proposals = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("timed out waiting for explicit retry completion".into());
        }
        let envelope = supervisor.next_event(remaining).await?;
        let worker_event = match envelope.message {
            WorkerMessage::OutputProposal { proposal } => {
                proposals.push(NativeImageOutputProposal::from_worker_proposal(proposal)?);
                continue;
            }
            WorkerMessage::Event { event } => {
                let Ok(worker_event) = postcard::from_bytes::<NativeImageWorkerEvent>(&event)
                else {
                    continue;
                };
                worker_event
            }
            _ => continue,
        };
        match worker_event {
            NativeImageWorkerEvent::Progress { progress } => {
                record_progress(presentation, &mut evidence, progress)?;
            }
            NativeImageWorkerEvent::Completed { result } => {
                let proposal_ids = proposals
                    .iter()
                    .map(NativeImageOutputProposal::proposal_id)
                    .collect::<Vec<_>>();
                if proposal_ids != result.output_proposal_ids {
                    return Err("explicit retry terminal result did not bind its proposals".into());
                }
                if count_png_files(roots)? != 0 {
                    return Err("worker published files before host commit".into());
                }
                let canonical = proposals
                    .iter()
                    .map(|proposal| proposal.output().clone())
                    .collect::<Vec<_>>();
                let authorization = authorize_native_output_committer(&roots.profile_id)?;
                let mut committer = OutputCommitter::open(roots.clone())?;
                let scope = OutputExecutionScope {
                    profile_id: result.report.profile_id,
                    prompt_id: result.report.prompt_id,
                    attempt_id: result.report.attempt_id,
                };
                let cancellation =
                    presentation.cancellation_token(scope.profile_id, scope.attempt_id)?;
                let mut assets = assets
                    .lock()
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                let receipts = committer.commit_scoped_proposal_batch_and_register_now(
                    &scope,
                    &canonical,
                    &mut assets,
                    &authorization,
                    &cancellation,
                )?;
                for receipt in &receipts {
                    recovery_journal.record_output_receipt(
                        scope.profile_id,
                        scope.prompt_id,
                        scope.attempt_id,
                        receipt,
                    )?;
                }
                apply_canonical_kind(
                    presentation,
                    scope.profile_id,
                    scope.prompt_id,
                    scope.attempt_id,
                    None,
                    AttemptEventKind::Succeeded,
                )?;
                return Ok(CompletedRetry {
                    result,
                    proposals,
                    receipts,
                });
            }
            NativeImageWorkerEvent::Failed { message, cancelled } => {
                return Err(
                    format!("explicit retry failed (cancelled={cancelled}): {message}").into(),
                );
            }
            NativeImageWorkerEvent::BackendUnavailable { unavailable } => {
                return Err(format!("explicit retry backend unavailable: {unavailable}").into());
            }
        }
    }
}

fn count_png_files(roots: &AssetRoots) -> Result<usize, Box<dyn Error>> {
    let mut count = 0;
    let mut pending = vec![
        roots.test_root_path(AssetNamespace::Output)?.to_path_buf(),
        roots
            .test_root_path(AssetNamespace::Temporary)?
            .to_path_buf(),
    ];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file()
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
    fixture: &NativeRecoveryFixture,
    cases: &BTreeMap<&str, bool>,
) -> Result<(), Box<dyn Error>> {
    let directory = target_directory()?.join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let artifact = json!({
        "validation_id": "VAL-RECOVERY-003",
        "validation": "VAL-RECOVERY-003",
        "scope": "native-image-worker-crash-recovery",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "worker_binary": "sim_native_image_worker_fixture",
            "process_ownership": "owned_process_tree",
            "python_or_javascript_runtime": false
        },
        "fixture_digests": {
            "input_declaration_sha256": format!("{:x}", Sha256::digest(INPUT_FIXTURE)),
            "input_png_sha256": format!("{:x}", Sha256::digest(&fixture.input_png)),
            "workflow_sha256": format!("{:x}", Sha256::digest(WORKFLOW_FIXTURE)),
        },
        "phase_coverage": [
            {"phase": "node", "status": "passed", "evidence": "LoadImage completion observed before worker loss"},
            {"phase": "tensor", "status": "passed", "evidence": "ImageScale and ImageInvert completion observed before worker loss; replacement worker was cold"},
            {"phase": "preview", "status": "passed", "evidence": "PreviewImage OutputPrepared observed before worker loss and reconciled as interrupted"},
            {"phase": "output", "status": "passed", "evidence": "SaveImage OutputPrepared observed before worker loss and reconciled as interrupted"},
            {"phase": "model", "status": "not_applicable", "reason": "Task 19's exact five-node native image registry contains no model node; model crash phases belong to their dedicated later native-model tasks"},
            {"phase": "sampler", "status": "not_applicable", "reason": "Task 19's exact five-node native image registry contains no sampler node; sampler crash phases belong to their dedicated later native-diffusion tasks"}
        ],
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0,
            "not_applicable": 2
        },
        "cases": cases,
        "skipped": [],
        "not_applicable": [
            {"phase": "model", "reason": "unavailable in the normative Task 19 five-node native image registry"},
            {"phase": "sampler", "reason": "unavailable in the normative Task 19 five-node native image registry"}
        ]
    });
    fs::write(
        directory.join("val-recovery-003.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}
