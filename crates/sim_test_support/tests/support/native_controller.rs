use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::json;
use sim_media::{PngLimits, decode_png, encode_png_frame};
use sim_runtime::{
    AssetNamespace, AssetRoots, AssetService, AttemptEvent, AttemptEventKind, AttemptState,
    ExecutionCommandAck, ExecutionCommandOutcome, ExecutionControlCommand,
    ExecutionControlCommandKind, ExecutionController, ExecutionDataSource, ExecutionEventBus,
    ExecutionFailure, ExecutionOutputAvailability, ExecutionPresentationService,
    ExecutionSnapshotStatus, NATIVE_IMAGE_REGISTRY_VERSION, NativeExecutionController,
    NativeExecutionControllerConfig, OutputMediaKind, RetryPromptSource,
    SharedExecutionPresentationService, SupervisorPolicy, WorkerLaunchConfig,
    compile_native_image_workflow,
};
use sim_types::{AttemptId, ProfileId, PromptId, RequestId, WorkerId};
use uuid::Uuid;

const INPUT_FIXTURE: &[u8] = include_bytes!("../../fixtures/native_image/input.json");
const WORKFLOW_FIXTURE: &[u8] = include_bytes!("../../fixtures/native_image/workflow.json");
const MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const PROFILE_UUID: Uuid = Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1910);
const EXECUTION_DELAY_MILLIS: u64 = 1_500;

#[derive(Deserialize)]
struct InputFixture {
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    pixels_bhwc: Vec<f32>,
}

struct NativeControllerFixture {
    _directory: tempfile::TempDir,
    roots: AssetRoots,
    worker_directory: PathBuf,
}

impl NativeControllerFixture {
    fn new() -> Result<Self, Box<dyn Error>> {
        let input: InputFixture = serde_json::from_slice(INPUT_FIXTURE)?;
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
        fs::write(
            roots
                .test_root_path(AssetNamespace::Input)?
                .join("fixture.png"),
            input_bytes,
        )?;
        Ok(Self {
            _directory: directory,
            roots,
            worker_directory,
        })
    }

    fn launch_config(&self) -> WorkerLaunchConfig {
        let mut config = WorkerLaunchConfig::new(
            env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
            ProfileId(PROFILE_UUID),
            WorkerId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_1911)),
            NATIVE_IMAGE_REGISTRY_VERSION,
            MEMORY_LIMIT_BYTES,
        );
        config.working_directory = Some(self.worker_directory.clone());
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

pub(crate) fn run_native_controller_e2e() -> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
    let mut cases = BTreeMap::new();
    let fixture = NativeControllerFixture::new()?;
    let profile_id = ProfileId(PROFILE_UUID);
    let event_bus = ExecutionEventBus::new(512)?;
    let event_receiver = event_bus.subscribe();
    let assets = Arc::new(Mutex::new(AssetService::open(fixture.roots.clone())?));
    let mut presentation_service = ExecutionPresentationService::new(32)?;
    presentation_service.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation_service);
    let controller = NativeExecutionController::start(
        NativeExecutionControllerConfig::new(
            assets,
            presentation.clone(),
            fixture.launch_config(),
            true,
        )?,
        event_bus,
    )?;

    let mut initial_plan = delayed_plan(prompt_id(1))?;
    let initial_attempt = dispatch_assigned(
        &presentation,
        controller.as_ref(),
        command(
            1,
            ExecutionControlCommandKind::Queue {
                plan: initial_plan.clone(),
                priority: 0,
                front: false,
            },
        ),
    )?;
    let mut initial_events = wait_for_event(
        &event_receiver,
        &presentation,
        "initial Started",
        profile_id,
        initial_attempt,
        Duration::from_secs(10),
        |event| matches!(event.kind, AttemptEventKind::Started),
    )?;
    wait_for_controller_handoff(
        controller.as_ref(),
        "initial Started",
        initial_attempt,
        10_000,
    )?;

    let pending_attempt = dispatch_assigned(
        &presentation,
        controller.as_ref(),
        command(
            2,
            ExecutionControlCommandKind::Queue {
                plan: delayed_plan(prompt_id(2))?,
                priority: 0,
                front: false,
            },
        ),
    )?;
    let front_attempt = dispatch_assigned(
        &presentation,
        controller.as_ref(),
        command(
            3,
            ExecutionControlCommandKind::Queue {
                plan: delayed_plan(prompt_id(3))?,
                priority: 0,
                front: true,
            },
        ),
    )?;
    assert_eq!(
        presentation
            .snapshot(profile_id)?
            .queue
            .iter()
            .map(|queued| queued.attempt_id)
            .collect::<Vec<_>>(),
        vec![front_attempt, pending_attempt]
    );
    dispatch_accepted(
        &presentation,
        controller.as_ref(),
        command(
            4,
            ExecutionControlCommandKind::Reorder {
                attempt_id: pending_attempt,
                position: 0,
            },
        ),
    )?;
    assert_eq!(
        presentation
            .snapshot(profile_id)?
            .queue
            .iter()
            .map(|queued| queued.attempt_id)
            .collect::<Vec<_>>(),
        vec![pending_attempt, front_attempt]
    );
    dispatch_accepted(
        &presentation,
        controller.as_ref(),
        command(
            5,
            ExecutionControlCommandKind::ClearPending {
                reason: "Task 19 controller clear-pending validation".to_owned(),
            },
        ),
    )?;
    assert!(presentation.snapshot(profile_id)?.queue.is_empty());
    cases.insert(
        "native_controller_acknowledges_queue_front_reorder_and_clear_pending",
        true,
    );

    dispatch_accepted(
        &presentation,
        controller.as_ref(),
        command(
            6,
            ExecutionControlCommandKind::Cancel {
                attempt_id: initial_attempt,
                reason: "Task 19 running cancellation".to_owned(),
            },
        ),
    )?;
    initial_events.extend(wait_for_event(
        &event_receiver,
        &presentation,
        "initial Cancelled",
        profile_id,
        initial_attempt,
        Duration::from_secs(10),
        |event| matches!(event.kind, AttemptEventKind::Cancelled),
    )?);
    assert_monotonic_attempt_events(initial_attempt, &initial_events)?;
    assert_eq!(
        attempt_state(&presentation, profile_id, initial_attempt)?,
        AttemptState::Cancelled
    );
    assert_eq!(
        count_png_files(fixture.roots.test_root_path(AssetNamespace::Output)?)?,
        0
    );
    assert_eq!(
        count_png_files(fixture.roots.test_root_path(AssetNamespace::Temporary)?)?,
        0
    );
    assert!(
        receive_event(&event_receiver, Duration::from_millis(300))?.is_none(),
        "cancelled attempt emitted a late result"
    );
    cases.insert(
        "native_controller_running_cancel_rejects_late_results",
        true,
    );

    let interrupted_attempt = dispatch_assigned(
        &presentation,
        controller.as_ref(),
        command(
            7,
            ExecutionControlCommandKind::Retry {
                attempt_id: initial_attempt,
                source: RetryPromptSource::OriginalPrompt,
                replacement_plan: None,
            },
        ),
    )?;
    assert_ne!(interrupted_attempt, initial_attempt);
    let mut interrupted_events = wait_for_event(
        &event_receiver,
        &presentation,
        "retry Started",
        profile_id,
        interrupted_attempt,
        Duration::from_secs(10),
        |event| matches!(event.kind, AttemptEventKind::Started),
    )?;
    wait_for_controller_handoff(
        controller.as_ref(),
        "retry Started",
        interrupted_attempt,
        20_000,
    )?;
    dispatch_accepted(
        &presentation,
        controller.as_ref(),
        command(
            8,
            ExecutionControlCommandKind::Interrupt {
                attempt_id: interrupted_attempt,
                reason: "Task 19 running interrupt".to_owned(),
            },
        ),
    )?;
    interrupted_events.extend(wait_for_event(
        &event_receiver,
        &presentation,
        "retry Interrupted",
        profile_id,
        interrupted_attempt,
        Duration::from_secs(10),
        |event| matches!(event.kind, AttemptEventKind::Interrupted { .. }),
    )?);
    assert_monotonic_attempt_events(interrupted_attempt, &interrupted_events)?;
    assert!(matches!(
        interrupted_events.last().map(|event| &event.kind),
        Some(AttemptEventKind::Interrupted { reason })
            if reason == "Task 19 running interrupt"
    ));
    assert_eq!(
        attempt_state(&presentation, profile_id, interrupted_attempt)?,
        AttemptState::Interrupted
    );
    cases.insert(
        "native_controller_running_interrupt_and_retry_are_attempt_scoped",
        true,
    );

    initial_plan.extra_data.remove("zed_native_delay_millis");
    let completed_attempt = dispatch_assigned(
        &presentation,
        controller.as_ref(),
        command(
            9,
            ExecutionControlCommandKind::Retry {
                attempt_id: interrupted_attempt,
                source: RetryPromptSource::CurrentWorkflow {
                    revision: "task19-controller-e2e-final".to_owned(),
                },
                replacement_plan: Some(initial_plan),
            },
        ),
    )?;
    assert_ne!(completed_attempt, interrupted_attempt);
    let completed_events = wait_for_event(
        &event_receiver,
        &presentation,
        "retry Succeeded",
        profile_id,
        completed_attempt,
        Duration::from_secs(15),
        |event| matches!(event.kind, AttemptEventKind::Succeeded),
    )?;
    assert_monotonic_attempt_events(completed_attempt, &completed_events)?;
    assert_typed_outputs(&completed_events)?;
    assert_eq!(
        attempt_state(&presentation, profile_id, completed_attempt)?,
        AttemptState::Succeeded
    );
    assert_eq!(
        count_png_files(fixture.roots.test_root_path(AssetNamespace::Output)?)?,
        1
    );
    assert_eq!(
        count_png_files(fixture.roots.test_root_path(AssetNamespace::Temporary)?)?,
        1
    );
    cases.insert(
        "native_controller_events_are_monotonic_and_outputs_are_typed",
        true,
    );

    drop(controller);
    std::thread::sleep(Duration::from_millis(200));
    Ok(cases)
}

fn delayed_plan(prompt_id: PromptId) -> Result<sim_runtime::CompiledPlan, Box<dyn Error>> {
    let mut plan = compile_native_image_workflow(WORKFLOW_FIXTURE, &BTreeSet::new())?;
    plan.prompt_id = prompt_id;
    plan.extra_data.insert(
        "zed_native_delay_millis".to_owned(),
        json!(EXECUTION_DELAY_MILLIS),
    );
    Ok(plan)
}

fn prompt_id(sequence: u128) -> PromptId {
    PromptId(Uuid::from_u128(
        0x5349_4d00_0000_0000_0000_0000_0001_0000 + sequence,
    ))
}

fn command(sequence: u128, kind: ExecutionControlCommandKind) -> ExecutionControlCommand {
    ExecutionControlCommand {
        request_id: RequestId(Uuid::from_u128(
            0x5349_4d00_0000_0000_0000_0000_0002_0000 + sequence,
        )),
        profile_id: ProfileId(PROFILE_UUID),
        expected_revision: None,
        kind,
    }
}

fn dispatch_assigned(
    presentation: &SharedExecutionPresentationService,
    controller: &dyn ExecutionController,
    command: ExecutionControlCommand,
) -> Result<AttemptId, Box<dyn Error>> {
    let acknowledgement = smol::block_on(presentation.dispatch_durable(command, controller))?;
    match acknowledgement.outcome {
        ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(attempt_id),
        } => Ok(attempt_id),
        outcome => Err(format!("command did not receive an assigned attempt: {outcome:?}").into()),
    }
}

fn wait_for_controller_handoff(
    controller: &NativeExecutionController,
    phase: &'static str,
    attempt_id: AttemptId,
    first_sequence: u128,
) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let timeout = Duration::from_secs(10);
    let deadline = started + timeout;
    let mut sequence = first_sequence;
    let mut observation = ControllerHandoffObservation {
        phase,
        attempt_id,
        first_sequence,
        admitted_probe_count: 0,
        reservation_rejection_count: 0,
        saturation_rejection_count: 0,
        activation_commit_invoked: false,
        last_rejection: None,
    };
    // Hold the first probe so a healthy consumer cannot drain faster than
    // this producer fills the queue.
    let reservation = loop {
        let probe = command(sequence, ExecutionControlCommandKind::ClearHistory);
        match controller.prepare(&probe, None) {
            Ok(activation) => {
                observation.admitted_probe_count =
                    observation.admitted_probe_count.saturating_add(1);
                break activation;
            }
            Err(failure) if failure.code == "native_controller_backpressure" => {
                observation.reservation_rejection_count =
                    observation.reservation_rejection_count.saturating_add(1);
                observation.last_rejection = Some(failure);
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(failure) => {
                observation.last_rejection = Some(failure);
                return Err(observation
                    .diagnostic(
                        "probe failed",
                        "prepare_reservation",
                        sequence,
                        timeout,
                        started.elapsed(),
                    )
                    .into());
            }
        }
        if Instant::now() >= deadline {
            return Err(observation
                .diagnostic(
                    "timed out",
                    "prepare_reservation",
                    sequence,
                    timeout,
                    started.elapsed(),
                )
                .into());
        }
    };
    let mut reservation = Some(reservation);
    sequence = sequence
        .checked_add(1)
        .ok_or("controller handoff probe sequence overflowed")?;
    let mut observed_backpressure = false;
    loop {
        let probe = command(sequence, ExecutionControlCommandKind::ClearHistory);
        match controller.accept(&probe, None) {
            Ok(()) if observed_backpressure => return Ok(()),
            Ok(()) => {
                observation.admitted_probe_count =
                    observation.admitted_probe_count.saturating_add(1);
                sequence = sequence
                    .checked_add(1)
                    .ok_or("controller handoff probe sequence overflowed")?;
            }
            Err(failure) if failure.code == "native_controller_backpressure" => {
                observation.saturation_rejection_count =
                    observation.saturation_rejection_count.saturating_add(1);
                observation.last_rejection = Some(failure);
                if let Some(activation) = reservation.take() {
                    observation.activation_commit_invoked = true;
                    activation.commit();
                }
                observed_backpressure = true;
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(failure) => {
                observation.last_rejection = Some(failure);
                let stage = if observed_backpressure {
                    "await_acceptance_after_commit"
                } else {
                    "await_saturation"
                };
                return Err(observation
                    .diagnostic("probe failed", stage, sequence, timeout, started.elapsed())
                    .into());
            }
        }
        if Instant::now() >= deadline {
            let stage = if observed_backpressure {
                "await_acceptance_after_commit"
            } else {
                "await_saturation"
            };
            return Err(observation
                .diagnostic("timed out", stage, sequence, timeout, started.elapsed())
                .into());
        }
    }
}

struct ControllerHandoffObservation {
    phase: &'static str,
    attempt_id: AttemptId,
    first_sequence: u128,
    admitted_probe_count: u128,
    reservation_rejection_count: u128,
    saturation_rejection_count: u128,
    activation_commit_invoked: bool,
    last_rejection: Option<ExecutionFailure>,
}

impl ControllerHandoffObservation {
    fn diagnostic(
        &self,
        disposition: &str,
        stage: &str,
        sequence: u128,
        timeout: Duration,
        elapsed: Duration,
    ) -> String {
        format!(
            "native controller command handoff {disposition}: phase={:?}; stage={stage}; profile_id={:?}; attempt_id={:?}; first_sequence={}; current_sequence={sequence}; admitted_probe_count={}; reservation_rejection_count={}; saturation_rejection_count={}; activation_commit_invoked={}; timeout={timeout:?}; elapsed={elapsed:?}; original_rejection={:?}",
            self.phase,
            ProfileId(PROFILE_UUID),
            self.attempt_id,
            self.first_sequence,
            self.admitted_probe_count,
            self.reservation_rejection_count,
            self.saturation_rejection_count,
            self.activation_commit_invoked,
            self.last_rejection,
        )
    }
}

fn dispatch_accepted(
    presentation: &SharedExecutionPresentationService,
    controller: &dyn ExecutionController,
    command: ExecutionControlCommand,
) -> Result<ExecutionCommandAck, Box<dyn Error>> {
    let acknowledgement = smol::block_on(presentation.dispatch_durable(command, controller))?;
    if !matches!(
        acknowledgement.outcome,
        ExecutionCommandOutcome::Accepted { .. }
    ) {
        return Err(format!("controller rejected command: {acknowledgement:?}").into());
    }
    Ok(acknowledgement)
}

fn wait_for_event(
    receiver: &async_channel::Receiver<AttemptEvent>,
    presentation: &SharedExecutionPresentationService,
    phase: &str,
    profile_id: ProfileId,
    attempt_id: AttemptId,
    timeout: Duration,
    predicate: impl Fn(&AttemptEvent) -> bool,
) -> Result<Vec<AttemptEvent>, Box<dyn Error>> {
    if phase.trim().is_empty() {
        return Err("native controller event wait phase must be non-empty".into());
    }
    let deadline = Instant::now() + timeout;
    let mut events = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(wait_for_event_diagnostic(
                "timed out",
                phase,
                profile_id,
                attempt_id,
                timeout,
                &events,
                presentation,
            )
            .into());
        }
        let event = match receive_event(receiver, remaining) {
            Ok(Some(event)) => event,
            Ok(None) => {
                return Err(wait_for_event_diagnostic(
                    "timed out",
                    phase,
                    profile_id,
                    attempt_id,
                    timeout,
                    &events,
                    presentation,
                )
                .into());
            }
            Err(error)
                if error
                    .downcast_ref::<NativeControllerEventBusClosed>()
                    .is_some() =>
            {
                return Err(wait_for_event_diagnostic(
                    "event channel closed",
                    phase,
                    profile_id,
                    attempt_id,
                    timeout,
                    &events,
                    presentation,
                )
                .into());
            }
            Err(error) => return Err(error),
        };
        assert_event_is_canonical(presentation, &event)?;
        if event.attempt_id == attempt_id {
            let matched = predicate(&event);
            events.push(event);
            if matched {
                return Ok(events);
            }
        }
    }
}

fn wait_for_event_diagnostic(
    disposition: &str,
    phase: &str,
    profile_id: ProfileId,
    attempt_id: AttemptId,
    timeout: Duration,
    events: &[AttemptEvent],
    presentation: &SharedExecutionPresentationService,
) -> String {
    let observed_events = events
        .iter()
        .map(|event| (event.sequence, &event.kind))
        .collect::<Vec<_>>();
    let canonical = match presentation.snapshot(profile_id) {
        Ok(snapshot) => {
            let attempt = snapshot
                .attempts
                .iter()
                .find(|attempt| attempt.attempt_id == attempt_id);
            format!("attempt={attempt:#?}; snapshot={snapshot:#?}")
        }
        Err(error) => format!("snapshot lookup error: {error:?}"),
    };
    format!(
        "native controller event wait {disposition}: phase={phase:?}; profile_id={profile_id:?}; attempt_id={attempt_id:?}; timeout={timeout:?}; observed_target_events={observed_events:#?}; canonical={canonical}"
    )
}

#[derive(Debug)]
struct NativeControllerEventBusClosed;

impl std::fmt::Display for NativeControllerEventBusClosed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("native controller event bus closed")
    }
}

impl Error for NativeControllerEventBusClosed {}

fn assert_event_is_canonical(
    presentation: &SharedExecutionPresentationService,
    event: &AttemptEvent,
) -> Result<(), Box<dyn Error>> {
    let snapshot = presentation.snapshot(event.profile_id)?;
    let attempt = snapshot
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == event.attempt_id)
        .ok_or_else(|| {
            format!(
                "event-bus attempt {:?} is absent from the canonical snapshot",
                event.attempt_id
            )
        })?;
    let expected_event_count = usize::try_from(
        event
            .sequence
            .checked_add(1)
            .ok_or("event sequence exceeds the canonical counter")?,
    )?;
    if attempt
        .last_sequence
        .is_none_or(|sequence| sequence < event.sequence)
        || attempt.canonical_event_count < expected_event_count
    {
        return Err(format!(
            "event-bus event {:?}/{} was not applied by the canonical presentation service",
            event.attempt_id, event.sequence
        )
        .into());
    }
    Ok(())
}

fn receive_event(
    receiver: &async_channel::Receiver<AttemptEvent>,
    timeout: Duration,
) -> Result<Option<AttemptEvent>, Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    loop {
        match receiver.try_recv() {
            Ok(event) => return Ok(Some(event)),
            Err(async_channel::TryRecvError::Closed) => {
                return Err(NativeControllerEventBusClosed.into());
            }
            Err(async_channel::TryRecvError::Empty) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Ok(None);
                }
                std::thread::sleep(remaining.min(Duration::from_millis(5)));
            }
        }
    }
}

fn attempt_state(
    presentation: &SharedExecutionPresentationService,
    profile_id: ProfileId,
    attempt_id: AttemptId,
) -> Result<AttemptState, Box<dyn Error>> {
    presentation
        .snapshot(profile_id)?
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == attempt_id)
        .map(|attempt| attempt.state)
        .ok_or_else(|| format!("attempt {attempt_id:?} is absent from the snapshot").into())
}

fn assert_monotonic_attempt_events(
    attempt_id: AttemptId,
    events: &[AttemptEvent],
) -> Result<(), Box<dyn Error>> {
    if events.first().map(|event| event.sequence) != Some(0) {
        return Err(format!("attempt {attempt_id:?} did not start at sequence zero").into());
    }
    for (expected, event) in events.iter().enumerate() {
        let expected = u64::try_from(expected)?;
        if event.attempt_id != attempt_id || event.sequence != expected {
            return Err(format!(
                "attempt {attempt_id:?} event sequence mismatch: expected {expected}, got {:?}/{}",
                event.attempt_id, event.sequence
            )
            .into());
        }
    }
    Ok(())
}

fn assert_typed_outputs(events: &[AttemptEvent]) -> Result<(), Box<dyn Error>> {
    let previews = events
        .iter()
        .filter_map(|event| match &event.kind {
            AttemptEventKind::Preview { preview } => Some(preview),
            _ => None,
        })
        .collect::<Vec<_>>();
    if previews.len() != 1 {
        return Err(format!("expected one typed preview, got {}", previews.len()).into());
    }
    let preview = previews[0];
    let decoded_preview = decode_png(&preview.encoded_bytes, PngLimits::default())?;
    assert_eq!(preview.node_id.0, "4");
    assert_eq!(preview.media_kind, OutputMediaKind::Image);
    assert_eq!(preview.media_type, "image/png");
    assert_eq!((preview.width, preview.height), (Some(4), Some(2)));
    assert_eq!((decoded_preview.width, decoded_preview.height), (4, 2));

    let outputs = events
        .iter()
        .filter_map(|event| match &event.kind {
            AttemptEventKind::OutputAvailable { output } => Some(output),
            _ => None,
        })
        .collect::<Vec<_>>();
    if outputs.len() != 2 {
        return Err(format!(
            "expected preview and final typed outputs, got {}",
            outputs.len()
        )
        .into());
    }
    let temporary = outputs
        .iter()
        .find(|output| output.storage_type.as_deref() == Some("temp"))
        .ok_or("typed temporary output is absent")?;
    let final_output = outputs
        .iter()
        .find(|output| output.storage_type.as_deref() == Some("output"))
        .ok_or("typed final output is absent")?;
    for (output, expected_node, expected_prefix) in [
        (temporary, "4", "zed-asset://temp/"),
        (final_output, "5", "zed-asset://output/"),
    ] {
        assert_eq!(output.node_id.0, expected_node);
        assert_eq!(output.media_kind, OutputMediaKind::Image);
        assert_eq!(output.media_type, "image/png");
        assert!(
            output
                .view_reference
                .as_deref()
                .is_some_and(|reference| { reference.starts_with(expected_prefix) })
        );
        assert_eq!(output.view_reference, output.download_reference);
        assert!(output.metadata.contains_key("sha256"));
        match &output.availability {
            ExecutionOutputAvailability::Ready {
                reference,
                byte_length,
            } => {
                assert!(reference.starts_with(expected_prefix));
                assert!(*byte_length > 0);
            }
            availability => {
                return Err(format!("output was not ready: {availability:?}").into());
            }
        }
    }
    Ok(())
}

fn count_png_files(root: &std::path::Path) -> Result<usize, Box<dyn Error>> {
    let mut count = 0_usize;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            count = count
                .checked_add(count_png_files(&entry.path())?)
                .ok_or("PNG count overflowed")?;
        } else if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "png")
        {
            count = count.checked_add(1).ok_or("PNG count overflowed")?;
        }
    }
    Ok(count)
}
