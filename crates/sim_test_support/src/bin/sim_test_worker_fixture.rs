use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    io::{self, Read, Write},
    path::PathBuf,
    time::Duration,
};

use sim_runtime::{
    PluginAuthorizationVerifier, RuntimeSupervisor, SupervisorPolicy, WorkerLaunchConfig,
};
use sim_tensor::{BackendCapabilityMatrix, CpuBackend};
use sim_types::{
    MAX_WORKER_FRAME_BYTES, ProfileId, WORKER_PROTOCOL_VERSION, WorkerEnvelope, WorkerId,
    WorkerLifecycleEvent, WorkerMessage, WorkerRegistryDeploymentAck, decode_worker_frame,
    encode_worker_frame,
};
use uuid::Uuid;

fn main() -> Result<(), Box<dyn Error>> {
    if env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("--orphan-parent")) {
        return run_orphan_parent();
    }
    validate_launch_environment()?;
    parse_memory_limit()?;

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut expected_input_sequence = 0_u64;
    let mut next_output_sequence = 0_u64;
    let mut pending_registry = None;

    loop {
        let envelope = read_frame(&mut input)?;
        if envelope.version != WORKER_PROTOCOL_VERSION {
            return Err(format!("unsupported protocol version {}", envelope.version).into());
        }
        if envelope.sequence != expected_input_sequence {
            return Err(format!(
                "expected input sequence {expected_input_sequence}, received {}",
                envelope.sequence
            )
            .into());
        }
        expected_input_sequence = expected_input_sequence
            .checked_add(1)
            .ok_or("input sequence overflow")?;

        let responses = match envelope.message {
            WorkerMessage::Hello { ref backend } => {
                let requested = BackendCapabilityMatrix::try_from(backend.clone())?;
                let accepted = CpuBackend::capability_matrix().negotiate(&requested)?;
                if accepted.supported().is_empty() {
                    return Err("fixture requires common CPU backend support".into());
                }
                vec![
                    WorkerMessage::HelloAck {
                        accepted_backend: accepted.to_worker_capabilities()?,
                    },
                    WorkerMessage::Ready,
                ]
            }
            WorkerMessage::Execute { ref plan } => {
                if plan.is_empty() {
                    return Err("execution plan cannot be empty".into());
                }
                vec![WorkerMessage::Lifecycle {
                    event: WorkerLifecycleEvent::ExecutionStarted,
                }]
            }
            WorkerMessage::Cancel { ref reason } => vec![WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::CancellationRequested {
                    reason: reason.clone(),
                },
            }],
            WorkerMessage::Heartbeat => vec![WorkerMessage::Heartbeat],
            WorkerMessage::Shutdown => vec![WorkerMessage::Shutdown],
            WorkerMessage::RegistryDeploymentBegin { deployment } => {
                if pending_registry.is_some() {
                    return Err("fixture received overlapping registry deployments".into());
                }
                pending_registry = Some((
                    deployment.generation(),
                    deployment.registry_digest_sha256().clone(),
                    u32::try_from(deployment.components().len())?,
                ));
                Vec::new()
            }
            WorkerMessage::RegistryDeploymentChunk { chunk } => {
                let Some((generation, _, _)) = pending_registry.as_ref() else {
                    return Err("fixture received a registry chunk before begin".into());
                };
                if chunk.generation() != *generation {
                    return Err("fixture received a foreign registry chunk".into());
                }
                Vec::new()
            }
            WorkerMessage::RegistryDeploymentCommit { commit } => {
                let Some((generation, digest, component_count)) = pending_registry.take() else {
                    return Err("fixture received a registry commit before begin".into());
                };
                if commit.generation() != generation || commit.registry_digest_sha256() != &digest {
                    return Err("fixture received a stale registry commit".into());
                }
                vec![WorkerMessage::RegistryDeploymentAck {
                    acknowledgement: WorkerRegistryDeploymentAck::new(
                        generation,
                        digest,
                        component_count,
                    )?,
                }]
            }
            WorkerMessage::ExecutePlugin { .. }
            | WorkerMessage::PluginCapabilityResponse { .. }
            | WorkerMessage::ProviderStreamResponse { .. }
            | WorkerMessage::ProviderV2ProposalFinalization { .. }
            | WorkerMessage::ModelSourceResponse { .. } => {
                return Err("fixture does not implement the component worker protocol".into());
            }
            WorkerMessage::HelloAck { .. }
            | WorkerMessage::Ready
            | WorkerMessage::Event { .. }
            | WorkerMessage::OutputProposal { .. }
            | WorkerMessage::Lifecycle { .. }
            | WorkerMessage::RegistryDeploymentAck { .. }
            | WorkerMessage::RegistryDeploymentRejected { .. }
            | WorkerMessage::PluginCapabilityRequest { .. }
            | WorkerMessage::ProviderStreamRequest { .. }
            | WorkerMessage::ProviderV2ProposalFinalizationAck { .. }
            | WorkerMessage::ModelSourceRequest { .. }
            | WorkerMessage::PluginResult { .. }
            | WorkerMessage::Fatal { .. } => {
                return Err("supervisor sent a worker-only message".into());
            }
        };

        let shutdown = responses
            .iter()
            .any(|message| matches!(message, WorkerMessage::Shutdown));
        for message in responses {
            let response = WorkerEnvelope {
                version: envelope.version,
                profile_id: envelope.profile_id,
                worker_id: envelope.worker_id,
                request_id: envelope.request_id,
                prompt_id: envelope.prompt_id,
                attempt_id: envelope.attempt_id,
                sequence: next_output_sequence,
                registry_version: envelope.registry_version.clone(),
                message,
                extensions: BTreeMap::new(),
            };
            next_output_sequence = next_output_sequence
                .checked_add(1)
                .ok_or("output sequence overflow")?;
            write_frame(&mut output, &response)?;
        }
        if shutdown {
            return Ok(());
        }
    }
}

fn run_orphan_parent() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(2);
    let worker_binary = PathBuf::from(
        arguments
            .next()
            .ok_or("orphan parent omitted worker binary")?,
    );
    let pid_file = PathBuf::from(arguments.next().ok_or("orphan parent omitted PID file")?);
    let phase = arguments
        .next()
        .ok_or("orphan parent omitted crash phase")?;
    if arguments.next().is_some() {
        return Err("orphan parent received unexpected arguments".into());
    }
    let mut config = WorkerLaunchConfig::new(
        worker_binary,
        ProfileId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_09f1)),
        WorkerId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_09f2)),
        "orphan-resilience-v1",
        1024 * 1024 * 1024,
    );
    config.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(2),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(5),
        ready_timeout: Duration::from_secs(5),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let mut supervisor = smol::block_on(RuntimeSupervisor::start(config))?;
    let worker_process_id = supervisor
        .worker_process_id()
        .ok_or("orphan parent could not observe its worker process")?;
    fs::write(&pid_file, worker_process_id.to_string())?;
    match phase.to_str() {
        Some("before-shutdown") => {}
        Some("during-shutdown") => {
            smol::block_on(supervisor.request_shutdown())?;
        }
        _ => return Err("orphan parent received an unknown crash phase".into()),
    }
    std::process::abort()
}

fn validate_launch_environment() -> Result<(), Box<dyn Error>> {
    if env::var_os("SIM_TEST_EXPECT_EMPTY_PATH").is_some_and(|value| value != "1") {
        return Err("SIM_TEST_EXPECT_EMPTY_PATH must be 1 when present".into());
    }
    if env::var_os("SIM_TEST_EXPECT_EMPTY_PATH").is_some()
        && env::var_os("PATH").is_some_and(|value| !value.is_empty())
    {
        return Err("PATH was not empty in the isolated worker".into());
    }
    if let Some(expected_root) = env::var_os("SIM_TEST_ISOLATED_ROOT") {
        let expected_root = fs::canonicalize(expected_root)?;
        let current_root = fs::canonicalize(env::current_dir()?)?;
        if current_root != expected_root {
            return Err("worker did not start in the isolated root".into());
        }
        for source_path in ["projects/comfy", "ComfyUI", "ComfyUI-Frontend"] {
            if current_root.join(source_path).exists() {
                return Err(format!("source path {source_path} exists in isolated root").into());
            }
        }
    }
    Ok(())
}

fn parse_memory_limit() -> Result<u64, Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let mut memory_limit = None;
    let mut backend_seen = false;
    let mut authorization_verifier_seen = false;
    while let Some(argument) = arguments.next() {
        if argument == "--backend" {
            let backend = arguments.next().ok_or("--backend requires a value")?;
            if backend_seen || backend != "cpu" {
                return Err("test worker requires exactly one CPU backend selection".into());
            }
            backend_seen = true;
            continue;
        }
        if argument == "--plugin-authorization-verification-key" {
            if authorization_verifier_seen {
                return Err("authorization verifier was provided more than once".into());
            }
            let value = arguments
                .next()
                .ok_or("authorization verifier requires a value")?;
            let value = value
                .to_str()
                .ok_or("authorization verifier is not UTF-8")?;
            PluginAuthorizationVerifier::from_token(value)?;
            authorization_verifier_seen = true;
            continue;
        }
        if argument != "--memory-limit-bytes" {
            return Err(format!("unexpected worker argument {argument:?}").into());
        }
        let value = arguments
            .next()
            .ok_or("--memory-limit-bytes requires a value")?;
        let value = value
            .to_str()
            .ok_or("memory limit must be UTF-8 decimal bytes")?
            .parse::<u64>()?;
        if value == 0 {
            return Err("memory limit must be non-zero".into());
        }
        memory_limit = Some(value);
    }
    if !backend_seen {
        return Err("test worker requires an explicit CPU backend selection".into());
    }
    memory_limit.ok_or_else(|| "--memory-limit-bytes was not supplied".into())
}

fn read_frame(reader: &mut impl Read) -> Result<WorkerEnvelope, Box<dyn Error>> {
    let mut prefix = [0_u8; 4];
    reader.read_exact(&mut prefix)?;
    let length = u32::from_le_bytes(prefix) as usize;
    if length > MAX_WORKER_FRAME_BYTES {
        return Err(format!("worker frame exceeds {MAX_WORKER_FRAME_BYTES} bytes").into());
    }
    let mut frame = Vec::new();
    frame.try_reserve_exact(4_usize.saturating_add(length))?;
    frame.extend_from_slice(&prefix);
    frame.resize(4 + length, 0);
    let payload = frame
        .get_mut(4..)
        .ok_or("invalid worker frame allocation")?;
    reader.read_exact(payload)?;
    Ok(decode_worker_frame(&frame)?)
}

fn write_frame(writer: &mut impl Write, envelope: &WorkerEnvelope) -> Result<(), Box<dyn Error>> {
    let frame = encode_worker_frame(envelope)?;
    writer.write_all(&frame)?;
    writer.flush()?;
    Ok(())
}
