use std::{
    collections::BTreeMap,
    future::pending,
    io::{self, Read, Write},
    sync::Arc,
    time::{Duration, Instant},
};

use sim_runtime::{
    AttemptEvent, AttemptEventKind, AttemptState, ExecutionEventBus, InputBinding,
    NativeDiffusionProvider, NativeImageExecutor, NativeImageRuntimeError, NativeImageWorkerEvent,
    NativeImageWorkerPlan, NativeImageWorkerProgress, NativeImageWorkerProgressKind,
    NativeModelSourceTransportCall, NativeModelSourceTransportHost, NativeValue,
    NativeVideoCodecWorkerServices, PluginAuthorizationVerifier, WorkerBackendSelection,
    certify_general_video_codec_package,
};
use sim_tensor::{BackendWorkspaceAuthority, CancellationToken, CpuBackend, DeviceId};
use sim_types::{
    BackendUnavailable, DeviceKind, MAX_ENCODED_PREVIEW_BYTES, MAX_WORKER_FRAME_BYTES,
    WorkerEnvelope, WorkerMessage, WorkerModelSourceError, WorkerModelSourceRequest,
    WorkerModelSourceResponse, WorkerPluginExecutionOutcome, WorkerProtocolError,
    WorkerRegistryDeploymentRejectionReason, decode_worker_frame, encode_worker_frame,
};
use thiserror::Error;

pub mod memory_modes;
pub mod memory_planner;
mod plugin_runtime;
pub mod supervisor;

pub use memory_modes::*;
pub use memory_planner::*;
pub use supervisor::*;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

pub struct WorkerModelSourceTransportCall {
    inner: NativeModelSourceTransportCall,
}

impl WorkerModelSourceTransportCall {
    pub const fn call_id(&self) -> u64 {
        self.inner.call_id()
    }

    pub fn request(&self) -> &WorkerModelSourceRequest {
        self.inner.request()
    }

    pub async fn respond(
        self,
        call_id: u64,
        response: WorkerModelSourceResponse,
    ) -> Result<(), WorkerModelSourceError> {
        self.inner.respond(call_id, response).await
    }

    pub fn revoke(&self) -> Result<(), WorkerModelSourceError> {
        self.inner.revoke()
    }
}

pub struct WorkerModelSourceTransportHost {
    inner: NativeModelSourceTransportHost,
}

impl WorkerModelSourceTransportHost {
    pub async fn receive(
        &self,
    ) -> Result<WorkerModelSourceTransportCall, async_channel::RecvError> {
        self.inner
            .receive()
            .await
            .map(|inner| WorkerModelSourceTransportCall { inner })
    }
}

enum NextWorkerInput {
    Frame(Result<Result<WorkerEnvelope, FrameError>, async_channel::RecvError>),
    Heartbeat,
    NativeJob(
        Result<
            Result<sim_runtime::NativeImageExecutionResult, NativeImageRuntimeError>,
            async_channel::RecvError,
        >,
    ),
    PluginTask(Result<plugin_runtime::WorkerPluginTaskEvent, async_channel::RecvError>),
    PluginCapability(
        Result<plugin_runtime::WorkerCapabilityBridgeRequest, async_channel::RecvError>,
    ),
    ProviderV2Stream(Result<sim_plugin_host::ProviderV2WorkerStreamCall, async_channel::RecvError>),
    ModelSource(Result<WorkerModelSourceTransportCall, async_channel::RecvError>),
    JobEvent(Result<AttemptEvent, async_channel::RecvError>),
}

struct ActiveExecution {
    cancellation: CancellationToken,
    memory: Option<AttemptMemoryController>,
    native_result: Option<
        async_channel::Receiver<
            Result<sim_runtime::NativeImageExecutionResult, NativeImageRuntimeError>,
        >,
    >,
    plugin_result: Option<async_channel::Receiver<plugin_runtime::WorkerPluginTaskEvent>>,
    plugin_capabilities:
        Option<async_channel::Receiver<plugin_runtime::WorkerCapabilityBridgeRequest>>,
    provider_v2_streams:
        Option<async_channel::Receiver<sim_plugin_host::ProviderV2WorkerStreamCall>>,
    model_sources: Option<WorkerModelSourceTransportHost>,
    events: Option<async_channel::Receiver<AttemptEvent>>,
}

struct CommittedPluginRegistry {
    source: Arc<AssembledWorkerRegistry>,
    compiled: Arc<plugin_runtime::WorkerPluginRegistry>,
}

fn provider_registry_pin_is_available(
    plugin_registry: Option<&CommittedPluginRegistry>,
    provider_registry: &sim_runtime::NativeProviderRegistryPin,
) -> bool {
    plugin_registry.is_some_and(|registry| {
        registry
            .compiled
            .matches_provider_registry_pin(provider_registry)
    })
}

fn apply_compiled_registry_commit(
    current: &mut Option<CommittedPluginRegistry>,
    responses: &[WorkerEnvelope],
    source: Option<Arc<AssembledWorkerRegistry>>,
    compiled: Option<Arc<plugin_runtime::WorkerPluginRegistry>>,
) -> anyhow::Result<bool> {
    let acknowledged = matches!(
        responses,
        [WorkerEnvelope {
            message: WorkerMessage::RegistryDeploymentAck { .. },
            ..
        }]
    );
    if acknowledged {
        let source = source
            .ok_or_else(|| anyhow::anyhow!("worker registry commit produced no source registry"))?;
        let changed = current.as_ref().is_none_or(|registry| {
            registry.source.generation() != source.generation()
                || registry.source.registry_digest_sha256() != source.registry_digest_sha256()
        });
        *current = Some(CommittedPluginRegistry {
            source,
            compiled: compiled.ok_or_else(|| {
                anyhow::anyhow!("worker registry commit produced no compiled registry")
            })?,
        });
        return Ok(changed);
    }
    Ok(false)
}

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("worker frame exceeds {MAX_WORKER_FRAME_BYTES} bytes")]
    TooLarge,
    #[error("encoded worker event exceeds {MAX_ENCODED_PREVIEW_BYTES} bytes")]
    PreviewTooLarge,
    #[error("worker frame allocation failed: {0}")]
    Allocation(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Protocol(#[from] WorkerProtocolError),
}

pub fn validate_worker_payload(envelope: &WorkerEnvelope) -> Result<(), FrameError> {
    if let WorkerMessage::Event { event } = &envelope.message
        && event.len() > MAX_ENCODED_PREVIEW_BYTES
    {
        return Err(FrameError::PreviewTooLarge);
    }
    Ok(())
}

pub fn apply_worker_control_cancellation(
    message: &WorkerMessage,
    cancellation: &CancellationToken,
) -> Option<bool> {
    match message {
        WorkerMessage::Cancel { .. } | WorkerMessage::Shutdown => Some(cancellation.cancel()),
        _ => None,
    }
}

fn cancel_rejected_provider_v2_finalization(
    result: &Result<(), sim_types::WorkerProviderStreamError>,
    cancellation: &CancellationToken,
) {
    if result.is_err() {
        cancellation.cancel();
    }
}

fn clear_pending_provider_v2_after_cancellation<Proposal, Stream>(
    cancellation: &CancellationToken,
    pending_proposal: &mut Option<Proposal>,
    pending_streams: &mut BTreeMap<u64, Stream>,
) -> anyhow::Result<bool> {
    if !cancellation.is_cancelled() {
        return Err(anyhow::anyhow!(
            "provider-v2 pending work cannot be cleared before cancellation"
        ));
    }
    let cancelled_proposal = pending_proposal.take().is_some();
    pending_streams.clear();
    Ok(cancelled_proposal)
}

pub fn write_frame(mut writer: impl Write, envelope: &WorkerEnvelope) -> Result<(), FrameError> {
    validate_worker_payload(envelope)?;
    let frame = encode_worker_frame(envelope)?;
    writer.write_all(&frame)?;
    writer.flush()?;
    Ok(())
}

pub fn read_frame(mut reader: impl Read) -> Result<WorkerEnvelope, FrameError> {
    let mut prefix = [0_u8; 4];
    reader.read_exact(&mut prefix)?;
    let length = u32::from_le_bytes(prefix) as usize;
    if length > MAX_WORKER_FRAME_BYTES {
        return Err(FrameError::TooLarge);
    }

    let mut frame = Vec::new();
    frame
        .try_reserve_exact(4_usize.saturating_add(length))
        .map_err(|error| FrameError::Allocation(error.to_string()))?;
    frame.extend_from_slice(&prefix);
    frame.resize(4 + length, 0);
    let payload = frame
        .get_mut(4..)
        .ok_or_else(|| FrameError::Allocation("invalid frame allocation length".to_owned()))?;
    reader.read_exact(payload)?;
    let envelope = decode_worker_frame(&frame)?;
    validate_worker_payload(&envelope)?;
    Ok(envelope)
}

pub async fn run_worker_process(memory_limit_bytes: u64) -> anyhow::Result<()> {
    run_worker_process_with_authorization_verifier(memory_limit_bytes, None).await
}

pub async fn run_worker_process_with_authorization_verifier(
    memory_limit_bytes: u64,
    plugin_authorization_verifier: Option<PluginAuthorizationVerifier>,
) -> anyhow::Result<()> {
    run_worker_process_with_backend_selection(
        memory_limit_bytes,
        WorkerBackendSelection::Cpu,
        plugin_authorization_verifier,
    )
    .await
}

pub async fn run_worker_process_with_backend_selection(
    memory_limit_bytes: u64,
    backend_selection: WorkerBackendSelection,
    plugin_authorization_verifier: Option<PluginAuthorizationVerifier>,
) -> anyhow::Result<()> {
    run_worker_process_with_backend_selection_and_video_codec_package(
        memory_limit_bytes,
        backend_selection,
        None,
        plugin_authorization_verifier,
    )
    .await
}

pub async fn run_worker_process_with_backend_selection_and_video_codec_package(
    memory_limit_bytes: u64,
    backend_selection: WorkerBackendSelection,
    general_video_codec_package: Option<sim_runtime::NativeGeneralVideoCodecPackageSettings>,
    plugin_authorization_verifier: Option<PluginAuthorizationVerifier>,
) -> anyhow::Result<()> {
    run_worker_process_with_configuration(
        memory_limit_bytes,
        backend_selection,
        general_video_codec_package,
        None,
        plugin_authorization_verifier,
    )
    .await
}

pub async fn run_worker_process_with_diffusion_provider(
    memory_limit_bytes: u64,
    diffusion_provider: Option<Arc<dyn NativeDiffusionProvider>>,
) -> anyhow::Result<()> {
    run_worker_process_with_configuration(
        memory_limit_bytes,
        WorkerBackendSelection::Cpu,
        None,
        diffusion_provider,
        None,
    )
    .await
}

fn initialize_worker_backend(
    selection: WorkerBackendSelection,
    memory_limit_bytes: u64,
) -> (
    Result<WorkerBackendSession, BackendUnavailable>,
    Option<Arc<CpuBackend>>,
) {
    match selection {
        WorkerBackendSelection::Cpu => match WorkerBackendSession::cpu(memory_limit_bytes) {
            Ok((session, backend)) => (Ok(session), Some(backend)),
            Err(error) => (
                Err(BackendUnavailable::new(DeviceKind::Cpu, error.to_string())),
                None,
            ),
        },
        WorkerBackendSelection::DirectMl { package } => {
            #[cfg(feature = "directml")]
            {
                let cancellation = CancellationToken::default();
                let initialized =
                    sim_runtime::initialize_certified_directml_runtime(&package, &cancellation)
                        .and_then(|session| {
                            sim_tensor::generated_backend_directml_sim_model_0018::DirectMlTensorBackend::from_certified_session(
                                session,
                                memory_limit_bytes,
                                &cancellation,
                            )
                            .map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::DirectMl,
                                    "certified DirectML tensor backend initialization failed",
                                )
                            })
                        })
                        .and_then(|(backend, authority)| {
                            WorkerBackendSession::new(Arc::new(backend), authority).map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::DirectMl,
                                    "workspace, allocation, transfer, operator, event, or accounting readiness probe failed",
                                )
                            })
                        });
                (initialized, None)
            }
            #[cfg(not(feature = "directml"))]
            {
                let _ = (package, memory_limit_bytes);
                (
                    Err(BackendUnavailable::new(
                        DeviceKind::DirectMl,
                        "the packaged worker was built without the DirectML integration feature",
                    )),
                    None,
                )
            }
        }
        WorkerBackendSelection::Rocm {
            package,
            device_ordinal,
        } => {
            #[cfg(feature = "rocm")]
            {
                let cancellation = CancellationToken::default();
                let initialized =
                    sim_runtime::initialize_certified_rocm_runtime(&package, &cancellation)
                        .and_then(|certified| {
                            sim_tensor::RocmTensorBackend::from_certified_runtime(
                                certified.into_runtime(),
                                device_ordinal,
                                memory_limit_bytes,
                                &cancellation,
                            )
                            .map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Rocm,
                                    "driver or device initialization failed",
                                )
                            })
                        })
                        .and_then(|(backend, authority)| {
                            WorkerBackendSession::new(Arc::new(backend), authority).map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Rocm,
                                    "allocation, transfer, or event readiness probe failed",
                                )
                            })
                        });
                (initialized, None)
            }
            #[cfg(not(feature = "rocm"))]
            {
                let _ = (package, device_ordinal, memory_limit_bytes);
                (
                    Err(BackendUnavailable::new(
                        DeviceKind::Rocm,
                        "the packaged worker was built without the ROCm integration feature",
                    )),
                    None,
                )
            }
        }
        WorkerBackendSelection::Metal { package } => {
            #[cfg(feature = "metal")]
            {
                let cancellation = CancellationToken::default();
                let initialized =
                    sim_runtime::initialize_certified_metal_runtime(&package, &cancellation)
                        .and_then(|certified| {
                            let host_physical_memory_bytes =
                                certified.host_physical_memory_bytes();
                            let runtime = certified.into_runtime();
                            sim_tensor::MetalTensorBackend::from_certified_runtime(
                                runtime,
                                host_physical_memory_bytes,
                                memory_limit_bytes,
                                &cancellation,
                            )
                            .map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Metal,
                                    "certified Metal tensor backend initialization failed",
                                )
                            })
                        })
                        .and_then(|(backend, authority)| {
                            WorkerBackendSession::new(Arc::new(backend), authority).map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Metal,
                                    "workspace, buffer, transfer, command-buffer, or event readiness probe failed",
                                )
                            })
                        });
                (initialized, None)
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = (package, memory_limit_bytes);
                (
                    Err(BackendUnavailable::new(
                        DeviceKind::Metal,
                        "the packaged worker was built without the Metal integration feature",
                    )),
                    None,
                )
            }
        }
        WorkerBackendSelection::Mlu {
            package,
            device_ordinal,
        } => {
            #[cfg(feature = "mlu")]
            {
                let cancellation = CancellationToken::default();
                let initialized =
                    sim_runtime::initialize_certified_mlu_runtime(&package, &cancellation)
                        .and_then(|runtime| {
                            sim_tensor::MluTensorBackend::from_certified_runtime(
                                runtime,
                                device_ordinal,
                                memory_limit_bytes,
                                &cancellation,
                            )
                            .map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Mlu,
                                    "certified MLU tensor backend initialization failed",
                                )
                            })
                        })
                        .and_then(|(backend, authority)| {
                            WorkerBackendSession::new(Arc::new(backend), authority).map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Mlu,
                                    "workspace, allocation, transfer, kernel, event, or accounting readiness probe failed",
                                )
                            })
                        });
                (initialized, None)
            }
            #[cfg(not(feature = "mlu"))]
            {
                let _ = (package, device_ordinal, memory_limit_bytes);
                (
                    Err(BackendUnavailable::new(
                        DeviceKind::Mlu,
                        "the packaged worker was built without the MLU integration feature",
                    )),
                    None,
                )
            }
        }
        WorkerBackendSelection::Npu {
            package,
            device_ordinal,
        } => {
            #[cfg(feature = "npu")]
            {
                let cancellation = CancellationToken::default();
                let initialized = sim_runtime::initialize_certified_npu_runtime(
                    &package,
                    device_ordinal,
                    &cancellation,
                )
                        .and_then(|runtime| {
                            sim_tensor::NpuTensorBackend::from_certified_runtime(
                                runtime,
                                device_ordinal,
                                memory_limit_bytes,
                                &cancellation,
                            )
                            .map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Npu,
                                    "certified NPU tensor backend initialization failed",
                                )
                            })
                        })
                        .and_then(|(backend, authority)| {
                            WorkerBackendSession::new(Arc::new(backend), authority).map_err(|_| {
                                BackendUnavailable::new(
                                    DeviceKind::Npu,
                                    "workspace, allocation, transfer, kernel, event, or accounting readiness probe failed",
                                )
                            })
                        });
                (initialized, None)
            }
            #[cfg(not(feature = "npu"))]
            {
                let _ = (package, device_ordinal, memory_limit_bytes);
                (
                    Err(BackendUnavailable::new(
                        DeviceKind::Npu,
                        "the packaged worker was built without the NPU integration feature",
                    )),
                    None,
                )
            }
        }
        WorkerBackendSelection::Cuda {
            package,
            device_ordinal,
        } => {
            #[cfg(feature = "cuda")]
            {
                let cancellation = CancellationToken::default();
                let initialized = usize::try_from(device_ordinal)
                    .map_err(|_| {
                        BackendUnavailable::new(
                            DeviceKind::Cuda,
                            "CUDA device ordinal is not representable on this worker",
                        )
                    })
                    .and_then(|device_ordinal| {
                        sim_runtime::initialize_certified_cuda_runtime(
                            &package,
                            device_ordinal,
                            &cancellation,
                        )
                    })
                    .and_then(|session| {
                        sim_tensor::CudaTensorBackend::from_certified_session(
                            session,
                            memory_limit_bytes,
                            &cancellation,
                        )
                        .map_err(|_| {
                            BackendUnavailable::new(
                                DeviceKind::Cuda,
                                "certified CUDA tensor backend initialization failed",
                            )
                        })
                    })
                    .and_then(|(backend, authority)| {
                        WorkerBackendSession::new(Arc::new(backend), authority).map_err(|_| {
                            BackendUnavailable::new(
                                DeviceKind::Cuda,
                                "workspace, allocation, transfer, kernel, event, or accounting readiness probe failed",
                            )
                        })
                    });
                (initialized, None)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (package, device_ordinal, memory_limit_bytes);
                (
                    Err(BackendUnavailable::new(
                        DeviceKind::Cuda,
                        "the packaged worker was built without the CUDA integration feature",
                    )),
                    None,
                )
            }
        }
        WorkerBackendSelection::Xpu {
            package,
            device_ordinal,
        } => {
            #[cfg(feature = "xpu")]
            {
                let cancellation = CancellationToken::default();
                let initialized = usize::try_from(device_ordinal)
                    .map_err(|_| {
                        BackendUnavailable::new(
                            DeviceKind::Xpu,
                            "XPU device ordinal is not representable on this worker",
                        )
                    })
                    .and_then(|device_ordinal| {
                        sim_runtime::initialize_certified_xpu_runtime(
                            &package,
                            device_ordinal,
                            &cancellation,
                        )
                    })
                    .and_then(|session| {
                    sim_tensor::XpuTensorBackend::from_certified_session(
                        session,
                        memory_limit_bytes,
                        &cancellation,
                    )
                    .map_err(|_| {
                        BackendUnavailable::new(
                            DeviceKind::Xpu,
                            "certified XPU tensor backend initialization failed",
                        )
                    })
                })
                .and_then(|(backend, authority)| {
                    WorkerBackendSession::new(Arc::new(backend), authority).map_err(|_| {
                        BackendUnavailable::new(
                            DeviceKind::Xpu,
                            "workspace, allocation, transfer, kernel, event, or accounting readiness probe failed",
                        )
                    })
                });
                (initialized, None)
            }
            #[cfg(not(feature = "xpu"))]
            {
                let _ = (package, device_ordinal, memory_limit_bytes);
                (
                    Err(BackendUnavailable::new(
                        DeviceKind::Xpu,
                        "the packaged worker was built without the XPU integration feature",
                    )),
                    None,
                )
            }
        }
    }
}

async fn run_worker_process_with_configuration(
    memory_limit_bytes: u64,
    backend_selection: WorkerBackendSelection,
    general_video_codec_package: Option<sim_runtime::NativeGeneralVideoCodecPackageSettings>,
    diffusion_provider: Option<Arc<dyn NativeDiffusionProvider>>,
    plugin_authorization_verifier: Option<PluginAuthorizationVerifier>,
) -> anyhow::Result<()> {
    let (backend_session, cpu_executor_backend) =
        initialize_worker_backend(backend_selection, memory_limit_bytes);
    let video_codec_worker_services = initialize_general_video_codec_worker_services(
        general_video_codec_package,
        cpu_executor_backend.clone(),
        memory_limit_bytes,
    )?;
    if cpu_executor_backend.is_some() {
        sim_runtime::generated_native_node_registry_projection(diffusion_provider.clone())?;
    }
    let mut shader_prewarm_thread = None;
    let worker_result = run_initialized_worker_process(
        backend_session,
        cpu_executor_backend,
        video_codec_worker_services,
        diffusion_provider,
        plugin_authorization_verifier,
        &mut shader_prewarm_thread,
    )
    .await;
    let shader_prewarm_result = match shader_prewarm_thread {
        Some(thread) => thread.join().map_err(|panic| {
            let message = if let Some(message) = panic.downcast_ref::<String>() {
                message.as_str()
            } else if let Some(message) = panic.downcast_ref::<&str>() {
                message
            } else {
                "non-string panic payload"
            };
            anyhow::anyhow!("native shader prewarm thread panicked: {message}")
        }),
        None => Ok(()),
    };
    match (worker_result, shader_prewarm_result) {
        (Ok(()), result) => result,
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(prewarm_error)) => Err(error.context(format!(
            "native shader prewarm cleanup also failed: {prewarm_error:#}"
        ))),
    }
}

async fn run_initialized_worker_process(
    backend_session: Result<WorkerBackendSession, BackendUnavailable>,
    cpu_executor_backend: Option<Arc<CpuBackend>>,
    video_codec_worker_services: Option<NativeVideoCodecWorkerServices>,
    diffusion_provider: Option<Arc<dyn NativeDiffusionProvider>>,
    plugin_authorization_verifier: Option<PluginAuthorizationVerifier>,
    shader_prewarm_thread: &mut Option<std::thread::JoinHandle<()>>,
) -> anyhow::Result<()> {
    let mut session = WorkerSession::with_backend_result(backend_session);
    let (input_sender, input_receiver) = async_channel::bounded(8);
    let reader_thread = std::thread::Builder::new()
        .name("sim-worker-stdin".to_owned())
        .spawn(move || {
            let stdin = io::stdin();
            loop {
                let frame = read_frame(stdin.lock());
                let terminal = frame.is_err();
                if input_sender.send_blocking(frame).is_err() || terminal {
                    break;
                }
            }
        })?;

    if cpu_executor_backend.is_some() {
        *shader_prewarm_thread = Some(
            std::thread::Builder::new()
                .name("sim-worker-shader-prewarm".to_owned())
                .spawn(sim_runtime::prewarm_native_shader_executor)?,
        );
    }

    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    let mut next_heartbeat = Instant::now() + HEARTBEAT_INTERVAL;
    let mut active_execution: Option<ActiveExecution> = None;
    let mut native_image_executor: Option<NativeImageExecutor> = None;
    let mut native_image_executor_provider_registry = None;
    let mut native_provider_capabilities = None;
    let mut plugin_registry: Option<CommittedPluginRegistry> = None;
    let mut pending_plugin_capabilities: BTreeMap<u64, async_channel::Sender<Vec<u8>>> =
        BTreeMap::new();
    let mut pending_provider_v2_streams: BTreeMap<
        u64,
        sim_plugin_host::ProviderV2WorkerStreamCall,
    > = BTreeMap::new();
    let mut pending_provider_v2_proposal: Option<
        sim_plugin_host::ProviderV2WorkerPendingInvocation,
    > = None;
    let mut pending_model_source: Option<WorkerModelSourceTransportCall> = None;
    'worker: loop {
        let delay = next_heartbeat.saturating_duration_since(Instant::now());
        let next = smol::future::race(
            async { NextWorkerInput::Frame(input_receiver.recv().await) },
            smol::future::race(
                async {
                    if session.heartbeat_enabled() {
                        worker_delay(delay).await;
                        NextWorkerInput::Heartbeat
                    } else {
                        pending().await
                    }
                },
                smol::future::race(
                    async {
                        if let Some(active) = &active_execution {
                            if let Some(result) = active.native_result.as_ref() {
                                NextWorkerInput::NativeJob(result.recv().await)
                            } else {
                                pending().await
                            }
                        } else {
                            pending().await
                        }
                    },
                    smol::future::race(
                        async {
                            if let Some(result) = active_execution
                                .as_ref()
                                .and_then(|active| active.plugin_result.as_ref())
                            {
                                NextWorkerInput::PluginTask(result.recv().await)
                            } else {
                                pending().await
                            }
                        },
                        smol::future::race(
                            async {
                                if let Some(requests) = active_execution
                                    .as_ref()
                                    .and_then(|active| active.plugin_capabilities.as_ref())
                                {
                                    NextWorkerInput::PluginCapability(requests.recv().await)
                                } else {
                                    pending().await
                                }
                            },
                            smol::future::race(
                                async {
                                    if let Some(requests) = active_execution
                                        .as_ref()
                                        .and_then(|active| active.provider_v2_streams.as_ref())
                                    {
                                        NextWorkerInput::ProviderV2Stream(requests.recv().await)
                                    } else {
                                        pending().await
                                    }
                                },
                                smol::future::race(
                                    async {
                                        if let Some(requests) = active_execution
                                            .as_ref()
                                            .and_then(|active| active.model_sources.as_ref())
                                        {
                                            NextWorkerInput::ModelSource(requests.receive().await)
                                        } else {
                                            pending().await
                                        }
                                    },
                                    async {
                                        if let Some(events) = active_execution
                                            .as_ref()
                                            .and_then(|active| active.events.as_ref())
                                        {
                                            NextWorkerInput::JobEvent(events.recv().await)
                                        } else {
                                            pending().await
                                        }
                                    },
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        .await;

        match next {
            NextWorkerInput::Heartbeat => {
                if let Some(heartbeat) = session.heartbeat()? {
                    write_frame(&mut stdout, &heartbeat)?;
                }
                next_heartbeat = Instant::now() + HEARTBEAT_INTERVAL;
            }
            NextWorkerInput::Frame(Ok(Ok(envelope))) => {
                let handled = if matches!(
                    &envelope.message,
                    sim_types::WorkerMessage::RegistryDeploymentCommit { .. }
                ) {
                    let authorization_verifier = plugin_authorization_verifier.as_ref();
                    let mut compiled_candidate = None;
                    let responses =
                        session.handle_verified_registry_commit(envelope.clone(), |source| {
                            let candidate = plugin_runtime::WorkerPluginRegistry::from_assembled(
                                envelope.profile_id,
                                source,
                                sim_plugin_host::ComponentLimits::default(),
                                authorization_verifier.ok_or_else(|| {
                                    WorkerRegistryDeploymentRejectionReason::VerificationUnavailable
                                })?,
                            )
                            .map_err(|error| {
                                eprintln!(
                                    "sim-worker: registry component deployment failed: {error}"
                                );
                                error.deployment_rejection_reason()
                            })?;
                            compiled_candidate = Some(Arc::new(candidate));
                            Ok(())
                        });
                    if let Ok(responses) = &responses {
                        if apply_compiled_registry_commit(
                            &mut plugin_registry,
                            responses,
                            session.registry().cloned().map(Arc::new),
                            compiled_candidate,
                        )? {
                            native_image_executor = None;
                            native_image_executor_provider_registry = None;
                            native_provider_capabilities = None;
                        }
                    }
                    responses
                } else {
                    session.handle(envelope.clone())
                };
                match handled {
                    Ok(responses) => {
                        for response in responses {
                            write_frame(&mut stdout, &response)?;
                        }
                        match &envelope.message {
                            sim_types::WorkerMessage::Execute { plan } => {
                                let worker_plan: NativeImageWorkerPlan =
                                    serde_json::from_slice(plan).map_err(|error| {
                                        anyhow::anyhow!("invalid native image worker plan: {error}")
                                    })?;
                                worker_plan.validate()?;
                                validate_worker_plan_has_no_serialized_handles(&worker_plan)?;
                                if let Some(provider_registry) = &worker_plan.provider_registry
                                    && !provider_registry_pin_is_available(
                                        plugin_registry.as_ref(),
                                        provider_registry,
                                    )
                                {
                                    let event = NativeImageWorkerEvent::Failed {
                                        message: "native provider registry pin is unavailable or stale; execution was not dispatched".to_owned(),
                                        cancelled: false,
                                    };
                                    let encoded = postcard::to_stdvec(&event)?;
                                    let response = session.complete_execution(encoded)?;
                                    write_frame(&mut stdout, &response)?;
                                    continue 'worker;
                                }
                                if let Some(unavailable) =
                                    backend_neutral_executor_unavailable(session.backend_device())
                                {
                                    let event =
                                        NativeImageWorkerEvent::BackendUnavailable { unavailable };
                                    let encoded = postcard::to_stdvec(&event)?;
                                    let response = session.complete_execution(encoded)?;
                                    write_frame(&mut stdout, &response)?;
                                    continue 'worker;
                                }
                                let (mut memory, memory_configuration) =
                                    match prepare_native_image_memory(
                                        &session,
                                        &worker_plan,
                                        video_codec_worker_services.as_ref().map_or(
                                            0,
                                            NativeVideoCodecWorkerServices::codec_residency_bytes,
                                        ),
                                    ) {
                                        Ok(memory) => memory,
                                        Err(error) => {
                                            let event = NativeImageWorkerEvent::Failed {
                                                message: format!(
                                                    "native memory preflight failed without dispatch: {error}"
                                                ),
                                                cancelled: false,
                                            };
                                            let encoded = postcard::to_stdvec(&event)?;
                                            let response = session.complete_execution(encoded)?;
                                            write_frame(&mut stdout, &response)?;
                                            continue 'worker;
                                        }
                                    };
                                let memory_configuration = append_video_codec_cache_configuration(
                                    memory_configuration,
                                    video_codec_worker_services.as_ref(),
                                );
                                let diffusion_enabled = diffusion_provider.is_some();
                                let reuse_executor =
                                    native_image_executor.as_ref().is_some_and(|executor| {
                                        executor.profile_id() == envelope.profile_id
                                            && executor.metadata_enabled()
                                                == worker_plan.metadata_enabled
                                            && executor.diffusion_enabled() == diffusion_enabled
                                            && native_image_executor_provider_registry.as_ref()
                                                == worker_plan.provider_registry.as_ref()
                                    });
                                let executor_update = if reuse_executor {
                                    native_image_executor
                                        .as_ref()
                                        .ok_or_else(|| {
                                            NativeImageRuntimeError::Execution(
                                                "native executor vanished".to_owned(),
                                            )
                                        })
                                        .and_then(|executor| {
                                            executor.replace_input_assets(
                                                worker_plan.input_assets.clone(),
                                            )
                                        })
                                } else {
                                    let cpu_backend = cpu_executor_backend.clone().ok_or_else(|| {
                                        NativeImageRuntimeError::Execution(
                                            "CPU executor adapter is unavailable for the selected backend"
                                                .to_owned(),
                                        )
                                    })?;
                                    let created = if worker_plan.provider_registry.is_some() {
                                        let registry = plugin_registry
                                            .as_ref()
                                            .ok_or_else(|| {
                                                NativeImageRuntimeError::Execution(
                                                    "native provider registry is unavailable"
                                                        .to_owned(),
                                                )
                                            })?
                                            .compiled
                                            .clone();
                                        let (capability_sender, capability_receiver) =
                                            async_channel::bounded(8);
                                        let bridge =
                                            Arc::new(plugin_runtime::WorkerCapabilityBridge::new(
                                                capability_sender,
                                            ));
                                        native_provider_capabilities = Some(capability_receiver);
                                        if let Some(provider) = diffusion_provider.clone() {
                                            NativeImageExecutor::new_with_generated_registry_diffusion_and_registration(
                                                envelope.profile_id,
                                                worker_plan.input_assets.clone(),
                                                worker_plan.metadata_enabled,
                                                cpu_backend,
                                                provider,
                                                {
                                                    let registry = registry.clone();
                                                    let bridge = bridge.clone();
                                                    move |nodes| {
                                                    registry
                                                        .activate_native_provider_nodes(nodes, bridge)
                                                        .map_err(|error| {
                                                            NativeImageRuntimeError::Execution(
                                                                error.to_string(),
                                                            )
                                                        })
                                                    }
                                                },
                                            )
                                        } else {
                                            NativeImageExecutor::new_with_generated_registry_and_registration(
                                                envelope.profile_id,
                                                worker_plan.input_assets.clone(),
                                                worker_plan.metadata_enabled,
                                                cpu_backend,
                                                move |nodes| {
                                                    registry
                                                        .activate_native_provider_nodes(nodes, bridge)
                                                        .map_err(|error| {
                                                            NativeImageRuntimeError::Execution(
                                                                error.to_string(),
                                                            )
                                                        })
                                                },
                                            )
                                        }
                                    } else if let Some(provider) = diffusion_provider.clone() {
                                        native_provider_capabilities = None;
                                        NativeImageExecutor::new_with_generated_registry_and_diffusion_provider(
                                            envelope.profile_id,
                                            worker_plan.input_assets.clone(),
                                            worker_plan.metadata_enabled,
                                            cpu_backend,
                                            provider,
                                        )
                                    } else {
                                        native_provider_capabilities = None;
                                        NativeImageExecutor::new_with_generated_registry(
                                            envelope.profile_id,
                                            worker_plan.input_assets.clone(),
                                            worker_plan.metadata_enabled,
                                            cpu_backend,
                                        )
                                    };
                                    created.map(|executor| {
                                        let executor = if let Some(services) =
                                            video_codec_worker_services.as_ref()
                                        {
                                            executor
                                                .with_ltxv_preprocess_service(
                                                    services.ltxv_preprocess_service(),
                                                )
                                                .with_webm_encode_service(
                                                    services.webm_encode_service(),
                                                )
                                                .with_component_h264_mp4_backing_service(
                                                    services.component_h264_mp4_backing_service(),
                                                )
                                        } else {
                                            executor
                                        };
                                        native_image_executor = Some(executor);
                                        native_image_executor_provider_registry =
                                            worker_plan.provider_registry.clone();
                                    })
                                };
                                if let Err(error) = executor_update {
                                    let event = NativeImageWorkerEvent::Failed {
                                        message: format!(
                                            "native executor setup failed without dispatch: {error}"
                                        ),
                                        cancelled: false,
                                    };
                                    let encoded = postcard::to_stdvec(&event)?;
                                    let response = session.complete_execution(encoded)?;
                                    write_frame(&mut stdout, &response)?;
                                    continue 'worker;
                                }
                                let planned_workspace = memory.issue_workspace_authorization()?;
                                let workspace_authorization =
                                    session.authorize_planned_workspace(planned_workspace)?;
                                memory.begin()?;
                                let mut current = native_image_executor
                                    .as_ref()
                                    .ok_or_else(|| anyhow::anyhow!("native executor unavailable"))?
                                    .clone();
                                let attempt_id = envelope.attempt_id.ok_or_else(|| {
                                    anyhow::anyhow!("execute omitted attempt identity")
                                })?;
                                let cancellation = CancellationToken::default();
                                let model_sources =
                                    if let Some(binding) = worker_plan.model_source_service {
                                        let (bound, host) =
                                            current.with_model_source_worker(binding)?;
                                        current = bound;
                                        Some(WorkerModelSourceTransportHost { inner: host })
                                    } else {
                                        None
                                    };
                                let event_bus = ExecutionEventBus::new(32)?;
                                let events = event_bus.subscribe();
                                let (result_sender, result) = async_channel::bounded(1);
                                let cancellation_for_job = cancellation.clone();
                                smol::spawn(async move {
                                    let result = smol::unblock(move || {
                                        current.execute_blocking_with_event_bus_and_configuration(
                                            &worker_plan.plan,
                                            attempt_id,
                                            cancellation_for_job,
                                            worker_plan.injected_delay_millis,
                                            Some(event_bus),
                                            workspace_authorization,
                                            &memory_configuration,
                                        )
                                    })
                                    .await;
                                    if let Err(error) = result_sender.send(result).await {
                                        eprintln!(
                                            "sim-worker: execution result receiver closed: {error}"
                                        );
                                    }
                                })
                                .detach();
                                active_execution = Some(ActiveExecution {
                                    cancellation,
                                    memory: Some(memory),
                                    native_result: Some(result),
                                    plugin_result: None,
                                    plugin_capabilities: native_provider_capabilities.clone(),
                                    provider_v2_streams: None,
                                    model_sources,
                                    events: Some(events),
                                });
                            }
                            sim_types::WorkerMessage::ExecutePlugin { invocation } => {
                                let invocation =
                                    sim_plugin_host::WorkerPluginInvocation::from_bytes(
                                        invocation,
                                    )?;
                                if invocation.provider_v2().is_some() {
                                    session.mark_provider_v2_execution()?;
                                }
                                let component_limits = invocation.component_limits().clone();
                                let reuse_registry =
                                    plugin_registry.as_ref().is_some_and(|registry| {
                                        registry.compiled.uses_component_limits(&component_limits)
                                    });
                                if !reuse_registry {
                                    let registry = plugin_registry.as_mut().ok_or_else(|| {
                                        anyhow::anyhow!("worker plugin registry is unavailable")
                                    })?;
                                    let candidate = plugin_runtime::WorkerPluginRegistry::from_assembled(
                                        envelope.profile_id,
                                        &registry.source,
                                        component_limits,
                                        plugin_authorization_verifier.as_ref().ok_or_else(|| {
                                            anyhow::anyhow!(
                                                "worker plugin authorization verifier is unavailable"
                                            )
                                        })?,
                                    )?;
                                    registry.compiled = Arc::new(candidate);
                                }
                                let registry = plugin_registry
                                    .as_ref()
                                    .ok_or_else(|| {
                                        anyhow::anyhow!("worker plugin registry is unavailable")
                                    })?
                                    .compiled
                                    .clone();
                                let cancellation = CancellationToken::default();
                                let (capability_sender, plugin_capabilities) =
                                    async_channel::bounded(8);
                                let (provider_v2_sender, provider_v2_streams) =
                                    async_channel::bounded(1);
                                let bridge = Arc::new(
                                    plugin_runtime::WorkerCapabilityBridge::new(capability_sender)
                                        .with_provider_v2_sender(provider_v2_sender),
                                );
                                let (result_sender, plugin_result) = async_channel::bounded(1);
                                let cancellation_for_job = cancellation.clone();
                                smol::spawn(async move {
                                    let result = smol::unblock(move || {
                                        registry.execute(invocation, bridge, cancellation_for_job)
                                    })
                                    .await;
                                    let event = plugin_runtime::encode_plugin_task_event(result);
                                    if let Err(error) = result_sender.send(event).await {
                                        eprintln!(
                                            "sim-worker: plugin result receiver closed: {error}"
                                        );
                                    }
                                })
                                .detach();
                                active_execution = Some(ActiveExecution {
                                    cancellation,
                                    memory: None,
                                    native_result: None,
                                    plugin_result: Some(plugin_result),
                                    plugin_capabilities: Some(plugin_capabilities),
                                    provider_v2_streams: Some(provider_v2_streams),
                                    model_sources: None,
                                    events: None,
                                });
                            }
                            sim_types::WorkerMessage::PluginCapabilityResponse {
                                call_id,
                                response,
                            } => {
                                let response_sender = pending_plugin_capabilities
                                    .remove(call_id)
                                    .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "worker received an unknown plugin capability response"
                                    )
                                })?;
                                response_sender.send(response.clone()).await.map_err(|_| {
                                    anyhow::anyhow!(
                                        "worker plugin capability response receiver was closed"
                                    )
                                })?;
                            }
                            sim_types::WorkerMessage::ProviderStreamResponse {
                                call_id,
                                response,
                            } => {
                                let call = pending_provider_v2_streams.remove(call_id).ok_or_else(
                                    || {
                                        anyhow::anyhow!(
                                            "worker received an unknown provider-v2 stream response"
                                        )
                                    },
                                )?;
                                call.respond(response.clone())
                                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                            }
                            sim_types::WorkerMessage::ProviderV2ProposalFinalization {
                                finalization,
                            } => {
                                if active_execution.is_some() {
                                    let result = match pending_provider_v2_proposal.take() {
                                        Some(pending) => pending.finalize(finalization),
                                        None => {
                                            Err(sim_types::WorkerProviderStreamError::InvalidOrder)
                                        }
                                    };
                                    if let Some(active) = &active_execution {
                                        cancel_rejected_provider_v2_finalization(
                                            &result,
                                            &active.cancellation,
                                        );
                                    }
                                    pending_provider_v2_streams.clear();
                                    let acknowledgement =
                                        sim_types::WorkerProviderV2ProposalFinalizationAck {
                                            finalization: finalization.clone(),
                                            result,
                                        };
                                    let response = session
                                        .complete_provider_v2_finalization(acknowledgement)?;
                                    write_frame(&mut stdout, &response)?;
                                    active_execution = None;
                                }
                            }
                            sim_types::WorkerMessage::ModelSourceResponse { call_id, response } => {
                                let pending = pending_model_source.take().ok_or_else(|| {
                                    if let Some(active) = &active_execution {
                                        active.cancellation.cancel();
                                    }
                                    anyhow::anyhow!(
                                        "worker received a model-source response without a pending capacity-one call"
                                    )
                                })?;
                                if let Err(error) =
                                    pending.respond(*call_id, response.clone()).await
                                {
                                    if let Some(active) = &active_execution {
                                        active.cancellation.cancel();
                                    }
                                    return Err(anyhow::anyhow!(error.to_string()));
                                }
                            }
                            sim_types::WorkerMessage::Cancel { .. } => {
                                let active = active_execution.as_mut().ok_or_else(|| {
                                    anyhow::anyhow!("cancellation has no active execution")
                                })?;
                                if let Some(memory) = active.memory.as_mut() {
                                    memory.cancel()?;
                                }
                                if apply_worker_control_cancellation(
                                    &envelope.message,
                                    &active.cancellation,
                                )
                                .is_none()
                                {
                                    return Err(anyhow::anyhow!(
                                        "worker cancellation branch received a non-control message"
                                    ));
                                }
                                let cancelled_provider_v2_proposal =
                                    clear_pending_provider_v2_after_cancellation(
                                        &active.cancellation,
                                        &mut pending_provider_v2_proposal,
                                        &mut pending_provider_v2_streams,
                                    )?;
                                if let Some(pending) = pending_model_source.take() {
                                    pending
                                        .revoke()
                                        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                                }
                                if cancelled_provider_v2_proposal {
                                    let response = session.complete_plugin_execution(
                                        WorkerPluginExecutionOutcome::Failed(
                                            sim_types::WorkerPluginExecutionFailure::Cancelled,
                                        ),
                                    )?;
                                    write_frame(&mut stdout, &response)?;
                                    active_execution = None;
                                }
                            }
                            sim_types::WorkerMessage::Shutdown => {
                                if let Some(active) = &mut active_execution {
                                    if let Some(memory) = active.memory.as_mut()
                                        && matches!(
                                            memory.state(),
                                            AttemptMemoryState::Planned
                                                | AttemptMemoryState::Running
                                        )
                                    {
                                        memory.cancel()?;
                                    }
                                    if apply_worker_control_cancellation(
                                        &envelope.message,
                                        &active.cancellation,
                                    )
                                    .is_none()
                                    {
                                        return Err(anyhow::anyhow!(
                                            "worker shutdown branch received a non-control message"
                                        ));
                                    }
                                    clear_pending_provider_v2_after_cancellation(
                                        &active.cancellation,
                                        &mut pending_provider_v2_proposal,
                                        &mut pending_provider_v2_streams,
                                    )?;
                                    if let Some(pending) = pending_model_source.take() {
                                        pending
                                            .revoke()
                                            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                                    }
                                } else if pending_provider_v2_proposal.is_some()
                                    || !pending_provider_v2_streams.is_empty()
                                    || pending_model_source.is_some()
                                {
                                    return Err(anyhow::anyhow!(
                                        "worker shutdown found bridge work without an active execution"
                                    ));
                                }
                            }
                            sim_types::WorkerMessage::Hello { .. }
                            | sim_types::WorkerMessage::HelloAck { .. }
                            | sim_types::WorkerMessage::Ready
                            | sim_types::WorkerMessage::Event { .. }
                            | sim_types::WorkerMessage::OutputProposal { .. }
                            | sim_types::WorkerMessage::Heartbeat
                            | sim_types::WorkerMessage::Fatal { .. }
                            | sim_types::WorkerMessage::Lifecycle { .. }
                            | sim_types::WorkerMessage::RegistryDeploymentBegin { .. }
                            | sim_types::WorkerMessage::RegistryDeploymentChunk { .. }
                            | sim_types::WorkerMessage::RegistryDeploymentCommit { .. }
                            | sim_types::WorkerMessage::RegistryDeploymentAck { .. }
                            | sim_types::WorkerMessage::RegistryDeploymentRejected { .. }
                            | sim_types::WorkerMessage::PluginCapabilityRequest { .. }
                            | sim_types::WorkerMessage::PluginResult { .. }
                            | sim_types::WorkerMessage::ProviderStreamRequest { .. }
                            | sim_types::WorkerMessage::ProviderV2ProposalFinalizationAck {
                                ..
                            }
                            | sim_types::WorkerMessage::ModelSourceRequest { .. } => {}
                        }
                        if session.is_terminal() {
                            break;
                        }
                    }
                    Err(error) => {
                        eprintln!("sim-worker: {error}");
                        let fatal = session.fatal_for(&envelope, &error)?;
                        write_frame(&mut stdout, &fatal)?;
                        return Err(error.into());
                    }
                }
            }
            NextWorkerInput::Frame(Ok(Err(error))) => {
                eprintln!("sim-worker: {error}");
                return Err(error.into());
            }
            NextWorkerInput::Frame(Err(error)) => {
                return Err(anyhow::anyhow!("worker input channel closed: {error}"));
            }
            NextWorkerInput::NativeJob(Ok(result)) => {
                if !pending_plugin_capabilities.is_empty() {
                    return Err(anyhow::anyhow!(
                        "native provider execution finished with capability calls still pending"
                    ));
                }
                pending_plugin_capabilities.clear();
                if pending_model_source.is_some() {
                    return Err(anyhow::anyhow!(
                        "native execution finished with a model-source call still pending"
                    ));
                }
                if let Some(active) = &active_execution
                    && let Some(events) = &active.events
                {
                    while let Ok(event) = events.try_recv() {
                        write_execution_event(&mut session, &mut stdout, event)?;
                    }
                }
                let policy_cancelled = active_execution
                    .as_ref()
                    .and_then(|active| active.memory.as_ref())
                    .is_some_and(|memory| !memory.accepts_value());
                let cancelled = policy_cancelled
                    || matches!(result, Err(NativeImageRuntimeError::Cancelled))
                    || matches!(
                        result.as_ref().map(|value| value.report.state),
                        Ok(AttemptState::Cancelled | AttemptState::Interrupted)
                    );
                if let Some(memory) = active_execution
                    .as_mut()
                    .and_then(|active| active.memory.as_mut())
                {
                    if cancelled {
                        if matches!(
                            memory.state(),
                            AttemptMemoryState::Planned | AttemptMemoryState::Running
                        ) {
                            memory.cancel()?;
                        }
                    } else {
                        match result.as_ref().map(|value| value.report.state) {
                            Ok(AttemptState::Succeeded) => memory.complete()?,
                            Ok(AttemptState::Failed)
                            | Ok(
                                AttemptState::Queued
                                | AttemptState::Running
                                | AttemptState::Cancelling,
                            )
                            | Err(_) => memory.fail()?,
                            Ok(AttemptState::Cancelled | AttemptState::Interrupted) => {
                                memory.cancel()?;
                            }
                        }
                    }
                }
                let event = match result {
                    Ok(_) if policy_cancelled => NativeImageWorkerEvent::Failed {
                        message: "native image execution was cancelled; late values were discarded"
                            .to_owned(),
                        cancelled: true,
                    },
                    Ok(mut result) => {
                        let mut output_proposal_ids =
                            Vec::with_capacity(result.output_proposals.len());
                        for proposal in result.output_proposals.drain(..) {
                            output_proposal_ids.push(proposal.proposal_id());
                            let proposal = proposal.to_worker_proposal()?;
                            let response = session.output_proposal(proposal)?;
                            write_frame(&mut stdout, &response)?;
                        }
                        NativeImageWorkerEvent::Completed {
                            result: sim_runtime::NativeImageWorkerResult::from_execution_report(
                                result.report,
                                output_proposal_ids,
                                result.executed_node_count,
                            )?,
                        }
                    }
                    Err(error) => NativeImageWorkerEvent::Failed {
                        message: error.to_string(),
                        cancelled,
                    },
                };
                let encoded = postcard::to_stdvec(&event)?;
                let response = session.complete_execution(encoded)?;
                write_frame(&mut stdout, &response)?;
                active_execution = None;
            }
            NextWorkerInput::NativeJob(Err(error)) => {
                return Err(anyhow::anyhow!("worker execution channel closed: {error}"));
            }
            NextWorkerInput::PluginTask(Ok(plugin_runtime::WorkerPluginTaskEvent::Terminal(
                outcome,
            ))) => {
                if matches!(outcome, WorkerPluginExecutionOutcome::Succeeded(_))
                    && !pending_plugin_capabilities.is_empty()
                {
                    return Err(anyhow::anyhow!(
                        "plugin execution finished with capability calls still pending"
                    ));
                }
                pending_plugin_capabilities.clear();
                let response = session.complete_plugin_execution(outcome)?;
                write_frame(&mut stdout, &response)?;
                active_execution = None;
            }
            NextWorkerInput::PluginTask(Ok(
                plugin_runtime::WorkerPluginTaskEvent::ProviderV2Proposal { outcome, pending },
            )) => {
                if !pending_plugin_capabilities.is_empty()
                    || !pending_provider_v2_streams.is_empty()
                    || pending_provider_v2_proposal.is_some()
                {
                    return Err(anyhow::anyhow!(
                        "provider-v2 proposal arrived with unfinished bridge calls"
                    ));
                }
                let response = session.provider_v2_proposal(outcome)?;
                write_frame(&mut stdout, &response)?;
                pending_provider_v2_proposal = Some(pending);
                if let Some(active) = &mut active_execution {
                    active.plugin_result = None;
                }
            }
            NextWorkerInput::PluginTask(Err(error)) => {
                return Err(anyhow::anyhow!(
                    "worker plugin execution channel closed: {error}"
                ));
            }
            NextWorkerInput::ProviderV2Stream(Ok(call)) => {
                if pending_provider_v2_proposal.is_some() {
                    if let Some(active) = &active_execution {
                        active.cancellation.cancel();
                    }
                    return Err(anyhow::anyhow!(
                        "worker provider-v2 stream call arrived after its proposal"
                    ));
                }
                let call_id = call.call_id();
                if !pending_provider_v2_streams.is_empty() {
                    return Err(anyhow::anyhow!(
                        "worker provider-v2 capacity-one route received concurrent calls"
                    ));
                }
                if pending_provider_v2_streams.insert(call_id, call).is_some() {
                    return Err(anyhow::anyhow!(
                        "worker provider-v2 stream repeated a call identifier"
                    ));
                }
                let request = pending_provider_v2_streams
                    .get(&call_id)
                    .ok_or_else(|| anyhow::anyhow!("provider-v2 stream call vanished"))?
                    .request()
                    .clone();
                let response = session.provider_stream_request(call_id, request)?;
                write_frame(&mut stdout, &response)?;
            }
            NextWorkerInput::ProviderV2Stream(Err(_)) => {
                if let Some(active) = &mut active_execution {
                    active.provider_v2_streams = None;
                }
            }
            NextWorkerInput::ModelSource(Ok(call)) => {
                if pending_model_source.is_some() {
                    if let Some(active) = &active_execution {
                        active.cancellation.cancel();
                    }
                    call.revoke()
                        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                    return Err(anyhow::anyhow!(
                        "worker model-source capacity-one route received concurrent calls"
                    ));
                }
                let response =
                    session.model_source_request(call.call_id(), call.request().clone())?;
                pending_model_source = Some(call);
                write_frame(&mut stdout, &response)?;
            }
            NextWorkerInput::ModelSource(Err(_)) => {
                if let Some(active) = &mut active_execution {
                    active.model_sources = None;
                }
            }
            NextWorkerInput::PluginCapability(Ok(request)) => {
                if pending_plugin_capabilities
                    .insert(request.call_id, request.response_sender)
                    .is_some()
                {
                    return Err(anyhow::anyhow!(
                        "worker plugin repeated a capability call identifier"
                    ));
                }
                let response =
                    session.plugin_capability_request(request.call_id, request.request)?;
                write_frame(&mut stdout, &response)?;
            }
            NextWorkerInput::PluginCapability(Err(_)) => {
                if active_execution
                    .as_ref()
                    .and_then(|active| active.plugin_result.as_ref())
                    .is_none()
                {
                    return Err(anyhow::anyhow!(
                        "worker plugin capability channel closed before completion"
                    ));
                }
                if let Some(active) = &mut active_execution {
                    active.plugin_capabilities = None;
                }
            }
            NextWorkerInput::JobEvent(Ok(event)) => {
                write_execution_event(&mut session, &mut stdout, event)?;
            }
            NextWorkerInput::JobEvent(Err(_)) => {
                if let Some(active) = &mut active_execution {
                    active.events = None;
                }
            }
        }
    }

    // The reader owns a blocking stdin lock; process exit closes it after the
    // supervisor receives the shutdown acknowledgement.
    drop(reader_thread);
    Ok(())
}

fn initialize_general_video_codec_worker_services(
    settings: Option<sim_runtime::NativeGeneralVideoCodecPackageSettings>,
    cpu_executor_backend: Option<Arc<CpuBackend>>,
    memory_limit_bytes: u64,
) -> anyhow::Result<Option<NativeVideoCodecWorkerServices>> {
    let Some(settings) = settings else {
        return Ok(None);
    };
    let cancellation = CancellationToken::default();
    let closure =
        certify_general_video_codec_package(&settings, &cancellation).map_err(|error| {
            anyhow::anyhow!("general video codec package admission failed: {error}")
        })?;
    let codec_residency_bytes = closure
        .startup_resident_bytes()
        .checked_add(closure.codec_scratch_bytes())
        .ok_or_else(|| anyhow::anyhow!("general video codec startup budget overflowed"))?;
    MemoryPlanner::plan(
        memory_limit_bytes,
        0,
        MemoryPlanRequest {
            codec_bytes: codec_residency_bytes,
            ..MemoryPlanRequest::default()
        },
    )
    .map_err(|error| {
        anyhow::anyhow!("general video codec startup memory preflight failed: {error}")
    })?;
    let codec_backend = match cpu_executor_backend {
        Some(backend) => backend,
        None => {
            let (backend, _authority) = BackendWorkspaceAuthority::create_backend(
                closure.codec_scratch_bytes(),
            )
            .map_err(|error| {
                anyhow::anyhow!("general video codec CPU backend initialization failed: {error}")
            })?;
            Arc::new(backend)
        }
    };
    let services = NativeVideoCodecWorkerServices::start(closure, codec_backend, &cancellation)
        .map_err(|error| anyhow::anyhow!("general video codec worker startup failed: {error}"))?;
    Ok(Some(services))
}

fn append_video_codec_cache_configuration(
    memory_configuration: String,
    services: Option<&NativeVideoCodecWorkerServices>,
) -> String {
    append_video_codec_cache_identity(
        memory_configuration,
        services.map(NativeVideoCodecWorkerServices::cache_configuration_sha256),
    )
}

fn append_video_codec_cache_identity(
    memory_configuration: String,
    cache_configuration_sha256: Option<&str>,
) -> String {
    match cache_configuration_sha256 {
        Some(cache_configuration_sha256) => {
            format!("{memory_configuration}:codec={cache_configuration_sha256}")
        }
        None => memory_configuration,
    }
}

fn validate_worker_plan_has_no_serialized_handles(
    worker_plan: &NativeImageWorkerPlan,
) -> Result<(), NativeImageRuntimeError> {
    for node in worker_plan.plan.nodes.values() {
        for (input_name, binding) in &node.inputs {
            if let InputBinding::Literal { value } = binding
                && native_value_contains_handle(value)
            {
                return Err(NativeImageRuntimeError::Encoding(format!(
                    "native worker plan node {:?} input `{input_name}` contains a process-local handle",
                    node.id
                )));
            }
        }
    }
    Ok(())
}

fn native_value_contains_handle(value: &NativeValue) -> bool {
    match value {
        NativeValue::Handle { .. } => true,
        NativeValue::List { values } => values.iter().any(native_value_contains_handle),
        NativeValue::Primitive { .. } | NativeValue::PreservedUnknown { .. } => false,
    }
}

fn backend_neutral_executor_unavailable(device: Option<DeviceId>) -> Option<BackendUnavailable> {
    let device = device?;
    if device == DeviceId::CPU {
        return None;
    }
    Some(BackendUnavailable::new(
        device.kind(),
        "the selected instance has no backend-neutral native image/diffusion executor binding for this graph; CPU fallback is forbidden",
    ))
}

fn prepare_native_image_memory(
    session: &WorkerSession,
    worker_plan: &NativeImageWorkerPlan,
    codec_bytes: u64,
) -> anyhow::Result<(AttemptMemoryController, String)> {
    let input_asset_bytes = worker_plan
        .input_assets
        .values()
        .try_fold(0_u64, |total, bytes| {
            let bytes = u64::try_from(bytes.len())
                .map_err(|_| anyhow::anyhow!("native image input byte accounting overflowed"))?;
            total
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("native image input byte accounting overflowed"))
        })?;
    let node_count = u64::try_from(worker_plan.plan.nodes.len())
        .map_err(|_| anyhow::anyhow!("native image node accounting overflowed"))?;
    let memory_request = native_image_memory_request_with_codec(
        input_asset_bytes,
        node_count,
        worker_plan.metadata_enabled,
        codec_bytes,
    )?;
    let memory_snapshot = session.memory_snapshot()?;
    let backend = session
        .accepted_backend()
        .ok_or_else(|| anyhow::anyhow!("native memory preflight has no accepted backend"))?;
    let effective_mode = EffectiveMemoryMode::resolve(
        MemoryModeRequest::from_runtime_policy(worker_plan.memory_policy),
        MemoryModeCapabilities::from_backend(backend),
    )?;
    let memory_configuration = effective_mode.configuration_token();
    let controller = AttemptMemoryController::new(
        memory_snapshot.limit_bytes,
        memory_snapshot.current_bytes,
        memory_request,
    )
    .map_err(|error| {
        anyhow::anyhow!("{error}; effective native memory configuration is {memory_configuration}")
    })?;
    Ok((controller, memory_configuration))
}

fn write_execution_event(
    session: &mut WorkerSession,
    stdout: &mut impl Write,
    event: AttemptEvent,
) -> anyhow::Result<()> {
    let kind = match event.kind {
        AttemptEventKind::Started => NativeImageWorkerProgressKind::Started,
        AttemptEventKind::Progress { completed, total } => {
            NativeImageWorkerProgressKind::Progress { completed, total }
        }
        AttemptEventKind::CacheHit => NativeImageWorkerProgressKind::CacheHit,
        AttemptEventKind::OutputPrepared { transaction_id } => {
            NativeImageWorkerProgressKind::OutputPrepared { transaction_id }
        }
        AttemptEventKind::Preview { .. }
        | AttemptEventKind::OutputAvailable { .. }
        | AttemptEventKind::CancelRequested { .. }
        | AttemptEventKind::Succeeded
        | AttemptEventKind::Failed { .. }
        | AttemptEventKind::Cancelled
        | AttemptEventKind::Interrupted { .. }
        | AttemptEventKind::RecoveryInterrupted { .. } => return Ok(()),
    };
    let progress = NativeImageWorkerProgress {
        profile_id: event.profile_id,
        prompt_id: event.prompt_id,
        attempt_id: event.attempt_id,
        sequence: event.sequence,
        node_id: event.node_id,
        kind,
    };
    let event = postcard::to_stdvec(&NativeImageWorkerEvent::Progress { progress })?;
    let response = session.execution_event(event)?;
    write_frame(stdout, &response)?;
    Ok(())
}

// The separately packaged worker has no GPUI context, so its protocol clock
// has one explicit async timer owner.
#[allow(clippy::disallowed_methods)]
async fn worker_delay(duration: Duration) {
    smol::Timer::after(duration).await;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use sim_types::{
        ProfileId, RequestId, WORKER_PROTOCOL_VERSION, WorkerId, WorkerMessage,
        WorkerRegistryDeploymentAck, WorkerRegistryDeploymentRejection,
        WorkerRegistryDeploymentRejectionReason, WorkerRegistryGeneration, WorkerSha256Digest,
    };

    use super::*;

    fn envelope(message: WorkerMessage) -> WorkerEnvelope {
        WorkerEnvelope {
            version: WORKER_PROTOCOL_VERSION,
            profile_id: ProfileId(Default::default()),
            worker_id: WorkerId(Default::default()),
            request_id: RequestId(Default::default()),
            prompt_id: None,
            attempt_id: None,
            sequence: 0,
            registry_version: "registry-v1".to_owned(),
            message,
            extensions: BTreeMap::new(),
        }
    }

    #[test]
    fn canonical_frame_round_trip_uses_little_endian_length() {
        let original = envelope(WorkerMessage::Ready);
        let mut frame = Vec::new();
        write_frame(&mut frame, &original).expect("encode frame");
        let payload_length = frame.len() - 4;
        assert_eq!(
            frame.get(..4),
            Some((payload_length as u32).to_le_bytes().as_slice())
        );
        assert_eq!(
            read_frame(frame.as_slice()).expect("decode frame"),
            original
        );
    }

    #[test]
    fn oversized_prefix_is_rejected_before_payload_allocation() {
        let prefix = u32::try_from(MAX_WORKER_FRAME_BYTES + 1)
            .expect("frame limit fits u32")
            .to_le_bytes();
        assert!(matches!(
            read_frame(prefix.as_slice()),
            Err(FrameError::TooLarge)
        ));
    }

    #[test]
    fn encoded_event_limit_is_independent_from_frame_limit() {
        let oversized = envelope(WorkerMessage::Event {
            event: vec![0; MAX_ENCODED_PREVIEW_BYTES + 1],
        });
        assert!(matches!(
            write_frame(Vec::new(), &oversized),
            Err(FrameError::PreviewTooLarge)
        ));
    }

    #[test]
    fn val_cancel_001_worker_control_uses_canonical_state() {
        let cancellation = CancellationToken::default();
        let clone = cancellation.clone();
        let cancel = WorkerMessage::Cancel {
            reason: "operator".to_owned(),
        };

        assert_eq!(
            apply_worker_control_cancellation(&cancel, &cancellation),
            Some(true)
        );
        assert!(clone.is_cancelled());
        assert_eq!(
            apply_worker_control_cancellation(&WorkerMessage::Shutdown, &clone),
            Some(false)
        );
        assert_eq!(
            apply_worker_control_cancellation(&WorkerMessage::Heartbeat, &clone),
            None
        );
    }

    #[test]
    fn rejected_provider_v2_finalization_cancels_before_active_execution_drop() {
        let accepted = CancellationToken::default();
        cancel_rejected_provider_v2_finalization(&Ok(()), &accepted);
        assert!(!accepted.is_cancelled());

        let rejected = CancellationToken::default();
        cancel_rejected_provider_v2_finalization(
            &Err(sim_types::WorkerProviderStreamError::StaleGeneration),
            &rejected,
        );
        assert!(rejected.is_cancelled());
    }

    #[test]
    fn provider_v2_cancellation_precedes_pending_route_drop() {
        struct CancellationObservedOnDrop {
            cancellation: CancellationToken,
            observed: Arc<AtomicBool>,
        }

        impl Drop for CancellationObservedOnDrop {
            fn drop(&mut self) {
                self.observed
                    .store(self.cancellation.is_cancelled(), Ordering::Release);
            }
        }

        let cancellation = CancellationToken::default();
        let observed = Arc::new(AtomicBool::new(false));
        let mut pending_proposal = Some(());
        let mut pending_streams = BTreeMap::from([(
            1,
            CancellationObservedOnDrop {
                cancellation: cancellation.clone(),
                observed: observed.clone(),
            },
        )]);

        assert!(
            clear_pending_provider_v2_after_cancellation(
                &cancellation,
                &mut pending_proposal,
                &mut pending_streams,
            )
            .is_err()
        );
        assert!(pending_proposal.is_some());
        assert_eq!(pending_streams.len(), 1);
        assert!(!observed.load(Ordering::Acquire));

        cancellation.cancel();
        assert!(
            clear_pending_provider_v2_after_cancellation(
                &cancellation,
                &mut pending_proposal,
                &mut pending_streams,
            )
            .expect("cancelled pending work can be cleared")
        );
        assert!(pending_proposal.is_none());
        assert!(pending_streams.is_empty());
        assert!(observed.load(Ordering::Acquire));

        let shutdown_cancellation = CancellationToken::default();
        let shutdown_observed = Arc::new(AtomicBool::new(false));
        let mut shutdown_proposal = Some(());
        let mut shutdown_streams = BTreeMap::from([(
            1,
            CancellationObservedOnDrop {
                cancellation: shutdown_cancellation.clone(),
                observed: shutdown_observed.clone(),
            },
        )]);
        assert_eq!(
            apply_worker_control_cancellation(&WorkerMessage::Shutdown, &shutdown_cancellation,),
            Some(true)
        );
        clear_pending_provider_v2_after_cancellation(
            &shutdown_cancellation,
            &mut shutdown_proposal,
            &mut shutdown_streams,
        )
        .expect("shutdown cancellation permits pending teardown");
        assert!(shutdown_proposal.is_none());
        assert!(shutdown_streams.is_empty());
        assert!(shutdown_observed.load(Ordering::Acquire));
    }

    #[test]
    fn serialized_worker_plan_rejects_nested_process_local_handles() {
        let handle_type =
            sim_runtime::NativeHandleType::new(sim_runtime::NativeHandleKind::Model, "MODEL")
                .expect("valid model handle type");
        let store_identity = sim_runtime::NativeHandleStoreIdentity::new(
            uuid::Uuid::from_u128(1),
            uuid::Uuid::from_u128(2),
        )
        .expect("valid store identity");
        let handle = sim_runtime::NativeOpaqueHandle::new(
            handle_type.clone(),
            store_identity,
            "forged-worker-handle",
            1,
            Some("a".repeat(64)),
        )
        .expect("structurally valid forged handle");
        let node_id = sim_types::NodeId::from("1");
        let descriptor = sim_runtime::NativeNodeDescriptor {
            schema_version: 1,
            class_type: "ForgedInput".to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: None,
            inputs: Vec::new(),
            dynamic_inputs: Vec::new(),
            outputs: Vec::new(),
            output_node: true,
            effect: sim_runtime::NativeEffectClass::Pure,
            cache: sim_runtime::NativeCachePolicy::InputIdentity,
        };
        let plan = sim_runtime::CompiledPlan {
            prompt_id: sim_types::PromptId(uuid::Uuid::from_u128(3)),
            client_id: None,
            prompt_number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
            nodes: BTreeMap::from([(
                node_id.clone(),
                sim_runtime::CompiledNode {
                    id: node_id.clone(),
                    class_type: "ForgedInput".to_owned(),
                    descriptor,
                    inputs: BTreeMap::from([(
                        "model".to_owned(),
                        InputBinding::Literal {
                            value: NativeValue::List {
                                values: vec![NativeValue::Handle { value: handle }],
                            },
                        },
                    )]),
                    unknown: BTreeMap::new(),
                },
            )]),
            topological_order: vec![node_id.clone()],
            static_required_nodes: std::collections::BTreeSet::from([node_id.clone()]),
            output_nodes: vec![node_id],
            provider_execution: None,
            persistence_unknown_fields: BTreeMap::new(),
        };
        let worker_plan = NativeImageWorkerPlan::new(plan, BTreeMap::new(), true, 0)
            .expect("plan is valid before the private worker trust boundary");
        let encoded = serde_json::to_vec(&worker_plan).expect("worker plan serializes");
        let decoded: NativeImageWorkerPlan =
            serde_json::from_slice(&encoded).expect("forged worker plan remains structural JSON");
        let error = validate_worker_plan_has_no_serialized_handles(&decoded)
            .expect_err("process-local handles must not cross private worker IPC");
        assert!(error.to_string().contains("process-local handle"));

        let primitive = NativeValue::Primitive {
            value: sim_runtime::NativePrimitive::Integer(7),
        };
        assert!(!native_value_contains_handle(&primitive));
        assert!(native_value_contains_handle(&NativeValue::Handle {
            value: sim_runtime::NativeOpaqueHandle::new(
                handle_type,
                store_identity,
                "second-handle",
                1,
                None,
            )
            .expect("valid handle"),
        }));
    }

    #[test]
    fn selected_accelerators_fail_before_cpu_executor_or_memory_preflight() {
        assert!(backend_neutral_executor_unavailable(Some(DeviceId::CPU)).is_none());
        for device in [
            DeviceId::new(DeviceKind::Metal, 0),
            DeviceId::new(DeviceKind::Rocm, 2),
            DeviceId::new(DeviceKind::Mlu, 0),
            DeviceId::new(DeviceKind::Npu, 0),
            DeviceId::new(DeviceKind::Cuda, 0),
            DeviceId::new(DeviceKind::Xpu, 0),
            DeviceId::new(DeviceKind::DirectMl, 0),
        ] {
            let unavailable = backend_neutral_executor_unavailable(Some(device))
                .expect("accelerator graph must fail closed");
            assert_eq!(unavailable.device(), device.kind());
            assert!(unavailable.reason().contains("CPU fallback is forbidden"));
        }
        assert!(backend_neutral_executor_unavailable(None).is_none());

        let source = include_str!("sim_worker.rs");
        let rejection = source
            .find("backend_neutral_executor_unavailable(session.backend_device())")
            .expect("selected backend rejection is present");
        let memory = source
            .find("prepare_native_image_memory(")
            .expect("memory preflight is present");
        let executor = source
            .find("NativeImageExecutor::new_with_diffusion_provider")
            .expect("CPU executor construction is present");
        assert!(rejection < memory);
        assert!(rejection < executor);
    }

    #[test]
    fn video_codec_package_bootstrap_uses_canonical_codec_memory_and_cache_dimensions() {
        let codec_bytes = 512 * 1024 * 1024;
        let request = native_image_memory_request_with_codec(1, 1, false, codec_bytes)
            .expect("native memory request must be bounded");
        assert_eq!(request.codec_bytes, codec_bytes);
        let plan = MemoryPlanner::plan(2 * 1024 * 1024 * 1024, 17, request)
            .expect("codec reservation and canonical margin must fit");
        assert_eq!(plan.durable_baseline_bytes, 17);
        assert!(plan.reservations.iter().any(|reservation| {
            reservation.kind == MemoryReservationKind::Codec && reservation.bytes == codec_bytes
        }));
        assert!(plan.reservations.iter().any(|reservation| {
            reservation.kind == MemoryReservationKind::SafetyMargin
                && reservation.bytes == plan.safety_margin_bytes
        }));
        let exact_capacity = plan.committed_target_bytes;
        MemoryPlanner::plan(exact_capacity, 17, request)
            .expect("the exact canonical capacity must admit the codec reservation");
        assert!(MemoryPlanner::plan(exact_capacity - 1, 17, request).is_err());

        let memory_configuration = "memory-v1".to_owned();
        assert_eq!(
            append_video_codec_cache_identity(memory_configuration.clone(), None),
            memory_configuration
        );
        assert_eq!(
            append_video_codec_cache_identity("memory-v1".to_owned(), Some("codec-v1")),
            "memory-v1:codec=codec-v1"
        );

        let source = include_str!("sim_worker.rs");
        let bootstrap = source
            .find("initialize_general_video_codec_worker_services(")
            .expect("package bootstrap must be retained");
        let worker_loop = source
            .find("'worker: loop")
            .expect("worker loop must be explicit");
        let memory = source[worker_loop..]
            .find("prepare_native_image_memory(")
            .map(|offset| worker_loop + offset)
            .expect("attempt memory admission must be explicit");
        let cache = source[worker_loop..]
            .find("append_video_codec_cache_configuration(")
            .map(|offset| worker_loop + offset)
            .expect("cache configuration must bind the codec actor identity");
        let attach = source[worker_loop..]
            .find("with_ltxv_preprocess_service")
            .map(|offset| worker_loop + offset)
            .expect("executor must attach the existing actor ports");
        assert!(
            bootstrap < worker_loop && worker_loop < memory && memory < cache && cache < attach
        );
        assert!(source.contains("MemoryPlanner::plan(\n        memory_limit_bytes,\n        0,"));
        assert!(source.contains("NativeVideoCodecWorkerServices::codec_residency_bytes"));
    }

    #[cfg(feature = "mlu")]
    #[test]
    fn unavailable_mlu_selection_never_constructs_a_cpu_fallback() {
        let package = sim_runtime::NativeMluPackageSettings::from_public_authority(
            "/missing/reviewed-mlu-package",
            "mlu.release",
            &"44".repeat(32),
        )
        .expect("bounded public MLU authority");
        let (session, cpu_backend) = initialize_worker_backend(
            WorkerBackendSelection::Mlu {
                package,
                device_ordinal: 0,
            },
            4096,
        );
        let unavailable = match session {
            Ok(_) => panic!("missing MLU trust package must fail closed"),
            Err(unavailable) => unavailable,
        };
        assert_eq!(unavailable.device(), DeviceKind::Mlu);
        assert!(cpu_backend.is_none());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn unavailable_cuda_selection_never_constructs_a_cpu_fallback() {
        let package = sim_runtime::NativeCudaPackageSettings::from_public_authority(
            "/missing/reviewed-cuda-package",
            "cuda.release",
            &"56".repeat(32),
        )
        .expect("bounded public CUDA authority");
        let (session, cpu_backend) = initialize_worker_backend(
            WorkerBackendSelection::Cuda {
                package,
                device_ordinal: 0,
            },
            4096,
        );
        let unavailable = match session {
            Ok(_) => panic!("missing CUDA trust package must fail closed"),
            Err(unavailable) => unavailable,
        };
        assert_eq!(unavailable.device(), DeviceKind::Cuda);
        assert!(cpu_backend.is_none());
    }

    #[cfg(feature = "npu")]
    #[test]
    fn unavailable_npu_selection_never_constructs_a_cpu_fallback() {
        let package = sim_runtime::NativeNpuPackageSettings::from_public_authority(
            "/missing/reviewed-npu-package",
            "npu.release",
            &"45".repeat(32),
        )
        .expect("bounded public NPU authority");
        let (session, cpu_backend) = initialize_worker_backend(
            WorkerBackendSelection::Npu {
                package,
                device_ordinal: 0,
            },
            4096,
        );
        let unavailable = match session {
            Ok(_) => panic!("missing NPU trust package must fail closed"),
            Err(unavailable) => unavailable,
        };
        assert_eq!(unavailable.device(), DeviceKind::Npu);
        assert!(cpu_backend.is_none());
    }

    #[cfg(feature = "xpu")]
    #[test]
    fn unavailable_xpu_selection_never_constructs_a_cpu_fallback() {
        let package = sim_runtime::NativeXpuPackageSettings::from_public_authority(
            "/missing/reviewed-xpu-package",
            "xpu.release",
            &"46".repeat(32),
        )
        .expect("bounded public XPU authority");
        let (session, cpu_backend) = initialize_worker_backend(
            WorkerBackendSelection::Xpu {
                package,
                device_ordinal: 0,
            },
            4096,
        );
        let unavailable = match session {
            Ok(_) => panic!("missing XPU trust package must fail closed"),
            Err(unavailable) => unavailable,
        };
        assert_eq!(unavailable.device(), DeviceKind::Xpu);
        assert!(cpu_backend.is_none());
    }

    #[cfg(feature = "directml")]
    #[test]
    fn unavailable_directml_selection_never_constructs_a_cpu_fallback() {
        let package = sim_runtime::NativeDirectMlPackageSettings::from_public_authority(
            "/missing/reviewed-directml-package",
            "directml.release",
            &"44".repeat(32),
        )
        .expect("bounded public DirectML authority");
        let (session, cpu_backend) =
            initialize_worker_backend(WorkerBackendSelection::DirectMl { package }, 4096);
        let unavailable = match session {
            Ok(_) => panic!("missing DirectML trust package must fail closed"),
            Err(unavailable) => unavailable,
        };
        assert_eq!(unavailable.device(), DeviceKind::DirectMl);
        assert!(cpu_backend.is_none());
    }

    #[test]
    fn rejected_process_registry_keeps_the_compiled_generation_executable() {
        let profile_id = ProfileId(Default::default());
        let initial_generation = WorkerRegistryGeneration::new(1).expect("nonzero generation");
        let initial_digest = WorkerSha256Digest::new("a".repeat(64)).expect("valid digest");
        let initial_source = Arc::new(AssembledWorkerRegistry::empty_for_test(
            initial_generation,
            initial_digest.clone(),
        ));
        let initial_compiled = Arc::new(
            plugin_runtime::WorkerPluginRegistry::empty_for_test(
                profile_id,
                initial_generation,
                initial_digest.clone(),
            )
            .expect("empty compiled registry"),
        );
        let mut current = Some(CommittedPluginRegistry {
            source: initial_source,
            compiled: initial_compiled,
        });

        let replacement_generation = WorkerRegistryGeneration::new(2).expect("nonzero generation");
        let replacement_digest = WorkerSha256Digest::new("b".repeat(64)).expect("valid digest");
        let replacement_source = Arc::new(AssembledWorkerRegistry::empty_for_test(
            replacement_generation,
            replacement_digest.clone(),
        ));
        let rejection = envelope(WorkerMessage::RegistryDeploymentRejected {
            rejection: WorkerRegistryDeploymentRejection::new(
                replacement_generation,
                replacement_digest,
                WorkerRegistryDeploymentRejectionReason::ComponentCompilationFailed,
            ),
        });
        assert!(
            !apply_compiled_registry_commit(
                &mut current,
                &[rejection],
                Some(replacement_source),
                None,
            )
            .expect("rejection is a non-mutating process transition")
        );

        let current = current.expect("previous registry remains committed");
        assert_eq!(current.source.generation(), initial_generation);
        let invocation = sim_plugin_host::WorkerPluginInvocation::from_bytes(
            &serde_json::to_vec(&serde_json::json!({
                "registry_generation": initial_generation.get(),
                "registry_digest_sha256": initial_digest.as_str(),
                "extension_id": "missing.extension",
                "extension_version": "1.0.0",
                "plugin_identifier": "missing.plugin",
                "plugin_version": "1.0.0",
                "manifest_digest_sha256": "e".repeat(64),
                "component_digest_sha256": "c".repeat(64),
                "authorization_generation": "d".repeat(64),
                "node_id": "missing-node",
                "inputs": { "values": {} },
                "timeout_milliseconds": 1_000,
                "maximum_response_bytes": 1_024,
                "component_limits": sim_plugin_host::ComponentLimits::default(),
            }))
            .expect("old-generation invocation serializes"),
        )
        .expect("old-generation invocation is valid");
        let (capability_sender, _capability_receiver) = async_channel::bounded(1);
        let result = current.compiled.execute(
            invocation,
            Arc::new(plugin_runtime::WorkerCapabilityBridge::new(
                capability_sender,
            )),
            CancellationToken::default(),
        );
        assert!(matches!(
            result,
            Err(plugin_runtime::WorkerPluginRuntimeError::MissingComponent)
        ));
    }

    #[test]
    fn accepted_registry_change_requires_native_executor_recreation() {
        let profile_id = ProfileId(Default::default());
        let first_generation = WorkerRegistryGeneration::new(1).expect("nonzero generation");
        let first_digest = WorkerSha256Digest::new("a".repeat(64)).expect("valid digest");
        let mut current = Some(CommittedPluginRegistry {
            source: Arc::new(AssembledWorkerRegistry::empty_for_test(
                first_generation,
                first_digest.clone(),
            )),
            compiled: Arc::new(
                plugin_runtime::WorkerPluginRegistry::empty_for_test(
                    profile_id,
                    first_generation,
                    first_digest,
                )
                .expect("empty compiled registry"),
            ),
        });
        let next_generation = WorkerRegistryGeneration::new(2).expect("nonzero generation");
        let next_digest = WorkerSha256Digest::new("b".repeat(64)).expect("valid digest");
        let acknowledgement = envelope(WorkerMessage::RegistryDeploymentAck {
            acknowledgement: WorkerRegistryDeploymentAck::new(
                next_generation,
                next_digest.clone(),
                0,
            )
            .expect("bounded acknowledgement"),
        });
        let next_source = Arc::new(AssembledWorkerRegistry::empty_for_test(
            next_generation,
            next_digest.clone(),
        ));
        let next_compiled = Arc::new(
            plugin_runtime::WorkerPluginRegistry::empty_for_test(
                profile_id,
                next_generation,
                next_digest,
            )
            .expect("empty compiled registry"),
        );
        assert!(
            apply_compiled_registry_commit(
                &mut current,
                &[acknowledgement],
                Some(next_source),
                Some(next_compiled),
            )
            .expect("accepted replacement commits atomically")
        );
        assert_eq!(
            current
                .as_ref()
                .expect("replacement is committed")
                .source
                .generation(),
            next_generation
        );
    }

    #[test]
    fn provider_registry_pin_requires_a_matching_committed_registry() {
        let profile_id = ProfileId(Default::default());
        let generation = WorkerRegistryGeneration::new(3).expect("nonzero generation");
        let digest = WorkerSha256Digest::new("a".repeat(64)).expect("valid digest");
        let source = Arc::new(AssembledWorkerRegistry::empty_for_test(
            generation,
            digest.clone(),
        ));
        let compiled = Arc::new(
            plugin_runtime::WorkerPluginRegistry::empty_for_test(profile_id, generation, digest)
                .expect("empty compiled registry"),
        );
        let registry = CommittedPluginRegistry { source, compiled };
        let pin = sim_runtime::NativeProviderRegistryPin::checked(
            generation.get(),
            "a".repeat(64),
            vec!["b".repeat(64)],
        )
        .expect("valid provider registry pin");

        assert!(!provider_registry_pin_is_available(None, &pin));
        assert!(!provider_registry_pin_is_available(Some(&registry), &pin));
    }
}
