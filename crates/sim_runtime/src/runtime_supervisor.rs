use std::{
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sim_tensor::{BackendCapabilityMatrix, CpuBackend, DeviceId};
#[cfg(feature = "test-support")]
use sim_types::WorkerPluginExecutionFailure;
use sim_types::{
    AttemptId, BackendUnavailable, DeviceKind, MAX_ENCODED_PREVIEW_BYTES,
    MAX_WORKER_COMPONENT_CHUNK_BYTES, MAX_WORKER_FRAME_BYTES,
    MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES, ProfileId, PromptId, RequestId,
    WORKER_PROTOCOL_VERSION, WorkerComponentContent, WorkerEnvelope, WorkerId,
    WorkerLifecycleEvent, WorkerMessage, WorkerModelSourceResponse, WorkerPluginExecutionOutcome,
    WorkerProviderStreamRequest, WorkerProviderStreamResponse,
    WorkerProviderStreamTransportValidator, WorkerProviderV2ProposalFinalization,
    WorkerRegistryDeploymentAck, WorkerRegistryDeploymentBegin, WorkerRegistryDeploymentChunk,
    WorkerRegistryDeploymentCommit, decode_worker_frame, encode_worker_frame,
};
use smol::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use thiserror::Error;
use util::process::Child;
use uuid::Uuid;

use crate::{
    NativeCudaPackageSettings, NativeDirectMlPackageSettings,
    NativeGeneralVideoCodecPackageSettings, NativeMetalPackageSettings, NativeMluPackageSettings,
    NativeNpuPackageSettings, NativeRocmPackageSettings, NativeRuntimeProfile,
    NativeXpuPackageSettings, PluginAuthorizationVerifier, PluginCapabilityInvocation,
    PluginServiceWireFailure, PluginServiceWireRequest, PluginServiceWireResponse,
    ResolvedProviderResult,
};
use sim_plugin_sdk::ProviderResultReceiptSet;

pub const WORKER_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
pub const WORKER_MISSED_HEARTBEAT_LIMIT: u8 = 3;
pub const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
pub const WORKER_READY_TIMEOUT: Duration = Duration::from_secs(10);
pub const MAX_CAPTURED_WORKER_LOG_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_AUTOMATIC_WORKER_RESTARTS: u8 = 1;
pub const WORKER_RESTART_BACKOFF: Duration = Duration::from_millis(250);
pub const MAX_PENDING_WORKER_MESSAGES: usize = 64;
pub const MAX_TRACKED_WORKER_REQUESTS: usize = 8;

type WorkerInput = Box<dyn AsyncWrite + Send + Unpin>;

pub(crate) struct RuntimeProviderV2StreamCall {
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes provider-v2 stream call identities"
        )
    )]
    pub(crate) call_id: u64,
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes this provider-v2 worker route"
        )
    )]
    pub(crate) request: WorkerProviderStreamRequest,
    response: async_channel::Sender<WorkerProviderStreamResponse>,
}

impl RuntimeProviderV2StreamCall {
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes provider-v2 stream call identities"
        )
    )]
    pub(crate) const fn call_id(&self) -> u64 {
        self.call_id
    }

    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes provider-v2 stream requests"
        )
    )]
    pub(crate) fn request(&self) -> &WorkerProviderStreamRequest {
        &self.request
    }

    #[cfg_attr(
        not(any(test, feature = "test-support")),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes this provider-v2 worker route"
        )
    )]
    pub(crate) fn respond(
        self,
        response: WorkerProviderStreamResponse,
    ) -> Result<(), RuntimeSupervisorError> {
        self.response.try_send(response).map_err(|error| {
            RuntimeSupervisorError::Protocol(format!(
                "provider-v2 stream response could not be delivered: {error}"
            ))
        })
    }
}

pub(crate) struct RuntimeProviderV2Proposal {
    #[cfg_attr(
        not(any(test, feature = "test-support")),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes this provider-v2 worker route"
        )
    )]
    pub(crate) outcome: WorkerPluginExecutionOutcome,
    finalization: async_channel::Sender<RuntimeProviderV2FinalizedProposal>,
}

#[cfg_attr(
    not(any(test, feature = "test-support")),
    expect(
        dead_code,
        reason = "Task427 deployment actuator consumes finalized provider-v2 proposals"
    )
)]
pub(crate) struct RuntimeProviderV2FinalizedProposal {
    finalization: WorkerProviderV2ProposalFinalization,
    materialization: crate::ProviderTransportResponse,
}

impl RuntimeProviderV2Proposal {
    #[cfg_attr(
        not(feature = "test-support"),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes provider-v2 proposal outcomes"
        )
    )]
    pub(crate) fn outcome(&self) -> &WorkerPluginExecutionOutcome {
        &self.outcome
    }

    #[cfg_attr(
        not(any(test, feature = "test-support")),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes this provider-v2 worker route"
        )
    )]
    pub(crate) fn finalize(
        self,
        finalization: WorkerProviderV2ProposalFinalization,
        materialization: crate::ProviderTransportResponse,
    ) -> Result<(), RuntimeSupervisorError> {
        self.finalization
            .try_send(RuntimeProviderV2FinalizedProposal {
                finalization,
                materialization,
            })
            .map_err(|error| {
                RuntimeSupervisorError::Protocol(format!(
                    "provider-v2 finalization could not be delivered: {error}"
                ))
            })
    }
}

#[cfg_attr(
    not(feature = "test-support"),
    expect(
        dead_code,
        reason = "Task427 deployment actuator consumes the provider-v2 supervisor bridge"
    )
)]
pub(crate) struct RuntimeProviderV2Bridge {
    stream_calls: async_channel::Sender<RuntimeProviderV2StreamCall>,
    proposal: async_channel::Sender<RuntimeProviderV2Proposal>,
    validator: WorkerProviderStreamTransportValidator,
    cancellation: sim_types::CancellationToken,
    invocation_timeout: Duration,
}

#[cfg_attr(
    not(any(test, feature = "test-support")),
    expect(
        dead_code,
        reason = "Task427 deployment actuator consumes provider-v2 finalization phases"
    )
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProviderV2FinalizationPhase {
    PreCommit,
    Committed,
}

#[cfg_attr(
    not(any(test, feature = "test-support")),
    expect(
        dead_code,
        reason = "Task427 deployment actuator consumes provider-v2 cancellation transitions"
    )
)]
fn should_begin_provider_v2_cancellation(
    phase: ProviderV2FinalizationPhase,
    cancellation: &sim_types::CancellationToken,
    cancellation_request_id: Option<RequestId>,
) -> bool {
    phase == ProviderV2FinalizationPhase::PreCommit
        && cancellation.is_cancelled()
        && cancellation_request_id.is_none()
}

#[cfg_attr(
    not(any(test, feature = "test-support")),
    expect(
        dead_code,
        reason = "Task427 deployment actuator consumes provider-v2 precommit suppression"
    )
)]
fn suppress_provider_v2_precommit_message(
    phase: ProviderV2FinalizationPhase,
    cancellation_request_id: Option<RequestId>,
    message: &WorkerMessage,
) -> bool {
    phase == ProviderV2FinalizationPhase::PreCommit
        && cancellation_request_id.is_some()
        && matches!(
            message,
            WorkerMessage::ProviderStreamRequest { .. }
                | WorkerMessage::PluginResult {
                    outcome: WorkerPluginExecutionOutcome::Succeeded(_)
                }
        )
}

#[cfg_attr(
    not(any(test, feature = "test-support")),
    expect(
        dead_code,
        reason = "Task427 deployment actuator consumes provider-v2 cancellation close precedence"
    )
)]
fn provider_v2_wait_close_is_cancellation(
    phase: ProviderV2FinalizationPhase,
    cancellation: &sim_types::CancellationToken,
) -> bool {
    phase == ProviderV2FinalizationPhase::PreCommit && cancellation.is_cancelled()
}

impl RuntimeProviderV2Bridge {
    #[cfg_attr(
        not(any(test, feature = "test-support")),
        expect(
            dead_code,
            reason = "Task427 deployment actuator consumes this provider-v2 worker route"
        )
    )]
    pub(crate) fn capacity_one(
        context: sim_types::WorkerProviderInvocationContext,
        contract: sim_types::WorkerProviderStreamingContract,
        cancellation: sim_types::CancellationToken,
        invocation_timeout: Duration,
    ) -> Result<
        (
            Self,
            async_channel::Receiver<RuntimeProviderV2StreamCall>,
            async_channel::Receiver<RuntimeProviderV2Proposal>,
        ),
        RuntimeSupervisorError,
    > {
        if invocation_timeout.is_zero() {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "provider-v2 invocation timeout must be non-zero".to_owned(),
            ));
        }
        let (stream_calls, stream_receiver) = async_channel::bounded(1);
        let (proposal, proposal_receiver) = async_channel::bounded(1);
        Ok((
            Self {
                stream_calls,
                proposal,
                validator: WorkerProviderStreamTransportValidator::checked_for_host_session(
                    context,
                    contract,
                    cancellation.clone(),
                )
                .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))?,
                cancellation,
                invocation_timeout,
            },
            stream_receiver,
            proposal_receiver,
        ))
    }
}

pub struct RetainedPluginExecution {
    outcome: WorkerPluginExecutionOutcome,
    capability_invocation: Option<PluginCapabilityInvocation>,
}

impl RetainedPluginExecution {
    pub fn outcome(&self) -> &WorkerPluginExecutionOutcome {
        &self.outcome
    }

    pub fn is_provider_result_structured_transport_response(
        &self,
        receipt: &[u8],
        expected_identity: &crate::ProviderInvocationIdentity,
        context: &sim_nodes::NativeNodeContext,
    ) -> Result<bool, RuntimeSupervisorError> {
        self.capability_invocation
            .as_ref()
            .ok_or_else(|| {
                RuntimeSupervisorError::PluginCapabilityBroker(
                    "provider capability invocation is already terminal".to_owned(),
                )
            })?
            .is_provider_result_structured_transport_response(receipt, expected_identity, context)
            .map_err(|error| RuntimeSupervisorError::PluginCapabilityBroker(error.to_string()))
    }

    pub fn materialize_provider_downloaded_image(
        &self,
        receipt: &[u8],
        expected_identity: &crate::ProviderInvocationIdentity,
        context: &sim_nodes::NativeNodeContext,
    ) -> Result<sim_nodes::NativeStoredPayload, RuntimeSupervisorError> {
        self.capability_invocation
            .as_ref()
            .ok_or_else(|| {
                RuntimeSupervisorError::PluginCapabilityBroker(
                    "provider capability invocation is already terminal".to_owned(),
                )
            })?
            .materialize_provider_downloaded_image(receipt, expected_identity, context)
            .map_err(|error| RuntimeSupervisorError::PluginCapabilityBroker(error.to_string()))
    }

    #[cfg(test)]
    pub(crate) fn with_capability_invocation_for_test(
        outcome: WorkerPluginExecutionOutcome,
        capability_invocation: crate::PluginCapabilityInvocation,
    ) -> Self {
        Self {
            outcome,
            capability_invocation: Some(capability_invocation),
        }
    }

    pub fn resolve_provider_result_receipt_set(
        &mut self,
        receipt_set: &ProviderResultReceiptSet,
    ) -> Result<Vec<ResolvedProviderResult>, RuntimeSupervisorError> {
        self.capability_invocation
            .as_mut()
            .ok_or_else(|| {
                RuntimeSupervisorError::PluginCapabilityBroker(
                    "provider capability invocation is already terminal".to_owned(),
                )
            })?
            .resolve_provider_result_receipt_set(receipt_set)
            .map_err(|error| RuntimeSupervisorError::PluginCapabilityBroker(error.to_string()))
    }

    pub fn finish(mut self) -> Result<WorkerPluginExecutionOutcome, RuntimeSupervisorError> {
        if let Some(invocation) = self.capability_invocation.take() {
            invocation.finish().map_err(|error| {
                RuntimeSupervisorError::PluginCapabilityBroker(error.to_string())
            })?;
        }
        Ok(self.outcome)
    }

    pub fn abort(mut self) -> WorkerPluginExecutionOutcome {
        if let Some(invocation) = self.capability_invocation.take() {
            invocation.abort();
        }
        self.outcome
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SupervisorPolicy {
    pub heartbeat_interval: Duration,
    pub missed_heartbeat_limit: u8,
    pub shutdown_timeout: Duration,
    pub ready_timeout: Duration,
    pub maximum_automatic_restarts: u8,
    pub restart_backoff: Duration,
}

impl Default for SupervisorPolicy {
    fn default() -> Self {
        Self {
            heartbeat_interval: WORKER_HEARTBEAT_INTERVAL,
            missed_heartbeat_limit: WORKER_MISSED_HEARTBEAT_LIMIT,
            shutdown_timeout: WORKER_SHUTDOWN_TIMEOUT,
            ready_timeout: WORKER_READY_TIMEOUT,
            maximum_automatic_restarts: MAXIMUM_AUTOMATIC_WORKER_RESTARTS,
            restart_backoff: WORKER_RESTART_BACKOFF,
        }
    }
}

impl SupervisorPolicy {
    fn validate(self) -> Result<(), RuntimeSupervisorError> {
        if self.heartbeat_interval.is_zero()
            || self.missed_heartbeat_limit == 0
            || self.shutdown_timeout.is_zero()
            || self.ready_timeout.is_zero()
        {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker timeouts and missed-heartbeat limit must be non-zero".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct WorkerLaunchConfig {
    pub binary: PathBuf,
    pub arguments: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub environment: Vec<(String, String)>,
    pub profile_id: ProfileId,
    pub worker_id: WorkerId,
    pub registry_version: String,
    pub backend: BackendCapabilityMatrix,
    pub backend_selection: WorkerBackendSelection,
    pub general_video_codec_package: Option<NativeGeneralVideoCodecPackageSettings>,
    pub memory_limit_bytes: u64,
    pub policy: SupervisorPolicy,
    pub registry_deployment: Option<WorkerRegistryDeploymentPlan>,
    pub plugin_authorization_verifier: Option<PluginAuthorizationVerifier>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkerBackendSelection {
    Cpu,
    DirectMl {
        package: NativeDirectMlPackageSettings,
    },
    Rocm {
        package: NativeRocmPackageSettings,
        device_ordinal: u32,
    },
    Metal {
        package: NativeMetalPackageSettings,
    },
    Mlu {
        package: NativeMluPackageSettings,
        device_ordinal: u32,
    },
    Npu {
        package: NativeNpuPackageSettings,
        device_ordinal: u32,
    },
    Cuda {
        package: NativeCudaPackageSettings,
        device_ordinal: u32,
    },
    Xpu {
        package: NativeXpuPackageSettings,
        device_ordinal: u32,
    },
}

impl WorkerBackendSelection {
    pub const fn device(&self) -> DeviceId {
        match self {
            Self::Cpu => DeviceId::CPU,
            Self::DirectMl { .. } => DeviceId::new(DeviceKind::DirectMl, 0),
            Self::Rocm { device_ordinal, .. } => DeviceId::new(DeviceKind::Rocm, *device_ordinal),
            Self::Metal { .. } => DeviceId::new(DeviceKind::Metal, 0),
            Self::Mlu { device_ordinal, .. } => DeviceId::new(DeviceKind::Mlu, *device_ordinal),
            Self::Npu { device_ordinal, .. } => DeviceId::new(DeviceKind::Npu, *device_ordinal),
            Self::Cuda { device_ordinal, .. } => DeviceId::new(DeviceKind::Cuda, *device_ordinal),
            Self::Xpu { device_ordinal, .. } => DeviceId::new(DeviceKind::Xpu, *device_ordinal),
        }
    }

    fn launch_arguments(&self) -> Result<Vec<String>, RuntimeSupervisorError> {
        match self {
            Self::Cpu => Ok(vec!["--backend".to_owned(), "cpu".to_owned()]),
            Self::DirectMl { package } => {
                let package_root = package.package_root().to_str().ok_or_else(|| {
                    RuntimeSupervisorError::InvalidConfiguration(
                        "DirectML package root is not valid UTF-8".to_owned(),
                    )
                })?;
                Ok(vec![
                    "--backend".to_owned(),
                    "directml".to_owned(),
                    "--directml-package-root".to_owned(),
                    package_root.to_owned(),
                    "--directml-package-signer".to_owned(),
                    package.verification_key().signer().to_owned(),
                    "--directml-package-public-key".to_owned(),
                    package.public_key_hex(),
                ])
            }
            Self::Rocm {
                package,
                device_ordinal,
            } => {
                let package_root = package.package_root().to_str().ok_or_else(|| {
                    RuntimeSupervisorError::InvalidConfiguration(
                        "ROCm package root is not valid UTF-8".to_owned(),
                    )
                })?;
                Ok(vec![
                    "--backend".to_owned(),
                    "rocm".to_owned(),
                    "--backend-device-ordinal".to_owned(),
                    device_ordinal.to_string(),
                    "--rocm-package-root".to_owned(),
                    package_root.to_owned(),
                    "--rocm-package-signer".to_owned(),
                    package.verification_key().signer().to_owned(),
                    "--rocm-package-public-key".to_owned(),
                    package.public_key_hex(),
                ])
            }
            Self::Metal { package } => {
                let package_root = package.package_root().to_str().ok_or_else(|| {
                    RuntimeSupervisorError::InvalidConfiguration(
                        "Metal package root is not valid UTF-8".to_owned(),
                    )
                })?;
                Ok(vec![
                    "--backend".to_owned(),
                    "metal".to_owned(),
                    "--metal-package-root".to_owned(),
                    package_root.to_owned(),
                    "--metal-package-signer".to_owned(),
                    package.verification_key().signer().to_owned(),
                    "--metal-package-public-key".to_owned(),
                    package.public_key_hex(),
                ])
            }
            Self::Mlu {
                package,
                device_ordinal,
            } => {
                let package_root = package.package_root().to_str().ok_or_else(|| {
                    RuntimeSupervisorError::InvalidConfiguration(
                        "MLU package root is not valid UTF-8".to_owned(),
                    )
                })?;
                Ok(vec![
                    "--backend".to_owned(),
                    "mlu".to_owned(),
                    "--backend-device-ordinal".to_owned(),
                    device_ordinal.to_string(),
                    "--mlu-package-root".to_owned(),
                    package_root.to_owned(),
                    "--mlu-package-signer".to_owned(),
                    package.verification_key().signer().to_owned(),
                    "--mlu-package-public-key".to_owned(),
                    package.public_key_hex(),
                ])
            }
            Self::Npu {
                package,
                device_ordinal,
            } => {
                let package_root = package.package_root().to_str().ok_or_else(|| {
                    RuntimeSupervisorError::InvalidConfiguration(
                        "NPU package root is not valid UTF-8".to_owned(),
                    )
                })?;
                Ok(vec![
                    "--backend".to_owned(),
                    "npu".to_owned(),
                    "--backend-device-ordinal".to_owned(),
                    device_ordinal.to_string(),
                    "--npu-package-root".to_owned(),
                    package_root.to_owned(),
                    "--npu-package-signer".to_owned(),
                    package.verification_key().signer().to_owned(),
                    "--npu-package-public-key".to_owned(),
                    package.public_key_hex(),
                ])
            }
            Self::Cuda {
                package,
                device_ordinal,
            } => {
                let package_root = package.package_root().to_str().ok_or_else(|| {
                    RuntimeSupervisorError::InvalidConfiguration(
                        "CUDA package root is not valid UTF-8".to_owned(),
                    )
                })?;
                Ok(vec![
                    "--backend".to_owned(),
                    "cuda".to_owned(),
                    "--backend-device-ordinal".to_owned(),
                    device_ordinal.to_string(),
                    "--cuda-package-root".to_owned(),
                    package_root.to_owned(),
                    "--cuda-package-signer".to_owned(),
                    package.verification_key().signer().to_owned(),
                    "--cuda-package-public-key".to_owned(),
                    package.public_key_hex(),
                ])
            }
            Self::Xpu {
                package,
                device_ordinal,
            } => {
                let package_root = package.package_root().to_str().ok_or_else(|| {
                    RuntimeSupervisorError::InvalidConfiguration(
                        "XPU package root is not valid UTF-8".to_owned(),
                    )
                })?;
                Ok(vec![
                    "--backend".to_owned(),
                    "xpu".to_owned(),
                    "--backend-device-ordinal".to_owned(),
                    device_ordinal.to_string(),
                    "--xpu-package-root".to_owned(),
                    package_root.to_owned(),
                    "--xpu-package-signer".to_owned(),
                    package.verification_key().signer().to_owned(),
                    "--xpu-package-public-key".to_owned(),
                    package.public_key_hex(),
                ])
            }
        }
    }
}

fn general_video_codec_package_launch_arguments(
    package: Option<&NativeGeneralVideoCodecPackageSettings>,
) -> Result<Vec<String>, RuntimeSupervisorError> {
    let Some(package) = package else {
        return Ok(Vec::new());
    };
    let package_root = package.package_root().to_str().ok_or_else(|| {
        RuntimeSupervisorError::InvalidConfiguration(
            "general video codec package root is not valid UTF-8".to_owned(),
        )
    })?;
    Ok(vec![
        "--video-codec-package-root".to_owned(),
        package_root.to_owned(),
        "--video-codec-package-signer".to_owned(),
        package.verification_key().signer().to_owned(),
        "--video-codec-package-public-key".to_owned(),
        package.public_key_hex(),
    ])
}

#[derive(Clone, Debug)]
pub struct WorkerRegistryDeploymentPlan {
    begin: WorkerRegistryDeploymentBegin,
    chunks: Vec<WorkerRegistryDeploymentChunk>,
    authorization_verifier: PluginAuthorizationVerifier,
}

impl WorkerRegistryDeploymentPlan {
    pub fn new(
        begin: WorkerRegistryDeploymentBegin,
        chunks: Vec<WorkerRegistryDeploymentChunk>,
        authorization_verifier: PluginAuthorizationVerifier,
    ) -> Result<Self, RuntimeSupervisorError> {
        validate_registry_chunks(&begin, &chunks)?;
        Ok(Self {
            begin,
            chunks,
            authorization_verifier,
        })
    }

    pub fn begin(&self) -> &WorkerRegistryDeploymentBegin {
        &self.begin
    }

    pub fn chunks(&self) -> &[WorkerRegistryDeploymentChunk] {
        &self.chunks
    }

    pub fn authorization_verifier(&self) -> &PluginAuthorizationVerifier {
        &self.authorization_verifier
    }
}

impl WorkerLaunchConfig {
    pub fn for_packaged_worker(
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        Ok(Self::new(
            packaged_worker_binary()?,
            profile_id,
            worker_id,
            registry_version,
            memory_limit_bytes,
        ))
    }

    pub fn for_packaged_worker_device(
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        device: DeviceKind,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        Self::for_device(
            packaged_worker_binary()?,
            profile_id,
            worker_id,
            registry_version,
            device,
            memory_limit_bytes,
        )
    }

    pub fn for_packaged_worker_profile(
        profile: &NativeRuntimeProfile,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let profile_id = ProfileId(profile.id);
        let config = match profile.device {
            DeviceKind::Cpu => Ok(Self::new(
                packaged_worker_binary()?,
                profile_id,
                worker_id,
                registry_version,
                memory_limit_bytes,
            )),
            DeviceKind::Rocm => {
                let package = profile.rocm_package.clone().ok_or_else(|| {
                    BackendUnavailable::new(
                        DeviceKind::Rocm,
                        "the selected native profile has no signed ROCm package authority",
                    )
                })?;
                Self::for_rocm(
                    packaged_worker_binary()?,
                    profile_id,
                    worker_id,
                    registry_version,
                    package,
                    0,
                    memory_limit_bytes,
                )
            }
            DeviceKind::DirectMl => {
                let package = profile.directml_package.clone().ok_or_else(|| {
                    BackendUnavailable::new(
                        DeviceKind::DirectMl,
                        "the selected native profile has no signed DirectML package authority",
                    )
                })?;
                Self::for_directml(
                    packaged_worker_binary()?,
                    profile_id,
                    worker_id,
                    registry_version,
                    package,
                    memory_limit_bytes,
                )
            }
            DeviceKind::Metal => {
                let package = profile.metal_package.clone().ok_or_else(|| {
                    BackendUnavailable::new(
                        DeviceKind::Metal,
                        "the selected native profile has no signed Metal package authority",
                    )
                })?;
                Self::for_metal(
                    packaged_worker_binary()?,
                    profile_id,
                    worker_id,
                    registry_version,
                    package,
                    memory_limit_bytes,
                )
            }
            DeviceKind::Mlu => {
                let package = profile.mlu_package.clone().ok_or_else(|| {
                    BackendUnavailable::new(
                        DeviceKind::Mlu,
                        "the selected native profile has no signed MLU package authority",
                    )
                })?;
                Self::for_mlu(
                    packaged_worker_binary()?,
                    profile_id,
                    worker_id,
                    registry_version,
                    package,
                    0,
                    memory_limit_bytes,
                )
            }
            DeviceKind::Npu => {
                let package = profile.npu_package.clone().ok_or_else(|| {
                    BackendUnavailable::new(
                        DeviceKind::Npu,
                        "the selected native profile has no signed NPU package authority",
                    )
                })?;
                Self::for_npu(
                    packaged_worker_binary()?,
                    profile_id,
                    worker_id,
                    registry_version,
                    package,
                    0,
                    memory_limit_bytes,
                )
            }
            DeviceKind::Cuda => {
                let package = profile.cuda_package.clone().ok_or_else(|| {
                    BackendUnavailable::new(
                        DeviceKind::Cuda,
                        "the selected native profile has no signed CUDA package authority",
                    )
                })?;
                Self::for_cuda(
                    packaged_worker_binary()?,
                    profile_id,
                    worker_id,
                    registry_version,
                    package,
                    0,
                    memory_limit_bytes,
                )
            }
            DeviceKind::Xpu => {
                let package = profile.xpu_package.clone().ok_or_else(|| {
                    BackendUnavailable::new(
                        DeviceKind::Xpu,
                        "the selected native profile has no signed XPU package authority",
                    )
                })?;
                Self::for_xpu(
                    packaged_worker_binary()?,
                    profile_id,
                    worker_id,
                    registry_version,
                    package,
                    0,
                    memory_limit_bytes,
                )
            }
            device => Err(BackendUnavailable::new(
                device,
                "the selected native profile has no certified production worker session adapter",
            )
            .into()),
        }?;
        Ok(config.with_general_video_codec_package(profile.general_video_codec_package.clone()))
    }

    pub fn new(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        memory_limit_bytes: u64,
    ) -> Self {
        Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            CpuBackend::capability_matrix(),
            WorkerBackendSelection::Cpu,
            memory_limit_bytes,
        )
    }

    pub fn for_device(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        device: DeviceKind,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::for_native_device(DeviceId::new(device, 0))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::Cpu,
            memory_limit_bytes,
        ))
    }

    pub fn for_rocm(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        package: NativeRocmPackageSettings,
        device_ordinal: u32,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::worker_readiness_requirements(DeviceId::new(
            DeviceKind::Rocm,
            device_ordinal,
        ))
        .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::Rocm {
                package,
                device_ordinal,
            },
            memory_limit_bytes,
        ))
    }

    pub fn for_directml(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        package: NativeDirectMlPackageSettings,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::worker_readiness_requirements(DeviceId::new(
            DeviceKind::DirectMl,
            0,
        ))
        .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::DirectMl { package },
            memory_limit_bytes,
        ))
    }

    pub fn for_metal(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        package: NativeMetalPackageSettings,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::worker_readiness_requirements(DeviceId::new(
            DeviceKind::Metal,
            0,
        ))
        .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::Metal { package },
            memory_limit_bytes,
        ))
    }

    pub fn for_mlu(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        package: NativeMluPackageSettings,
        device_ordinal: u32,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::worker_readiness_requirements(DeviceId::new(
            DeviceKind::Mlu,
            device_ordinal,
        ))
        .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::Mlu {
                package,
                device_ordinal,
            },
            memory_limit_bytes,
        ))
    }

    pub fn for_npu(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        package: NativeNpuPackageSettings,
        device_ordinal: u32,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::worker_readiness_requirements(DeviceId::new(
            DeviceKind::Npu,
            device_ordinal,
        ))
        .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::Npu {
                package,
                device_ordinal,
            },
            memory_limit_bytes,
        ))
    }

    pub fn for_xpu(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        package: NativeXpuPackageSettings,
        device_ordinal: u32,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::worker_readiness_requirements(DeviceId::new(
            DeviceKind::Xpu,
            device_ordinal,
        ))
        .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::Xpu {
                package,
                device_ordinal,
            },
            memory_limit_bytes,
        ))
    }

    pub fn for_cuda(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        package: NativeCudaPackageSettings,
        device_ordinal: u32,
        memory_limit_bytes: u64,
    ) -> Result<Self, RuntimeSupervisorError> {
        let backend = BackendCapabilityMatrix::worker_readiness_requirements(DeviceId::new(
            DeviceKind::Cuda,
            device_ordinal,
        ))
        .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        Ok(Self::with_backend(
            binary,
            profile_id,
            worker_id,
            registry_version,
            backend,
            WorkerBackendSelection::Cuda {
                package,
                device_ordinal,
            },
            memory_limit_bytes,
        ))
    }

    fn with_backend(
        binary: impl Into<PathBuf>,
        profile_id: ProfileId,
        worker_id: WorkerId,
        registry_version: impl Into<String>,
        backend: BackendCapabilityMatrix,
        backend_selection: WorkerBackendSelection,
        memory_limit_bytes: u64,
    ) -> Self {
        Self {
            binary: binary.into(),
            arguments: Vec::new(),
            working_directory: None,
            environment: Vec::new(),
            profile_id,
            worker_id,
            registry_version: registry_version.into(),
            backend,
            backend_selection,
            general_video_codec_package: None,
            memory_limit_bytes,
            policy: SupervisorPolicy::default(),
            registry_deployment: None,
            plugin_authorization_verifier: None,
        }
    }

    pub fn with_registry_deployment(mut self, deployment: WorkerRegistryDeploymentPlan) -> Self {
        self.plugin_authorization_verifier = Some(deployment.authorization_verifier().clone());
        self.registry_deployment = Some(deployment);
        self
    }

    pub fn with_general_video_codec_package(
        mut self,
        package: Option<NativeGeneralVideoCodecPackageSettings>,
    ) -> Self {
        self.general_video_codec_package = package;
        self
    }

    fn validate(&self) -> Result<(), RuntimeSupervisorError> {
        self.policy.validate()?;
        if self.binary.as_os_str().is_empty() {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker binary path is empty".to_owned(),
            ));
        }
        if self.registry_version.is_empty() {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker registry version is empty".to_owned(),
            ));
        }
        if self.backend.supported().is_empty() {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker backend capability request is empty".to_owned(),
            ));
        }
        if self.backend.device() != self.backend_selection.device() {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker backend selection and readiness requirements identify different devices"
                    .to_owned(),
            ));
        }
        if self.registry_deployment.as_ref().is_some_and(|deployment| {
            self.plugin_authorization_verifier.as_ref() != Some(deployment.authorization_verifier())
        }) {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "registry deployment authorization verifier differs from the worker launch verifier"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

fn packaged_worker_binary() -> Result<PathBuf, RuntimeSupervisorError> {
    let executable = std::env::current_exe().map_err(|error| {
        RuntimeSupervisorError::InvalidConfiguration(format!(
            "Zed executable path is unavailable: {error}"
        ))
    })?;
    packaged_worker_binary_for_executable(&executable)
}

fn packaged_worker_binary_for_executable(
    executable: &Path,
) -> Result<PathBuf, RuntimeSupervisorError> {
    let binary_name = if cfg!(windows) {
        "sim-worker.exe"
    } else {
        "sim-worker"
    };
    let directory = executable.parent().ok_or_else(|| {
        RuntimeSupervisorError::InvalidConfiguration(
            "Zed executable has no containing directory".to_owned(),
        )
    })?;
    Ok(directory.join(binary_name))
}

fn validate_registry_chunks(
    begin: &WorkerRegistryDeploymentBegin,
    chunks: &[WorkerRegistryDeploymentChunk],
) -> Result<(), RuntimeSupervisorError> {
    let mut cursor = 0_usize;
    for (component_index, descriptor) in begin.components().iter().enumerate() {
        let component_index = u32::try_from(component_index).map_err(|_| {
            RuntimeSupervisorError::InvalidRegistryDeployment(
                "component index exceeds the worker protocol".to_owned(),
            )
        })?;
        for content in [
            WorkerComponentContent::Manifest,
            WorkerComponentContent::Authorization,
            WorkerComponentContent::Component,
        ] {
            let (byte_length, chunk_count) = match content {
                WorkerComponentContent::Manifest => (
                    descriptor.manifest_bytes(),
                    descriptor.manifest_chunk_count(),
                ),
                WorkerComponentContent::Authorization => (
                    descriptor.authorization_bytes(),
                    descriptor.authorization_chunk_count(),
                ),
                WorkerComponentContent::Component => (
                    descriptor.component_bytes(),
                    descriptor.component_chunk_count(),
                ),
            };
            for chunk_index in 0..chunk_count {
                let chunk = chunks.get(cursor).ok_or_else(|| {
                    RuntimeSupervisorError::InvalidRegistryDeployment(
                        "component content is incomplete".to_owned(),
                    )
                })?;
                if chunk.generation() != begin.generation()
                    || chunk.component_index() != component_index
                    || chunk.content() != content
                    || chunk.chunk_index() != chunk_index
                {
                    return Err(RuntimeSupervisorError::InvalidRegistryDeployment(
                        "component chunks are not in canonical generation/content order".to_owned(),
                    ));
                }
                let chunk_bytes =
                    u64::try_from(MAX_WORKER_COMPONENT_CHUNK_BYTES).map_err(|_| {
                        RuntimeSupervisorError::InvalidRegistryDeployment(
                            "component chunk bound cannot be represented".to_owned(),
                        )
                    })?;
                let offset = u64::from(chunk_index)
                    .checked_mul(chunk_bytes)
                    .ok_or_else(|| {
                        RuntimeSupervisorError::InvalidRegistryDeployment(
                            "component chunk offset overflowed".to_owned(),
                        )
                    })?;
                let expected = byte_length
                    .checked_sub(offset)
                    .map(|remaining| remaining.min(chunk_bytes))
                    .and_then(|length| usize::try_from(length).ok())
                    .ok_or_else(|| {
                        RuntimeSupervisorError::InvalidRegistryDeployment(
                            "component chunk length overflowed".to_owned(),
                        )
                    })?;
                if chunk.bytes().len() != expected {
                    return Err(RuntimeSupervisorError::InvalidRegistryDeployment(
                        "component chunk length does not match its descriptor".to_owned(),
                    ));
                }
                cursor = cursor.checked_add(1).ok_or_else(|| {
                    RuntimeSupervisorError::InvalidRegistryDeployment(
                        "component chunk count overflowed".to_owned(),
                    )
                })?;
            }
        }
    }
    if cursor != chunks.len() {
        return Err(RuntimeSupervisorError::InvalidRegistryDeployment(
            "component deployment contains trailing chunks".to_owned(),
        ));
    }
    Ok(())
}

fn validate_registry_deployment_response(
    message: WorkerMessage,
    expected: &WorkerRegistryDeploymentBegin,
) -> Result<WorkerRegistryDeploymentAck, RuntimeSupervisorError> {
    match message {
        WorkerMessage::RegistryDeploymentAck { acknowledgement } => Ok(acknowledgement),
        WorkerMessage::RegistryDeploymentRejected { rejection } => {
            if rejection.generation() != expected.generation()
                || rejection.registry_digest_sha256() != expected.registry_digest_sha256()
            {
                return Err(RuntimeSupervisorError::Protocol(
                    "worker registry rejection does not match the deployed generation".to_owned(),
                ));
            }
            Err(RuntimeSupervisorError::InvalidRegistryDeployment(format!(
                "worker rejected registry generation {}: {:?}",
                rejection.generation().get(),
                rejection.reason()
            )))
        }
        _ => Err(RuntimeSupervisorError::Protocol(
            "worker did not acknowledge or reject the registry deployment".to_owned(),
        )),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessOwnership {
    ProcessGroup,
    WindowsJobObject,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerEnvironmentSource {
    InheritedSim,
    InheritedSimWithExplicitOverrides,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerLaunchRecord {
    pub executable: String,
    pub arguments: Vec<String>,
    pub working_directory: Option<String>,
    pub environment_names: Vec<String>,
    pub environment_source: WorkerEnvironmentSource,
    pub ownership: ProcessOwnership,
    pub started_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerHealth {
    Starting,
    ProcessAlive,
    IpcReachable,
    BackendReady,
    Degraded { reason: String },
    Cancelling,
    Hung,
    Lost,
    ProtocolIncompatible { reason: String },
    Exited { success: bool, code: Option<i32> },
}

impl WorkerHealth {
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Lost | Self::ProtocolIncompatible { .. } | Self::Exited { .. }
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerOperationStage {
    Prepare,
    Spawn,
    Handshake,
    Ready,
    Cancel,
    Shutdown,
    Kill,
    Exit,
    Fail,
    Recover,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerOperationTransition {
    pub stage: WorkerOperationStage,
    pub at: DateTime<Utc>,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerOperationRecord {
    pub operation_id: Uuid,
    pub profile_id: ProfileId,
    pub worker_id: WorkerId,
    pub transitions: Vec<WorkerOperationTransition>,
    pub terminal_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSupervisorSnapshot {
    pub profile_id: ProfileId,
    pub worker_id: WorkerId,
    pub registry_version: String,
    pub health: WorkerHealth,
    pub missed_heartbeats: u8,
    pub active_prompt_id: Option<PromptId>,
    pub active_attempt_id: Option<AttemptId>,
    pub launch: WorkerLaunchRecord,
    pub operation: WorkerOperationRecord,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RuntimeSupervisorError {
    #[error("invalid worker configuration: {0}")]
    InvalidConfiguration(String),
    #[error(transparent)]
    BackendUnavailable(#[from] BackendUnavailable),
    #[error("failed to spawn native worker: {0}")]
    Spawn(String),
    #[error("native worker did not expose its {0} pipe")]
    MissingPipe(&'static str),
    #[error("native worker I/O failed: {0}")]
    Io(String),
    #[error("native worker protocol failed: {0}")]
    Protocol(String),
    #[error("worker identity or request scope did not match the supervisor")]
    IdentityMismatch,
    #[error("worker response sequence expected {expected}, received {actual}")]
    Sequence { expected: u64, actual: u64 },
    #[error("native worker timed out during {stage}")]
    Timeout { stage: &'static str },
    #[error("native worker channel closed")]
    ChannelClosed,
    #[error("native worker reported fatal {code}: {message}")]
    WorkerFatal { code: String, message: String },
    #[error("native worker is not running")]
    NotRunning,
    #[error("native worker exited unsuccessfully with code {code:?}")]
    ExitFailure { code: Option<i32> },
    #[error("worker state {health:?} is not eligible for automatic recovery")]
    RecoveryNotEligible { health: WorkerHealth },
    #[error("automatic worker restart budget of {maximum} is exhausted")]
    RecoveryBudgetExhausted { maximum: u8 },
    #[error("worker registry deployment plan is invalid: {0}")]
    InvalidRegistryDeployment(String),
    #[error("canonical plugin capability broker failed: {0}")]
    PluginCapabilityBroker(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RequestScope {
    prompt_id: Option<PromptId>,
    attempt_id: Option<AttemptId>,
    kind: RequestKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestKind {
    Hello,
    RegistryDeployment,
    Execute,
    ExecutePlugin,
    ExecuteProviderV2,
    Cancel,
    Shutdown,
    Other,
}

impl From<&WorkerMessage> for RequestKind {
    fn from(message: &WorkerMessage) -> Self {
        match message {
            WorkerMessage::Hello { .. } => Self::Hello,
            WorkerMessage::RegistryDeploymentBegin { .. }
            | WorkerMessage::RegistryDeploymentChunk { .. }
            | WorkerMessage::RegistryDeploymentCommit { .. } => Self::RegistryDeployment,
            WorkerMessage::Execute { .. } => Self::Execute,
            WorkerMessage::ExecutePlugin { .. } => Self::ExecutePlugin,
            WorkerMessage::Cancel { .. } => Self::Cancel,
            WorkerMessage::Shutdown => Self::Shutdown,
            WorkerMessage::HelloAck { .. }
            | WorkerMessage::Ready
            | WorkerMessage::Event { .. }
            | WorkerMessage::OutputProposal { .. }
            | WorkerMessage::Lifecycle { .. }
            | WorkerMessage::RegistryDeploymentAck { .. }
            | WorkerMessage::RegistryDeploymentRejected { .. }
            | WorkerMessage::PluginCapabilityRequest { .. }
            | WorkerMessage::PluginCapabilityResponse { .. }
            | WorkerMessage::ProviderStreamRequest { .. }
            | WorkerMessage::ProviderStreamResponse { .. }
            | WorkerMessage::ProviderV2ProposalFinalization { .. }
            | WorkerMessage::ProviderV2ProposalFinalizationAck { .. }
            | WorkerMessage::ModelSourceRequest { .. }
            | WorkerMessage::ModelSourceResponse { .. }
            | WorkerMessage::PluginResult { .. }
            | WorkerMessage::Heartbeat
            | WorkerMessage::Fatal { .. } => Self::Other,
        }
    }
}

struct SupervisorShared {
    snapshot: RuntimeSupervisorSnapshot,
    requested_backend: BackendCapabilityMatrix,
    accepted_backend: Option<BackendCapabilityMatrix>,
    request_scopes: HashMap<RequestId, RequestScope>,
    request_order: VecDeque<RequestId>,
    last_worker_sequence: Option<u64>,
    last_heartbeat: Option<Instant>,
    monitoring_active: bool,
    shutdown_requested: bool,
    logs: BoundedLog,
}

impl SupervisorShared {
    fn record_exit(&mut self, status: ExitStatus) {
        self.snapshot.health = WorkerHealth::Exited {
            success: status.success(),
            code: status.code(),
        };
        self.record(
            WorkerOperationStage::Exit,
            format!(
                "worker exited with code {:?}; status={status}",
                status.code()
            ),
        );
        self.monitoring_active = false;
    }

    fn record(&mut self, stage: WorkerOperationStage, detail: impl Into<String>) {
        self.snapshot
            .operation
            .transitions
            .push(WorkerOperationTransition {
                stage,
                at: Utc::now(),
                detail: detail.into(),
            });
    }

    fn fail(&mut self, error: &RuntimeSupervisorError) {
        let detail = error.to_string();
        self.snapshot.operation.terminal_error = Some(detail.clone());
        self.record(WorkerOperationStage::Fail, detail);
    }

    fn register_request(&mut self, request_id: RequestId, scope: RequestScope) {
        self.request_scopes.insert(request_id, scope);
        self.request_order.push_back(request_id);
        while self.request_order.len() > MAX_TRACKED_WORKER_REQUESTS {
            if let Some(expired) = self.request_order.pop_front() {
                self.request_scopes.remove(&expired);
            }
        }
        self.snapshot.active_prompt_id = scope.prompt_id;
        self.snapshot.active_attempt_id = scope.attempt_id;
    }

    #[cfg_attr(
        not(any(test, feature = "test-support")),
        expect(
            dead_code,
            reason = "Task427 deployment actuator retires provider-v2 execution scopes"
        )
    )]
    fn retire_provider_v2_request(&mut self, request_id: RequestId) {
        if self
            .request_scopes
            .remove(&request_id)
            .is_some_and(|scope| scope.kind == RequestKind::ExecuteProviderV2)
        {
            self.request_order.retain(|tracked| *tracked != request_id);
            self.snapshot.active_prompt_id = None;
            self.snapshot.active_attempt_id = None;
            if !self.snapshot.health.is_terminal() {
                self.snapshot.health = WorkerHealth::BackendReady;
            }
        }
    }

    #[cfg_attr(
        not(any(test, feature = "test-support")),
        expect(
            dead_code,
            reason = "Task427 deployment actuator retires provider-v2 cancellation scopes"
        )
    )]
    fn retire_provider_v2_cancel_request(&mut self, request_id: RequestId) {
        if self
            .request_scopes
            .remove(&request_id)
            .is_some_and(|scope| scope.kind == RequestKind::Cancel)
        {
            self.request_order.retain(|tracked| *tracked != request_id);
        }
    }

    fn accept(&mut self, envelope: &WorkerEnvelope) -> Result<(), RuntimeSupervisorError> {
        if envelope.version != WORKER_PROTOCOL_VERSION
            || envelope.profile_id != self.snapshot.profile_id
            || envelope.worker_id != self.snapshot.worker_id
            || envelope.registry_version != self.snapshot.registry_version
        {
            let error = RuntimeSupervisorError::IdentityMismatch;
            self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                reason: error.to_string(),
            };
            self.fail(&error);
            return Err(error);
        }
        let expected = self
            .last_worker_sequence
            .map_or(0, |sequence| sequence.saturating_add(1));
        if envelope.sequence != expected {
            let error = RuntimeSupervisorError::Sequence {
                expected,
                actual: envelope.sequence,
            };
            self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                reason: error.to_string(),
            };
            self.fail(&error);
            return Err(error);
        }
        let scope = self
            .request_scopes
            .get(&envelope.request_id)
            .copied()
            .ok_or(RuntimeSupervisorError::IdentityMismatch)?;
        if scope.prompt_id != envelope.prompt_id || scope.attempt_id != envelope.attempt_id {
            let error = RuntimeSupervisorError::IdentityMismatch;
            self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                reason: error.to_string(),
            };
            self.fail(&error);
            return Err(error);
        }
        self.last_worker_sequence = Some(envelope.sequence);
        match &envelope.message {
            WorkerMessage::HelloAck { accepted_backend } => {
                let accepted_backend =
                    match BackendCapabilityMatrix::try_from(accepted_backend.clone()) {
                        Ok(accepted_backend) => accepted_backend,
                        Err(source) => {
                            let error = RuntimeSupervisorError::Protocol(format!(
                                "worker acknowledged an invalid backend matrix: {source}"
                            ));
                            self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                                reason: error.to_string(),
                            };
                            self.fail(&error);
                            return Err(error);
                        }
                    };
                if !self.requested_backend.is_subset_of(&accepted_backend) {
                    let error = RuntimeSupervisorError::Protocol(
                        "worker instance does not satisfy the requested readiness matrix"
                            .to_owned(),
                    );
                    self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                        reason: error.to_string(),
                    };
                    self.fail(&error);
                    return Err(error);
                }
                self.accepted_backend = Some(accepted_backend);
                self.snapshot.health = WorkerHealth::IpcReachable;
                self.record(WorkerOperationStage::Handshake, "private IPC negotiated");
            }
            WorkerMessage::Ready => {
                if self.accepted_backend.is_none() {
                    let error = RuntimeSupervisorError::Protocol(
                        "worker became ready before capability negotiation".to_owned(),
                    );
                    self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                        reason: error.to_string(),
                    };
                    self.fail(&error);
                    return Err(error);
                }
                self.snapshot.health = WorkerHealth::BackendReady;
                self.snapshot.missed_heartbeats = 0;
                self.last_heartbeat = Some(Instant::now());
                self.monitoring_active = true;
                self.record(WorkerOperationStage::Ready, "native backend ready");
            }
            WorkerMessage::Heartbeat => {
                self.last_heartbeat = Some(Instant::now());
                self.snapshot.missed_heartbeats = 0;
            }
            WorkerMessage::Fatal { code, message } => {
                let error = RuntimeSupervisorError::WorkerFatal {
                    code: code.clone(),
                    message: message.clone(),
                };
                self.snapshot.health = WorkerHealth::Degraded {
                    reason: error.to_string(),
                };
                self.fail(&error);
            }
            WorkerMessage::Shutdown => {
                self.monitoring_active = false;
                self.record(
                    WorkerOperationStage::Shutdown,
                    "worker acknowledged shutdown",
                );
            }
            WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::ExecutionStarted,
            } if matches!(
                scope.kind,
                RequestKind::Execute | RequestKind::ExecutePlugin | RequestKind::ExecuteProviderV2
            ) => {}
            WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::CancellationRequested { .. },
            } if scope.kind == RequestKind::Cancel => {}
            WorkerMessage::Lifecycle { event } => {
                let error = RuntimeSupervisorError::Protocol(format!(
                    "worker lifecycle event {event:?} does not match request kind {:?}",
                    scope.kind
                ));
                self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                    reason: error.to_string(),
                };
                self.fail(&error);
                return Err(error);
            }
            WorkerMessage::OutputProposal { .. } if scope.kind == RequestKind::Execute => {}
            WorkerMessage::PluginCapabilityRequest { .. }
                if matches!(
                    scope.kind,
                    RequestKind::Execute | RequestKind::ExecutePlugin
                ) => {}
            WorkerMessage::PluginResult { .. } if scope.kind == RequestKind::ExecutePlugin => {
                self.snapshot.health = WorkerHealth::BackendReady;
                self.snapshot.active_prompt_id = None;
                self.snapshot.active_attempt_id = None;
            }
            WorkerMessage::PluginResult {
                outcome: WorkerPluginExecutionOutcome::Failed(_),
            } if scope.kind == RequestKind::ExecuteProviderV2 => {
                self.snapshot.health = WorkerHealth::BackendReady;
                self.snapshot.active_prompt_id = None;
                self.snapshot.active_attempt_id = None;
            }
            WorkerMessage::PluginResult {
                outcome: WorkerPluginExecutionOutcome::Succeeded(_),
            } if scope.kind == RequestKind::ExecuteProviderV2 => {}
            WorkerMessage::ProviderStreamRequest { .. }
                if scope.kind == RequestKind::ExecuteProviderV2 => {}
            WorkerMessage::ProviderV2ProposalFinalizationAck { .. }
                if scope.kind == RequestKind::ExecuteProviderV2 =>
            {
                self.snapshot.health = WorkerHealth::BackendReady;
                self.snapshot.active_prompt_id = None;
                self.snapshot.active_attempt_id = None;
            }
            WorkerMessage::ModelSourceRequest { .. } if scope.kind == RequestKind::Execute => {}
            WorkerMessage::RegistryDeploymentAck { .. }
                if scope.kind == RequestKind::RegistryDeployment => {}
            WorkerMessage::RegistryDeploymentRejected { .. }
                if scope.kind == RequestKind::RegistryDeployment => {}
            WorkerMessage::ProviderStreamRequest { .. }
            | WorkerMessage::ProviderStreamResponse { .. }
            | WorkerMessage::ProviderV2ProposalFinalization { .. }
            | WorkerMessage::ProviderV2ProposalFinalizationAck { .. }
            | WorkerMessage::ModelSourceRequest { .. }
            | WorkerMessage::ModelSourceResponse { .. } => {
                let error = RuntimeSupervisorError::Protocol(
                    "worker stream message has an invalid direction or execution scope".to_owned(),
                );
                self.snapshot.health = WorkerHealth::ProtocolIncompatible {
                    reason: error.to_string(),
                };
                self.fail(&error);
                return Err(error);
            }
            WorkerMessage::Event { event }
                if matches!(scope.kind, RequestKind::Execute | RequestKind::Cancel)
                    && postcard::from_bytes::<crate::NativeImageWorkerEvent>(event).is_ok_and(
                        |event| {
                            matches!(
                                event,
                                crate::NativeImageWorkerEvent::Completed { .. }
                                    | crate::NativeImageWorkerEvent::BackendUnavailable { .. }
                                    | crate::NativeImageWorkerEvent::Failed { .. }
                            )
                        },
                    ) =>
            {
                self.snapshot.health = WorkerHealth::BackendReady;
                self.snapshot.active_prompt_id = None;
                self.snapshot.active_attempt_id = None;
                if scope.kind == RequestKind::Cancel {
                    self.record(
                        WorkerOperationStage::Recover,
                        "attempt cancellation converged",
                    );
                }
            }
            WorkerMessage::Event { .. }
            | WorkerMessage::OutputProposal { .. }
            | WorkerMessage::Hello { .. }
            | WorkerMessage::RegistryDeploymentBegin { .. }
            | WorkerMessage::RegistryDeploymentChunk { .. }
            | WorkerMessage::RegistryDeploymentCommit { .. }
            | WorkerMessage::RegistryDeploymentAck { .. }
            | WorkerMessage::RegistryDeploymentRejected { .. }
            | WorkerMessage::Execute { .. }
            | WorkerMessage::ExecutePlugin { .. }
            | WorkerMessage::PluginCapabilityRequest { .. }
            | WorkerMessage::PluginCapabilityResponse { .. }
            | WorkerMessage::PluginResult { .. }
            | WorkerMessage::Cancel { .. } => {}
        }
        Ok(())
    }

    fn evaluate_heartbeat(&mut self, now: Instant, policy: SupervisorPolicy) {
        if !self.monitoring_active || self.snapshot.health.is_terminal() {
            return;
        }
        let Some(last_heartbeat) = self.last_heartbeat else {
            return;
        };
        let interval_millis = policy.heartbeat_interval.as_millis().max(1);
        let missed = now.duration_since(last_heartbeat).as_millis() / interval_millis;
        self.snapshot.missed_heartbeats = u8::try_from(missed).unwrap_or(u8::MAX);
        if self.snapshot.missed_heartbeats >= policy.missed_heartbeat_limit {
            self.snapshot.health = WorkerHealth::Lost;
            self.monitoring_active = false;
            let error = RuntimeSupervisorError::Timeout { stage: "heartbeat" };
            self.fail(&error);
        }
    }
}

struct BoundedLog {
    entries: VecDeque<String>,
    bytes: usize,
}

impl BoundedLog {
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        let entry = sanitize_log(bytes);
        if entry.is_empty() {
            return;
        }
        self.bytes = self.bytes.saturating_add(entry.len());
        self.entries.push_back(entry);
        while self.bytes > MAX_CAPTURED_WORKER_LOG_BYTES {
            let Some(removed) = self.entries.pop_front() else {
                self.bytes = 0;
                break;
            };
            self.bytes = self.bytes.saturating_sub(removed.len());
        }
    }

    fn entries(&self) -> Vec<String> {
        self.entries.iter().cloned().collect()
    }
}

pub struct RuntimeSupervisor {
    child: Option<Child>,
    input: Option<WorkerInput>,
    incoming: async_channel::Receiver<Result<WorkerEnvelope, RuntimeSupervisorError>>,
    shared: Arc<Mutex<SupervisorShared>>,
    profile_id: ProfileId,
    worker_id: WorkerId,
    registry_version: String,
    next_input_sequence: u64,
    policy: SupervisorPolicy,
    launch_config: WorkerLaunchConfig,
    automatic_restarts: u8,
    _reader_task: smol::Task<()>,
    _log_task: smol::Task<()>,
    _heartbeat_task: smol::Task<()>,
}

impl RuntimeSupervisor {
    pub async fn start(config: WorkerLaunchConfig) -> Result<Self, RuntimeSupervisorError> {
        config.validate()?;
        let requested_backend_wire = config
            .backend
            .to_worker_capabilities()
            .map_err(|error| RuntimeSupervisorError::InvalidConfiguration(error.to_string()))?;
        let mut command = std::process::Command::new(&config.binary);
        command.args(&config.arguments);
        command.args(config.backend_selection.launch_arguments()?);
        command.args(general_video_codec_package_launch_arguments(
            config.general_video_codec_package.as_ref(),
        )?);
        command
            .arg("--memory-limit-bytes")
            .arg(config.memory_limit_bytes.to_string());
        if let Some(verifier) = &config.plugin_authorization_verifier {
            command
                .arg("--plugin-authorization-verification-key")
                .arg(verifier.to_token());
        }
        if let Some(working_directory) = &config.working_directory {
            command.current_dir(working_directory);
        }
        for (name, value) in &config.environment {
            command.env(name, value);
        }
        let mut child = Child::spawn(command, Stdio::piped(), Stdio::piped(), Stdio::piped())
            .map_err(|error| RuntimeSupervisorError::Spawn(error.to_string()))?;
        let input = match child.stdin.take() {
            Some(input) => input,
            None => return Err(missing_pipe_after_spawn(&mut child, "stdin")),
        };
        let output = match child.stdout.take() {
            Some(output) => output,
            None => return Err(missing_pipe_after_spawn(&mut child, "stdout")),
        };
        let error_output = match child.stderr.take() {
            Some(error_output) => error_output,
            None => return Err(missing_pipe_after_spawn(&mut child, "stderr")),
        };

        let launch = launch_record(&config)?;
        let operation = WorkerOperationRecord {
            operation_id: Uuid::new_v4(),
            profile_id: config.profile_id,
            worker_id: config.worker_id,
            transitions: vec![WorkerOperationTransition {
                stage: WorkerOperationStage::Prepare,
                at: Utc::now(),
                detail: "native worker launch prepared".to_owned(),
            }],
            terminal_error: None,
        };
        let shared = Arc::new(Mutex::new(SupervisorShared {
            snapshot: RuntimeSupervisorSnapshot {
                profile_id: config.profile_id,
                worker_id: config.worker_id,
                registry_version: config.registry_version.clone(),
                health: WorkerHealth::ProcessAlive,
                missed_heartbeats: 0,
                active_prompt_id: None,
                active_attempt_id: None,
                launch,
                operation,
            },
            requested_backend: config.backend.clone(),
            accepted_backend: None,
            request_scopes: HashMap::new(),
            request_order: VecDeque::new(),
            last_worker_sequence: None,
            last_heartbeat: None,
            monitoring_active: false,
            shutdown_requested: false,
            logs: BoundedLog::new(),
        }));
        shared
            .lock()
            .record(WorkerOperationStage::Spawn, "owned process started");

        let (incoming_sender, incoming) = async_channel::bounded(MAX_PENDING_WORKER_MESSAGES);
        let reader_task = smol::spawn(read_worker_output(output, shared.clone(), incoming_sender));
        let log_task = smol::spawn(read_worker_logs(error_output, shared.clone()));
        let heartbeat_task = smol::spawn(monitor_heartbeats(shared.clone(), config.policy));
        let launch_config = config.clone();
        let mut supervisor = Self {
            child: Some(child),
            input: Some(Box::new(input)),
            incoming,
            shared,
            profile_id: config.profile_id,
            worker_id: config.worker_id,
            registry_version: config.registry_version,
            next_input_sequence: 0,
            policy: config.policy,
            launch_config,
            automatic_restarts: 0,
            _reader_task: reader_task,
            _log_task: log_task,
            _heartbeat_task: heartbeat_task,
        };

        supervisor
            .send(
                None,
                None,
                WorkerMessage::Hello {
                    backend: requested_backend_wire,
                },
            )
            .await?;
        let hello_ack = supervisor
            .receive_one_at_stage(config.policy.ready_timeout, "worker HelloAck")
            .await?;
        match hello_ack.message {
            WorkerMessage::HelloAck { .. } => {}
            WorkerMessage::Fatal { code, message } if code == "backend_unavailable" => {
                let unavailable =
                    serde_json::from_str::<BackendUnavailable>(&message).map_err(|error| {
                        RuntimeSupervisorError::Protocol(format!(
                            "worker emitted malformed backend-unavailable detail: {error}"
                        ))
                    })?;
                return Err(RuntimeSupervisorError::BackendUnavailable(unavailable));
            }
            WorkerMessage::Fatal { code, message } => {
                return Err(RuntimeSupervisorError::WorkerFatal { code, message });
            }
            _ => {
                return Err(RuntimeSupervisorError::Protocol(
                    "worker did not begin with HelloAck".to_owned(),
                ));
            }
        }
        let ready = supervisor
            .receive_one_at_stage(config.policy.ready_timeout, "worker Ready")
            .await?;
        if !matches!(ready.message, WorkerMessage::Ready) {
            return Err(RuntimeSupervisorError::Protocol(
                "worker did not become ready after HelloAck".to_owned(),
            ));
        }
        if let Some(deployment) = config.registry_deployment.as_ref() {
            supervisor.deploy_registry(deployment).await?;
        }
        Ok(supervisor)
    }

    pub fn snapshot(&self) -> RuntimeSupervisorSnapshot {
        self.shared.lock().snapshot.clone()
    }

    pub fn logs(&self) -> Vec<String> {
        self.shared.lock().logs.entries()
    }

    pub fn accepted_backend(&self) -> Option<BackendCapabilityMatrix> {
        self.shared.lock().accepted_backend.clone()
    }

    pub fn worker_process_id(&self) -> Option<u32> {
        self.child.as_ref().map(|child| child.id())
    }

    pub async fn deploy_registry(
        &mut self,
        deployment: &WorkerRegistryDeploymentPlan,
    ) -> Result<WorkerRegistryDeploymentAck, RuntimeSupervisorError> {
        if self.launch_config.plugin_authorization_verifier.as_ref()
            != Some(deployment.authorization_verifier())
        {
            return Err(RuntimeSupervisorError::InvalidRegistryDeployment(
                "registry authorization verifier differs from the worker launch verifier"
                    .to_owned(),
            ));
        }
        validate_registry_chunks(deployment.begin(), deployment.chunks())?;
        self.send(
            None,
            None,
            WorkerMessage::RegistryDeploymentBegin {
                deployment: deployment.begin().clone(),
            },
        )
        .await?;
        for chunk in deployment.chunks() {
            self.send(
                None,
                None,
                WorkerMessage::RegistryDeploymentChunk {
                    chunk: chunk.clone(),
                },
            )
            .await?;
        }
        self.send(
            None,
            None,
            WorkerMessage::RegistryDeploymentCommit {
                commit: WorkerRegistryDeploymentCommit::new(
                    deployment.begin().generation(),
                    deployment.begin().registry_digest_sha256().clone(),
                ),
            },
        )
        .await?;
        let envelope = self
            .receive_one_at_stage(self.policy.ready_timeout, "worker RegistryDeploymentAck")
            .await?;
        let acknowledgement =
            validate_registry_deployment_response(envelope.message, deployment.begin())?;
        let component_count =
            u32::try_from(deployment.begin().components().len()).map_err(|_| {
                RuntimeSupervisorError::InvalidRegistryDeployment(
                    "component count exceeds the worker acknowledgement".to_owned(),
                )
            })?;
        if acknowledgement.generation() != deployment.begin().generation()
            || acknowledgement.registry_digest_sha256()
                != deployment.begin().registry_digest_sha256()
            || acknowledgement.component_count() != component_count
        {
            return Err(RuntimeSupervisorError::Protocol(
                "worker registry acknowledgement does not match the deployed generation".to_owned(),
            ));
        }
        Ok(acknowledgement)
    }

    pub async fn execute(
        &mut self,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        plan: Vec<u8>,
    ) -> Result<RequestId, RuntimeSupervisorError> {
        if plan.is_empty() {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker execute plan is empty".to_owned(),
            ));
        }
        self.send(
            Some(prompt_id),
            Some(attempt_id),
            WorkerMessage::Execute { plan },
        )
        .await
    }

    pub async fn execute_plugin(
        &mut self,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        invocation: Vec<u8>,
        capability_invocation: PluginCapabilityInvocation,
    ) -> Result<WorkerPluginExecutionOutcome, RuntimeSupervisorError> {
        self.execute_plugin_retaining_capabilities(
            prompt_id,
            attempt_id,
            invocation,
            capability_invocation,
        )
        .await?
        .finish()
    }

    pub async fn execute_plugin_retaining_capabilities(
        &mut self,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        invocation: Vec<u8>,
        capability_invocation: PluginCapabilityInvocation,
    ) -> Result<RetainedPluginExecution, RuntimeSupervisorError> {
        if invocation.is_empty() {
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker plugin invocation is empty".to_owned(),
            ));
        }
        let request_id = self
            .send(
                Some(prompt_id),
                Some(attempt_id),
                WorkerMessage::ExecutePlugin { invocation },
            )
            .await?;
        let cancellation = capability_invocation.context().cancellation().clone();
        let mut capability_invocation = Some(capability_invocation);
        let mut cancellation_request_id = None;
        let response_deadline = Instant::now()
            .checked_add(self.policy.ready_timeout)
            .ok_or(RuntimeSupervisorError::Timeout {
                stage: "plugin execution",
            })?;
        loop {
            if cancellation.is_cancelled() && cancellation_request_id.is_none() {
                cancellation_request_id = Some(
                    self.cancel(prompt_id, attempt_id, "plugin invocation cancelled")
                        .await?,
                );
            }
            let remaining = response_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if let Some(invocation) = capability_invocation.take() {
                    invocation.abort();
                }
                return Err(RuntimeSupervisorError::Timeout {
                    stage: "plugin execution",
                });
            }
            let poll_interval = remaining.min(Duration::from_millis(10));
            let envelope = match self.receive_one(poll_interval).await {
                Err(RuntimeSupervisorError::Timeout { .. }) => continue,
                result => result?,
            };
            match envelope.message {
                WorkerMessage::Lifecycle {
                    event: WorkerLifecycleEvent::ExecutionStarted,
                }
                | WorkerMessage::Heartbeat => {}
                WorkerMessage::Lifecycle {
                    event: WorkerLifecycleEvent::CancellationRequested { .. },
                } if cancellation_request_id == Some(envelope.request_id) => {}
                WorkerMessage::PluginCapabilityRequest { call_id, request } => {
                    if envelope.request_id != request_id {
                        return Err(RuntimeSupervisorError::IdentityMismatch);
                    }
                    let response = match PluginServiceWireRequest::from_bytes(&request) {
                        Ok(request) => capability_invocation
                            .as_mut()
                            .ok_or_else(|| {
                                RuntimeSupervisorError::PluginCapabilityBroker(
                                    "invocation is already terminal".to_owned(),
                                )
                            })?
                            .handle_wire_request(request),
                        Err(_) => PluginServiceWireResponse::Failure(
                            PluginServiceWireFailure::InvalidRequest,
                        ),
                    };
                    let maximum = u64::try_from(MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES)
                        .map_err(|error| {
                            RuntimeSupervisorError::PluginCapabilityBroker(error.to_string())
                        })?;
                    let response = response
                        .to_bytes(maximum)
                        .or_else(|_| {
                            PluginServiceWireResponse::Failure(
                                PluginServiceWireFailure::ResponseTooLarge,
                            )
                            .to_bytes(maximum)
                        })
                        .map_err(|_| {
                            RuntimeSupervisorError::PluginCapabilityBroker(
                                "bounded capability response cannot be encoded".to_owned(),
                            )
                        })?;
                    self.send_for_existing_request(
                        request_id,
                        Some(prompt_id),
                        Some(attempt_id),
                        WorkerMessage::PluginCapabilityResponse { call_id, response },
                    )
                    .await?;
                }
                WorkerMessage::PluginResult { outcome } => {
                    if envelope.request_id != request_id {
                        return Err(RuntimeSupervisorError::IdentityMismatch);
                    }
                    let invocation = capability_invocation.take().ok_or_else(|| {
                        RuntimeSupervisorError::PluginCapabilityBroker(
                            "invocation completed more than once".to_owned(),
                        )
                    })?;
                    return Ok(match outcome {
                        WorkerPluginExecutionOutcome::Succeeded(bytes) => RetainedPluginExecution {
                            outcome: WorkerPluginExecutionOutcome::Succeeded(bytes),
                            capability_invocation: Some(invocation),
                        },
                        WorkerPluginExecutionOutcome::Failed(failure) => {
                            invocation.abort();
                            RetainedPluginExecution {
                                outcome: WorkerPluginExecutionOutcome::Failed(failure),
                                capability_invocation: None,
                            }
                        }
                    });
                }
                WorkerMessage::Fatal { code, message } => {
                    if let Some(invocation) = capability_invocation.take() {
                        invocation.abort();
                    }
                    return Err(RuntimeSupervisorError::WorkerFatal { code, message });
                }
                message => {
                    if let Some(invocation) = capability_invocation.take() {
                        invocation.abort();
                    }
                    return Err(RuntimeSupervisorError::Protocol(format!(
                        "worker emitted {message:?} during plugin execution"
                    )));
                }
            }
        }
    }

    #[cfg(feature = "test-support")]
    pub async fn execute_provider_v2(
        &mut self,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        invocation: Vec<u8>,
        mut supervised_route: crate::NativeProviderWorkerV2SupervisorBridge,
        capability_invocation: &mut PluginCapabilityInvocation,
    ) -> Result<
        (
            WorkerPluginExecutionOutcome,
            Option<crate::ProviderTransportResponse>,
        ),
        RuntimeSupervisorError,
    > {
        let mut bridge = supervised_route.take()?;
        if invocation.is_empty() {
            bridge.cancellation.cancel();
            return Err(RuntimeSupervisorError::InvalidConfiguration(
                "worker provider-v2 invocation is empty".to_owned(),
            ));
        }
        let request_id = match self
            .send_with_kind(
                Some(prompt_id),
                Some(attempt_id),
                WorkerMessage::ExecutePlugin { invocation },
                RequestKind::ExecuteProviderV2,
            )
            .await
        {
            Ok(request_id) => request_id,
            Err(error) => {
                bridge.cancellation.cancel();
                return Err(error);
            }
        };
        let response_deadline = Instant::now()
            .checked_add(bridge.invocation_timeout)
            .ok_or(RuntimeSupervisorError::Timeout {
                stage: "provider-v2 execution",
            })?;
        let mut cancellation_request_id = None;
        let result = async {
            let mut proposal_outcome = None;
            let mut previous_capability_call_id = 0;
            let mut expected_finalization = None;
            let mut proposal_materialization = None;
            let mut finalization_phase = ProviderV2FinalizationPhase::PreCommit;
            let mut cancellation_lifecycle_observed = false;
            loop {
                if should_begin_provider_v2_cancellation(
                    finalization_phase,
                    &bridge.cancellation,
                    cancellation_request_id,
                ) {
                    cancellation_request_id = Some(
                        self.cancel(prompt_id, attempt_id, "provider-v2 app route was cancelled")
                            .await?,
                    );
                }
                let remaining = response_deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(RuntimeSupervisorError::Timeout {
                        stage: "provider-v2 execution",
                    });
                }
                let envelope = self
                    .receive_one(remaining.min(Duration::from_millis(10)))
                    .await;
                let envelope = match envelope {
                    Err(RuntimeSupervisorError::Timeout { .. }) => continue,
                    result => result?,
                };
                if should_begin_provider_v2_cancellation(
                    finalization_phase,
                    &bridge.cancellation,
                    cancellation_request_id,
                ) {
                    cancellation_request_id = Some(
                        self.cancel(prompt_id, attempt_id, "provider-v2 app route was cancelled")
                            .await?,
                    );
                }
                let cancellation_lifecycle = cancellation_request_id == Some(envelope.request_id)
                    && matches!(
                        &envelope.message,
                        WorkerMessage::Lifecycle {
                            event: WorkerLifecycleEvent::CancellationRequested { .. }
                        }
                    );
                if envelope.request_id != request_id
                    && !cancellation_lifecycle
                    && !matches!(&envelope.message, WorkerMessage::Heartbeat)
                {
                    return Err(RuntimeSupervisorError::IdentityMismatch);
                }
                if suppress_provider_v2_precommit_message(
                    finalization_phase,
                    cancellation_request_id,
                    &envelope.message,
                ) {
                    continue;
                }
                match envelope.message {
                    WorkerMessage::Lifecycle {
                        event: WorkerLifecycleEvent::ExecutionStarted,
                    }
                    | WorkerMessage::Heartbeat => {}
                    WorkerMessage::Lifecycle {
                        event: WorkerLifecycleEvent::CancellationRequested { .. },
                    } if cancellation_request_id == Some(envelope.request_id) => {
                        cancellation_lifecycle_observed = true;
                    }
                    WorkerMessage::PluginCapabilityRequest { call_id, request } => {
                        if proposal_outcome.is_some() || call_id <= previous_capability_call_id {
                            return Err(RuntimeSupervisorError::IdentityMismatch);
                        }
                        previous_capability_call_id = call_id;
                        let request = PluginServiceWireRequest::from_bytes(&request)
                            .map_err(|_| RuntimeSupervisorError::Protocol(
                                "provider-v2 capability request is malformed".to_owned(),
                            ))?;
                        if !matches!(request, PluginServiceWireRequest::CredentialIsPresent { .. }) {
                            return Err(RuntimeSupervisorError::Protocol(
                                "provider-v2 input host only permits credential presence queries".to_owned(),
                            ));
                        }
                        let response = capability_invocation.handle_wire_request(request);
                        let maximum = u64::try_from(MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES)
                            .map_err(|error| RuntimeSupervisorError::PluginCapabilityBroker(error.to_string()))?;
                        let response = response.to_bytes(maximum)
                            .map_err(|error| RuntimeSupervisorError::PluginCapabilityBroker(error.to_string()))?;
                        self.send_for_existing_request(
                            request_id,
                            Some(prompt_id),
                            Some(attempt_id),
                            WorkerMessage::PluginCapabilityResponse { call_id, response },
                        ).await?;
                    }
                    WorkerMessage::ProviderStreamRequest { call_id, request } => {
                        if proposal_outcome.is_some() {
                            return Err(RuntimeSupervisorError::Protocol(
                                "provider-v2 stream request arrived after its proposal".to_owned(),
                            ));
                        }
                        bridge
                            .validator
                            .validate_request(call_id, &request)
                            .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))?;
                        let response_timeout = match &request {
                            WorkerProviderStreamRequest::WaitResponse(request) => {
                                Duration::from_millis(request.timeout_milliseconds).min(remaining)
                            }
                            _ => remaining,
                        };
                        let (response, response_receiver) = async_channel::bounded(1);
                        bridge
                            .stream_calls
                            .try_send(RuntimeProviderV2StreamCall {
                                call_id,
                                request,
                                response,
                            })
                            .map_err(|error| {
                                RuntimeSupervisorError::Protocol(format!(
                                    "provider-v2 capacity-one route rejected a call: {error}"
                                ))
                            })?;
                        let call_response_deadline = Instant::now()
                            .checked_add(response_timeout)
                            .ok_or(RuntimeSupervisorError::Timeout {
                                stage: "provider-v2 stream response",
                            })?;
                        let response = loop {
                            if should_begin_provider_v2_cancellation(
                                finalization_phase,
                                &bridge.cancellation,
                                cancellation_request_id,
                            ) {
                                cancellation_request_id = Some(
                                    self.cancel(
                                        prompt_id,
                                        attempt_id,
                                        "provider-v2 app route was cancelled",
                                    )
                                    .await?,
                                );
                                break None;
                            }
                            let response_remaining = call_response_deadline
                                .saturating_duration_since(Instant::now());
                            if response_remaining.is_zero() {
                                return Err(RuntimeSupervisorError::Timeout {
                                    stage: "provider-v2 stream response",
                                });
                            }
                            let poll = response_remaining.min(Duration::from_millis(10));
                            let received = smol::future::race(
                                async {
                                    response_receiver.recv().await.map(Some).map_err(|_| {
                                        RuntimeSupervisorError::Protocol(
                                            "provider-v2 stream route lost its response".to_owned(),
                                        )
                                    })
                                },
                                async {
                                    supervisor_delay(poll).await;
                                    Ok(None)
                                },
                            )
                            .await;
                            let received = match received {
                                Ok(received) => received,
                                Err(_)
                                    if provider_v2_wait_close_is_cancellation(
                                        finalization_phase,
                                        &bridge.cancellation,
                                    ) =>
                                {
                                    if cancellation_request_id.is_none() {
                                        cancellation_request_id = Some(
                                            self.cancel(
                                                prompt_id,
                                                attempt_id,
                                                "provider-v2 app route was cancelled",
                                            )
                                            .await?,
                                        );
                                    }
                                    break None;
                                }
                                Err(error) => return Err(error),
                            };
                            if received.is_some() {
                                break received;
                            }
                        };
                        let Some(response) = response else {
                            continue;
                        };
                        if should_begin_provider_v2_cancellation(
                            finalization_phase,
                            &bridge.cancellation,
                            cancellation_request_id,
                        ) {
                            cancellation_request_id = Some(
                                self.cancel(
                                    prompt_id,
                                    attempt_id,
                                    "provider-v2 app route was cancelled",
                                )
                                .await?,
                            );
                            continue;
                        }
                        bridge
                            .validator
                            .validate_response(call_id, &response)
                            .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))?;
                        self.send_for_existing_request(
                            request_id,
                            Some(prompt_id),
                            Some(attempt_id),
                            WorkerMessage::ProviderStreamResponse { call_id, response },
                        )
                        .await?;
                    }
                    WorkerMessage::PluginResult { outcome } if proposal_outcome.is_none() => {
                        if should_begin_provider_v2_cancellation(
                            finalization_phase,
                            &bridge.cancellation,
                            cancellation_request_id,
                        ) {
                            cancellation_request_id = Some(
                                self.cancel(
                                    prompt_id,
                                    attempt_id,
                                    "provider-v2 app route was cancelled",
                                )
                                .await?,
                            );
                        }
                        if cancellation_request_id.is_some() {
                            match &outcome {
                                WorkerPluginExecutionOutcome::Failed(
                                    WorkerPluginExecutionFailure::Cancelled,
                                ) if cancellation_lifecycle_observed => {
                                    return Ok((outcome, None));
                                }
                                WorkerPluginExecutionOutcome::Failed(
                                    WorkerPluginExecutionFailure::Cancelled,
                                ) => {
                                    return Err(RuntimeSupervisorError::Protocol(
                                        "provider-v2 cancellation terminal preceded its lifecycle"
                                            .to_owned(),
                                    ));
                                }
                                WorkerPluginExecutionOutcome::Succeeded(_) => continue,
                                WorkerPluginExecutionOutcome::Failed(failure) => {
                                    return Err(RuntimeSupervisorError::Protocol(
                                        format!(
                                            "provider-v2 cancelled execution returned a non-cancelled terminal: {failure:?}"
                                        ),
                                    ));
                                }
                            }
                        }
                        if matches!(outcome, WorkerPluginExecutionOutcome::Failed(_)) {
                            return Ok((outcome, None));
                        }
                        let (finalization, finalization_receiver) = async_channel::bounded(1);
                        bridge
                            .proposal
                            .try_send(RuntimeProviderV2Proposal {
                                outcome: outcome.clone(),
                                finalization,
                            })
                            .map_err(|error| {
                                RuntimeSupervisorError::Protocol(format!(
                                    "provider-v2 proposal route rejected the proposal: {error}"
                                ))
                            })?;
                        let finalized = loop {
                            if should_begin_provider_v2_cancellation(
                                finalization_phase,
                                &bridge.cancellation,
                                cancellation_request_id,
                            ) {
                                cancellation_request_id = Some(
                                    self.cancel(
                                        prompt_id,
                                        attempt_id,
                                        "provider-v2 app route was cancelled",
                                    )
                                    .await?,
                                );
                                break None;
                            }
                            let finalization_remaining = response_deadline
                                .saturating_duration_since(Instant::now());
                            if finalization_remaining.is_zero() {
                                return Err(RuntimeSupervisorError::Timeout {
                                    stage: "provider-v2 proposal finalization",
                                });
                            }
                            let poll = finalization_remaining.min(Duration::from_millis(10));
                            let received = smol::future::race(
                                async {
                                    finalization_receiver.recv().await.map(Some).map_err(|_| {
                                        RuntimeSupervisorError::Protocol(
                                            "provider-v2 proposal was dropped before finalization"
                                                .to_owned(),
                                        )
                                    })
                                },
                                async {
                                    supervisor_delay(poll).await;
                                    Ok(None)
                                },
                            )
                            .await;
                            let received = match received {
                                Ok(received) => received,
                                Err(_)
                                    if provider_v2_wait_close_is_cancellation(
                                        finalization_phase,
                                        &bridge.cancellation,
                                    ) =>
                                {
                                    if cancellation_request_id.is_none() {
                                        cancellation_request_id = Some(
                                            self.cancel(
                                                prompt_id,
                                                attempt_id,
                                                "provider-v2 app route was cancelled",
                                            )
                                            .await?,
                                        );
                                    }
                                    break None;
                                }
                                Err(error) => return Err(error),
                            };
                            if received.is_some() {
                                break received;
                            }
                        };
                        let Some(finalized) = finalized else {
                            continue;
                        };
                        let RuntimeProviderV2FinalizedProposal {
                            finalization,
                            materialization,
                        } = finalized;
                        if should_begin_provider_v2_cancellation(
                            finalization_phase,
                            &bridge.cancellation,
                            cancellation_request_id,
                        ) {
                            cancellation_request_id = Some(
                                self.cancel(
                                    prompt_id,
                                    attempt_id,
                                    "provider-v2 app route was cancelled",
                                )
                                .await?,
                            );
                            continue;
                        }
                        self.send_for_existing_request(
                            request_id,
                            Some(prompt_id),
                            Some(attempt_id),
                            WorkerMessage::ProviderV2ProposalFinalization {
                                finalization: finalization.clone(),
                            },
                        )
                        .await?;
                        finalization_phase = ProviderV2FinalizationPhase::Committed;
                        proposal_outcome = Some(outcome);
                        expected_finalization = Some(finalization);
                        proposal_materialization = Some(materialization);
                    }
                    WorkerMessage::ProviderV2ProposalFinalizationAck { acknowledgement } => {
                        acknowledgement.validate().map_err(|error| {
                            RuntimeSupervisorError::Protocol(format!(
                                "provider-v2 finalization acknowledgement is invalid: {error}"
                            ))
                        })?;
                        let expected = expected_finalization.take().ok_or_else(|| {
                            RuntimeSupervisorError::Protocol(
                                "provider-v2 finalization acknowledgement preceded its proposal"
                                    .to_owned(),
                            )
                        })?;
                        if acknowledgement.finalization != expected {
                            return Err(RuntimeSupervisorError::IdentityMismatch);
                        }
                        acknowledgement.result.map_err(|error| {
                            RuntimeSupervisorError::Protocol(format!(
                                "provider-v2 finalization was rejected: {error}"
                            ))
                        })?;
                        let outcome = proposal_outcome.take().ok_or_else(|| {
                            RuntimeSupervisorError::Protocol(
                                "provider-v2 finalization lost its proposal outcome".to_owned(),
                            )
                        })?;
                        let materialization = proposal_materialization.take().ok_or_else(|| {
                            RuntimeSupervisorError::Protocol(
                                "provider-v2 finalization lost its retained materialization"
                                    .to_owned(),
                            )
                        })?;
                        return Ok((outcome, Some(materialization)));
                    }
                    WorkerMessage::Fatal { code, message } => {
                        return Err(RuntimeSupervisorError::WorkerFatal { code, message });
                    }
                    message => {
                        return Err(RuntimeSupervisorError::Protocol(format!(
                            "worker emitted {message:?} during provider-v2 execution"
                        )));
                    }
                }
            }
        }
        .await;
        bridge.stream_calls.close();
        bridge.proposal.close();
        bridge.validator.revoke();
        let successful = matches!(
            &result,
            Ok((WorkerPluginExecutionOutcome::Succeeded(_), Some(_)))
        );
        if !successful {
            bridge.cancellation.cancel();
        }
        let mut shared = self.shared.lock();
        if let Some(cancellation_request_id) = cancellation_request_id {
            shared.retire_provider_v2_cancel_request(cancellation_request_id);
        }
        shared.retire_provider_v2_request(request_id);
        result
    }

    pub async fn cancel(
        &mut self,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        reason: impl Into<String>,
    ) -> Result<RequestId, RuntimeSupervisorError> {
        {
            let mut shared = self.shared.lock();
            shared.snapshot.health = WorkerHealth::Cancelling;
            shared.record(
                WorkerOperationStage::Cancel,
                "attempt cancellation requested",
            );
        }
        self.send(
            Some(prompt_id),
            Some(attempt_id),
            WorkerMessage::Cancel {
                reason: reason.into(),
            },
        )
        .await
    }

    pub async fn next_event(
        &self,
        timeout: Duration,
    ) -> Result<WorkerEnvelope, RuntimeSupervisorError> {
        self.receive_one(timeout).await
    }

    pub async fn shutdown(&mut self) -> Result<ExitStatus, RuntimeSupervisorError> {
        if self.child.is_none() {
            return Err(RuntimeSupervisorError::NotRunning);
        }
        self.request_shutdown().await?;
        let deadline = Instant::now() + self.policy.shutdown_timeout;
        let mut acknowledged = false;
        while Instant::now() < deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match self.receive_one(remaining).await {
                Ok(envelope) if matches!(envelope.message, WorkerMessage::Shutdown) => {
                    acknowledged = true;
                    break;
                }
                Ok(_) => {}
                Err(RuntimeSupervisorError::Timeout { .. }) => break,
                Err(error) => return Err(error),
            }
        }
        drop(self.input.take());
        let remaining = deadline.saturating_duration_since(Instant::now());
        let status = if acknowledged {
            self.wait_for_exit(remaining).await?
        } else {
            None
        };
        let status = match status {
            Some(status) => status,
            None => {
                self.shared
                    .lock()
                    .record(WorkerOperationStage::Kill, "shutdown deadline expired");
                self.kill_process()?;
                self.wait_for_exit(self.policy.shutdown_timeout)
                    .await?
                    .ok_or(RuntimeSupervisorError::Timeout {
                        stage: "forced worker termination",
                    })?
            }
        };
        self.record_exit(status);
        drop(self.child.take());
        if status.success() {
            Ok(status)
        } else {
            Err(RuntimeSupervisorError::ExitFailure {
                code: status.code(),
            })
        }
    }

    pub async fn request_shutdown(&mut self) -> Result<RequestId, RuntimeSupervisorError> {
        if self.child.is_none() {
            return Err(RuntimeSupervisorError::NotRunning);
        }
        {
            let mut shared = self.shared.lock();
            shared.shutdown_requested = true;
            shared.snapshot.health = WorkerHealth::Cancelling;
            shared.record(
                WorkerOperationStage::Shutdown,
                "graceful shutdown requested",
            );
        }
        self.send(None, None, WorkerMessage::Shutdown).await
    }

    pub async fn terminate(&mut self) -> Result<ExitStatus, RuntimeSupervisorError> {
        if self.child.is_none() {
            return Err(RuntimeSupervisorError::NotRunning);
        }
        self.shared.lock().record(
            WorkerOperationStage::Kill,
            "explicit process-tree termination",
        );
        self.kill_process()?;
        drop(self.input.take());
        let status = self
            .wait_for_exit(self.policy.shutdown_timeout)
            .await?
            .ok_or(RuntimeSupervisorError::Timeout {
                stage: "worker termination",
            })?;
        self.record_exit(status);
        drop(self.child.take());
        Ok(status)
    }

    pub async fn recover(mut self) -> Result<Self, RuntimeSupervisorError> {
        let health = self.snapshot().health;
        if !matches!(
            health,
            WorkerHealth::Degraded { .. }
                | WorkerHealth::Hung
                | WorkerHealth::Lost
                | WorkerHealth::Exited { success: false, .. }
        ) {
            return Err(RuntimeSupervisorError::RecoveryNotEligible { health });
        }
        if self.automatic_restarts >= self.policy.maximum_automatic_restarts {
            return Err(RuntimeSupervisorError::RecoveryBudgetExhausted {
                maximum: self.policy.maximum_automatic_restarts,
            });
        }
        let next_restart = self.automatic_restarts.checked_add(1).ok_or(
            RuntimeSupervisorError::RecoveryBudgetExhausted {
                maximum: self.policy.maximum_automatic_restarts,
            },
        )?;
        self.shared.lock().record(
            WorkerOperationStage::Recover,
            format!("bounded automatic restart {next_restart} requested"),
        );
        if self.child.is_some() {
            self.kill_process()?;
            drop(self.input.take());
            let status = self
                .wait_for_exit(self.policy.shutdown_timeout)
                .await?
                .ok_or(RuntimeSupervisorError::Timeout {
                    stage: "failed-worker termination",
                })?;
            self.record_exit(status);
            drop(self.child.take());
        }
        supervisor_delay(self.policy.restart_backoff).await;
        let mut replacement = Self::start(self.launch_config.clone()).await?;
        replacement.automatic_restarts = next_restart;
        replacement.shared.lock().record(
            WorkerOperationStage::Recover,
            format!("replacement worker ready after restart {next_restart}"),
        );
        Ok(replacement)
    }

    async fn send(
        &mut self,
        prompt_id: Option<PromptId>,
        attempt_id: Option<AttemptId>,
        message: WorkerMessage,
    ) -> Result<RequestId, RuntimeSupervisorError> {
        let kind = RequestKind::from(&message);
        self.send_with_kind(prompt_id, attempt_id, message, kind)
            .await
    }

    async fn send_with_kind(
        &mut self,
        prompt_id: Option<PromptId>,
        attempt_id: Option<AttemptId>,
        message: WorkerMessage,
        kind: RequestKind,
    ) -> Result<RequestId, RuntimeSupervisorError> {
        let request_id = RequestId(Uuid::new_v4());
        let sequence = self.next_input_sequence;
        self.next_input_sequence = self.next_input_sequence.checked_add(1).ok_or_else(|| {
            RuntimeSupervisorError::Protocol("input sequence overflow".to_owned())
        })?;
        let envelope = WorkerEnvelope {
            version: WORKER_PROTOCOL_VERSION,
            profile_id: self.profile_id,
            worker_id: self.worker_id,
            request_id,
            prompt_id,
            attempt_id,
            sequence,
            registry_version: self.registry_version.clone(),
            message,
            extensions: Default::default(),
        };
        self.shared.lock().register_request(
            request_id,
            RequestScope {
                prompt_id,
                attempt_id,
                kind,
            },
        );
        let input = self
            .input
            .as_mut()
            .ok_or(RuntimeSupervisorError::NotRunning)?;
        write_async_frame(input, &envelope).await?;
        Ok(request_id)
    }

    async fn send_for_existing_request(
        &mut self,
        request_id: RequestId,
        prompt_id: Option<PromptId>,
        attempt_id: Option<AttemptId>,
        message: WorkerMessage,
    ) -> Result<(), RuntimeSupervisorError> {
        let scope = self
            .shared
            .lock()
            .request_scopes
            .get(&request_id)
            .copied()
            .ok_or(RuntimeSupervisorError::IdentityMismatch)?;
        if scope.prompt_id != prompt_id
            || scope.attempt_id != attempt_id
            || !matches!(
                scope.kind,
                RequestKind::Execute | RequestKind::ExecutePlugin | RequestKind::ExecuteProviderV2
            )
        {
            return Err(RuntimeSupervisorError::IdentityMismatch);
        }
        let sequence = self.next_input_sequence;
        self.next_input_sequence = self.next_input_sequence.checked_add(1).ok_or_else(|| {
            RuntimeSupervisorError::Protocol("input sequence overflow".to_owned())
        })?;
        let envelope = WorkerEnvelope {
            version: WORKER_PROTOCOL_VERSION,
            profile_id: self.profile_id,
            worker_id: self.worker_id,
            request_id,
            prompt_id,
            attempt_id,
            sequence,
            registry_version: self.registry_version.clone(),
            message,
            extensions: Default::default(),
        };
        let input = self
            .input
            .as_mut()
            .ok_or(RuntimeSupervisorError::NotRunning)?;
        write_async_frame(input, &envelope).await
    }

    pub async fn respond_plugin_capability(
        &mut self,
        request_id: RequestId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        call_id: u64,
        response: Vec<u8>,
    ) -> Result<(), RuntimeSupervisorError> {
        self.send_for_existing_request(
            request_id,
            Some(prompt_id),
            Some(attempt_id),
            WorkerMessage::PluginCapabilityResponse { call_id, response },
        )
        .await
    }

    pub async fn respond_model_source(
        &mut self,
        request_id: RequestId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        call_id: u64,
        response: WorkerModelSourceResponse,
    ) -> Result<(), RuntimeSupervisorError> {
        self.send_for_existing_request(
            request_id,
            Some(prompt_id),
            Some(attempt_id),
            WorkerMessage::ModelSourceResponse { call_id, response },
        )
        .await
    }

    async fn receive_one(
        &self,
        timeout: Duration,
    ) -> Result<WorkerEnvelope, RuntimeSupervisorError> {
        self.receive_one_at_stage(timeout, "worker response").await
    }

    async fn receive_one_at_stage(
        &self,
        timeout: Duration,
        stage: &'static str,
    ) -> Result<WorkerEnvelope, RuntimeSupervisorError> {
        smol::future::race(
            async {
                self.incoming
                    .recv()
                    .await
                    .map_err(|_| RuntimeSupervisorError::ChannelClosed)?
            },
            async {
                supervisor_delay(timeout).await;
                Err(RuntimeSupervisorError::Timeout { stage })
            },
        )
        .await
    }

    async fn wait_for_exit(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<ExitStatus>, RuntimeSupervisorError> {
        let child = self
            .child
            .as_mut()
            .ok_or(RuntimeSupervisorError::NotRunning)?;
        smol::future::race(
            async {
                child
                    .status()
                    .await
                    .map(Some)
                    .map_err(|error| RuntimeSupervisorError::Io(error.to_string()))
            },
            async {
                supervisor_delay(timeout).await;
                Ok(None)
            },
        )
        .await
    }

    fn kill_process(&mut self) -> Result<(), RuntimeSupervisorError> {
        self.child
            .as_mut()
            .ok_or(RuntimeSupervisorError::NotRunning)?
            .kill()
            .map_err(|error| RuntimeSupervisorError::Io(error.to_string()))
    }

    fn record_exit(&self, status: ExitStatus) {
        self.shared.lock().record_exit(status);
    }
}

impl Drop for RuntimeSupervisor {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child
            && let Err(error) = child.kill()
        {
            let error = RuntimeSupervisorError::Io(error.to_string());
            self.shared.lock().fail(&error);
        }
    }
}

async fn read_worker_output(
    mut output: impl AsyncRead + Send + Unpin + 'static,
    shared: Arc<Mutex<SupervisorShared>>,
    sender: async_channel::Sender<Result<WorkerEnvelope, RuntimeSupervisorError>>,
) {
    loop {
        match read_async_frame(&mut output).await {
            Ok(envelope) => {
                let accepted = shared.lock().accept(&envelope);
                match accepted {
                    Ok(()) if matches!(envelope.message, WorkerMessage::Heartbeat) => {}
                    Ok(()) => {
                        if sender.send(Ok(envelope)).await.is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        if sender.send(Err(error)).await.is_err() {
                            break;
                        }
                        break;
                    }
                }
            }
            Err(error) => {
                let shutdown_requested = shared.lock().shutdown_requested;
                if !shutdown_requested {
                    {
                        let mut shared = shared.lock();
                        shared.snapshot.health = WorkerHealth::Degraded {
                            reason: error.to_string(),
                        };
                        shared.fail(&error);
                    }
                    if sender.send(Err(error)).await.is_err() {
                        break;
                    }
                }
                break;
            }
        }
    }
}

async fn read_worker_logs(
    mut error_output: impl AsyncRead + Send + Unpin + 'static,
    shared: Arc<Mutex<SupervisorShared>>,
) {
    let mut buffer = [0_u8; 4096];
    loop {
        match error_output.read(&mut buffer).await {
            Ok(0) => break,
            Ok(length) => {
                let Some(bytes) = buffer.get(..length) else {
                    break;
                };
                shared.lock().logs.push(bytes);
            }
            Err(error) => {
                let error = RuntimeSupervisorError::Io(error.to_string());
                shared.lock().fail(&error);
                break;
            }
        }
    }
}

async fn monitor_heartbeats(shared: Arc<Mutex<SupervisorShared>>, policy: SupervisorPolicy) {
    loop {
        supervisor_delay(policy.heartbeat_interval).await;
        let mut shared = shared.lock();
        shared.evaluate_heartbeat(Instant::now(), policy);
        if shared.snapshot.health.is_terminal() {
            break;
        }
    }
}

// Runtime supervision has no GPUI context; this single owner keeps production
// deadlines async while GPUI-facing callers retain their own executor timers.
#[allow(clippy::disallowed_methods)]
pub(crate) async fn supervisor_delay(duration: Duration) {
    smol::Timer::after(duration).await;
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub async fn supervisor_test_delay(duration: Duration) {
    supervisor_delay(duration).await;
}

async fn write_async_frame(
    writer: &mut WorkerInput,
    envelope: &WorkerEnvelope,
) -> Result<(), RuntimeSupervisorError> {
    validate_event_limit(envelope)?;
    let frame = encode_worker_frame(envelope)
        .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))?;
    writer
        .write_all(&frame)
        .await
        .map_err(|error| RuntimeSupervisorError::Io(error.to_string()))?;
    writer
        .flush()
        .await
        .map_err(|error| RuntimeSupervisorError::Io(error.to_string()))
}

async fn read_async_frame(
    reader: &mut (impl AsyncRead + Unpin),
) -> Result<WorkerEnvelope, RuntimeSupervisorError> {
    let mut prefix = [0_u8; 4];
    reader
        .read_exact(&mut prefix)
        .await
        .map_err(|error| RuntimeSupervisorError::Io(error.to_string()))?;
    let length = u32::from_le_bytes(prefix) as usize;
    if length > MAX_WORKER_FRAME_BYTES {
        return Err(RuntimeSupervisorError::Protocol(format!(
            "worker frame exceeds {MAX_WORKER_FRAME_BYTES} bytes"
        )));
    }
    let mut frame = Vec::new();
    frame
        .try_reserve_exact(4_usize.saturating_add(length))
        .map_err(|error| RuntimeSupervisorError::Io(error.to_string()))?;
    frame.extend_from_slice(&prefix);
    frame.resize(4 + length, 0);
    let payload = frame
        .get_mut(4..)
        .ok_or_else(|| RuntimeSupervisorError::Protocol("invalid frame length".to_owned()))?;
    reader
        .read_exact(payload)
        .await
        .map_err(|error| RuntimeSupervisorError::Io(error.to_string()))?;
    let envelope = decode_worker_frame(&frame)
        .map_err(|error| RuntimeSupervisorError::Protocol(error.to_string()))?;
    validate_event_limit(&envelope)?;
    Ok(envelope)
}

fn validate_event_limit(envelope: &WorkerEnvelope) -> Result<(), RuntimeSupervisorError> {
    if let WorkerMessage::Event { event } = &envelope.message
        && event.len() > MAX_ENCODED_PREVIEW_BYTES
    {
        return Err(RuntimeSupervisorError::Protocol(format!(
            "worker event exceeds {MAX_ENCODED_PREVIEW_BYTES} bytes"
        )));
    }
    Ok(())
}

fn launch_record(
    config: &WorkerLaunchConfig,
) -> Result<WorkerLaunchRecord, RuntimeSupervisorError> {
    let mut arguments = config.arguments.clone();
    arguments.extend(config.backend_selection.launch_arguments()?);
    arguments.extend(general_video_codec_package_launch_arguments(
        config.general_video_codec_package.as_ref(),
    )?);
    arguments.push("--memory-limit-bytes".to_owned());
    arguments.push(config.memory_limit_bytes.to_string());
    Ok(WorkerLaunchRecord {
        executable: sanitized_path(&config.binary),
        arguments: sanitize_arguments(&arguments),
        working_directory: config.working_directory.as_deref().map(sanitized_path),
        environment_names: config
            .environment
            .iter()
            .map(|(name, _)| name.clone())
            .collect(),
        environment_source: if config.environment.is_empty() {
            WorkerEnvironmentSource::InheritedSim
        } else {
            WorkerEnvironmentSource::InheritedSimWithExplicitOverrides
        },
        ownership: if cfg!(windows) {
            ProcessOwnership::WindowsJobObject
        } else {
            ProcessOwnership::ProcessGroup
        },
        started_at: Utc::now(),
    })
}

fn missing_pipe_after_spawn(child: &mut Child, pipe: &'static str) -> RuntimeSupervisorError {
    match child.kill() {
        Ok(()) => RuntimeSupervisorError::MissingPipe(pipe),
        Err(error) => RuntimeSupervisorError::Spawn(format!(
            "worker had no {pipe} pipe and process-tree cleanup failed: {error}"
        )),
    }
}

fn sanitized_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<non-utf8-path>")
        .to_owned()
}

fn sanitize_arguments(arguments: &[String]) -> Vec<String> {
    let mut redact_next = false;
    arguments
        .iter()
        .map(|argument| {
            if redact_next {
                redact_next = false;
                return "[redacted]".to_owned();
            }
            let lowercase = argument.to_ascii_lowercase();
            let sensitive = [
                "token",
                "secret",
                "password",
                "credential",
                "api-key",
                "package-root",
                "public-key",
            ]
            .iter()
            .any(|needle| lowercase.contains(needle));
            if sensitive {
                if let Some((name, _)) = argument.split_once('=') {
                    return format!("{name}=[redacted]");
                }
                redact_next = true;
            }
            argument.clone()
        })
        .collect()
}

fn sanitize_log(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\r' | '\t'))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    #[cfg(unix)]
    use std::collections::BTreeSet;

    use super::*;

    fn identifiers() -> (ProfileId, WorkerId, RequestId) {
        (
            ProfileId(Uuid::new_v4()),
            WorkerId(Uuid::new_v4()),
            RequestId(Uuid::new_v4()),
        )
    }

    fn shared() -> (SupervisorShared, RequestId) {
        let (profile_id, worker_id, request_id) = identifiers();
        let config = WorkerLaunchConfig::new(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            1024,
        );
        let mut shared = SupervisorShared {
            snapshot: RuntimeSupervisorSnapshot {
                profile_id,
                worker_id,
                registry_version: "registry-v1".to_owned(),
                health: WorkerHealth::ProcessAlive,
                missed_heartbeats: 0,
                active_prompt_id: None,
                active_attempt_id: None,
                launch: launch_record(&config).expect("valid test launch record"),
                operation: WorkerOperationRecord {
                    operation_id: Uuid::new_v4(),
                    profile_id,
                    worker_id,
                    transitions: Vec::new(),
                    terminal_error: None,
                },
            },
            requested_backend: CpuBackend::capability_matrix(),
            accepted_backend: None,
            request_scopes: HashMap::new(),
            request_order: VecDeque::new(),
            last_worker_sequence: None,
            last_heartbeat: None,
            monitoring_active: false,
            shutdown_requested: false,
            logs: BoundedLog::new(),
        };
        shared.register_request(
            request_id,
            RequestScope {
                prompt_id: None,
                attempt_id: None,
                kind: RequestKind::Hello,
            },
        );
        (shared, request_id)
    }

    #[cfg(unix)]
    fn observe_exit_without_protocol_changes(
        status: ExitStatus,
        expected_health: WorkerHealth,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let (mut state, _) = shared();
        state.snapshot.health = WorkerHealth::BackendReady;
        state.snapshot.missed_heartbeats = 2;
        state.snapshot.operation.terminal_error = Some("retained earlier observation".to_owned());
        state.register_request(
            RequestId(Uuid::new_v4()),
            RequestScope {
                prompt_id: Some(PromptId(Uuid::new_v4())),
                attempt_id: Some(AttemptId(Uuid::new_v4())),
                kind: RequestKind::Execute,
            },
        );
        state.accepted_backend = Some(CpuBackend::capability_matrix());
        state.last_worker_sequence = Some(31);
        state.last_heartbeat = Some(Instant::now());
        state.shutdown_requested = true;
        state.monitoring_active = true;
        state.logs.push(b"retained worker stderr\n");
        state.record(WorkerOperationStage::Prepare, "retained operation prefix");

        let before = state.snapshot.clone();
        let request_scopes = state.request_scopes.clone();
        let request_order = state.request_order.clone();
        let requested_backend = state.requested_backend.clone();
        let accepted_backend = state.accepted_backend.clone();
        let last_worker_sequence = state.last_worker_sequence;
        let last_heartbeat = state.last_heartbeat;
        let shutdown_requested = state.shutdown_requested;
        let logs = state.logs.entries();

        state.record_exit(status);

        assert_eq!(state.snapshot.health, expected_health);
        assert!(state.snapshot.health.is_terminal());
        assert!(!state.monitoring_active);
        assert_eq!(state.request_scopes, request_scopes);
        assert_eq!(state.request_order, request_order);
        assert_eq!(state.requested_backend, requested_backend);
        assert_eq!(state.accepted_backend, accepted_backend);
        assert_eq!(state.last_worker_sequence, last_worker_sequence);
        assert_eq!(state.last_heartbeat, last_heartbeat);
        assert_eq!(state.shutdown_requested, shutdown_requested);
        assert_eq!(state.logs.entries(), logs);
        assert_eq!(
            state.snapshot.operation.transitions.len(),
            before.operation.transitions.len() + 1
        );
        let mut after = state.snapshot;
        let transition = after
            .operation
            .transitions
            .pop()
            .ok_or("exit transition is absent")?;
        assert_eq!(transition.stage, WorkerOperationStage::Exit);
        after.health = before.health.clone();
        assert_eq!(after, before);
        Ok(transition.detail)
    }

    #[cfg(unix)]
    #[test]
    fn unix_exit_normal_diagnostic_preserves_existing_classification()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::process::ExitStatusExt as _;

        let mut details = BTreeSet::new();
        for code in [0, 7] {
            let status = ExitStatus::from_raw(code << 8);
            assert_eq!(status.code(), Some(code));
            assert_eq!(status.signal(), None);
            assert!(!status.core_dumped());
            let detail = observe_exit_without_protocol_changes(
                status,
                WorkerHealth::Exited {
                    success: code == 0,
                    code: Some(code),
                },
            )?;
            assert!(detail.contains(&format!("code Some({code})")));
            assert!(detail.contains(&format!("status=exit status: {code}")));
            details.insert(detail);
        }
        assert_eq!(details.len(), 2);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn unix_exit_sigterm_diagnostic_preserves_existing_classification()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::process::ExitStatusExt as _;

        let status = ExitStatus::from_raw(libc::SIGTERM);
        assert_eq!(status.code(), None);
        assert_eq!(status.signal(), Some(libc::SIGTERM));
        assert!(!status.core_dumped());
        let termination = observe_exit_without_protocol_changes(
            status,
            WorkerHealth::Exited {
                success: false,
                code: None,
            },
        )?;
        let crash = observe_exit_without_protocol_changes(
            ExitStatus::from_raw(libc::SIGSEGV),
            WorkerHealth::Exited {
                success: false,
                code: None,
            },
        )?;
        let normal = observe_exit_without_protocol_changes(
            ExitStatus::from_raw(0),
            WorkerHealth::Exited {
                success: true,
                code: Some(0),
            },
        )?;
        assert!(termination.contains("code None"));
        assert!(termination.contains(&format!("status=signal: {}", libc::SIGTERM)));
        assert!(!termination.contains("core dumped"));
        assert_ne!(termination, crash);
        assert_ne!(termination, normal);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn unix_exit_sigsegv_diagnostic_preserves_existing_classification()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::process::ExitStatusExt as _;

        let mut details = BTreeSet::new();
        for core_dumped in [false, true] {
            let status = ExitStatus::from_raw(libc::SIGSEGV | if core_dumped { 0x80 } else { 0 });
            assert_eq!(status.code(), None);
            assert_eq!(status.signal(), Some(libc::SIGSEGV));
            assert_eq!(status.core_dumped(), core_dumped);
            let detail = observe_exit_without_protocol_changes(
                status,
                WorkerHealth::Exited {
                    success: false,
                    code: None,
                },
            )?;
            assert!(detail.contains("code None"));
            assert!(detail.contains(&format!("status=signal: {}", libc::SIGSEGV)));
            assert_eq!(detail.contains("core dumped"), core_dumped);
            details.insert(detail);
        }
        assert_eq!(details.len(), 2);
        Ok(())
    }

    fn empty_registry_plan(
        seed: [u8; 32],
    ) -> Result<WorkerRegistryDeploymentPlan, Box<dyn std::error::Error>> {
        let begin = WorkerRegistryDeploymentBegin::new(
            sim_types::WorkerRegistryGeneration::new(1)?,
            sim_types::WorkerSha256Digest::new("0".repeat(64))?,
            Vec::new(),
        )?;
        let verifier = crate::PluginAuthorizationSealer::from_seed(
            seed,
            crate::PermissionPolicyGeneration::new(1)?,
        )?
        .verifier()?;
        Ok(WorkerRegistryDeploymentPlan::new(
            begin,
            Vec::new(),
            verifier,
        )?)
    }

    #[test]
    fn worker_launch_and_deployment_are_bound_to_one_authorization_verifier()
    -> Result<(), Box<dyn std::error::Error>> {
        let (profile_id, worker_id, _) = identifiers();
        let deployment = empty_registry_plan([3; 32])?;
        let mut config = WorkerLaunchConfig::new(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            1024,
        )
        .with_registry_deployment(deployment.clone());
        assert_eq!(
            config.plugin_authorization_verifier.as_ref(),
            Some(deployment.authorization_verifier())
        );
        config.validate()?;

        config.plugin_authorization_verifier = Some(
            crate::PluginAuthorizationSealer::from_seed(
                [4; 32],
                crate::PermissionPolicyGeneration::new(1)?,
            )?
            .verifier()?,
        );
        assert!(matches!(
            config.validate(),
            Err(RuntimeSupervisorError::InvalidConfiguration(message))
                if message.contains("authorization verifier")
        ));
        Ok(())
    }

    #[test]
    fn registry_rejection_is_nonfatal_and_bound_to_the_requested_generation()
    -> Result<(), Box<dyn std::error::Error>> {
        let deployment = empty_registry_plan([5; 32])?;
        let rejection = sim_types::WorkerRegistryDeploymentRejection::new(
            deployment.begin().generation(),
            deployment.begin().registry_digest_sha256().clone(),
            sim_types::WorkerRegistryDeploymentRejectionReason::ComponentCompilationFailed,
        );
        assert!(matches!(
            validate_registry_deployment_response(
                WorkerMessage::RegistryDeploymentRejected {
                    rejection: rejection.clone(),
                },
                deployment.begin(),
            ),
            Err(RuntimeSupervisorError::InvalidRegistryDeployment(message))
                if message.contains("ComponentCompilationFailed")
        ));

        let (mut shared, _) = shared();
        let request_id = RequestId(Uuid::new_v4());
        shared.snapshot.health = WorkerHealth::BackendReady;
        shared.register_request(
            request_id,
            RequestScope {
                prompt_id: None,
                attempt_id: None,
                kind: RequestKind::RegistryDeployment,
            },
        );
        shared
            .accept(&envelope(
                &shared,
                request_id,
                0,
                WorkerMessage::RegistryDeploymentRejected { rejection },
            ))
            .expect("matching registry rejection is a valid worker response");
        assert_eq!(shared.snapshot.health, WorkerHealth::BackendReady);
        assert!(shared.snapshot.operation.terminal_error.is_none());
        Ok(())
    }

    fn envelope(
        shared: &SupervisorShared,
        request_id: RequestId,
        sequence: u64,
        message: WorkerMessage,
    ) -> WorkerEnvelope {
        WorkerEnvelope {
            version: WORKER_PROTOCOL_VERSION,
            profile_id: shared.snapshot.profile_id,
            worker_id: shared.snapshot.worker_id,
            request_id,
            prompt_id: None,
            attempt_id: None,
            sequence,
            registry_version: shared.snapshot.registry_version.clone(),
            message,
            extensions: BTreeMap::new(),
        }
    }

    fn cpu_backend_wire() -> sim_types::WorkerBackendCapabilities {
        CpuBackend::capability_matrix()
            .to_worker_capabilities()
            .expect("CPU capabilities project to worker protocol")
    }

    #[test]
    fn handshake_ready_and_heartbeat_states_are_distinct() {
        let (mut shared, request_id) = shared();
        let hello = envelope(
            &shared,
            request_id,
            0,
            WorkerMessage::HelloAck {
                accepted_backend: cpu_backend_wire(),
            },
        );
        shared.accept(&hello).expect("hello accepted");
        assert_eq!(shared.snapshot.health, WorkerHealth::IpcReachable);
        let ready = envelope(&shared, request_id, 1, WorkerMessage::Ready);
        shared.accept(&ready).expect("ready accepted");
        assert_eq!(shared.snapshot.health, WorkerHealth::BackendReady);
        let heartbeat = envelope(&shared, request_id, 2, WorkerMessage::Heartbeat);
        shared.accept(&heartbeat).expect("heartbeat accepted");
        assert_eq!(shared.snapshot.missed_heartbeats, 0);
    }

    #[test]
    fn protocol_sequence_skew_isolated_as_domain_state() {
        let (mut shared, request_id) = shared();
        let skewed = envelope(
            &shared,
            request_id,
            2,
            WorkerMessage::HelloAck {
                accepted_backend: cpu_backend_wire(),
            },
        );
        assert_eq!(
            shared.accept(&skewed),
            Err(RuntimeSupervisorError::Sequence {
                expected: 0,
                actual: 2
            })
        );
        assert!(matches!(
            shared.snapshot.health,
            WorkerHealth::ProtocolIncompatible { .. }
        ));
    }

    #[test]
    fn provider_stream_messages_are_rejected_until_the_canonical_bridge_is_active() {
        for message in [
            WorkerMessage::ProviderStreamRequest {
                call_id: 1,
                request: sim_types::WorkerProviderStreamRequest::CheckCancelled(
                    sim_types::WorkerProviderStreamHandle {
                        session_id: Uuid::from_u128(1),
                        session_generation: 1,
                        invocation: 1,
                        slot: 1,
                        generation: 1,
                    },
                ),
            },
            WorkerMessage::ProviderStreamResponse {
                call_id: 1,
                response: sim_types::WorkerProviderStreamResponse::Unit(Ok(())),
            },
        ] {
            let (mut shared, request_id) = shared();
            let message = envelope(&shared, request_id, 0, message);
            let error = shared
                .accept(&message)
                .expect_err("provider streaming requires an active canonical bridge");
            assert!(
                matches!(
                    &error,
                    RuntimeSupervisorError::Protocol(message)
                        if message.contains("invalid direction or execution scope")
                ),
                "unexpected pre-bridge provider-stream rejection: {error:?}"
            );
            assert!(matches!(
                shared.snapshot.health,
                WorkerHealth::ProtocolIncompatible { .. }
            ));
        }
    }

    #[test]
    fn provider_v2_bridge_is_capacity_one_and_finalization_is_consuming() {
        let context = sim_types::WorkerProviderInvocationContext {
            session_id: uuid::Uuid::from_u128(0x425_300),
            session_generation: 3,
            invocation: 5,
            generation: 7,
        };
        let contract = sim_types::WorkerProviderStreamingContract {
            methods: vec![sim_types::WorkerProviderHttpMethod::Post],
            maximum_headers: 4,
            maximum_header_bytes: 1024,
            maximum_request_body_bytes: 1024,
            maximum_response_body_bytes: 1024,
            maximum_chunk_bytes: 256,
            maximum_ndjson_line_bytes: 256,
            maximum_wait_milliseconds: 100,
            maximum_uploads: 1,
            maximum_upload_body_bytes: 1024,
            maximum_cost_requests: 1,
            maximum_progress_total: 100,
            uploads: true,
            cost_requests: true,
        };
        let (bridge, stream_receiver, proposal_receiver) = RuntimeProviderV2Bridge::capacity_one(
            context.clone(),
            contract,
            sim_types::CancellationToken::default(),
            Duration::from_secs(1),
        )
        .expect("checked bridge");
        let request = sim_types::WorkerProviderStreamRequest::CheckCancelled(
            sim_types::WorkerProviderStreamHandle {
                session_id: uuid::Uuid::from_u128(0x425_300),
                session_generation: 3,
                invocation: 5,
                slot: 1,
                generation: 7,
            },
        );
        let (first_response, first_receiver) = async_channel::bounded(1);
        bridge
            .stream_calls
            .try_send(RuntimeProviderV2StreamCall {
                call_id: 1,
                request: request.clone(),
                response: first_response,
            })
            .expect("first provider call occupies the route");
        let (second_response, _) = async_channel::bounded(1);
        assert!(
            bridge
                .stream_calls
                .try_send(RuntimeProviderV2StreamCall {
                    call_id: 2,
                    request,
                    response: second_response,
                })
                .is_err()
        );
        stream_receiver
            .recv_blocking()
            .expect("first provider call remains queued")
            .respond(sim_types::WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("response returns to the execution route");
        assert_eq!(
            first_receiver
                .recv_blocking()
                .expect("response is retained"),
            sim_types::WorkerProviderStreamResponse::Unit(Ok(()))
        );

        let finalization = WorkerProviderV2ProposalFinalization {
            handle: sim_types::WorkerProviderStreamHandle {
                session_id: context.session_id,
                session_generation: context.session_generation,
                invocation: context.invocation,
                slot: 1,
                generation: context.generation,
            },
            context,
            proposal_generation: 11,
            finalization_nonce: [0x42; 32],
            receipt_identity_sha256: sim_types::WorkerSha256Digest::new("a".repeat(64))
                .expect("receipt identity"),
            materialization_identity_sha256: sim_types::WorkerSha256Digest::new("b".repeat(64))
                .expect("materialization identity"),
        };
        let (sender, receiver) = async_channel::bounded(1);
        bridge
            .proposal
            .try_send(RuntimeProviderV2Proposal {
                outcome: WorkerPluginExecutionOutcome::Succeeded(vec![1]),
                finalization: sender,
            })
            .expect("first proposal occupies the route");
        let proposal = proposal_receiver
            .recv_blocking()
            .expect("proposal remains armed");
        assert_eq!(
            proposal.outcome,
            WorkerPluginExecutionOutcome::Succeeded(vec![1])
        );
        let materialization = crate::ProviderTransportResponse::checked("fixture", Vec::new())
            .expect("checked materialization");
        proposal
            .finalize(finalization.clone(), materialization.clone())
            .expect("consuming proposal retains its finalization bundle once");
        let retained = receiver
            .recv_blocking()
            .expect("finalization bundle retained");
        assert_eq!(retained.finalization, finalization);
        assert_eq!(retained.materialization, materialization);

        let (closed_sender, closed_receiver) = async_channel::bounded(1);
        closed_receiver.close();
        let dropped = RuntimeProviderV2Proposal {
            outcome: WorkerPluginExecutionOutcome::Succeeded(vec![2]),
            finalization: closed_sender,
        };
        assert!(matches!(
            dropped.finalize(
                retained.finalization,
                crate::ProviderTransportResponse::checked("fixture", Vec::new())
                    .expect("checked rejected materialization"),
            ),
            Err(RuntimeSupervisorError::Protocol(message))
                if message.contains("could not be delivered")
        ));
    }

    #[test]
    fn provider_v2_cancellation_stops_at_the_finalization_commit_boundary() {
        let cancellation = sim_types::CancellationToken::default();
        let cancellation_request_id = RequestId(Uuid::new_v4());
        let succeeded = WorkerMessage::PluginResult {
            outcome: WorkerPluginExecutionOutcome::Succeeded(vec![0x42]),
        };
        let stream_request = WorkerMessage::ProviderStreamRequest {
            call_id: 1,
            request: sim_types::WorkerProviderStreamRequest::CheckCancelled(
                sim_types::WorkerProviderStreamHandle {
                    session_id: Uuid::from_u128(0x425_500),
                    session_generation: 3,
                    invocation: 5,
                    slot: 1,
                    generation: 7,
                },
            ),
        };

        assert!(!should_begin_provider_v2_cancellation(
            ProviderV2FinalizationPhase::PreCommit,
            &cancellation,
            None,
        ));
        cancellation.cancel();
        assert!(should_begin_provider_v2_cancellation(
            ProviderV2FinalizationPhase::PreCommit,
            &cancellation,
            None,
        ));
        assert!(provider_v2_wait_close_is_cancellation(
            ProviderV2FinalizationPhase::PreCommit,
            &cancellation,
        ));
        assert!(suppress_provider_v2_precommit_message(
            ProviderV2FinalizationPhase::PreCommit,
            Some(cancellation_request_id),
            &succeeded,
        ));
        assert!(suppress_provider_v2_precommit_message(
            ProviderV2FinalizationPhase::PreCommit,
            Some(cancellation_request_id),
            &stream_request,
        ));
        assert!(!should_begin_provider_v2_cancellation(
            ProviderV2FinalizationPhase::Committed,
            &cancellation,
            None,
        ));
        assert!(!provider_v2_wait_close_is_cancellation(
            ProviderV2FinalizationPhase::Committed,
            &cancellation,
        ));
        assert!(!provider_v2_wait_close_is_cancellation(
            ProviderV2FinalizationPhase::PreCommit,
            &sim_types::CancellationToken::default(),
        ));
        assert!(!suppress_provider_v2_precommit_message(
            ProviderV2FinalizationPhase::Committed,
            Some(cancellation_request_id),
            &succeeded,
        ));
        assert!(!suppress_provider_v2_precommit_message(
            ProviderV2FinalizationPhase::PreCommit,
            Some(cancellation_request_id),
            &WorkerMessage::ProviderV2ProposalFinalizationAck {
                acknowledgement: sim_types::WorkerProviderV2ProposalFinalizationAck {
                    finalization: WorkerProviderV2ProposalFinalization {
                        handle: sim_types::WorkerProviderStreamHandle {
                            session_id: Uuid::from_u128(0x425_500),
                            session_generation: 3,
                            invocation: 5,
                            slot: 1,
                            generation: 7,
                        },
                        context: sim_types::WorkerProviderInvocationContext {
                            session_id: Uuid::from_u128(0x425_500),
                            session_generation: 3,
                            invocation: 5,
                            generation: 7,
                        },
                        proposal_generation: 11,
                        finalization_nonce: [0x42; 32],
                        receipt_identity_sha256:
                            sim_types::WorkerSha256Digest::new("a".repeat(64),)
                                .expect("receipt identity"),
                        materialization_identity_sha256: sim_types::WorkerSha256Digest::new(
                            "b".repeat(64),
                        )
                        .expect("materialization identity"),
                    },
                    result: Ok(()),
                },
            },
        ));
    }

    #[test]
    fn provider_v2_cancellation_retires_both_request_scopes_before_retry() {
        let (mut shared, _) = shared();
        shared.request_scopes.clear();
        shared.request_order.clear();
        let prompt_id = PromptId(Uuid::new_v4());
        let attempt_id = AttemptId(Uuid::new_v4());

        for _ in 0..2 {
            let execution_request_id = RequestId(Uuid::new_v4());
            let cancellation_request_id = RequestId(Uuid::new_v4());
            shared.register_request(
                execution_request_id,
                RequestScope {
                    prompt_id: Some(prompt_id),
                    attempt_id: Some(attempt_id),
                    kind: RequestKind::ExecuteProviderV2,
                },
            );
            shared.register_request(
                cancellation_request_id,
                RequestScope {
                    prompt_id: Some(prompt_id),
                    attempt_id: Some(attempt_id),
                    kind: RequestKind::Cancel,
                },
            );

            shared.retire_provider_v2_cancel_request(cancellation_request_id);
            shared.retire_provider_v2_request(execution_request_id);

            assert!(shared.request_scopes.is_empty());
            assert!(shared.request_order.is_empty());
            assert_eq!(shared.snapshot.active_prompt_id, None);
            assert_eq!(shared.snapshot.active_attempt_id, None);
        }
    }

    #[test]
    fn lifecycle_events_are_typed_and_bound_to_the_originating_request_kind() {
        let (mut shared, _) = shared();
        let request_id = RequestId(Uuid::new_v4());
        shared.register_request(
            request_id,
            RequestScope {
                prompt_id: None,
                attempt_id: None,
                kind: RequestKind::Cancel,
            },
        );
        let mismatched = envelope(
            &shared,
            request_id,
            0,
            WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::ExecutionStarted,
            },
        );
        assert!(matches!(
            shared.accept(&mismatched),
            Err(RuntimeSupervisorError::Protocol(message))
                if message.contains("does not match request kind")
        ));
        assert!(matches!(
            shared.snapshot.health,
            WorkerHealth::ProtocolIncompatible { .. }
        ));
    }

    #[test]
    fn worker_instance_must_satisfy_every_requested_readiness_row() {
        let (mut shared, request_id) = shared();
        let unrequested_resize = sim_types::WorkerOperationSupport::for_tensor_v2(
            sim_types::WorkerPrimitiveOperationV2::Resize(sim_types::WorkerResizeModeV1::Bilinear),
            sim_types::WorkerTensorRoleV1::Input,
            sim_types::WorkerDType::I64,
            sim_types::WorkerLayout::Contiguous,
        )
        .expect("resize is a tensor primitive");
        let unrequested = sim_types::WorkerBackendCapabilities::new(
            sim_types::DeviceKind::Cpu,
            0,
            vec![unrequested_resize],
            vec![],
        )
        .expect("valid bounded wire declaration");
        let hello = envelope(
            &shared,
            request_id,
            0,
            WorkerMessage::HelloAck {
                accepted_backend: unrequested,
            },
        );
        assert!(matches!(
            shared.accept(&hello),
            Err(RuntimeSupervisorError::Protocol(message))
                if message.contains("does not satisfy the requested readiness matrix")
        ));
        assert!(matches!(
            shared.snapshot.health,
            WorkerHealth::ProtocolIncompatible { .. }
        ));
    }

    #[test]
    fn worker_backend_wire_semantics_are_checked_by_canonical_matrix() {
        let (mut shared, request_id) = shared();
        let unary = sim_types::WorkerOperationSupport::for_tensor_v2(
            sim_types::WorkerPrimitiveOperationV2::Unary(
                sim_types::WorkerUnaryOperationV1::Absolute,
            ),
            sim_types::WorkerTensorRoleV1::Input,
            sim_types::WorkerDType::F32,
            sim_types::WorkerLayout::Contiguous,
        )
        .expect("unary is a tensor primitive");
        let unsupported_resize = sim_types::WorkerOperationSupport::for_tensor_v2(
            sim_types::WorkerPrimitiveOperationV2::Resize(sim_types::WorkerResizeModeV1::Bilinear),
            sim_types::WorkerTensorRoleV1::Input,
            sim_types::WorkerDType::F32,
            sim_types::WorkerLayout::Contiguous,
        )
        .expect("resize is a tensor primitive");
        let invalid = sim_types::WorkerBackendCapabilities::new(
            sim_types::DeviceKind::Cpu,
            0,
            vec![unary],
            vec![unsupported_resize],
        )
        .expect("wire representation is structurally valid");
        let hello = envelope(
            &shared,
            request_id,
            0,
            WorkerMessage::HelloAck {
                accepted_backend: invalid,
            },
        );
        assert!(matches!(
            shared.accept(&hello),
            Err(RuntimeSupervisorError::Protocol(message))
                if message.contains("invalid backend matrix")
        ));
        assert!(matches!(
            shared.snapshot.health,
            WorkerHealth::ProtocolIncompatible { .. }
        ));
    }

    #[test]
    fn three_missed_heartbeats_mark_worker_lost() {
        let (mut shared, request_id) = shared();
        let hello = envelope(
            &shared,
            request_id,
            0,
            WorkerMessage::HelloAck {
                accepted_backend: cpu_backend_wire(),
            },
        );
        shared.accept(&hello).expect("hello");
        let ready = envelope(&shared, request_id, 1, WorkerMessage::Ready);
        shared.accept(&ready).expect("ready");
        let last = shared.last_heartbeat.expect("heartbeat start");
        shared.evaluate_heartbeat(
            last + WORKER_HEARTBEAT_INTERVAL * 3,
            SupervisorPolicy::default(),
        );
        assert_eq!(shared.snapshot.missed_heartbeats, 3);
        assert_eq!(shared.snapshot.health, WorkerHealth::Lost);
    }

    #[test]
    fn launch_records_redact_secrets_and_own_process_tree() {
        let (profile_id, worker_id, _) = identifiers();
        let mut config = WorkerLaunchConfig::new(
            "/private/user/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            1024,
        );
        config.arguments = vec!["--api-key".to_owned(), "secret-value".to_owned()];
        config.environment = vec![("TOKEN".to_owned(), "secret-value".to_owned())];
        let record = launch_record(&config).expect("valid launch record");
        assert_eq!(record.executable, "sim-worker");
        assert_eq!(record.arguments[1], "[redacted]");
        assert_eq!(record.environment_names, ["TOKEN"]);
        assert!(!format!("{record:?}").contains("secret-value"));
    }

    #[test]
    fn general_video_codec_package_launch_is_backend_independent_and_redacted() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeGeneralVideoCodecPackageSettings::from_public_authority(
            "/reviewed/general-video",
            "codec.release",
            &"11".repeat(32),
        )
        .expect("checked general-video public authority");
        let direct = WorkerLaunchConfig::new(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            1024,
        )
        .with_general_video_codec_package(Some(package.clone()));
        let arguments = general_video_codec_package_launch_arguments(
            direct.general_video_codec_package.as_ref(),
        )
        .expect("general-video arguments");
        assert_eq!(
            arguments,
            vec![
                "--video-codec-package-root".to_owned(),
                "/reviewed/general-video".to_owned(),
                "--video-codec-package-signer".to_owned(),
                "codec.release".to_owned(),
                "--video-codec-package-public-key".to_owned(),
                "11".repeat(32),
            ]
        );
        let record = launch_record(&direct).expect("sanitized general-video launch record");
        assert!(record.arguments.contains(&"codec.release".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/general-video"));
        assert!(!format!("{record:?}").contains(&"11".repeat(32)));

        let mut profile =
            NativeRuntimeProfile::disabled_migration_replacement(profile_id.0, "General video")
                .expect("valid profile");
        profile.general_video_codec_package = Some(package);
        let projected = WorkerLaunchConfig::for_packaged_worker_profile(
            &profile,
            worker_id,
            "registry-v1",
            1024,
        )
        .expect("CPU profile accepts general-video authority");
        assert!(projected.general_video_codec_package.is_some());
    }

    #[test]
    fn worker_launch_device_selection_uses_the_canonical_backend_matrix() {
        let (profile_id, worker_id, _) = identifiers();
        for device in DeviceKind::ALL {
            let result = WorkerLaunchConfig::for_device(
                "/package/sim-worker",
                profile_id,
                worker_id,
                "registry-v1",
                device,
                1024,
            );
            if device == DeviceKind::Cpu {
                let config = result.expect("CPU has a canonical capability matrix");
                assert_eq!(config.backend.device(), DeviceId::CPU);
                assert!(!config.backend.supported().is_empty());
            } else {
                assert!(matches!(
                    result,
                    Err(RuntimeSupervisorError::BackendUnavailable(error))
                        if error.device() == device && !error.reason().is_empty()
                ));
            }
        }
    }

    #[test]
    fn rocm_launch_projects_only_checked_public_profile_authority() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeRocmPackageSettings::from_public_authority(
            "/reviewed/rocm-package",
            "rocm.release",
            &"11".repeat(32),
        )
        .expect("checked public package authority");
        let config = WorkerLaunchConfig::for_rocm(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            package,
            2,
            1024,
        )
        .expect("ROCm selection contract");
        assert_eq!(config.backend.device(), DeviceId::new(DeviceKind::Rocm, 2));
        assert_eq!(config.backend_selection.device(), config.backend.device());
        let record = launch_record(&config).expect("sanitized ROCm launch record");
        assert!(record.arguments.contains(&"rocm".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/rocm-package"));
        assert!(!format!("{record:?}").contains(&"11".repeat(32)));
    }

    #[test]
    fn directml_launch_projects_only_checked_public_profile_authority_and_canonical_readiness() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeDirectMlPackageSettings::from_public_authority(
            "/reviewed/directml-package",
            "directml.release",
            &"44".repeat(32),
        )
        .expect("checked public package authority");
        let config = WorkerLaunchConfig::for_directml(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            package,
            4096,
        )
        .expect("DirectML selection contract");
        let device = DeviceId::new(DeviceKind::DirectMl, 0);
        assert_eq!(config.backend.device(), device);
        assert_eq!(
            config.backend,
            BackendCapabilityMatrix::worker_readiness_requirements(device)
                .expect("canonical worker readiness requirements")
        );
        assert_eq!(config.backend.supported().len(), 7);
        assert_eq!(config.backend.deterministic().len(), 2);
        assert_eq!(config.backend_selection.device(), config.backend.device());
        let arguments = config
            .backend_selection
            .launch_arguments()
            .expect("checked DirectML launch arguments");
        assert_eq!(
            arguments,
            [
                "--backend",
                "directml",
                "--directml-package-root",
                "/reviewed/directml-package",
                "--directml-package-signer",
                "directml.release",
                "--directml-package-public-key",
                &"44".repeat(32),
            ]
        );
        assert!(!arguments.contains(&"--backend-device-ordinal".to_owned()));
        let record = launch_record(&config).expect("sanitized DirectML launch record");
        assert!(record.arguments.contains(&"directml".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/directml-package"));
        assert!(!format!("{record:?}").contains(&"44".repeat(32)));
    }

    #[test]
    fn packaged_directml_profile_maps_to_the_fixed_worker_device() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeDirectMlPackageSettings::from_public_authority(
            "/reviewed/directml-package",
            "directml.release",
            &"44".repeat(32),
        )
        .expect("checked public package authority");
        let profile = NativeRuntimeProfile {
            id: profile_id.0,
            name: "DirectML".to_owned(),
            model_roots: Vec::new(),
            device: DeviceKind::DirectMl,
            memory_policy: crate::MemoryPolicy::Balanced,
            api_host: crate::NativeApiHostPolicy::default(),
            plugin_policy: crate::PluginPolicy::Disabled,
            rocm_package: None,
            metal_package: None,
            mlu_package: None,
            npu_package: None,
            cuda_package: None,
            xpu_package: None,
            directml_package: Some(package),
            general_video_codec_package: None,
            provider_scope: "local".to_owned(),
            compatibility_version: crate::CURRENT_NATIVE_PROFILE_VERSION,
            unknown_fields: BTreeMap::new(),
        };
        let config = WorkerLaunchConfig::for_packaged_worker_profile(
            &profile,
            worker_id,
            "registry-v1",
            4096,
        )
        .expect("DirectML profile maps to the packaged worker");
        assert_eq!(
            config.backend_selection.device(),
            DeviceId::new(DeviceKind::DirectMl, 0)
        );
        assert!(matches!(
            config.backend_selection,
            WorkerBackendSelection::DirectMl { .. }
        ));
    }

    #[test]
    fn packaged_cuda_profile_maps_to_the_exact_worker_device_and_public_authority() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeCudaPackageSettings::from_public_authority(
            "/reviewed/cuda-package",
            "cuda.release",
            &"56".repeat(32),
        )
        .expect("checked public package authority");
        let profile = NativeRuntimeProfile {
            id: profile_id.0,
            name: "CUDA".to_owned(),
            model_roots: Vec::new(),
            device: DeviceKind::Cuda,
            memory_policy: crate::MemoryPolicy::Balanced,
            api_host: crate::NativeApiHostPolicy::default(),
            plugin_policy: crate::PluginPolicy::Disabled,
            rocm_package: None,
            metal_package: None,
            mlu_package: None,
            npu_package: None,
            cuda_package: Some(package),
            xpu_package: None,
            directml_package: None,
            general_video_codec_package: None,
            provider_scope: "local".to_owned(),
            compatibility_version: crate::CURRENT_NATIVE_PROFILE_VERSION,
            unknown_fields: BTreeMap::new(),
        };
        let config = WorkerLaunchConfig::for_packaged_worker_profile(
            &profile,
            worker_id,
            "registry-v1",
            4096,
        )
        .expect("CUDA profile maps to the packaged worker");
        let device = DeviceId::new(DeviceKind::Cuda, 0);
        assert_eq!(config.backend_selection.device(), device);
        assert_eq!(
            config.backend,
            BackendCapabilityMatrix::worker_readiness_requirements(device)
                .expect("canonical worker readiness requirements")
        );
        assert!(matches!(
            config.backend_selection,
            WorkerBackendSelection::Cuda { .. }
        ));
        let arguments = config
            .backend_selection
            .launch_arguments()
            .expect("checked CUDA launch arguments");
        assert_eq!(
            arguments,
            [
                "--backend",
                "cuda",
                "--backend-device-ordinal",
                "0",
                "--cuda-package-root",
                "/reviewed/cuda-package",
                "--cuda-package-signer",
                "cuda.release",
                "--cuda-package-public-key",
                &"56".repeat(32),
            ]
        );
        let record = launch_record(&config).expect("sanitized CUDA launch record");
        assert!(record.arguments.contains(&"cuda".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/cuda-package"));
        assert!(!format!("{record:?}").contains(&"56".repeat(32)));
    }

    #[test]
    fn packaged_npu_profile_maps_to_the_exact_worker_device_and_authority() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeNpuPackageSettings::from_public_authority(
            "/reviewed/npu-package",
            "npu.release",
            &"35".repeat(32),
        )
        .expect("checked public package authority");
        let profile = NativeRuntimeProfile {
            id: profile_id.0,
            name: "NPU".to_owned(),
            model_roots: Vec::new(),
            device: DeviceKind::Npu,
            memory_policy: crate::MemoryPolicy::Balanced,
            api_host: crate::NativeApiHostPolicy::default(),
            plugin_policy: crate::PluginPolicy::Disabled,
            rocm_package: None,
            metal_package: None,
            mlu_package: None,
            npu_package: Some(package),
            cuda_package: None,
            xpu_package: None,
            directml_package: None,
            general_video_codec_package: None,
            provider_scope: "local".to_owned(),
            compatibility_version: crate::CURRENT_NATIVE_PROFILE_VERSION,
            unknown_fields: BTreeMap::new(),
        };
        let config = WorkerLaunchConfig::for_packaged_worker_profile(
            &profile,
            worker_id,
            "registry-v1",
            4096,
        )
        .expect("NPU profile maps to the packaged worker");
        assert_eq!(
            config.backend_selection.device(),
            DeviceId::new(DeviceKind::Npu, 0)
        );
        assert!(matches!(
            config.backend_selection,
            WorkerBackendSelection::Npu { .. }
        ));
    }

    #[test]
    fn packaged_xpu_profile_maps_to_the_exact_worker_device_and_public_authority() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeXpuPackageSettings::from_public_authority(
            "/reviewed/xpu-package",
            "xpu.release",
            &"46".repeat(32),
        )
        .expect("checked public package authority");
        let profile = NativeRuntimeProfile {
            id: profile_id.0,
            name: "XPU".to_owned(),
            model_roots: Vec::new(),
            device: DeviceKind::Xpu,
            memory_policy: crate::MemoryPolicy::Balanced,
            api_host: crate::NativeApiHostPolicy::default(),
            plugin_policy: crate::PluginPolicy::Disabled,
            rocm_package: None,
            metal_package: None,
            mlu_package: None,
            npu_package: None,
            cuda_package: None,
            xpu_package: Some(package),
            directml_package: None,
            general_video_codec_package: None,
            provider_scope: "local".to_owned(),
            compatibility_version: crate::CURRENT_NATIVE_PROFILE_VERSION,
            unknown_fields: BTreeMap::new(),
        };
        let config = WorkerLaunchConfig::for_packaged_worker_profile(
            &profile,
            worker_id,
            "registry-v1",
            4096,
        )
        .expect("XPU profile maps to the packaged worker");
        let device = DeviceId::new(DeviceKind::Xpu, 0);
        assert_eq!(config.backend_selection.device(), device);
        assert_eq!(
            config.backend,
            BackendCapabilityMatrix::worker_readiness_requirements(device)
                .expect("canonical worker readiness requirements")
        );
        let arguments = config
            .backend_selection
            .launch_arguments()
            .expect("checked XPU launch arguments");
        assert_eq!(
            arguments,
            [
                "--backend",
                "xpu",
                "--backend-device-ordinal",
                "0",
                "--xpu-package-root",
                "/reviewed/xpu-package",
                "--xpu-package-signer",
                "xpu.release",
                "--xpu-package-public-key",
                &"46".repeat(32),
            ]
        );
        let record = launch_record(&config).expect("sanitized XPU launch record");
        assert!(record.arguments.contains(&"xpu".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/xpu-package"));
        assert!(!format!("{record:?}").contains(&"46".repeat(32)));
    }

    #[test]
    fn metal_launch_projects_only_checked_public_profile_authority_and_canonical_readiness() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeMetalPackageSettings::from_public_authority(
            "/reviewed/metal-package",
            "metal.release",
            &"22".repeat(32),
        )
        .expect("checked public package authority");
        let config = WorkerLaunchConfig::for_metal(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            package,
            1024,
        )
        .expect("Metal selection contract");
        assert_eq!(config.backend.device(), DeviceId::new(DeviceKind::Metal, 0));
        assert_eq!(config.backend.supported().len(), 7);
        assert_eq!(config.backend.deterministic().len(), 2);
        assert!(config.backend.device_properties().is_none());
        assert_eq!(config.backend_selection.device(), config.backend.device());
        let record = launch_record(&config).expect("sanitized Metal launch record");
        assert!(record.arguments.contains(&"metal".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/metal-package"));
        assert!(!format!("{record:?}").contains(&"22".repeat(32)));
    }

    #[test]
    fn mlu_launch_projects_only_checked_public_profile_authority_and_canonical_readiness() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeMluPackageSettings::from_public_authority(
            "/reviewed/mlu-package",
            "mlu.release",
            &"33".repeat(32),
        )
        .expect("checked public package authority");
        let config = WorkerLaunchConfig::for_mlu(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            package,
            3,
            4096,
        )
        .expect("MLU selection contract");
        assert_eq!(config.backend.device(), DeviceId::new(DeviceKind::Mlu, 3));
        assert_eq!(config.backend.supported().len(), 7);
        assert_eq!(config.backend.deterministic().len(), 2);
        assert!(config.backend.device_properties().is_none());
        assert_eq!(config.backend_selection.device(), config.backend.device());
        let record = launch_record(&config).expect("sanitized MLU launch record");
        assert!(record.arguments.contains(&"mlu".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/mlu-package"));
        assert!(!format!("{record:?}").contains(&"33".repeat(32)));
    }

    #[test]
    fn npu_launch_projects_only_checked_public_profile_authority_and_canonical_readiness() {
        let (profile_id, worker_id, _) = identifiers();
        let package = NativeNpuPackageSettings::from_public_authority(
            "/reviewed/npu-package",
            "npu.release",
            &"35".repeat(32),
        )
        .expect("checked public package authority");
        let config = WorkerLaunchConfig::for_npu(
            "/package/sim-worker",
            profile_id,
            worker_id,
            "registry-v1",
            package,
            2,
            4096,
        )
        .expect("NPU selection contract");
        assert_eq!(config.backend.device(), DeviceId::new(DeviceKind::Npu, 2));
        assert_eq!(config.backend.supported().len(), 7);
        assert_eq!(config.backend.deterministic().len(), 2);
        assert!(config.backend.device_properties().is_none());
        assert_eq!(config.backend_selection.device(), config.backend.device());
        let record = launch_record(&config).expect("sanitized NPU launch record");
        assert!(record.arguments.contains(&"npu".to_owned()));
        assert!(!format!("{record:?}").contains("/reviewed/npu-package"));
        assert!(!format!("{record:?}").contains(&"35".repeat(32)));
    }

    #[test]
    fn packaged_worker_path_is_owned_once_and_strictly_sibling_relative() {
        let executable = Path::new("/Applications/Zed.app/Contents/MacOS/zed");
        let expected_name = if cfg!(windows) {
            "sim-worker.exe"
        } else {
            "sim-worker"
        };
        assert_eq!(
            packaged_worker_binary_for_executable(executable),
            Ok(executable
                .parent()
                .expect("fixture executable has a parent")
                .join(expected_name))
        );
    }

    #[test]
    fn captured_logs_are_control_sanitized_and_bounded() {
        let mut log = BoundedLog::new();
        log.push(b"visible\x00line\n");
        assert_eq!(log.entries(), ["visibleline\n"]);
        for _ in 0..300 {
            log.push(&vec![b'x'; 4096]);
        }
        assert!(log.bytes <= MAX_CAPTURED_WORKER_LOG_BYTES);
    }

    #[test]
    fn production_policy_uses_normative_timing() {
        let policy = SupervisorPolicy::default();
        assert_eq!(policy.heartbeat_interval, Duration::from_secs(2));
        assert_eq!(policy.missed_heartbeat_limit, 3);
        assert_eq!(policy.shutdown_timeout, Duration::from_secs(5));
        assert_eq!(policy.maximum_automatic_restarts, 1);
    }
}
