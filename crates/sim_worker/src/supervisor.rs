use std::{collections::BTreeMap, sync::Arc};

use sha2::{Digest, Sha256};
use sim_tensor::{
    BackendCapabilityMatrix, BackendMemorySnapshot, BackendWorkspaceAuthority, BinaryOperation,
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, ScratchReservation, StreamId,
    TensorBackend, TensorDescriptor, TensorError,
};
use sim_types::{
    AttemptId, BackendUnavailable, MAX_WORKER_COMPONENT_CHUNK_BYTES, ProfileId, PromptId,
    RequestId, WORKER_PROTOCOL_VERSION, WorkerComponentContent, WorkerComponentDescriptor,
    WorkerEnvelope, WorkerId, WorkerLifecycleEvent, WorkerMessage, WorkerOutputProposal,
    WorkerPluginExecutionOutcome, WorkerRegistryDeploymentAck, WorkerRegistryDeploymentBegin,
    WorkerRegistryDeploymentChunk, WorkerRegistryDeploymentCommit,
    WorkerRegistryDeploymentRejection, WorkerRegistryDeploymentRejectionReason,
    WorkerRegistryGeneration, WorkerSha256Digest,
};
use thiserror::Error;

use crate::{PlannedWorkspaceAuthorization, validate_worker_payload};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerLifecycle {
    Booting,
    Ready,
    Running,
    Cancelling,
    Stopping,
    Stopped,
    Fatal,
}

pub use sim_types::WorkerLifecycleEvent as PrivateWorkerEvent;

pub struct WorkerBackendSession {
    backend: Arc<dyn TensorBackend>,
    workspace_authority: BackendWorkspaceAuthority,
}

impl WorkerBackendSession {
    pub fn new(
        backend: Arc<dyn TensorBackend>,
        workspace_authority: BackendWorkspaceAuthority,
    ) -> Result<Self, TensorError> {
        let session = Self {
            backend,
            workspace_authority,
        };
        session.run_readiness_probe()?;
        Ok(session)
    }

    pub fn cpu(memory_limit_bytes: u64) -> Result<(Self, Arc<CpuBackend>), WorkerSessionError> {
        let (backend, workspace_authority) =
            BackendWorkspaceAuthority::create_backend(memory_limit_bytes)
                .map_err(|error| WorkerSessionError::BackendInitialization(error.to_string()))?;
        let backend = Arc::new(backend);
        let session = Self::new(backend.clone(), workspace_authority)
            .map_err(|error| WorkerSessionError::BackendInitialization(error.to_string()))?;
        Ok((session, backend))
    }

    pub fn backend(&self) -> Arc<dyn TensorBackend> {
        self.backend.clone()
    }

    pub fn memory_snapshot(&self) -> BackendMemorySnapshot {
        self.workspace_authority.memory_snapshot()
    }

    fn run_readiness_probe(&self) -> Result<(), TensorError> {
        let baseline = self.workspace_authority.memory_snapshot().current_bytes;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: self.workspace_authority.authorize_workspace(8)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let workspace = self.backend.reserve_workspace(&context, 8)?;
        drop(workspace);
        let descriptor = TensorDescriptor::contiguous(
            vec![1],
            DType::F32,
            self.backend.device(),
            StreamId::DEFAULT,
        )?;
        let (source, allocation_event) = self.backend.allocate(descriptor.clone(), &context)?;
        self.backend.wait_event(allocation_event, &context)?;
        let (destination, transfer_event) = self.backend.copy(&source, descriptor, &context)?;
        self.backend.wait_event(transfer_event, &context)?;
        let output_descriptor = destination.descriptor().clone();
        let (output, kernel_event) = self.backend.binary(
            BinaryOperation::Add,
            &source,
            &destination,
            output_descriptor,
            &context,
        )?;
        self.backend.wait_event(kernel_event, &context)?;
        let recorded_event = self.backend.record_event(&context)?;
        self.backend.wait_event(recorded_event, &context)?;
        drop(output);
        drop(destination);
        drop(source);
        let snapshot = self.workspace_authority.memory_snapshot();
        if snapshot.current_bytes != baseline {
            return Err(TensorError::Faulted {
                reason: "backend readiness probe retained device allocations".to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExecutionScope {
    prompt_id: PromptId,
    attempt_id: AttemptId,
    request_id: RequestId,
    kind: ExecutionKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutionKind {
    Native,
    Plugin,
    ProviderV2Plugin,
}

#[derive(Clone, Debug)]
struct SessionIdentity {
    profile_id: ProfileId,
    worker_id: WorkerId,
    request_id: RequestId,
    prompt_id: Option<PromptId>,
    attempt_id: Option<AttemptId>,
    registry_version: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssembledWorkerComponent {
    extension_id: String,
    extension_version: String,
    plugin_identifier: String,
    plugin_version: String,
    authorization_generation: WorkerSha256Digest,
    manifest_digest_sha256: WorkerSha256Digest,
    component_digest_sha256: WorkerSha256Digest,
    manifest_bytes: Vec<u8>,
    authorization_bytes: Vec<u8>,
    component_bytes: Vec<u8>,
}

impl AssembledWorkerComponent {
    pub fn extension_id(&self) -> &str {
        &self.extension_id
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub fn plugin_identifier(&self) -> &str {
        &self.plugin_identifier
    }

    pub fn plugin_version(&self) -> &str {
        &self.plugin_version
    }

    pub fn authorization_generation(&self) -> &WorkerSha256Digest {
        &self.authorization_generation
    }

    pub fn manifest_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.manifest_digest_sha256
    }

    pub fn component_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.component_digest_sha256
    }

    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }

    pub fn component_bytes(&self) -> &[u8] {
        &self.component_bytes
    }

    pub fn authorization_bytes(&self) -> &[u8] {
        &self.authorization_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssembledWorkerRegistry {
    generation: WorkerRegistryGeneration,
    registry_digest_sha256: WorkerSha256Digest,
    components: Vec<AssembledWorkerComponent>,
}

impl AssembledWorkerRegistry {
    pub const fn generation(&self) -> WorkerRegistryGeneration {
        self.generation
    }

    pub fn registry_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.registry_digest_sha256
    }

    pub fn components(&self) -> &[AssembledWorkerComponent] {
        &self.components
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test(
        generation: WorkerRegistryGeneration,
        registry_digest_sha256: WorkerSha256Digest,
    ) -> Self {
        Self {
            generation,
            registry_digest_sha256,
            components: Vec::new(),
        }
    }
}

struct PendingWorkerRegistry {
    begin: WorkerRegistryDeploymentBegin,
    component_index: usize,
    content: WorkerComponentContent,
    chunk_index: u32,
    manifest_bytes: Vec<u8>,
    authorization_bytes: Vec<u8>,
    component_bytes: Vec<u8>,
    assembled: Vec<AssembledWorkerComponent>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerSessionError {
    #[error("worker backend initialization failed: {0}")]
    BackendInitialization(String),
    #[error("worker received protocol version {actual}, expected {expected}")]
    ProtocolVersion { expected: u16, actual: u16 },
    #[error("worker expected input sequence {expected}, received {actual}")]
    Sequence { expected: u64, actual: u64 },
    #[error("worker profile, worker, or registry identity changed during the session")]
    IdentityChanged,
    #[error("worker expected {expected} while in {actual:?}")]
    InvalidState {
        expected: &'static str,
        actual: WorkerLifecycle,
    },
    #[error("worker and supervisor have no common backend capability")]
    CapabilityMismatch,
    #[error("execute and cancellation messages require prompt and attempt identities")]
    MissingAttemptIdentity,
    #[error("attempt identity does not match the running execution")]
    StaleAttempt,
    #[error("execute plan is empty")]
    EmptyPlan,
    #[error("worker output sequence overflowed")]
    OutputSequenceOverflow,
    #[error("worker payload is invalid: {0}")]
    Payload(String),
    #[error("message direction is invalid for the worker")]
    InvalidDirection,
    #[error("worker envelope contains unsupported opaque extensions")]
    OpaqueExtensions,
    #[error("a worker registry deployment is already pending")]
    DeploymentAlreadyPending,
    #[error("worker registry generation {actual} is not newer than {current}")]
    StaleRegistryGeneration { current: u64, actual: u64 },
    #[error("worker registry deployment digest does not match its descriptors")]
    RegistryDigestMismatch,
    #[error("worker registry deployment has not begun")]
    DeploymentNotBegun,
    #[error("worker registry deployment generation does not match the pending generation")]
    DeploymentGenerationMismatch,
    #[error("worker registry deployment chunk targeted component {actual}, expected {expected}")]
    UnexpectedComponentIndex { expected: u32, actual: u32 },
    #[error("worker registry deployment chunk targeted {actual:?}, expected {expected:?}")]
    UnexpectedComponentContent {
        expected: WorkerComponentContent,
        actual: WorkerComponentContent,
    },
    #[error("worker registry deployment chunk index {actual} does not match {expected}")]
    UnexpectedChunkIndex { expected: u32, actual: u32 },
    #[error("worker registry deployment chunk contains {actual} bytes, expected {expected}")]
    UnexpectedChunkLength { expected: usize, actual: usize },
    #[error("worker registry deployment {0:?} digest does not match its content address")]
    ContentDigestMismatch(WorkerComponentContent),
    #[error("worker registry deployment commit arrived before all content")]
    DeploymentIncomplete,
    #[error("worker registry deployment allocation failed")]
    DeploymentAllocation,
    #[error("worker registry deployment component count overflowed")]
    DeploymentComponentCountOverflow,
    #[error("worker registry deployment is pending while execution was requested")]
    DeploymentPending,
    #[error("worker plugin execution requires a deployed component registry")]
    MissingRegistry,
    #[error("worker registry deployment acknowledgement is invalid: {0}")]
    InvalidDeploymentAcknowledgement(String),
    #[error("worker registry deployment requires component verification before commit")]
    RegistryVerificationRequired,
}

pub struct WorkerSession {
    lifecycle: WorkerLifecycle,
    identity: Option<SessionIdentity>,
    accepted_backend: Option<BackendCapabilityMatrix>,
    last_input_sequence: Option<u64>,
    next_output_sequence: u64,
    execution: Option<ExecutionScope>,
    provider_v2_proposal_pending: bool,
    last_provider_v2_finalization: Option<(
        ExecutionScope,
        sim_types::WorkerProviderV2ProposalFinalization,
    )>,
    pending_registry: Option<PendingWorkerRegistry>,
    registry: Option<AssembledWorkerRegistry>,
    backend_session: Option<WorkerBackendSession>,
    backend_unavailable: Option<BackendUnavailable>,
}

impl WorkerSession {
    pub fn new(memory_limit_bytes: u64) -> Result<Self, WorkerSessionError> {
        let (backend_session, _) = WorkerBackendSession::cpu(memory_limit_bytes)?;
        Ok(Self::with_backend_result(Ok(backend_session)))
    }

    pub fn with_backend_result(backend: Result<WorkerBackendSession, BackendUnavailable>) -> Self {
        let (backend_session, backend_unavailable) = match backend {
            Ok(backend) => (Some(backend), None),
            Err(error) => (None, Some(error)),
        };
        Self {
            lifecycle: WorkerLifecycle::Booting,
            identity: None,
            accepted_backend: None,
            last_input_sequence: None,
            next_output_sequence: 0,
            execution: None,
            provider_v2_proposal_pending: false,
            last_provider_v2_finalization: None,
            pending_registry: None,
            registry: None,
            backend_session,
            backend_unavailable,
        }
    }

    pub fn lifecycle(&self) -> WorkerLifecycle {
        self.lifecycle
    }

    pub fn accepted_backend(&self) -> Option<&BackendCapabilityMatrix> {
        self.accepted_backend.as_ref()
    }

    pub fn memory_snapshot(&self) -> Result<BackendMemorySnapshot, WorkerSessionError> {
        self.backend_session
            .as_ref()
            .map(WorkerBackendSession::memory_snapshot)
            .ok_or_else(|| {
                WorkerSessionError::BackendInitialization(
                    "selected backend session is unavailable".to_owned(),
                )
            })
    }

    pub fn backend(&self) -> Option<Arc<dyn TensorBackend>> {
        self.backend_session
            .as_ref()
            .map(WorkerBackendSession::backend)
    }

    pub fn backend_device(&self) -> Option<DeviceId> {
        self.backend_session
            .as_ref()
            .map(|session| session.backend.device())
    }

    pub fn authorize_planned_workspace(
        &self,
        planned: PlannedWorkspaceAuthorization,
    ) -> Result<ScratchReservation, TensorError> {
        self.backend_session
            .as_ref()
            .ok_or_else(|| TensorError::Faulted {
                reason: "selected backend session is unavailable".to_owned(),
            })?
            .workspace_authority
            .authorize_workspace(planned.bytes())
    }

    pub fn registry(&self) -> Option<&AssembledWorkerRegistry> {
        self.registry.as_ref()
    }

    pub fn heartbeat_enabled(&self) -> bool {
        matches!(
            self.lifecycle,
            WorkerLifecycle::Ready | WorkerLifecycle::Running | WorkerLifecycle::Cancelling
        )
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.lifecycle,
            WorkerLifecycle::Stopped | WorkerLifecycle::Fatal
        )
    }

    pub fn handle(
        &mut self,
        envelope: WorkerEnvelope,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        self.validate_input(&envelope)?;
        if let WorkerMessage::ProviderV2ProposalFinalization { finalization } = &envelope.message
            && self.lifecycle == WorkerLifecycle::Ready
            && self
                .last_provider_v2_finalization
                .as_ref()
                .is_some_and(|(scope, completed)| {
                    completed == finalization
                        && envelope.request_id == scope.request_id
                        && envelope.prompt_id == Some(scope.prompt_id)
                        && envelope.attempt_id == Some(scope.attempt_id)
                })
        {
            return Ok(vec![self.response_to(
                &envelope,
                WorkerMessage::ProviderV2ProposalFinalizationAck {
                    acknowledgement: sim_types::WorkerProviderV2ProposalFinalizationAck {
                        finalization: finalization.clone(),
                        result: Err(sim_types::WorkerProviderStreamError::InvalidOrder),
                    },
                },
            )?]);
        }
        let messages = match &envelope.message {
            WorkerMessage::Hello { backend } => self.handle_hello(&envelope, backend)?,
            WorkerMessage::RegistryDeploymentBegin { deployment } => {
                self.handle_registry_begin(&envelope, deployment)?
            }
            WorkerMessage::RegistryDeploymentChunk { chunk } => {
                self.handle_registry_chunk(chunk)?
            }
            WorkerMessage::RegistryDeploymentCommit { .. } => {
                return Err(WorkerSessionError::RegistryVerificationRequired);
            }
            WorkerMessage::Execute { plan } => self.handle_execute(&envelope, plan)?,
            WorkerMessage::ExecutePlugin { invocation } => {
                self.handle_execute_plugin(&envelope, invocation)?
            }
            WorkerMessage::PluginCapabilityResponse { .. } => {
                self.handle_plugin_capability_response(&envelope)?
            }
            WorkerMessage::ProviderStreamResponse { .. }
            | WorkerMessage::ProviderV2ProposalFinalization { .. } => {
                self.handle_provider_v2_response(&envelope)?
            }
            WorkerMessage::ModelSourceResponse { .. } => {
                self.handle_model_source_response(&envelope)?
            }
            WorkerMessage::Cancel { reason } => self.handle_cancel(&envelope, reason)?,
            WorkerMessage::Heartbeat => {
                vec![self.response_to(&envelope, WorkerMessage::Heartbeat)?]
            }
            WorkerMessage::Shutdown => self.handle_shutdown(&envelope)?,
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
            | WorkerMessage::Fatal { .. } => return Err(WorkerSessionError::InvalidDirection),
        };
        Ok(messages)
    }

    pub fn heartbeat(&mut self) -> Result<Option<WorkerEnvelope>, WorkerSessionError> {
        if !self.heartbeat_enabled() {
            return Ok(None);
        }
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        Ok(Some(self.envelope_from_identity(
            identity,
            WorkerMessage::Heartbeat,
        )?))
    }

    pub fn fatal_for(
        &mut self,
        envelope: &WorkerEnvelope,
        error: &WorkerSessionError,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        self.lifecycle = WorkerLifecycle::Fatal;
        self.response_to(
            envelope,
            WorkerMessage::Fatal {
                code: "worker_protocol_error".to_owned(),
                message: error.to_string(),
            },
        )
    }

    fn validate_input(&mut self, envelope: &WorkerEnvelope) -> Result<(), WorkerSessionError> {
        if envelope.version != WORKER_PROTOCOL_VERSION {
            return Err(WorkerSessionError::ProtocolVersion {
                expected: WORKER_PROTOCOL_VERSION,
                actual: envelope.version,
            });
        }
        if !envelope.extensions.is_empty() {
            return Err(WorkerSessionError::OpaqueExtensions);
        }
        validate_worker_payload(envelope)
            .map_err(|error| WorkerSessionError::Payload(error.to_string()))?;
        let expected = self
            .last_input_sequence
            .map_or(0, |sequence| sequence.saturating_add(1));
        if envelope.sequence != expected {
            return Err(WorkerSessionError::Sequence {
                expected,
                actual: envelope.sequence,
            });
        }
        if let Some(identity) = &self.identity
            && (identity.profile_id != envelope.profile_id
                || identity.worker_id != envelope.worker_id
                || identity.registry_version != envelope.registry_version)
        {
            return Err(WorkerSessionError::IdentityChanged);
        }
        self.last_input_sequence = Some(envelope.sequence);
        if let Some(identity) = &mut self.identity {
            identity.request_id = envelope.request_id;
            identity.prompt_id = envelope.prompt_id;
            identity.attempt_id = envelope.attempt_id;
        }
        Ok(())
    }

    fn handle_hello(
        &mut self,
        envelope: &WorkerEnvelope,
        backend: &sim_types::WorkerBackendCapabilities,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        if self.lifecycle != WorkerLifecycle::Booting || self.identity.is_some() {
            return Err(WorkerSessionError::InvalidState {
                expected: "one initial hello",
                actual: self.lifecycle,
            });
        }
        self.identity = Some(SessionIdentity {
            profile_id: envelope.profile_id,
            worker_id: envelope.worker_id,
            request_id: envelope.request_id,
            prompt_id: envelope.prompt_id,
            attempt_id: envelope.attempt_id,
            registry_version: envelope.registry_version.clone(),
        });
        if let Some(unavailable) = self.backend_unavailable.take() {
            self.lifecycle = WorkerLifecycle::Fatal;
            let message = serde_json::to_string(&unavailable)
                .map_err(|error| WorkerSessionError::BackendInitialization(error.to_string()))?;
            return Ok(vec![self.response_to(
                envelope,
                WorkerMessage::Fatal {
                    code: "backend_unavailable".to_owned(),
                    message,
                },
            )?]);
        }
        let requested = BackendCapabilityMatrix::try_from(backend.clone())
            .map_err(|_| WorkerSessionError::CapabilityMismatch)?;
        let available = self
            .backend_session
            .as_ref()
            .ok_or_else(|| {
                WorkerSessionError::BackendInitialization(
                    "selected backend session disappeared".to_owned(),
                )
            })?
            .backend
            .capabilities();
        if !requested.is_subset_of(available) {
            return Err(WorkerSessionError::CapabilityMismatch);
        }
        let accepted_backend = available.clone();
        let accepted_wire = accepted_backend
            .to_worker_capabilities()
            .map_err(|error| WorkerSessionError::BackendInitialization(error.to_string()))?;
        self.accepted_backend = Some(accepted_backend);
        self.lifecycle = WorkerLifecycle::Ready;
        Ok(vec![
            self.response_to(
                envelope,
                WorkerMessage::HelloAck {
                    accepted_backend: accepted_wire,
                },
            )?,
            self.response_to(envelope, WorkerMessage::Ready)?,
        ])
    }

    fn handle_registry_begin(
        &mut self,
        _envelope: &WorkerEnvelope,
        deployment: &WorkerRegistryDeploymentBegin,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        if self.lifecycle != WorkerLifecycle::Ready {
            return Err(WorkerSessionError::InvalidState {
                expected: "ready",
                actual: self.lifecycle,
            });
        }
        if self.pending_registry.is_some() {
            return Err(WorkerSessionError::DeploymentAlreadyPending);
        }
        if let Some(current) = &self.registry
            && deployment.generation() <= current.generation()
        {
            return Err(WorkerSessionError::StaleRegistryGeneration {
                current: current.generation().get(),
                actual: deployment.generation().get(),
            });
        }
        verify_digest(
            deployment.registry_digest_sha256(),
            &deployment.digest_material(),
            WorkerComponentContent::Manifest,
        )
        .map_err(|_| WorkerSessionError::RegistryDigestMismatch)?;
        self.pending_registry = Some(PendingWorkerRegistry::new(deployment.clone())?);
        Ok(Vec::new())
    }

    fn handle_registry_chunk(
        &mut self,
        chunk: &WorkerRegistryDeploymentChunk,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        if self.lifecycle != WorkerLifecycle::Ready {
            return Err(WorkerSessionError::InvalidState {
                expected: "ready",
                actual: self.lifecycle,
            });
        }
        self.pending_registry
            .as_mut()
            .ok_or(WorkerSessionError::DeploymentNotBegun)?
            .apply_chunk(chunk)?;
        Ok(Vec::new())
    }

    pub(crate) fn handle_verified_registry_commit<F>(
        &mut self,
        envelope: WorkerEnvelope,
        verify: F,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError>
    where
        F: FnOnce(&AssembledWorkerRegistry) -> Result<(), WorkerRegistryDeploymentRejectionReason>,
    {
        self.validate_input(&envelope)?;
        let WorkerMessage::RegistryDeploymentCommit { commit } = &envelope.message else {
            return Err(WorkerSessionError::InvalidDirection);
        };
        self.handle_registry_commit(&envelope, commit, verify)
    }

    fn handle_registry_commit<F>(
        &mut self,
        envelope: &WorkerEnvelope,
        commit: &WorkerRegistryDeploymentCommit,
        verify: F,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError>
    where
        F: FnOnce(&AssembledWorkerRegistry) -> Result<(), WorkerRegistryDeploymentRejectionReason>,
    {
        if self.lifecycle != WorkerLifecycle::Ready {
            return Err(WorkerSessionError::InvalidState {
                expected: "ready",
                actual: self.lifecycle,
            });
        }
        let pending = self
            .pending_registry
            .take()
            .ok_or(WorkerSessionError::DeploymentNotBegun)?;
        if commit.generation() != pending.begin.generation() {
            self.pending_registry = Some(pending);
            return Err(WorkerSessionError::DeploymentGenerationMismatch);
        }
        if commit.registry_digest_sha256() != pending.begin.registry_digest_sha256() {
            self.pending_registry = Some(pending);
            return Err(WorkerSessionError::RegistryDigestMismatch);
        }
        if !pending.is_complete() {
            self.pending_registry = Some(pending);
            return Err(WorkerSessionError::DeploymentIncomplete);
        }
        let registry = pending.finish();
        if let Err(reason) = verify(&registry) {
            let rejection = WorkerRegistryDeploymentRejection::new(
                registry.generation,
                registry.registry_digest_sha256,
                reason,
            );
            return Ok(vec![self.response_to(
                envelope,
                WorkerMessage::RegistryDeploymentRejected { rejection },
            )?]);
        }
        let component_count = u32::try_from(registry.components.len())
            .map_err(|_| WorkerSessionError::DeploymentComponentCountOverflow)?;
        let acknowledgement = WorkerRegistryDeploymentAck::new(
            registry.generation,
            registry.registry_digest_sha256.clone(),
            component_count,
        )
        .map_err(|error| WorkerSessionError::InvalidDeploymentAcknowledgement(error.to_string()))?;
        self.registry = Some(registry);
        Ok(vec![self.response_to(
            envelope,
            WorkerMessage::RegistryDeploymentAck { acknowledgement },
        )?])
    }

    fn handle_execute(
        &mut self,
        envelope: &WorkerEnvelope,
        plan: &[u8],
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        if self.lifecycle != WorkerLifecycle::Ready {
            return Err(WorkerSessionError::InvalidState {
                expected: "ready",
                actual: self.lifecycle,
            });
        }
        if self.pending_registry.is_some() {
            return Err(WorkerSessionError::DeploymentPending);
        }
        if plan.is_empty() {
            return Err(WorkerSessionError::EmptyPlan);
        }
        let (prompt_id, attempt_id) = attempt_identity(envelope)?;
        self.execution = Some(ExecutionScope {
            prompt_id,
            attempt_id,
            request_id: envelope.request_id,
            kind: ExecutionKind::Native,
        });
        self.provider_v2_proposal_pending = false;
        self.last_provider_v2_finalization = None;
        self.lifecycle = WorkerLifecycle::Running;
        Ok(vec![self.event_response(
            envelope,
            WorkerLifecycleEvent::ExecutionStarted,
        )?])
    }

    fn handle_execute_plugin(
        &mut self,
        envelope: &WorkerEnvelope,
        invocation: &[u8],
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        if self.lifecycle != WorkerLifecycle::Ready {
            return Err(WorkerSessionError::InvalidState {
                expected: "ready",
                actual: self.lifecycle,
            });
        }
        if self.pending_registry.is_some() {
            return Err(WorkerSessionError::DeploymentPending);
        }
        if self.registry.is_none() {
            return Err(WorkerSessionError::MissingRegistry);
        }
        if invocation.is_empty() {
            return Err(WorkerSessionError::EmptyPlan);
        }
        let (prompt_id, attempt_id) = attempt_identity(envelope)?;
        self.execution = Some(ExecutionScope {
            prompt_id,
            attempt_id,
            request_id: envelope.request_id,
            kind: ExecutionKind::Plugin,
        });
        self.provider_v2_proposal_pending = false;
        self.last_provider_v2_finalization = None;
        self.lifecycle = WorkerLifecycle::Running;
        Ok(vec![self.event_response(
            envelope,
            WorkerLifecycleEvent::ExecutionStarted,
        )?])
    }

    fn handle_plugin_capability_response(
        &self,
        envelope: &WorkerEnvelope,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        self.require_running_attempt(envelope)?;
        Ok(Vec::new())
    }

    pub fn mark_provider_v2_execution(&mut self) -> Result<(), WorkerSessionError> {
        let execution = self
            .execution
            .as_mut()
            .ok_or(WorkerSessionError::StaleAttempt)?;
        if self.lifecycle != WorkerLifecycle::Running || execution.kind != ExecutionKind::Plugin {
            return Err(WorkerSessionError::InvalidDirection);
        }
        execution.kind = ExecutionKind::ProviderV2Plugin;
        Ok(())
    }

    fn handle_provider_v2_response(
        &mut self,
        envelope: &WorkerEnvelope,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        let (prompt_id, attempt_id) = attempt_identity(envelope)?;
        let execution = self
            .execution
            .as_ref()
            .ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.prompt_id != prompt_id || execution.attempt_id != attempt_id {
            return Err(WorkerSessionError::StaleAttempt);
        }
        if execution.kind != ExecutionKind::ProviderV2Plugin {
            return Err(WorkerSessionError::InvalidDirection);
        }
        match &envelope.message {
            WorkerMessage::ProviderStreamResponse { .. } if self.provider_v2_proposal_pending => {
                return Err(WorkerSessionError::InvalidDirection);
            }
            WorkerMessage::ProviderV2ProposalFinalization { finalization } => {
                finalization
                    .validate()
                    .map_err(|error| WorkerSessionError::Payload(error.to_string()))?;
            }
            _ => {}
        }
        Ok(Vec::new())
    }

    fn handle_model_source_response(
        &mut self,
        envelope: &WorkerEnvelope,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        let (prompt_id, attempt_id) = attempt_identity(envelope)?;
        let execution = self
            .execution
            .as_ref()
            .ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.prompt_id != prompt_id || execution.attempt_id != attempt_id {
            return Err(WorkerSessionError::StaleAttempt);
        }
        if execution.kind != ExecutionKind::Native {
            return Err(WorkerSessionError::InvalidDirection);
        }
        Ok(Vec::new())
    }

    fn handle_cancel(
        &mut self,
        envelope: &WorkerEnvelope,
        reason: &str,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        if self.lifecycle != WorkerLifecycle::Running {
            return Err(WorkerSessionError::InvalidState {
                expected: "running",
                actual: self.lifecycle,
            });
        }
        self.require_running_attempt(envelope)?;
        self.lifecycle = WorkerLifecycle::Cancelling;
        if let Some(execution) = &mut self.execution
            && execution.kind == ExecutionKind::Native
        {
            execution.request_id = envelope.request_id;
        }
        let response = self.event_response(
            envelope,
            WorkerLifecycleEvent::CancellationRequested {
                reason: reason.to_owned(),
            },
        )?;
        Ok(vec![response])
    }

    pub fn complete_execution(
        &mut self,
        event: Vec<u8>,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        if !matches!(
            self.lifecycle,
            WorkerLifecycle::Running | WorkerLifecycle::Cancelling
        ) {
            return Err(WorkerSessionError::InvalidState {
                expected: "running or cancelling",
                actual: self.lifecycle,
            });
        }
        let execution = self
            .execution
            .take()
            .ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.kind != ExecutionKind::Native {
            self.execution = Some(execution);
            return Err(WorkerSessionError::InvalidDirection);
        }
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.lifecycle = WorkerLifecycle::Ready;
        self.envelope_from_identity(identity, WorkerMessage::Event { event })
    }

    pub fn execution_event(
        &mut self,
        event: Vec<u8>,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        if !matches!(
            self.lifecycle,
            WorkerLifecycle::Running | WorkerLifecycle::Cancelling
        ) {
            return Err(WorkerSessionError::InvalidState {
                expected: "running or cancelling",
                actual: self.lifecycle,
            });
        }
        let execution = self.execution.ok_or(WorkerSessionError::StaleAttempt)?;
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.envelope_from_identity(identity, WorkerMessage::Event { event })
    }

    pub fn output_proposal(
        &mut self,
        proposal: WorkerOutputProposal,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        if self.lifecycle != WorkerLifecycle::Running {
            return Err(WorkerSessionError::InvalidState {
                expected: "running",
                actual: self.lifecycle,
            });
        }
        let execution = self.execution.ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.kind != ExecutionKind::Native {
            return Err(WorkerSessionError::InvalidDirection);
        }
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.envelope_from_identity(identity, WorkerMessage::OutputProposal { proposal })
    }

    pub fn plugin_capability_request(
        &mut self,
        call_id: u64,
        request: Vec<u8>,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        let execution = self.execution.ok_or(WorkerSessionError::StaleAttempt)?;
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.envelope_from_identity(
            identity,
            WorkerMessage::PluginCapabilityRequest { call_id, request },
        )
    }

    pub fn provider_stream_request(
        &mut self,
        call_id: u64,
        request: sim_types::WorkerProviderStreamRequest,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        let execution = self.execution.ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.kind != ExecutionKind::ProviderV2Plugin {
            return Err(WorkerSessionError::InvalidDirection);
        }
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.envelope_from_identity(
            identity,
            WorkerMessage::ProviderStreamRequest { call_id, request },
        )
    }

    pub fn model_source_request(
        &mut self,
        call_id: u64,
        request: sim_types::WorkerModelSourceRequest,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        let execution = self.execution.ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.kind != ExecutionKind::Native {
            return Err(WorkerSessionError::InvalidDirection);
        }
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.envelope_from_identity(
            identity,
            WorkerMessage::ModelSourceRequest { call_id, request },
        )
    }

    pub fn provider_v2_proposal(
        &mut self,
        outcome: WorkerPluginExecutionOutcome,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        let execution = self.execution.ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.kind != ExecutionKind::ProviderV2Plugin
            || self.lifecycle != WorkerLifecycle::Running
            || self.provider_v2_proposal_pending
        {
            return Err(WorkerSessionError::InvalidDirection);
        }
        self.provider_v2_proposal_pending = true;
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.envelope_from_identity(identity, WorkerMessage::PluginResult { outcome })
    }

    pub fn complete_provider_v2_finalization(
        &mut self,
        acknowledgement: sim_types::WorkerProviderV2ProposalFinalizationAck,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        acknowledgement
            .validate()
            .map_err(|error| WorkerSessionError::Payload(error.to_string()))?;
        let execution = self
            .execution
            .take()
            .ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.kind != ExecutionKind::ProviderV2Plugin {
            self.execution = Some(execution);
            return Err(WorkerSessionError::InvalidDirection);
        }
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.provider_v2_proposal_pending = false;
        self.last_provider_v2_finalization =
            Some((execution, acknowledgement.finalization.clone()));
        self.lifecycle = WorkerLifecycle::Ready;
        self.envelope_from_identity(
            identity,
            WorkerMessage::ProviderV2ProposalFinalizationAck { acknowledgement },
        )
    }

    pub fn complete_plugin_execution(
        &mut self,
        outcome: WorkerPluginExecutionOutcome,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        let execution = self
            .execution
            .take()
            .ok_or(WorkerSessionError::StaleAttempt)?;
        if !matches!(
            execution.kind,
            ExecutionKind::Plugin | ExecutionKind::ProviderV2Plugin
        ) {
            self.execution = Some(execution);
            return Err(WorkerSessionError::InvalidDirection);
        }
        self.provider_v2_proposal_pending = false;
        let identity = self
            .identity
            .clone()
            .ok_or(WorkerSessionError::IdentityChanged)?;
        let identity = SessionIdentity {
            request_id: execution.request_id,
            prompt_id: Some(execution.prompt_id),
            attempt_id: Some(execution.attempt_id),
            ..identity
        };
        self.lifecycle = WorkerLifecycle::Ready;
        self.envelope_from_identity(identity, WorkerMessage::PluginResult { outcome })
    }

    fn handle_shutdown(
        &mut self,
        envelope: &WorkerEnvelope,
    ) -> Result<Vec<WorkerEnvelope>, WorkerSessionError> {
        if self.is_terminal() {
            return Err(WorkerSessionError::InvalidState {
                expected: "non-terminal",
                actual: self.lifecycle,
            });
        }
        self.lifecycle = WorkerLifecycle::Stopping;
        self.execution = None;
        self.pending_registry = None;
        let response = self.response_to(envelope, WorkerMessage::Shutdown)?;
        self.lifecycle = WorkerLifecycle::Stopped;
        Ok(vec![response])
    }

    fn require_running_attempt(&self, envelope: &WorkerEnvelope) -> Result<(), WorkerSessionError> {
        self.require_running_execution(envelope, self.execution_kind()?)
    }

    fn require_running_execution(
        &self,
        envelope: &WorkerEnvelope,
        expected_kind: ExecutionKind,
    ) -> Result<(), WorkerSessionError> {
        let (prompt_id, attempt_id) = attempt_identity(envelope)?;
        let execution = self
            .execution
            .as_ref()
            .ok_or(WorkerSessionError::StaleAttempt)?;
        if execution.prompt_id != prompt_id
            || execution.attempt_id != attempt_id
            || execution.kind != expected_kind
        {
            return Err(WorkerSessionError::StaleAttempt);
        }
        Ok(())
    }

    fn execution_kind(&self) -> Result<ExecutionKind, WorkerSessionError> {
        self.execution
            .map(|execution| execution.kind)
            .ok_or(WorkerSessionError::StaleAttempt)
    }

    fn event_response(
        &mut self,
        envelope: &WorkerEnvelope,
        event: WorkerLifecycleEvent,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        self.response_to(envelope, WorkerMessage::Lifecycle { event })
    }

    fn response_to(
        &mut self,
        envelope: &WorkerEnvelope,
        message: WorkerMessage,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        let identity = SessionIdentity {
            profile_id: envelope.profile_id,
            worker_id: envelope.worker_id,
            request_id: envelope.request_id,
            prompt_id: envelope.prompt_id,
            attempt_id: envelope.attempt_id,
            registry_version: envelope.registry_version.clone(),
        };
        self.envelope_from_identity(identity, message)
    }

    fn envelope_from_identity(
        &mut self,
        identity: SessionIdentity,
        message: WorkerMessage,
    ) -> Result<WorkerEnvelope, WorkerSessionError> {
        let sequence = self.next_output_sequence;
        self.next_output_sequence = self
            .next_output_sequence
            .checked_add(1)
            .ok_or(WorkerSessionError::OutputSequenceOverflow)?;
        Ok(WorkerEnvelope {
            version: WORKER_PROTOCOL_VERSION,
            profile_id: identity.profile_id,
            worker_id: identity.worker_id,
            request_id: identity.request_id,
            prompt_id: identity.prompt_id,
            attempt_id: identity.attempt_id,
            sequence,
            registry_version: identity.registry_version,
            message,
            extensions: BTreeMap::new(),
        })
    }
}

impl PendingWorkerRegistry {
    fn new(begin: WorkerRegistryDeploymentBegin) -> Result<Self, WorkerSessionError> {
        let mut value = Self {
            assembled: Vec::new(),
            begin,
            component_index: 0,
            content: WorkerComponentContent::Manifest,
            chunk_index: 0,
            manifest_bytes: Vec::new(),
            authorization_bytes: Vec::new(),
            component_bytes: Vec::new(),
        };
        value.reserve_current_content()?;
        Ok(value)
    }

    fn apply_chunk(
        &mut self,
        chunk: &WorkerRegistryDeploymentChunk,
    ) -> Result<(), WorkerSessionError> {
        if chunk.generation() != self.begin.generation() {
            return Err(WorkerSessionError::DeploymentGenerationMismatch);
        }
        let expected_component_index = u32::try_from(self.component_index)
            .map_err(|_| WorkerSessionError::DeploymentComponentCountOverflow)?;
        if chunk.component_index() != expected_component_index {
            return Err(WorkerSessionError::UnexpectedComponentIndex {
                expected: expected_component_index,
                actual: chunk.component_index(),
            });
        }
        if chunk.content() != self.content {
            return Err(WorkerSessionError::UnexpectedComponentContent {
                expected: self.content,
                actual: chunk.content(),
            });
        }
        if chunk.chunk_index() != self.chunk_index {
            return Err(WorkerSessionError::UnexpectedChunkIndex {
                expected: self.chunk_index,
                actual: chunk.chunk_index(),
            });
        }
        let descriptor = self
            .begin
            .components()
            .get(self.component_index)
            .cloned()
            .ok_or(WorkerSessionError::DeploymentIncomplete)?;
        let expected_length = expected_chunk_length(&descriptor, self.content, self.chunk_index)?;
        if chunk.bytes().len() != expected_length {
            return Err(WorkerSessionError::UnexpectedChunkLength {
                expected: expected_length,
                actual: chunk.bytes().len(),
            });
        }
        match self.content {
            WorkerComponentContent::Manifest => {
                self.manifest_bytes.extend_from_slice(chunk.bytes())
            }
            WorkerComponentContent::Authorization => {
                self.authorization_bytes.extend_from_slice(chunk.bytes())
            }
            WorkerComponentContent::Component => {
                self.component_bytes.extend_from_slice(chunk.bytes())
            }
        }
        self.chunk_index = self
            .chunk_index
            .checked_add(1)
            .ok_or(WorkerSessionError::DeploymentComponentCountOverflow)?;
        let expected_chunk_count = match self.content {
            WorkerComponentContent::Manifest => descriptor.manifest_chunk_count(),
            WorkerComponentContent::Authorization => descriptor.authorization_chunk_count(),
            WorkerComponentContent::Component => descriptor.component_chunk_count(),
        };
        if self.chunk_index != expected_chunk_count {
            return Ok(());
        }
        match self.content {
            WorkerComponentContent::Manifest => {
                verify_digest(
                    descriptor.manifest_digest_sha256(),
                    &self.manifest_bytes,
                    WorkerComponentContent::Manifest,
                )?;
                self.content = WorkerComponentContent::Authorization;
                self.chunk_index = 0;
                self.reserve_current_content()?;
            }
            WorkerComponentContent::Authorization => {
                verify_digest(
                    descriptor.authorization_generation(),
                    &self.authorization_bytes,
                    WorkerComponentContent::Authorization,
                )?;
                self.content = WorkerComponentContent::Component;
                self.chunk_index = 0;
                self.reserve_current_content()?;
            }
            WorkerComponentContent::Component => {
                verify_digest(
                    descriptor.component_digest_sha256(),
                    &self.component_bytes,
                    WorkerComponentContent::Component,
                )?;
                self.assembled.push(AssembledWorkerComponent {
                    extension_id: descriptor.extension_id().to_owned(),
                    extension_version: descriptor.extension_version().to_owned(),
                    plugin_identifier: descriptor.plugin_identifier().to_owned(),
                    plugin_version: descriptor.plugin_version().to_owned(),
                    authorization_generation: descriptor.authorization_generation().clone(),
                    manifest_digest_sha256: descriptor.manifest_digest_sha256().clone(),
                    component_digest_sha256: descriptor.component_digest_sha256().clone(),
                    manifest_bytes: std::mem::take(&mut self.manifest_bytes),
                    authorization_bytes: std::mem::take(&mut self.authorization_bytes),
                    component_bytes: std::mem::take(&mut self.component_bytes),
                });
                self.component_index = self
                    .component_index
                    .checked_add(1)
                    .ok_or(WorkerSessionError::DeploymentComponentCountOverflow)?;
                self.content = WorkerComponentContent::Manifest;
                self.chunk_index = 0;
                self.reserve_current_content()?;
            }
        }
        Ok(())
    }

    fn reserve_current_content(&mut self) -> Result<(), WorkerSessionError> {
        let Some(descriptor) = self.begin.components().get(self.component_index) else {
            return Ok(());
        };
        let byte_length = match self.content {
            WorkerComponentContent::Manifest => descriptor.manifest_bytes(),
            WorkerComponentContent::Authorization => descriptor.authorization_bytes(),
            WorkerComponentContent::Component => descriptor.component_bytes(),
        };
        let byte_length =
            usize::try_from(byte_length).map_err(|_| WorkerSessionError::DeploymentAllocation)?;
        let bytes = match self.content {
            WorkerComponentContent::Manifest => &mut self.manifest_bytes,
            WorkerComponentContent::Authorization => &mut self.authorization_bytes,
            WorkerComponentContent::Component => &mut self.component_bytes,
        };
        bytes
            .try_reserve_exact(byte_length)
            .map_err(|_| WorkerSessionError::DeploymentAllocation)
    }

    fn is_complete(&self) -> bool {
        self.component_index == self.begin.components().len()
            && self.manifest_bytes.is_empty()
            && self.authorization_bytes.is_empty()
            && self.component_bytes.is_empty()
            && self.chunk_index == 0
    }

    fn finish(self) -> AssembledWorkerRegistry {
        AssembledWorkerRegistry {
            generation: self.begin.generation(),
            registry_digest_sha256: self.begin.registry_digest_sha256().clone(),
            components: self.assembled,
        }
    }
}

fn expected_chunk_length(
    descriptor: &WorkerComponentDescriptor,
    content: WorkerComponentContent,
    chunk_index: u32,
) -> Result<usize, WorkerSessionError> {
    let total = match content {
        WorkerComponentContent::Manifest => descriptor.manifest_bytes(),
        WorkerComponentContent::Authorization => descriptor.authorization_bytes(),
        WorkerComponentContent::Component => descriptor.component_bytes(),
    };
    let chunk_bytes = u64::try_from(MAX_WORKER_COMPONENT_CHUNK_BYTES)
        .map_err(|_| WorkerSessionError::DeploymentAllocation)?;
    let offset = u64::from(chunk_index)
        .checked_mul(chunk_bytes)
        .ok_or(WorkerSessionError::DeploymentAllocation)?;
    let remaining = total
        .checked_sub(offset)
        .ok_or(WorkerSessionError::DeploymentIncomplete)?;
    usize::try_from(remaining.min(chunk_bytes))
        .map_err(|_| WorkerSessionError::DeploymentAllocation)
}

fn verify_digest(
    expected: &WorkerSha256Digest,
    bytes: &[u8],
    content: WorkerComponentContent,
) -> Result<(), WorkerSessionError> {
    let actual: [u8; 32] = Sha256::digest(bytes).into();
    if actual != expected.bytes() {
        return Err(WorkerSessionError::ContentDigestMismatch(content));
    }
    Ok(())
}

fn attempt_identity(
    envelope: &WorkerEnvelope,
) -> Result<(PromptId, AttemptId), WorkerSessionError> {
    match (envelope.prompt_id, envelope.attempt_id) {
        (Some(prompt_id), Some(attempt_id)) => Ok((prompt_id, attempt_id)),
        _ => Err(WorkerSessionError::MissingAttemptIdentity),
    }
}

#[cfg(test)]
mod tests {
    use sim_types::{
        AttemptId, ProfileId, PromptId, RequestId, WorkerComponentDescriptor, WorkerId,
        WorkerRegistryDeploymentBegin, WorkerSha256Digest,
    };

    use super::*;

    const AUTHORIZATION_BYTES: &[u8] = b"authorization-generation";

    fn envelope(sequence: u64, message: WorkerMessage) -> WorkerEnvelope {
        WorkerEnvelope {
            version: WORKER_PROTOCOL_VERSION,
            profile_id: ProfileId(Default::default()),
            worker_id: WorkerId(Default::default()),
            request_id: RequestId(Default::default()),
            prompt_id: None,
            attempt_id: None,
            sequence,
            registry_version: "registry-v1".to_owned(),
            message,
            extensions: BTreeMap::new(),
        }
    }

    fn attempt_envelope(sequence: u64, message: WorkerMessage) -> WorkerEnvelope {
        WorkerEnvelope {
            prompt_id: Some(PromptId(Default::default())),
            attempt_id: Some(AttemptId(Default::default())),
            ..envelope(sequence, message)
        }
    }

    fn cpu_backend_wire() -> sim_types::WorkerBackendCapabilities {
        CpuBackend::capability_matrix()
            .to_worker_capabilities()
            .expect("CPU capabilities project to worker protocol")
    }

    fn sha256(bytes: &[u8]) -> WorkerSha256Digest {
        let digest = Sha256::digest(bytes);
        let value = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        WorkerSha256Digest::new(value).expect("SHA-256 is canonical lowercase hex")
    }

    fn deployment(
        generation: u64,
        manifest: &[u8],
        component: &[u8],
    ) -> WorkerRegistryDeploymentBegin {
        let generation = WorkerRegistryGeneration::new(generation).expect("nonzero generation");
        let descriptor = WorkerComponentDescriptor::new(
            "test-extension",
            "1.0.0",
            "test.plugin",
            "1.0.0",
            sha256(AUTHORIZATION_BYTES),
            sha256(manifest),
            sha256(component),
            u64::try_from(manifest.len()).expect("bounded manifest"),
            u64::try_from(AUTHORIZATION_BYTES.len()).expect("bounded authorization"),
            u64::try_from(component.len()).expect("bounded component"),
        )
        .expect("bounded descriptor");
        let provisional = WorkerRegistryDeploymentBegin::new(
            generation,
            WorkerSha256Digest::new("0".repeat(64)).expect("zero digest is structural"),
            vec![descriptor],
        )
        .expect("bounded deployment");
        WorkerRegistryDeploymentBegin::new(
            generation,
            sha256(&provisional.digest_material()),
            provisional.components().to_vec(),
        )
        .expect("content-addressed deployment")
    }

    fn ready_session() -> WorkerSession {
        let mut session = WorkerSession::new(1024 * 1024).expect("worker session");
        let responses = session
            .handle(envelope(
                0,
                WorkerMessage::Hello {
                    backend: cpu_backend_wire(),
                },
            ))
            .expect("handshake");
        assert!(matches!(
            responses[0].message,
            WorkerMessage::HelloAck { .. }
        ));
        assert!(matches!(responses[1].message, WorkerMessage::Ready));
        session
    }

    fn provider_v2_finalization() -> sim_types::WorkerProviderV2ProposalFinalization {
        let context = sim_types::WorkerProviderInvocationContext {
            session_id: uuid::Uuid::from_u128(0x425_200),
            session_generation: 3,
            invocation: 5,
            generation: 7,
        };
        sim_types::WorkerProviderV2ProposalFinalization {
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
            receipt_identity_sha256: WorkerSha256Digest::new("a".repeat(64))
                .expect("receipt identity"),
            materialization_identity_sha256: WorkerSha256Digest::new("b".repeat(64))
                .expect("materialization identity"),
        }
    }

    fn stage_single_component_deployment(
        session: &mut WorkerSession,
        first_sequence: u64,
        deployment: &WorkerRegistryDeploymentBegin,
        manifest: &[u8],
        component: &[u8],
    ) {
        let generation = deployment.generation();
        for (sequence, content, bytes) in [
            (
                first_sequence,
                WorkerComponentContent::Manifest,
                manifest.to_vec(),
            ),
            (
                first_sequence + 1,
                WorkerComponentContent::Authorization,
                AUTHORIZATION_BYTES.to_vec(),
            ),
            (
                first_sequence + 2,
                WorkerComponentContent::Component,
                component.to_vec(),
            ),
        ] {
            session
                .handle(envelope(
                    sequence,
                    WorkerMessage::RegistryDeploymentChunk {
                        chunk: WorkerRegistryDeploymentChunk::new(generation, 0, content, 0, bytes)
                            .expect("single bounded deployment chunk"),
                    },
                ))
                .expect("deployment content assembles");
        }
    }

    #[test]
    fn handshake_negotiates_capabilities_and_monotonic_output() {
        let session = ready_session();
        assert_eq!(session.lifecycle(), WorkerLifecycle::Ready);
        assert_eq!(
            session
                .accepted_backend()
                .map(BackendCapabilityMatrix::device),
            Some(sim_tensor::DeviceId::CPU)
        );
        assert_eq!(
            session
                .memory_snapshot()
                .expect("ready session has memory authority")
                .current_bytes,
            0
        );
        let first = session.backend().expect("ready session backend");
        let second = session.backend().expect("ready session backend");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn unavailable_backend_is_typed_bounded_and_never_becomes_ready() {
        let unavailable = BackendUnavailable::new(
            sim_types::DeviceKind::Rocm,
            "signed package verification failed",
        );
        let mut session = WorkerSession::with_backend_result(Err(unavailable.clone()));
        let responses = session
            .handle(envelope(
                0,
                WorkerMessage::Hello {
                    backend: cpu_backend_wire(),
                },
            ))
            .expect("typed unavailable response");
        let [response] = responses.as_slice() else {
            panic!("unavailable backend emitted an unexpected response count");
        };
        let WorkerMessage::Fatal { code, message } = &response.message else {
            panic!("unavailable backend did not emit Fatal");
        };
        assert_eq!(code, "backend_unavailable");
        assert_eq!(
            serde_json::from_str::<BackendUnavailable>(message).expect("typed fatal payload"),
            unavailable
        );
        assert_eq!(session.lifecycle(), WorkerLifecycle::Fatal);
        assert!(session.accepted_backend().is_none());
    }

    #[test]
    fn registry_deployment_assembles_and_acknowledges_verified_content() {
        let manifest = b"signed manifest";
        let component = b"component model bytes";
        let deployment = deployment(1, manifest, component);
        let generation = deployment.generation();
        let registry_digest = deployment.registry_digest_sha256().clone();
        let mut session = ready_session();
        assert!(
            session
                .handle(envelope(
                    1,
                    WorkerMessage::RegistryDeploymentBegin {
                        deployment: deployment.clone(),
                    },
                ))
                .expect("deployment begins")
                .is_empty()
        );
        assert!(
            session
                .handle(envelope(
                    2,
                    WorkerMessage::RegistryDeploymentChunk {
                        chunk: WorkerRegistryDeploymentChunk::new(
                            generation,
                            0,
                            WorkerComponentContent::Manifest,
                            0,
                            manifest.to_vec(),
                        )
                        .expect("manifest chunk"),
                    },
                ))
                .expect("manifest assembles")
                .is_empty()
        );
        assert!(
            session
                .handle(envelope(
                    3,
                    WorkerMessage::RegistryDeploymentChunk {
                        chunk: WorkerRegistryDeploymentChunk::new(
                            generation,
                            0,
                            WorkerComponentContent::Authorization,
                            0,
                            AUTHORIZATION_BYTES.to_vec(),
                        )
                        .expect("authorization chunk"),
                    },
                ))
                .expect("authorization assembles")
                .is_empty()
        );
        assert!(
            session
                .handle(envelope(
                    4,
                    WorkerMessage::RegistryDeploymentChunk {
                        chunk: WorkerRegistryDeploymentChunk::new(
                            generation,
                            0,
                            WorkerComponentContent::Component,
                            0,
                            component.to_vec(),
                        )
                        .expect("component chunk"),
                    },
                ))
                .expect("component assembles")
                .is_empty()
        );
        let responses = session
            .handle_verified_registry_commit(
                envelope(
                    5,
                    WorkerMessage::RegistryDeploymentCommit {
                        commit: WorkerRegistryDeploymentCommit::new(
                            generation,
                            registry_digest.clone(),
                        ),
                    },
                ),
                |_| Ok::<_, WorkerRegistryDeploymentRejectionReason>(()),
            )
            .expect("verified deployment commits");
        assert!(matches!(
            &responses[0].message,
            WorkerMessage::RegistryDeploymentAck { acknowledgement }
                if acknowledgement.generation() == generation
                    && acknowledgement.registry_digest_sha256() == &registry_digest
                    && acknowledgement.component_count() == 1
        ));
        let registry = session.registry().expect("committed registry snapshot");
        assert_eq!(registry.generation(), generation);
        assert_eq!(registry.registry_digest_sha256(), &registry_digest);
        assert_eq!(registry.components()[0].extension_id(), "test-extension");
        assert_eq!(registry.components()[0].extension_version(), "1.0.0");
        assert_eq!(registry.components()[0].plugin_identifier(), "test.plugin");
        assert_eq!(registry.components()[0].plugin_version(), "1.0.0");
        assert_eq!(
            registry.components()[0].authorization_generation(),
            deployment.components()[0].authorization_generation()
        );
        assert_eq!(registry.components()[0].manifest_bytes(), manifest);
        assert_eq!(
            registry.components()[0].authorization_bytes(),
            AUTHORIZATION_BYTES
        );
        assert_eq!(registry.components()[0].component_bytes(), component);
        assert_eq!(
            registry.components()[0].manifest_digest_sha256(),
            deployment.components()[0].manifest_digest_sha256()
        );
        assert_eq!(
            registry.components()[0].component_digest_sha256(),
            deployment.components()[0].component_digest_sha256()
        );

        assert_eq!(
            session.handle(envelope(
                6,
                WorkerMessage::RegistryDeploymentBegin { deployment }
            )),
            Err(WorkerSessionError::StaleRegistryGeneration {
                current: 1,
                actual: 1,
            })
        );
    }

    #[test]
    fn registry_replacement_is_atomic_and_unverified_commit_has_no_bypass() {
        let mut session = ready_session();
        let initial_manifest = b"initial manifest";
        let initial_component = b"initial component";
        let initial = deployment(1, initial_manifest, initial_component);
        session
            .handle(envelope(
                1,
                WorkerMessage::RegistryDeploymentBegin {
                    deployment: initial.clone(),
                },
            ))
            .expect("initial deployment begins");
        stage_single_component_deployment(
            &mut session,
            2,
            &initial,
            initial_manifest,
            initial_component,
        );
        session
            .handle_verified_registry_commit(
                envelope(
                    5,
                    WorkerMessage::RegistryDeploymentCommit {
                        commit: WorkerRegistryDeploymentCommit::new(
                            initial.generation(),
                            initial.registry_digest_sha256().clone(),
                        ),
                    },
                ),
                |_| Ok::<_, WorkerRegistryDeploymentRejectionReason>(()),
            )
            .expect("initial registry verifies and commits");
        let committed = session
            .registry()
            .expect("initial registry remains available")
            .clone();

        let replacement_manifest = b"replacement manifest";
        let replacement_component = b"replacement component";
        let replacement = deployment(2, replacement_manifest, replacement_component);
        session
            .handle(envelope(
                6,
                WorkerMessage::RegistryDeploymentBegin {
                    deployment: replacement.clone(),
                },
            ))
            .expect("replacement deployment begins");
        stage_single_component_deployment(
            &mut session,
            7,
            &replacement,
            replacement_manifest,
            replacement_component,
        );
        let rejection = session
            .handle_verified_registry_commit(
                envelope(
                    10,
                    WorkerMessage::RegistryDeploymentCommit {
                        commit: WorkerRegistryDeploymentCommit::new(
                            replacement.generation(),
                            replacement.registry_digest_sha256().clone(),
                        ),
                    },
                ),
                |_| Err(WorkerRegistryDeploymentRejectionReason::ComponentCompilationFailed),
            )
            .expect("invalid replacement is rejected without terminating the session");
        assert!(matches!(
            &rejection[0].message,
            WorkerMessage::RegistryDeploymentRejected { rejection }
                if rejection.generation() == replacement.generation()
                    && rejection.registry_digest_sha256()
                        == replacement.registry_digest_sha256()
                    && rejection.reason()
                        == WorkerRegistryDeploymentRejectionReason::ComponentCompilationFailed
        ));
        assert_eq!(session.registry(), Some(&committed));
        let started = session
            .handle(attempt_envelope(
                11,
                WorkerMessage::ExecutePlugin {
                    invocation: vec![1],
                },
            ))
            .expect("previously committed registry remains executable");
        assert!(matches!(
            started[0].message,
            WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::ExecutionStarted
            }
        ));
        session
            .complete_plugin_execution(WorkerPluginExecutionOutcome::Failed(
                sim_types::WorkerPluginExecutionFailure::InvalidInvocation,
            ))
            .expect("old-registry execution can converge");

        let retry = deployment(2, replacement_manifest, replacement_component);
        session
            .handle(envelope(
                12,
                WorkerMessage::RegistryDeploymentBegin {
                    deployment: retry.clone(),
                },
            ))
            .expect("failed replacement does not consume its generation");
        stage_single_component_deployment(
            &mut session,
            13,
            &retry,
            replacement_manifest,
            replacement_component,
        );
        assert_eq!(
            session.handle(envelope(
                16,
                WorkerMessage::RegistryDeploymentCommit {
                    commit: WorkerRegistryDeploymentCommit::new(
                        retry.generation(),
                        retry.registry_digest_sha256().clone(),
                    ),
                },
            )),
            Err(WorkerSessionError::RegistryVerificationRequired)
        );
        assert_eq!(session.registry(), Some(&committed));
    }

    #[test]
    fn registry_deployment_rejects_digest_order_and_partial_commit() {
        let manifest = b"manifest";
        let component = vec![7; MAX_WORKER_COMPONENT_CHUNK_BYTES + 1];

        let mut invalid_begin = deployment(1, manifest, &component);
        invalid_begin = WorkerRegistryDeploymentBegin::new(
            invalid_begin.generation(),
            WorkerSha256Digest::new("f".repeat(64)).expect("structural digest"),
            invalid_begin.components().to_vec(),
        )
        .expect("structurally valid deployment");
        let mut session = ready_session();
        assert_eq!(
            session.handle(envelope(
                1,
                WorkerMessage::RegistryDeploymentBegin {
                    deployment: invalid_begin,
                },
            )),
            Err(WorkerSessionError::RegistryDigestMismatch)
        );

        let deployment = deployment(1, manifest, &component);
        let generation = deployment.generation();
        let mut session = ready_session();
        session
            .handle(envelope(
                1,
                WorkerMessage::RegistryDeploymentBegin {
                    deployment: deployment.clone(),
                },
            ))
            .expect("deployment begins");
        assert!(matches!(
            session.handle(envelope(
                2,
                WorkerMessage::RegistryDeploymentChunk {
                    chunk: WorkerRegistryDeploymentChunk::new(
                        generation,
                        0,
                        WorkerComponentContent::Component,
                        0,
                        vec![1],
                    )
                    .expect("bounded wrong-order chunk"),
                },
            )),
            Err(WorkerSessionError::UnexpectedComponentContent {
                expected: WorkerComponentContent::Manifest,
                actual: WorkerComponentContent::Component,
            })
        ));

        let mut session = ready_session();
        session
            .handle(envelope(
                1,
                WorkerMessage::RegistryDeploymentBegin {
                    deployment: deployment.clone(),
                },
            ))
            .expect("deployment begins");
        session
            .handle(envelope(
                2,
                WorkerMessage::RegistryDeploymentChunk {
                    chunk: WorkerRegistryDeploymentChunk::new(
                        generation,
                        0,
                        WorkerComponentContent::Manifest,
                        0,
                        manifest.to_vec(),
                    )
                    .expect("manifest chunk"),
                },
            ))
            .expect("manifest assembles");
        session
            .handle(envelope(
                3,
                WorkerMessage::RegistryDeploymentChunk {
                    chunk: WorkerRegistryDeploymentChunk::new(
                        generation,
                        0,
                        WorkerComponentContent::Authorization,
                        0,
                        AUTHORIZATION_BYTES.to_vec(),
                    )
                    .expect("authorization chunk"),
                },
            ))
            .expect("authorization assembles");
        session
            .handle(envelope(
                4,
                WorkerMessage::RegistryDeploymentChunk {
                    chunk: WorkerRegistryDeploymentChunk::new(
                        generation,
                        0,
                        WorkerComponentContent::Component,
                        0,
                        component[..MAX_WORKER_COMPONENT_CHUNK_BYTES].to_vec(),
                    )
                    .expect("first component chunk"),
                },
            ))
            .expect("first component chunk assembles");
        assert_eq!(
            session.handle(envelope(
                5,
                WorkerMessage::RegistryDeploymentChunk {
                    chunk: WorkerRegistryDeploymentChunk::new(
                        generation,
                        0,
                        WorkerComponentContent::Component,
                        0,
                        component[..MAX_WORKER_COMPONENT_CHUNK_BYTES].to_vec(),
                    )
                    .expect("duplicate component chunk"),
                },
            )),
            Err(WorkerSessionError::UnexpectedChunkIndex {
                expected: 1,
                actual: 0,
            })
        );

        let mut session = ready_session();
        session
            .handle(envelope(
                1,
                WorkerMessage::RegistryDeploymentBegin {
                    deployment: deployment.clone(),
                },
            ))
            .expect("deployment begins");
        assert_eq!(
            session.handle_verified_registry_commit(
                envelope(
                    2,
                    WorkerMessage::RegistryDeploymentCommit {
                        commit: WorkerRegistryDeploymentCommit::new(
                            generation,
                            deployment.registry_digest_sha256().clone(),
                        ),
                    },
                ),
                |_| Ok::<_, WorkerRegistryDeploymentRejectionReason>(())
            ),
            Err(WorkerSessionError::DeploymentIncomplete)
        );
    }

    #[test]
    fn registry_deployment_rejects_changed_content_and_opaque_extensions() {
        let manifest = b"manifest";
        let component = b"component";
        let deployment = deployment(1, manifest, component);
        let generation = deployment.generation();
        let mut session = ready_session();
        session
            .handle(envelope(
                1,
                WorkerMessage::RegistryDeploymentBegin { deployment },
            ))
            .expect("deployment begins");
        assert_eq!(
            session.handle(envelope(
                2,
                WorkerMessage::RegistryDeploymentChunk {
                    chunk: WorkerRegistryDeploymentChunk::new(
                        generation,
                        0,
                        WorkerComponentContent::Manifest,
                        0,
                        b"manifesx".to_vec(),
                    )
                    .expect("same-length changed manifest"),
                },
            )),
            Err(WorkerSessionError::ContentDigestMismatch(
                WorkerComponentContent::Manifest
            ))
        );

        let mut session = WorkerSession::new(1024).expect("session");
        let mut hello = envelope(
            0,
            WorkerMessage::Hello {
                backend: cpu_backend_wire(),
            },
        );
        hello.extensions.insert("escape".into(), vec![1]);
        assert_eq!(
            session.handle(hello),
            Err(WorkerSessionError::OpaqueExtensions)
        );
    }

    #[test]
    fn duplicate_or_gapped_input_sequence_is_rejected() {
        let mut session = ready_session();
        assert_eq!(
            session.handle(attempt_envelope(
                2,
                WorkerMessage::Execute { plan: vec![1] }
            )),
            Err(WorkerSessionError::Sequence {
                expected: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn execute_cancel_completion_and_shutdown_converge() {
        let mut session = ready_session();
        let started = session
            .handle(attempt_envelope(
                1,
                WorkerMessage::Execute { plan: vec![1] },
            ))
            .expect("execution starts");
        assert_eq!(session.lifecycle(), WorkerLifecycle::Running);
        assert!(matches!(
            started[0].message,
            WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::ExecutionStarted
            }
        ));
        let cancelled = session
            .handle(attempt_envelope(
                2,
                WorkerMessage::Cancel {
                    reason: "operator".to_owned(),
                },
            ))
            .expect("cancellation is requested");
        let WorkerMessage::Lifecycle { event } = &cancelled[0].message else {
            panic!("expected cancellation event");
        };
        assert_eq!(
            event,
            &WorkerLifecycleEvent::CancellationRequested {
                reason: "operator".to_owned()
            }
        );
        assert_eq!(session.lifecycle(), WorkerLifecycle::Cancelling);
        let completed = session
            .complete_execution(vec![9])
            .expect("cancellation completion converges");
        assert!(matches!(completed.message, WorkerMessage::Event { event } if event == vec![9]));
        assert_eq!(session.lifecycle(), WorkerLifecycle::Ready);
        let stopped = session
            .handle(envelope(3, WorkerMessage::Shutdown))
            .expect("shutdown");
        assert!(matches!(stopped[0].message, WorkerMessage::Shutdown));
        assert!(session.is_terminal());
    }

    #[test]
    fn output_proposal_is_worker_to_host_only_and_does_not_own_commit_state() {
        let mut session = ready_session();
        session
            .handle(attempt_envelope(
                1,
                WorkerMessage::Execute { plan: vec![1] },
            ))
            .expect("execution starts");
        let proposal = WorkerOutputProposal::new(Default::default(), vec![1], vec![2])
            .expect("bounded output proposal");
        let response = session
            .output_proposal(proposal.clone())
            .expect("worker emits output proposal");
        assert!(matches!(
            response.message,
            WorkerMessage::OutputProposal { proposal: emitted } if emitted == proposal
        ));
        assert_eq!(session.lifecycle(), WorkerLifecycle::Running);
        assert_eq!(
            session.handle(attempt_envelope(
                2,
                WorkerMessage::OutputProposal { proposal },
            )),
            Err(WorkerSessionError::InvalidDirection)
        );
    }

    #[test]
    fn provider_stream_messages_are_rejected_until_the_canonical_bridge_is_active() {
        let handle = sim_types::WorkerProviderStreamHandle {
            session_id: uuid::Uuid::from_u128(1),
            session_generation: 1,
            invocation: 1,
            slot: 1,
            generation: 1,
        };
        assert_eq!(
            ready_session().handle(envelope(
                1,
                WorkerMessage::ProviderStreamRequest {
                    call_id: 1,
                    request: sim_types::WorkerProviderStreamRequest::CheckCancelled(handle),
                },
            )),
            Err(WorkerSessionError::InvalidDirection)
        );
        let response = || WorkerMessage::ProviderStreamResponse {
            call_id: 1,
            response: sim_types::WorkerProviderStreamResponse::Unit(Ok(())),
        };
        assert_eq!(
            ready_session().handle(envelope(1, response())),
            Err(WorkerSessionError::MissingAttemptIdentity)
        );
        assert_eq!(
            ready_session().handle(attempt_envelope(1, response())),
            Err(WorkerSessionError::StaleAttempt)
        );
    }

    #[test]
    fn model_source_worker_message_consumers_are_exhaustive() {
        let source_names = vec!["fixture.safetensors".to_owned()];
        let selection =
            sim_types::worker_model_source_selection_sha256("checkpoints", &source_names)
                .expect("bounded source selection");
        let context = sim_types::WorkerModelSourceContext {
            session_id: uuid::Uuid::from_u128(0x39910),
            attempt_id: AttemptId(Default::default()),
            attempt_generation: 1,
            node_id: "loader".to_owned(),
            node_generation: 1,
            service_id: uuid::Uuid::from_u128(0x39911),
            service_generation: 1,
            ordered_source_identity_sha256: selection,
        };
        let request = sim_types::WorkerModelSourceRequest {
            context: context.clone(),
            call_ordinal: 1,
            operation: sim_types::WorkerModelSourceOperation::Open {
                folder_category: "checkpoints".to_owned(),
                source_names,
            },
        };
        let response = sim_types::WorkerModelSourceResponse::rejected(
            context.session_id,
            1,
            sim_types::WorkerModelSourceError::HostFailure,
        )
        .expect("bounded model-source response");

        let mut native = ready_session();
        native
            .handle(attempt_envelope(
                1,
                WorkerMessage::Execute { plan: vec![1] },
            ))
            .expect("native execution starts");
        let outgoing = native
            .model_source_request(9, request.clone())
            .expect("native execution may request model bytes");
        assert!(matches!(
            outgoing.message,
            WorkerMessage::ModelSourceRequest { call_id: 9, .. }
        ));
        assert!(
            native
                .handle(attempt_envelope(
                    2,
                    WorkerMessage::ModelSourceResponse {
                        call_id: 9,
                        response: response.clone(),
                    },
                ))
                .expect("app response is consumed by the native execution")
                .is_empty()
        );

        let mut wrong_direction = ready_session();
        wrong_direction
            .handle(attempt_envelope(
                1,
                WorkerMessage::Execute { plan: vec![1] },
            ))
            .expect("native execution starts");
        assert_eq!(
            wrong_direction.handle(attempt_envelope(
                2,
                WorkerMessage::ModelSourceRequest {
                    call_id: 9,
                    request,
                },
            )),
            Err(WorkerSessionError::InvalidDirection),
        );
        assert_eq!(
            ready_session().handle(attempt_envelope(
                1,
                WorkerMessage::ModelSourceResponse {
                    call_id: 9,
                    response,
                },
            )),
            Err(WorkerSessionError::StaleAttempt),
        );
    }

    #[test]
    fn provider_v2_finalization_is_ordered_typed_and_one_use() {
        let executing_session = || {
            let mut session = ready_session();
            session.registry = Some(AssembledWorkerRegistry::empty_for_test(
                WorkerRegistryGeneration::new(1).expect("generation"),
                WorkerSha256Digest::new("c".repeat(64)).expect("registry digest"),
            ));
            session
                .handle(attempt_envelope(
                    1,
                    WorkerMessage::ExecutePlugin {
                        invocation: vec![1],
                    },
                ))
                .expect("provider-v2 execution starts");
            session
                .mark_provider_v2_execution()
                .expect("decoded invocation selects the provider-v2 route");
            session
        };
        let finalization = provider_v2_finalization();
        let mut ordinary = ready_session();
        ordinary.registry = Some(AssembledWorkerRegistry::empty_for_test(
            WorkerRegistryGeneration::new(1).expect("generation"),
            WorkerSha256Digest::new("c".repeat(64)).expect("registry digest"),
        ));
        ordinary
            .handle(attempt_envelope(
                1,
                WorkerMessage::ExecutePlugin {
                    invocation: vec![1],
                },
            ))
            .expect("ordinary plugin execution starts");
        assert_eq!(
            ordinary.handle(attempt_envelope(
                2,
                WorkerMessage::ProviderV2ProposalFinalization {
                    finalization: finalization.clone(),
                },
            )),
            Err(WorkerSessionError::InvalidDirection)
        );

        let mut foreign_response = executing_session();
        let mut response_envelope = attempt_envelope(
            2,
            WorkerMessage::ProviderStreamResponse {
                call_id: 1,
                response: sim_types::WorkerProviderStreamResponse::Unit(Ok(())),
            },
        );
        response_envelope.attempt_id = Some(AttemptId(uuid::Uuid::from_u128(2)));
        assert_eq!(
            foreign_response.handle(response_envelope),
            Err(WorkerSessionError::StaleAttempt)
        );

        let mut foreign_finalization = executing_session();
        let mut finalization_envelope = attempt_envelope(
            2,
            WorkerMessage::ProviderV2ProposalFinalization {
                finalization: finalization.clone(),
            },
        );
        finalization_envelope.attempt_id = Some(AttemptId(uuid::Uuid::from_u128(2)));
        assert_eq!(
            foreign_finalization.handle(finalization_envelope),
            Err(WorkerSessionError::StaleAttempt)
        );

        let mut wrong_order = executing_session();
        assert!(
            wrong_order
                .handle(attempt_envelope(
                    2,
                    WorkerMessage::ProviderV2ProposalFinalization {
                        finalization: finalization.clone(),
                    },
                ))
                .expect("structurally valid finalization reaches the retained-state owner")
                .is_empty()
        );
        let wrong_order_acknowledgement = sim_types::WorkerProviderV2ProposalFinalizationAck {
            finalization: finalization.clone(),
            result: Err(sim_types::WorkerProviderStreamError::InvalidOrder),
        };
        wrong_order
            .complete_provider_v2_finalization(wrong_order_acknowledgement)
            .expect("wrong-order finalization returns a typed acknowledgement");

        let mut session = executing_session();
        session
            .provider_v2_proposal(WorkerPluginExecutionOutcome::Succeeded(vec![1]))
            .expect("worker retains a proposal before finalization");
        assert_eq!(
            session.handle(attempt_envelope(
                2,
                WorkerMessage::ProviderStreamResponse {
                    call_id: 1,
                    response: sim_types::WorkerProviderStreamResponse::Unit(Ok(())),
                },
            )),
            Err(WorkerSessionError::InvalidDirection)
        );
        let mut malformed = finalization.clone();
        malformed.finalization_nonce = [0; 32];
        assert!(matches!(
            session.handle(attempt_envelope(
                3,
                WorkerMessage::ProviderV2ProposalFinalization {
                    finalization: malformed,
                },
            )),
            Err(WorkerSessionError::Payload(_))
        ));
        assert!(
            session
                .handle(attempt_envelope(
                    4,
                    WorkerMessage::ProviderV2ProposalFinalization {
                        finalization: finalization.clone(),
                    },
                ))
                .expect("valid finalization reaches the retained worker proposal")
                .is_empty()
        );
        let acknowledgement = sim_types::WorkerProviderV2ProposalFinalizationAck {
            finalization: finalization.clone(),
            result: Ok(()),
        };
        let response = session
            .complete_provider_v2_finalization(acknowledgement.clone())
            .expect("exact acknowledgement completes the worker execution");
        assert_eq!(
            response.message,
            WorkerMessage::ProviderV2ProposalFinalizationAck { acknowledgement }
        );
        assert_eq!(session.lifecycle(), WorkerLifecycle::Ready);
        let duplicate = session
            .handle(attempt_envelope(
                5,
                WorkerMessage::ProviderV2ProposalFinalization {
                    finalization: finalization.clone(),
                },
            ))
            .expect("duplicate finalization returns a typed acknowledgement");
        assert!(matches!(
            duplicate.as_slice(),
            [WorkerEnvelope {
                message: WorkerMessage::ProviderV2ProposalFinalizationAck { acknowledgement },
                ..
            }] if acknowledgement.finalization == finalization
                && acknowledgement.result
                    == Err(sim_types::WorkerProviderStreamError::InvalidOrder)
        ));
    }

    #[test]
    fn provider_v2_cancellation_revokes_an_armed_proposal_before_terminal_result() {
        let mut session = ready_session();
        session.registry = Some(AssembledWorkerRegistry::empty_for_test(
            WorkerRegistryGeneration::new(1).expect("generation"),
            WorkerSha256Digest::new("c".repeat(64)).expect("registry digest"),
        ));
        session
            .handle(attempt_envelope(
                1,
                WorkerMessage::ExecutePlugin {
                    invocation: vec![1],
                },
            ))
            .expect("provider-v2 execution starts");
        session
            .mark_provider_v2_execution()
            .expect("decoded invocation selects the provider-v2 route");
        session
            .provider_v2_proposal(WorkerPluginExecutionOutcome::Succeeded(vec![1]))
            .expect("proposal remains armed");
        let cancellation = session
            .handle(attempt_envelope(
                2,
                WorkerMessage::Cancel {
                    reason: "test cancellation".to_owned(),
                },
            ))
            .expect("cancellation is observed");
        assert!(matches!(
            cancellation.as_slice(),
            [WorkerEnvelope {
                message: WorkerMessage::Lifecycle {
                    event: WorkerLifecycleEvent::CancellationRequested { .. }
                },
                ..
            }]
        ));
        let terminal = session
            .complete_plugin_execution(WorkerPluginExecutionOutcome::Failed(
                sim_types::WorkerPluginExecutionFailure::Cancelled,
            ))
            .expect("cancelled proposal converges without finalization");
        assert!(matches!(
            terminal.message,
            WorkerMessage::PluginResult {
                outcome: WorkerPluginExecutionOutcome::Failed(
                    sim_types::WorkerPluginExecutionFailure::Cancelled
                )
            }
        ));
        assert_eq!(session.lifecycle(), WorkerLifecycle::Ready);
        assert!(!session.provider_v2_proposal_pending);
    }

    #[test]
    fn protocol_skew_and_capability_mismatch_are_typed_failures() {
        let mut session = WorkerSession::new(1024).expect("session");
        let mut skewed = envelope(
            0,
            WorkerMessage::Hello {
                backend: cpu_backend_wire(),
            },
        );
        skewed.version = WORKER_PROTOCOL_VERSION + 1;
        assert!(matches!(
            session.handle(skewed),
            Err(WorkerSessionError::ProtocolVersion { .. })
        ));

        let mut session = WorkerSession::new(1024).expect("session");
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
        let invalid_backend = sim_types::WorkerBackendCapabilities::new(
            sim_types::DeviceKind::Cpu,
            0,
            vec![unary],
            vec![unsupported_resize],
        )
        .expect("wire representation is structurally valid");
        assert_eq!(
            session.handle(envelope(
                0,
                WorkerMessage::Hello {
                    backend: invalid_backend,
                }
            )),
            Err(WorkerSessionError::CapabilityMismatch)
        );

        let mut session = WorkerSession::new(1024).expect("session");
        let allocation = sim_types::WorkerOperationSupport::for_tensor_v2(
            sim_types::WorkerPrimitiveOperationV2::Allocation,
            sim_types::WorkerTensorRoleV1::Output,
            sim_types::WorkerDType::F32,
            sim_types::WorkerLayout::Contiguous,
        )
        .expect("allocation is a tensor primitive");
        let cuda_backend = sim_types::WorkerBackendCapabilities::new(
            sim_types::DeviceKind::Cuda,
            0,
            vec![allocation],
            vec![],
        )
        .expect("valid CUDA fixture");
        assert_eq!(
            session.handle(envelope(
                0,
                WorkerMessage::Hello {
                    backend: cuda_backend,
                }
            )),
            Err(WorkerSessionError::CapabilityMismatch)
        );
    }
}
