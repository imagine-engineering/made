use crate::assets::{AssetIdentity, NativeAssetResolverRegistry};
use crate::native_execution_controller::{
    NativeRuntimeModelCompileService, NativeRuntimeSdPoseService,
    NativeRuntimeTextGenerationService,
};
use crate::{
    AttemptEvent, AttemptEventKind, AttemptState, CacheEntry, CacheKey, CompiledNode, CompiledPlan,
    EventBusError, ExecutionEventBus, InputBinding, NativeCache, PromptCompileError,
    validate_native_provider_schemas,
};
use chrono::Utc;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_model::{
    NativeAudioEncoder, NativeAudioEncoderCheckpoint, NativeBackgroundRemovalCheckpoint,
    NativeBackgroundRemovalResource, NativeClipVision, NativeDepthAnything3Checkpoint,
    NativeDepthAnything3Resource, NativeFrameInterpolationModel, NativeGligenCheckpoint,
    NativeGligenResource, NativeLatentUpscaleCheckpoint, NativeLatentUpscaleModelResource,
    NativeModelPayload, NativeModelResourceRole, NativeMogeCheckpoint, NativeMogeResource,
    NativePhotoMakerCheckpoint, NativePhotoMakerCheckpointEntry, NativePhotoMakerResource,
    NativeStyleModelCheckpoint, NativeStyleModelResource, ParserLimits,
    clip::{
        NativeClipRecipeAuxiliaryArtifact, NativeClipRecipeAuxiliaryFormat, NativeClipRecipeSource,
        NativeClipResource,
    },
    model_family::{NativeFamilyModelResource, NativeModelRecipeSourceForm},
    vae::{NativeVaeRecipeResource, load_native_vae_recipe},
    vision_models::NativeRaftLarge,
};
use sim_nodes::execution::NativeModelResourcePreflight;
use sim_nodes::{
    NativeAssetReference, NativeAssetServiceError, NativeComponentH264Mp4BackingService,
    NativeHandleStore, NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType,
    NativeLtxvPreprocessService, NativeModelResourceRequest, NativeModelResourceService,
    NativeModelResourceServiceError, NativeModelTrainingService, NativeNodeBindingDisposition,
    NativeNodeComputeSession, NativeNodeContractError, NativeNodeServiceIdentity,
    NativeNodeServices, NativeOpaqueHandle, NativePayloadResidency, NativePreparedEffectKind,
    NativePreparedEffectService, NativeProviderExecutionIdentity, NativeResidentAllocationId,
    NativeResolvedPayload, NativeResolvedPayloadRetention, NativeStoredPayload,
    NativeStructuredValue, NativeValue, NativeWebmEncodeService, NodeRegistry,
};
pub use sim_nodes::{
    NativeCacheDependencies as CacheDependencies, NativeCachePolicy as RuntimeCachePolicy,
    NativeEffectClass as EffectClass, NativeNode, NativeNodeBinding,
    NativeNodeContext as NodeContext, NativeNodeDescriptor as RuntimeNodeDescriptor,
    NativeNodeFailure as NodeFailure, NativeNodeFailureKind as NodeFailureKind,
    NativeNodeOutcome as NodeOutcome, NativeNodePresentation as RuntimeNodePresentation,
    NativeOutputDescriptor as RuntimeOutputDescriptor, NativePortCardinality as InputMode,
    NativePreparedEffectRequest as PreparedEffectRequest,
};
#[cfg(test)]
use sim_nodes::{NativeOutputEffectRequest, NativeOutputNamespace, NativeOutputShape};
use sim_plugin_sdk::{CanonicalTypeId, ProviderBindingClaim, ProviderBindingSet};
use sim_tensor::{
    BackendCapabilityMatrix, CpuBackend, DType, DeviceId, ExecutionContext, NativeShaderExecutor,
    ScratchReservation, StreamId, Tensor, TensorBackend, TensorDescriptor,
};
#[cfg(test)]
use sim_tensor::{CpuWorkspaceAuthority, NativeDeviceProperties};
use sim_types::{
    AttemptId, CancellationToken, DeviceKind, NodeId, ProfileId, PromptId, PromptSubmission,
    WorkerModelSourceContext, WorkerModelSourceError, WorkerModelSourceFormat,
    WorkerModelSourceManifest, WorkerModelSourceNestedStateDisposition, WorkerModelSourceOperation,
    WorkerModelSourceRequest, WorkerModelSourceResponse, WorkerModelSourceTransportValidator,
    worker_model_source_selection_sha256,
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    sync::{
        Arc, Mutex as StdMutex, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use thiserror::Error;
use uuid::Uuid;

pub const MAX_EXPANSION_DEPTH: usize = 64;
pub const MAX_EFFECTS_PER_NODE: usize = 4_096;
pub const MAX_NATIVE_COMPILE_OPTIONS: usize = 64;
pub const MAX_NATIVE_COMPILE_TEXT_BYTES: usize = 4_096;
pub const MAX_RUNTIME_NATIVE_HANDLES: usize = 1_000_000;
pub const MAX_RUNTIME_NATIVE_HANDLE_BYTES: usize = 8 * 1024 * 1024 * 1024;

const MODEL_SOURCE_TRANSPORT_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[doc(hidden)]
pub struct NativeModelSourceTransportCall {
    call_id: u64,
    request: WorkerModelSourceRequest,
    validator: Arc<StdMutex<WorkerModelSourceTransportValidator>>,
    response_sender: async_channel::Sender<WorkerModelSourceResponse>,
}

impl NativeModelSourceTransportCall {
    pub const fn call_id(&self) -> u64 {
        self.call_id
    }

    pub fn request(&self) -> &WorkerModelSourceRequest {
        &self.request
    }

    pub async fn respond(
        self,
        call_id: u64,
        response: WorkerModelSourceResponse,
    ) -> Result<(), WorkerModelSourceError> {
        self.validator
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?
            .validate_response(call_id, &response)?;
        self.response_sender
            .send(response)
            .await
            .map_err(|_| WorkerModelSourceError::Closed)
    }

    pub fn revoke(&self) -> Result<(), WorkerModelSourceError> {
        self.validator
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?
            .revoke();
        Ok(())
    }
}

#[doc(hidden)]
pub struct NativeModelSourceTransportHost {
    receiver: async_channel::Receiver<NativeModelSourceTransportCall>,
}

impl NativeModelSourceTransportHost {
    pub async fn receive(
        &self,
    ) -> Result<NativeModelSourceTransportCall, async_channel::RecvError> {
        self.receiver.recv().await
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub(crate) struct NativeModelSourceTransport {
    sender: async_channel::Sender<NativeModelSourceTransportCall>,
    next_call_id: Arc<AtomicU64>,
}

impl NativeModelSourceTransport {
    pub(crate) fn channel() -> (Self, NativeModelSourceTransportHost) {
        let (sender, receiver) = async_channel::bounded(1);
        (
            Self {
                sender,
                next_call_id: Arc::new(AtomicU64::new(1)),
            },
            NativeModelSourceTransportHost { receiver },
        )
    }

    fn open_session(
        &self,
        context: WorkerModelSourceContext,
    ) -> Result<NativeModelSourceTransportSession, WorkerModelSourceError> {
        Ok(NativeModelSourceTransportSession {
            transport: self.clone(),
            validator: Arc::new(StdMutex::new(WorkerModelSourceTransportValidator::checked(
                context,
            )?)),
            next_call_ordinal: AtomicU64::new(1),
            opened: AtomicBool::new(false),
        })
    }
}

#[doc(hidden)]
struct NativeModelSourceTransportSession {
    transport: NativeModelSourceTransport,
    validator: Arc<StdMutex<WorkerModelSourceTransportValidator>>,
    next_call_ordinal: AtomicU64,
    opened: AtomicBool,
}

impl NativeModelSourceTransportSession {
    #[allow(
        clippy::disallowed_methods,
        reason = "the private native model loader runs on a blocking execution thread and cooperatively polls its capacity-one transport"
    )]
    fn call(
        &self,
        request: WorkerModelSourceRequest,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        cancellation
            .check()
            .map_err(|_| WorkerModelSourceError::Cancelled)?;
        let call_id = self.transport.next_call_id.fetch_add(1, Ordering::Relaxed);
        if call_id == 0 {
            self.revoke()?;
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        self.validator
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?
            .validate_request(call_id, &request)?;
        let request_call_ordinal = request.call_ordinal;
        let request_is_open = matches!(&request.operation, WorkerModelSourceOperation::Open { .. });
        let (response_sender, response_receiver) = async_channel::bounded(1);
        let mut call = NativeModelSourceTransportCall {
            call_id,
            request,
            validator: self.validator.clone(),
            response_sender,
        };
        loop {
            if cancellation.is_cancelled() {
                self.revoke()?;
                return Err(WorkerModelSourceError::Cancelled);
            }
            match self.transport.sender.try_send(call) {
                Ok(()) => break,
                Err(async_channel::TrySendError::Full(returned)) => {
                    call = returned;
                    std::thread::sleep(MODEL_SOURCE_TRANSPORT_POLL_INTERVAL);
                }
                Err(async_channel::TrySendError::Closed(_)) => {
                    self.revoke()?;
                    return Err(WorkerModelSourceError::Closed);
                }
            }
        }
        loop {
            if cancellation.is_cancelled() {
                self.revoke()?;
                return Err(WorkerModelSourceError::Cancelled);
            }
            match response_receiver.try_recv() {
                Ok(response) => {
                    let next_call_ordinal = request_call_ordinal
                        .checked_add(1)
                        .ok_or(WorkerModelSourceError::InvalidOrder)?;
                    self.next_call_ordinal
                        .store(next_call_ordinal, Ordering::Release);
                    if request_is_open && matches!(&response, WorkerModelSourceResponse::Opened(_))
                    {
                        self.opened.store(true, Ordering::Release);
                    }
                    if matches!(
                        &response,
                        WorkerModelSourceResponse::Closed(_)
                            | WorkerModelSourceResponse::Rejected { .. }
                    ) {
                        self.opened.store(false, Ordering::Release);
                    }
                    return Ok(response);
                }
                Err(async_channel::TryRecvError::Empty) => {
                    std::thread::sleep(MODEL_SOURCE_TRANSPORT_POLL_INTERVAL);
                }
                Err(async_channel::TryRecvError::Closed) => {
                    self.revoke()?;
                    return Err(WorkerModelSourceError::Closed);
                }
            }
        }
    }

    fn revoke(&self) -> Result<(), WorkerModelSourceError> {
        self.opened.store(false, Ordering::Release);
        self.validator
            .lock()
            .map_err(|_| WorkerModelSourceError::HostFailure)?
            .revoke();
        Ok(())
    }

    fn has_open_session(&self) -> bool {
        self.opened.load(Ordering::Acquire)
    }

    fn next_call_ordinal(&self) -> u64 {
        self.next_call_ordinal.load(Ordering::Acquire)
    }
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub struct NativeModelSourceTestTransport {
    inner: NativeModelSourceTransport,
}

#[cfg(feature = "test-support")]
impl NativeModelSourceTestTransport {
    pub fn channel() -> (Self, NativeModelSourceTransportHost) {
        let (inner, host) = NativeModelSourceTransport::channel();
        (Self { inner }, host)
    }

    pub fn open_session(
        &self,
        context: WorkerModelSourceContext,
    ) -> Result<NativeModelSourceTestSession, WorkerModelSourceError> {
        Ok(NativeModelSourceTestSession {
            inner: self.inner.open_session(context)?,
        })
    }
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub struct NativeModelSourceTestSession {
    inner: NativeModelSourceTransportSession,
}

#[cfg(feature = "test-support")]
impl NativeModelSourceTestSession {
    pub fn call(
        &self,
        request: WorkerModelSourceRequest,
        cancellation: &CancellationToken,
    ) -> Result<WorkerModelSourceResponse, WorkerModelSourceError> {
        self.inner.call(request, cancellation)
    }
}

#[derive(Clone, Copy, Debug, Default)]
enum NativeModelConstructionProfile {
    #[default]
    SourceExact,
    #[cfg(feature = "test-support")]
    ReducedFixture,
}

#[derive(Clone)]
struct NativeModelResourceServiceFactory {
    binding: crate::NativeModelSourceWorkerBinding,
    transport: NativeModelSourceTransport,
    backend: Arc<CpuBackend>,
    construction_profile: NativeModelConstructionProfile,
}

impl NativeModelResourceServiceFactory {
    fn node_service(
        &self,
        identity: NativeNodeServiceIdentity,
    ) -> Arc<dyn NativeModelResourceService> {
        Arc::new(RuntimeNativeModelResourceService {
            expected_reconstruction: self
                .binding
                .model_resource_reconstruction()
                .get(identity.node_id())
                .cloned()
                .unwrap_or_default(),
            identity,
            binding: self.binding.clone(),
            transport: self.transport.clone(),
            backend: self.backend.clone(),
            construction_profile: self.construction_profile,
            pending_preflight: Mutex::new(VecDeque::new()),
        })
    }
}

struct RuntimeNativeModelResourceService {
    expected_reconstruction: Vec<crate::persistence::PersistedNativeModelResourceEvidence>,
    identity: NativeNodeServiceIdentity,
    binding: crate::NativeModelSourceWorkerBinding,
    transport: NativeModelSourceTransport,
    backend: Arc<CpuBackend>,
    construction_profile: NativeModelConstructionProfile,
    pending_preflight: Mutex<VecDeque<NativeModelResourcePreflight>>,
}

impl fmt::Debug for RuntimeNativeModelResourceService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeNativeModelResourceService")
            .field("identity", &self.identity)
            .field("binding", &self.binding)
            .field("construction_profile", &self.construction_profile)
            .finish_non_exhaustive()
    }
}

impl NativeModelResourceService for RuntimeNativeModelResourceService {
    fn identity(&self) -> &NativeNodeServiceIdentity {
        &self.identity
    }

    fn preflight(
        &self,
        context: &NodeContext,
        request: &NativeModelResourceRequest,
    ) -> Result<NativeModelResourcePreflight, NativeModelResourceServiceError> {
        validate_available_model_resource_recipe(request)?;
        let session_context = self.session_context(context, request)?;
        let session = self
            .transport
            .open_session(session_context.clone())
            .map_err(map_model_source_error)?;
        let opened = open_native_model_source_session(
            &session,
            &session_context,
            request,
            &context.cancellation,
        );
        let closed = close_native_model_source_session(&session, &session_context);
        let evidence = match (opened, closed) {
            (Ok(opened), Ok(())) => model_resource_preflight(request, &opened)?,
            (Ok(_), Err(error)) | (Err(error), _) => return Err(error),
        };
        self.validate_expected_preflight(request, &evidence)?;
        self.pending_preflight.lock().push_back(evidence.clone());
        Ok(evidence)
    }

    fn load(
        &self,
        context: &NodeContext,
        request: &NativeModelResourceRequest,
    ) -> Result<Arc<NativeModelPayload>, NativeModelResourceServiceError> {
        validate_available_model_resource_recipe(request)?;
        context
            .cancellation
            .check()
            .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
        let session_context = self.session_context(context, request)?;
        let session = self
            .transport
            .open_session(session_context.clone())
            .map_err(map_model_source_error)?;
        let pending_preflight = self.pending_preflight.lock().pop_front();
        let loaded = load_native_model_resource_from_session(
            &session,
            session_context.clone(),
            request,
            &self.backend,
            context,
            self.construction_profile,
            pending_preflight.as_ref(),
        );
        let closed = close_native_model_source_session(&session, &session_context);
        match (loaded, closed) {
            (Ok(payload), Ok(())) => {
                context
                    .cancellation
                    .check()
                    .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
                self.validate_expected_resource(
                    request,
                    pending_preflight.as_ref(),
                    payload.payload.identity(),
                )?;
                Ok(payload.payload)
            }
            (Ok(_), Err(error)) => Err(error),
            (Err(error), _) => Err(error),
        }
    }
}

impl RuntimeNativeModelResourceService {
    fn expected_reconstruction(
        &self,
        request: &NativeModelResourceRequest,
    ) -> Result<
        Option<&crate::persistence::PersistedNativeModelResourceEvidence>,
        NativeModelResourceServiceError,
    > {
        if self.expected_reconstruction.is_empty() {
            return Ok(None);
        }
        let mut matching = self.expected_reconstruction.iter().filter(|evidence| {
            evidence.preflight.role() == request.role()
                && evidence.preflight.folder_category() == request.folder_category()
                && evidence.preflight.source_names() == request.source_names()
        });
        let expected = matching
            .next()
            .ok_or(NativeModelResourceServiceError::SourceChanged)?;
        if matching.next().is_some() {
            return Err(NativeModelResourceServiceError::SourceChanged);
        }
        Ok(Some(expected))
    }

    fn validate_expected_preflight(
        &self,
        request: &NativeModelResourceRequest,
        observed: &NativeModelResourcePreflight,
    ) -> Result<(), NativeModelResourceServiceError> {
        if self
            .expected_reconstruction(request)?
            .is_some_and(|expected| expected.preflight != *observed)
        {
            return Err(NativeModelResourceServiceError::SourceChanged);
        }
        Ok(())
    }

    fn validate_expected_resource(
        &self,
        request: &NativeModelResourceRequest,
        observed_preflight: Option<&NativeModelResourcePreflight>,
        observed_identity: &sim_model::NativeModelResourceIdentity,
    ) -> Result<(), NativeModelResourceServiceError> {
        let Some(expected) = self.expected_reconstruction(request)? else {
            return Ok(());
        };
        if observed_preflight != Some(&expected.preflight)
            || observed_identity != &expected.resource_identity
        {
            return Err(NativeModelResourceServiceError::SourceChanged);
        }
        Ok(())
    }

    fn session_context(
        &self,
        context: &NodeContext,
        request: &NativeModelResourceRequest,
    ) -> Result<WorkerModelSourceContext, NativeModelResourceServiceError> {
        let ordered_source_identity_sha256 =
            worker_model_source_selection_sha256(request.folder_category(), request.source_names())
                .map_err(map_model_source_error)?;
        Ok(WorkerModelSourceContext {
            session_id: Uuid::new_v4(),
            attempt_id: context.attempt_id,
            attempt_generation: self.binding.attempt_generation(),
            node_id: context.node_id.0.clone(),
            node_generation: self.binding.node_generation(),
            service_id: self.binding.service_id(),
            service_generation: self.binding.service_generation(),
            ordered_source_identity_sha256,
        })
    }
}

fn validate_available_model_resource_recipe(
    request: &NativeModelResourceRequest,
) -> Result<(), NativeModelResourceServiceError> {
    request.validate_recipe()?;
    if matches!(
        request.role(),
        NativeModelResourceRole::ControlNet
            | NativeModelResourceRole::FaceDetection
            | NativeModelResourceRole::Hooks
            | NativeModelResourceRole::HookKeyframes
            | NativeModelResourceRole::LatentOperation
            | NativeModelResourceRole::LoraModel
            | NativeModelResourceRole::ModelPatch
            | NativeModelResourceRole::UpscaleModel
    ) {
        return Err(NativeModelResourceServiceError::Unavailable);
    }
    Ok(())
}

struct LoadedNativeModelPayload {
    payload: Arc<NativeModelPayload>,
}

struct LoadedNativeModelSource {
    manifest: WorkerModelSourceManifest,
    ordered_state: Vec<(String, Tensor)>,
    auxiliary_artifacts: Vec<LoadedNativeModelAuxiliaryArtifact>,
}

struct LoadedNativeModelAuxiliaryArtifact {
    format: WorkerModelSourceFormat,
    artifact_sha256: String,
    bytes: Vec<u8>,
}

fn close_native_model_source_session(
    session: &NativeModelSourceTransportSession,
    session_context: &WorkerModelSourceContext,
) -> Result<(), NativeModelResourceServiceError> {
    if !session.has_open_session() {
        return session.revoke().map_err(map_model_source_error);
    }
    let close_ordinal = session.next_call_ordinal();
    let cancellation = CancellationToken::default();
    let response = match session.call(
        WorkerModelSourceRequest {
            context: session_context.clone(),
            call_ordinal: close_ordinal,
            operation: WorkerModelSourceOperation::Close,
        },
        &cancellation,
    ) {
        Ok(response) => response,
        Err(error) => {
            session.revoke().map_err(map_model_source_error)?;
            return Err(map_model_source_error(error));
        }
    };
    response.validate().map_err(map_model_source_error)?;
    match response {
        WorkerModelSourceResponse::Closed(closed)
            if closed.session_id == session_context.session_id
                && closed.call_ordinal == close_ordinal =>
        {
            Ok(())
        }
        WorkerModelSourceResponse::Rejected { error, .. } => Err(map_model_source_error(error)),
        _ => Err(NativeModelResourceServiceError::MalformedSource(
            "close response does not match its request",
        )),
    }
}

fn open_native_model_source_session(
    session: &NativeModelSourceTransportSession,
    session_context: &WorkerModelSourceContext,
    request: &NativeModelResourceRequest,
    cancellation: &CancellationToken,
) -> Result<sim_types::WorkerModelSourceOpened, NativeModelResourceServiceError> {
    let open = session
        .call(
            WorkerModelSourceRequest {
                context: session_context.clone(),
                call_ordinal: 1,
                operation: WorkerModelSourceOperation::Open {
                    folder_category: request.folder_category().to_owned(),
                    source_names: request.source_names().to_vec(),
                },
            },
            cancellation,
        )
        .map_err(map_model_source_error)?;
    open.validate().map_err(map_model_source_error)?;
    match open {
        WorkerModelSourceResponse::Opened(opened)
            if opened.session_id == session_context.session_id
                && opened.call_ordinal == 1
                && opened.ordered_source_identity_sha256
                    == session_context.ordered_source_identity_sha256
                && opened.sources.len() == request.source_names().len() =>
        {
            Ok(opened)
        }
        WorkerModelSourceResponse::Rejected { error, .. } => Err(map_model_source_error(error)),
        _ => Err(NativeModelResourceServiceError::MalformedSource(
            "open response does not match its request",
        )),
    }
}

pub(crate) fn model_resource_preflight(
    request: &NativeModelResourceRequest,
    opened: &sim_types::WorkerModelSourceOpened,
) -> Result<NativeModelResourcePreflight, NativeModelResourceServiceError> {
    let encoded = postcard::to_stdvec(&opened.sources).map_err(|_| {
        NativeModelResourceServiceError::MalformedSource(
            "source manifest evidence could not be encoded",
        )
    })?;
    let mut digest = Sha256::new();
    digest.update(b"zed.sim.native-model-resource-manifest.v1");
    digest.update(&encoded);
    NativeModelResourcePreflight::checked(
        request,
        opened.ordered_source_identity_sha256.as_str(),
        opened
            .sources
            .iter()
            .map(|source| source.model_identity_sha256.as_str().to_owned())
            .collect(),
        format!("{:x}", digest.finalize()),
    )
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub fn persisted_model_resource_evidence_for_test(
    node_id: NodeId,
    request: &NativeModelResourceRequest,
    opened: &sim_types::WorkerModelSourceOpened,
    resource_identity: sim_model::NativeModelResourceIdentity,
) -> Result<crate::persistence::PersistedNativeModelResourceEvidence, NativeModelResourceServiceError>
{
    crate::persistence::PersistedNativeModelResourceEvidence::checked(
        node_id,
        model_resource_preflight(request, opened)?,
        resource_identity,
    )
    .map_err(|_| NativeModelResourceServiceError::SourceChanged)
}

fn load_native_model_resource_from_session(
    session: &NativeModelSourceTransportSession,
    session_context: WorkerModelSourceContext,
    request: &NativeModelResourceRequest,
    backend: &CpuBackend,
    node_context: &NodeContext,
    construction_profile: NativeModelConstructionProfile,
    expected_preflight: Option<&NativeModelResourcePreflight>,
) -> Result<LoadedNativeModelPayload, NativeModelResourceServiceError> {
    let opened = open_native_model_source_session(
        session,
        &session_context,
        request,
        &node_context.cancellation,
    )?;
    let observed_preflight = model_resource_preflight(request, &opened)?;
    if expected_preflight.is_some_and(|expected| expected != &observed_preflight) {
        return Err(NativeModelResourceServiceError::SourceChanged);
    }
    let mut aggregate_tensor_bytes = 0_u64;
    for (source_ordinal, manifest) in opened.sources.iter().enumerate() {
        if usize::try_from(manifest.source_ordinal).ok() != Some(source_ordinal)
            || manifest.tensors.is_empty()
            || opened.maximum_chunk_bytes == 0
        {
            return Err(NativeModelResourceServiceError::MalformedSource(
                "source manifest has an invalid ordinal, tensor set, or chunk bound",
            ));
        }
        aggregate_tensor_bytes = aggregate_tensor_bytes
            .checked_add(manifest.aggregate_tensor_bytes)
            .ok_or(NativeModelResourceServiceError::OutOfMemory)?;
        for artifact in manifest
            .artifacts
            .iter()
            .filter(|artifact| artifact.format.supports_auxiliary_artifact_stream())
        {
            aggregate_tensor_bytes = aggregate_tensor_bytes
                .checked_add(artifact.byte_size)
                .ok_or(NativeModelResourceServiceError::OutOfMemory)?;
        }
    }
    if aggregate_tensor_bytes > node_context.scratch.bytes() {
        return Err(NativeModelResourceServiceError::OutOfMemory);
    }
    let tensor_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: node_context.scratch.clone(),
        rng_phase: None,
        cancellation: &node_context.cancellation,
    };
    let mut call_ordinal = 2_u64;
    let mut loaded_sources = Vec::new();
    loaded_sources
        .try_reserve_exact(opened.sources.len())
        .map_err(|_| NativeModelResourceServiceError::OutOfMemory)?;
    for manifest in opened.sources {
        let ordered_state = load_manifest_tensors(
            session,
            &session_context,
            &manifest,
            opened.maximum_chunk_bytes,
            backend,
            &tensor_context,
            &mut call_ordinal,
        )?;
        let auxiliary_artifacts = load_manifest_auxiliary_artifacts(
            session,
            &session_context,
            &manifest,
            opened.maximum_chunk_bytes,
            &node_context.cancellation,
            &mut call_ordinal,
        )?;
        loaded_sources.push(LoadedNativeModelSource {
            manifest,
            ordered_state,
            auxiliary_artifacts,
        });
    }
    if request.role() == NativeModelResourceRole::Clip {
        let source_form = if request.folder_category() == "checkpoints" {
            NativeModelRecipeSourceForm::Checkpoint
        } else {
            NativeModelRecipeSourceForm::Standalone
        };
        let mut clip_sources = Vec::new();
        clip_sources
            .try_reserve_exact(loaded_sources.len())
            .map_err(|_| NativeModelResourceServiceError::OutOfMemory)?;
        for source in loaded_sources {
            let mut auxiliary_artifacts = Vec::new();
            for artifact in &source.auxiliary_artifacts {
                let (format, target_ordinal) = match artifact.format {
                    WorkerModelSourceFormat::JsonConfig => {
                        (NativeClipRecipeAuxiliaryFormat::JsonConfiguration, None)
                    }
                    WorkerModelSourceFormat::JsonTokenizer => {
                        (NativeClipRecipeAuxiliaryFormat::JsonTokenizer, None)
                    }
                    WorkerModelSourceFormat::SentencePiece => {
                        (NativeClipRecipeAuxiliaryFormat::SentencePiece, None)
                    }
                    WorkerModelSourceFormat::Tiktoken => {
                        (NativeClipRecipeAuxiliaryFormat::Tiktoken, None)
                    }
                    WorkerModelSourceFormat::ClipConfiguration { target_ordinal } => (
                        NativeClipRecipeAuxiliaryFormat::JsonConfiguration,
                        Some(target_ordinal),
                    ),
                    WorkerModelSourceFormat::ClipTokenizerJson { target_ordinal } => (
                        NativeClipRecipeAuxiliaryFormat::JsonTokenizer,
                        Some(target_ordinal),
                    ),
                    WorkerModelSourceFormat::ClipSentencePiece { target_ordinal } => (
                        NativeClipRecipeAuxiliaryFormat::SentencePiece,
                        Some(target_ordinal),
                    ),
                    WorkerModelSourceFormat::ClipTiktoken { target_ordinal } => (
                        NativeClipRecipeAuxiliaryFormat::Tiktoken,
                        Some(target_ordinal),
                    ),
                    WorkerModelSourceFormat::SafetensorsMetadata { .. }
                    | WorkerModelSourceFormat::YamlConfig
                    | WorkerModelSourceFormat::Safetensors
                    | WorkerModelSourceFormat::PytorchArchive
                    | WorkerModelSourceFormat::Gguf => continue,
                };
                auxiliary_artifacts.push(NativeClipRecipeAuxiliaryArtifact {
                    format,
                    target_ordinal,
                    artifact_sha256: artifact.artifact_sha256.clone(),
                    bytes: artifact.bytes.clone(),
                });
            }
            clip_sources.push(NativeClipRecipeSource {
                source_form,
                artifact_sha256: source.manifest.model_identity_sha256.as_str().to_owned(),
                ordered_state: source.ordered_state,
                parser_limits: ParserLimits {
                    version: source.manifest.parser_limits.version,
                    manifest_bytes: source.manifest.parser_limits.manifest_bytes,
                    maximum_depth: source.manifest.parser_limits.maximum_depth,
                    maximum_tensors: source.manifest.parser_limits.maximum_tensors,
                    maximum_tensor_bytes: source.manifest.parser_limits.maximum_tensor_bytes,
                    maximum_aggregate_tensor_bytes: source
                        .manifest
                        .parser_limits
                        .maximum_aggregate_tensor_bytes,
                    maximum_name_bytes: source.manifest.parser_limits.maximum_name_bytes,
                    maximum_archive_entries: source.manifest.parser_limits.maximum_archive_entries,
                    maximum_metadata_values: source.manifest.parser_limits.maximum_metadata_values,
                },
                auxiliary_artifacts,
            });
        }
        let resource =
            NativeClipResource::from_recipe_sources(clip_sources, backend, &tensor_context)?;
        return Ok(LoadedNativeModelPayload {
            payload: Arc::new(NativeModelPayload::native_clip(Arc::new(resource))?),
        });
    }
    let mut loaded_sources = loaded_sources.into_iter();
    let source = loaded_sources
        .next()
        .ok_or(NativeModelResourceServiceError::MalformedSource(
            "opened response omitted its source manifest",
        ))?;
    let manifest = &source.manifest;
    let artifact_sha256 = manifest.model_identity_sha256.as_str().to_owned();
    let ordered_state = source.ordered_state;
    let auxiliary_artifacts = source.auxiliary_artifacts;
    let memory_budget_bytes = node_context.scratch.bytes();
    let payload = match request.role() {
        NativeModelResourceRole::Model => {
            let source_form = if request.folder_category() == "checkpoints" {
                NativeModelRecipeSourceForm::Checkpoint
            } else {
                NativeModelRecipeSourceForm::Standalone
            };
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeFamilyModelResource::from_recipe_state(
                        source_form,
                        artifact_sha256,
                        ordered_state,
                        memory_budget_bytes,
                        backend,
                        &tensor_context,
                    )?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeFamilyModelResource::from_reduced_recipe_state(
                        source_form,
                        artifact_sha256,
                        ordered_state,
                        memory_budget_bytes,
                        backend,
                        &tensor_context,
                    )?
                }
            };
            NativeModelPayload::native_family_model(Arc::new(resource))?
        }
        NativeModelResourceRole::Vae => {
            let source_form = if request.folder_category() == "checkpoints" {
                NativeModelRecipeSourceForm::Checkpoint
            } else {
                NativeModelRecipeSourceForm::Standalone
            };
            let parser_limits = model_source_parser_limits(manifest);
            let mut explicit_configuration = None;
            for artifact in &auxiliary_artifacts {
                let configuration = match artifact.format {
                    WorkerModelSourceFormat::JsonConfig => {
                        sim_model::formats::checked_vae_configuration_json(
                            &artifact.bytes,
                            &artifact.artifact_sha256,
                            &parser_limits,
                            &node_context.cancellation,
                        )
                        .map_err(|_| {
                            NativeModelResourceServiceError::MalformedSource(
                                "VAE semantic configuration is invalid",
                            )
                        })?
                    }
                    WorkerModelSourceFormat::SafetensorsMetadata { .. } => {
                        sim_model::formats::checked_safetensors_metadata_json(
                            &artifact.bytes,
                            &artifact.artifact_sha256,
                            &parser_limits,
                            &node_context.cancellation,
                        )
                        .map_err(|_| {
                            NativeModelResourceServiceError::MalformedSource(
                                "VAE safetensors metadata is invalid",
                            )
                        })?
                        .remove("config")
                    }
                    _ => None,
                };
                if let Some(configuration) = configuration {
                    if explicit_configuration.replace(configuration).is_some() {
                        return Err(NativeModelResourceServiceError::MalformedSource(
                            "VAE source has multiple semantic configurations",
                        ));
                    }
                }
            }
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => load_native_vae_recipe(
                    source_form,
                    &artifact_sha256,
                    ordered_state,
                    explicit_configuration.as_deref(),
                    backend,
                    &tensor_context,
                )?,
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    sim_model::vae::load_native_vae_reduced_recipe(
                        source_form,
                        &artifact_sha256,
                        ordered_state,
                        explicit_configuration.as_deref(),
                        backend,
                        &tensor_context,
                    )?
                }
            };
            match resource {
                NativeVaeRecipeResource::Native(resource) => {
                    NativeModelPayload::native_vae(Arc::new(resource))?
                }
                NativeVaeRecipeResource::Structured(resource) => {
                    NativeModelPayload::native_structured_vae(Arc::new(resource))?
                }
            }
        }
        NativeModelResourceRole::ClipVision => {
            let parser_limits = model_source_parser_limits(manifest);
            let configurations = auxiliary_artifacts
                .iter()
                .filter(|artifact| artifact.format == WorkerModelSourceFormat::JsonConfig)
                .map(
                    |artifact| sim_model::clip_vision::NativeClipVisionRecipeConfiguration {
                        artifact_sha256: artifact.artifact_sha256.clone(),
                        bytes: artifact.bytes.clone(),
                        parser_limits: parser_limits.clone(),
                    },
                )
                .collect();
            let resource = NativeClipVision::from_recipe_state(
                NativeModelRecipeSourceForm::Standalone,
                configurations,
                ordered_state,
                &node_context.cancellation,
            )?;
            NativeModelPayload::clip_vision(Arc::new(resource))?
        }
        NativeModelResourceRole::AudioEncoder => {
            let resource = NativeAudioEncoder::from_checkpoint(
                NativeAudioEncoderCheckpoint {
                    artifact_sha256,
                    ordered_state,
                    memory_budget_bytes,
                },
                &tensor_context,
            )?;
            NativeModelPayload::audio_encoder(Arc::new(resource))?
        }
        NativeModelResourceRole::BackgroundRemoval => {
            let checkpoint = NativeBackgroundRemovalCheckpoint {
                artifact_sha256,
                ordered_state,
                memory_budget_bytes,
            };
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeBackgroundRemovalResource::from_checkpoint(checkpoint, &tensor_context)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeBackgroundRemovalResource::from_reduced_fixture(
                        checkpoint,
                        &tensor_context,
                    )?
                }
            };
            match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeModelPayload::background_removal(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeModelPayload::background_removal_test_fixture(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
            }
        }
        NativeModelResourceRole::Da3Model => {
            let checkpoint = NativeDepthAnything3Checkpoint {
                artifact_sha256,
                ordered_state,
                memory_budget_bytes,
            };
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeDepthAnything3Resource::from_checkpoint(
                        backend,
                        checkpoint,
                        &tensor_context,
                    )?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeDepthAnything3Resource::from_reduced_fixture(
                        backend,
                        checkpoint,
                        &tensor_context,
                    )?
                }
            };
            match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeModelPayload::depth_anything_3(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeModelPayload::depth_anything_3_test_fixture(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
            }
        }
        NativeModelResourceRole::FrameInterpolation => {
            let state = checked_model_resource_state(ordered_state)?;
            let resource = NativeFrameInterpolationModel::from_checkpoint(
                artifact_sha256,
                state,
                &node_context.cancellation,
            )?;
            NativeModelPayload::frame_interpolation(Arc::new(resource))?
        }
        NativeModelResourceRole::LatentUpscaleModel => {
            let parser_limits = model_source_parser_limits(manifest);
            let mut metadata = BTreeMap::new();
            for artifact in &auxiliary_artifacts {
                let configuration = match artifact.format {
                    WorkerModelSourceFormat::JsonConfig => {
                        sim_model::formats::checked_latent_upscale_configuration_json(
                            &artifact.bytes,
                            &artifact.artifact_sha256,
                            &parser_limits,
                            &node_context.cancellation,
                        )
                        .map_err(|_| {
                            NativeModelResourceServiceError::MalformedSource(
                                "latent upscale configuration is invalid",
                            )
                        })?
                    }
                    WorkerModelSourceFormat::SafetensorsMetadata { .. } => {
                        sim_model::formats::checked_safetensors_metadata_json(
                            &artifact.bytes,
                            &artifact.artifact_sha256,
                            &parser_limits,
                            &node_context.cancellation,
                        )
                        .map_err(|_| {
                            NativeModelResourceServiceError::MalformedSource(
                                "latent upscale safetensors metadata is invalid",
                            )
                        })?
                        .remove("config")
                    }
                    _ => None,
                };
                if let Some(configuration) = configuration {
                    if metadata
                        .insert("config".to_owned(), configuration)
                        .is_some()
                    {
                        return Err(NativeModelResourceServiceError::MalformedSource(
                            "latent upscale source has multiple semantic configurations",
                        ));
                    }
                }
            }
            let resource = NativeLatentUpscaleModelResource::from_checkpoint(
                NativeLatentUpscaleCheckpoint {
                    artifact_sha256,
                    metadata,
                    ordered_state,
                    memory_budget_bytes,
                },
                &tensor_context,
            )?;
            NativeModelPayload::latent_upscale_model(Arc::new(resource))?
        }
        NativeModelResourceRole::MogeModel => {
            let checkpoint = NativeMogeCheckpoint {
                artifact_sha256,
                ordered_state,
                memory_budget_bytes,
            };
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeMogeResource::from_checkpoint(backend, checkpoint, &tensor_context)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeMogeResource::from_reduced_fixture(backend, checkpoint, &tensor_context)?
                }
            };
            match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeModelPayload::moge(Arc::new(resource), &node_context.cancellation)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeModelPayload::moge_test_fixture(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
            }
        }
        NativeModelResourceRole::OpticalFlow => {
            let resource =
                NativeRaftLarge::from_recipe_state(ordered_state, &node_context.cancellation)?;
            NativeModelPayload::optical_flow(Arc::new(resource))?
        }
        NativeModelResourceRole::StyleModel => {
            let checkpoint = NativeStyleModelCheckpoint {
                artifact_sha256,
                ordered_state,
                memory_budget_bytes,
            };
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeStyleModelResource::from_checkpoint(backend, checkpoint, &tensor_context)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeStyleModelResource::from_reduced_fixture(
                        backend,
                        checkpoint,
                        &tensor_context,
                    )?
                }
            };
            match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeModelPayload::style_model(Arc::new(resource), &node_context.cancellation)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeModelPayload::style_model_test_fixture(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
            }
        }
        NativeModelResourceRole::Photomaker => {
            let ordered_entries = photomaker_entries(manifest, ordered_state)?;
            let checkpoint = NativePhotoMakerCheckpoint {
                artifact_sha256,
                ordered_entries,
                memory_budget_bytes,
            };
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativePhotoMakerResource::from_checkpoint(backend, checkpoint, &tensor_context)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativePhotoMakerResource::from_reduced_fixture(
                        backend,
                        checkpoint,
                        &tensor_context,
                    )?
                }
            };
            match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeModelPayload::photomaker(Arc::new(resource), &node_context.cancellation)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeModelPayload::photomaker_test_fixture(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
            }
        }
        NativeModelResourceRole::Gligen => {
            let checkpoint = NativeGligenCheckpoint {
                artifact_sha256,
                ordered_state,
                memory_budget_bytes,
            };
            let resource = match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeGligenResource::from_checkpoint(backend, checkpoint, &tensor_context)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeGligenResource::from_reduced_fixture(
                        backend,
                        checkpoint,
                        &tensor_context,
                    )?
                }
            };
            match construction_profile {
                NativeModelConstructionProfile::SourceExact => {
                    NativeModelPayload::gligen(Arc::new(resource), &node_context.cancellation)?
                }
                #[cfg(feature = "test-support")]
                NativeModelConstructionProfile::ReducedFixture => {
                    NativeModelPayload::gligen_test_fixture(
                        Arc::new(resource),
                        &node_context.cancellation,
                    )?
                }
            }
        }
        NativeModelResourceRole::Clip => {
            return Err(NativeModelResourceServiceError::Unavailable);
        }
        NativeModelResourceRole::ControlNet
        | NativeModelResourceRole::FaceDetection
        | NativeModelResourceRole::Hooks
        | NativeModelResourceRole::HookKeyframes
        | NativeModelResourceRole::LatentOperation
        | NativeModelResourceRole::LoraModel
        | NativeModelResourceRole::ModelPatch
        | NativeModelResourceRole::UpscaleModel => {
            return Err(NativeModelResourceServiceError::Unavailable);
        }
    };
    Ok(LoadedNativeModelPayload {
        payload: Arc::new(payload),
    })
}

fn checked_model_resource_state(
    ordered_state: Vec<(String, Tensor)>,
) -> Result<BTreeMap<String, Tensor>, NativeModelResourceServiceError> {
    let mut state = BTreeMap::new();
    for (name, tensor) in ordered_state {
        if state.insert(name.clone(), tensor).is_some() {
            return Err(NativeModelResourceServiceError::MalformedSource(
                "model source contains a duplicate tensor name",
            ));
        }
    }
    Ok(state)
}

fn model_source_parser_limits(manifest: &WorkerModelSourceManifest) -> ParserLimits {
    ParserLimits {
        version: manifest.parser_limits.version,
        manifest_bytes: manifest.parser_limits.manifest_bytes,
        maximum_depth: manifest.parser_limits.maximum_depth,
        maximum_tensors: manifest.parser_limits.maximum_tensors,
        maximum_tensor_bytes: manifest.parser_limits.maximum_tensor_bytes,
        maximum_aggregate_tensor_bytes: manifest.parser_limits.maximum_aggregate_tensor_bytes,
        maximum_name_bytes: manifest.parser_limits.maximum_name_bytes,
        maximum_archive_entries: manifest.parser_limits.maximum_archive_entries,
        maximum_metadata_values: manifest.parser_limits.maximum_metadata_values,
    }
}

fn load_manifest_auxiliary_artifacts(
    session: &NativeModelSourceTransportSession,
    session_context: &WorkerModelSourceContext,
    manifest: &WorkerModelSourceManifest,
    maximum_chunk_bytes: u32,
    cancellation: &CancellationToken,
    call_ordinal: &mut u64,
) -> Result<Vec<LoadedNativeModelAuxiliaryArtifact>, NativeModelResourceServiceError> {
    let mut loaded = Vec::new();
    loaded
        .try_reserve_exact(manifest.artifacts.len())
        .map_err(|_| NativeModelResourceServiceError::OutOfMemory)?;
    for (artifact_index, artifact) in manifest.artifacts.iter().enumerate() {
        if !artifact.format.supports_auxiliary_artifact_stream() {
            continue;
        }
        cancellation
            .check()
            .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
        let byte_size = usize::try_from(artifact.byte_size)
            .map_err(|_| NativeModelResourceServiceError::OutOfMemory)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_size)
            .map_err(|_| NativeModelResourceServiceError::OutOfMemory)?;
        bytes.resize(byte_size, 0);
        let artifact_ordinal = u32::try_from(artifact_index).map_err(|_| {
            NativeModelResourceServiceError::MalformedSource(
                "source artifact ordinal does not fit the wire protocol",
            )
        })?;
        let mut byte_offset = 0_u64;
        while byte_offset < artifact.byte_size {
            let remaining = artifact.byte_size - byte_offset;
            let byte_length = u32::try_from(remaining.min(u64::from(maximum_chunk_bytes)))
                .map_err(|_| {
                    NativeModelResourceServiceError::MalformedSource(
                        "source artifact chunk length does not fit the wire protocol",
                    )
                })?;
            let response = session
                .call(
                    WorkerModelSourceRequest {
                        context: session_context.clone(),
                        call_ordinal: *call_ordinal,
                        operation: WorkerModelSourceOperation::ReadArtifact {
                            source_ordinal: manifest.source_ordinal,
                            artifact_ordinal,
                            artifact_sha256: artifact.sha256.clone(),
                            byte_offset,
                            byte_length,
                        },
                    },
                    cancellation,
                )
                .map_err(map_model_source_error)?;
            response.validate().map_err(map_model_source_error)?;
            let chunk = match response {
                WorkerModelSourceResponse::ArtifactChunk(chunk)
                    if chunk.session_id == session_context.session_id
                        && chunk.call_ordinal == *call_ordinal
                        && chunk.source_ordinal == manifest.source_ordinal
                        && chunk.artifact_ordinal == artifact_ordinal
                        && chunk.artifact_sha256 == artifact.sha256
                        && chunk.byte_offset == byte_offset
                        && chunk.bytes.len()
                            == usize::try_from(byte_length).unwrap_or(usize::MAX) =>
                {
                    chunk
                }
                WorkerModelSourceResponse::Rejected { error, .. } => {
                    return Err(map_model_source_error(error));
                }
                _ => {
                    return Err(NativeModelResourceServiceError::MalformedSource(
                        "source artifact response does not match its request",
                    ));
                }
            };
            let start = usize::try_from(byte_offset).map_err(|_| {
                NativeModelResourceServiceError::MalformedSource(
                    "source artifact offset does not fit address space",
                )
            })?;
            let end = start.checked_add(chunk.bytes.len()).ok_or(
                NativeModelResourceServiceError::MalformedSource(
                    "source artifact destination range overflowed",
                ),
            )?;
            bytes
                .get_mut(start..end)
                .ok_or(NativeModelResourceServiceError::MalformedSource(
                    "source artifact destination is outside the artifact",
                ))?
                .copy_from_slice(&chunk.bytes);
            byte_offset = byte_offset.checked_add(u64::from(byte_length)).ok_or(
                NativeModelResourceServiceError::MalformedSource(
                    "source artifact offset overflowed",
                ),
            )?;
            *call_ordinal = (*call_ordinal).checked_add(1).ok_or(
                NativeModelResourceServiceError::MalformedSource("source call ordinal overflowed"),
            )?;
        }
        loaded.push(LoadedNativeModelAuxiliaryArtifact {
            format: artifact.format,
            artifact_sha256: artifact.sha256.as_str().to_owned(),
            bytes,
        });
    }
    Ok(loaded)
}

fn load_manifest_tensors(
    session: &NativeModelSourceTransportSession,
    session_context: &WorkerModelSourceContext,
    manifest: &WorkerModelSourceManifest,
    maximum_chunk_bytes: u32,
    backend: &CpuBackend,
    tensor_context: &ExecutionContext<'_>,
    call_ordinal: &mut u64,
) -> Result<Vec<(String, Tensor)>, NativeModelResourceServiceError> {
    let mut ordered_state = Vec::new();
    ordered_state
        .try_reserve_exact(manifest.tensors.len())
        .map_err(|_| NativeModelResourceServiceError::OutOfMemory)?;
    for (tensor_index, tensor_manifest) in manifest.tensors.iter().enumerate() {
        tensor_context
            .cancellation
            .check()
            .map_err(|_| NativeModelResourceServiceError::Cancelled)?;
        let dtype = model_source_dtype(&tensor_manifest.data_type).ok_or(
            NativeModelResourceServiceError::MalformedSource("source tensor dtype is unsupported"),
        )?;
        let descriptor = TensorDescriptor::contiguous(
            tensor_manifest.shape.clone(),
            dtype,
            DeviceId::CPU,
            tensor_context.stream,
        )?;
        if descriptor.byte_len()? != tensor_manifest.byte_length {
            return Err(NativeModelResourceServiceError::MalformedSource(
                "source tensor byte length does not match its descriptor",
            ));
        }
        let (mut tensor, allocation) = backend.allocate(descriptor, tensor_context)?;
        backend.wait_event(allocation, tensor_context)?;
        let mut write = tensor.write()?;
        let destination = write.bytes_mut()?;
        let mut byte_offset = 0_u64;
        while byte_offset < tensor_manifest.byte_length {
            let remaining = tensor_manifest.byte_length - byte_offset;
            let byte_length = remaining.min(u64::from(maximum_chunk_bytes));
            let byte_length = u32::try_from(byte_length).map_err(|_| {
                NativeModelResourceServiceError::MalformedSource(
                    "source chunk length does not fit the wire protocol",
                )
            })?;
            let tensor_ordinal = u32::try_from(tensor_index).map_err(|_| {
                NativeModelResourceServiceError::MalformedSource(
                    "source tensor ordinal does not fit the wire protocol",
                )
            })?;
            let response = session
                .call(
                    WorkerModelSourceRequest {
                        context: session_context.clone(),
                        call_ordinal: *call_ordinal,
                        operation: WorkerModelSourceOperation::Read {
                            source_ordinal: manifest.source_ordinal,
                            tensor_ordinal,
                            byte_offset,
                            byte_length,
                        },
                    },
                    tensor_context.cancellation,
                )
                .map_err(map_model_source_error)?;
            response.validate().map_err(map_model_source_error)?;
            let chunk = match response {
                WorkerModelSourceResponse::Chunk(chunk)
                    if chunk.session_id == session_context.session_id
                        && chunk.call_ordinal == *call_ordinal
                        && chunk.source_ordinal == manifest.source_ordinal
                        && chunk.tensor_ordinal == tensor_ordinal
                        && chunk.byte_offset == byte_offset
                        && chunk.bytes.len()
                            == usize::try_from(byte_length).unwrap_or(usize::MAX) =>
                {
                    chunk
                }
                WorkerModelSourceResponse::Rejected { error, .. } => {
                    return Err(map_model_source_error(error));
                }
                _ => {
                    return Err(NativeModelResourceServiceError::MalformedSource(
                        "source chunk response does not match its request",
                    ));
                }
            };
            let start = usize::try_from(byte_offset).map_err(|_| {
                NativeModelResourceServiceError::MalformedSource(
                    "source chunk offset does not fit address space",
                )
            })?;
            let end = start.checked_add(chunk.bytes.len()).ok_or(
                NativeModelResourceServiceError::MalformedSource(
                    "source chunk destination range overflowed",
                ),
            )?;
            let destination = destination.get_mut(start..end).ok_or(
                NativeModelResourceServiceError::MalformedSource(
                    "source chunk destination is outside the tensor",
                ),
            )?;
            destination.copy_from_slice(&chunk.bytes);
            byte_offset = byte_offset.checked_add(u64::from(byte_length)).ok_or(
                NativeModelResourceServiceError::MalformedSource("source chunk offset overflowed"),
            )?;
            *call_ordinal = (*call_ordinal).checked_add(1).ok_or(
                NativeModelResourceServiceError::MalformedSource("source call ordinal overflowed"),
            )?;
        }
        drop(write);
        ordered_state.push((tensor_manifest.name.clone(), tensor));
    }
    Ok(ordered_state)
}

fn photomaker_entries(
    manifest: &WorkerModelSourceManifest,
    ordered_state: Vec<(String, Tensor)>,
) -> Result<Vec<NativePhotoMakerCheckpointEntry>, NativeModelResourceServiceError> {
    let nested = manifest.artifacts.iter().any(|artifact| {
        artifact.nested_state_disposition
            == WorkerModelSourceNestedStateDisposition::NestedStringToParam
    });
    if nested {
        if manifest.artifacts.len() != 1 {
            return Err(NativeModelResourceServiceError::MalformedSource(
                "nested PhotoMaker state spans more than one artifact",
            ));
        }
        Ok(vec![NativePhotoMakerCheckpointEntry::Mapping {
            key: "id_encoder".to_owned(),
            ordered_state,
        }])
    } else {
        Ok(ordered_state
            .into_iter()
            .map(|(key, tensor)| NativePhotoMakerCheckpointEntry::Tensor { key, tensor })
            .collect())
    }
}

fn model_source_dtype(value: &str) -> Option<DType> {
    Some(match value {
        "BOOL" | "torch.BoolStorage" => DType::Bool,
        "U8" | "torch.ByteStorage" => DType::U8,
        "I8" | "torch.CharStorage" => DType::I8,
        "I16" | "torch.ShortStorage" => DType::I16,
        "I32" | "torch.IntStorage" => DType::I32,
        "I64" | "torch.LongStorage" => DType::I64,
        "F16" | "torch.HalfStorage" => DType::F16,
        "BF16" | "torch.BFloat16Storage" => DType::Bf16,
        "F32" | "torch.FloatStorage" => DType::F32,
        "F64" | "torch.DoubleStorage" => DType::F64,
        "C64" | "torch.ComplexFloatStorage" => DType::Complex64,
        "C128" | "torch.ComplexDoubleStorage" => DType::Complex128,
        _ => return None,
    })
}

fn map_model_source_error(error: WorkerModelSourceError) -> NativeModelResourceServiceError {
    match error {
        WorkerModelSourceError::Cancelled => NativeModelResourceServiceError::Cancelled,
        WorkerModelSourceError::StaleAttempt
        | WorkerModelSourceError::StaleNode
        | WorkerModelSourceError::StaleService
        | WorkerModelSourceError::ForeignSession
        | WorkerModelSourceError::Late => NativeModelResourceServiceError::StaleIdentity,
        WorkerModelSourceError::SourceChanged => NativeModelResourceServiceError::SourceChanged,
        WorkerModelSourceError::HostFailure | WorkerModelSourceError::Closed => {
            NativeModelResourceServiceError::SourceUnavailable
        }
        WorkerModelSourceError::InvalidOrder => NativeModelResourceServiceError::MalformedSource(
            "model-source transport rejected call order",
        ),
        WorkerModelSourceError::Replay => NativeModelResourceServiceError::MalformedSource(
            "model-source transport rejected a replay",
        ),
        WorkerModelSourceError::InvalidSourceSelection => {
            NativeModelResourceServiceError::MalformedSource(
                "model-source transport rejected source selection",
            )
        }
        WorkerModelSourceError::InvalidSourceOrdinal => {
            NativeModelResourceServiceError::MalformedSource(
                "model-source transport rejected source ordinal",
            )
        }
        WorkerModelSourceError::InvalidTensorOrdinal => {
            NativeModelResourceServiceError::MalformedSource(
                "model-source transport rejected tensor ordinal",
            )
        }
        WorkerModelSourceError::RangeLimit => NativeModelResourceServiceError::MalformedSource(
            "model-source transport rejected byte range",
        ),
        WorkerModelSourceError::ChunkLimit => NativeModelResourceServiceError::MalformedSource(
            "model-source transport rejected chunk length",
        ),
        WorkerModelSourceError::DigestMismatch => NativeModelResourceServiceError::MalformedSource(
            "model-source transport rejected response digest",
        ),
    }
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn load_reduced_fixture_native_model_resource_for_test<R>(
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    cancellation: CancellationToken,
    scratch: ScratchReservation,
    handle_store: Arc<dyn NativeHandleStore>,
    binding: crate::NativeModelSourceWorkerBinding,
    backend: Arc<CpuBackend>,
    request: NativeModelResourceRequest,
    host_operation: impl FnOnce(NativeModelSourceTransportHost) -> Result<R, WorkerModelSourceError>
    + Send
    + 'static,
) -> Result<
    (
        Result<Arc<NativeModelPayload>, NativeModelResourceServiceError>,
        R,
    ),
    NativeModelResourceServiceError,
>
where
    R: Send + 'static,
{
    let (transport, host) = NativeModelSourceTransport::channel();
    let host_thread = std::thread::spawn(move || host_operation(host));
    let load_result = (|| {
        let identity =
            NativeNodeServiceIdentity::checked(Uuid::new_v4(), attempt_id, node_id.clone())
                .map_err(|_| NativeModelResourceServiceError::StaleIdentity)?;
        let service = NativeModelResourceServiceFactory {
            binding,
            transport,
            backend,
            construction_profile: NativeModelConstructionProfile::ReducedFixture,
        }
        .node_service(identity);
        let services = NativeNodeServices::checked(None, None, None)
            .and_then(|services| services.with_model_resources(service))
            .map_err(|_| NativeModelResourceServiceError::StaleIdentity)?;
        let context = NodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            services,
        )
        .map_err(|_| NativeModelResourceServiceError::StaleIdentity)?;
        context.load_native_model_resource(&request)
    })();
    let host_result = host_thread
        .join()
        .map_err(|_| {
            NativeModelResourceServiceError::MalformedSource(
                "model-source test host thread panicked",
            )
        })?
        .map_err(map_model_source_error)?;
    Ok((load_result, host_result))
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn execute_reduced_fixture_native_model_resource_for_test<R>(
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    cancellation: CancellationToken,
    scratch: ScratchReservation,
    handle_store: Arc<dyn NativeHandleStore>,
    binding: crate::NativeModelSourceWorkerBinding,
    backend: Arc<CpuBackend>,
    request: NativeModelResourceRequest,
    host_operation: impl FnOnce(NativeModelSourceTransportHost) -> Result<R, WorkerModelSourceError>
    + Send
    + 'static,
) -> Result<
    (
        Result<
            (
                Arc<NativeModelPayload>,
                crate::persistence::PersistedNativeModelResourceEvidence,
            ),
            NodeFailure,
        >,
        R,
    ),
    NativeModelResourceServiceError,
>
where
    R: Send + 'static,
{
    let (transport, host) = NativeModelSourceTransport::channel();
    let host_thread = std::thread::spawn(move || host_operation(host));
    let execution_result = (|| {
        let identity =
            NativeNodeServiceIdentity::checked(Uuid::new_v4(), attempt_id, node_id.clone())
                .map_err(|error| NodeFailure {
                    code: "native_model_resource_service_unavailable".to_owned(),
                    message: error.to_string(),
                    kind: NodeFailureKind::Failure,
                    retryable: true,
                })?;
        let service = NativeModelResourceServiceFactory {
            binding,
            transport,
            backend,
            construction_profile: NativeModelConstructionProfile::ReducedFixture,
        }
        .node_service(identity);
        let services = NativeNodeServices::checked(None, None, None)
            .and_then(|services| services.with_model_resources(service))
            .map_err(|error| NodeFailure {
                code: "native_model_resource_service_unavailable".to_owned(),
                message: error.to_string(),
                kind: NodeFailureKind::Failure,
                retryable: true,
            })?;
        let context = NodeContext::new_with_services(
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            services,
        )
        .map_err(|error| NodeFailure {
            code: "native_model_resource_service_unavailable".to_owned(),
            message: error.to_string(),
            kind: NodeFailureKind::Failure,
            retryable: true,
        })?;
        let (value, evidence) = crate::load_and_publish_native_model_resource(&request, &context)?;
        let NativeValue::Handle { value: handle } = value else {
            return Err(NodeFailure {
                code: "invalid_native_model_resource".to_owned(),
                message: "model-resource execution returned a non-handle value".to_owned(),
                kind: NodeFailureKind::Failure,
                retryable: false,
            });
        };
        let resolved = context
            .handle_store()
            .resolve(&handle, handle.handle_type(), &context.cancellation)
            .map_err(|error| NodeFailure {
                code: "native_model_resource_resolution_failed".to_owned(),
                message: error.to_string(),
                kind: NodeFailureKind::Failure,
                retryable: false,
            });
        let payload = match resolved {
            Ok(resolved) => match resolved.as_ref() {
                NativeStoredPayload::Model(model) => Ok(model.model_payload().clone()),
                _ => Err(NodeFailure {
                    code: "invalid_native_model_resource".to_owned(),
                    message: "model-resource handle resolved to another payload kind".to_owned(),
                    kind: NodeFailureKind::Failure,
                    retryable: false,
                }),
            },
            Err(error) => Err(error),
        };
        context
            .handle_store()
            .revoke(&handle, &context.cancellation)
            .map_err(|error| NodeFailure {
                code: "native_model_resource_rollback_failed".to_owned(),
                message: error.to_string(),
                kind: NodeFailureKind::Failure,
                retryable: false,
            })?;
        Ok((payload?, evidence))
    })();
    let host_result = host_thread
        .join()
        .map_err(|_| {
            NativeModelResourceServiceError::MalformedSource(
                "model-source test host thread panicked",
            )
        })?
        .map_err(map_model_source_error)?;
    Ok((execution_result, host_result))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCompileBackend {
    Graph,
    CudaGraphs,
}

impl NativeCompileBackend {
    pub fn from_source_name(value: &str) -> Result<Self, NativeCompileError> {
        match value {
            "inductor" | "native" => Ok(Self::Graph),
            "cudagraphs" => Ok(Self::CudaGraphs),
            _ => Err(NativeCompileError::UnsupportedBackend(value.to_owned())),
        }
    }

    const fn cache_dimension(self) -> &'static str {
        match self {
            Self::Graph => "native-graph",
            Self::CudaGraphs => "native-cudagraphs",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeCompilePolicy {
    backend: NativeCompileBackend,
    guard_policy: NativeCompileGuardPolicy,
    mode: Option<String>,
    fullgraph: bool,
    dynamic: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCompileGuardPolicy {
    ExactTypedInputs,
    SkipTransformerOptionsDictionary,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeCompilePhase {
    #[default]
    Eager,
    CapturingGraph,
}

pub fn compiler_is_compiling_exact_native(
    phase: NativeCompilePhase,
    cancellation: &CancellationToken,
) -> Result<bool, NativeCompileError> {
    cancellation
        .check()
        .map_err(|_| NativeCompileError::Cancelled)?;
    Ok(phase == NativeCompilePhase::CapturingGraph)
}

impl NativeCompilePolicy {
    pub fn from_source_configuration(
        backend: &str,
        mut options: BTreeMap<String, String>,
        mode: Option<String>,
        fullgraph: bool,
        dynamic: Option<bool>,
    ) -> Result<Self, NativeCompileError> {
        if options.len() > MAX_NATIVE_COMPILE_OPTIONS {
            return Err(NativeCompileError::TooManyOptions);
        }
        for (key, value) in &options {
            validate_compile_text("option key", key, false)?;
            validate_compile_text("option value", value, true)?;
        }
        let guard_policy = match options.remove("guard_filter_fn") {
            None => NativeCompileGuardPolicy::ExactTypedInputs,
            Some(value) if value == "skip_torch_compile_dict" => {
                NativeCompileGuardPolicy::SkipTransformerOptionsDictionary
            }
            Some(value) => {
                return Err(NativeCompileError::UnsupportedOption {
                    key: "guard_filter_fn".to_owned(),
                    value,
                });
            }
        };
        if let Some((key, value)) = options.into_iter().next() {
            return Err(NativeCompileError::UnsupportedOption { key, value });
        }
        if let Some(mode) = mode.as_deref() {
            validate_compile_text("mode", mode, false)?;
            if !matches!(
                mode,
                "default" | "reduce-overhead" | "max-autotune" | "max-autotune-no-cudagraphs"
            ) {
                return Err(NativeCompileError::UnsupportedMode(mode.to_owned()));
            }
        }
        Ok(Self {
            backend: NativeCompileBackend::from_source_name(backend)?,
            guard_policy,
            mode,
            fullgraph,
            dynamic,
        })
    }

    pub const fn backend(&self) -> NativeCompileBackend {
        self.backend
    }

    pub const fn guard_policy(&self) -> NativeCompileGuardPolicy {
        self.guard_policy
    }

    pub fn mode(&self) -> Option<&str> {
        self.mode.as_deref()
    }

    pub const fn fullgraph(&self) -> bool {
        self.fullgraph
    }

    pub const fn dynamic(&self) -> Option<bool> {
        self.dynamic
    }

    pub fn cache_token(&self) -> Result<String, NativeCompileError> {
        let encoded = serde_json::to_vec(self).map_err(NativeCompileError::Encode)?;
        Ok(format!("native-compile-v2:{:x}", Sha256::digest(encoded)))
    }

    pub fn invocation_guard_token(
        &self,
        typed_inputs: &BTreeMap<String, String>,
    ) -> Result<String, NativeCompileError> {
        if typed_inputs.len() > MAX_NATIVE_COMPILE_OPTIONS {
            return Err(NativeCompileError::TooManyGuardEntries);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-compile-guards.v1");
        hasher.update(self.cache_token()?.as_bytes());
        for (name, type_identity) in typed_inputs {
            validate_compile_text("guard name", name, false)?;
            validate_compile_text("guard type identity", type_identity, false)?;
            if self.guard_policy == NativeCompileGuardPolicy::SkipTransformerOptionsDictionary
                && name == "transformer_options"
            {
                continue;
            }
            hasher.update((name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            hasher.update((type_identity.len() as u64).to_le_bytes());
            hasher.update(type_identity.as_bytes());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[derive(Clone, Debug)]
pub struct NativeCompiledModel<Model> {
    model: Model,
    policy: NativeCompilePolicy,
    invocation_guard_token: String,
}

impl<Model> NativeCompiledModel<Model> {
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn policy(&self) -> &NativeCompilePolicy {
        &self.policy
    }

    pub fn invoke<Output>(
        &self,
        cancellation: &CancellationToken,
        operation: impl FnOnce(&Model) -> Output,
    ) -> Result<Output, NativeCompileError> {
        self.invoke_with_typed_inputs(cancellation, &BTreeMap::new(), operation)
    }

    pub fn invoke_with_typed_inputs<Output>(
        &self,
        cancellation: &CancellationToken,
        typed_inputs: &BTreeMap<String, String>,
        operation: impl FnOnce(&Model) -> Output,
    ) -> Result<Output, NativeCompileError> {
        cancellation
            .check()
            .map_err(|_| NativeCompileError::Cancelled)?;
        if self.policy.invocation_guard_token(typed_inputs)? != self.invocation_guard_token {
            return Err(NativeCompileError::InvocationGuardMismatch);
        }
        let output = operation(&self.model);
        cancellation
            .check()
            .map_err(|_| NativeCompileError::Cancelled)?;
        Ok(output)
    }

    pub fn into_inner(self) -> Model {
        self.model
    }
}

pub fn compile_exact_native<Model>(
    model: Model,
    policy: NativeCompilePolicy,
    capabilities: Option<&BackendCapabilityMatrix>,
    typed_inputs: &BTreeMap<String, String>,
    cancellation: &CancellationToken,
) -> Result<NativeCompiledModel<Model>, NativeCompileError> {
    cancellation
        .check()
        .map_err(|_| NativeCompileError::Cancelled)?;
    policy.cache_token()?;
    let invocation_guard_token = policy.invocation_guard_token(typed_inputs)?;
    if policy.backend() == NativeCompileBackend::CudaGraphs {
        let capabilities = capabilities.ok_or(NativeCompileError::UncertifiedCudaGraphs)?;
        if !matches!(
            capabilities.device().kind(),
            DeviceKind::Cuda | DeviceKind::Rocm
        ) || capabilities.device_properties().is_none()
        {
            return Err(NativeCompileError::UncertifiedCudaGraphs);
        }
    }
    cancellation
        .check()
        .map_err(|_| NativeCompileError::Cancelled)?;
    Ok(NativeCompiledModel {
        model,
        policy,
        invocation_guard_token,
    })
}

fn validate_compile_text(
    field: &'static str,
    value: &str,
    allow_empty: bool,
) -> Result<(), NativeCompileError> {
    if (!allow_empty && value.is_empty())
        || value.len() > MAX_NATIVE_COMPILE_TEXT_BYTES
        || value.contains('\0')
    {
        return Err(NativeCompileError::InvalidText { field });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum NativeCompileError {
    #[error("source compile backend {0:?} has no native Rust execution strategy")]
    UnsupportedBackend(String),
    #[error("source compile mode {0:?} has no native Rust execution strategy")]
    UnsupportedMode(String),
    #[error("source compile option {key:?}={value:?} has no native Rust execution strategy")]
    UnsupportedOption { key: String, value: String },
    #[error("native compile policy has too many options")]
    TooManyOptions,
    #[error("native compile invocation has too many typed guard entries")]
    TooManyGuardEntries,
    #[error("native compile invocation guard no longer matches its compiled inputs")]
    InvocationGuardMismatch,
    #[error("native compile {field} is empty, oversized, or contains NUL")]
    InvalidText { field: &'static str },
    #[error("native compile policy serialization failed: {0}")]
    Encode(serde_json::Error),
    #[error("native CUDA-graphs compilation requires a certified CUDA or ROCm capability matrix")]
    UncertifiedCudaGraphs,
    #[error("native compile operation was cancelled")]
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedEffect {
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub node_id: NodeId,
    pub service_id: Uuid,
    pub transaction_id: Uuid,
    pub kind: NativePreparedEffectKind,
    pub request_digest_sha256: String,
}

#[derive(Clone, Default)]
pub struct NativeNodeRegistry {
    nodes: BTreeMap<String, Arc<dyn NativeNode>>,
    descriptors: BTreeMap<String, RuntimeNodeDescriptor>,
    presentations: BTreeMap<String, RuntimeNodePresentation>,
    bindings: BTreeMap<String, RegistryBindingState>,
}

#[derive(Clone, Debug)]
struct RegistryBindingState {
    disposition: NativeNodeBindingDisposition,
    feature_id: String,
    provider_activation_sha256: Option<String>,
    implementation_namespace: Option<String>,
    catalog_source: String,
    reason: Option<String>,
}

#[derive(Clone)]
pub struct NativeProviderBindingActivation {
    claim: ProviderBindingClaim,
    node: Arc<dyn NativeNode>,
}

impl NativeProviderBindingActivation {
    pub fn new(claim: ProviderBindingClaim, node: Arc<dyn NativeNode>) -> Self {
        Self { claim, node }
    }
}

#[derive(Clone)]
pub struct NativeProviderBindingActivationSet {
    binding_set: ProviderBindingSet,
    activation_sha256: String,
    bindings: Vec<NativeProviderBindingActivation>,
}

impl NativeProviderBindingActivationSet {
    pub fn checked(
        profile_id: impl Into<String>,
        component_generation: u64,
        component_snapshot_sha256: impl Into<String>,
        component_digest_sha256: impl Into<String>,
        authorization_generation_sha256: impl Into<String>,
        binding_set: ProviderBindingSet,
        bindings: Vec<NativeProviderBindingActivation>,
    ) -> Result<Self, NativeNodeRegistryError> {
        let profile_id = profile_id.into();
        let component_snapshot_sha256 = component_snapshot_sha256.into();
        let component_digest_sha256 = component_digest_sha256.into();
        let authorization_generation_sha256 = authorization_generation_sha256.into();
        if profile_id.is_empty()
            || component_generation == 0
            || !valid_registry_sha256(&component_snapshot_sha256)
            || !valid_registry_sha256(&component_digest_sha256)
            || !valid_registry_sha256(&authorization_generation_sha256)
            || binding_set.bindings_sha256
                != binding_set
                    .canonical_bindings_sha256()
                    .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?
            || binding_set.bindings.len() != bindings.len()
        {
            return Err(NativeNodeRegistryError::InvalidProviderActivation);
        }
        let activation_sha256 = provider_activation_sha256(
            &profile_id,
            component_generation,
            &component_snapshot_sha256,
            &component_digest_sha256,
            &authorization_generation_sha256,
            &binding_set,
        )?;
        Ok(Self {
            binding_set,
            activation_sha256,
            bindings,
        })
    }

    pub fn activation_sha256(&self) -> &str {
        &self.activation_sha256
    }

    pub fn binding_set(&self) -> &ProviderBindingSet {
        &self.binding_set
    }
}

impl NativeNodeRegistry {
    pub fn register(&mut self, node: Arc<dyn NativeNode>) -> Result<(), ExecutionError> {
        let class_type = node.class_type().to_owned();
        let implementation_version = node.implementation_version().to_owned();
        if class_type.is_empty() || implementation_version.is_empty() {
            return Err(ExecutionError::InvalidNodeImplementation(class_type));
        }
        if self.nodes.contains_key(&class_type) {
            return Err(ExecutionError::DuplicateNodeImplementation(class_type));
        }
        if let Some(descriptor) = self.descriptors.get(&class_type)
            && descriptor.implementation_version != implementation_version
        {
            return Err(ExecutionError::DescriptorImplementationVersionMismatch {
                class_type,
                expected: descriptor.implementation_version.clone(),
                actual: implementation_version,
            });
        }
        let implementation_namespace = node.implementation_namespace().to_owned();
        if implementation_namespace.trim().is_empty() {
            return Err(ExecutionError::InvalidNodeImplementation(class_type));
        }
        self.nodes.insert(class_type.clone(), node);
        self.bindings.insert(
            class_type,
            RegistryBindingState {
                disposition: NativeNodeBindingDisposition::Executable,
                feature_id: "runtime-bound".to_owned(),
                provider_activation_sha256: None,
                implementation_namespace: Some(implementation_namespace.clone()),
                catalog_source: implementation_namespace,
                reason: None,
            },
        );
        Ok(())
    }

    pub fn node(&self, class_type: &str) -> Option<Arc<dyn NativeNode>> {
        self.nodes.get(class_type).cloned()
    }

    pub fn node_len(&self) -> usize {
        self.nodes.len()
    }

    pub fn register_descriptor(
        &mut self,
        descriptor: RuntimeNodeDescriptor,
    ) -> Result<(), PromptCompileError> {
        if descriptor.validate().is_err() {
            return Err(PromptCompileError::InvalidRuntimeDescriptor(
                descriptor.class_type,
            ));
        }
        if self.descriptors.contains_key(&descriptor.class_type) {
            return Err(PromptCompileError::DuplicateRuntimeDescriptor(
                descriptor.class_type,
            ));
        }
        if let Some(node) = self.nodes.get(&descriptor.class_type)
            && node.implementation_version() != descriptor.implementation_version
        {
            return Err(PromptCompileError::InvalidRuntimeDescriptor(
                descriptor.class_type,
            ));
        }
        self.descriptors
            .insert(descriptor.class_type.clone(), descriptor);
        Ok(())
    }

    pub fn descriptor(&self, class_type: &str) -> Option<&RuntimeNodeDescriptor> {
        self.descriptors.get(class_type)
    }

    pub fn descriptors(&self) -> impl Iterator<Item = (&str, &RuntimeNodeDescriptor)> {
        self.descriptors
            .iter()
            .map(|(class_type, descriptor)| (class_type.as_str(), descriptor))
    }

    pub fn descriptor_len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn descriptors_are_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn descriptors_are_fully_bound(&self) -> bool {
        !self.descriptors.is_empty()
            && self
                .descriptors
                .keys()
                .all(|class_type| self.nodes.contains_key(class_type))
    }

    pub fn descriptor_is_bound(&self, class_type: &str) -> bool {
        self.descriptors.contains_key(class_type) && self.nodes.contains_key(class_type)
    }

    pub fn validate_comprehensive_bindings(&self) -> Result<(), NativeNodeRegistryError> {
        for (class_type, descriptor) in &self.descriptors {
            descriptor.validate()?;
            let binding = self
                .bindings
                .get(class_type)
                .ok_or_else(|| NativeNodeRegistryError::IncompleteRegistry(class_type.clone()))?;
            let presentation = self
                .presentations
                .get(class_type)
                .ok_or_else(|| NativeNodeRegistryError::IncompleteRegistry(class_type.clone()))?;
            presentation.validate()?;
            let expected_output_names = descriptor
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>();
            if binding.catalog_source.is_empty()
                || presentation
                    .output_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    != expected_output_names
            {
                return Err(NativeNodeRegistryError::IncompleteRegistry(
                    class_type.clone(),
                ));
            }
            let has_node = self.nodes.contains_key(class_type);
            let node_matches = self.nodes.get(class_type).is_none_or(|node| {
                node.class_type() == class_type
                    && node.implementation_version() == descriptor.implementation_version
                    && binding.implementation_namespace.as_deref()
                        == Some(node.implementation_namespace())
            });
            let valid = match binding.disposition {
                NativeNodeBindingDisposition::Executable => {
                    has_node
                        && node_matches
                        && binding.provider_activation_sha256.is_none()
                        && binding.reason.is_none()
                        && binding
                            .implementation_namespace
                            .as_deref()
                            .is_some_and(|value| !value.is_empty())
                }
                NativeNodeBindingDisposition::ProviderRequired => {
                    has_node == binding.provider_activation_sha256.is_some()
                        && node_matches
                        && binding
                            .reason
                            .as_deref()
                            .is_some_and(|value| !value.is_empty())
                        && binding
                            .implementation_namespace
                            .as_deref()
                            .is_some_and(|value| !value.is_empty())
                }
                NativeNodeBindingDisposition::Unavailable => {
                    !has_node
                        && binding.provider_activation_sha256.is_none()
                        && binding
                            .reason
                            .as_deref()
                            .is_some_and(|value| !value.is_empty())
                        && binding.implementation_namespace.is_none()
                }
            };
            if !valid {
                return Err(NativeNodeRegistryError::IncompleteRegistry(
                    class_type.clone(),
                ));
            }
        }
        if self.nodes.keys().any(|class_type| {
            !self.descriptors.contains_key(class_type) || !self.bindings.contains_key(class_type)
        }) || self.presentations.keys().any(|class_type| {
            !self.descriptors.contains_key(class_type) || !self.bindings.contains_key(class_type)
        }) || self.bindings.keys().any(|class_type| {
            !self.descriptors.contains_key(class_type)
                || !self.presentations.contains_key(class_type)
        }) {
            return Err(NativeNodeRegistryError::IncompleteRegistry(
                "registry key sets differ".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn bindings_are_comprehensive(&self) -> bool {
        self.validate_comprehensive_bindings().is_ok()
    }

    pub fn implementation_namespace(&self, class_type: &str) -> Option<&str> {
        self.nodes
            .get(class_type)
            .map(|node| node.implementation_namespace())
    }

    pub fn binding_disposition(&self, class_type: &str) -> Option<NativeNodeBindingDisposition> {
        self.bindings.get(class_type).map(|binding| {
            if binding.provider_activation_sha256.is_some() {
                NativeNodeBindingDisposition::Executable
            } else {
                binding.disposition
            }
        })
    }

    pub fn binding_declared_disposition(
        &self,
        class_type: &str,
    ) -> Option<NativeNodeBindingDisposition> {
        self.bindings
            .get(class_type)
            .map(|binding| binding.disposition)
    }

    pub fn provider_binding_is_activated(&self, class_type: &str) -> Option<bool> {
        self.bindings
            .get(class_type)
            .map(|binding| binding.provider_activation_sha256.is_some())
    }

    pub fn binding_source(&self, class_type: &str) -> Option<&str> {
        self.bindings
            .get(class_type)
            .map(|binding| binding.catalog_source.as_str())
    }

    pub fn binding_implementation_namespace(&self, class_type: &str) -> Option<&str> {
        self.bindings
            .get(class_type)
            .and_then(|binding| binding.implementation_namespace.as_deref())
    }

    pub fn unavailable_reason(&self, class_type: &str) -> Option<&str> {
        if self.nodes.contains_key(class_type) {
            return None;
        }
        self.bindings
            .get(class_type)
            .filter(|binding| binding.disposition != NativeNodeBindingDisposition::Executable)
            .and_then(|binding| binding.reason.as_deref())
    }

    pub fn presentation(&self, class_type: &str) -> Option<&RuntimeNodePresentation> {
        self.presentations.get(class_type)
    }

    pub fn register_bound_batch(
        &mut self,
        bindings: impl IntoIterator<Item = (RuntimeNodeDescriptor, Arc<dyn NativeNode>)>,
    ) -> Result<(), NativeNodeRegistryError> {
        self.register_bound_batch_internal(
            bindings
                .into_iter()
                .map(|(descriptor, node)| (descriptor, node, None)),
        )
    }

    pub fn register_bound_batch_with_presentations(
        &mut self,
        bindings: impl IntoIterator<
            Item = (
                RuntimeNodeDescriptor,
                Arc<dyn NativeNode>,
                RuntimeNodePresentation,
            ),
        >,
    ) -> Result<(), NativeNodeRegistryError> {
        self.register_bound_batch_internal(
            bindings
                .into_iter()
                .map(|(descriptor, node, presentation)| (descriptor, node, Some(presentation))),
        )
    }

    pub fn register_native_bindings(
        &mut self,
        bindings: impl IntoIterator<Item = NativeNodeBinding>,
    ) -> Result<(), NativeNodeRegistryError> {
        let catalog = NodeRegistry::built_in()?;
        let bindings = bindings.into_iter().collect::<Vec<_>>();
        for binding in &bindings {
            catalog.validate_native_binding(binding)?;
        }
        let mut next = self.clone();
        for binding in bindings {
            binding.validate()?;
            let class_type = binding.descriptor().class_type.clone();
            let feature_id = binding.feature_id().to_owned();
            if next.descriptors.contains_key(&class_type)
                || next.nodes.contains_key(&class_type)
                || next.presentations.contains_key(&class_type)
                || next.bindings.contains_key(&class_type)
            {
                return Err(NativeNodeRegistryError::DuplicateBinding(class_type));
            }
            let descriptor = binding.descriptor().clone();
            let presentation = binding.presentation().clone();
            let catalog_source = catalog
                .descriptor(&class_type)
                .map(|descriptor| descriptor.source_file.clone())
                .ok_or_else(|| NativeNodeRegistryError::BindingMismatch(class_type.clone()))?;
            match binding {
                NativeNodeBinding::Executable { node, .. } => {
                    let source = node.implementation_namespace().to_owned();
                    next.register_descriptor(descriptor)?;
                    next.register(node)?;
                    next.presentations.insert(class_type.clone(), presentation);
                    next.bindings.insert(
                        class_type,
                        RegistryBindingState {
                            disposition: NativeNodeBindingDisposition::Executable,
                            feature_id,
                            provider_activation_sha256: None,
                            implementation_namespace: Some(source),
                            catalog_source,
                            reason: None,
                        },
                    );
                }
                NativeNodeBinding::ProviderRequired {
                    provider, reason, ..
                } => {
                    next.register_descriptor(descriptor)?;
                    next.presentations.insert(class_type.clone(), presentation);
                    next.bindings.insert(
                        class_type,
                        RegistryBindingState {
                            disposition: NativeNodeBindingDisposition::ProviderRequired,
                            feature_id,
                            provider_activation_sha256: None,
                            implementation_namespace: Some(provider),
                            catalog_source,
                            reason: Some(reason),
                        },
                    );
                }
                NativeNodeBinding::Unavailable { reason, .. } => {
                    next.register_descriptor(descriptor)?;
                    next.presentations.insert(class_type.clone(), presentation);
                    next.bindings.insert(
                        class_type,
                        RegistryBindingState {
                            disposition: NativeNodeBindingDisposition::Unavailable,
                            feature_id,
                            provider_activation_sha256: None,
                            implementation_namespace: None,
                            catalog_source,
                            reason: Some(reason),
                        },
                    );
                }
            }
        }
        *self = next;
        Ok(())
    }

    pub(crate) fn replace_native_bindings(
        &mut self,
        bindings: impl IntoIterator<Item = NativeNodeBinding>,
    ) -> Result<(), NativeNodeRegistryError> {
        let bindings = bindings.into_iter().collect::<Vec<_>>();
        let mut next = self.clone();
        for binding in &bindings {
            let class_type = &binding.descriptor().class_type;
            next.descriptors.remove(class_type);
            next.nodes.remove(class_type);
            next.presentations.remove(class_type);
            next.bindings.remove(class_type);
        }
        next.register_native_bindings(bindings)?;
        *self = next;
        Ok(())
    }

    pub fn provider_binding_contract_sha256(
        &self,
        class_type: &str,
        transport_schema: &str,
        materializer_schema: &str,
    ) -> Result<Option<String>, NativeNodeRegistryError> {
        let transport_schema: CanonicalTypeId = transport_schema
            .parse()
            .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?;
        let materializer_schema: CanonicalTypeId = materializer_schema
            .parse()
            .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?;
        validate_native_provider_schemas(&transport_schema, &materializer_schema)
            .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?;
        let Some(binding) = self.bindings.get(class_type) else {
            return Ok(None);
        };
        if binding.disposition != NativeNodeBindingDisposition::ProviderRequired {
            return Ok(None);
        }
        let descriptor = self
            .descriptors
            .get(class_type)
            .ok_or_else(|| NativeNodeRegistryError::BindingMismatch(class_type.to_owned()))?;
        let presentation = self
            .presentations
            .get(class_type)
            .ok_or_else(|| NativeNodeRegistryError::BindingMismatch(class_type.to_owned()))?;
        let implementation_namespace = binding
            .implementation_namespace
            .as_deref()
            .ok_or_else(|| NativeNodeRegistryError::BindingMismatch(class_type.to_owned()))?;
        Ok(Some(provider_contract_sha256(
            &binding.feature_id,
            implementation_namespace,
            descriptor,
            presentation,
            &transport_schema.to_string(),
            &materializer_schema.to_string(),
        )?))
    }

    pub fn activate_provider_binding_set(
        &mut self,
        activation: NativeProviderBindingActivationSet,
    ) -> Result<(), NativeNodeRegistryError> {
        if activation.binding_set.implementation_namespace.is_empty()
            || activation.binding_set.bindings_sha256
                != activation
                    .binding_set
                    .canonical_bindings_sha256()
                    .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?
        {
            return Err(NativeNodeRegistryError::InvalidProviderActivation);
        }
        let mut next = self.clone();
        let expected = next
            .bindings
            .iter()
            .filter_map(|(class_type, binding)| {
                (binding.disposition == NativeNodeBindingDisposition::ProviderRequired
                    && binding.implementation_namespace.as_deref()
                        == Some(activation.binding_set.implementation_namespace.as_str()))
                .then_some(class_type.clone())
            })
            .collect::<BTreeSet<_>>();
        let claimed = activation
            .binding_set
            .bindings
            .iter()
            .map(|claim| claim.node_id.clone())
            .collect::<BTreeSet<_>>();
        if expected.is_empty()
            || expected != claimed
            || activation.bindings.len() != activation.binding_set.bindings.len()
        {
            return Err(NativeNodeRegistryError::IncompleteProviderActivation(
                activation.binding_set.implementation_namespace.clone(),
            ));
        }
        let claims = activation
            .binding_set
            .bindings
            .iter()
            .map(|claim| (claim.node_id.as_str(), claim))
            .collect::<BTreeMap<_, _>>();
        let mut activated = BTreeSet::new();
        for activation_binding in &activation.bindings {
            let node = &activation_binding.node;
            let claim = &activation_binding.claim;
            let class_type = node.class_type().to_owned();
            if !activated.insert(class_type.clone())
                || next.nodes.contains_key(&class_type)
                || claims.get(class_type.as_str()).copied() != Some(claim)
            {
                return Err(NativeNodeRegistryError::DuplicateBinding(class_type));
            }
            let descriptor = next
                .descriptors
                .get(&class_type)
                .ok_or_else(|| NativeNodeRegistryError::BindingMismatch(class_type.clone()))?;
            let binding = next
                .bindings
                .get(&class_type)
                .ok_or_else(|| NativeNodeRegistryError::BindingMismatch(class_type.clone()))?;
            if binding.disposition != NativeNodeBindingDisposition::ProviderRequired
                || binding.feature_id != claim.feature_id
                || descriptor.implementation_version != node.implementation_version()
                || binding.implementation_namespace.as_deref()
                    != Some(node.implementation_namespace())
                || binding.provider_activation_sha256.is_some()
                || self.provider_binding_contract_sha256(
                    &class_type,
                    &claim.transport_schema.to_string(),
                    &claim.materializer_schema.to_string(),
                )? != Some(claim.contract_sha256.clone())
            {
                return Err(NativeNodeRegistryError::BindingMismatch(class_type));
            }
            next.nodes.insert(class_type.clone(), node.clone());
            if let Some(binding) = next.bindings.get_mut(&class_type) {
                binding.provider_activation_sha256 = Some(activation.activation_sha256.clone());
            }
        }
        if activated != expected {
            return Err(NativeNodeRegistryError::IncompleteProviderActivation(
                activation.binding_set.implementation_namespace.clone(),
            ));
        }
        next.validate_comprehensive_bindings()?;
        *self = next;
        Ok(())
    }

    fn register_bound_batch_internal(
        &mut self,
        bindings: impl IntoIterator<
            Item = (
                RuntimeNodeDescriptor,
                Arc<dyn NativeNode>,
                Option<RuntimeNodePresentation>,
            ),
        >,
    ) -> Result<(), NativeNodeRegistryError> {
        let mut next = self.clone();
        for (descriptor, node, presentation) in bindings {
            let class_type = descriptor.class_type.clone();
            if descriptor.class_type != node.class_type()
                || descriptor.implementation_version != node.implementation_version()
            {
                return Err(NativeNodeRegistryError::BindingMismatch(class_type));
            }
            if let Some(presentation) = presentation.as_ref() {
                if presentation.validate().is_err()
                    || presentation.output_names.len() != descriptor.outputs.len()
                {
                    return Err(NativeNodeRegistryError::InvalidPresentation(class_type));
                }
                if next.presentations.contains_key(&class_type) {
                    return Err(NativeNodeRegistryError::DuplicatePresentation(class_type));
                }
            }
            next.register_descriptor(descriptor)?;
            next.register(node)?;
            if let Some(presentation) = presentation {
                next.presentations.insert(class_type, presentation);
            }
        }
        *self = next;
        Ok(())
    }
}

fn provider_contract_sha256(
    feature_id: &str,
    implementation_namespace: &str,
    descriptor: &RuntimeNodeDescriptor,
    presentation: &RuntimeNodePresentation,
    transport_schema: &str,
    materializer_schema: &str,
) -> Result<String, NativeNodeRegistryError> {
    let descriptor = serde_json::to_vec(descriptor)
        .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?;
    let presentation = serde_json::to_vec(presentation)
        .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?;
    let mut hasher = Sha256::new();
    for field in [
        b"zed:sim-native-provider-contract@1".as_slice(),
        feature_id.as_bytes(),
        implementation_namespace.as_bytes(),
        descriptor.as_slice(),
        presentation.as_slice(),
        transport_schema.as_bytes(),
        materializer_schema.as_bytes(),
    ] {
        hash_registry_field(&mut hasher, field)?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn provider_activation_sha256(
    profile_id: &str,
    component_generation: u64,
    component_snapshot_sha256: &str,
    component_digest_sha256: &str,
    authorization_generation_sha256: &str,
    binding_set: &ProviderBindingSet,
) -> Result<String, NativeNodeRegistryError> {
    let binding_bytes = binding_set
        .canonical_binding_bytes()
        .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?;
    let mut hasher = Sha256::new();
    for field in [
        b"zed:sim-provider-activation-set@1".as_slice(),
        profile_id.as_bytes(),
        component_snapshot_sha256.as_bytes(),
        component_digest_sha256.as_bytes(),
        authorization_generation_sha256.as_bytes(),
        binding_set.bindings_sha256.as_bytes(),
        binding_bytes.as_slice(),
    ] {
        hash_registry_field(&mut hasher, field)?;
    }
    hasher.update(component_generation.to_le_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_registry_field(hasher: &mut Sha256, value: &[u8]) -> Result<(), NativeNodeRegistryError> {
    let length = u64::try_from(value.len())
        .map_err(|_| NativeNodeRegistryError::InvalidProviderActivation)?;
    hasher.update(length.to_le_bytes());
    hasher.update(value);
    Ok(())
}

fn valid_registry_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Debug, Error)]
pub enum NativeNodeRegistryError {
    #[error(transparent)]
    Descriptor(#[from] PromptCompileError),
    #[error(transparent)]
    Execution(#[from] ExecutionError),
    #[error("native node binding for `{0}` does not match its implementation")]
    BindingMismatch(String),
    #[error("native node presentation for `{0}` is invalid")]
    InvalidPresentation(String),
    #[error("native node presentation for `{0}` is already registered")]
    DuplicatePresentation(String),
    #[error("native node binding for `{0}` is already registered")]
    DuplicateBinding(String),
    #[error("native node registry is incomplete at `{0}`")]
    IncompleteRegistry(String),
    #[error("native provider binding activation is invalid")]
    InvalidProviderActivation,
    #[error("native provider binding activation for `{0}` is incomplete")]
    IncompleteProviderActivation(String),
    #[error(transparent)]
    Contract(#[from] NativeNodeContractError),
    #[error(transparent)]
    Catalog(#[from] sim_nodes::NodeRegistryError),
}

#[derive(Clone)]
pub struct NativeHandleStoreGeneration {
    state: Arc<NativeHandleStoreGenerationState>,
}

struct NativeHandleStoreGenerationState {
    identity: NativeHandleStoreIdentity,
    next_generation: AtomicU64,
    data: Mutex<NativeHandleStoreData>,
    sessions: Mutex<BTreeMap<Uuid, Weak<RuntimeNativeHandleStoreSession>>>,
    active_attempts: Mutex<BTreeSet<Uuid>>,
    capacity: usize,
    byte_capacity: usize,
    #[cfg(test)]
    test_hooks: Mutex<NativeHandleStoreTestHooks>,
}

#[cfg(test)]
#[derive(Default)]
struct NativeHandleStoreTestHooks {
    after_publish_insert: Option<Arc<dyn Fn() + Send + Sync>>,
    after_resolve_increment: Option<Arc<dyn Fn() + Send + Sync>>,
    after_cache_insert: Option<Arc<dyn Fn() + Send + Sync>>,
}

#[derive(Default)]
struct NativeHandleStoreData {
    values: BTreeMap<String, StoredNativeHandle>,
    allocations: BTreeMap<NativeResidentAllocationId, StoredResidentAllocation>,
    resident_bytes: usize,
}

#[derive(Clone)]
struct StoredResidentAllocation {
    resident_bytes: usize,
    handle_references: usize,
}

fn remove_stored_handle(
    data: &mut NativeHandleStoreData,
    identifier: &str,
) -> Option<StoredNativeHandle> {
    let residency = data.values.get(identifier)?.residency.clone();
    let removed_shared_bytes =
        residency
            .shared_allocations()
            .iter()
            .try_fold(0usize, |bytes, allocation| {
                let stored = data.allocations.get(allocation.id())?;
                if stored.resident_bytes != allocation.resident_bytes()
                    || stored.handle_references == 0
                {
                    return None;
                }
                if stored.handle_references == 1 {
                    bytes.checked_add(stored.resident_bytes)
                } else {
                    Some(bytes)
                }
            })?;
    let removed_bytes = residency
        .exclusive_bytes()
        .checked_add(removed_shared_bytes)?;
    let next_resident_bytes = data.resident_bytes.checked_sub(removed_bytes)?;
    let removed = data.values.remove(identifier)?;
    for allocation in residency.shared_allocations() {
        let remove = data
            .allocations
            .get(allocation.id())
            .is_some_and(|stored| stored.handle_references == 1);
        if remove {
            data.allocations.remove(allocation.id());
        } else if let Some(stored) = data.allocations.get_mut(allocation.id()) {
            stored.handle_references -= 1;
        }
    }
    data.resident_bytes = next_resident_bytes;
    Some(removed)
}

fn retire_stored_handle(data: &mut NativeHandleStoreData, identifier: &str) -> bool {
    let should_remove = match data.values.get_mut(identifier) {
        Some(record) => {
            record.retired = true;
            record.roots == 0 && record.resolved_roots == 0
        }
        None => return false,
    };
    if should_remove {
        remove_stored_handle(data, identifier).is_some()
    } else {
        true
    }
}

#[derive(Clone)]
struct StoredNativeHandle {
    handle_type: NativeHandleType,
    generation: u64,
    digest_sha256: String,
    value: Arc<NativeStoredPayload>,
    committed: bool,
    published_by: AttemptId,
    residency: NativePayloadResidency,
    roots: usize,
    resolved_roots: usize,
    retired: bool,
}

#[derive(Clone)]
pub struct NativeHandleLease {
    inner: Arc<NativeHandleLeaseInner>,
}

struct NativeHandleLeaseInner {
    generation: NativeHandleStoreGeneration,
    handles: Vec<NativeOpaqueHandle>,
}

#[derive(Debug)]
struct RuntimeResolvedPayloadRetention {
    generation: NativeHandleStoreGeneration,
    handle: NativeOpaqueHandle,
}

impl NativeResolvedPayloadRetention for RuntimeResolvedPayloadRetention {}

impl fmt::Debug for NativeHandleLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeHandleLease")
            .field("handle_count", &self.inner.handles.len())
            .finish_non_exhaustive()
    }
}

impl Drop for NativeHandleLeaseInner {
    fn drop(&mut self) {
        self.generation.release_roots(&self.handles);
    }
}

impl Drop for RuntimeResolvedPayloadRetention {
    fn drop(&mut self) {
        self.generation.release_resolved_root(&self.handle);
    }
}

impl PartialEq for NativeHandleLease {
    fn eq(&self, other: &Self) -> bool {
        self.inner.generation.identity() == other.inner.generation.identity()
            && self.inner.handles == other.inner.handles
    }
}

impl NativeHandleLease {
    pub(crate) fn covers_values(&self, values: &[NativeValue]) -> bool {
        let mut handles = Vec::new();
        for value in values {
            collect_native_value_handles(value, &mut handles);
        }
        let unique = handles
            .into_iter()
            .map(|handle| {
                (
                    (handle.identifier().to_owned(), handle.generation()),
                    handle.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        unique.len() == self.inner.handles.len()
            && unique
                .values()
                .zip(self.inner.handles.iter())
                .all(|(expected, actual)| expected == actual)
    }

    pub(crate) fn store_identity(&self) -> NativeHandleStoreIdentity {
        self.inner.generation.identity()
    }
}

impl fmt::Debug for NativeHandleStoreGeneration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeHandleStoreGeneration")
            .field("identity", &self.state.identity)
            .finish_non_exhaustive()
    }
}

impl NativeHandleStoreGeneration {
    pub fn new() -> Result<Self, NativeHandleStoreError> {
        Self::with_capacities(MAX_RUNTIME_NATIVE_HANDLES, MAX_RUNTIME_NATIVE_HANDLE_BYTES)
    }

    #[cfg(feature = "test-support")]
    pub(crate) fn live_handle_count_for_test(&self) -> usize {
        self.state.data.lock().values.len()
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, NativeHandleStoreError> {
        Self::with_capacities(capacity, MAX_RUNTIME_NATIVE_HANDLE_BYTES)
    }

    pub fn with_capacities(
        capacity: usize,
        byte_capacity: usize,
    ) -> Result<Self, NativeHandleStoreError> {
        Self::with_identity(
            NativeHandleStoreIdentity::new(Uuid::new_v4(), Uuid::new_v4())?,
            capacity,
            byte_capacity,
        )
    }

    pub(crate) fn from_identity(
        identity: NativeHandleStoreIdentity,
    ) -> Result<Self, NativeHandleStoreError> {
        Self::with_identity(
            identity,
            MAX_RUNTIME_NATIVE_HANDLES,
            MAX_RUNTIME_NATIVE_HANDLE_BYTES,
        )
    }

    fn with_identity(
        identity: NativeHandleStoreIdentity,
        capacity: usize,
        byte_capacity: usize,
    ) -> Result<Self, NativeHandleStoreError> {
        if capacity == 0 || byte_capacity == 0 {
            return Err(NativeHandleStoreError::Rejected(
                "native handle store capacities must be nonzero".to_owned(),
            ));
        }
        Ok(Self {
            state: Arc::new(NativeHandleStoreGenerationState {
                identity,
                next_generation: AtomicU64::new(1),
                data: Mutex::new(NativeHandleStoreData::default()),
                sessions: Mutex::new(BTreeMap::new()),
                active_attempts: Mutex::new(BTreeSet::new()),
                capacity,
                byte_capacity,
                #[cfg(test)]
                test_hooks: Mutex::new(NativeHandleStoreTestHooks::default()),
            }),
        })
    }

    pub fn fresh_generation(&self) -> Result<Self, NativeHandleStoreError> {
        Self::with_identity(
            NativeHandleStoreIdentity::new(self.identity().store_id, Uuid::new_v4())?,
            self.state.capacity,
            self.state.byte_capacity,
        )
    }

    pub fn identity(&self) -> NativeHandleStoreIdentity {
        self.state.identity
    }

    pub fn len(&self) -> usize {
        self.state.data.lock().values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn resident_bytes(&self) -> usize {
        self.state.data.lock().resident_bytes
    }

    fn acquire_lease<'a>(
        &self,
        handles: impl IntoIterator<Item = &'a NativeOpaqueHandle>,
    ) -> Result<Option<NativeHandleLease>, NativeHandleStoreError> {
        let mut handles_by_key = BTreeMap::new();
        for handle in handles {
            handle.validate()?;
            if handle.store_identity().store_id != self.identity().store_id {
                return Err(NativeHandleStoreError::WrongStore);
            }
            if handle.store_identity().generation_id != self.identity().generation_id {
                return Err(NativeHandleStoreError::WrongGeneration);
            }
            let key = (handle.identifier().to_owned(), handle.generation());
            if let Some(existing) = handles_by_key.insert(key, handle.clone())
                && existing != *handle
            {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
        }
        let handles = handles_by_key;
        if handles.is_empty() {
            return Ok(None);
        }
        let mut data = self.state.data.lock();
        for ((identifier, generation), handle) in &handles {
            let record = data
                .values
                .get(identifier)
                .ok_or_else(|| NativeHandleStoreError::Missing(identifier.clone()))?;
            if record.retired {
                return Err(NativeHandleStoreError::Missing(identifier.clone()));
            }
            if !record.committed || record.generation != *generation {
                return Err(NativeHandleStoreError::WrongGeneration);
            }
            if record.handle_type != *handle.handle_type() {
                return Err(NativeHandleStoreError::WrongType {
                    expected: record.handle_type.type_id.clone(),
                    actual: handle.handle_type().type_id.clone(),
                });
            }
            if Some(record.digest_sha256.as_str()) != handle.digest_sha256() {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            if record.roots == usize::MAX {
                return Err(NativeHandleStoreError::Rejected(
                    "native handle root count overflowed".to_owned(),
                ));
            }
        }
        for (identifier, _) in handles.keys() {
            if let Some(record) = data.values.get_mut(identifier) {
                record.roots += 1;
            }
        }
        Ok(Some(NativeHandleLease {
            inner: Arc::new(NativeHandleLeaseInner {
                generation: self.clone(),
                handles: handles.into_values().collect(),
            }),
        }))
    }

    fn release_roots(&self, handles: &[NativeOpaqueHandle]) {
        let mut data = self.state.data.lock();
        let mut removals = Vec::new();
        for handle in handles {
            if let Some(record) = data.values.get_mut(handle.identifier())
                && record.generation == handle.generation()
                && record.roots > 0
            {
                record.roots -= 1;
                if record.roots == 0 && record.committed {
                    removals.push(handle.identifier().to_owned());
                }
            }
        }
        for identifier in removals {
            retire_stored_handle(&mut data, &identifier);
        }
    }

    fn release_resolved_root(&self, handle: &NativeOpaqueHandle) {
        let mut data = self.state.data.lock();
        let should_remove = if let Some(record) = data.values.get_mut(handle.identifier()) {
            if record.generation != handle.generation()
                || record.handle_type != *handle.handle_type()
                || Some(record.digest_sha256.as_str()) != handle.digest_sha256()
                || record.resolved_roots == 0
            {
                false
            } else {
                record.resolved_roots -= 1;
                record.retired && record.roots == 0 && record.resolved_roots == 0
            }
        } else {
            false
        };
        if should_remove {
            let removed = remove_stored_handle(&mut data, handle.identifier());
            debug_assert!(removed.is_some());
        }
    }

    fn collect_unrooted_attempt(&self, attempt_id: AttemptId) {
        let mut data = self.state.data.lock();
        let identifiers = data
            .values
            .iter()
            .filter(|(_, record)| {
                record.committed
                    && !record.retired
                    && record.published_by == attempt_id
                    && record.roots == 0
            })
            .map(|(identifier, _)| identifier.clone())
            .collect::<Vec<_>>();
        for identifier in identifiers {
            retire_stored_handle(&mut data, &identifier);
        }
    }

    fn session(&self, attempt_id: AttemptId) -> Arc<RuntimeNativeHandleStoreSession> {
        loop {
            let mut sessions = self.state.sessions.lock();
            match sessions.get(&attempt_id.0) {
                Some(session) => {
                    if let Some(session) = session.upgrade() {
                        return session;
                    }
                    drop(sessions);
                    std::thread::yield_now();
                }
                None => {
                    let session = Arc::new(RuntimeNativeHandleStoreSession {
                        generation: self.clone(),
                        attempt_id,
                        staged: Mutex::new(NativeHandleStoreSessionStage::default()),
                    });
                    sessions.insert(attempt_id.0, Arc::downgrade(&session));
                    return session;
                }
            }
        }
    }

    pub fn handle_store_for_attempt(&self, attempt_id: AttemptId) -> Arc<dyn NativeHandleStore> {
        self.session(attempt_id)
    }

    fn try_claim_attempt(&self, attempt_id: AttemptId) -> Option<NativeExecutionAttemptClaim> {
        let mut active_attempts = self.state.active_attempts.lock();
        active_attempts
            .insert(attempt_id.0)
            .then(|| NativeExecutionAttemptClaim {
                generation: self.clone(),
                attempt_id,
            })
    }

    #[cfg(test)]
    fn set_after_publish_insert_hook(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        self.state.test_hooks.lock().after_publish_insert = Some(hook);
    }

    #[cfg(test)]
    fn set_after_resolve_increment_hook(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        self.state.test_hooks.lock().after_resolve_increment = Some(hook);
    }

    #[cfg(test)]
    fn set_after_cache_insert_hook(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        self.state.test_hooks.lock().after_cache_insert = Some(hook);
    }

    #[cfg(test)]
    fn run_after_publish_insert_hook(&self) {
        let hook = self.state.test_hooks.lock().after_publish_insert.clone();
        if let Some(hook) = hook {
            hook();
        }
    }

    #[cfg(test)]
    fn run_after_resolve_increment_hook(&self) {
        let hook = self.state.test_hooks.lock().after_resolve_increment.clone();
        if let Some(hook) = hook {
            hook();
        }
    }

    #[cfg(test)]
    fn run_after_cache_insert_hook(&self) {
        let hook = self.state.test_hooks.lock().after_cache_insert.clone();
        if let Some(hook) = hook {
            hook();
        }
    }
}

struct RuntimeNativeHandleStoreSession {
    generation: NativeHandleStoreGeneration,
    attempt_id: AttemptId,
    staged: Mutex<NativeHandleStoreSessionStage>,
}

struct NativeExecutionAttemptClaim {
    generation: NativeHandleStoreGeneration,
    attempt_id: AttemptId,
}

impl Drop for NativeExecutionAttemptClaim {
    fn drop(&mut self) {
        self.generation
            .state
            .active_attempts
            .lock()
            .remove(&self.attempt_id.0);
    }
}

#[derive(Default)]
struct NativeHandleStoreSessionStage {
    identifiers: Vec<String>,
    closed: bool,
}

impl fmt::Debug for RuntimeNativeHandleStoreSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeNativeHandleStoreSession")
            .field("identity", &self.generation.identity())
            .field("attempt_id", &self.attempt_id)
            .finish_non_exhaustive()
    }
}

impl Drop for RuntimeNativeHandleStoreSession {
    fn drop(&mut self) {
        let mut sessions = self.generation.state.sessions.lock();
        self.rollback_all();
        let registered_self = sessions
            .get(&self.attempt_id.0)
            .is_some_and(|session| std::ptr::eq(session.as_ptr(), self));
        if registered_self {
            sessions.remove(&self.attempt_id.0);
        }
    }
}

impl RuntimeNativeHandleStoreSession {
    fn checkpoint(&self) -> usize {
        self.staged.lock().identifiers.len()
    }

    fn rollback_from(&self, checkpoint: usize) {
        let identifiers = {
            let mut staged = self.staged.lock();
            if checkpoint >= staged.identifiers.len() {
                return;
            }
            staged.identifiers.split_off(checkpoint)
        };
        let mut data = self.generation.state.data.lock();
        for identifier in identifiers {
            if data
                .values
                .get(&identifier)
                .is_some_and(|record| !record.committed && record.published_by == self.attempt_id)
            {
                retire_stored_handle(&mut data, &identifier);
            }
        }
    }

    fn rollback_all(&self) {
        let identifiers = {
            let mut staged = self.staged.lock();
            staged.closed = true;
            std::mem::take(&mut staged.identifiers)
        };
        let mut data = self.generation.state.data.lock();
        for identifier in identifiers {
            if data
                .values
                .get(&identifier)
                .is_some_and(|record| !record.committed && record.published_by == self.attempt_id)
            {
                retire_stored_handle(&mut data, &identifier);
            }
        }
    }

    fn commit(&self) {
        let identifiers = {
            let mut staged = self.staged.lock();
            staged.closed = true;
            std::mem::take(&mut staged.identifiers)
        };
        let mut data = self.generation.state.data.lock();
        for identifier in identifiers {
            if let Some(record) = data.values.get_mut(&identifier)
                && record.published_by == self.attempt_id
                && !record.retired
            {
                record.committed = true;
            }
        }
    }

    fn validate_handle(
        &self,
        handle: &NativeOpaqueHandle,
        expected_type: &NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeHandleStoreError> {
        cancellation
            .check()
            .map_err(|_| NativeHandleStoreError::Cancelled)?;
        handle.validate()?;
        let identity = self.generation.identity();
        if handle.store_identity().store_id != identity.store_id {
            return Err(NativeHandleStoreError::WrongStore);
        }
        if handle.store_identity().generation_id != identity.generation_id {
            return Err(NativeHandleStoreError::WrongGeneration);
        }
        if handle.handle_type() != expected_type {
            return Err(NativeHandleStoreError::WrongType {
                expected: expected_type.type_id.clone(),
                actual: handle.handle_type().type_id.clone(),
            });
        }
        Ok(())
    }
}

impl NativeHandleStore for RuntimeNativeHandleStoreSession {
    fn identity(&self) -> NativeHandleStoreIdentity {
        self.generation.identity()
    }

    fn attempt_id(&self) -> AttemptId {
        self.attempt_id
    }

    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected_type: &NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
        self.validate_handle(handle, expected_type, cancellation)?;
        let mut data = self.generation.state.data.lock();
        let record = data
            .values
            .get_mut(handle.identifier())
            .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
        if record.retired {
            return Err(NativeHandleStoreError::Missing(
                handle.identifier().to_owned(),
            ));
        }
        if !record.committed && record.published_by != self.attempt_id {
            return Err(NativeHandleStoreError::Missing(
                handle.identifier().to_owned(),
            ));
        }
        if record.generation != handle.generation() {
            return Err(NativeHandleStoreError::WrongGeneration);
        }
        if record.handle_type != *expected_type {
            return Err(NativeHandleStoreError::WrongType {
                expected: record.handle_type.type_id.clone(),
                actual: expected_type.type_id.clone(),
            });
        }
        if Some(record.digest_sha256.as_str()) != handle.digest_sha256() {
            return Err(NativeHandleStoreError::DigestMismatch);
        }
        record.value.validate()?;
        let residency = record.value.residency()?;
        if record.value.handle_type()? != record.handle_type
            || record.value.digest_sha256() != record.digest_sha256
            || residency != record.residency
        {
            return Err(sim_nodes::NativeStoredPayloadError::ProjectionChanged.into());
        }
        record.resolved_roots = record.resolved_roots.checked_add(1).ok_or_else(|| {
            NativeHandleStoreError::Rejected(
                "native resolved payload root count overflowed".to_owned(),
            )
        })?;
        let payload = record.value.clone();
        drop(data);
        let resolved = NativeResolvedPayload::checked(
            payload,
            Arc::new(RuntimeResolvedPayloadRetention {
                generation: self.generation.clone(),
                handle: handle.clone(),
            }),
        )?;
        #[cfg(test)]
        self.generation.run_after_resolve_increment_hook();
        cancellation
            .check()
            .map_err(|_| NativeHandleStoreError::Cancelled)?;
        Ok(resolved)
    }

    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
        cancellation
            .check()
            .map_err(|_| NativeHandleStoreError::Cancelled)?;
        payload.validate()?;
        let handle_type = payload.handle_type()?;
        let digest_sha256 = payload.digest_sha256();
        let residency = payload.residency()?;
        let mut staged = self.staged.lock();
        if staged.closed {
            return Err(NativeHandleStoreError::Rejected(
                "native handle store attempt session is closed".to_owned(),
            ));
        }
        let mut data = self.generation.state.data.lock();
        let mut resident_delta = residency.exclusive_bytes();
        for allocation in residency.shared_allocations() {
            match data.allocations.get(allocation.id()) {
                Some(stored) => {
                    if stored.resident_bytes != allocation.resident_bytes() {
                        return Err(
                            sim_nodes::NativeStoredPayloadError::ResidentAllocationChanged.into(),
                        );
                    }
                    stored.handle_references.checked_add(1).ok_or_else(|| {
                        NativeHandleStoreError::Rejected(
                            "native resident allocation reference count overflowed".to_owned(),
                        )
                    })?;
                }
                None => {
                    resident_delta = resident_delta
                        .checked_add(allocation.resident_bytes())
                        .ok_or_else(|| {
                            NativeHandleStoreError::Rejected(
                                "native handle resident byte count overflowed".to_owned(),
                            )
                        })?;
                }
            }
        }
        let next_resident_bytes =
            data.resident_bytes
                .checked_add(resident_delta)
                .ok_or_else(|| {
                    NativeHandleStoreError::Rejected(
                        "native handle resident byte count overflowed".to_owned(),
                    )
                })?;
        if data.values.len() >= self.generation.state.capacity
            || next_resident_bytes > self.generation.state.byte_capacity
        {
            return Err(NativeHandleStoreError::Rejected(
                "native handle store capacity is exhausted".to_owned(),
            ));
        }
        let generation = self
            .generation
            .state
            .next_generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                generation.checked_add(1).filter(|next| *next != 0)
            })
            .map_err(|_| {
                NativeHandleStoreError::Rejected(
                    "native handle generation was exhausted".to_owned(),
                )
            })?;
        let identifier = format!("native-{generation:016x}");
        let handle = NativeOpaqueHandle::new(
            handle_type.clone(),
            self.generation.identity(),
            identifier.clone(),
            generation,
            Some(digest_sha256.clone()),
        )?;
        for allocation in residency.shared_allocations() {
            match data.allocations.entry(allocation.id().clone()) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().handle_references += 1;
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(StoredResidentAllocation {
                        resident_bytes: allocation.resident_bytes(),
                        handle_references: 1,
                    });
                }
            }
        }
        data.values.insert(
            identifier.clone(),
            StoredNativeHandle {
                handle_type,
                generation,
                digest_sha256,
                value: Arc::new(payload),
                committed: false,
                published_by: self.attempt_id,
                residency,
                roots: 0,
                resolved_roots: 0,
                retired: false,
            },
        );
        data.resident_bytes = next_resident_bytes;
        drop(data);
        staged.identifiers.push(identifier);
        #[cfg(test)]
        self.generation.run_after_publish_insert_hook();
        if cancellation.is_cancelled() {
            staged.identifiers.pop();
            let mut data = self.generation.state.data.lock();
            retire_stored_handle(&mut data, handle.identifier());
            return Err(NativeHandleStoreError::Cancelled);
        }
        drop(staged);
        Ok(handle)
    }

    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeHandleStoreError> {
        self.validate_handle(handle, handle.handle_type(), cancellation)?;
        let mut staged = self.staged.lock();
        if staged.closed {
            return Err(NativeHandleStoreError::Rejected(
                "native handle store attempt session is closed".to_owned(),
            ));
        }
        let mut data = self.generation.state.data.lock();
        let removable = data.values.get(handle.identifier()).is_some_and(|record| {
            !record.committed && !record.retired && record.published_by == self.attempt_id
        });
        if !removable {
            return Err(NativeHandleStoreError::Rejected(
                "native handles may only be revoked by their publishing attempt before commit"
                    .to_owned(),
            ));
        }
        if !retire_stored_handle(&mut data, handle.identifier()) {
            return Err(NativeHandleStoreError::Missing(
                handle.identifier().to_owned(),
            ));
        }
        staged
            .identifiers
            .retain(|identifier| identifier != handle.identifier());
        Ok(())
    }
}

pub trait EffectCoordinator: Send + Sync {
    fn node_service(
        &self,
        identity: NativeNodeServiceIdentity,
        prompt_id: PromptId,
    ) -> Result<Arc<dyn NativePreparedEffectService>, String>;
    fn prepared_effect(
        &self,
        ticket: &PreparedEffectRequest,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: &NodeId,
    ) -> Result<PreparedEffect, String>;
    fn commit_batch(
        &self,
        effects: &[PreparedEffect],
        cancellation: &CancellationToken,
    ) -> Result<(), String>;
    fn rollback_batch(&self, effects: &[PreparedEffect]) -> Result<(), String>;
}

#[cfg(test)]
fn prepared_effect_transaction_id(
    identity: &NativeNodeServiceIdentity,
    ordinal: u64,
    request_digest_sha256: &str,
) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.prepared-effect-transaction.v1");
    hasher.update(identity.service_id().as_bytes());
    hasher.update(identity.attempt_id().0.as_bytes());
    hasher.update(identity.node_id().0.as_bytes());
    hasher.update(ordinal.to_le_bytes());
    hasher.update(request_digest_sha256.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes)
}

fn node_effect_service_id(prompt_id: PromptId, attempt_id: AttemptId, node_id: &NodeId) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.node-effect-service.v1");
    hasher.update(prompt_id.0.as_bytes());
    hasher.update(attempt_id.0.as_bytes());
    hasher.update(node_id.0.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes)
}

fn rollback_node_prepared_effects(
    service: &dyn NativePreparedEffectService,
    primary: ExecutionError,
) -> ExecutionError {
    match service.rollback_all_prepared() {
        Ok(()) => primary,
        Err(rollback) => ExecutionError::Effect(format!(
            "{primary}; rolling back the node's prepared effects failed: {rollback}"
        )),
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
struct EffectCoordinatorCalls {
    prepared: Vec<PreparedEffect>,
    prepared_history: Vec<Uuid>,
    node_rolled_back: Vec<Uuid>,
    committed_batches: Vec<Vec<PreparedEffect>>,
    rolled_back_batches: Vec<Vec<PreparedEffect>>,
}

#[cfg(test)]
#[derive(Default)]
struct RecordingEffectCoordinator {
    calls: Arc<Mutex<EffectCoordinatorCalls>>,
}

#[cfg(test)]
#[derive(Debug)]
struct RecordingPreparedEffectService {
    identity: NativeNodeServiceIdentity,
    prompt_id: PromptId,
    ordinal: AtomicU64,
    calls: Arc<Mutex<EffectCoordinatorCalls>>,
}

#[cfg(test)]
impl NativePreparedEffectService for RecordingPreparedEffectService {
    fn identity(&self) -> &NativeNodeServiceIdentity {
        &self.identity
    }

    fn maximum_output_bytes(&self) -> u64 {
        2 * 1024 * 1024 * 1024
    }

    fn prepare_output(
        &self,
        request: NativeOutputEffectRequest,
        cancellation: &CancellationToken,
    ) -> Result<PreparedEffectRequest, sim_nodes::NativeEffectServiceError> {
        cancellation
            .check()
            .map_err(|_| sim_nodes::NativeEffectServiceError::Cancelled)?;
        let ordinal = self
            .ordinal
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| sim_nodes::NativeEffectServiceError::Rejected)?;
        let transaction_id = prepared_effect_transaction_id(
            &self.identity,
            ordinal,
            request.request_digest_sha256(),
        );
        let ticket = PreparedEffectRequest::checked(
            self.identity.service_id(),
            transaction_id,
            NativePreparedEffectKind::Output,
            request.request_digest_sha256(),
        )
        .map_err(|_| sim_nodes::NativeEffectServiceError::Rejected)?;
        let prepared = PreparedEffect {
            prompt_id: self.prompt_id,
            attempt_id: self.identity.attempt_id(),
            node_id: self.identity.node_id().clone(),
            service_id: self.identity.service_id(),
            transaction_id,
            kind: NativePreparedEffectKind::Output,
            request_digest_sha256: request.request_digest_sha256().to_owned(),
        };
        let mut calls = self.calls.lock();
        calls.prepared_history.push(transaction_id);
        calls.prepared.push(prepared.clone());
        drop(calls);
        if cancellation.check().is_err() {
            self.calls
                .lock()
                .prepared
                .retain(|candidate| candidate != &prepared);
            return Err(sim_nodes::NativeEffectServiceError::Cancelled);
        }
        Ok(ticket)
    }

    fn rollback_prepared(
        &self,
        request: &PreparedEffectRequest,
    ) -> Result<(), sim_nodes::NativeEffectServiceError> {
        if request.service_id() != self.identity.service_id() {
            return Err(sim_nodes::NativeEffectServiceError::InvalidTicket);
        }
        let mut calls = self.calls.lock();
        let Some(index) = calls.prepared.iter().position(|prepared| {
            prepared.transaction_id == request.transaction_id()
                && prepared.service_id == request.service_id()
                && prepared.kind == request.kind()
                && prepared.request_digest_sha256 == request.request_digest_sha256()
        }) else {
            return Err(sim_nodes::NativeEffectServiceError::InvalidTicket);
        };
        let prepared = calls.prepared.remove(index);
        calls.node_rolled_back.push(prepared.transaction_id);
        Ok(())
    }

    fn rollback_all_prepared(&self) -> Result<(), sim_nodes::NativeEffectServiceError> {
        let mut calls = self.calls.lock();
        let mut retained = Vec::with_capacity(calls.prepared.len());
        let mut rolled_back = Vec::new();
        for prepared in calls.prepared.drain(..) {
            if prepared.service_id == self.identity.service_id()
                && prepared.attempt_id == self.identity.attempt_id()
                && prepared.node_id == *self.identity.node_id()
            {
                rolled_back.push(prepared.transaction_id);
            } else {
                retained.push(prepared);
            }
        }
        calls.prepared = retained;
        calls.node_rolled_back.extend(rolled_back);
        Ok(())
    }
}

#[cfg(test)]
impl RecordingEffectCoordinator {
    pub fn prepared_history(&self) -> BTreeSet<Uuid> {
        self.calls.lock().prepared_history.iter().copied().collect()
    }

    pub fn node_rolled_back(&self) -> BTreeSet<Uuid> {
        self.calls.lock().node_rolled_back.iter().copied().collect()
    }

    pub fn prepared(&self) -> BTreeSet<Uuid> {
        self.calls
            .lock()
            .prepared
            .iter()
            .map(|effect| effect.transaction_id)
            .collect()
    }

    pub fn committed(&self) -> BTreeSet<Uuid> {
        self.calls
            .lock()
            .committed_batches
            .iter()
            .flatten()
            .map(|effect| effect.transaction_id)
            .collect()
    }

    pub fn rolled_back(&self) -> BTreeSet<Uuid> {
        self.calls
            .lock()
            .rolled_back_batches
            .iter()
            .flatten()
            .map(|effect| effect.transaction_id)
            .collect()
    }
}

#[cfg(test)]
impl EffectCoordinator for RecordingEffectCoordinator {
    fn node_service(
        &self,
        identity: NativeNodeServiceIdentity,
        prompt_id: PromptId,
    ) -> Result<Arc<dyn NativePreparedEffectService>, String> {
        Ok(Arc::new(RecordingPreparedEffectService {
            identity,
            prompt_id,
            ordinal: AtomicU64::new(0),
            calls: self.calls.clone(),
        }))
    }

    fn prepared_effect(
        &self,
        ticket: &PreparedEffectRequest,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: &NodeId,
    ) -> Result<PreparedEffect, String> {
        self.calls
            .lock()
            .prepared
            .iter()
            .find(|effect| {
                effect.prompt_id == prompt_id
                    && effect.attempt_id == attempt_id
                    && &effect.node_id == node_id
                    && effect.service_id == ticket.service_id()
                    && effect.transaction_id == ticket.transaction_id()
                    && effect.kind == ticket.kind()
                    && effect.request_digest_sha256 == ticket.request_digest_sha256()
            })
            .cloned()
            .ok_or_else(|| "prepared effect ticket is absent or belongs to another node".to_owned())
    }

    fn commit_batch(
        &self,
        effects: &[PreparedEffect],
        cancellation: &CancellationToken,
    ) -> Result<(), String> {
        cancellation
            .check()
            .map_err(|_| "effect commit was cancelled".to_owned())?;
        self.calls.lock().committed_batches.push(effects.to_vec());
        Ok(())
    }

    fn rollback_batch(&self, effects: &[PreparedEffect]) -> Result<(), String> {
        self.calls.lock().rolled_back_batches.push(effects.to_vec());
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ExecutionError {
    #[error("native node implementation `{0}` is invalid")]
    InvalidNodeImplementation(String),
    #[error("native node implementation `{0}` is already registered")]
    DuplicateNodeImplementation(String),
    #[error("compiled node {node:?} has no native implementation for `{class_type}`")]
    MissingNodeImplementation { node: NodeId, class_type: String },
    #[error(
        "compiled node {node:?} expects `{class_type}` version `{expected}`, but the registered implementation is `{actual}`"
    )]
    ImplementationVersionMismatch {
        node: NodeId,
        class_type: String,
        expected: String,
        actual: String,
    },
    #[error(
        "native node `{class_type}` descriptor expects implementation version `{expected}`, but registration supplied `{actual}`"
    )]
    DescriptorImplementationVersionMismatch {
        class_type: String,
        expected: String,
        actual: String,
    },
    #[error("compiled plan references unknown node {0:?}")]
    UnknownNode(NodeId),
    #[error("runtime dependency cycle reached node {0:?}")]
    DependencyCycle(NodeId),
    #[error("node {node:?} demanded invalid lazy input `{input}`")]
    InvalidLazyDemand { node: NodeId, input: String },
    #[error("node {node:?} output {output_index} is missing")]
    MissingOutput { node: NodeId, output_index: usize },
    #[error("node {node:?} returned {actual} outputs, expected {expected}")]
    OutputArity {
        node: NodeId,
        expected: usize,
        actual: usize,
    },
    #[error("node {node:?} output {output_index} violates its declared type or list cardinality")]
    InvalidOutput { node: NodeId, output_index: usize },
    #[error("node {node:?} returned transactional effects while declared {effect:?}")]
    UnexpectedEffect { node: NodeId, effect: EffectClass },
    #[error("node {node:?} was blocked: {reason}")]
    Blocked { node: NodeId, reason: String },
    #[error("node {node:?} failed: {failure}")]
    Node { node: NodeId, failure: NodeFailure },
    #[error("node {node:?} was interrupted: {failure}")]
    Interrupted { node: NodeId, failure: NodeFailure },
    #[error("execution was cancelled")]
    Cancelled,
    #[error("execution attempt {0:?} is already active")]
    AttemptAlreadyActive(AttemptId),
    #[error("expansion depth exceeds {MAX_EXPANSION_DEPTH}")]
    ExpansionDepth,
    #[error("expanded plan does not contain output node {0:?}")]
    InvalidExpansionOutput(NodeId),
    #[error("expanded prompt failed compilation: {0}")]
    ExpansionCompile(PromptCompileError),
    #[error("native handle store failed: {0}")]
    HandleStore(String),
    #[error("cache operation failed: {0}")]
    Cache(String),
    #[error("effect coordination failed: {0}")]
    Effect(String),
    #[error("execution precommit validation failed: {0}")]
    Precommit(String),
    #[error("execution event publication failed: {0}")]
    EventBus(String),
    #[error("execution event sequence is exhausted")]
    SequenceExhausted,
    #[error("execution progress exceeds the supported counter range")]
    ProgressOverflow,
    #[error("execution expansion scope sequence is exhausted")]
    ExpansionSequenceExhausted,
    #[error("node {node:?} returned too many transactional effects")]
    TooManyEffects { node: NodeId },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub profile_id: ProfileId,
    pub prompt_id: PromptId,
    pub attempt_id: AttemptId,
    pub state: AttemptState,
    pub outputs: BTreeMap<NodeId, Vec<NativeValue>>,
    #[serde(default)]
    pub ui_outputs: BTreeMap<NodeId, Value>,
    pub events: Vec<AttemptEvent>,
    pub cache_hits: usize,
    pub error: Option<String>,
    #[serde(skip)]
    pub(crate) handle_lease: Option<NativeHandleLease>,
}

impl PartialEq for ExecutionReport {
    fn eq(&self, other: &Self) -> bool {
        self.profile_id == other.profile_id
            && self.prompt_id == other.prompt_id
            && self.attempt_id == other.attempt_id
            && self.state == other.state
            && self.outputs == other.outputs
            && self.ui_outputs == other.ui_outputs
            && self.events == other.events
            && self.cache_hits == other.cache_hits
            && self.error == other.error
    }
}

pub struct ExecutionEngine {
    profile_id: ProfileId,
    nodes: Arc<NativeNodeRegistry>,
    cache: Arc<Mutex<NativeCache>>,
    effects: Arc<dyn EffectCoordinator>,
    event_bus: Option<ExecutionEventBus>,
    registry_version: String,
    backend: String,
    dtype_policy: String,
    configuration_token: String,
    scratch: ScratchReservation,
    compute_backend: Option<Arc<CpuBackend>>,
    shader_executor: Option<Arc<dyn NativeShaderExecutor>>,
    ltxv_preprocess_service: Option<Arc<dyn NativeLtxvPreprocessService>>,
    webm_encode_service: Option<Arc<dyn NativeWebmEncodeService>>,
    component_h264_mp4_backing_service: Option<Arc<dyn NativeComponentH264Mp4BackingService>>,
    asset_resolvers: Option<Arc<NativeAssetResolverRegistry>>,
    model_resource_services: Option<NativeModelResourceServiceFactory>,
    model_training_service: Option<Arc<dyn NativeModelTrainingService>>,
    precommit_validator: Option<Arc<ExecutionPrecommitValidator>>,
    handle_store_generation: NativeHandleStoreGeneration,
}

type ExecutionPrecommitValidator = dyn Fn(
        &BTreeMap<NodeId, Vec<NativeValue>>,
        &BTreeMap<NodeId, Value>,
        &dyn NativeHandleStore,
        &CancellationToken,
    ) -> Result<(), ExecutionError>
    + Send
    + Sync;

impl ExecutionEngine {
    pub fn new_with_workspace_authorization(
        profile_id: ProfileId,
        nodes: Arc<NativeNodeRegistry>,
        cache: Arc<Mutex<NativeCache>>,
        effects: Arc<dyn EffectCoordinator>,
        registry_version: impl Into<String>,
        scratch: ScratchReservation,
    ) -> Result<Self, ExecutionError> {
        Self::new_with_handle_store_generation(
            profile_id,
            nodes,
            cache,
            effects,
            registry_version,
            scratch,
            NativeHandleStoreGeneration::new()
                .map_err(|error| ExecutionError::HandleStore(error.to_string()))?,
        )
    }

    pub fn new_with_handle_store_generation(
        profile_id: ProfileId,
        nodes: Arc<NativeNodeRegistry>,
        cache: Arc<Mutex<NativeCache>>,
        effects: Arc<dyn EffectCoordinator>,
        registry_version: impl Into<String>,
        scratch: ScratchReservation,
        handle_store_generation: NativeHandleStoreGeneration,
    ) -> Result<Self, ExecutionError> {
        let registry_version = registry_version.into();
        if registry_version.is_empty() {
            return Err(ExecutionError::Cache(
                "registry version is empty".to_owned(),
            ));
        }
        Ok(Self {
            profile_id,
            nodes,
            cache,
            effects,
            event_bus: None,
            registry_version,
            backend: "cpu".to_owned(),
            dtype_policy: "default".to_owned(),
            configuration_token: "default".to_owned(),
            scratch,
            compute_backend: None,
            shader_executor: None,
            ltxv_preprocess_service: None,
            webm_encode_service: None,
            component_h264_mp4_backing_service: None,
            asset_resolvers: None,
            model_resource_services: None,
            model_training_service: None,
            precommit_validator: None,
            handle_store_generation,
        })
    }

    pub fn handle_store_generation(&self) -> &NativeHandleStoreGeneration {
        &self.handle_store_generation
    }

    pub fn with_event_bus(mut self, event_bus: ExecutionEventBus) -> Self {
        self.event_bus = Some(event_bus);
        self
    }

    pub(crate) fn with_precommit_validator(
        mut self,
        validator: Arc<ExecutionPrecommitValidator>,
    ) -> Self {
        self.precommit_validator = Some(validator);
        self
    }

    pub fn with_compute_backend(
        mut self,
        backend: Arc<CpuBackend>,
    ) -> Result<Self, ExecutionError> {
        backend
            .validate_scratch_reservation(&self.scratch)
            .map_err(|error| ExecutionError::Cache(error.to_string()))?;
        self.compute_backend = Some(backend);
        Ok(self)
    }

    pub fn with_shader_executor(mut self, shader: Arc<dyn NativeShaderExecutor>) -> Self {
        self.shader_executor = Some(shader);
        self
    }

    pub fn with_model_training_service(
        mut self,
        service: Arc<dyn NativeModelTrainingService>,
    ) -> Self {
        self.model_training_service = Some(service);
        self
    }

    pub fn with_ltxv_preprocess_service(
        mut self,
        service: Arc<dyn NativeLtxvPreprocessService>,
    ) -> Self {
        self.ltxv_preprocess_service = Some(service);
        self
    }

    pub fn with_webm_encode_service(mut self, service: Arc<dyn NativeWebmEncodeService>) -> Self {
        self.webm_encode_service = Some(service);
        self
    }

    pub fn with_component_h264_mp4_backing_service(
        mut self,
        service: Arc<dyn NativeComponentH264Mp4BackingService>,
    ) -> Self {
        self.component_h264_mp4_backing_service = Some(service);
        self
    }

    pub fn with_asset_resolvers(
        mut self,
        asset_resolvers: Arc<NativeAssetResolverRegistry>,
    ) -> Self {
        self.asset_resolvers = Some(asset_resolvers);
        self
    }

    pub(crate) fn with_model_source_transport(
        mut self,
        binding: crate::NativeModelSourceWorkerBinding,
        transport: NativeModelSourceTransport,
        backend: Arc<CpuBackend>,
    ) -> Result<Self, ExecutionError> {
        if binding.service_id().is_nil()
            || binding.service_generation() == 0
            || binding.attempt_generation() == 0
            || binding.node_generation() == 0
        {
            return Err(ExecutionError::Cache(
                "native model-source service binding is invalid".to_owned(),
            ));
        }
        backend
            .validate_scratch_reservation(&self.scratch)
            .map_err(|error| ExecutionError::Cache(error.to_string()))?;
        #[cfg(not(feature = "test-support"))]
        let construction_profile = NativeModelConstructionProfile::SourceExact;
        #[cfg(feature = "test-support")]
        let construction_profile = if binding.reduced_fixture_model_resources() {
            NativeModelConstructionProfile::ReducedFixture
        } else {
            NativeModelConstructionProfile::SourceExact
        };
        self.model_resource_services = Some(NativeModelResourceServiceFactory {
            binding,
            transport,
            backend,
            construction_profile,
        });
        Ok(self)
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn with_reduced_fixture_model_resources(mut self) -> Self {
        if let Some(factory) = self.model_resource_services.as_mut() {
            factory.construction_profile = NativeModelConstructionProfile::ReducedFixture;
        }
        self
    }

    pub fn seal_asset_for_node(
        &self,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        node_id: NodeId,
        identity: AssetIdentity,
        source_type_id: impl Into<String>,
    ) -> Result<NativeAssetReference, NativeAssetServiceError> {
        let service_identity = NativeNodeServiceIdentity::checked(
            node_effect_service_id(prompt_id, attempt_id, &node_id),
            attempt_id,
            node_id,
        )
        .map_err(|_| NativeAssetServiceError::InvalidReference)?;
        self.asset_resolvers
            .as_ref()
            .ok_or(NativeAssetServiceError::Unavailable)?
            .seal_for_node(&service_identity, identity, source_type_id)
    }

    pub fn with_backend(mut self, backend: impl Into<String>) -> Result<Self, ExecutionError> {
        self.backend = nonempty_cache_dimension("backend", backend.into())?;
        Ok(self)
    }

    pub fn with_dtype_policy(
        mut self,
        dtype_policy: impl Into<String>,
    ) -> Result<Self, ExecutionError> {
        self.dtype_policy = nonempty_cache_dimension("dtype policy", dtype_policy.into())?;
        Ok(self)
    }

    pub fn with_configuration_token(
        mut self,
        configuration_token: impl Into<String>,
    ) -> Result<Self, ExecutionError> {
        self.configuration_token =
            nonempty_cache_dimension("configuration token", configuration_token.into())?;
        Ok(self)
    }

    pub fn with_native_compile_policy(
        mut self,
        policy: &NativeCompilePolicy,
    ) -> Result<Self, ExecutionError> {
        self.backend = policy.backend().cache_dimension().to_owned();
        self.configuration_token = policy
            .cache_token()
            .map_err(|error| ExecutionError::Cache(error.to_string()))?;
        Ok(self)
    }

    fn publish_cache_batch(
        &self,
        entries: Vec<(CacheKey, CacheEntry, Option<NativeHandleLease>)>,
        cancellation: &CancellationToken,
    ) -> Result<(), ExecutionError> {
        if cancellation.is_cancelled() {
            return Err(ExecutionError::Cancelled);
        }
        let mut cache = self.cache.lock();
        let prior_cache = cache.clone();
        if !cache.insert_batch_with_handle_leases(entries) {
            *cache = prior_cache;
            return Err(ExecutionError::Cache(
                "native cache handle leases did not cover the committed batch".to_owned(),
            ));
        }
        #[cfg(test)]
        self.handle_store_generation.run_after_cache_insert_hook();
        if cancellation.is_cancelled() {
            *cache = prior_cache;
            return Err(ExecutionError::Cancelled);
        }
        Ok(())
    }

    pub async fn execute(
        &self,
        plan: &CompiledPlan,
        attempt_id: AttemptId,
        cancellation: CancellationToken,
    ) -> ExecutionReport {
        let _attempt_claim = match self.handle_store_generation.try_claim_attempt(attempt_id) {
            Some(claim) => claim,
            None => {
                return ExecutionReport {
                    profile_id: self.profile_id,
                    prompt_id: plan.prompt_id,
                    attempt_id,
                    state: AttemptState::Failed,
                    outputs: BTreeMap::new(),
                    ui_outputs: BTreeMap::new(),
                    events: Vec::new(),
                    cache_hits: 0,
                    error: Some(ExecutionError::AttemptAlreadyActive(attempt_id).to_string()),
                    handle_lease: None,
                };
            }
        };
        let handle_store = self.handle_store_generation.session(attempt_id);
        let mut report_handle_lease = None;
        let mut state = RunState::new(
            self.profile_id,
            plan.prompt_id,
            attempt_id,
            cancellation,
            handle_store,
        );
        let result = async {
            state.emit(self.event_bus.as_ref(), None, AttemptEventKind::Started)?;
            self.run_plan(plan, &mut state, 0).await?;
            if state.cancellation.is_cancelled() {
                return Err(ExecutionError::Cancelled);
            }
            if let Some(validator) = &self.precommit_validator {
                validator(
                    &state.outputs,
                    &state.ui_outputs,
                    state.handle_store.as_ref(),
                    &state.cancellation,
                )?;
            }
            if state.cancellation.is_cancelled() {
                return Err(ExecutionError::Cancelled);
            }
            state.handle_store.commit();
            let pending_cache_entries = std::mem::take(&mut state.pending_cache_entries);
            let mut leased_cache_entries = Vec::with_capacity(pending_cache_entries.len());
            for (key, entry) in pending_cache_entries {
                let mut handles = Vec::new();
                for output in &entry.outputs {
                    collect_native_value_handles(output, &mut handles);
                }
                let lease = match self.handle_store_generation.acquire_lease(handles.iter()) {
                    Ok(lease) => lease,
                    Err(error) => {
                        drop(leased_cache_entries);
                        self.handle_store_generation
                            .collect_unrooted_attempt(attempt_id);
                        return Err(ExecutionError::HandleStore(error.to_string()));
                    }
                };
                leased_cache_entries.push((key, entry, lease));
            }
            let mut report_handles = Vec::new();
            for values in state.outputs.values() {
                for output in values {
                    collect_native_value_handles(output, &mut report_handles);
                }
            }
            let report_lease = match self
                .handle_store_generation
                .acquire_lease(report_handles.iter())
            {
                Ok(lease) => lease,
                Err(error) => {
                    drop(leased_cache_entries);
                    self.handle_store_generation
                        .collect_unrooted_attempt(attempt_id);
                    return Err(ExecutionError::HandleStore(error.to_string()));
                }
            };
            if state.cancellation.is_cancelled() {
                drop(report_lease);
                drop(leased_cache_entries);
                self.handle_store_generation
                    .collect_unrooted_attempt(attempt_id);
                return Err(ExecutionError::Cancelled);
            }
            if let Err(error) = self
                .effects
                .commit_batch(&state.prepared_effects, &state.cancellation)
            {
                drop(report_lease);
                drop(leased_cache_entries);
                self.handle_store_generation
                    .collect_unrooted_attempt(attempt_id);
                return Err(ExecutionError::Effect(error));
            }
            if state.cancellation.is_cancelled() {
                drop(report_lease);
                drop(leased_cache_entries);
                self.handle_store_generation
                    .collect_unrooted_attempt(attempt_id);
                return Err(ExecutionError::Cancelled);
            }
            if let Err(error) = self.publish_cache_batch(leased_cache_entries, &state.cancellation)
            {
                drop(report_lease);
                self.handle_store_generation
                    .collect_unrooted_attempt(attempt_id);
                return Err(error);
            }
            report_handle_lease = report_lease;
            self.handle_store_generation
                .collect_unrooted_attempt(attempt_id);
            if let Err(error) =
                state.emit(self.event_bus.as_ref(), None, AttemptEventKind::Succeeded)
            {
                state.diagnostics.push(error.to_string());
            }
            Ok(())
        }
        .await;

        if let Some(asset_resolvers) = &self.asset_resolvers {
            asset_resolvers.retire_attempt(attempt_id);
        }

        match result {
            Ok(()) => ExecutionReport {
                profile_id: self.profile_id,
                prompt_id: plan.prompt_id,
                attempt_id,
                state: AttemptState::Succeeded,
                outputs: state.outputs,
                ui_outputs: state.ui_outputs,
                events: state.events,
                cache_hits: state.cache_hits,
                error: (!state.diagnostics.is_empty()).then(|| state.diagnostics.join("; ")),
                handle_lease: report_handle_lease,
            },
            Err(error) => {
                state.handle_store.rollback_all();
                let mut error_message = error.to_string();
                if let Err(rollback_error) = self.effects.rollback_batch(&state.prepared_effects) {
                    error_message.push_str("; rollback failed: ");
                    error_message.push_str(&rollback_error);
                }
                let (terminal_state, terminal_event) = match &error {
                    ExecutionError::Cancelled => {
                        (AttemptState::Cancelled, AttemptEventKind::Cancelled)
                    }
                    ExecutionError::Interrupted { .. } => (
                        AttemptState::Interrupted,
                        AttemptEventKind::Interrupted {
                            reason: error.to_string(),
                        },
                    ),
                    _ => (
                        AttemptState::Failed,
                        AttemptEventKind::Failed {
                            failure: crate::ExecutionFailure::new(
                                "native_execution_failed",
                                error.to_string(),
                            ),
                        },
                    ),
                };
                if let Err(event_error) = state.emit(self.event_bus.as_ref(), None, terminal_event)
                {
                    error_message.push_str("; terminal event failed: ");
                    error_message.push_str(&event_error.to_string());
                }
                ExecutionReport {
                    profile_id: self.profile_id,
                    prompt_id: plan.prompt_id,
                    attempt_id,
                    state: terminal_state,
                    outputs: BTreeMap::new(),
                    ui_outputs: BTreeMap::new(),
                    events: state.events,
                    cache_hits: state.cache_hits,
                    error: Some(error_message),
                    handle_lease: None,
                }
            }
        }
    }

    fn run_plan<'a>(
        &'a self,
        plan: &'a CompiledPlan,
        state: &'a mut RunState,
        expansion_depth: usize,
    ) -> BoxFuture<'a, Result<(), ExecutionError>> {
        Box::pin(async move {
            if expansion_depth > MAX_EXPANSION_DEPTH {
                return Err(ExecutionError::ExpansionDepth);
            }
            for output_node in &plan.output_nodes {
                self.run_node(plan, output_node, state, expansion_depth)
                    .await?;
            }
            Ok(())
        })
    }

    fn run_node<'a>(
        &'a self,
        plan: &'a CompiledPlan,
        node_id: &'a NodeId,
        state: &'a mut RunState,
        expansion_depth: usize,
    ) -> BoxFuture<'a, Result<Vec<NativeValue>, ExecutionError>> {
        Box::pin(async move {
            if state.cancellation.is_cancelled() {
                return Err(ExecutionError::Cancelled);
            }
            if let Some(outputs) = state.outputs.get(node_id) {
                return Ok(outputs.clone());
            }
            if !state.visiting.insert(node_id.clone()) {
                return Err(ExecutionError::DependencyCycle(node_id.clone()));
            }
            let result = self
                .run_node_inner(plan, node_id, state, expansion_depth)
                .await;
            state.visiting.remove(node_id);
            result
        })
    }

    async fn run_node_inner(
        &self,
        plan: &CompiledPlan,
        node_id: &NodeId,
        state: &mut RunState,
        expansion_depth: usize,
    ) -> Result<Vec<NativeValue>, ExecutionError> {
        let node = plan
            .nodes
            .get(node_id)
            .cloned()
            .ok_or_else(|| ExecutionError::UnknownNode(node_id.clone()))?;
        let implementation = self.nodes.node(&node.class_type).ok_or_else(|| {
            ExecutionError::MissingNodeImplementation {
                node: node_id.clone(),
                class_type: node.class_type.clone(),
            }
        })?;
        if implementation.implementation_version() != node.descriptor.implementation_version {
            return Err(ExecutionError::ImplementationVersionMismatch {
                node: node_id.clone(),
                class_type: node.class_type.clone(),
                expected: node.descriptor.implementation_version.clone(),
                actual: implementation.implementation_version().to_owned(),
            });
        }
        let service_identity = NativeNodeServiceIdentity::checked(
            node_effect_service_id(plan.prompt_id, state.attempt_id, node_id),
            state.attempt_id,
            node_id.clone(),
        )
        .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        let asset_resolver = self
            .asset_resolvers
            .as_ref()
            .map(|registry| registry.node_service(service_identity.clone()));
        let effect_service = self
            .effects
            .node_service(service_identity.clone(), plan.prompt_id)
            .map_err(ExecutionError::Effect)?;
        let compute = self
            .compute_backend
            .as_ref()
            .map(|backend| {
                NativeNodeComputeSession::checked(
                    service_identity.clone(),
                    backend.clone(),
                    StreamId::DEFAULT,
                    &self.scratch,
                )
            })
            .transpose()
            .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        let mut services =
            NativeNodeServices::checked(asset_resolver, Some(effect_service.clone()), compute)
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        if let Some(factory) = &self.model_resource_services {
            services = services
                .with_model_resources(factory.node_service(service_identity.clone()))
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if let Some(service) = &self.model_training_service {
            services = services
                .with_model_training(service.clone())
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if node.class_type == "SDPoseKeypointExtractor" {
            services = services
                .with_sdpose(Arc::new(NativeRuntimeSdPoseService::new(
                    service_identity.clone(),
                )))
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if matches!(
            node.class_type.as_str(),
            "TorchCompileModel"
                | "UNetSelfAttentionMultiply"
                | "UNetCrossAttentionMultiply"
                | "UNetTemporalAttentionMultiply"
        ) {
            let capabilities = self
                .compute_backend
                .as_ref()
                .map(|backend| backend.capabilities().clone());
            services = services
                .with_model_compile(Arc::new(NativeRuntimeModelCompileService::new(
                    service_identity.clone(),
                    capabilities,
                )))
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if matches!(
            node.class_type.as_str(),
            "TextGenerate" | "TextGenerateLTX2Prompt"
        ) {
            services = services
                .with_text_generation(Arc::new(NativeRuntimeTextGenerationService::new(
                    service_identity.clone(),
                )))
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if let Some(shader) = &self.shader_executor {
            services = services.with_shader(shader.clone());
        }
        if let Some(ltxv_preprocess) = &self.ltxv_preprocess_service {
            services = services
                .with_ltxv_preprocess(ltxv_preprocess.clone())
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if let Some(webm_encode) = &self.webm_encode_service {
            services = services
                .with_webm_encode(webm_encode.clone())
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if let Some(component_h264_mp4_backing) = &self.component_h264_mp4_backing_service {
            services = services
                .with_component_h264_mp4_backing(component_h264_mp4_backing.clone())
                .map_err(|error| ExecutionError::Effect(error.to_string()))?;
        }
        if let Some(provider_execution) = &plan.provider_execution {
            services = services.with_provider_execution(
                NativeProviderExecutionIdentity::checked(
                    provider_execution.compiled_plan_sha256().to_owned(),
                )
                .map_err(|error| ExecutionError::Effect(error.to_string()))?,
            );
        }
        let context = NodeContext::new_with_services(
            plan.prompt_id,
            state.attempt_id,
            node_id.clone(),
            state.cancellation.clone(),
            self.scratch.clone(),
            state.handle_store.clone(),
            services,
        )
        .map_err(|error| ExecutionError::HandleStore(error.to_string()))?;
        let mut inputs = BTreeMap::new();
        for (name, binding) in &node.inputs {
            match binding {
                InputBinding::Literal { value } => {
                    inputs.insert(name.clone(), value.clone());
                }
                InputBinding::Link {
                    source,
                    output_index,
                    lazy: false,
                    mode,
                } => {
                    let outputs = self.run_node(plan, source, state, expansion_depth).await?;
                    inputs.insert(
                        name.clone(),
                        linked_value(source, *output_index, *mode, &outputs)?,
                    );
                }
                InputBinding::Link { lazy: true, .. } => {}
            }
        }
        let node_inputs = assemble_structured_inputs(&node.descriptor, &inputs)?;
        let demanded = implementation
            .demanded_lazy_inputs(&context, &node_inputs)
            .map_err(|failure| execution_node_failure(node_id.clone(), failure))?;
        for name in demanded {
            let binding =
                node.inputs
                    .get(&name)
                    .ok_or_else(|| ExecutionError::InvalidLazyDemand {
                        node: node_id.clone(),
                        input: name.clone(),
                    })?;
            let InputBinding::Link {
                source,
                output_index,
                lazy: true,
                mode,
            } = binding
            else {
                return Err(ExecutionError::InvalidLazyDemand {
                    node: node_id.clone(),
                    input: name,
                });
            };
            let outputs = self.run_node(plan, source, state, expansion_depth).await?;
            inputs.insert(name, linked_value(source, *output_index, *mode, &outputs)?);
        }

        let node_inputs = assemble_structured_inputs(&node.descriptor, &inputs)?;

        if context.cancellation.is_cancelled() {
            return Err(ExecutionError::Cancelled);
        }
        let change_token = implementation
            .cache_change_token(&node_inputs)
            .map_err(|failure| execution_node_failure(node_id.clone(), failure))?;
        let cache_dependencies = implementation
            .cache_dependencies(&context, &node_inputs)
            .map_err(|failure| execution_node_failure(node_id.clone(), failure))?;
        let demanded_dependencies = demanded_dependency_identities(&node, &inputs, state)?;
        let cache_key = CacheKey::from_inputs_with_dependencies(
            &node.class_type,
            &node.descriptor.implementation_version,
            &inputs,
            demanded_dependencies,
            cache_dependencies.artifact_digests,
            &self.backend,
            &self.dtype_policy,
            cache_dependencies.plugin_digest,
            cache_dependencies.rng_phase,
            &self.configuration_token,
            &self.registry_version,
            change_token,
        )
        .map_err(|error| ExecutionError::Cache(error.to_string()))?;
        let cache_identity = cache_key
            .identity()
            .map_err(|error| ExecutionError::Cache(error.to_string()))?;
        if node.descriptor.cache == RuntimeCachePolicy::InputIdentity {
            let cached = self.cache.lock().get_with_handle_lease(&cache_key);
            if let Some((entry, cache_lease)) = cached {
                let mut handles = Vec::new();
                for output in &entry.outputs {
                    collect_native_value_handles(output, &mut handles);
                }
                let attempt_lease = if handles.is_empty() {
                    cache_lease.is_none().then_some(None)
                } else if cache_lease.as_ref().is_some_and(|lease| {
                    lease.store_identity() == self.handle_store_generation.identity()
                        && lease.covers_values(&entry.outputs)
                }) {
                    self.handle_store_generation
                        .acquire_lease(handles.iter())
                        .ok()
                } else {
                    None
                };
                if let Some(attempt_lease) = attempt_lease {
                    if let Some(lease) = attempt_lease {
                        state.cache_handle_leases.push(lease);
                    }
                    state.cache_hits = state.cache_hits.saturating_add(1);
                    state.emit(
                        self.event_bus.as_ref(),
                        Some(node_id.clone()),
                        AttemptEventKind::CacheHit,
                    )?;
                    state.outputs.insert(node_id.clone(), entry.outputs.clone());
                    if let Some(ui) = entry.ui {
                        state.ui_outputs.insert(node_id.clone(), ui);
                    }
                    state
                        .cache_identities
                        .insert(node_id.clone(), cache_identity);
                    emit_progress(plan, state, self.event_bus.as_ref(), node_id.clone())?;
                    return Ok(entry.outputs);
                }
                self.cache.lock().remove(&cache_key);
            }
        }

        let execution = self
            .execute_mapped(
                &node,
                implementation.as_ref(),
                context,
                inputs,
                state,
                expansion_depth,
            )
            .await;
        let (outputs, ui, effects) = match execution {
            Ok(execution) => execution,
            Err(error) => {
                return Err(rollback_node_prepared_effects(
                    effect_service.as_ref(),
                    error,
                ));
            }
        };
        let validated_effects = (|| {
            if outputs.len() != node.descriptor.outputs.len() {
                return Err(ExecutionError::OutputArity {
                    node: node_id.clone(),
                    expected: node.descriptor.outputs.len(),
                    actual: outputs.len(),
                });
            }
            if !effects.is_empty()
                && matches!(
                    node.descriptor.effect,
                    EffectClass::Pure | EffectClass::ReadsArtifact
                )
            {
                return Err(ExecutionError::UnexpectedEffect {
                    node: node_id.clone(),
                    effect: node.descriptor.effect,
                });
            }
            if effects.len() > MAX_EFFECTS_PER_NODE {
                return Err(ExecutionError::TooManyEffects {
                    node: node_id.clone(),
                });
            }
            let mut effect_transactions = BTreeSet::new();
            let mut prepared_effects = Vec::new();
            for effect in effects {
                effect
                    .validate()
                    .map_err(|error| ExecutionError::Effect(error.to_string()))?;
                if !effect_transactions.insert(effect.transaction_id()) {
                    return Err(ExecutionError::Effect(format!(
                        "duplicate prepared effect ticket {}",
                        effect.transaction_id()
                    )));
                }
                prepared_effects.push(
                    self.effects
                        .prepared_effect(&effect, plan.prompt_id, state.attempt_id, node_id)
                        .map_err(ExecutionError::Effect)?,
                );
            }
            Ok(prepared_effects)
        })();
        let validated_effects = match validated_effects {
            Ok(effects) => effects,
            Err(error) => {
                return Err(rollback_node_prepared_effects(
                    effect_service.as_ref(),
                    error,
                ));
            }
        };
        for prepared in validated_effects {
            state.prepared_effects.push(prepared.clone());
            state.emit(
                self.event_bus.as_ref(),
                Some(node_id.clone()),
                AttemptEventKind::OutputPrepared {
                    transaction_id: prepared.transaction_id,
                },
            )?;
        }
        if node.descriptor.cache == RuntimeCachePolicy::InputIdentity
            && state
                .prepared_effects
                .iter()
                .all(|effect| effect.node_id != *node_id)
        {
            state.pending_cache_entries.push((
                cache_key,
                CacheEntry {
                    outputs: outputs.clone(),
                    ui: ui.clone(),
                },
            ));
        }
        state.outputs.insert(node_id.clone(), outputs.clone());
        if let Some(ui) = ui {
            state.ui_outputs.insert(node_id.clone(), ui);
        }
        state
            .cache_identities
            .insert(node_id.clone(), cache_identity);
        emit_progress(plan, state, self.event_bus.as_ref(), node_id.clone())?;
        Ok(outputs)
    }

    async fn execute_mapped(
        &self,
        node: &CompiledNode,
        implementation: &dyn NativeNode,
        context: NodeContext,
        inputs: BTreeMap<String, NativeValue>,
        state: &mut RunState,
        expansion_depth: usize,
    ) -> Result<(Vec<NativeValue>, Option<Value>, Vec<PreparedEffectRequest>), ExecutionError> {
        let mapped = inputs
            .iter()
            .filter_map(|(name, value)| {
                crate::prompt_compiler::resolve_input_descriptor(&node.descriptor, name)
                    .filter(|descriptor| descriptor.cardinality == InputMode::Mapped)
                    .and_then(|_| {
                        native_list_values(value).map(|values| (name.clone(), values.len()))
                    })
            })
            .collect::<Vec<_>>();
        if mapped.is_empty() {
            let inputs = assemble_structured_inputs(&node.descriptor, &inputs)?;
            return self
                .execute_once(
                    implementation,
                    &node.descriptor.outputs,
                    context,
                    inputs,
                    state,
                    expansion_depth,
                )
                .await;
        }
        if mapped.iter().any(|(_, length)| *length == 0) {
            return Ok((
                node.descriptor
                    .outputs
                    .iter()
                    .map(|_| NativeValue::List { values: Vec::new() })
                    .collect(),
                None,
                Vec::new(),
            ));
        }
        let iterations = mapped
            .iter()
            .map(|(_, length)| *length)
            .max()
            .unwrap_or_default();
        let mut collected = vec![Vec::with_capacity(iterations); node.descriptor.outputs.len()];
        let mut combined_ui = Vec::new();
        let mut effects = Vec::new();
        for index in 0..iterations {
            if state.cancellation.is_cancelled() {
                return Err(ExecutionError::Cancelled);
            }
            let mut iteration_inputs = inputs.clone();
            for (name, _) in &mapped {
                let values = inputs
                    .get(name)
                    .and_then(native_list_values)
                    .ok_or_else(|| ExecutionError::InvalidLazyDemand {
                        node: node.id.clone(),
                        input: name.clone(),
                    })?;
                let value_index = index.min(values.len().saturating_sub(1));
                let value = values.get(value_index).cloned().ok_or_else(|| {
                    ExecutionError::InvalidLazyDemand {
                        node: node.id.clone(),
                        input: name.clone(),
                    }
                })?;
                iteration_inputs.insert(name.clone(), value);
            }
            let (outputs, ui, iteration_effects) = self
                .execute_once(
                    implementation,
                    &node.descriptor.outputs,
                    context.clone(),
                    assemble_structured_inputs(&node.descriptor, &iteration_inputs)?,
                    state,
                    expansion_depth,
                )
                .await?;
            if outputs.len() != collected.len() {
                return Err(ExecutionError::OutputArity {
                    node: node.id.clone(),
                    expected: collected.len(),
                    actual: outputs.len(),
                });
            }
            for (output_index, ((output, values), descriptor)) in outputs
                .into_iter()
                .zip(&mut collected)
                .zip(&node.descriptor.outputs)
                .enumerate()
            {
                if descriptor.is_list {
                    let output_values = native_list_values(&output).ok_or_else(|| {
                        ExecutionError::InvalidOutput {
                            node: node.id.clone(),
                            output_index,
                        }
                    })?;
                    values.extend(output_values.iter().cloned());
                } else {
                    values.push(output);
                }
            }
            if let Some(ui) = ui {
                combined_ui.push(ui);
            }
            effects.extend(iteration_effects);
        }
        Ok((
            collected
                .into_iter()
                .map(|values| NativeValue::List { values })
                .collect(),
            (!combined_ui.is_empty()).then_some(Value::Array(combined_ui)),
            effects,
        ))
    }

    async fn execute_once(
        &self,
        implementation: &dyn NativeNode,
        output_descriptors: &[RuntimeOutputDescriptor],
        context: NodeContext,
        inputs: BTreeMap<String, NativeValue>,
        state: &mut RunState,
        expansion_depth: usize,
    ) -> Result<(Vec<NativeValue>, Option<Value>, Vec<PreparedEffectRequest>), ExecutionError> {
        if state.cancellation.is_cancelled() {
            return Err(ExecutionError::Cancelled);
        }
        let node_id = context.node_id.clone();
        let handle_checkpoint = state.handle_store.checkpoint();
        let outcome = implementation.execute(context, inputs).await;
        if state.cancellation.is_cancelled() {
            state.handle_store.rollback_from(handle_checkpoint);
            return Err(ExecutionError::Cancelled);
        }
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(failure) => {
                state.handle_store.rollback_from(handle_checkpoint);
                return Err(execution_node_failure(node_id, failure));
            }
        };
        if let Err(error) = outcome.validate() {
            state.handle_store.rollback_from(handle_checkpoint);
            return Err(ExecutionError::HandleStore(error.to_string()));
        }
        let result = async {
            match outcome {
                NodeOutcome::Values {
                    outputs,
                    ui,
                    effects,
                } => Ok((outputs, ui, effects)),
                NodeOutcome::Blocked { reason } => Err(ExecutionError::Blocked {
                    node: node_id.clone(),
                    reason,
                }),
                NodeOutcome::Expansion {
                    prompt,
                    output_node,
                } => {
                    if expansion_depth >= MAX_EXPANSION_DEPTH {
                        return Err(ExecutionError::ExpansionDepth);
                    }
                    let plan = crate::PromptCompiler::new(&self.nodes)
                        .compile(PromptSubmission {
                            prompt,
                            prompt_id: Some(state.prompt_id),
                            client_id: None,
                            number: None,
                            extra_data: BTreeMap::new(),
                            unknown: BTreeMap::new(),
                        })
                        .map_err(ExecutionError::ExpansionCompile)?;
                    if !plan.nodes.contains_key(&output_node) {
                        return Err(ExecutionError::InvalidExpansionOutput(output_node));
                    }
                    let scope = state.next_expansion_scope;
                    let next_scope = scope
                        .checked_add(1)
                        .ok_or(ExecutionError::ExpansionSequenceExhausted)?;
                    let (plan, output_node) = namespace_expansion(
                        &plan,
                        &output_node,
                        &node_id,
                        state.prompt_id,
                        expansion_depth,
                        scope,
                    )?;
                    state.next_expansion_scope = next_scope;
                    self.run_node(&plan, &output_node, state, expansion_depth + 1)
                        .await
                        .map(|outputs| (outputs, None, Vec::new()))
                }
            }
        }
        .await;
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                state.handle_store.rollback_from(handle_checkpoint);
                return Err(error);
            }
        };
        if let Err(error) = validate_outputs(&node_id, output_descriptors, &result.0) {
            state.handle_store.rollback_from(handle_checkpoint);
            return Err(error);
        }
        Ok(result)
    }
}

fn emit_progress(
    plan: &CompiledPlan,
    state: &mut RunState,
    event_bus: Option<&ExecutionEventBus>,
    node_id: NodeId,
) -> Result<(), ExecutionError> {
    let completed = u64::try_from(
        plan.static_required_nodes
            .iter()
            .filter(|node_id| state.outputs.contains_key(*node_id))
            .count(),
    )
    .map_err(|_| ExecutionError::ProgressOverflow)?;
    let total = u64::try_from(plan.static_required_nodes.len())
        .map_err(|_| ExecutionError::ProgressOverflow)?;
    state.emit(
        event_bus,
        Some(node_id),
        AttemptEventKind::Progress { completed, total },
    )
}

fn execution_node_failure(node: NodeId, failure: NodeFailure) -> ExecutionError {
    if failure.kind == NodeFailureKind::Interrupted {
        ExecutionError::Interrupted { node, failure }
    } else {
        ExecutionError::Node { node, failure }
    }
}

fn nonempty_cache_dimension(name: &'static str, value: String) -> Result<String, ExecutionError> {
    if value.is_empty() {
        Err(ExecutionError::Cache(format!("{name} is empty")))
    } else {
        Ok(value)
    }
}

fn demanded_dependency_identities(
    node: &CompiledNode,
    inputs: &BTreeMap<String, NativeValue>,
    state: &RunState,
) -> Result<BTreeMap<String, String>, ExecutionError> {
    node.inputs
        .iter()
        .filter_map(|(name, binding)| {
            if !inputs.contains_key(name) {
                return None;
            }
            let InputBinding::Link {
                source,
                output_index,
                mode,
                ..
            } = binding
            else {
                return None;
            };
            Some((name, source, output_index, mode))
        })
        .map(|(name, source, output_index, mode)| {
            let source_identity = state.cache_identities.get(source).ok_or_else(|| {
                ExecutionError::Cache(format!(
                    "demanded dependency {source:?} has no cache identity"
                ))
            })?;
            let mode = match mode {
                InputMode::Scalar => "scalar",
                InputMode::List => "list",
                InputMode::Mapped => "mapped",
            };
            Ok((
                name.clone(),
                format!("{source_identity}:output={output_index}:mode={mode}"),
            ))
        })
        .collect()
}

fn validate_outputs(
    node_id: &NodeId,
    descriptors: &[RuntimeOutputDescriptor],
    outputs: &[NativeValue],
) -> Result<(), ExecutionError> {
    if outputs.len() != descriptors.len() {
        return Err(ExecutionError::OutputArity {
            node: node_id.clone(),
            expected: descriptors.len(),
            actual: outputs.len(),
        });
    }
    for (output_index, (descriptor, output)) in descriptors.iter().zip(outputs).enumerate() {
        let valid = if descriptor.is_list {
            native_list_values(output).is_some_and(|values| {
                values
                    .iter()
                    .all(|value| native_output_type_accepts(&descriptor.produced_type, value))
            })
        } else {
            !matches!(output, NativeValue::List { .. })
                && native_output_type_accepts(&descriptor.produced_type, output)
        };
        if !valid {
            return Err(ExecutionError::InvalidOutput {
                node: node_id.clone(),
                output_index,
            });
        }
    }
    Ok(())
}

fn linked_value(
    source: &NodeId,
    output_index: usize,
    mode: InputMode,
    outputs: &[NativeValue],
) -> Result<NativeValue, ExecutionError> {
    let value =
        outputs
            .get(output_index)
            .cloned()
            .ok_or_else(|| ExecutionError::MissingOutput {
                node: source.clone(),
                output_index,
            })?;
    if mode == InputMode::List && !matches!(value, NativeValue::List { .. }) {
        Ok(NativeValue::List {
            values: vec![value],
        })
    } else {
        Ok(value)
    }
}

fn native_list_values(value: &NativeValue) -> Option<&[NativeValue]> {
    match value {
        NativeValue::List { values } => Some(values),
        _ => None,
    }
}

#[derive(Default)]
struct StructuredInputTree {
    value: Option<NativeValue>,
    children: BTreeMap<String, StructuredInputTree>,
}

fn assemble_structured_inputs(
    descriptor: &RuntimeNodeDescriptor,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<BTreeMap<String, NativeValue>, ExecutionError> {
    let Some(source_schema) = &descriptor.source_schema else {
        return Ok(inputs.clone());
    };
    let mut result = inputs.clone();
    for schema in &source_schema.inputs {
        let structured = schema
            .structured_options()
            .map_err(|error| ExecutionError::HandleStore(error.to_string()))?;
        if structured.is_empty() || !inputs.contains_key(&schema.name) {
            continue;
        }
        let mut tree = StructuredInputTree::default();
        tree.value = result.remove(&schema.name).map(normalize_structured_field);
        let prefix = format!("{}.", schema.name);
        let nested_names = result
            .keys()
            .filter(|name| name.starts_with(&prefix))
            .cloned()
            .collect::<Vec<_>>();
        for name in nested_names {
            let value = result.remove(&name).ok_or_else(|| {
                ExecutionError::HandleStore("structured input disappeared".to_owned())
            })?;
            let path = name.strip_prefix(&prefix).ok_or_else(|| {
                ExecutionError::HandleStore("structured input path is invalid".to_owned())
            })?;
            insert_structured_input_path(&mut tree, path, normalize_structured_field(value))?;
        }
        let type_name = schema.source_type_names.first().ok_or_else(|| {
            ExecutionError::HandleStore("structured input type is missing".to_owned())
        })?;
        let value = structured_tree_value(&schema.name, type_name, tree)?;
        result.insert(schema.name.clone(), value);
    }
    Ok(result)
}

fn insert_structured_input_path(
    tree: &mut StructuredInputTree,
    path: &str,
    value: NativeValue,
) -> Result<(), ExecutionError> {
    let mut parts = path.split('.').peekable();
    let mut current = tree;
    while let Some(part) = parts.next() {
        if part.is_empty() {
            return Err(ExecutionError::HandleStore(
                "structured input path contains an empty field".to_owned(),
            ));
        }
        current = current.children.entry(part.to_owned()).or_default();
        if parts.peek().is_none() {
            if current.value.replace(value).is_some() {
                return Err(ExecutionError::HandleStore(
                    "structured input path is duplicated".to_owned(),
                ));
            }
            return Ok(());
        }
    }
    Err(ExecutionError::HandleStore(
        "structured input path is empty".to_owned(),
    ))
}

fn structured_tree_value(
    field_name: &str,
    type_name: &str,
    tree: StructuredInputTree,
) -> Result<NativeValue, ExecutionError> {
    let mut fields = BTreeMap::new();
    if let Some(value) = tree.value {
        fields.insert(field_name.to_owned(), value);
    }
    for (name, child) in tree.children {
        let value = if child.children.is_empty() {
            child.value.ok_or_else(|| {
                ExecutionError::HandleStore("structured input field has no value".to_owned())
            })?
        } else {
            structured_tree_value(&name, "zed.structured@1", child)?
        };
        if fields.insert(name, value).is_some() {
            return Err(ExecutionError::HandleStore(
                "structured input field is duplicated".to_owned(),
            ));
        }
    }
    NativeStructuredValue::checked(type_name, fields)
        .and_then(NativeStructuredValue::into_runtime_value)
        .map_err(|error| ExecutionError::HandleStore(error.to_string()))
}

fn normalize_structured_field(value: NativeValue) -> NativeValue {
    let NativeValue::PreservedUnknown { value, .. } = value else {
        return value;
    };
    match value {
        Value::Null => NativeValue::Primitive {
            value: sim_nodes::NativePrimitive::Null,
        },
        Value::Bool(value) => NativeValue::Primitive {
            value: sim_nodes::NativePrimitive::Boolean(value),
        },
        Value::Number(value) => {
            let primitive = value
                .as_i64()
                .map(sim_nodes::NativePrimitive::Integer)
                .or_else(|| {
                    value
                        .as_u64()
                        .map(sim_nodes::NativePrimitive::UnsignedInteger)
                })
                .or_else(|| {
                    value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .map(sim_nodes::NativePrimitive::Number)
                });
            match primitive {
                Some(value) => NativeValue::Primitive { value },
                None => NativeValue::PreservedUnknown {
                    type_name: "zed.json@1".to_owned(),
                    value: Value::Number(value),
                },
            }
        }
        Value::String(value) => NativeValue::Primitive {
            value: sim_nodes::NativePrimitive::String(value),
        },
        value @ (Value::Array(_) | Value::Object(_)) => NativeValue::PreservedUnknown {
            type_name: "zed.json@1".to_owned(),
            value,
        },
    }
}

fn collect_native_value_handles(value: &NativeValue, handles: &mut Vec<NativeOpaqueHandle>) {
    match value {
        NativeValue::Handle { value } => handles.push(value.clone()),
        NativeValue::List { values } => {
            for value in values {
                collect_native_value_handles(value, handles);
            }
        }
        NativeValue::Primitive { .. } | NativeValue::PreservedUnknown { .. } => {}
    }
}

fn native_output_type_accepts(expected: &sim_nodes::NativeValueType, value: &NativeValue) -> bool {
    match (expected, value) {
        (sim_nodes::NativeValueType::Any, _) => true,
        (
            _,
            NativeValue::Primitive {
                value: sim_nodes::NativePrimitive::Null,
            },
        ) => true,
        (sim_nodes::NativeValueType::Primitive(expected), NativeValue::Primitive { value }) => {
            *expected == value.primitive_type()
                || (*expected == sim_nodes::NativePrimitiveType::Number
                    && value.primitive_type() == sim_nodes::NativePrimitiveType::Integer)
        }
        (sim_nodes::NativeValueType::Handle(expected), NativeValue::Handle { value }) => {
            expected == value.handle_type()
        }
        (sim_nodes::NativeValueType::PreservedUnknown, NativeValue::PreservedUnknown { .. }) => {
            true
        }
        (
            sim_nodes::NativeValueType::NamedPreservedUnknown(expected),
            NativeValue::PreservedUnknown { type_name, .. },
        ) => expected == type_name,
        _ => false,
    }
}

fn namespace_expansion(
    plan: &CompiledPlan,
    output_node: &NodeId,
    owner: &NodeId,
    prompt_id: PromptId,
    depth: usize,
    scope: u64,
) -> Result<(CompiledPlan, NodeId), ExecutionError> {
    if !plan.nodes.contains_key(output_node) {
        return Err(ExecutionError::InvalidExpansionOutput(output_node.clone()));
    }
    let prefix = format!("{}::expansion-{depth}-{scope}::", owner.0);
    let translate = |identifier: &NodeId| NodeId(format!("{prefix}{}", identifier.0));
    let mut nodes = BTreeMap::new();
    for node in plan.nodes.values() {
        let mut node = node.clone();
        node.id = translate(&node.id);
        for binding in node.inputs.values_mut() {
            if let InputBinding::Link { source, .. } = binding {
                *source = translate(source);
            }
        }
        nodes.insert(node.id.clone(), node);
    }
    Ok((
        CompiledPlan {
            prompt_id,
            client_id: plan.client_id.clone(),
            prompt_number: plan.prompt_number,
            extra_data: plan.extra_data.clone(),
            unknown: plan.unknown.clone(),
            nodes,
            topological_order: plan.topological_order.iter().map(translate).collect(),
            static_required_nodes: plan.static_required_nodes.iter().map(translate).collect(),
            output_nodes: plan.output_nodes.iter().map(translate).collect(),
            provider_execution: plan.provider_execution.clone(),
            persistence_unknown_fields: plan.persistence_unknown_fields.clone(),
        },
        translate(output_node),
    ))
}

struct RunState {
    profile_id: ProfileId,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    cancellation: CancellationToken,
    outputs: BTreeMap<NodeId, Vec<NativeValue>>,
    ui_outputs: BTreeMap<NodeId, Value>,
    cache_identities: BTreeMap<NodeId, String>,
    visiting: BTreeSet<NodeId>,
    prepared_effects: Vec<PreparedEffect>,
    events: Vec<AttemptEvent>,
    next_sequence: u64,
    next_expansion_scope: u64,
    cache_hits: usize,
    diagnostics: Vec<String>,
    handle_store: Arc<RuntimeNativeHandleStoreSession>,
    pending_cache_entries: Vec<(CacheKey, CacheEntry)>,
    cache_handle_leases: Vec<NativeHandleLease>,
}

impl RunState {
    fn new(
        profile_id: ProfileId,
        prompt_id: PromptId,
        attempt_id: AttemptId,
        cancellation: CancellationToken,
        handle_store: Arc<RuntimeNativeHandleStoreSession>,
    ) -> Self {
        Self {
            profile_id,
            prompt_id,
            attempt_id,
            cancellation,
            outputs: BTreeMap::new(),
            ui_outputs: BTreeMap::new(),
            cache_identities: BTreeMap::new(),
            visiting: BTreeSet::new(),
            prepared_effects: Vec::new(),
            events: Vec::new(),
            next_sequence: 0,
            next_expansion_scope: 0,
            cache_hits: 0,
            diagnostics: Vec::new(),
            handle_store,
            pending_cache_entries: Vec::new(),
            cache_handle_leases: Vec::new(),
        }
    }

    fn emit(
        &mut self,
        event_bus: Option<&ExecutionEventBus>,
        node_id: Option<NodeId>,
        kind: AttemptEventKind,
    ) -> Result<(), ExecutionError> {
        let event = AttemptEvent {
            profile_id: self.profile_id,
            prompt_id: self.prompt_id,
            attempt_id: self.attempt_id,
            sequence: self.next_sequence,
            node_id,
            at: Utc::now(),
            kind,
            data: None,
        };
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(ExecutionError::SequenceExhausted)?;
        self.events.push(event.clone());
        if let Some(event_bus) = event_bus {
            event_bus
                .publish(event)
                .map_err(|error| ExecutionError::EventBus(error.to_string()))?;
        }
        Ok(())
    }
}

impl From<EventBusError> for ExecutionError {
    fn from(error: EventBusError) -> Self {
        Self::EventBus(error.to_string())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{InputMode, PromptCompiler, RuntimeNodeDescriptor, RuntimeOutputDescriptor};
    use serde_json::json;
    use sim_model::{
        ClipVisionActivation, ClipVisionConfiguration, ClipVisionLayerWeights, ClipVisionModelType,
        ClipVisionOutput, ClipVisionWeights, NativeClipVision, NativeModelPayload,
    };
    use sim_nodes::{
        NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeHandleKind, NativeInputDescriptor,
        NativePrimitive, NativePrimitiveType, NativeProviderPayload, NativeStoredModelPayload,
        NativeTypeUnion, NativeValueType,
    };
    use sim_tensor::{DType, StreamId, Tensor, TensorDescriptor};
    use sim_types::{ApiPrompt, PromptNode, PromptSubmission};
    use std::mem;
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    const TEST_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;

    #[derive(Clone)]
    enum ValueType {
        Any,
        Boolean,
        Integer,
        Number,
    }

    struct RuntimeInputDescriptor {
        value_type: ValueType,
        required: bool,
        hidden: bool,
        lazy: bool,
        mode: InputMode,
        allows_literal: bool,
    }

    fn native_integer(value: i64) -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }
    }

    fn native_boolean(value: bool) -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::Boolean(value),
        }
    }

    fn native_string(value: &str) -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::String(value.to_owned()),
        }
    }

    fn native_null() -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::Null,
        }
    }

    fn prepare_test_effect(
        context: &NodeContext,
        content: &'static [u8],
    ) -> Result<PreparedEffectRequest, NodeFailure> {
        let service = context.prepared_effects().map_err(|error| NodeFailure {
            code: "test_effect_service_unavailable".to_owned(),
            message: error.to_string(),
            kind: NodeFailureKind::Failure,
            retryable: false,
        })?;
        let request = NativeOutputEffectRequest::checked(
            NativeOutputNamespace::Temporary,
            "test-effect",
            "bin",
            0,
            NativeOutputShape::File,
            Arc::from(content),
            service.maximum_output_bytes(),
        )
        .map_err(|error| NodeFailure {
            code: "test_effect_request_invalid".to_owned(),
            message: error.to_string(),
            kind: NodeFailureKind::Failure,
            retryable: false,
        })?;
        service
            .prepare_output(request, &context.cancellation)
            .map_err(|error| NodeFailure {
                code: "test_effect_prepare_failed".to_owned(),
                message: error.to_string(),
                kind: NodeFailureKind::Failure,
                retryable: false,
            })
    }

    #[allow(clippy::result_large_err)]
    fn stored_test_payload(
        abi_bytes: Vec<u8>,
    ) -> Result<NativeStoredPayload, sim_nodes::NativeStoredPayloadError> {
        let semantic_digest_sha256 = format!("{:x}", Sha256::digest(&abi_bytes));
        Ok(NativeStoredPayload::Provider(Arc::new(
            NativeProviderPayload::checked(
                NativeHandleType::new(NativeHandleKind::ProviderTask, "TEST_PROVIDER_TASK")?,
                "zed.test.provider",
                semantic_digest_sha256,
                abi_bytes,
            )?,
        )))
    }

    fn clip_vision_test_tensor(
        backend: &sim_tensor::CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
        shape: Vec<u64>,
        value: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let elements = shape
            .iter()
            .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension))
            .ok_or("clip vision test tensor element count overflowed")?;
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(
                elements
                    .checked_mul(4)
                    .ok_or("clip vision test tensor workspace byte count overflowed")?,
            )?,
            cancellation,
        );
        Ok(backend
            .upload_f32(
                descriptor,
                &vec![value; usize::try_from(elements)?],
                &context,
            )?
            .0)
    }

    fn tiny_clip_vision_resource() -> Result<Arc<NativeClipVision>, Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let zero_2 = clip_vision_test_tensor(&backend, &authority, &cancellation, vec![2], 0.0)?;
        let zero_2x2 =
            clip_vision_test_tensor(&backend, &authority, &cancellation, vec![2, 2], 0.0)?;
        let layer = ClipVisionLayerWeights {
            layer_norm_1_weight: clip_vision_test_tensor(
                &backend,
                &authority,
                &cancellation,
                vec![2],
                1.0,
            )?,
            layer_norm_1_bias: zero_2.clone(),
            query_weight: zero_2x2.clone(),
            query_bias: zero_2.clone(),
            key_weight: zero_2x2.clone(),
            key_bias: zero_2.clone(),
            value_weight: zero_2x2.clone(),
            value_bias: zero_2.clone(),
            output_weight: zero_2x2.clone(),
            output_bias: zero_2.clone(),
            layer_norm_2_weight: clip_vision_test_tensor(
                &backend,
                &authority,
                &cancellation,
                vec![2],
                1.0,
            )?,
            layer_norm_2_bias: zero_2.clone(),
            feed_forward_1_weight: zero_2x2.clone(),
            feed_forward_1_bias: zero_2.clone(),
            feed_forward_2_weight: zero_2x2,
            feed_forward_2_bias: zero_2.clone(),
        };
        Ok(Arc::new(NativeClipVision::new(
            ClipVisionConfiguration {
                model_type: ClipVisionModelType::Clip,
                dtype: DType::F32,
                device: DeviceId::CPU,
                hidden_size: 2,
                intermediate_size: 2,
                attention_heads: 1,
                layer_count: 1,
                image_size: 2,
                patch_size: 1,
                num_channels: 3,
                max_num_patches: 4,
                activation: ClipVisionActivation::QuickGelu,
                projection_dimension: None,
                llava_projection_dimension: None,
            },
            ClipVisionWeights {
                patch_embedding_weight: clip_vision_test_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![2, 3, 1, 1],
                    0.0,
                )?,
                patch_embedding_bias: None,
                class_embedding: Some(zero_2.clone()),
                position_embedding: clip_vision_test_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![5, 2],
                    0.0,
                )?,
                pre_layer_norm_weight: Some(clip_vision_test_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![2],
                    1.0,
                )?),
                pre_layer_norm_bias: Some(zero_2.clone()),
                layers: vec![layer],
                post_layer_norm_weight: clip_vision_test_tensor(
                    &backend,
                    &authority,
                    &cancellation,
                    vec![2],
                    1.0,
                )?,
                post_layer_norm_bias: zero_2,
                visual_projection_weight: None,
                llava_linear_1_weight: None,
                llava_linear_1_bias: None,
                llava_linear_2_weight: None,
                llava_linear_2_bias: None,
            },
        )?))
    }

    fn test_cache_key(node_class: &str) -> Result<CacheKey, crate::NativeCacheError> {
        CacheKey::from_inputs(
            node_class,
            "1",
            &BTreeMap::new(),
            BTreeMap::new(),
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )
    }

    fn native_integer_value(value: &NativeValue) -> Option<i64> {
        match value {
            NativeValue::Primitive {
                value: NativePrimitive::Integer(value),
            } => Some(*value),
            _ => None,
        }
    }

    struct FixtureNode {
        class_type: String,
        calls: Arc<AtomicUsize>,
    }

    #[test]
    fn native_model_resource_service_recipes_reject_before_transport()
    -> Result<(), Box<dyn std::error::Error>> {
        let (transport, host) = NativeModelSourceTransport::channel();
        let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1 << 20)?;
        let backend = Arc::new(backend);
        let attempt_id = AttemptId(Uuid::from_u128(0x4004));
        let node_id = NodeId("model-resource".to_owned());
        let identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x4005),
            attempt_id,
            node_id.clone(),
        )?;
        let service = NativeModelResourceServiceFactory {
            binding: crate::NativeModelSourceWorkerBinding::checked(
                Uuid::from_u128(0x4006),
                1,
                1,
                1,
            )?,
            transport,
            backend,
            construction_profile: NativeModelConstructionProfile::SourceExact,
        }
        .node_service(identity);
        let services =
            NativeNodeServices::checked(None, None, None)?.with_model_resources(service)?;
        let handle_store_generation = NativeHandleStoreGeneration::new()?;
        let context = NodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x4007)),
            attempt_id,
            node_id,
            CancellationToken::default(),
            workspace_authority.authorize_workspace(1 << 20)?,
            handle_store_generation.handle_store_for_attempt(attempt_id),
            services,
        )?;
        for (request, expected) in [
            (
                NativeModelResourceRequest::checked(
                    NativeModelResourceRole::ControlNet,
                    "controlnet",
                    vec!["control.safetensors".to_owned()],
                )?,
                NativeModelResourceServiceError::Unavailable,
            ),
            (
                NativeModelResourceRequest::checked(
                    NativeModelResourceRole::StyleModel,
                    "checkpoints",
                    vec!["style.safetensors".to_owned()],
                )?,
                NativeModelResourceServiceError::InvalidRequest,
            ),
            (
                NativeModelResourceRequest::checked(
                    NativeModelResourceRole::StyleModel,
                    "style_models",
                    vec!["one.safetensors".to_owned(), "two.safetensors".to_owned()],
                )?,
                NativeModelResourceServiceError::InvalidRequest,
            ),
        ] {
            let error = context
                .load_native_model_resource(&request)
                .err()
                .ok_or("unsupported model-resource recipe unexpectedly loaded")?;
            assert_eq!(mem::discriminant(&error), mem::discriminant(&expected));
        }
        assert!(matches!(
            host.receiver.try_recv(),
            Err(async_channel::TryRecvError::Empty)
        ));
        Ok(())
    }

    struct WorkspaceRecordingNode {
        observed: Arc<Mutex<Vec<u64>>>,
    }

    impl NativeNode for WorkspaceRecordingNode {
        fn class_type(&self) -> &str {
            "WorkspaceRecording"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn demanded_lazy_inputs(
            &self,
            context: &NodeContext,
            _available: &BTreeMap<String, NativeValue>,
        ) -> Result<BTreeSet<String>, NodeFailure> {
            self.observed.lock().push(context.scratch.bytes());
            Ok(BTreeSet::new())
        }

        fn execute<'a>(
            &'a self,
            context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            self.observed.lock().push(context.scratch.bytes());
            Box::pin(async {
                Ok(NodeOutcome::Values {
                    outputs: vec![native_integer(1)],
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    impl NativeNode for FixtureNode {
        fn class_type(&self) -> &str {
            &self.class_type
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn demanded_lazy_inputs(
            &self,
            _context: &NodeContext,
            available: &BTreeMap<String, NativeValue>,
        ) -> Result<BTreeSet<String>, NodeFailure> {
            if self.class_type == "Choose"
                && available.get("condition") == Some(&native_boolean(true))
            {
                Ok(BTreeSet::from(["value".to_owned()]))
            } else {
                Ok(BTreeSet::new())
            }
        }

        fn execute<'a>(
            &'a self,
            context: NodeContext,
            inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                smol::future::yield_now().await;
                if context.cancellation.is_cancelled() {
                    return Err(NodeFailure {
                        code: "cancelled".to_owned(),
                        message: "fixture cancelled".to_owned(),
                        kind: NodeFailureKind::Interrupted,
                        retryable: true,
                    });
                }
                let outputs = match self.class_type.as_str() {
                    "Source" => vec![NativeValue::List {
                        values: vec![native_integer(1), native_integer(2), native_integer(3)],
                    }],
                    "LazySource" | "InnerOutput" => vec![native_integer(42)],
                    "Double" => vec![native_integer(
                        inputs
                            .get("value")
                            .and_then(native_integer_value)
                            .unwrap_or_default()
                            * 2,
                    )],
                    "Pair" => vec![native_integer(
                        inputs
                            .get("left")
                            .and_then(native_integer_value)
                            .unwrap_or_default()
                            * 10
                            + inputs
                                .get("right")
                                .and_then(native_integer_value)
                                .unwrap_or_default(),
                    )],
                    "ListMap" => vec![NativeValue::List {
                        values: vec![native_integer(
                            inputs
                                .get("value")
                                .and_then(native_integer_value)
                                .unwrap_or_default(),
                        )],
                    }],
                    "Choose" => vec![
                        inputs
                            .get("value")
                            .cloned()
                            .unwrap_or_else(|| native_integer(0)),
                    ],
                    "Output" => vec![inputs.get("value").cloned().unwrap_or_else(native_null)],
                    "Write" => {
                        return Ok(NodeOutcome::Values {
                            outputs: vec![native_string("prepared")],
                            ui: None,
                            effects: vec![prepare_test_effect(&context, b"output")?],
                        });
                    }
                    "PrepareThenFail" => {
                        let _prepared_effect = prepare_test_effect(&context, b"failure")?;
                        return Err(NodeFailure {
                            code: "fixture_failure".to_owned(),
                            message: "fixture failed after preparing an effect".to_owned(),
                            kind: NodeFailureKind::Failure,
                            retryable: false,
                        });
                    }
                    "PrepareThenInvalidOutput" => {
                        return Ok(NodeOutcome::Values {
                            outputs: vec![native_string("invalid")],
                            ui: None,
                            effects: vec![prepare_test_effect(&context, b"invalid-output")?],
                        });
                    }
                    "Block" => {
                        return Ok(NodeOutcome::Blocked {
                            reason: "fixture blocker".to_owned(),
                        });
                    }
                    _ => vec![native_null()],
                };
                Ok(NodeOutcome::Values {
                    outputs,
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    struct ExpansionNode {
        prompt: ApiPrompt,
    }

    struct PublishingMalformedExpansionNode;

    struct PublishingHandleNode {
        calls: Arc<AtomicUsize>,
        cancel_after_publish: bool,
    }

    struct BlockingPublishingHandleNode {
        calls: Arc<AtomicUsize>,
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
        published_handle: Arc<Mutex<Option<NativeOpaqueHandle>>>,
    }

    impl NativeNode for PublishingHandleNode {
        fn class_type(&self) -> &str {
            "PublishingHandle"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                let payload =
                    stored_test_payload(call.to_le_bytes().to_vec()).map_err(|error| {
                        NodeFailure {
                            code: "test_payload".to_owned(),
                            message: error.to_string(),
                            kind: NodeFailureKind::Failure,
                            retryable: false,
                        }
                    })?;
                let handle = context
                    .handle_store()
                    .publish(payload, &context.cancellation)
                    .map_err(|error| NodeFailure {
                        code: "publish".to_owned(),
                        message: error.to_string(),
                        kind: NodeFailureKind::Failure,
                        retryable: false,
                    })?;
                if self.cancel_after_publish {
                    context.cancellation.cancel();
                }
                Ok(NodeOutcome::Values {
                    outputs: vec![NativeValue::Handle { value: handle }],
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    impl NativeNode for BlockingPublishingHandleNode {
        fn class_type(&self) -> &str {
            "PublishingHandle"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                let payload =
                    stored_test_payload(b"claimed-attempt".to_vec()).map_err(|error| {
                        NodeFailure {
                            code: "test_payload".to_owned(),
                            message: error.to_string(),
                            kind: NodeFailureKind::Failure,
                            retryable: false,
                        }
                    })?;
                let handle = context
                    .handle_store()
                    .publish(payload, &context.cancellation)
                    .map_err(|error| NodeFailure {
                        code: "publish".to_owned(),
                        message: error.to_string(),
                        kind: NodeFailureKind::Failure,
                        retryable: false,
                    })?;
                *self.published_handle.lock() = Some(handle.clone());
                self.entered.wait();
                self.release.wait();
                Ok(NodeOutcome::Values {
                    outputs: vec![NativeValue::Handle { value: handle }],
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    impl NativeNode for PublishingMalformedExpansionNode {
        fn class_type(&self) -> &str {
            "PublishingMalformedExpansion"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            Box::pin(async move {
                context
                    .handle_store()
                    .publish(
                        stored_test_payload(vec![9]).map_err(|error| NodeFailure {
                            code: "payload".to_owned(),
                            message: error.to_string(),
                            kind: NodeFailureKind::Failure,
                            retryable: false,
                        })?,
                        &context.cancellation,
                    )
                    .map_err(|error| NodeFailure {
                        code: "publish".to_owned(),
                        message: error.to_string(),
                        kind: NodeFailureKind::Failure,
                        retryable: false,
                    })?;
                Ok(NodeOutcome::Expansion {
                    prompt: ApiPrompt(BTreeMap::new()),
                    output_node: NodeId("missing".to_owned()),
                })
            })
        }
    }

    impl NativeNode for ExpansionNode {
        fn class_type(&self) -> &str {
            "Expand"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            _context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            let prompt = self.prompt.clone();
            Box::pin(async move {
                Ok(NodeOutcome::Expansion {
                    prompt,
                    output_node: NodeId::from("inner"),
                })
            })
        }
    }

    struct CancellingNode;

    impl NativeNode for CancellingNode {
        fn class_type(&self) -> &str {
            "CancelDuringExecution"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            Box::pin(async move {
                context.cancellation.cancel();
                Err(NodeFailure {
                    code: "interrupted".to_owned(),
                    message: "node observed cancellation".to_owned(),
                    kind: NodeFailureKind::Interrupted,
                    retryable: true,
                })
            })
        }
    }

    struct CancelBeforeCacheNode {
        phases: Arc<Mutex<Vec<&'static str>>>,
    }

    impl NativeNode for CancelBeforeCacheNode {
        fn class_type(&self) -> &str {
            "CancelBeforeCache"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn demanded_lazy_inputs(
            &self,
            context: &NodeContext,
            _available_inputs: &BTreeMap<String, NativeValue>,
        ) -> Result<BTreeSet<String>, NodeFailure> {
            self.phases.lock().push("demand");
            context.cancellation.cancel();
            Ok(BTreeSet::new())
        }

        fn cache_change_token(
            &self,
            _inputs: &BTreeMap<String, NativeValue>,
        ) -> Result<String, NodeFailure> {
            self.phases.lock().push("change");
            Ok("stable".to_owned())
        }

        fn cache_dependencies(
            &self,
            _context: &NodeContext,
            _inputs: &BTreeMap<String, NativeValue>,
        ) -> Result<CacheDependencies, NodeFailure> {
            self.phases.lock().push("dependencies");
            Ok(CacheDependencies::default())
        }

        fn execute<'a>(
            &'a self,
            _context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            self.phases.lock().push("execute");
            Box::pin(async move {
                Ok(NodeOutcome::Values {
                    outputs: vec![native_integer(1)],
                    ui: Some(json!({"preview": "must-not-publish"})),
                    effects: Vec::new(),
                })
            })
        }
    }

    struct UiNode {
        calls: Arc<AtomicUsize>,
    }

    impl NativeNode for UiNode {
        fn class_type(&self) -> &str {
            "Ui"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            _context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Ok(NodeOutcome::Values {
                    outputs: vec![native_integer(1)],
                    ui: Some(json!({"preview": "ready"})),
                    effects: Vec::new(),
                })
            })
        }
    }

    struct VersionedSourceNode {
        version: &'static str,
    }

    struct ConfiguredNode {
        class_type: String,
        version: String,
        namespace: String,
    }

    impl NativeNode for ConfiguredNode {
        fn class_type(&self) -> &str {
            &self.class_type
        }

        fn implementation_version(&self) -> &str {
            &self.version
        }

        fn implementation_namespace(&self) -> &str {
            &self.namespace
        }

        fn execute<'a>(
            &'a self,
            _context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            Box::pin(async {
                Ok(NodeOutcome::Values {
                    outputs: Vec::new(),
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    impl NativeNode for VersionedSourceNode {
        fn class_type(&self) -> &str {
            "VersionedSource"
        }

        fn implementation_version(&self) -> &str {
            self.version
        }

        fn execute<'a>(
            &'a self,
            _context: NodeContext,
            _inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            Box::pin(async {
                Ok(NodeOutcome::Values {
                    outputs: vec![native_integer(7)],
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    struct PassthroughNode {
        calls: Arc<AtomicUsize>,
    }

    impl NativeNode for PassthroughNode {
        fn class_type(&self) -> &str {
            "Passthrough"
        }

        fn implementation_version(&self) -> &str {
            "1"
        }

        fn execute<'a>(
            &'a self,
            _context: NodeContext,
            inputs: BTreeMap<String, NativeValue>,
        ) -> BoxFuture<'a, Result<NodeOutcome, NodeFailure>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(NodeOutcome::Values {
                    outputs: vec![inputs.get("value").cloned().unwrap_or_else(native_null)],
                    ui: None,
                    effects: Vec::new(),
                })
            })
        }
    }

    fn runtime_descriptor(
        class_type: &str,
        output_node: bool,
        inputs: BTreeMap<String, RuntimeInputDescriptor>,
        effect: EffectClass,
    ) -> Result<RuntimeNodeDescriptor, NativeNodeContractError> {
        let inputs = inputs
            .into_iter()
            .map(|(name, input)| {
                Ok(NativeInputDescriptor {
                    name,
                    accepted_types: NativeTypeUnion::new([match input.value_type {
                        ValueType::Any => NativeValueType::Any,
                        ValueType::Boolean => {
                            NativeValueType::Primitive(NativePrimitiveType::Boolean)
                        }
                        ValueType::Integer => {
                            NativeValueType::Primitive(NativePrimitiveType::Integer)
                        }
                        ValueType::Number => {
                            NativeValueType::Primitive(NativePrimitiveType::Number)
                        }
                    }])?,
                    required: input.required,
                    hidden: input.hidden,
                    lazy: input.lazy,
                    cardinality: input.mode,
                    allows_literal: input.allows_literal,
                })
            })
            .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
        let source_schema = sim_nodes::NativeDescriptorSchemaMetadata::synthetic(
            inputs.iter().map(|input| input.name.clone()),
            std::iter::empty(),
            ["value".to_owned()],
        );
        Ok(RuntimeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: "1".to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: vec![RuntimeOutputDescriptor {
                name: "value".to_owned(),
                produced_type: if class_type == "Write" {
                    NativeValueType::Any
                } else {
                    NativeValueType::Primitive(NativePrimitiveType::Number)
                },
                is_list: class_type == "Source" || class_type == "Output",
            }],
            output_node,
            effect,
            cache: if effect == EffectClass::Pure {
                RuntimeCachePolicy::InputIdentity
            } else {
                RuntimeCachePolicy::Never
            },
        })
    }

    fn publishing_handle_descriptor() -> Result<RuntimeNodeDescriptor, NativeNodeContractError> {
        let mut descriptor =
            runtime_descriptor("PublishingHandle", true, BTreeMap::new(), EffectClass::Pure)?;
        descriptor.outputs = vec![RuntimeOutputDescriptor {
            name: "value".to_owned(),
            produced_type: NativeValueType::Any,
            is_list: false,
        }];
        Ok(descriptor)
    }

    #[test]
    fn resolved_dotted_fields_assemble_without_serializing_handles()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut descriptor = runtime_descriptor(
            "Structured",
            true,
            BTreeMap::from([(
                "resize_type".to_owned(),
                RuntimeInputDescriptor {
                    value_type: ValueType::Any,
                    required: true,
                    hidden: false,
                    lazy: false,
                    mode: InputMode::Scalar,
                    allows_literal: true,
                },
            )]),
            EffectClass::Pure,
        )?;
        descriptor.inputs[0].accepted_types =
            NativeTypeUnion::new([NativeValueType::NamedPreservedUnknown(
                "COMFY_DYNAMICCOMBO_V3".to_owned(),
            )])?;
        let expression = json!({
            "arguments": [
                {"kind": "literal", "value": "match size"},
                {"kind": "list", "items": [{
                    "arguments": [
                        {"kind": "literal", "value": "match"},
                        {"kind": "list", "items": [
                            {"kind": "attribute", "name": "io.Image"},
                            {"kind": "attribute", "name": "io.Mask"}
                        ]}
                    ],
                    "keywords": [],
                    "kind": "call",
                    "name": "io.MultiType.Input"
                }]}
            ],
            "keywords": [],
            "kind": "call",
            "name": "io.DynamicCombo.Option"
        });
        let expression = serde_json::to_string(&expression)?;
        let mut source_schema = sim_nodes::NativeDescriptorSchemaMetadata::compatibility(
            sim_nodes::NativeSchemaProvenance::SourceV3,
            [("resize_type".to_owned(), "COMFY_DYNAMICCOMBO_V3".to_owned())],
            std::iter::empty(),
            [("value".to_owned(), "FLOAT".to_owned())],
        );
        source_schema.inputs[0].choices = vec![sim_nodes::NativeSchemaValue::PreservedExpression {
            sha256: format!("{:x}", Sha256::digest(expression.as_bytes())),
            source: expression,
        }];
        descriptor.source_schema = Some(source_schema);

        let store = NativeHandleStoreGeneration::new()?;
        let session = store.session(AttemptId(Uuid::from_u128(400)));
        let payload = stored_test_payload(b"structured".to_vec())?;
        let handle = session.publish(payload, &CancellationToken::default())?;
        let inputs = BTreeMap::from([
            (
                "resize_type".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "COMFY_DYNAMICCOMBO_V3".to_owned(),
                    value: json!("match size"),
                },
            ),
            (
                "resize_type.match".to_owned(),
                NativeValue::Handle {
                    value: handle.clone(),
                },
            ),
        ]);
        let assembled = assemble_structured_inputs(&descriptor, &inputs)?;
        assert_eq!(assembled.len(), 1);
        let structured = NativeStructuredValue::from_native_value(
            assembled
                .get("resize_type")
                .ok_or("missing structured input")?,
        )?
        .ok_or("structured input did not retain its typed representation")?;
        assert_eq!(
            structured.get("resize_type"),
            Some(&native_string("match size"))
        );
        assert_eq!(
            structured.get("match"),
            Some(&NativeValue::Handle { value: handle })
        );
        Ok(())
    }

    fn report_handle(report: &ExecutionReport) -> Option<NativeOpaqueHandle> {
        report
            .outputs
            .values()
            .flat_map(|values| values.iter())
            .find_map(|value| match value {
                NativeValue::Handle { value } => Some(value.clone()),
                NativeValue::Primitive { .. }
                | NativeValue::List { .. }
                | NativeValue::PreservedUnknown { .. } => None,
            })
    }

    fn input(
        value_type: ValueType,
        lazy: bool,
        mode: InputMode,
        literal: bool,
    ) -> RuntimeInputDescriptor {
        RuntimeInputDescriptor {
            value_type,
            required: true,
            hidden: false,
            lazy,
            mode,
            allows_literal: literal,
        }
    }

    #[test]
    fn component_presentation_registration_is_checked_and_atomic()
    -> Result<(), Box<dyn std::error::Error>> {
        let calls = Arc::new(AtomicUsize::new(0));
        let descriptor =
            runtime_descriptor("Component", false, BTreeMap::new(), EffectClass::Pure)?;
        let node: Arc<dyn NativeNode> = Arc::new(FixtureNode {
            class_type: "Component".to_owned(),
            calls,
        });
        let mut registry = NativeNodeRegistry::default();
        let error = registry
            .register_bound_batch_with_presentations([(
                descriptor.clone(),
                node.clone(),
                RuntimeNodePresentation {
                    display_name: "Signed Component".to_owned(),
                    category: "signed/category".to_owned(),
                    description: String::new(),
                    output_names: Vec::new(),
                    search_aliases: Vec::new(),
                    is_deprecated: false,
                    is_experimental: false,
                },
            )])
            .expect_err("output-name arity must match the execution descriptor");
        assert!(matches!(
            error,
            NativeNodeRegistryError::InvalidPresentation(class_type)
                if class_type == "Component"
        ));
        assert_eq!(registry.node_len(), 0);
        assert_eq!(registry.descriptor_len(), 0);
        assert!(registry.presentation("Component").is_none());

        let presentation = RuntimeNodePresentation {
            display_name: "Signed Component".to_owned(),
            category: "signed/category".to_owned(),
            description: String::new(),
            output_names: vec!["value".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: false,
        };
        registry.register_bound_batch_with_presentations([(
            descriptor,
            node,
            presentation.clone(),
        )])?;
        assert!(registry.descriptor_is_bound("Component"));
        assert_eq!(registry.presentation("Component"), Some(&presentation));
        assert_eq!(
            registry.binding_source("Component"),
            Some("zed.native_rust")
        );
        registry.validate_comprehensive_bindings()?;
        Ok(())
    }

    #[test]
    fn native_handle_store_sessions_isolate_stage_commit_and_revoke()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(vec![7])?;
        let payload_bytes = payload.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(4, payload_bytes * 4)?;
        let first_attempt = AttemptId(Uuid::from_u128(10));
        let second_attempt = AttemptId(Uuid::from_u128(11));
        let first = generation.session(first_attempt);
        let second = generation.session(second_attempt);
        let handle_type = payload.handle_type()?;
        let handle = first.publish(payload, &CancellationToken::default())?;

        assert!(matches!(
            second.resolve(&handle, &handle_type, &CancellationToken::default()),
            Err(NativeHandleStoreError::Missing(_))
        ));
        assert!(matches!(
            first
                .resolve(&handle, &handle_type, &CancellationToken::default())?
                .as_ref(),
            NativeStoredPayload::Provider(_)
        ));

        first.commit();
        assert!(matches!(
            first.publish(stored_test_payload(vec![8])?, &CancellationToken::default(),),
            Err(NativeHandleStoreError::Rejected(_))
        ));
        assert!(
            second
                .resolve(&handle, &handle_type, &CancellationToken::default())
                .is_ok()
        );
        assert!(matches!(
            second.revoke(&handle, &CancellationToken::default()),
            Err(NativeHandleStoreError::Rejected(_))
        ));

        let lease = generation
            .acquire_lease([&handle])?
            .ok_or("committed handle did not produce a lease")?;
        generation.collect_unrooted_attempt(first_attempt);
        assert_eq!(generation.len(), 1);
        drop(lease);
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn native_handle_store_abandoned_session_rolls_back_staged_values()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(b"abandoned-session".to_vec())?;
        let generation =
            NativeHandleStoreGeneration::with_capacities(1, payload.resident_bytes()?)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x2f20));
        let session = generation.handle_store_for_attempt(attempt_id);
        session.publish(payload, &CancellationToken::default())?;
        assert_eq!(generation.len(), 1);

        drop(session);

        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_handle_store_duplicate_attempt_reuses_one_session_and_close_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(b"duplicate-attempt".to_vec())?;
        let generation =
            NativeHandleStoreGeneration::with_capacities(1, payload.resident_bytes()?)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x2f21));
        let first = generation.session(attempt_id);
        let duplicate = generation.session(attempt_id);
        assert!(Arc::ptr_eq(&first, &duplicate));

        let handle = first.publish(payload, &CancellationToken::default())?;
        let resolved =
            duplicate.resolve(&handle, handle.handle_type(), &CancellationToken::default())?;
        assert!(matches!(
            resolved.as_ref(),
            NativeStoredPayload::Provider(_)
        ));
        drop(resolved);

        duplicate.rollback_all();
        assert!(generation.is_empty());
        assert!(matches!(
            first.publish(
                stored_test_payload(b"closed-duplicate".to_vec())?,
                &CancellationToken::default(),
            ),
            Err(NativeHandleStoreError::Rejected(_))
        ));
        drop(duplicate);
        drop(first);

        let reopened = generation.session(attempt_id);
        let reopened_handle = reopened.publish(
            stored_test_payload(b"reopened-attempt".to_vec())?,
            &CancellationToken::default(),
        )?;
        reopened.revoke(&reopened_handle, &CancellationToken::default())?;
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn native_executor_rejects_concurrent_duplicate_attempt_before_store_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let descriptor = publishing_handle_descriptor()?;
        let plan = Arc::new(compile_plan(
            vec![descriptor.clone()],
            BTreeMap::from([(
                NodeId("publish".to_owned()),
                PromptNode {
                    class_type: descriptor.class_type.clone(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            )]),
        )?);
        let calls = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let published_handle = Arc::new(Mutex::new(None));
        let mut registry = NativeNodeRegistry::default();
        registry.register_descriptor(descriptor)?;
        registry.register(Arc::new(BlockingPublishingHandleNode {
            calls: calls.clone(),
            entered: entered.clone(),
            release: release.clone(),
            published_handle: published_handle.clone(),
        }))?;
        let payload_bytes = stored_test_payload(b"claimed-attempt".to_vec())?.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(1, payload_bytes)?;
        let cache = Arc::new(Mutex::new(NativeCache::new(1)?));
        let (_backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
        let engine = Arc::new(ExecutionEngine::new_with_handle_store_generation(
            ProfileId(Uuid::from_u128(0x2f24)),
            Arc::new(registry),
            cache.clone(),
            Arc::new(RecordingEffectCoordinator::default()),
            "registry-v1",
            workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            generation.clone(),
        )?);
        let attempt_id = AttemptId(Uuid::from_u128(0x2f25));
        let first_execution = std::thread::spawn({
            let engine = engine.clone();
            let plan = plan.clone();
            move || {
                smol::block_on(engine.execute(
                    plan.as_ref(),
                    attempt_id,
                    CancellationToken::default(),
                ))
            }
        });
        entered.wait();

        let duplicate =
            smol::block_on(engine.execute(plan.as_ref(), attempt_id, CancellationToken::default()));
        assert_eq!(duplicate.state, AttemptState::Failed);
        assert!(duplicate.outputs.is_empty());
        assert!(duplicate.events.is_empty());
        assert!(
            duplicate
                .error
                .as_deref()
                .is_some_and(|error| error.contains("already active"))
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let handle = published_handle
            .lock()
            .clone()
            .ok_or("first execution did not publish its staged handle")?;
        {
            let data = generation.state.data.lock();
            let record = data
                .values
                .get(handle.identifier())
                .ok_or("duplicate execution removed the first execution's handle")?;
            assert!(!record.committed);
            assert!(!record.retired);
        }
        let shared_session = generation.session(attempt_id);
        let resolved =
            shared_session.resolve(&handle, handle.handle_type(), &CancellationToken::default())?;
        drop(resolved);

        release.wait();
        let first = first_execution
            .join()
            .map_err(|_| "first execution thread panicked")?;
        assert_eq!(first.state, AttemptState::Succeeded);
        assert_eq!(report_handle(&first), Some(handle.clone()));
        {
            let data = generation.state.data.lock();
            let record = data
                .values
                .get(handle.identifier())
                .ok_or("successful first execution lost its handle")?;
            assert!(record.committed);
            assert!(!record.retired);
        }
        drop(shared_session);
        drop(first);
        cache.lock().clear();
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_handle_resolve_cancellation_after_root_increment_is_atomic()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(b"resolve-cancellation".to_vec())?;
        let generation =
            NativeHandleStoreGeneration::with_capacities(1, payload.resident_bytes()?)?;
        let session = generation.session(AttemptId(Uuid::from_u128(0x2f22)));
        let handle = session.publish(payload, &CancellationToken::default())?;
        let cancellation = CancellationToken::default();
        generation.set_after_resolve_increment_hook(Arc::new({
            let cancellation = cancellation.clone();
            move || {
                cancellation.cancel();
            }
        }));

        assert!(matches!(
            session.resolve(&handle, handle.handle_type(), &cancellation),
            Err(NativeHandleStoreError::Cancelled)
        ));
        let data = generation.state.data.lock();
        let record = data
            .values
            .get(handle.identifier())
            .ok_or("cancelled resolve removed its staged handle")?;
        assert_eq!(record.resolved_roots, 0);
        drop(data);
        session.revoke(&handle, &CancellationToken::default())?;
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn native_handle_root_overflow_rejection_is_atomic() -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(b"root-overflow".to_vec())?;
        let generation =
            NativeHandleStoreGeneration::with_capacities(1, payload.resident_bytes()?)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x2f23));
        let session = generation.session(attempt_id);
        let handle = session.publish(payload, &CancellationToken::default())?;
        session.commit();
        {
            let mut data = generation.state.data.lock();
            let record = data
                .values
                .get_mut(handle.identifier())
                .ok_or("committed handle was missing")?;
            record.roots = usize::MAX;
        }
        assert!(matches!(
            generation.acquire_lease([&handle]),
            Err(NativeHandleStoreError::Rejected(_))
        ));
        {
            let mut data = generation.state.data.lock();
            let record = data
                .values
                .get_mut(handle.identifier())
                .ok_or("committed handle was missing after lease overflow")?;
            assert_eq!(record.roots, usize::MAX);
            record.roots = 0;
            record.resolved_roots = usize::MAX;
        }
        assert!(matches!(
            session.resolve(&handle, handle.handle_type(), &CancellationToken::default(),),
            Err(NativeHandleStoreError::Rejected(_))
        ));
        {
            let mut data = generation.state.data.lock();
            let record = data
                .values
                .get_mut(handle.identifier())
                .ok_or("committed handle was missing after resolve overflow")?;
            assert_eq!(record.resolved_roots, usize::MAX);
            record.resolved_roots = 0;
        }
        generation.collect_unrooted_attempt(attempt_id);
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn native_handle_store_clip_vision_payloads_enforce_identity_and_alias_residency()
    -> Result<(), Box<dyn std::error::Error>> {
        let clip_vision = tiny_clip_vision_resource()?;
        let model_owner = Arc::new(NativeModelPayload::clip_vision(clip_vision)?);
        let stored_model = Arc::new(NativeStoredModelPayload::model_resource(
            model_owner.clone(),
        )?);
        let model_payload = NativeStoredPayload::Model(stored_model);
        let model_type = model_payload.handle_type()?;
        assert_eq!(model_type.type_id, "CLIP_VISION");

        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let hidden =
            clip_vision_test_tensor(&backend, &authority, &cancellation, vec![1, 2, 2], 1.0)?;
        let embeds = clip_vision_test_tensor(&backend, &authority, &cancellation, vec![1, 2], 2.0)?;
        let output_owner = Arc::new(ClipVisionOutput::checked(
            hidden.clone(),
            Some(hidden),
            embeds,
            None,
            vec![[3, 32, 32]],
        )?);
        let output_parts = output_owner.resident_parts()?;
        assert_eq!(output_parts.tensor_allocations().len(), 2);
        let output_payload = NativeStoredPayload::ClipVisionOutput(output_owner.clone());
        let output_type = output_payload.handle_type()?;
        assert_eq!(output_type.type_id, ClipVisionOutput::SOURCE_TYPE_ID);
        assert_ne!(model_type, output_type);
        assert_eq!(
            output_payload.residency()?.resident_bytes()?,
            usize::try_from(output_parts.resident_bytes()?)?
        );
        assert_eq!(
            output_payload.resident_bytes()?,
            usize::try_from(output_owner.resident_bytes())?
        );

        let byte_capacity = model_payload
            .resident_bytes()?
            .checked_add(output_payload.resident_bytes()?)
            .ok_or("clip vision store capacity overflowed")?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, byte_capacity)?;
        let session = generation.session(AttemptId(Uuid::from_u128(0x2f26)));
        let model_handle = session.publish(model_payload, &CancellationToken::default())?;
        let output_handle =
            session.publish(output_payload.clone(), &CancellationToken::default())?;
        assert_eq!(generation.len(), 2);
        assert_eq!(generation.resident_bytes(), byte_capacity);

        let resolved_model =
            session.resolve(&model_handle, &model_type, &CancellationToken::default())?;
        let NativeStoredPayload::Model(resolved_model_payload) = resolved_model.as_ref() else {
            return Err("CLIP vision model handle resolved to the wrong payload".into());
        };
        assert!(Arc::ptr_eq(
            resolved_model_payload.model_payload(),
            &model_owner
        ));
        let resolved_output =
            session.resolve(&output_handle, &output_type, &CancellationToken::default())?;
        let NativeStoredPayload::ClipVisionOutput(resolved_output_payload) =
            resolved_output.as_ref()
        else {
            return Err("CLIP vision output handle resolved to the wrong payload".into());
        };
        assert!(Arc::ptr_eq(resolved_output_payload, &output_owner));
        assert!(matches!(
            session.resolve(&model_handle, &output_type, &CancellationToken::default(),),
            Err(NativeHandleStoreError::WrongType { .. })
        ));
        assert!(matches!(
            session.resolve(&output_handle, &model_type, &CancellationToken::default(),),
            Err(NativeHandleStoreError::WrongType { .. })
        ));
        let forged_output = NativeOpaqueHandle::new(
            output_type,
            generation.identity(),
            output_handle.identifier(),
            output_handle.generation(),
            Some("f".repeat(64)),
        )?;
        assert!(matches!(
            session.resolve(
                &forged_output,
                forged_output.handle_type(),
                &CancellationToken::default(),
            ),
            Err(NativeHandleStoreError::DigestMismatch)
        ));
        drop(resolved_output);
        drop(resolved_model);
        session.rollback_all();
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);

        let cancelled_generation =
            NativeHandleStoreGeneration::with_capacities(1, output_payload.resident_bytes()?)?;
        let cancelled_session = cancelled_generation.session(AttemptId(Uuid::from_u128(0x2f27)));
        let late_cancellation = CancellationToken::default();
        cancelled_generation.set_after_publish_insert_hook(Arc::new({
            let late_cancellation = late_cancellation.clone();
            move || {
                late_cancellation.cancel();
            }
        }));
        assert!(matches!(
            cancelled_session.publish(output_payload, &late_cancellation),
            Err(NativeHandleStoreError::Cancelled)
        ));
        assert!(cancelled_generation.is_empty());
        assert_eq!(cancelled_generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_handle_store_capacity_and_lease_validation_are_atomic()
    -> Result<(), Box<dyn std::error::Error>> {
        let first_payload = stored_test_payload(vec![1])?;
        let second_payload = stored_test_payload(vec![2])?;
        let oversized_payload = stored_test_payload(vec![3; 4_096])?;
        let maximum_bytes = first_payload
            .resident_bytes()?
            .checked_add(second_payload.resident_bytes()?)
            .ok_or("test capacity overflowed")?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, maximum_bytes)?;
        let attempt_id = AttemptId(Uuid::from_u128(12));
        let session = generation.session(attempt_id);
        let handle_type = first_payload.handle_type()?;
        let first = session.publish(first_payload, &CancellationToken::default())?;
        assert!(matches!(
            session.publish(oversized_payload, &CancellationToken::default(),),
            Err(NativeHandleStoreError::Rejected(_))
        ));
        let second = session.publish(second_payload, &CancellationToken::default())?;
        assert_eq!(generation.len(), 2);
        assert_eq!(generation.resident_bytes(), maximum_bytes);
        session.commit();

        let forged = NativeOpaqueHandle::new(
            handle_type,
            generation.identity(),
            second.identifier(),
            second.generation(),
            Some("d".repeat(64)),
        )?;
        assert!(matches!(
            generation.acquire_lease([&first, &forged]),
            Err(NativeHandleStoreError::DigestMismatch)
        ));
        let data = generation.state.data.lock();
        assert!(data.values.values().all(|record| record.roots == 0));
        drop(data);
        generation.collect_unrooted_attempt(attempt_id);
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn native_handle_store_deduplicates_shared_allocations_until_final_owner()
    -> Result<(), Box<dyn std::error::Error>> {
        let semantic_digest_sha256 = format!("{:x}", Sha256::digest(b"shared-payload"));
        let shared = Arc::new(NativeProviderPayload::checked(
            NativeHandleType::new(NativeHandleKind::ProviderTask, "TEST_PROVIDER_TASK")?,
            "zed.test.provider",
            semantic_digest_sha256,
            b"shared-payload".to_vec(),
        )?);
        let payload = NativeStoredPayload::Provider(shared.clone());
        let payload_bytes = payload.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, payload_bytes)?;
        let attempt_id = AttemptId(Uuid::from_u128(0x2f01));
        let session = generation.session(attempt_id);
        let first = session.publish(
            NativeStoredPayload::Provider(shared.clone()),
            &CancellationToken::default(),
        )?;
        let second = session.publish(
            NativeStoredPayload::Provider(shared),
            &CancellationToken::default(),
        )?;
        assert_eq!(generation.len(), 2);
        assert_eq!(generation.resident_bytes(), payload_bytes);

        session.commit();
        let first_lease = generation
            .acquire_lease([&first])?
            .ok_or("first handle did not produce a lease")?;
        let second_lease = generation
            .acquire_lease([&second])?
            .ok_or("second handle did not produce a lease")?;
        generation.collect_unrooted_attempt(attempt_id);
        drop(first_lease);
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), payload_bytes);
        drop(second_lease);
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn resolved_payload_guard_retires_logically_and_releases_shared_capacity_on_final_drop()
    -> Result<(), Box<dyn std::error::Error>> {
        let semantic_digest_sha256 = format!("{:x}", Sha256::digest(b"guarded-shared-payload"));
        let shared = Arc::new(NativeProviderPayload::checked(
            NativeHandleType::new(NativeHandleKind::ProviderTask, "TEST_PROVIDER_TASK")?,
            "zed.test.provider",
            semantic_digest_sha256,
            b"guarded-shared-payload".to_vec(),
        )?);
        let payload = NativeStoredPayload::Provider(shared.clone());
        let payload_bytes = payload.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(3, payload_bytes)?;
        let cancellation = CancellationToken::default();
        let first_session = generation.session(AttemptId(Uuid::from_u128(0x2f10)));
        let first = first_session.publish(payload, &cancellation)?;
        let resolved = first_session.resolve(&first, first.handle_type(), &cancellation)?;
        let resolved_clone = resolved.clone();

        first_session.rollback_all();
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), payload_bytes);
        assert!(matches!(
            first_session.resolve(&first, first.handle_type(), &cancellation),
            Err(NativeHandleStoreError::Missing(_))
        ));

        let second_session = generation.session(AttemptId(Uuid::from_u128(0x2f11)));
        let second = second_session.publish(
            NativeStoredPayload::Provider(shared),
            &CancellationToken::default(),
        )?;
        assert_eq!(generation.len(), 2);
        assert_eq!(generation.resident_bytes(), payload_bytes);
        assert!(matches!(
            second_session.publish(
                stored_test_payload(b"distinct-capacity-allocation".to_vec())?,
                &CancellationToken::default(),
            ),
            Err(NativeHandleStoreError::Rejected(_))
        ));
        assert_eq!(generation.len(), 2);
        assert_eq!(generation.resident_bytes(), payload_bytes);

        second_session.revoke(&second, &CancellationToken::default())?;
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), payload_bytes);
        drop(resolved);
        assert_eq!(generation.len(), 1);
        drop(resolved_clone);
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_handle_store_allocation_capacity_rejection_is_atomic()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = stored_test_payload(b"first-allocation".to_vec())?;
        let first_bytes = first.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, first_bytes)?;
        let session = generation.session(AttemptId(Uuid::from_u128(0x2f02)));
        session.publish(first, &CancellationToken::default())?;
        let second = stored_test_payload(b"second-distinct-allocation".to_vec())?;
        assert!(matches!(
            session.publish(second, &CancellationToken::default()),
            Err(NativeHandleStoreError::Rejected(_))
        ));
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), first_bytes);
        session.rollback_all();
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_handle_store_accepts_zero_byte_payloads_and_never_wraps_generation()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            sim_tensor::StreamId::DEFAULT,
            workspace_authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let tensor =
            sim_tensor::generated_native_diffusion::tensor_from_f32(&backend, &[0], &[], &context)?;
        let payload =
            NativeStoredPayload::Tensor(Arc::new(sim_tensor::NativeTensorPayload::from_tensor(
                sim_tensor::NativeTensorRole::Sigmas,
                tensor,
            )?));
        assert_eq!(payload.resident_bytes()?, 0);
        let generation = NativeHandleStoreGeneration::with_capacities(2, 1)?;
        let session = generation.session(AttemptId(Uuid::from_u128(121)));
        let first = session.publish(payload.clone(), &cancellation)?;
        let second = session.publish(payload.clone(), &cancellation)?;
        assert_eq!(generation.resident_bytes(), 0);
        assert_eq!(generation.len(), 2);
        assert!(matches!(
            session.publish(payload.clone(), &cancellation),
            Err(NativeHandleStoreError::Rejected(message))
                if message.contains("capacity is exhausted")
        ));
        session.revoke(&first, &cancellation)?;
        session.revoke(&second, &cancellation)?;
        assert!(generation.is_empty());

        generation
            .state
            .next_generation
            .store(u64::MAX - 1, Ordering::Release);
        let final_handle = session.publish(payload.clone(), &cancellation)?;
        assert_eq!(final_handle.generation(), u64::MAX - 1);
        assert_eq!(final_handle.identifier(), "native-fffffffffffffffe");
        session.revoke(&final_handle, &cancellation)?;
        assert_eq!(
            generation.state.next_generation.load(Ordering::Acquire),
            u64::MAX
        );
        assert!(matches!(
            session.publish(payload, &cancellation),
            Err(NativeHandleStoreError::Rejected(message))
                if message.contains("generation was exhausted")
        ));
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_handle_publish_cancellation_is_atomic_before_validation_and_after_insert()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(vec![1, 2, 3])?;
        let generation =
            NativeHandleStoreGeneration::with_capacities(2, payload.resident_bytes()?)?;
        let session = generation.session(AttemptId(Uuid::from_u128(124)));

        let pre_validation = CancellationToken::default();
        pre_validation.cancel();
        assert!(matches!(
            session.publish(payload.clone(), &pre_validation),
            Err(NativeHandleStoreError::Cancelled)
        ));
        assert!(generation.is_empty());
        assert_eq!(generation.state.next_generation.load(Ordering::Acquire), 1);

        let post_insert = CancellationToken::default();
        generation.set_after_publish_insert_hook(Arc::new({
            let post_insert = post_insert.clone();
            move || {
                post_insert.cancel();
            }
        }));
        assert!(matches!(
            session.publish(payload, &post_insert),
            Err(NativeHandleStoreError::Cancelled)
        ));
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        assert_eq!(generation.state.next_generation.load(Ordering::Acquire), 2);
        Ok(())
    }

    #[test]
    fn native_handle_publish_racing_session_close_never_leaks_staged_values()
    -> Result<(), Box<dyn std::error::Error>> {
        for rollback in [false, true] {
            for iteration in 0..64_u128 {
                let payload = stored_test_payload(vec![u8::try_from(iteration)?])?;
                let generation =
                    NativeHandleStoreGeneration::with_capacities(1, payload.resident_bytes()?)?;
                let session = generation.session(AttemptId(Uuid::from_u128(200 + iteration)));
                let barrier = Arc::new(Barrier::new(2));
                let publisher = std::thread::spawn({
                    let barrier = barrier.clone();
                    let session = session.clone();
                    move || {
                        barrier.wait();
                        session.publish(payload, &CancellationToken::default())
                    }
                });
                let closer = std::thread::spawn({
                    let barrier = barrier.clone();
                    let session = session.clone();
                    move || {
                        barrier.wait();
                        if rollback {
                            session.rollback_all();
                        } else {
                            session.commit();
                        }
                    }
                });
                let published = publisher.join().map_err(|_| "publisher thread panicked")?;
                closer.join().map_err(|_| "closer thread panicked")?;

                if rollback {
                    assert!(generation.is_empty());
                    assert!(matches!(
                        published,
                        Ok(_) | Err(NativeHandleStoreError::Rejected(_))
                    ));
                } else {
                    match published {
                        Ok(handle) => {
                            let reader = generation.session(AttemptId(Uuid::from_u128(10_000)));
                            assert!(
                                reader
                                    .resolve(
                                        &handle,
                                        handle.handle_type(),
                                        &CancellationToken::default(),
                                    )
                                    .is_ok()
                            );
                            let lease = generation
                                .acquire_lease([&handle])?
                                .ok_or("committed handle lease was missing")?;
                            drop(lease);
                            assert!(generation.is_empty());
                        }
                        Err(NativeHandleStoreError::Rejected(_)) => {
                            assert!(generation.is_empty());
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                assert_eq!(generation.resident_bytes(), 0);
            }
        }
        Ok(())
    }

    #[test]
    fn native_handle_resolve_rejects_forged_store_generation_type_and_digest()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(vec![9])?;
        let generation =
            NativeHandleStoreGeneration::with_capacities(1, payload.resident_bytes()?)?;
        let attempt_id = AttemptId(Uuid::from_u128(122));
        let session = generation.session(attempt_id);
        let handle = session.publish(payload, &CancellationToken::default())?;
        let expected = handle.handle_type().clone();
        let other = NativeHandleStoreGeneration::new()?;
        let wrong_store = NativeOpaqueHandle::new(
            expected.clone(),
            other.identity(),
            handle.identifier(),
            handle.generation(),
            handle.digest_sha256().map(ToOwned::to_owned),
        )?;
        let wrong_generation = NativeOpaqueHandle::new(
            expected.clone(),
            NativeHandleStoreIdentity::new(generation.identity().store_id, Uuid::from_u128(123))?,
            handle.identifier(),
            handle.generation(),
            handle.digest_sha256().map(ToOwned::to_owned),
        )?;
        let wrong_type = NativeHandleType::new(NativeHandleKind::Model, "MODEL")?;
        let wrong_type_handle = NativeOpaqueHandle::new(
            wrong_type.clone(),
            generation.identity(),
            handle.identifier(),
            handle.generation(),
            handle.digest_sha256().map(ToOwned::to_owned),
        )?;
        let wrong_digest = NativeOpaqueHandle::new(
            expected.clone(),
            generation.identity(),
            handle.identifier(),
            handle.generation(),
            Some("f".repeat(64)),
        )?;
        assert!(matches!(
            session.resolve(&wrong_store, &expected, &CancellationToken::default()),
            Err(NativeHandleStoreError::WrongStore)
        ));
        assert!(matches!(
            session.resolve(&wrong_generation, &expected, &CancellationToken::default()),
            Err(NativeHandleStoreError::WrongGeneration)
        ));
        assert!(matches!(
            session.resolve(&handle, &wrong_type, &CancellationToken::default()),
            Err(NativeHandleStoreError::WrongType { .. })
        ));
        assert!(matches!(
            session.resolve(
                &wrong_type_handle,
                &wrong_type,
                &CancellationToken::default(),
            ),
            Err(NativeHandleStoreError::WrongType { .. })
        ));
        assert!(matches!(
            session.resolve(&wrong_digest, &expected, &CancellationToken::default()),
            Err(NativeHandleStoreError::DigestMismatch)
        ));
        assert_eq!(generation.len(), 1);
        Ok(())
    }

    #[test]
    fn native_cache_leases_cover_aliases_and_release_on_lru_and_invalidation()
    -> Result<(), Box<dyn std::error::Error>> {
        let first_payload = stored_test_payload(vec![1])?;
        let second_payload = stored_test_payload(vec![2])?;
        let maximum_bytes = first_payload
            .resident_bytes()?
            .checked_add(second_payload.resident_bytes()?)
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or("test capacity overflowed")?;
        let generation = NativeHandleStoreGeneration::with_capacities(4, maximum_bytes)?;
        let attempt_id = AttemptId(Uuid::from_u128(13));
        let session = generation.session(attempt_id);
        let first = session.publish(first_payload, &CancellationToken::default())?;
        let second = session.publish(second_payload, &CancellationToken::default())?;
        session.commit();
        let first_entry = CacheEntry {
            outputs: vec![
                NativeValue::Handle {
                    value: first.clone(),
                },
                NativeValue::List {
                    values: vec![
                        NativeValue::Handle {
                            value: first.clone(),
                        },
                        NativeValue::List {
                            values: vec![NativeValue::Handle {
                                value: first.clone(),
                            }],
                        },
                    ],
                },
            ],
            ui: None,
        };
        let second_entry = CacheEntry {
            outputs: vec![NativeValue::Handle {
                value: second.clone(),
            }],
            ui: None,
        };
        let first_key = CacheKey::from_inputs(
            "First",
            "1",
            &BTreeMap::new(),
            BTreeMap::new(),
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )?;
        let second_key = CacheKey::from_inputs(
            "Second",
            "1",
            &BTreeMap::new(),
            BTreeMap::new(),
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )?;
        let mut cache = NativeCache::new(1)?;
        assert!(!cache.insert(first_key.clone(), first_entry.clone()));
        assert!(cache.is_empty());
        assert!(cache.insert_with_handle_lease(
            first_key,
            first_entry,
            generation.acquire_lease([&first])?,
        ));
        assert!(cache.insert_with_handle_lease(
            second_key,
            second_entry,
            generation.acquire_lease([&second])?,
        ));
        assert_eq!(generation.len(), 1);
        assert_eq!(cache.invalidate_node("Second"), 1);
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn cache_eviction_retires_handles_but_resolved_guard_retains_shared_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        let semantic_digest_sha256 = format!("{:x}", Sha256::digest(b"cache-shared-payload"));
        let shared = Arc::new(NativeProviderPayload::checked(
            NativeHandleType::new(NativeHandleKind::ProviderTask, "TEST_PROVIDER_TASK")?,
            "zed.test.provider",
            semantic_digest_sha256,
            b"cache-shared-payload".to_vec(),
        )?);
        let payload_bytes = NativeStoredPayload::Provider(shared.clone()).resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, payload_bytes)?;
        let session = generation.session(AttemptId(Uuid::from_u128(0x2f12)));
        let first = session.publish(
            NativeStoredPayload::Provider(shared.clone()),
            &CancellationToken::default(),
        )?;
        let second = session.publish(
            NativeStoredPayload::Provider(shared),
            &CancellationToken::default(),
        )?;
        session.commit();
        let reader = generation.session(AttemptId(Uuid::from_u128(0x2f13)));
        let resolved =
            reader.resolve(&first, first.handle_type(), &CancellationToken::default())?;
        let first_entry = CacheEntry {
            outputs: vec![NativeValue::Handle {
                value: first.clone(),
            }],
            ui: None,
        };
        let second_entry = CacheEntry {
            outputs: vec![NativeValue::Handle {
                value: second.clone(),
            }],
            ui: None,
        };
        let mut cache = NativeCache::new(1)?;
        assert!(cache.insert_with_handle_lease(
            test_cache_key("GuardedFirst")?,
            first_entry,
            generation.acquire_lease([&first])?,
        ));
        assert!(cache.insert_with_handle_lease(
            test_cache_key("GuardedSecond")?,
            second_entry,
            generation.acquire_lease([&second])?,
        ));
        assert_eq!(generation.len(), 2);
        assert_eq!(generation.resident_bytes(), payload_bytes);
        assert!(matches!(
            reader.resolve(&first, first.handle_type(), &CancellationToken::default(),),
            Err(NativeHandleStoreError::Missing(_))
        ));

        cache.clear();
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.resident_bytes(), payload_bytes);
        drop(resolved);
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn native_cache_same_key_replacement_transfers_exact_handle_roots()
    -> Result<(), Box<dyn std::error::Error>> {
        let first_payload = stored_test_payload(vec![1])?;
        let second_payload = stored_test_payload(vec![2])?;
        let byte_capacity = first_payload
            .resident_bytes()?
            .checked_add(second_payload.resident_bytes()?)
            .ok_or("test byte capacity overflowed")?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, byte_capacity)?;
        let session = generation.session(AttemptId(Uuid::from_u128(310)));
        let first = session.publish(first_payload, &CancellationToken::default())?;
        let second = session.publish(second_payload, &CancellationToken::default())?;
        session.commit();

        let key = test_cache_key("Replacement")?;
        let first_entry = CacheEntry {
            outputs: vec![NativeValue::Handle {
                value: first.clone(),
            }],
            ui: None,
        };
        let second_entry = CacheEntry {
            outputs: vec![NativeValue::Handle {
                value: second.clone(),
            }],
            ui: None,
        };
        let mut cache = NativeCache::new(1)?;
        assert!(cache.insert_with_handle_lease(
            key.clone(),
            first_entry.clone(),
            generation.acquire_lease([&first])?,
        ));
        assert!(cache.insert_with_handle_lease(
            key.clone(),
            first_entry,
            generation.acquire_lease([&first])?,
        ));
        assert_eq!(generation.len(), 2);
        assert!(cache.insert_with_handle_lease(
            key,
            second_entry,
            generation.acquire_lease([&second])?,
        ));
        assert_eq!(generation.len(), 1);
        assert!(matches!(
            session.resolve(&first, first.handle_type(), &CancellationToken::default(),),
            Err(NativeHandleStoreError::Missing(_))
        ));
        assert!(
            session
                .resolve(&second, second.handle_type(), &CancellationToken::default(),)
                .is_ok()
        );
        cache.clear();
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn cancelled_cache_publication_restores_replaced_entry_and_exact_roots()
    -> Result<(), Box<dyn std::error::Error>> {
        let first_payload = stored_test_payload(vec![1])?;
        let second_payload = stored_test_payload(vec![2])?;
        let byte_capacity = first_payload
            .resident_bytes()?
            .checked_add(second_payload.resident_bytes()?)
            .ok_or("test byte capacity overflowed")?;
        let generation = NativeHandleStoreGeneration::with_capacities(2, byte_capacity)?;
        let session = generation.session(AttemptId(Uuid::from_u128(320)));
        let first = session.publish(first_payload, &CancellationToken::default())?;
        let second = session.publish(second_payload, &CancellationToken::default())?;
        session.commit();
        let key = test_cache_key("AtomicReplacement")?;
        let first_entry = CacheEntry {
            outputs: vec![NativeValue::Handle {
                value: first.clone(),
            }],
            ui: None,
        };
        let second_entry = CacheEntry {
            outputs: vec![NativeValue::Handle {
                value: second.clone(),
            }],
            ui: None,
        };
        let cache = Arc::new(Mutex::new(NativeCache::new(1)?));
        assert!(cache.lock().insert_with_handle_lease(
            key.clone(),
            first_entry.clone(),
            generation.acquire_lease([&first])?,
        ));
        let (_backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
        let engine = ExecutionEngine::new_with_handle_store_generation(
            ProfileId(Uuid::from_u128(321)),
            Arc::new(NativeNodeRegistry::default()),
            cache.clone(),
            Arc::new(RecordingEffectCoordinator::default()),
            "registry-v1",
            workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            generation.clone(),
        )?;
        let pre_publication = CancellationToken::default();
        pre_publication.cancel();
        assert!(matches!(
            engine.publish_cache_batch(Vec::new(), &pre_publication),
            Err(ExecutionError::Cancelled)
        ));
        assert_eq!(cache.lock().get(&key), Some(first_entry.clone()));
        assert_eq!(generation.len(), 2);
        let cancellation = CancellationToken::default();
        generation.set_after_cache_insert_hook(Arc::new({
            let cancellation = cancellation.clone();
            move || {
                cancellation.cancel();
            }
        }));
        assert!(matches!(
            engine.publish_cache_batch(
                vec![(
                    key.clone(),
                    second_entry,
                    generation.acquire_lease([&second])?,
                )],
                &cancellation,
            ),
            Err(ExecutionError::Cancelled)
        ));
        assert_eq!(cache.lock().get(&key), Some(first_entry));
        assert_eq!(generation.len(), 1);
        assert!(
            session
                .resolve(&first, first.handle_type(), &CancellationToken::default(),)
                .is_ok()
        );
        assert!(matches!(
            session.resolve(&second, second.handle_type(), &CancellationToken::default(),),
            Err(NativeHandleStoreError::Missing(_))
        ));
        cache.lock().clear();
        assert!(generation.is_empty());
        Ok(())
    }

    #[test]
    fn cloned_native_handle_lease_releases_roots_once_under_concurrent_drop()
    -> Result<(), Box<dyn std::error::Error>> {
        let payload = stored_test_payload(vec![7])?;
        let byte_capacity = payload.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(1, byte_capacity)?;
        let attempt_id = AttemptId(Uuid::from_u128(14));
        let session = generation.session(attempt_id);
        let handle = session.publish(payload, &CancellationToken::default())?;
        session.commit();
        let lease = generation
            .acquire_lease([&handle])?
            .ok_or("test lease was not acquired")?;
        let second = lease.clone();
        let barrier = Arc::new(Barrier::new(3));
        std::thread::scope(|scope| {
            for lease in [lease, second] {
                let barrier = barrier.clone();
                scope.spawn(move || {
                    barrier.wait();
                    drop(lease);
                });
            }
            barrier.wait();
        });
        assert!(generation.is_empty());
        assert_eq!(generation.resident_bytes(), 0);
        Ok(())
    }

    #[test]
    fn provider_activation_is_checked_atomic_and_preserves_declared_provenance()
    -> Result<(), Box<dyn std::error::Error>> {
        let catalog = NodeRegistry::built_in()?;
        let catalog_descriptor = catalog
            .registered()
            .values()
            .find(|descriptor| {
                descriptor.catalog_status == sim_nodes::CatalogNodeStatus::ProviderRequired
            })
            .ok_or("generated catalog did not include a provider-required binding")?;
        let presentation = RuntimeNodePresentation {
            display_name: catalog_descriptor.display_name.clone(),
            category: match catalog_descriptor.category.as_str() {
                "(empty root category declared by source)" => String::new(),
                category => category.to_owned(),
            },
            description: String::new(),
            output_names: vec!["value".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: false,
        };
        let descriptor = RuntimeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: catalog_descriptor.node_identifier.clone(),
            implementation_version: "provider-v1".to_owned(),
            source_schema: Some(sim_nodes::NativeDescriptorSchemaMetadata::synthetic(
                std::iter::empty(),
                std::iter::empty(),
                ["value".to_owned()],
            )),
            inputs: Vec::new(),
            dynamic_inputs: Vec::new(),
            outputs: vec![RuntimeOutputDescriptor {
                name: "value".to_owned(),
                produced_type: NativeValueType::Any,
                is_list: false,
            }],
            output_node: catalog_descriptor.output_node,
            effect: EffectClass::Provider,
            cache: RuntimeCachePolicy::Never,
        };
        let binding = NativeNodeBinding::ProviderRequired {
            feature_id: catalog_descriptor.feature_id.clone(),
            descriptor,
            presentation,
            provider: "zed.provider.test".to_owned(),
            reason: "verified provider activation is required".to_owned(),
        };
        let NativeNodeBinding::ProviderRequired {
            descriptor,
            presentation: _,
            provider,
            ..
        } = binding.clone()
        else {
            return Err("selected binding was not provider-required".into());
        };
        let node: Arc<dyn NativeNode> = Arc::new(ConfiguredNode {
            class_type: descriptor.class_type.clone(),
            version: descriptor.implementation_version.clone(),
            namespace: provider,
        });
        let mut registry = NativeNodeRegistry::default();
        registry.register_native_bindings([binding])?;
        registry.validate_comprehensive_bindings()?;

        let transport_schema: sim_plugin_sdk::CanonicalTypeId =
            "zed:sim-provider-transport@1".parse()?;
        let materializer_schema: sim_plugin_sdk::CanonicalTypeId =
            "zed:sim-provider-materializer@1".parse()?;
        assert!(matches!(
            registry.provider_binding_contract_sha256(
                &descriptor.class_type,
                "zed:unsupported-provider-transport@1",
                &materializer_schema.to_string(),
            ),
            Err(NativeNodeRegistryError::InvalidProviderActivation)
        ));
        assert!(matches!(
            registry.provider_binding_contract_sha256(
                &descriptor.class_type,
                &transport_schema.to_string(),
                "zed:unsupported-provider-materializer@1",
            ),
            Err(NativeNodeRegistryError::InvalidProviderActivation)
        ));
        let contract_sha256 = registry
            .provider_binding_contract_sha256(
                &descriptor.class_type,
                &transport_schema.to_string(),
                &materializer_schema.to_string(),
            )?
            .ok_or("provider contract was not projected")?;
        let claim = ProviderBindingClaim {
            feature_id: catalog_descriptor.feature_id.clone(),
            node_id: descriptor.class_type.clone(),
            contract_sha256,
            transport_schema,
            materializer_schema,
        };
        let mut binding_set = ProviderBindingSet {
            schema_version: sim_plugin_sdk::PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: "zed.provider.test".to_owned(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![claim.clone()],
        };
        binding_set.bindings_sha256 = binding_set.canonical_bindings_sha256()?;
        let mismatched_node: Arc<dyn NativeNode> = Arc::new(ConfiguredNode {
            class_type: descriptor.class_type.clone(),
            version: descriptor.implementation_version.clone(),
            namespace: "zed.provider.mismatch".to_owned(),
        });
        let mismatched_activation = NativeProviderBindingActivationSet::checked(
            "profile-a",
            1,
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            binding_set.clone(),
            vec![NativeProviderBindingActivation::new(
                claim.clone(),
                mismatched_node,
            )],
        )?;
        assert!(matches!(
            registry.activate_provider_binding_set(mismatched_activation),
            Err(NativeNodeRegistryError::BindingMismatch(class_type))
                if class_type == descriptor.class_type
        ));
        assert_eq!(
            registry.provider_binding_is_activated(&descriptor.class_type),
            Some(false)
        );
        assert!(registry.node(&descriptor.class_type).is_none());

        let activation = NativeProviderBindingActivationSet::checked(
            "profile-a",
            1,
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            binding_set,
            vec![NativeProviderBindingActivation::new(claim, node)],
        )?;
        registry.activate_provider_binding_set(activation)?;
        assert_eq!(
            registry.binding_declared_disposition(&descriptor.class_type),
            Some(NativeNodeBindingDisposition::ProviderRequired)
        );
        assert_eq!(
            registry.binding_disposition(&descriptor.class_type),
            Some(NativeNodeBindingDisposition::Executable)
        );
        assert_eq!(
            registry.provider_binding_is_activated(&descriptor.class_type),
            Some(true)
        );
        registry.validate_comprehensive_bindings()?;
        Ok(())
    }

    #[test]
    fn provider_activation_requires_the_complete_signed_namespace_set()
    -> Result<(), Box<dyn std::error::Error>> {
        let catalog = NodeRegistry::built_in()?;
        let catalog_descriptors = catalog
            .registered()
            .values()
            .filter(|descriptor| {
                descriptor.catalog_status == sim_nodes::CatalogNodeStatus::ProviderRequired
            })
            .take(2)
            .cloned()
            .collect::<Vec<_>>();
        if catalog_descriptors.len() != 2 {
            return Err("generated catalog did not include two provider bindings".into());
        }
        let provider = "zed.provider.complete";
        let bindings = catalog_descriptors
            .iter()
            .map(|catalog_descriptor| NativeNodeBinding::ProviderRequired {
                feature_id: catalog_descriptor.feature_id.clone(),
                descriptor: RuntimeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: catalog_descriptor.node_identifier.clone(),
                    implementation_version: "provider-v1".to_owned(),
                    source_schema: Some(sim_nodes::NativeDescriptorSchemaMetadata::synthetic(
                        std::iter::empty(),
                        std::iter::empty(),
                        ["value".to_owned()],
                    )),
                    inputs: Vec::new(),
                    dynamic_inputs: Vec::new(),
                    outputs: vec![RuntimeOutputDescriptor {
                        name: "value".to_owned(),
                        produced_type: NativeValueType::Any,
                        is_list: false,
                    }],
                    output_node: catalog_descriptor.output_node,
                    effect: EffectClass::Provider,
                    cache: RuntimeCachePolicy::Never,
                },
                presentation: RuntimeNodePresentation {
                    display_name: catalog_descriptor.display_name.clone(),
                    category: match catalog_descriptor.category.as_str() {
                        "(empty root category declared by source)" => String::new(),
                        category => category.to_owned(),
                    },
                    description: String::new(),
                    output_names: vec!["value".to_owned()],
                    search_aliases: Vec::new(),
                    is_deprecated: false,
                    is_experimental: false,
                },
                provider: provider.to_owned(),
                reason: "verified complete provider activation is required".to_owned(),
            })
            .collect::<Vec<_>>();
        let mut registry = NativeNodeRegistry::default();
        registry.register_native_bindings(bindings.clone())?;
        let transport_schema: sim_plugin_sdk::CanonicalTypeId =
            "zed:sim-provider-transport@1".parse()?;
        let materializer_schema: sim_plugin_sdk::CanonicalTypeId =
            "zed:sim-provider-materializer@1".parse()?;
        let mut claims = Vec::new();
        let mut activations = Vec::new();
        for binding in bindings {
            let class_type = binding.descriptor().class_type.clone();
            let claim = ProviderBindingClaim {
                feature_id: binding.feature_id().to_owned(),
                node_id: class_type.clone(),
                contract_sha256: registry
                    .provider_binding_contract_sha256(
                        &class_type,
                        &transport_schema.to_string(),
                        &materializer_schema.to_string(),
                    )?
                    .ok_or("provider contract was not projected")?,
                transport_schema: transport_schema.clone(),
                materializer_schema: materializer_schema.clone(),
            };
            let node: Arc<dyn NativeNode> = Arc::new(ConfiguredNode {
                class_type,
                version: binding.descriptor().implementation_version.clone(),
                namespace: provider.to_owned(),
            });
            activations.push(NativeProviderBindingActivation::new(claim.clone(), node));
            claims.push(claim);
        }
        claims.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        activations.sort_by(|left, right| left.claim.node_id.cmp(&right.claim.node_id));

        let mut incomplete_set = ProviderBindingSet {
            schema_version: sim_plugin_sdk::PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: provider.to_owned(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![claims[0].clone()],
        };
        incomplete_set.bindings_sha256 = incomplete_set.canonical_bindings_sha256()?;
        let incomplete = NativeProviderBindingActivationSet::checked(
            "profile-a",
            1,
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            incomplete_set,
            vec![activations[0].clone()],
        )?;
        assert!(matches!(
            registry.activate_provider_binding_set(incomplete),
            Err(NativeNodeRegistryError::IncompleteProviderActivation(namespace))
                if namespace == provider
        ));
        for claim in &claims {
            assert_eq!(
                registry.provider_binding_is_activated(&claim.node_id),
                Some(false)
            );
            assert!(registry.node(&claim.node_id).is_none());
        }

        let mut complete_set = ProviderBindingSet {
            schema_version: sim_plugin_sdk::PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: provider.to_owned(),
            bindings_sha256: "0".repeat(64),
            bindings: claims.clone(),
        };
        complete_set.bindings_sha256 = complete_set.canonical_bindings_sha256()?;
        let complete = NativeProviderBindingActivationSet::checked(
            "profile-a",
            1,
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            complete_set,
            activations,
        )?;
        registry.activate_provider_binding_set(complete)?;
        for claim in &claims {
            assert_eq!(
                registry.provider_binding_is_activated(&claim.node_id),
                Some(true)
            );
            assert!(registry.node(&claim.node_id).is_some());
        }
        registry.validate_comprehensive_bindings()?;
        Ok(())
    }

    #[test]
    fn malformed_expansion_rolls_back_published_handles_and_cache_state()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let descriptor = runtime_descriptor(
                "PublishingMalformedExpansion",
                true,
                BTreeMap::new(),
                EffectClass::Pure,
            )?;
            let plan = compile_plan(
                vec![descriptor.clone()],
                BTreeMap::from([(
                    NodeId("expand".to_owned()),
                    PromptNode {
                        class_type: descriptor.class_type.clone(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let mut registry = NativeNodeRegistry::default();
            registry.register_descriptor(descriptor)?;
            registry.register(Arc::new(PublishingMalformedExpansionNode))?;
            let cache = Arc::new(Mutex::new(NativeCache::new(4)?));
            let effects = Arc::new(RecordingEffectCoordinator::default());
            let payload_bytes = stored_test_payload(vec![9])?.resident_bytes()?;
            let generation = NativeHandleStoreGeneration::with_capacities(4, payload_bytes * 4)?;
            let (_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let engine = ExecutionEngine::new_with_handle_store_generation(
                ProfileId(Uuid::from_u128(41)),
                Arc::new(registry),
                cache.clone(),
                effects.clone(),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
                generation.clone(),
            )?;
            let report = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(42)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(report.state, AttemptState::Failed);
            assert!(report.outputs.is_empty());
            assert!(report.ui_outputs.is_empty());
            assert!(generation.is_empty());
            assert!(cache.lock().is_empty());
            assert!(effects.committed().is_empty());
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn native_prepared_effects_roll_back_before_node_failure_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            for (ordinal, class_type) in ["PrepareThenFail", "PrepareThenInvalidOutput"]
                .into_iter()
                .enumerate()
            {
                let descriptor = runtime_descriptor(
                    class_type,
                    true,
                    BTreeMap::new(),
                    EffectClass::WritesArtifact,
                )?;
                let plan = compile_plan(
                    vec![descriptor],
                    BTreeMap::from([(
                        NodeId::from("effect"),
                        PromptNode {
                            class_type: class_type.to_owned(),
                            inputs: BTreeMap::new(),
                            unknown: BTreeMap::new(),
                        },
                    )]),
                )?;
                let mut registry = NativeNodeRegistry::default();
                registry.register(Arc::new(FixtureNode {
                    class_type: class_type.to_owned(),
                    calls: Arc::new(AtomicUsize::new(0)),
                }))?;
                let effects = Arc::new(RecordingEffectCoordinator::default());
                let engine = ExecutionEngine::new_with_workspace_authorization(
                    ProfileId(Uuid::from_u128(71)),
                    Arc::new(registry),
                    Arc::new(Mutex::new(NativeCache::new(4)?)),
                    effects.clone(),
                    "registry-v1",
                    workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
                )?;
                let report = engine
                    .execute(
                        &plan,
                        AttemptId(Uuid::from_u128(72 + u128::try_from(ordinal)?)),
                        CancellationToken::default(),
                    )
                    .await;
                assert_eq!(report.state, AttemptState::Failed);
                assert!(report.outputs.is_empty());
                assert!(effects.prepared().is_empty());
                assert!(effects.committed().is_empty());
                assert!(effects.rolled_back().is_empty());
                assert_eq!(effects.prepared_history(), effects.node_rolled_back());
                assert_eq!(effects.node_rolled_back().len(), 1);
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn cancellation_before_store_commit_rolls_back_handles_and_cache()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let descriptor = publishing_handle_descriptor()?;
            let plan = compile_plan(
                vec![descriptor.clone()],
                BTreeMap::from([(
                    NodeId("publish".to_owned()),
                    PromptNode {
                        class_type: descriptor.class_type.clone(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let calls = Arc::new(AtomicUsize::new(0));
            let mut registry = NativeNodeRegistry::default();
            registry.register_descriptor(descriptor)?;
            registry.register(Arc::new(PublishingHandleNode {
                calls: calls.clone(),
                cancel_after_publish: true,
            }))?;
            let cache = Arc::new(Mutex::new(NativeCache::new(2)?));
            let payload_bytes =
                stored_test_payload(vec![0; mem::size_of::<usize>()])?.resident_bytes()?;
            let generation = NativeHandleStoreGeneration::with_capacities(2, payload_bytes * 2)?;
            let (_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let engine = ExecutionEngine::new_with_handle_store_generation(
                ProfileId(Uuid::from_u128(401)),
                Arc::new(registry),
                cache.clone(),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
                generation.clone(),
            )?;
            let report = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(402)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(report.state, AttemptState::Cancelled);
            assert!(report.outputs.is_empty());
            assert!(generation.is_empty());
            assert_eq!(generation.resident_bytes(), 0);
            assert!(cache.lock().is_empty());
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn cache_and_report_hold_independent_handle_leases() -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let descriptor = publishing_handle_descriptor()?;
            let plan = compile_plan(
                vec![descriptor.clone()],
                BTreeMap::from([(
                    NodeId("publish".to_owned()),
                    PromptNode {
                        class_type: descriptor.class_type.clone(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let calls = Arc::new(AtomicUsize::new(0));
            let mut registry = NativeNodeRegistry::default();
            registry.register_descriptor(descriptor)?;
            registry.register(Arc::new(PublishingHandleNode {
                calls,
                cancel_after_publish: false,
            }))?;
            let cache = Arc::new(Mutex::new(NativeCache::new(2)?));
            let payload_bytes =
                stored_test_payload(vec![0; mem::size_of::<usize>()])?.resident_bytes()?;
            let generation = NativeHandleStoreGeneration::with_capacities(2, payload_bytes * 2)?;
            let (_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let engine = ExecutionEngine::new_with_handle_store_generation(
                ProfileId(Uuid::from_u128(411)),
                Arc::new(registry),
                cache.clone(),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
                generation.clone(),
            )?;
            let report = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(412)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(report.state, AttemptState::Succeeded);
            let handle = report_handle(&report).ok_or("report handle was missing")?;
            assert_eq!(generation.len(), 1);
            assert_eq!(cache.lock().invalidate_node("PublishingHandle"), 1);
            assert_eq!(generation.len(), 1);
            assert!(
                generation
                    .session(AttemptId(Uuid::from_u128(413)))
                    .resolve(&handle, handle.handle_type(), &CancellationToken::default(),)
                    .is_ok()
            );
            drop(report);
            assert!(generation.is_empty());
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn stale_store_generation_cache_entry_is_evicted_and_recomputed()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let descriptor = publishing_handle_descriptor()?;
            let plan = compile_plan(
                vec![descriptor.clone()],
                BTreeMap::from([(
                    NodeId("publish".to_owned()),
                    PromptNode {
                        class_type: descriptor.class_type.clone(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let calls = Arc::new(AtomicUsize::new(0));
            let mut registry = NativeNodeRegistry::default();
            registry.register_descriptor(descriptor)?;
            registry.register(Arc::new(PublishingHandleNode {
                calls: calls.clone(),
                cancel_after_publish: false,
            }))?;
            let registry = Arc::new(registry);
            let cache = Arc::new(Mutex::new(NativeCache::new(2)?));
            let payload_bytes =
                stored_test_payload(vec![0; mem::size_of::<usize>()])?.resident_bytes()?;
            let first_generation =
                NativeHandleStoreGeneration::with_capacities(2, payload_bytes * 2)?;
            let second_generation =
                NativeHandleStoreGeneration::with_capacities(2, payload_bytes * 2)?;
            let (_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES * 2)?;
            let first_engine = ExecutionEngine::new_with_handle_store_generation(
                ProfileId(Uuid::from_u128(421)),
                registry.clone(),
                cache.clone(),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
                first_generation.clone(),
            )?;
            let first_report = first_engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(422)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(first_report.state, AttemptState::Succeeded);
            drop(first_report);
            assert_eq!(first_generation.len(), 1);

            let second_engine = ExecutionEngine::new_with_handle_store_generation(
                ProfileId(Uuid::from_u128(423)),
                registry,
                cache.clone(),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
                second_generation.clone(),
            )?;
            let second_report = second_engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(424)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(second_report.state, AttemptState::Succeeded);
            assert_eq!(second_report.cache_hits, 0);
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert!(first_generation.is_empty());
            assert_eq!(second_generation.len(), 1);
            drop(second_report);
            assert_eq!(second_generation.len(), 1);
            cache.lock().clear();
            assert!(second_generation.is_empty());
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    fn val_domain_004_native_handle_lifecycle_is_atomic() -> Result<(), Box<dyn std::error::Error>>
    {
        native_handle_store_sessions_isolate_stage_commit_and_revoke()?;
        native_handle_store_abandoned_session_rolls_back_staged_values()?;
        native_handle_store_duplicate_attempt_reuses_one_session_and_close_state()?;
        native_executor_rejects_concurrent_duplicate_attempt_before_store_mutation()?;
        native_handle_resolve_cancellation_after_root_increment_is_atomic()?;
        native_handle_root_overflow_rejection_is_atomic()?;
        native_handle_store_clip_vision_payloads_enforce_identity_and_alias_residency()?;
        native_handle_store_capacity_and_lease_validation_are_atomic()?;
        native_handle_store_deduplicates_shared_allocations_until_final_owner()?;
        resolved_payload_guard_retires_logically_and_releases_shared_capacity_on_final_drop()?;
        native_handle_store_allocation_capacity_rejection_is_atomic()?;
        native_handle_store_accepts_zero_byte_payloads_and_never_wraps_generation()?;
        native_handle_publish_cancellation_is_atomic_before_validation_and_after_insert()?;
        native_handle_publish_racing_session_close_never_leaks_staged_values()?;
        native_handle_resolve_rejects_forged_store_generation_type_and_digest()?;
        native_cache_leases_cover_aliases_and_release_on_lru_and_invalidation()?;
        cache_eviction_retires_handles_but_resolved_guard_retains_shared_bytes()?;
        native_cache_same_key_replacement_transfers_exact_handle_roots()?;
        cancelled_cache_publication_restores_replaced_entry_and_exact_roots()?;
        cloned_native_handle_lease_releases_roots_once_under_concurrent_drop()?;
        malformed_expansion_rolls_back_published_handles_and_cache_state()?;
        cancellation_before_store_commit_rolls_back_handles_and_cache()?;
        cache_and_report_hold_independent_handle_leases()?;
        stale_store_generation_cache_entry_is_evicted_and_recomputed()?;
        Ok(())
    }

    fn compile_plan(
        descriptors: Vec<RuntimeNodeDescriptor>,
        nodes: BTreeMap<NodeId, PromptNode>,
    ) -> Result<CompiledPlan, Box<dyn std::error::Error>> {
        let mut registry = NativeNodeRegistry::default();
        for descriptor in descriptors {
            registry.register_descriptor(descriptor)?;
        }
        Ok(PromptCompiler::new(&registry).compile(PromptSubmission {
            prompt: ApiPrompt(nodes),
            prompt_id: Some(PromptId(Uuid::from_u128(1))),
            client_id: None,
            number: None,
            extra_data: BTreeMap::new(),
            unknown: BTreeMap::new(),
        })?)
    }

    #[test]
    fn native_executor_forwards_planned_workspace_without_reauthorizing()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let descriptor = runtime_descriptor(
                "WorkspaceRecording",
                true,
                BTreeMap::new(),
                EffectClass::Pure,
            )?;
            let plan = compile_plan(
                vec![descriptor.clone()],
                BTreeMap::from([(
                    NodeId::from("workspace"),
                    PromptNode {
                        class_type: "WorkspaceRecording".to_owned(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let observed = Arc::new(Mutex::new(Vec::new()));
            let mut registry = NativeNodeRegistry::default();
            registry.register_descriptor(descriptor)?;
            registry.register(Arc::new(WorkspaceRecordingNode {
                observed: observed.clone(),
            }))?;
            let (_backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024)?;
            let workspace = workspace_authority.authorize_workspace(321)?;
            let engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(registry),
                Arc::new(Mutex::new(NativeCache::new(4)?)),
                Arc::new(RecordingEffectCoordinator::default()),
                "workspace-registry-v1",
                workspace,
            )?;
            let report = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(321)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(report.state, AttemptState::Succeeded);
            assert_eq!(&*observed.lock(), &[321, 321]);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn val_domain_004_graph_list_lazy_async_cache_and_effects_execute_transactionally()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_workspace_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let descriptors = vec![
                runtime_descriptor("Source", false, BTreeMap::new(), EffectClass::Pure)?,
                runtime_descriptor(
                    "Double",
                    false,
                    BTreeMap::from([(
                        "value".to_owned(),
                        input(ValueType::Number, false, InputMode::Mapped, false),
                    )]),
                    EffectClass::Pure,
                )?,
                runtime_descriptor(
                    "Output",
                    true,
                    BTreeMap::from([(
                        "value".to_owned(),
                        input(ValueType::Number, false, InputMode::List, false),
                    )]),
                    EffectClass::Pure,
                )?,
            ];
            let plan = compile_plan(
                descriptors,
                BTreeMap::from([
                    (
                        NodeId::from("source"),
                        PromptNode {
                            class_type: "Source".to_owned(),
                            inputs: BTreeMap::new(),
                            unknown: BTreeMap::new(),
                        },
                    ),
                    (
                        NodeId::from("double"),
                        PromptNode {
                            class_type: "Double".to_owned(),
                            inputs: BTreeMap::from([("value".to_owned(), json!(["source", 0]))]),
                            unknown: BTreeMap::new(),
                        },
                    ),
                    (
                        NodeId::from("output"),
                        PromptNode {
                            class_type: "Output".to_owned(),
                            inputs: BTreeMap::from([("value".to_owned(), json!(["double", 0]))]),
                            unknown: BTreeMap::new(),
                        },
                    ),
                ]),
            )?;
            let calls = Arc::new(AtomicUsize::new(0));
            let mut nodes = NativeNodeRegistry::default();
            for class_type in ["Source", "Double", "Output"] {
                nodes.register(Arc::new(FixtureNode {
                    class_type: class_type.to_owned(),
                    calls: calls.clone(),
                }))?;
            }
            let effects = Arc::new(RecordingEffectCoordinator::default());
            let engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(nodes),
                Arc::new(Mutex::new(NativeCache::new(32)?)),
                effects,
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let first = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(2)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(first.state, AttemptState::Succeeded);
            assert_eq!(
                first.outputs[&NodeId::from("output")],
                [NativeValue::List {
                    values: vec![native_integer(2), native_integer(4), native_integer(6)],
                }]
            );
            let first_calls = calls.load(Ordering::SeqCst);
            let second = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(3)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(second.state, AttemptState::Succeeded);
            assert!(second.cache_hits > 0);
            assert_eq!(calls.load(Ordering::SeqCst), first_calls);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn val_domain_004_repeat_last_and_output_list_flattening_are_exact()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_workspace_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let pair = runtime_descriptor(
                "Pair",
                true,
                BTreeMap::from([
                    (
                        "left".to_owned(),
                        input(ValueType::Integer, false, InputMode::Mapped, true),
                    ),
                    (
                        "right".to_owned(),
                        input(ValueType::Integer, false, InputMode::Mapped, true),
                    ),
                ]),
                EffectClass::Pure,
            )?;
            let mut list_map = runtime_descriptor(
                "ListMap",
                true,
                BTreeMap::from([(
                    "value".to_owned(),
                    input(ValueType::Integer, false, InputMode::Mapped, true),
                )]),
                EffectClass::Pure,
            )?;
            list_map.outputs[0].is_list = true;
            let plan = compile_plan(
                vec![pair, list_map],
                BTreeMap::from([
                    (
                        NodeId::from("pair"),
                        PromptNode {
                            class_type: "Pair".to_owned(),
                            inputs: BTreeMap::from([
                                ("left".to_owned(), json!([1, 2, 3])),
                                ("right".to_owned(), json!([4, 5])),
                            ]),
                            unknown: BTreeMap::new(),
                        },
                    ),
                    (
                        NodeId::from("list"),
                        PromptNode {
                            class_type: "ListMap".to_owned(),
                            inputs: BTreeMap::from([("value".to_owned(), json!([1, 2]))]),
                            unknown: BTreeMap::new(),
                        },
                    ),
                ]),
            )?;
            let calls = Arc::new(AtomicUsize::new(0));
            let mut registry = NativeNodeRegistry::default();
            for class_type in ["Pair", "ListMap"] {
                registry.register(Arc::new(FixtureNode {
                    class_type: class_type.to_owned(),
                    calls: calls.clone(),
                }))?;
            }
            let engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(registry),
                Arc::new(Mutex::new(NativeCache::new(8)?)),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let report = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(20)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(report.state, AttemptState::Succeeded);
            assert_eq!(
                report.outputs[&NodeId::from("pair")],
                [NativeValue::List {
                    values: vec![native_integer(14), native_integer(25), native_integer(35)],
                }]
            );
            assert_eq!(
                report.outputs[&NodeId::from("list")],
                [NativeValue::List {
                    values: vec![native_integer(1), native_integer(2)],
                }]
            );
            assert_eq!(calls.load(Ordering::SeqCst), 5);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn val_domain_004_canonical_cancellation_and_ui_cache_semantics()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_workspace_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let cancel_plan = compile_plan(
                vec![runtime_descriptor(
                    "CancelDuringExecution",
                    true,
                    BTreeMap::new(),
                    EffectClass::Pure,
                )?],
                BTreeMap::from([(
                    NodeId::from("cancel"),
                    PromptNode {
                        class_type: "CancelDuringExecution".to_owned(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let mut cancel_registry = NativeNodeRegistry::default();
            cancel_registry.register(Arc::new(CancellingNode))?;
            let cancel_engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(cancel_registry),
                Arc::new(Mutex::new(NativeCache::new(4)?)),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let cancelled = cancel_engine
                .execute(
                    &cancel_plan,
                    AttemptId(Uuid::from_u128(21)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(cancelled.state, AttemptState::Cancelled);
            assert!(cancelled.outputs.is_empty());
            assert!(cancelled.ui_outputs.is_empty());

            let before_cache_plan = compile_plan(
                vec![runtime_descriptor(
                    "CancelBeforeCache",
                    true,
                    BTreeMap::new(),
                    EffectClass::WritesArtifact,
                )?],
                BTreeMap::from([(
                    NodeId::from("before-cache"),
                    PromptNode {
                        class_type: "CancelBeforeCache".to_owned(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let phases = Arc::new(Mutex::new(Vec::new()));
            let mut before_cache_registry = NativeNodeRegistry::default();
            before_cache_registry.register(Arc::new(CancelBeforeCacheNode {
                phases: phases.clone(),
            }))?;
            let before_cache = Arc::new(Mutex::new(NativeCache::new(4)?));
            let before_cache_effects = Arc::new(RecordingEffectCoordinator::default());
            let before_cache_engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(before_cache_registry),
                before_cache.clone(),
                before_cache_effects.clone(),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;

            let pre_cancelled = CancellationToken::default();
            pre_cancelled.cancel();
            let report = before_cache_engine
                .execute(
                    &before_cache_plan,
                    AttemptId(Uuid::from_u128(0x2101)),
                    pre_cancelled,
                )
                .await;
            assert_eq!(report.state, AttemptState::Cancelled);
            assert!(phases.lock().is_empty());
            assert_eq!(report.cache_hits, 0);
            assert!(before_cache.lock().is_empty());
            assert!(before_cache_effects.prepared().is_empty());
            assert!(before_cache_effects.committed().is_empty());
            assert!(before_cache_effects.rolled_back().is_empty());
            assert!(report.outputs.is_empty());
            assert!(report.ui_outputs.is_empty());
            assert_eq!(report.events.len(), 2);
            assert!(matches!(report.events[0].kind, AttemptEventKind::Started));
            assert!(matches!(report.events[1].kind, AttemptEventKind::Cancelled));

            let report = before_cache_engine
                .execute(
                    &before_cache_plan,
                    AttemptId(Uuid::from_u128(0x2102)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(report.state, AttemptState::Cancelled);
            assert_eq!(phases.lock().as_slice(), &["demand"]);
            assert_eq!(report.cache_hits, 0);
            assert!(before_cache.lock().is_empty());
            assert!(before_cache_effects.prepared().is_empty());
            assert!(before_cache_effects.committed().is_empty());
            assert!(before_cache_effects.rolled_back().is_empty());
            assert!(report.outputs.is_empty());
            assert!(report.ui_outputs.is_empty());
            assert_eq!(report.events.len(), 2);
            assert!(matches!(report.events[0].kind, AttemptEventKind::Started));
            assert!(matches!(report.events[1].kind, AttemptEventKind::Cancelled));

            let ui_plan = compile_plan(
                vec![runtime_descriptor(
                    "Ui",
                    true,
                    BTreeMap::new(),
                    EffectClass::Pure,
                )?],
                BTreeMap::from([(
                    NodeId::from("ui"),
                    PromptNode {
                        class_type: "Ui".to_owned(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let ui_calls = Arc::new(AtomicUsize::new(0));
            let mut ui_registry = NativeNodeRegistry::default();
            ui_registry.register(Arc::new(UiNode {
                calls: ui_calls.clone(),
            }))?;
            let ui_engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(ui_registry),
                Arc::new(Mutex::new(NativeCache::new(4)?)),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            for attempt in [22_u128, 23] {
                let report = ui_engine
                    .execute(
                        &ui_plan,
                        AttemptId(Uuid::from_u128(attempt)),
                        CancellationToken::default(),
                    )
                    .await;
                assert_eq!(report.state, AttemptState::Succeeded);
                assert_eq!(
                    report.ui_outputs[&NodeId::from("ui")],
                    json!({"preview": "ready"})
                );
            }
            assert_eq!(ui_calls.load(Ordering::SeqCst), 1);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn val_domain_004_cache_tracks_demanded_dependency_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_workspace_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let compile = |source_version: &str| {
                let mut source = runtime_descriptor(
                    "VersionedSource",
                    false,
                    BTreeMap::new(),
                    EffectClass::Pure,
                )?;
                source.implementation_version = source_version.to_owned();
                compile_plan(
                    vec![
                        source,
                        runtime_descriptor(
                            "Passthrough",
                            true,
                            BTreeMap::from([(
                                "value".to_owned(),
                                input(ValueType::Number, false, InputMode::Scalar, false),
                            )]),
                            EffectClass::Pure,
                        )?,
                    ],
                    BTreeMap::from([
                        (
                            NodeId::from("source"),
                            PromptNode {
                                class_type: "VersionedSource".to_owned(),
                                inputs: BTreeMap::new(),
                                unknown: BTreeMap::new(),
                            },
                        ),
                        (
                            NodeId::from("output"),
                            PromptNode {
                                class_type: "Passthrough".to_owned(),
                                inputs: BTreeMap::from([(
                                    "value".to_owned(),
                                    json!(["source", 0]),
                                )]),
                                unknown: BTreeMap::new(),
                            },
                        ),
                    ]),
                )
            };
            let first_plan = compile("1")?;
            let second_plan = compile("2")?;
            let passthrough_calls = Arc::new(AtomicUsize::new(0));
            let cache = Arc::new(Mutex::new(NativeCache::new(16)?));
            for (attempt, version, plan) in
                [(30_u128, "1", &first_plan), (31_u128, "2", &second_plan)]
            {
                let mut registry = NativeNodeRegistry::default();
                registry.register(Arc::new(VersionedSourceNode { version }))?;
                registry.register(Arc::new(PassthroughNode {
                    calls: passthrough_calls.clone(),
                }))?;
                let engine = ExecutionEngine::new_with_workspace_authorization(
                    ProfileId(Uuid::nil()),
                    Arc::new(registry),
                    cache.clone(),
                    Arc::new(RecordingEffectCoordinator::default()),
                    "registry-v1",
                    workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
                )?;
                let report = engine
                    .execute(
                        plan,
                        AttemptId(Uuid::from_u128(attempt)),
                        CancellationToken::default(),
                    )
                    .await;
                assert_eq!(report.state, AttemptState::Succeeded);
                assert_eq!(report.outputs[&NodeId::from("output")], [native_integer(7)]);
            }
            assert_eq!(passthrough_calls.load(Ordering::SeqCst), 2);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn val_domain_004_cancellation_and_blockers_publish_no_partial_outputs()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_workspace_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let plan = compile_plan(
                vec![runtime_descriptor(
                    "Block",
                    true,
                    BTreeMap::new(),
                    EffectClass::Pure,
                )?],
                BTreeMap::from([(
                    NodeId::from("block"),
                    PromptNode {
                        class_type: "Block".to_owned(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let mut nodes = NativeNodeRegistry::default();
            nodes.register(Arc::new(FixtureNode {
                class_type: "Block".to_owned(),
                calls: Arc::new(AtomicUsize::new(0)),
            }))?;
            let engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(nodes),
                Arc::new(Mutex::new(NativeCache::new(4)?)),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let blocked = engine
                .execute(
                    &plan,
                    AttemptId(Uuid::from_u128(2)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(blocked.state, AttemptState::Failed);
            assert!(blocked.outputs.is_empty());
            let cancellation = CancellationToken::default();
            cancellation.cancel();
            let cancelled = engine
                .execute(&plan, AttemptId(Uuid::from_u128(3)), cancellation)
                .await;
            assert_eq!(cancelled.state, AttemptState::Cancelled);
            assert!(cancelled.outputs.is_empty());
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn val_domain_004_lazy_dependencies_execute_only_when_demanded()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_workspace_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let descriptors = vec![
                runtime_descriptor("LazySource", false, BTreeMap::new(), EffectClass::Pure)?,
                runtime_descriptor(
                    "Choose",
                    true,
                    BTreeMap::from([
                        (
                            "condition".to_owned(),
                            input(ValueType::Boolean, false, InputMode::Scalar, true),
                        ),
                        (
                            "value".to_owned(),
                            input(ValueType::Number, true, InputMode::Scalar, false),
                        ),
                    ]),
                    EffectClass::Pure,
                )?,
            ];
            let nodes = |condition| {
                BTreeMap::from([
                    (
                        NodeId::from("source"),
                        PromptNode {
                            class_type: "LazySource".to_owned(),
                            inputs: BTreeMap::new(),
                            unknown: BTreeMap::new(),
                        },
                    ),
                    (
                        NodeId::from("choose"),
                        PromptNode {
                            class_type: "Choose".to_owned(),
                            inputs: BTreeMap::from([
                                ("condition".to_owned(), json!(condition)),
                                ("value".to_owned(), json!(["source", 0])),
                            ]),
                            unknown: BTreeMap::new(),
                        },
                    ),
                ])
            };
            let false_plan = compile_plan(descriptors.clone(), nodes(false))?;
            let true_plan = compile_plan(descriptors, nodes(true))?;
            let source_calls = Arc::new(AtomicUsize::new(0));
            let mut registry = NativeNodeRegistry::default();
            registry.register(Arc::new(FixtureNode {
                class_type: "LazySource".to_owned(),
                calls: source_calls.clone(),
            }))?;
            registry.register(Arc::new(FixtureNode {
                class_type: "Choose".to_owned(),
                calls: Arc::new(AtomicUsize::new(0)),
            }))?;
            let engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(registry),
                Arc::new(Mutex::new(NativeCache::new(8)?)),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let skipped = engine
                .execute(
                    &false_plan,
                    AttemptId(Uuid::from_u128(2)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(skipped.state, AttemptState::Succeeded);
            assert_eq!(source_calls.load(Ordering::SeqCst), 0);
            let demanded = engine
                .execute(
                    &true_plan,
                    AttemptId(Uuid::from_u128(3)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(demanded.state, AttemptState::Succeeded);
            assert_eq!(
                demanded.outputs[&NodeId::from("choose")],
                [native_integer(42)]
            );
            assert_eq!(source_calls.load(Ordering::SeqCst), 1);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    #[test]
    fn val_domain_004_expansion_and_transactional_effect_batches_are_real_execution_paths()
    -> Result<(), Box<dyn std::error::Error>> {
        smol::block_on(async {
            let (_workspace_backend, workspace_authority) =
                CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
            let inner = ApiPrompt(BTreeMap::from([(
                NodeId::from("inner"),
                PromptNode {
                    class_type: "InnerOutput".to_owned(),
                    inputs: BTreeMap::new(),
                    unknown: BTreeMap::new(),
                },
            )]));
            let outer = compile_plan(
                vec![runtime_descriptor(
                    "Expand",
                    true,
                    BTreeMap::from([(
                        "value".to_owned(),
                        input(ValueType::Integer, false, InputMode::Mapped, true),
                    )]),
                    EffectClass::Pure,
                )?],
                BTreeMap::from([(
                    NodeId::from("expand"),
                    PromptNode {
                        class_type: "Expand".to_owned(),
                        inputs: BTreeMap::from([("value".to_owned(), json!([1, 2]))]),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let mut expansion_registry = NativeNodeRegistry::default();
            expansion_registry.register_descriptor(runtime_descriptor(
                "InnerOutput",
                true,
                BTreeMap::new(),
                EffectClass::Pure,
            )?)?;
            expansion_registry.register(Arc::new(ExpansionNode { prompt: inner }))?;
            expansion_registry.register(Arc::new(FixtureNode {
                class_type: "InnerOutput".to_owned(),
                calls: Arc::new(AtomicUsize::new(0)),
            }))?;
            let expansion_engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(expansion_registry),
                Arc::new(Mutex::new(NativeCache::new(8)?)),
                Arc::new(RecordingEffectCoordinator::default()),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let expanded = expansion_engine
                .execute(
                    &outer,
                    AttemptId(Uuid::from_u128(2)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(expanded.state, AttemptState::Succeeded);
            assert_eq!(
                expanded.outputs[&NodeId::from("expand")],
                [NativeValue::List {
                    values: vec![native_integer(42), native_integer(42)],
                }]
            );
            assert_eq!(
                expanded
                    .outputs
                    .keys()
                    .filter(|node_id| node_id.0.starts_with("expand::expansion-0-"))
                    .count(),
                2
            );

            let write = compile_plan(
                vec![runtime_descriptor(
                    "Write",
                    true,
                    BTreeMap::new(),
                    EffectClass::WritesArtifact,
                )?],
                BTreeMap::from([(
                    NodeId::from("write"),
                    PromptNode {
                        class_type: "Write".to_owned(),
                        inputs: BTreeMap::new(),
                        unknown: BTreeMap::new(),
                    },
                )]),
            )?;
            let mut write_registry = NativeNodeRegistry::default();
            write_registry.register(Arc::new(FixtureNode {
                class_type: "Write".to_owned(),
                calls: Arc::new(AtomicUsize::new(0)),
            }))?;
            let effects = Arc::new(RecordingEffectCoordinator::default());
            let write_engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(write_registry),
                Arc::new(Mutex::new(NativeCache::new(8)?)),
                effects.clone(),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let committed = write_engine
                .execute(
                    &write,
                    AttemptId(Uuid::from_u128(3)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(committed.state, AttemptState::Succeeded);
            assert_eq!(effects.committed(), effects.prepared());
            assert_eq!(effects.committed().len(), 1);
            assert!(effects.rolled_back().is_empty());

            let failed_write = compile_plan(
                vec![
                    runtime_descriptor(
                        "Write",
                        false,
                        BTreeMap::new(),
                        EffectClass::WritesArtifact,
                    )?,
                    runtime_descriptor(
                        "Block",
                        true,
                        BTreeMap::from([(
                            "value".to_owned(),
                            input(ValueType::Any, false, InputMode::Scalar, false),
                        )]),
                        EffectClass::Pure,
                    )?,
                ],
                BTreeMap::from([
                    (
                        NodeId::from("write"),
                        PromptNode {
                            class_type: "Write".to_owned(),
                            inputs: BTreeMap::new(),
                            unknown: BTreeMap::new(),
                        },
                    ),
                    (
                        NodeId::from("block"),
                        PromptNode {
                            class_type: "Block".to_owned(),
                            inputs: BTreeMap::from([("value".to_owned(), json!(["write", 0]))]),
                            unknown: BTreeMap::new(),
                        },
                    ),
                ]),
            )?;
            let mut failed_registry = NativeNodeRegistry::default();
            for class_type in ["Write", "Block"] {
                failed_registry.register(Arc::new(FixtureNode {
                    class_type: class_type.to_owned(),
                    calls: Arc::new(AtomicUsize::new(0)),
                }))?;
            }
            let failed_effects = Arc::new(RecordingEffectCoordinator::default());
            let failed_engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(failed_registry),
                Arc::new(Mutex::new(NativeCache::new(8)?)),
                failed_effects.clone(),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?;
            let failed = failed_engine
                .execute(
                    &failed_write,
                    AttemptId(Uuid::from_u128(4)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(failed.state, AttemptState::Failed);
            assert!(failed.outputs.is_empty());
            assert!(failed_effects.committed().is_empty());
            assert_eq!(failed_effects.rolled_back(), failed_effects.prepared());
            assert_eq!(failed_effects.rolled_back().len(), 1);

            let mut event_failure_registry = NativeNodeRegistry::default();
            event_failure_registry.register(Arc::new(FixtureNode {
                class_type: "Write".to_owned(),
                calls: Arc::new(AtomicUsize::new(0)),
            }))?;
            let event_failure_effects = Arc::new(RecordingEffectCoordinator::default());
            let event_bus = ExecutionEventBus::new(1)?;
            let _receiver = event_bus.subscribe();
            let event_failure_engine = ExecutionEngine::new_with_workspace_authorization(
                ProfileId(Uuid::nil()),
                Arc::new(event_failure_registry),
                Arc::new(Mutex::new(NativeCache::new(8)?)),
                event_failure_effects.clone(),
                "registry-v1",
                workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
            )?
            .with_event_bus(event_bus);
            let event_failure = event_failure_engine
                .execute(
                    &write,
                    AttemptId(Uuid::from_u128(5)),
                    CancellationToken::default(),
                )
                .await;
            assert_eq!(event_failure.state, AttemptState::Failed);
            assert!(event_failure.outputs.is_empty());
            assert!(event_failure_effects.committed().is_empty());
            assert_eq!(
                event_failure_effects.rolled_back(),
                event_failure_effects.prepared()
            );
            assert_eq!(event_failure_effects.rolled_back().len(), 1);
            Ok::<(), Box<dyn std::error::Error>>(())
        })
    }

    pub(crate) fn val_domain_004_executor_case_results()
    -> Result<Vec<(&'static str, bool)>, Box<dyn std::error::Error>> {
        val_domain_004_graph_list_lazy_async_cache_and_effects_execute_transactionally()?;
        val_domain_004_repeat_last_and_output_list_flattening_are_exact()?;
        val_domain_004_canonical_cancellation_and_ui_cache_semantics()?;
        val_domain_004_cache_tracks_demanded_dependency_identity()?;
        val_domain_004_cancellation_and_blockers_publish_no_partial_outputs()?;
        val_domain_004_lazy_dependencies_execute_only_when_demanded()?;
        val_domain_004_expansion_and_transactional_effect_batches_are_real_execution_paths()?;
        val_domain_004_native_handle_lifecycle_is_atomic()?;
        Ok(vec![
            ("executor_graph_list_async_cache_effects", true),
            ("executor_repeat_last_output_list", true),
            ("executor_canonical_cancel_ui_cache", true),
            ("executor_demanded_dependency_cache_identity", true),
            ("executor_blocker_output_fence", true),
            ("executor_lazy_demand", true),
            ("executor_expansion_effect_atomicity", true),
            ("native_handle_session_commit_rollback_revoke", true),
            ("native_handle_abandoned_session_rollback", true),
            ("native_handle_duplicate_attempt_single_session", true),
            ("native_handle_concurrent_attempt_claim", true),
            ("native_handle_resolve_cancellation_atomicity", true),
            ("native_handle_root_overflow_atomicity", true),
            ("native_handle_clip_vision_payload_lifecycle", true),
            ("native_handle_capacity_lease_atomicity", true),
            ("native_handle_shared_allocation_dedup", true),
            ("native_handle_resolved_guard_retirement", true),
            ("native_handle_allocation_capacity_atomicity", true),
            ("native_handle_zero_byte_identifier_exhaustion", true),
            ("native_handle_publish_cancellation_atomicity", true),
            ("native_handle_publish_session_close_race", true),
            ("native_handle_forged_identity_type_digest_rejection", true),
            ("native_handle_cache_alias_lru_invalidation_leases", true),
            ("native_handle_cache_eviction_resolved_guard", true),
            ("native_handle_cache_same_key_replacement", true),
            ("native_handle_cancelled_cache_publication_rollback", true),
            ("native_handle_cloned_lease_final_drop", true),
            ("native_handle_failed_expansion_rollback", true),
            ("native_handle_precommit_cancellation_rollback", true),
            ("native_handle_cache_report_independent_leases", true),
            ("native_handle_restart_generation_recompute", true),
        ])
    }

    #[test]
    fn native_compile_policy_seals_invocation_and_canonical_cache_dimensions()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_workspace_backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(TEST_WORKSPACE_BYTES)?;
        assert!(!compiler_is_compiling_exact_native(
            NativeCompilePhase::Eager,
            &CancellationToken::default(),
        )?);
        assert!(compiler_is_compiling_exact_native(
            NativeCompilePhase::CapturingGraph,
            &CancellationToken::default(),
        )?);
        let policy = NativeCompilePolicy::from_source_configuration(
            "inductor",
            BTreeMap::from([(
                "guard_filter_fn".to_owned(),
                "skip_torch_compile_dict".to_owned(),
            )]),
            Some("default".to_owned()),
            true,
            Some(false),
        )?;
        let same = NativeCompilePolicy::from_source_configuration(
            "inductor",
            BTreeMap::from([(
                "guard_filter_fn".to_owned(),
                "skip_torch_compile_dict".to_owned(),
            )]),
            Some("default".to_owned()),
            true,
            Some(false),
        )?;
        assert_eq!(
            policy.guard_policy(),
            NativeCompileGuardPolicy::SkipTransformerOptionsDictionary
        );
        assert_eq!(policy.cache_token()?, same.cache_token()?);
        let compiled = compile_exact_native(
            41_u32,
            policy.clone(),
            None,
            &BTreeMap::new(),
            &CancellationToken::default(),
        )?;
        assert_eq!(
            compiled.invoke(&CancellationToken::default(), |model| model + 1)?,
            42
        );
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(compiled.invoke(&cancelled, |model| *model).is_err());
        assert!(matches!(
            compile_exact_native(
                41_u32,
                NativeCompilePolicy::from_source_configuration(
                    "cudagraphs",
                    BTreeMap::new(),
                    None,
                    false,
                    None,
                )?,
                None,
                &BTreeMap::new(),
                &cancelled,
            ),
            Err(NativeCompileError::Cancelled)
        ));

        let registry = Arc::new(NativeNodeRegistry::default());
        let engine = ExecutionEngine::new_with_workspace_authorization(
            ProfileId(Uuid::nil()),
            registry,
            Arc::new(Mutex::new(NativeCache::new(1)?)),
            Arc::new(RecordingEffectCoordinator::default()),
            "registry-v1",
            workspace_authority.authorize_workspace(TEST_WORKSPACE_BYTES)?,
        )?
        .with_native_compile_policy(&policy)?;
        assert_eq!(engine.backend, "native-graph");
        assert_eq!(engine.configuration_token, policy.cache_token()?);
        assert!(
            NativeCompilePolicy::from_source_configuration(
                "python-dynamo",
                BTreeMap::new(),
                None,
                false,
                None,
            )
            .is_err()
        );
        assert!(matches!(
            NativeCompilePolicy::from_source_configuration(
                "inductor",
                BTreeMap::from([("epilogue_fusion".to_owned(), "true".to_owned())]),
                None,
                false,
                None,
            ),
            Err(NativeCompileError::UnsupportedOption { .. })
        ));
        let cuda_graphs = NativeCompilePolicy::from_source_configuration(
            "cudagraphs",
            BTreeMap::new(),
            None,
            false,
            None,
        )?;
        assert!(matches!(
            compile_exact_native(
                41_u32,
                cuda_graphs.clone(),
                None,
                &BTreeMap::new(),
                &CancellationToken::default()
            ),
            Err(NativeCompileError::UncertifiedCudaGraphs)
        ));
        let cpu_capabilities = BackendCapabilityMatrix::new_with_properties(
            DeviceId::CPU,
            Vec::new(),
            Vec::new(),
            Some(NativeDeviceProperties::new(
                DeviceId::CPU,
                "CPU",
                1,
                0,
                0,
                None,
                true,
            )?),
        )?;
        assert!(matches!(
            compile_exact_native(
                41_u32,
                cuda_graphs.clone(),
                Some(&cpu_capabilities),
                &BTreeMap::new(),
                &CancellationToken::default()
            ),
            Err(NativeCompileError::UncertifiedCudaGraphs)
        ));
        let cuda_device = DeviceId::new(DeviceKind::Cuda, 0);
        let cuda_capabilities = BackendCapabilityMatrix::new_with_properties(
            cuda_device,
            Vec::new(),
            Vec::new(),
            Some(NativeDeviceProperties::new(
                cuda_device,
                "Certified fixture CUDA",
                1,
                9,
                0,
                Some("sm_90".to_owned()),
                true,
            )?),
        )?;
        compile_exact_native(
            41_u32,
            cuda_graphs,
            Some(&cuda_capabilities),
            &BTreeMap::new(),
            &CancellationToken::default(),
        )?;
        Ok(())
    }

    #[test]
    fn task408_compile_policy_seals_source_backends_and_enforces_guard_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        let skip_policy = NativeCompilePolicy::from_source_configuration(
            "inductor",
            BTreeMap::from([(
                "guard_filter_fn".to_owned(),
                "skip_torch_compile_dict".to_owned(),
            )]),
            None,
            false,
            None,
        )?;
        let exact_policy = NativeCompilePolicy::from_source_configuration(
            "inductor",
            BTreeMap::new(),
            None,
            false,
            None,
        )?;
        let first = BTreeMap::from([
            ("model".to_owned(), "model-v1".to_owned()),
            ("transformer_options".to_owned(), "dictionary-v1".to_owned()),
        ]);
        let second = BTreeMap::from([
            ("model".to_owned(), "model-v1".to_owned()),
            ("transformer_options".to_owned(), "dictionary-v2".to_owned()),
        ]);
        assert_eq!(
            skip_policy.invocation_guard_token(&first)?,
            skip_policy.invocation_guard_token(&second)?
        );
        assert_ne!(
            exact_policy.invocation_guard_token(&first)?,
            exact_policy.invocation_guard_token(&second)?
        );
        let changed_model = BTreeMap::from([
            ("model".to_owned(), "model-v2".to_owned()),
            ("transformer_options".to_owned(), "dictionary-v1".to_owned()),
        ]);
        assert_ne!(
            skip_policy.invocation_guard_token(&first)?,
            skip_policy.invocation_guard_token(&changed_model)?
        );

        let cuda_policy = NativeCompilePolicy::from_source_configuration(
            "cudagraphs",
            BTreeMap::from([(
                "guard_filter_fn".to_owned(),
                "skip_torch_compile_dict".to_owned(),
            )]),
            None,
            false,
            None,
        )?;
        assert_ne!(skip_policy.cache_token()?, cuda_policy.cache_token()?);
        let cuda_device = DeviceId::new(DeviceKind::Cuda, 0);
        let capabilities = BackendCapabilityMatrix::new_with_properties(
            cuda_device,
            Vec::new(),
            Vec::new(),
            Some(NativeDeviceProperties::new(
                cuda_device,
                "Task408 certified CUDA fixture",
                1,
                9,
                0,
                Some("sm_90".to_owned()),
                true,
            )?),
        )?;
        let compiled = compile_exact_native(
            408_u64,
            cuda_policy.clone(),
            Some(&capabilities),
            &first,
            &CancellationToken::default(),
        )?;
        assert_eq!(compiled.policy(), &cuda_policy);
        assert_eq!(
            compiled.invoke_with_typed_inputs(
                &CancellationToken::default(),
                &first,
                |model| model + 1,
            )?,
            409
        );
        assert!(matches!(
            compiled.invoke_with_typed_inputs(
                &CancellationToken::default(),
                &changed_model,
                |model| model + 1,
            ),
            Err(NativeCompileError::InvocationGuardMismatch)
        ));
        assert!(matches!(
            NativeCompilePolicy::from_source_configuration(
                "inductor",
                BTreeMap::from([(
                    "guard_filter_fn".to_owned(),
                    "accepted-but-ignored".to_owned()
                )]),
                None,
                false,
                None,
            ),
            Err(NativeCompileError::UnsupportedOption { .. })
        ));
        Ok(())
    }

    #[test]
    fn typed_outputs_accept_source_null_without_weakening_inputs()
    -> Result<(), Box<dyn std::error::Error>> {
        let audio = sim_nodes::NativeHandleType::new(NativeHandleKind::Audio, "AUDIO")?;
        let expected = NativeValueType::Handle(audio.clone());
        let null = native_null();
        assert!(native_output_type_accepts(&expected, &null));
        assert!(!native_output_type_accepts(&expected, &native_integer(1)));
        let input = NativeTypeUnion::new([NativeValueType::Handle(audio)])?;
        assert!(!input.accepts(&null));
        Ok(())
    }
}
