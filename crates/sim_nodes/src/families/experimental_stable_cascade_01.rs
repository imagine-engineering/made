use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativeResolvedPayload, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_tensor::{
    ImageTensor, Layout, MemoryFormatReference, NativeLatentBundle, NativeLatentMetadata,
    NativeTensorPayload, NativeTensorRole, generated_native_diffusion::tensor_from_f32,
    generated_shape_layout_transform_02::torch_movedim_exact_native,
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
#[cfg(test)]
use sim_tensor::NativeLatentSamples;
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["StableCascade_SuperResolutionControlnet"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const FEATURE_ID: &str = "COMFY-NODE-0638";
const CLASS_TYPE: &str = "StableCascade_SuperResolutionControlnet";
const IMPLEMENTATION_VERSION: &str = "source-c11f471e-v1";
const CACHE_CHANGE_TOKEN: &str = "source-c11f471e-stable-cascade-super-resolution-v1";

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let image_type = image_type()?;
    let latent_type = latent_type()?;
    let source_schema = built_in_source_schema(CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["image".to_owned(), "vae".to_owned()],
            &[],
            &[
                "controlnet_input".to_owned(),
                "stage_c".to_owned(),
                "stage_b".to_owned(),
            ],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(vec![NativeNodeBinding::Executable {
        feature_id: FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: CLASS_TYPE.to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![
                handle_input("image", image_type.clone())?,
                handle_input("vae", vae_type()?)?,
            ],
            dynamic_inputs: Vec::new(),
            outputs: vec![
                NativeOutputDescriptor {
                    name: "controlnet_input".to_owned(),
                    produced_type: NativeValueType::Handle(image_type),
                    is_list: false,
                },
                NativeOutputDescriptor {
                    name: "stage_c".to_owned(),
                    produced_type: NativeValueType::Handle(latent_type.clone()),
                    is_list: false,
                },
                NativeOutputDescriptor {
                    name: "stage_b".to_owned(),
                    produced_type: NativeValueType::Handle(latent_type),
                    is_list: false,
                },
            ],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: CLASS_TYPE.to_owned(),
            category: "experimental/stable cascade".to_owned(),
            description: String::new(),
            output_names: vec![
                "controlnet_input".to_owned(),
                "stage_c".to_owned(),
                "stage_b".to_owned(),
            ],
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: true,
        },
        node: Arc::new(StableCascadeSuperResolutionControlnet),
    }])
}

fn handle_input(
    name: &str,
    handle_type: NativeHandleType,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([NativeValueType::Handle(handle_type)])?,
        required: true,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: false,
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

fn vae_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Vae, "VAE")
}

fn latent_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Latent, "LATENT")
}

#[derive(Debug)]
struct StableCascadeSuperResolutionControlnet;

impl NativeNode for StableCascadeSuperResolutionControlnet {
    fn class_type(&self) -> &str {
        CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context)?;
        input_handles(available_inputs)?;
        Ok(BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        input_handles(inputs)?;
        Ok(CACHE_CHANGE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context)?;
        let (_, vae_handle) = input_handles(inputs)?;
        let expected_type = vae_type().map_err(|error| invalid_inputs(error.to_string()))?;
        let resolved = resolve_handle(context, vae_handle, expected_type, "vae")?;
        let vae = resolved_vae(&resolved)?;
        Ok(NativeCacheDependencies {
            artifact_digests: BTreeMap::from([
                (
                    "vae-artifact".to_owned(),
                    vae.descriptor().identity().artifact_sha256().to_owned(),
                ),
                ("vae-execution".to_owned(), vae.execution_digest()),
            ]),
            ..Default::default()
        })
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context)?;
            let (image_handle, vae_handle) = input_handles(&inputs)?;
            let expected_image = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
            let expected_vae = vae_type().map_err(|error| invalid_inputs(error.to_string()))?;
            let resolved_image = resolve_handle(&context, image_handle, expected_image, "image")?;
            let resolved_vae_payload =
                resolve_handle(&context, vae_handle, expected_vae, "vae")?;
            let image = resolved_image_tensor(&resolved_image)?;
            let vae = resolved_vae(&resolved_vae_payload)?;
            let (batch, height, width, channels) = image
                .dimensions()
                .map_err(|error| tensor_failure(&context, error.to_string()))?;
            if channels < 3 {
                return Err(invalid_inputs(
                    "image must contain at least three channels for VAE encoding",
                ));
            }
            let stage_c_shape = [batch, 16, height / 16, width / 16];
            let stage_b_shape = [batch, 4, height / 2, width / 2];
            if stage_c_shape.contains(&0) || stage_b_shape.contains(&0) {
                return Err(invalid_inputs(
                    "image height and width must both be at least 16",
                ));
            }

            let compute = context.compute_session().map_err(compute_failure)?;
            let execution = compute
                .execution_context(&context)
                .map_err(|error| compute_failure(error.to_string()))?;
            if image.tensor().descriptor().stream() != compute.stream() {
                return Err(invalid_inputs(
                    "image and native VAE compute streams must match",
                ));
            }
            let pixels = torch_movedim_exact_native(
                image.tensor(),
                &[-1],
                &[1],
                execution.cancellation,
            )
            .map_err(|error| tensor_failure(&context, error.to_string()))?;
            let encoded = vae
                .encode(compute.backend(), &pixels, &execution)
                .map_err(|error| vae_failure(&context, error))?;
            let controlnet_input = torch_movedim_exact_native(
                &encoded,
                &[1],
                &[-1],
                execution.cancellation,
            )
            .map_err(|error| tensor_failure(&context, error.to_string()))?;
            let controlnet_input = contiguous_with_context_exact_native(
                compute.backend(),
                &controlnet_input,
                MemoryFormatReference::Layout(Layout::Contiguous),
                &execution,
            )
            .map_err(|error| tensor_failure(&context, error.to_string()))?;
            let controlnet_input = ImageTensor::from_tensor(controlnet_input)
                .map_err(|error| tensor_failure(&context, error.to_string()))?;

            let stage_c = zero_tensor(compute.backend(), &stage_c_shape, &execution)
                .map_err(|error| tensor_failure(&context, error))?;
            let stage_b = zero_tensor(compute.backend(), &stage_b_shape, &execution)
                .map_err(|error| tensor_failure(&context, error))?;
            let stage_c = NativeLatentBundle::single(
                stage_c,
                None,
                None,
                NativeLatentMetadata::default(),
                &execution,
            )
            .map_err(|error| tensor_failure(&context, error.to_string()))?;
            let stage_b = NativeLatentBundle::single(
                stage_b,
                None,
                None,
                NativeLatentMetadata::default(),
                &execution,
            )
            .map_err(|error| tensor_failure(&context, error.to_string()))?;
            check_cancellation(&context)?;

            let payloads = [
                NativeStoredPayload::Tensor(Arc::new(
                    NativeTensorPayload::from_image(NativeTensorRole::Image, controlnet_input)
                        .map_err(|error| tensor_failure(&context, error.to_string()))?,
                )),
                NativeStoredPayload::Latent(Arc::new(stage_c)),
                NativeStoredPayload::Latent(Arc::new(stage_b)),
            ];
            let outputs = publish_outputs(&context, payloads)?;
            let outcome = NativeNodeOutcome::Values {
                outputs,
                ui: None,
                effects: Vec::new(),
            };
            if let Err(error) = outcome.validate() {
                revoke_outputs(&context, &outcome_handles(&outcome)?)?;
                return Err(invalid_inputs(error.to_string()));
            }
            drop(resolved_image);
            drop(resolved_vae_payload);
            Ok(outcome)
        })
    }
}

fn input_handles(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(&NativeOpaqueHandle, &NativeOpaqueHandle), NativeNodeFailure> {
    if inputs.len() != 2 {
        return Err(invalid_inputs(
            "StableCascade_SuperResolutionControlnet requires exactly image and vae inputs",
        ));
    }
    let image = exact_handle(inputs, "image", NativeHandleKind::Image, "IMAGE")?;
    let vae = exact_handle(inputs, "vae", NativeHandleKind::Vae, "VAE")?;
    Ok((image, vae))
}

fn exact_handle<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
    kind: NativeHandleKind,
    type_id: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = inputs.get(name) else {
        return Err(invalid_inputs(format!(
            "{name} must be an exact {type_id} handle"
        )));
    };
    if value.handle_type().kind != kind || value.handle_type().type_id != type_id {
        return Err(invalid_inputs(format!(
            "{name} must be an exact {type_id} handle"
        )));
    }
    Ok(value)
}

fn resolve_handle(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
    expected_type: NativeHandleType,
    input: &'static str,
) -> Result<NativeResolvedPayload, NativeNodeFailure> {
    context
        .handle_store()
        .resolve(handle, &expected_type, &context.cancellation)
        .map_err(|error| handle_failure(context, input, error))
}

fn resolved_image_tensor(
    resolved: &NativeResolvedPayload,
) -> Result<&ImageTensor, NativeNodeFailure> {
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(invalid_inputs(
            "image handle does not contain a native tensor payload",
        ));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(invalid_inputs("image handle has the wrong tensor role"));
    }
    payload
        .image()
        .ok_or_else(|| invalid_inputs("image handle has no canonical ImageTensor"))
}

fn resolved_vae(
    resolved: &NativeResolvedPayload,
) -> Result<&Arc<sim_model::NativeVae>, NativeNodeFailure> {
    let NativeStoredPayload::Model(payload) = resolved.as_ref() else {
        return Err(invalid_inputs(
            "vae handle does not contain a native model payload",
        ));
    };
    payload
        .model_payload()
        .vae()
        .ok_or_else(|| invalid_inputs("vae handle has no canonical NativeVae resource"))
}

fn zero_tensor(
    backend: &sim_tensor::CpuBackend,
    shape: &[u64],
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, String> {
    let count = shape.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension)
    });
    let count = count
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| "Stable Cascade latent shape overflowed".to_owned())?;
    let mut values = backend
        .workspace_vec(context, count)
        .map_err(|error| error.to_string())?;
    for index in 0..count {
        if index.is_multiple_of(1_024) {
            context.check().map_err(|error| error.to_string())?;
        }
        values.try_push(0.0).map_err(|error| error.to_string())?;
    }
    tensor_from_f32(backend, shape, &values, context).map_err(|error| error.to_string())
}

fn publish_outputs(
    context: &NativeNodeContext,
    payloads: [NativeStoredPayload; 3],
) -> Result<Vec<NativeValue>, NativeNodeFailure> {
    let mut published = Vec::with_capacity(payloads.len());
    for payload in payloads {
        match context
            .handle_store()
            .publish(payload, &context.cancellation)
        {
            Ok(handle) => published.push(handle),
            Err(error) => {
                revoke_outputs(context, &published)?;
                return Err(handle_failure(context, "output", error));
            }
        }
    }
    Ok(published
        .into_iter()
        .map(|value| NativeValue::Handle { value })
        .collect())
}

fn outcome_handles(outcome: &NativeNodeOutcome) -> Result<Vec<NativeOpaqueHandle>, NativeNodeFailure> {
    let NativeNodeOutcome::Values { outputs, .. } = outcome else {
        return Err(invalid_inputs("Stable Cascade output disposition changed"));
    };
    outputs
        .iter()
        .map(|value| match value {
            NativeValue::Handle { value } => Ok(value.clone()),
            _ => Err(invalid_inputs("Stable Cascade output type changed")),
        })
        .collect()
}

fn revoke_outputs(
    context: &NativeNodeContext,
    published: &[NativeOpaqueHandle],
) -> Result<(), NativeNodeFailure> {
    let cleanup_cancellation = sim_types::CancellationToken::default();
    for handle in published.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &cleanup_cancellation)
            .map_err(|error| NativeNodeFailure {
                code: "output_rollback_failed".to_owned(),
                message: format!("Stable Cascade output rollback failed: {error}"),
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            })?;
    }
    Ok(())
}

fn check_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure())
}

fn handle_failure(
    context: &NativeNodeContext,
    input: &'static str,
    error: NativeHandleStoreError,
) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() || matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure()
    } else {
        NativeNodeFailure {
            code: format!("invalid_{input}_handle"),
            message: format!("Stable Cascade {input} handle is unavailable: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "native_compute_unavailable".to_owned(),
        message: format!("Stable Cascade compute session is unavailable: {error}"),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
    }
}

fn vae_failure(context: &NativeNodeContext, error: sim_model::VaeError) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        interrupted_failure()
    } else {
        let retryable = matches!(error, sim_model::VaeError::Allocation(_));
        NativeNodeFailure {
            code: "native_vae_encode_failed".to_owned(),
            message: format!("Stable Cascade VAE encode failed: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable,
        }
    }
}

fn tensor_failure(context: &NativeNodeContext, message: impl Into<String>) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        interrupted_failure()
    } else {
        NativeNodeFailure {
            code: "native_tensor_failed".to_owned(),
            message: format!("Stable Cascade tensor execution failed: {}", message.into()),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure() -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: "StableCascade_SuperResolutionControlnet execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativeNodeBindingDisposition,
        NativeNodeComputeSession, NativeNodeServices, NativeResolvedPayloadRetention, NodeRegistry,
    };
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelStore, NativeModelPayload, ParserLimits,
        PatchGraph, VaeArchitectureRegistry, VaeBoundary, VaeDescriptor,
        vae_image::load_image_vae_from_model_store_with_context,
    };
    use sim_sampler::NativeDiffusionPayload;
    use sim_tensor::{CpuWorkspaceAuthority, StreamId, generated_native_diffusion::tensor_to_f32};
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::{Value, json};
    use std::{
        error::Error,
        fs,
        path::Path,
        sync::{
            Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/experimental-stable-cascade-comfy-node-0638/fixture.json"
    ));

    #[derive(Debug)]
    struct InertStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: sim_types::AttemptId,
    }

    impl NativeHandleStore for InertStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.identity
        }

        fn attempt_id(&self) -> sim_types::AttemptId {
            self.attempt_id
        }

        fn resolve(
            &self,
            _handle: &NativeOpaqueHandle,
            _expected_type: &NativeHandleType,
            _cancellation: &sim_types::CancellationToken,
        ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
            Err(NativeHandleStoreError::Rejected(
                "cancelled test context must not resolve handles".to_owned(),
            ))
        }

        fn publish(
            &self,
            _payload: NativeStoredPayload,
            _cancellation: &sim_types::CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            Err(NativeHandleStoreError::Rejected(
                "cancelled test context must not publish handles".to_owned(),
            ))
        }

        fn revoke(
            &self,
            _handle: &NativeOpaqueHandle,
            _cancellation: &sim_types::CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            Err(NativeHandleStoreError::Rejected(
                "cancelled test context must not revoke handles".to_owned(),
            ))
        }
    }

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        fail_publish_at: AtomicU64,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(
            store_id: u128,
            generation_id: u128,
            attempt_id: AttemptId,
        ) -> Result<Arc<Self>, Box<dyn Error>> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    uuid::Uuid::from_u128(store_id),
                    uuid::Uuid::from_u128(generation_id),
                )?,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                fail_publish_at: AtomicU64::new(u64::MAX),
                values: Mutex::new(BTreeMap::new()),
            }))
        }

        fn fail_publish_at(&self, identifier: u64) {
            self.fail_publish_at.store(identifier, Ordering::Release);
        }

        fn value_count(&self) -> Result<usize, NativeHandleStoreError> {
            self.values
                .lock()
                .map(|values| values.len())
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock poisoned".into()))
        }
    }

    impl NativeHandleStore for TestStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.identity
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
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            if handle.store_identity().store_id != self.identity.store_id {
                return Err(NativeHandleStoreError::WrongStore);
            }
            if handle.store_identity().generation_id != self.identity.generation_id {
                return Err(NativeHandleStoreError::WrongGeneration);
            }
            if handle.handle_type() != expected_type {
                return Err(NativeHandleStoreError::WrongType {
                    expected: expected_type.type_id.clone(),
                    actual: handle.handle_type().type_id.clone(),
                });
            }
            let payload = self
                .values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock poisoned".into()))?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            if handle.digest_sha256() != Some(payload.digest_sha256().as_str()) {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            NativeResolvedPayload::checked(payload, Arc::new(TestRetention)).map_err(Into::into)
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
            let digest = payload.digest_sha256();
            let identifier_number = self.next_identifier.fetch_add(1, Ordering::AcqRel);
            if identifier_number == self.fail_publish_at.load(Ordering::Acquire) {
                cancellation.cancel();
                return Err(NativeHandleStoreError::Cancelled);
            }
            let identifier = format!("stable-cascade-{identifier_number}");
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock poisoned".into()))?
                .insert(identifier.clone(), Arc::new(payload));
            NativeOpaqueHandle::new(handle_type, self.identity, identifier, 1, Some(digest))
                .map_err(Into::into)
        }

        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock poisoned".into()))?
                .remove(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            Ok(())
        }
    }

    struct ExecutionHarness {
        store: Arc<TestStore>,
        backend: Arc<sim_tensor::CpuBackend>,
        workspace: CpuWorkspaceAuthority,
        attempt_id: AttemptId,
        node_id: NodeId,
    }

    struct TemporaryFixture(std::path::PathBuf);

    impl Drop for TemporaryFixture {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                eprintln!(
                    "failed to remove Stable Cascade temporary VAE fixture {}: {error}",
                    self.0.display()
                );
            }
        }
    }

    fn write_pixel_space_vae(path: &Path) -> Result<(), Box<dyn Error>> {
        let mut header =
            br#"{"pixel_space_vae":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#.to_vec();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let header_length = u64::try_from(header.len())?;
        let mut bytes = Vec::with_capacity(8 + header.len() + 4);
        bytes.extend_from_slice(&header_length.to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        fs::write(path, bytes)?;
        Ok(())
    }

    impl ExecutionHarness {
        fn new(store_id: u128, generation_id: u128) -> Result<Self, Box<dyn Error>> {
            const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
            let attempt_id = AttemptId(uuid::Uuid::from_u128(0x63810));
            let (backend, workspace) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
            Ok(Self {
                store: TestStore::new(store_id, generation_id, attempt_id)?,
                backend: Arc::new(backend),
                workspace,
                attempt_id,
                node_id: NodeId("stable-cascade-execution-test".to_owned()),
            })
        }

        fn load_vae(&self) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
            let cancellation = CancellationToken::default();
            let fixture_root = std::env::temp_dir().join(format!(
                "zed-stable-cascade-pixel-vae-{}",
                uuid::Uuid::new_v4()
            ));
            fs::create_dir(&fixture_root)?;
            let _cleanup = TemporaryFixture(fixture_root.clone());
            write_pixel_space_vae(&fixture_root.join("pixel-space.safetensors"))?;
            let mut index = ArtifactIndex::default();
            index.add_root(ArtifactRoot::canonical(
                "stable-cascade-test-vae",
                "checkpoint",
                &fixture_root,
                ["safetensors"],
            )?)?;
            index.refresh(&cancellation)?;
            let key = ArtifactKey::new("stable-cascade-test-vae", "pixel-space.safetensors")?;
            let artifact = index
                .record(&key)
                .cloned()
                .ok_or("pixel-space native VAE fixture is absent")?;
            let mut model_store = ModelStore::new(ParserLimits::default())?;
            let loaded = model_store.load(&index, &artifact.key, &cancellation)?;
            let registry = VaeArchitectureRegistry::checked()?;
            let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
            let probe = model_store.family_probe(&loaded, &cancellation)?;
            let selection = registry.select(&probe, &cancellation)?;
            let target = registry.intended_target(
                &selection,
                &family_registry,
                &latent_registry,
                &cancellation,
            )?;
            let latent_definition = latent_registry
                .get(target.latent_format())
                .ok_or("pixel-space latent format is absent")?;
            let patch = PatchGraph::checked_semantic(&artifact.sha256, Vec::new())?.identity();
            let descriptor = VaeDescriptor::checked_selection(
                &artifact,
                &selection,
                &target,
                &family_registry,
                &latent_registry,
                patch,
                VaeBoundary::image(3)?,
                [0.0, 1.0],
                &cancellation,
            )?;
            let execution = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(MEMORY_LIMIT)?,
                &cancellation,
            );
            let vae = Arc::new(load_image_vae_from_model_store_with_context(
                &self.backend,
                &model_store,
                &index,
                loaded,
                descriptor,
                latent_definition,
                &execution,
            )?);
            let model = Arc::new(NativeModelPayload::native_vae(vae)?);
            let diffusion = Arc::new(NativeDiffusionPayload::vae(model)?);
            self.store
                .publish(
                    NativeStoredPayload::Model(Arc::new(
                        crate::NativeStoredModelPayload::native_diffusion(diffusion)?,
                    )),
                    &cancellation,
                )
                .map_err(Into::into)
        }

        fn image(&self) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            self.image_with_dimensions(1, 32, 32, 4)
        }

        fn image_with_dimensions(
            &self,
            batch: u64,
            height: u64,
            width: u64,
            channels: u64,
        ) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
            let cancellation = CancellationToken::default();
            let execution = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(MEMORY_LIMIT)?,
                &cancellation,
            );
            let count = [batch, height, width, channels]
                .into_iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(dimension))
                .and_then(|count| usize::try_from(count).ok())
                .ok_or("test image dimensions overflowed")?;
            let values = vec![0.5_f32; count];
            let image = ImageTensor::from_f32(
                &self.backend,
                &execution,
                batch,
                height,
                width,
                channels,
                &values,
            )?;
            self.store
                .publish(
                    NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                        NativeTensorRole::Image,
                        image,
                    )?)),
                    &cancellation,
                )
                .map_err(Into::into)
        }

        fn context(
            &self,
            cancellation: CancellationToken,
        ) -> Result<NativeNodeContext, Box<dyn Error>> {
            const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
            let scratch = self.workspace.authorize_workspace(MEMORY_LIMIT)?;
            let identity = crate::NativeNodeServiceIdentity::checked(
                uuid::Uuid::from_u128(0x63811),
                self.attempt_id,
                self.node_id.clone(),
            )?;
            let compute = NativeNodeComputeSession::checked(
                identity,
                self.backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            Ok(NativeNodeContext::new_with_services(
                PromptId(uuid::Uuid::from_u128(0x63812)),
                self.attempt_id,
                self.node_id.clone(),
                cancellation,
                scratch,
                self.store.clone(),
                NativeNodeServices::checked(None, None, Some(compute))?,
            )?)
        }
    }

    fn executable() -> Result<
        (
            NativeNodeDescriptor,
            NativeNodePresentation,
            Arc<dyn NativeNode>,
        ),
        Box<dyn Error>,
    > {
        let mut bindings = native_node_bindings()?;
        let binding = bindings.pop().ok_or("Stable Cascade binding is absent")?;
        if !bindings.is_empty() {
            return Err("Stable Cascade family emitted extra bindings".into());
        }
        match binding {
            NativeNodeBinding::Executable {
                descriptor,
                presentation,
                node,
                ..
            } => Ok((descriptor, presentation, node)),
            _ => Err("Stable Cascade binding is not executable".into()),
        }
    }

    #[test]
    fn source_fixture_schema_and_registry_are_exact() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(fixture["feature_id"], FEATURE_ID);
        assert_eq!(
            fixture.pointer("/source/sha256").and_then(Value::as_str),
            Some("c11f471ef730405e43e66fbadca3adcf4bc503b3fb5997ea262fca9da7aaa09a")
        );
        assert_eq!(NODE_DESCRIPTOR_IDS, &[CLASS_TYPE]);
        let (descriptor, presentation, node) = executable()?;
        descriptor.validate_exact_schema_v2()?;
        assert_eq!(descriptor.class_type, CLASS_TYPE);
        assert_eq!(descriptor.inputs.len(), 2);
        assert_eq!(descriptor.inputs[0].name, "image");
        assert_eq!(descriptor.inputs[1].name, "vae");
        assert!(descriptor.inputs.iter().all(|input| {
            input.required
                && !input.hidden
                && !input.lazy
                && input.cardinality == NativePortCardinality::Scalar
                && !input.allows_literal
        }));
        assert_eq!(
            descriptor
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            ["controlnet_input", "stage_c", "stage_b"]
        );
        assert!(descriptor.outputs.iter().all(|output| !output.is_list));
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        assert!(!descriptor.output_node);
        assert_eq!(presentation.display_name, CLASS_TYPE);
        assert_eq!(presentation.category, "experimental/stable cascade");
        assert!(presentation.is_experimental);
        assert!(!presentation.is_deprecated);
        assert_eq!(node.class_type(), CLASS_TYPE);
        assert_eq!(node.implementation_version(), IMPLEMENTATION_VERSION);

        let binding = native_node_bindings()?.remove(0);
        assert_eq!(binding.disposition(), NativeNodeBindingDisposition::Executable);
        binding.validate()?;
        NodeRegistry::built_in()?.validate_native_binding(&binding)?;
        Ok(())
    }

    #[test]
    fn fixture_pins_source_equations_boundaries_and_persistence() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture.pointer("/behavior/stage_c/shape"),
            Some(&json!(["batch", 16, "height // 16", "width // 16"]))
        );
        assert_eq!(
            fixture.pointer("/behavior/stage_b/shape"),
            Some(&json!(["batch", 4, "height // 2", "width // 2"]))
        );
        assert_eq!(
            fixture.pointer("/behavior/controlnet_input/equation"),
            Some(&json!("vae.encode(image[:, :, :, :3]).movedim(1, -1)"))
        );
        let persistence = fixture
            .pointer("/behavior/persistence")
            .cloned()
            .ok_or("persistence fixture is absent")?;
        assert_eq!(
            serde_json::from_slice::<Value>(&serde_json::to_vec(&persistence)?)?,
            persistence
        );
        assert_eq!(
            persistence,
            json!({
                "class_type": CLASS_TYPE,
                "inputs": { "image": [7, 0], "vae": [8, 0] },
                "unknown_data": { "preserve": true, "source_extension": "nodes_stable_cascade" }
            })
        );
        Ok(())
    }

    #[test]
    fn native_vae_success_latent_boundaries_cache_and_recovery_are_exact()
    -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x63820, 0x63821)?;
        let image = harness.image()?;
        let vae = harness.load_vae()?;
        let inputs = BTreeMap::from([
            (
                "image".to_owned(),
                NativeValue::Handle { value: image },
            ),
            (
                "vae".to_owned(),
                NativeValue::Handle { value: vae },
            ),
        ]);
        let context = harness.context(CancellationToken::default())?;
        assert!(node.demanded_lazy_inputs(&context, &inputs)?.is_empty());
        assert_eq!(node.cache_change_token(&inputs)?, CACHE_CHANGE_TOKEN);
        let dependencies = node.cache_dependencies(&context, &inputs)?;
        assert_eq!(dependencies.artifact_digests.len(), 2);
        assert!(dependencies.artifact_digests.contains_key("vae-artifact"));
        assert!(dependencies.artifact_digests.contains_key("vae-execution"));

        let before_count = harness.store.value_count()?;
        let outcome = futures::executor::block_on(node.execute(context, inputs.clone()))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("Stable Cascade node did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        let [
            NativeValue::Handle {
                value: controlnet_input,
            },
            NativeValue::Handle { value: stage_c },
            NativeValue::Handle { value: stage_b },
        ] = outputs.as_slice()
        else {
            return Err("Stable Cascade outputs changed type or cardinality".into());
        };
        assert_eq!(harness.store.value_count()?, before_count + 3);

        let cancellation = CancellationToken::default();
        let controlnet = harness
            .store
            .resolve(controlnet_input, &image_type()?, &cancellation)?;
        let NativeStoredPayload::Tensor(controlnet) = controlnet.as_ref() else {
            return Err("controlnet_input is not a native tensor".into());
        };
        assert_eq!(controlnet.role(), NativeTensorRole::Image);
        assert_eq!(
            controlnet
                .image()
                .ok_or("controlnet_input has no ImageTensor")?
                .dimensions()?,
            (1, 32, 32, 3)
        );

        for (handle, expected_shape) in [
            (stage_c, vec![1, 16, 2, 2]),
            (stage_b, vec![1, 4, 16, 16]),
        ] {
            let resolved = harness
                .store
                .resolve(handle, &latent_type()?, &cancellation)?;
            let NativeStoredPayload::Latent(payload) = resolved.as_ref() else {
                return Err("Stable Cascade latent output is not a native latent bundle".into());
            };
            let NativeLatentSamples::Tensor(samples) = payload.samples() else {
                return Err("Stable Cascade latent output is unexpectedly nested".into());
            };
            assert_eq!(samples.descriptor().shape(), expected_shape);
            let execution = harness.backend.execution_context(
                StreamId::DEFAULT,
                harness.workspace.authorize_workspace(2 * 1024 * 1024 * 1024)?,
                &cancellation,
            );
            assert!(
                tensor_to_f32(&harness.backend, samples, &execution)?
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }

        let restarted = ExecutionHarness::new(0x63820, 0x63822)?;
        let stale_error = futures::executor::block_on(node.execute(
            restarted.context(CancellationToken::default())?,
            inputs,
        ))
        .expect_err("stale worker handles must fail");
        assert!(matches!(
            stale_error.code.as_str(),
            "invalid_image_handle" | "invalid_vae_handle"
        ));
        Ok(())
    }

    #[test]
    fn image_boundaries_fail_before_publication() -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x63830, 0x63831)?;
        let vae = harness.load_vae()?;
        for image in [
            harness.image_with_dimensions(1, 32, 32, 1)?,
            harness.image_with_dimensions(1, 15, 32, 4)?,
        ] {
            let inputs = BTreeMap::from([
                ("image".to_owned(), NativeValue::Handle { value: image }),
                (
                    "vae".to_owned(),
                    NativeValue::Handle { value: vae.clone() },
                ),
            ]);
            let before_count = harness.store.value_count()?;
            let error = futures::executor::block_on(node.execute(
                harness.context(CancellationToken::default())?,
                inputs,
            ))
            .expect_err("invalid image boundary must fail");
            assert_eq!(error.code, "invalid_node_inputs");
            assert_eq!(harness.store.value_count()?, before_count);
        }
        Ok(())
    }

    #[test]
    fn interrupted_output_publication_rolls_back_atomically() -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x63840, 0x63841)?;
        let image = harness.image()?;
        let vae = harness.load_vae()?;
        let inputs = BTreeMap::from([
            ("image".to_owned(), NativeValue::Handle { value: image }),
            ("vae".to_owned(), NativeValue::Handle { value: vae }),
        ]);
        let before_count = harness.store.value_count()?;
        harness.store.fail_publish_at(4);
        let error = futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            inputs,
        ))
        .expect_err("injected output publication cancellation must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.value_count()?, before_count);
        Ok(())
    }

    #[test]
    fn exact_handle_validation_cache_token_and_cancellation_are_structural()
    -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let identity = crate::NativeHandleStoreIdentity::new(
            uuid::Uuid::from_u128(0x63801),
            uuid::Uuid::from_u128(0x63802),
        )?;
        let image = NativeOpaqueHandle::new(
            image_type()?,
            identity,
            "image-1",
            1,
            Some("1".repeat(64)),
        )?;
        let vae = NativeOpaqueHandle::new(
            vae_type()?,
            identity,
            "vae-1",
            1,
            Some("2".repeat(64)),
        )?;
        let inputs = BTreeMap::from([
            ("image".to_owned(), NativeValue::Handle { value: image }),
            ("vae".to_owned(), NativeValue::Handle { value: vae }),
        ]);
        assert_eq!(node.cache_change_token(&inputs)?, CACHE_CHANGE_TOKEN);

        let mut missing = inputs;
        missing.remove("vae");
        assert_eq!(
            node.cache_change_token(&missing)
                .expect_err("missing VAE must fail")
                .code,
            "invalid_node_inputs"
        );
        let cancellation = sim_types::CancellationToken::default();
        cancellation.cancel();
        let error = check_cancellation(&NativeNodeContext::new(
            sim_types::PromptId(uuid::Uuid::from_u128(0x63803)),
            sim_types::AttemptId(uuid::Uuid::from_u128(0x63804)),
            sim_types::NodeId("stable-cascade-cancelled".to_owned()),
            cancellation,
            sim_tensor::CpuWorkspaceAuthority::create_backend(1)?.1.authorize_workspace(0)?,
            Arc::new(InertStore {
                identity: crate::NativeHandleStoreIdentity::new(
                    uuid::Uuid::from_u128(0x63805),
                    uuid::Uuid::from_u128(0x63806),
                )?,
                attempt_id: sim_types::AttemptId(uuid::Uuid::from_u128(0x63804)),
            }),
        )?)
        .expect_err("pre-cancelled context must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        Ok(())
    }
}
