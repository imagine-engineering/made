use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativeResolvedPayload, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_model::NativeBackgroundRemovalResource;
use sim_tensor::{ImageTensor, NativeTensorPayload, NativeTensorRole};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["RemoveBackground"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
const FEATURE_ID: &str = "COMFY-NODE-0532";
const CLASS_TYPE: &str = "RemoveBackground";
const IMPLEMENTATION_VERSION: &str = "source-75b415ac-v1";
const CACHE_CHANGE_TOKEN: &str = "source-75b415ac-background-removal-v1";

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let source_schema = built_in_source_schema(CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["bg_removal_model".to_owned(), "image".to_owned()],
            &[],
            &["mask".to_owned()],
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
                handle_input("bg_removal_model", model_type()?)?,
                handle_input("image", image_type()?)?,
            ],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "mask".to_owned(),
                produced_type: NativeValueType::Handle(mask_type()?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "Remove Background".to_owned(),
            category: "image/background removal".to_owned(),
            description: "Generates a foreground mask to remove the background from an image using a background removal model.".to_owned(),
            output_names: vec!["mask".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(RemoveBackground),
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

fn model_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Model, "BACKGROUND_REMOVAL")
}

fn mask_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Mask, "MASK")
}

#[derive(Debug)]
struct RemoveBackground;
impl NativeNode for RemoveBackground {
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
        let (_, model_handle) = input_handles(inputs)?;
        let resolved = resolve_handle(
            context,
            model_handle,
            model_type().map_err(|error| invalid_inputs(error.to_string()))?,
            "bg_removal_model",
        )?;
        let model = resolved_model(&resolved)?;
        Ok(NativeCacheDependencies {
            artifact_digests: BTreeMap::from([
                (
                    "background-removal-artifact".to_owned(),
                    model.artifact_sha256().to_owned(),
                ),
                (
                    "background-removal-execution".to_owned(),
                    model.semantic_digest_sha256().to_owned(),
                ),
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
            let (image_handle, model_handle) = input_handles(&inputs)?;
            let image_payload = resolve_handle(
                &context,
                image_handle,
                image_type().map_err(|error| invalid_inputs(error.to_string()))?,
                "image",
            )?;
            let model_payload = resolve_handle(
                &context,
                model_handle,
                model_type().map_err(|error| invalid_inputs(error.to_string()))?,
                "bg_removal_model",
            )?;
            let image = resolved_image_tensor(&image_payload)?;
            let model = resolved_model(&model_payload)?;
            let compute = context.compute_session().map_err(compute_failure)?;
            let execution = compute
                .execution_context(&context)
                .map_err(compute_failure)?;
            let mask = model
                .encode_image(compute.backend(), image, &execution)
                .map_err(|error| model_failure(&context, error))?;
            let payload = NativeTensorPayload::from_tensor(NativeTensorRole::Mask, mask)
                .map_err(|error| invalid_inputs(error.to_string()))?;
            check_cancellation(&context)?;
            let handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Tensor(Arc::new(payload)),
                    &context.cancellation,
                )
                .map_err(|error| handle_failure(&context, "output", error))?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle {
                    value: handle.clone(),
                }],
                ui: None,
                effects: Vec::new(),
            };
            if let Err(error) = check_cancellation(&context).and_then(|()| {
                outcome
                    .validate()
                    .map_err(|error| invalid_inputs(error.to_string()))
            }) {
                context
                    .handle_store()
                    .revoke(&handle, &sim_types::CancellationToken::default())
                    .map_err(|error| NativeNodeFailure {
                        code: "output_rollback_failed".to_owned(),
                        message: format!("Background removal output rollback failed: {error}"),
                        kind: NativeNodeFailureKind::Failure,
                        retryable: false,
                    })?;
                return Err(error);
            }
            Ok(outcome)
        })
    }
}

fn input_handles(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(&NativeOpaqueHandle, &NativeOpaqueHandle), NativeNodeFailure> {
    if inputs.len() != 2 {
        return Err(invalid_inputs(
            "RemoveBackground requires exactly bg_removal_model and image inputs",
        ));
    }
    Ok((
        exact_handle(inputs, "image", NativeHandleKind::Image, "IMAGE")?,
        exact_handle(
            inputs,
            "bg_removal_model",
            NativeHandleKind::Model,
            "BACKGROUND_REMOVAL",
        )?,
    ))
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

fn resolved_model(
    resolved: &NativeResolvedPayload,
) -> Result<&Arc<NativeBackgroundRemovalResource>, NativeNodeFailure> {
    let NativeStoredPayload::Model(payload) = resolved.as_ref() else {
        return Err(invalid_inputs(
            "bg_removal_model handle does not contain a native model payload",
        ));
    };
    payload
        .model_payload()
        .background_removal_resource()
        .ok_or_else(|| {
            invalid_inputs("bg_removal_model handle has no canonical background-removal resource")
        })
}

fn model_failure(
    context: &NativeNodeContext,
    error: sim_model::NativeBackgroundRemovalError,
) -> NativeNodeFailure {
    if context.cancellation.is_cancelled()
        || matches!(error, sim_model::NativeBackgroundRemovalError::Cancelled)
    {
        return interrupted_failure();
    }
    NativeNodeFailure {
        code: "native_background_removal_failed".to_owned(),
        message: format!("Background removal failed: {error}"),
        kind: NativeNodeFailureKind::Failure,
        retryable: matches!(
            error,
            sim_model::NativeBackgroundRemovalError::Allocation
                | sim_model::NativeBackgroundRemovalError::OutOfMemory { .. }
        ),
    }
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
            message: format!("Background removal {input} handle is unavailable: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "native_compute_unavailable".to_owned(),
        message: format!("Background removal compute session is unavailable: {error}"),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
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
        message: "RemoveBackground execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativeNodeComputeSession, NativeNodeServices,
        NativeResolvedPayloadRetention, NativeStoredModelPayload, NodeRegistry,
    };
    use sim_model::{BackgroundRemovalFixtureMutation, NativeModelPayload};
    use sim_tensor::{
        CpuWorkspaceAuthority, StreamId, generated_native_diffusion::tensor_to_f32,
    };
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::Value;
    use std::{
        error::Error,
        sync::{
            Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };
    const MEMORY_LIMIT: u64 = 512 * 1024 * 1024;
    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-background-removal-comfy-node-0532/fixture.json"
    ));
    const ORACLE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/background-removal-resource-foundation/oracle.json"
    ));
    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        fail_publish_at: AtomicU64,
        cancel_after_publish_at: AtomicU64,
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
                cancel_after_publish_at: AtomicU64::new(u64::MAX),
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
            let identifier = format!("background-removal-{identifier_number}");
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock poisoned".into()))?
                .insert(identifier.clone(), Arc::new(payload));
            if identifier_number == self.cancel_after_publish_at.load(Ordering::Acquire) {
                cancellation.cancel();
            }
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

    impl ExecutionHarness {
        fn new(store_id: u128, generation_id: u128) -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(uuid::Uuid::from_u128(0x53210));
            let (backend, workspace) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
            Ok(Self {
                store: TestStore::new(store_id, generation_id, attempt_id)?,
                backend: Arc::new(backend),
                workspace,
                attempt_id,
                node_id: NodeId("background-removal-test".to_owned()),
            })
        }
        fn model(
            &self,
            mutation: BackgroundRemovalFixtureMutation,
            memory_budget: Option<u64>,
        ) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            let cancellation = CancellationToken::default();
            let execution = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(MEMORY_LIMIT)?,
                &cancellation,
            );
            let mut checkpoint = sim_model::deterministic_reduced_test_checkpoint(
                &self.backend,
                &execution,
                mutation,
            )?;
            if let Some(budget) = memory_budget {
                checkpoint.memory_budget_bytes = budget;
            }
            let resource =
                NativeBackgroundRemovalResource::from_reduced_fixture(checkpoint, &execution)?;
            let payload = NativeModelPayload::background_removal_test_fixture(
                Arc::new(resource),
                &cancellation,
            )?;
            Ok(self.store.publish(
                NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::model_resource(
                    Arc::new(payload),
                )?)),
                &cancellation,
            )?)
        }
        fn image(&self, batch: u64, channels: u64) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            let fixture: Value = serde_json::from_str(FIXTURE)?;
            let pixels: Vec<f32> = serde_json::from_value(fixture["input_rgba"].clone())?;
            let mut values = Vec::new();
            for _ in 0..batch {
                for pixel in pixels.chunks_exact(4) {
                    values.extend(pixel.iter().take(usize::try_from(channels)?));
                }
            }
            let cancellation = CancellationToken::default();
            let execution = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(MEMORY_LIMIT)?,
                &cancellation,
            );
            let image =
                ImageTensor::from_f32(&self.backend, &execution, batch, 3, 5, channels, &values)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                    NativeTensorRole::Image,
                    image,
                )?)),
                &cancellation,
            )?)
        }
        fn context(
            &self,
            cancellation: CancellationToken,
        ) -> Result<NativeNodeContext, Box<dyn Error>> {
            let scratch = self.workspace.authorize_workspace(MEMORY_LIMIT)?;
            let identity = crate::NativeNodeServiceIdentity::checked(
                uuid::Uuid::from_u128(0x53211),
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
                PromptId(uuid::Uuid::from_u128(0x53212)),
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
        let binding = bindings
            .pop()
            .ok_or("Background removal binding is absent")?;
        if !bindings.is_empty() {
            return Err("Background removal family emitted extra bindings".into());
        }
        match binding {
            NativeNodeBinding::Executable {
                descriptor,
                presentation,
                node,
                ..
            } => Ok((descriptor, presentation, node)),
            _ => Err("Background removal binding is not executable".into()),
        }
    }

    fn inputs(
        image: NativeOpaqueHandle,
        model: NativeOpaqueHandle,
    ) -> BTreeMap<String, NativeValue> {
        BTreeMap::from([
            ("image".to_owned(), NativeValue::Handle { value: image }),
            (
                "bg_removal_model".to_owned(),
                NativeValue::Handle { value: model },
            ),
        ])
    }

    fn output_handle(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("mask values are absent".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        let [NativeValue::Handle { value }] = outputs.as_slice() else {
            return Err("exactly one mask handle is required".into());
        };
        Ok(value.clone())
    }

    fn mask_bits(
        harness: &ExecutionHarness,
        handle: &NativeOpaqueHandle,
        batch: u64,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let resolved = harness
            .store
            .resolve(handle, &mask_type()?, &cancellation)?;
        let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
            return Err("output is not a native tensor".into());
        };
        assert_eq!(payload.role(), NativeTensorRole::Mask);
        assert_eq!(payload.tensor().descriptor().shape(), &[batch, 3, 5]);
        let execution = harness.backend.execution_context(
            StreamId::DEFAULT,
            harness.workspace.authorize_workspace(MEMORY_LIMIT)?,
            &cancellation,
        );
        Ok(
            tensor_to_f32(&harness.backend, payload.tensor(), &execution)?
                .iter()
                .map(|value| value.to_bits())
                .collect(),
        )
    }

    #[test]
    fn exact_source_schema_registry_and_serialization() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(fixture["feature_id"], FEATURE_ID);
        assert_eq!(
            fixture["source"]["sha256"],
            "c2cf4b42f10cfb1bb057b60a8745fb96a2462e1f1a2bd275e00795bb3f758cce"
        );
        assert_eq!(
            fixture["source"]["symbol"]["sha256"],
            "75b415acedaeaa1a694aeba2e4b0367524c6878e3e4a1f48b2a62898c68109f9"
        );
        assert_eq!(NODE_DESCRIPTOR_IDS, &[CLASS_TYPE]);
        let (descriptor, presentation, node) = executable()?;
        descriptor.validate_exact_schema_v2()?;
        assert_eq!(
            descriptor
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            ["bg_removal_model", "image"]
        );
        assert!(descriptor.inputs.iter().all(|input| input.required
            && !input.hidden
            && !input.lazy
            && !input.allows_literal
            && input.cardinality == NativePortCardinality::Scalar));
        assert!(descriptor.dynamic_inputs.is_empty());
        assert_eq!(descriptor.outputs.len(), 1);
        let output = descriptor
            .outputs
            .first()
            .ok_or("mask descriptor is absent")?;
        assert_eq!(output.name, "mask");
        assert_eq!(output.produced_type, NativeValueType::Handle(mask_type()?));
        assert!(!output.is_list);
        assert!(!descriptor.output_node);
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        assert_eq!(
            presentation.display_name,
            fixture["schema"]["presentation"]["display_name"]
        );
        assert_eq!(
            presentation.description,
            fixture["schema"]["presentation"]["description"]
        );
        assert_eq!(presentation.category, "image/background removal");
        assert_eq!(presentation.output_names, ["mask"]);
        assert!(!presentation.is_deprecated && !presentation.is_experimental);
        assert!(presentation.search_aliases.is_empty());
        assert_eq!(node.class_type(), CLASS_TYPE);
        assert_eq!(node.implementation_version(), IMPLEMENTATION_VERSION);
        let schema = serde_json::to_value(
            descriptor
                .source_schema
                .as_ref()
                .ok_or("source schema absent")?,
        )?;
        assert_eq!(schema["node"], fixture["schema"]["node"]);
        assert_eq!(
            schema["dynamic_inputs"],
            fixture["schema"]["dynamic_inputs"]
        );
        let expected_inputs = fixture["schema"]["inputs"]
            .as_array()
            .ok_or("fixture inputs absent")?;
        let mut normalized_inputs = Vec::new();
        for input in expected_inputs {
            let mut input = input
                .as_object()
                .ok_or("fixture input is not an object")?
                .clone();
            input.remove("requirement");
            normalized_inputs.push(Value::Object(input));
        }
        assert_eq!(schema["inputs"], Value::Array(normalized_inputs));
        let mut expected_output = fixture["schema"]["outputs"]
            .as_array()
            .and_then(|outputs| outputs.first())
            .and_then(Value::as_object)
            .ok_or("fixture output absent")?
            .clone();
        let name = expected_output
            .remove("source_name")
            .ok_or("source output name absent")?;
        expected_output.remove("ordinal");
        expected_output.insert("name".to_owned(), name);
        assert_eq!(
            schema["outputs"],
            Value::Array(vec![Value::Object(expected_output)])
        );
        let encoded = serde_json::to_value(&descriptor)?;
        let restored: NativeNodeDescriptor = serde_json::from_value(encoded.clone())?;
        restored.validate_exact_schema_v2()?;
        assert_eq!(serde_json::to_value(restored)?, encoded);
        let restored_prompt: Value =
            serde_json::from_slice(&serde_json::to_vec(&fixture["prompt"])?)?;
        assert_eq!(restored_prompt, fixture["prompt"]);
        let binding = native_node_bindings()?.pop().ok_or("binding absent")?;
        binding.validate()?;
        NodeRegistry::built_in()?.validate_native_binding(&binding)?;
        Ok(())
    }

    #[test]
    fn canonical_masks_match_oracle_preserve_inputs_and_map_batches() -> Result<(), Box<dyn Error>>
    {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x53220, 0x53221)?;
        let model = harness.model(BackgroundRemovalFixtureMutation::None, None)?;
        let oracle: Value = serde_json::from_str(ORACLE)?;
        let expected: Vec<u32> = serde_json::from_value(oracle["output_bits"].clone())?;
        for (batch, channels) in [(1, 4), (1, 3), (2, 4)] {
            let image = harness.image(batch, channels)?;
            let cancellation = CancellationToken::default();
            let original = harness
                .store
                .resolve(&image, &image_type()?, &cancellation)?;
            let original_digest = original.as_ref().digest_sha256();
            let values = inputs(image.clone(), model.clone());
            let context = harness.context(cancellation)?;
            assert!(node.demanded_lazy_inputs(&context, &values)?.is_empty());
            assert_eq!(node.cache_change_token(&values)?, CACHE_CHANGE_TOKEN);
            let dependencies = node.cache_dependencies(&context, &values)?;
            assert_eq!(dependencies.artifact_digests.len(), 2);
            let count = harness.store.value_count()?;
            let handle = output_handle(futures::executor::block_on(
                node.execute(context, values.clone()),
            )?)?;
            assert_eq!(harness.store.value_count()?, count + 1);
            let actual = mask_bits(&harness, &handle, batch)?;
            assert_eq!(actual, expected.repeat(usize::try_from(batch)?));
            let after =
                harness
                    .store
                    .resolve(&image, &image_type()?, &CancellationToken::default())?;
            assert_eq!(after.as_ref().digest_sha256(), original_digest);
            assert_eq!(
                serde_json::to_value(node.cache_dependencies(
                    &harness.context(CancellationToken::default())?,
                    &values
                )?)?,
                serde_json::to_value(dependencies)?
            );
            if batch == 1 && channels == 4 {
                if let Some(path) = std::env::var_os("SIM_BACKGROUND_MASK_EVIDENCE") {
                    std::fs::write(
                        path,
                        actual
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn semantic_state_partitions_cache_and_restart_reacquires_handles() -> Result<(), Box<dyn Error>>
    {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x53230, 0x53231)?;
        let image = harness.image(1, 4)?;
        let first = inputs(
            image.clone(),
            harness.model(BackgroundRemovalFixtureMutation::None, None)?,
        );
        let changed = inputs(
            image,
            harness.model(BackgroundRemovalFixtureMutation::AsppDilatedBranch, None)?,
        );
        let first_dependencies =
            node.cache_dependencies(&harness.context(CancellationToken::default())?, &first)?;
        let changed_dependencies =
            node.cache_dependencies(&harness.context(CancellationToken::default())?, &changed)?;
        assert_ne!(
            first_dependencies
                .artifact_digests
                .get("background-removal-execution"),
            changed_dependencies
                .artifact_digests
                .get("background-removal-execution")
        );
        let first_mask = output_handle(futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            first.clone(),
        ))?)?;
        let changed_mask = output_handle(futures::executor::block_on(
            node.execute(harness.context(CancellationToken::default())?, changed),
        )?)?;
        let oracle: Value = serde_json::from_str(ORACLE)?;
        let changed_expected: Vec<u32> = serde_json::from_value(
            oracle["mutations"]["aspp-dilated-branch"]["output_bits"].clone(),
        )?;
        assert_eq!(mask_bits(&harness, &changed_mask, 1)?, changed_expected);
        let restarted = ExecutionHarness::new(0x53230, 0x53232)?;
        let stale = futures::executor::block_on(
            node.execute(restarted.context(CancellationToken::default())?, first),
        )
        .expect_err("old generation must fail");
        assert_eq!(stale.code, "invalid_image_handle");
        assert_eq!(restarted.store.value_count()?, 0);
        let fresh_inputs = inputs(
            restarted.image(1, 4)?,
            restarted.model(BackgroundRemovalFixtureMutation::None, None)?,
        );
        let fresh_context = restarted.context(CancellationToken::default())?;
        let fresh_dependencies = node.cache_dependencies(&fresh_context, &fresh_inputs)?;
        assert_eq!(
            first_dependencies.artifact_digests,
            fresh_dependencies.artifact_digests
        );
        let recovered = output_handle(futures::executor::block_on(
            node.execute(fresh_context, fresh_inputs),
        )?)?;
        assert_eq!(
            mask_bits(&harness, &first_mask, 1)?,
            mask_bits(&restarted, &recovered, 1)?
        );
        Ok(())
    }

    #[test]
    fn invalid_inputs_and_memory_failure_publish_nothing() -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x53240, 0x53241)?;
        let image = harness.image(1, 4)?;
        let model = harness.model(BackgroundRemovalFixtureMutation::None, None)?;
        let good = inputs(image.clone(), model.clone());
        let mut missing = good.clone();
        missing.remove("bg_removal_model");
        let mut extra = good.clone();
        extra.insert(
            "unexpected".to_owned(),
            NativeValue::Handle {
                value: image.clone(),
            },
        );
        let mut list = good.clone();
        list.insert(
            "image".to_owned(),
            NativeValue::List {
                values: vec![NativeValue::Handle {
                    value: image.clone(),
                }],
            },
        );
        for invalid in [missing, extra, list, inputs(model.clone(), image)] {
            assert!(node.cache_change_token(&invalid).is_err());
            assert!(
                node.demanded_lazy_inputs(
                    &harness.context(CancellationToken::default())?,
                    &invalid
                )
                .is_err()
            );
            let count = harness.store.value_count()?;
            let error = futures::executor::block_on(
                node.execute(harness.context(CancellationToken::default())?, invalid),
            )
            .expect_err("invalid inputs must fail");
            assert_eq!(error.code, "invalid_node_inputs");
            assert_eq!(harness.store.value_count()?, count);
        }
        let count = harness.store.value_count()?;
        for (batch, channels) in [(1, 1), (0, 4)] {
            let invalid = inputs(harness.image(batch, channels)?, model.clone());
            let before = harness.store.value_count()?;
            let error = futures::executor::block_on(
                node.execute(harness.context(CancellationToken::default())?, invalid),
            )
            .expect_err("unsupported image channels must fail");
            assert_eq!(error.code, "native_background_removal_failed");
            assert!(!error.retryable);
            assert_eq!(harness.store.value_count()?, before);
        }
        assert_eq!(harness.store.value_count()?, count + 2);
        assert!(harness.image(1, 2).is_err());
        let resolved =
            harness
                .store
                .resolve(&model, &model_type()?, &CancellationToken::default())?;
        let resident = resolved_model(&resolved)?.resident_bytes();
        let limited = harness.model(BackgroundRemovalFixtureMutation::None, Some(resident))?;
        let invalid = inputs(harness.image(1, 4)?, limited);
        let before = harness.store.value_count()?;
        let error = futures::executor::block_on(
            node.execute(harness.context(CancellationToken::default())?, invalid),
        )
        .expect_err("invocation memory limit must fail");
        assert_eq!(error.code, "native_background_removal_failed");
        assert!(error.retryable);
        assert_eq!(harness.store.value_count()?, before);
        let recovered = output_handle(futures::executor::block_on(
            node.execute(harness.context(CancellationToken::default())?, good),
        )?)?;
        assert_eq!(mask_bits(&harness, &recovered, 1)?.len(), 15);
        Ok(())
    }

    #[test]
    fn cancellation_before_and_during_publication_is_atomic_and_retryable()
    -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x53250, 0x53251)?;
        let values = inputs(
            harness.image(1, 4)?,
            harness.model(BackgroundRemovalFixtureMutation::None, None)?,
        );
        let count = harness.store.value_count()?;
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let context = harness.context(cancelled)?;
        assert_eq!(
            node.demanded_lazy_inputs(&context, &values)
                .expect_err("cancelled lazy validation")
                .kind,
            NativeNodeFailureKind::Interrupted
        );
        assert_eq!(
            node.cache_dependencies(&context, &values)
                .expect_err("cancelled cache dependency validation")
                .kind,
            NativeNodeFailureKind::Interrupted
        );
        let error = futures::executor::block_on(node.execute(context, values.clone()))
            .expect_err("pre-cancelled execution");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.value_count()?, count);
        harness
            .store
            .fail_publish_at(harness.store.next_identifier.load(Ordering::Acquire));
        let error = futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            values.clone(),
        ))
        .expect_err("cancelled publication");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.value_count()?, count);
        harness.store.cancel_after_publish_at.store(
            harness.store.next_identifier.load(Ordering::Acquire),
            Ordering::Release,
        );
        let error = futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            values.clone(),
        ))
        .expect_err("late cancelled output must be revoked");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.value_count()?, count);
        let output = output_handle(futures::executor::block_on(
            node.execute(harness.context(CancellationToken::default())?, values),
        )?)?;
        assert_eq!(mask_bits(&harness, &output, 1)?.len(), 15);
        assert_eq!(harness.store.value_count()?, count + 1);
        Ok(())
    }
    #[test]
    fn handles_reject_wrong_store_digest_and_revoked_model() -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let harness = ExecutionHarness::new(0x53260, 0x53261)?;
        let image = harness.image(1, 4)?;
        let model = harness.model(BackgroundRemovalFixtureMutation::None, None)?;
        let wrong_digest = NativeOpaqueHandle::new(
            image_type()?,
            image.store_identity(),
            image.identifier(),
            1,
            Some("f".repeat(64)),
        )?;
        let foreign = ExecutionHarness::new(0x53262, 0x53261)?;
        for invalid in [wrong_digest, foreign.image(1, 4)?] {
            let before = harness.store.value_count()?;
            let error = futures::executor::block_on(node.execute(
                harness.context(CancellationToken::default())?,
                inputs(invalid, model.clone()),
            ))
            .expect_err("forged or foreign handle must fail");
            assert_eq!(error.code, "invalid_image_handle");
            assert_eq!(harness.store.value_count()?, before);
        }
        harness
            .store
            .revoke(&model, &CancellationToken::default())?;
        let values = inputs(image, model);
        let values: BTreeMap<String, NativeValue> =
            serde_json::from_slice(&serde_json::to_vec(&values)?)?;
        let context = harness.context(CancellationToken::default())?;
        assert_eq!(
            node.cache_dependencies(&context, &values)
                .expect_err("revoked model cache dependency")
                .code,
            "invalid_bg_removal_model_handle"
        );
        let error = futures::executor::block_on(node.execute(context, values))
            .expect_err("revoked model must fail");
        assert_eq!(error.code, "invalid_bg_removal_model_handle");
        assert_eq!(harness.store.value_count()?, 1);
        Ok(())
    }
}
