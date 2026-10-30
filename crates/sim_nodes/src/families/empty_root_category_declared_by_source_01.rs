use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativeTypeUnion, NativeValue, NativeValueType,
    built_in_source_schema,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["wanBlockSwap"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const FEATURE_ID: &str = "COMFY-NODE-0757";
const CLASS_TYPE: &str = "wanBlockSwap";
const IMPLEMENTATION_VERSION: &str = "source-4f9130b7-v1";
const CACHE_CHANGE_TOKEN: &str = "wan-block-swap-identity-source-4f9130b7-v1";
const DESCRIPTION: &str =
    "Intercept wanBlockSwap custom node that causes major instability and make it no-op.";

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let model_type = model_type()?;
    let source_schema = built_in_source_schema(CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(&["model".to_owned()], &[], &["model".to_owned()])
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(vec![NativeNodeBinding::Executable {
        feature_id: FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: CLASS_TYPE.to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![NativeInputDescriptor {
                name: "model".to_owned(),
                accepted_types: NativeTypeUnion::new([NativeValueType::Handle(
                    model_type.clone(),
                )])?,
                required: true,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: false,
            }],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "model".to_owned(),
                produced_type: NativeValueType::Handle(model_type),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: CLASS_TYPE.to_owned(),
            category: String::new(),
            description: DESCRIPTION.to_owned(),
            output_names: vec!["model".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: true,
            is_experimental: false,
        },
        node: Arc::new(WanBlockSwap),
    }])
}

fn model_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Model, "MODEL")
}

#[derive(Debug)]
struct WanBlockSwap;

impl NativeNode for WanBlockSwap {
    fn class_type(&self) -> &str {
        CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        _available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context)?;
        Ok(BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        model_handle(inputs)?;
        Ok(CACHE_CHANGE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context)?;
        model_handle(inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context)?;
            let handle = model_handle(&inputs)?.clone();
            let expected_type = model_type().map_err(|error| invalid_inputs(error.to_string()))?;
            let resolved = context
                .handle_store()
                .resolve(&handle, &expected_type, &context.cancellation)
                .map_err(handle_failure)?;
            check_cancellation(&context)?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle { value: handle }],
                ui: None,
                effects: Vec::new(),
            };
            outcome
                .validate()
                .map_err(|error| invalid_inputs(error.to_string()))?;
            drop(resolved);
            Ok(outcome)
        })
    }
}

fn model_handle(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<&NativeOpaqueHandle, NativeNodeFailure> {
    if inputs.len() != 1 {
        return Err(invalid_inputs(
            "wanBlockSwap requires exactly one model input",
        ));
    }
    let Some(NativeValue::Handle { value }) = inputs.get("model") else {
        return Err(invalid_inputs(
            "wanBlockSwap model input must be a MODEL handle",
        ));
    };
    if value.handle_type().kind != NativeHandleKind::Model || value.handle_type().type_id != "MODEL"
    {
        return Err(invalid_inputs(
            "wanBlockSwap model input must be an exact MODEL handle",
        ));
    }
    Ok(value)
}

fn check_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure())
}

fn handle_failure(error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure()
    } else {
        NativeNodeFailure {
            code: "invalid_model_handle".to_owned(),
            message: format!("wanBlockSwap model handle is not available: {error}"),
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
        message: "wanBlockSwap execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativePrimitive, NativeResolvedPayload,
        NativeResolvedPayloadRetention, NativeStoredModelPayload, NativeStoredPayload,
        NodeRegistry,
    };
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelStore, NativeModelPayload, ParserLimits,
        PatchGraph,
        generated_native_diffusion::{Sd15DetectorProjection, Sd15TinyModel},
    };
    use sim_sampler::{NativeConditioningPayload, NativeDiffusionPayload};
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::{Value, json};
    use std::{
        error::Error,
        fs,
        path::Path,
        sync::{
            Arc, OnceLock,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/empty-root-category-declared-by-source-comfy-node-0757/fixture.json"
    ));

    #[derive(Debug)]
    struct TestResolvedPayloadRetention;

    impl NativeResolvedPayloadRetention for TestResolvedPayloadRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        identifier: String,
        generation: u64,
        digest: String,
        value: Arc<NativeStoredPayload>,
        resolve_count: AtomicUsize,
        publish_count: AtomicUsize,
        revoke_count: AtomicUsize,
        cancel_after_resolve: AtomicBool,
    }

    impl TestStore {
        fn new(
            store_id: u128,
            generation_id: u128,
            attempt_id: AttemptId,
        ) -> Result<Arc<Self>, Box<dyn Error>> {
            let value = canonical_model_payload()?;
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(store_id),
                    Uuid::from_u128(generation_id),
                )?,
                attempt_id,
                identifier: "model-1".to_owned(),
                generation: 1,
                digest: value.digest_sha256(),
                value,
                resolve_count: AtomicUsize::new(0),
                publish_count: AtomicUsize::new(0),
                revoke_count: AtomicUsize::new(0),
                cancel_after_resolve: AtomicBool::new(false),
            }))
        }

        fn handle(&self) -> Result<NativeOpaqueHandle, NativeNodeContractError> {
            NativeOpaqueHandle::new(
                model_type()?,
                self.identity,
                self.identifier.clone(),
                self.generation,
                Some(self.digest.clone()),
            )
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
            if handle.identifier() != self.identifier || handle.generation() != self.generation {
                return Err(NativeHandleStoreError::Missing(
                    handle.identifier().to_owned(),
                ));
            }
            if handle.digest_sha256() != Some(self.digest.as_str()) {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            self.value.validate().map_err(|error| {
                NativeHandleStoreError::Rejected(format!(
                    "wanBlockSwap test MODEL payload is invalid: {error}"
                ))
            })?;
            let payload_type = self.value.handle_type().map_err(|error| {
                NativeHandleStoreError::Rejected(format!(
                    "wanBlockSwap test MODEL payload type is invalid: {error}"
                ))
            })?;
            if &payload_type != expected_type {
                return Err(NativeHandleStoreError::WrongType {
                    expected: expected_type.type_id.clone(),
                    actual: payload_type.type_id,
                });
            }
            self.resolve_count.fetch_add(1, Ordering::AcqRel);
            if self.cancel_after_resolve.load(Ordering::Acquire) {
                cancellation.cancel();
            }
            Ok(NativeResolvedPayload::checked(
                self.value.clone(),
                Arc::new(TestResolvedPayloadRetention),
            )?)
        }

        fn publish(
            &self,
            _payload: NativeStoredPayload,
            _cancellation: &CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            self.publish_count.fetch_add(1, Ordering::AcqRel);
            Err(NativeHandleStoreError::Rejected(
                "wanBlockSwap must not publish handles".to_owned(),
            ))
        }

        fn revoke(
            &self,
            _handle: &NativeOpaqueHandle,
            _cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            self.revoke_count.fetch_add(1, Ordering::AcqRel);
            Err(NativeHandleStoreError::Rejected(
                "wanBlockSwap must not revoke handles".to_owned(),
            ))
        }
    }

    fn canonical_model_payload() -> Result<Arc<NativeStoredPayload>, Box<dyn Error>> {
        static MODEL_PAYLOAD: OnceLock<Arc<NativeStoredPayload>> = OnceLock::new();
        if let Some(payload) = MODEL_PAYLOAD.get() {
            return Ok(payload.clone());
        }

        let fixture_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sim_test_support/fixtures/models/sd15-tiny-v1");
        let cancellation = CancellationToken::default();
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "wan-block-swap-model",
            "checkpoint",
            &fixture_root,
            ["safetensors"],
        )?)?;
        index.refresh(&cancellation)?;
        let key = ArtifactKey::new("wan-block-swap-model", "model.safetensors")?;
        let artifact = index
            .record(&key)
            .cloned()
            .ok_or("wanBlockSwap canonical MODEL fixture is absent")?;
        let mut model_store = ModelStore::new(ParserLimits::default())?;
        let loaded = model_store.load(&index, &artifact.key, &cancellation)?;
        let projection: Sd15DetectorProjection = serde_json::from_slice(&fs::read(
            fixture_root.join("sd15-detector-projection.json"),
        )?)?;
        let patch_graph = Arc::new(PatchGraph::checked_semantic(loaded.identity(), Vec::new())?);
        let memory_limit = 2 * 1024 * 1024 * 1024;
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
        let backend = Arc::new(backend);
        let scratch = workspace.authorize_workspace(memory_limit)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let model = Arc::new(Sd15TinyModel::load_production_with_patch_graph(
            &model_store,
            &index,
            &loaded,
            &projection,
            &patch_graph,
            backend,
            &context,
        )?);
        let model_owner = Arc::new(NativeModelPayload::sd15_model(model.clone())?);
        let conditioning = Arc::new(NativeConditioningPayload::checked_sd15(
            &artifact.sha256,
            model.as_ref(),
            patch_graph,
            None,
        )?);
        let diffusion = Arc::new(NativeDiffusionPayload::model(model_owner, conditioning)?);
        let payload = Arc::new(NativeStoredPayload::Model(Arc::new(
            NativeStoredModelPayload::native_diffusion(diffusion)?,
        )));
        payload.validate()?;
        if payload.handle_type()? != model_type()? {
            return Err(
                "wanBlockSwap canonical fixture did not produce an exact MODEL payload".into(),
            );
        }

        match MODEL_PAYLOAD.set(payload.clone()) {
            Ok(()) => Ok(payload),
            Err(_) => MODEL_PAYLOAD
                .get()
                .cloned()
                .ok_or_else(|| "wanBlockSwap canonical MODEL cache was not initialized".into()),
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
        let binding = bindings.pop().ok_or("wanBlockSwap binding is absent")?;
        if !bindings.is_empty() {
            return Err("wanBlockSwap family emitted extra bindings".into());
        }
        match binding {
            NativeNodeBinding::Executable {
                descriptor,
                presentation,
                node,
                ..
            } => Ok((descriptor, presentation, node)),
            _ => Err("wanBlockSwap binding is not executable".into()),
        }
    }

    fn context(
        store: Arc<TestStore>,
        cancellation: CancellationToken,
    ) -> Result<NativeNodeContext, Box<dyn Error>> {
        let (_backend, workspace) = CpuWorkspaceAuthority::create_backend(1)?;
        Ok(NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x75701)),
            store.attempt_id,
            NodeId("wan-block-swap-test".to_owned()),
            cancellation,
            workspace.authorize_workspace(0)?,
            store,
        )?)
    }

    fn inputs(handle: NativeOpaqueHandle) -> BTreeMap<String, NativeValue> {
        BTreeMap::from([("model".to_owned(), NativeValue::Handle { value: handle })])
    }

    #[test]
    fn source_fixture_and_schema_are_exact() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let (descriptor, presentation, node) = executable()?;
        assert_eq!(NODE_DESCRIPTOR_IDS, &[CLASS_TYPE]);
        assert_eq!(
            fixture.pointer("/feature_id").and_then(Value::as_str),
            Some(FEATURE_ID)
        );
        assert_eq!(
            fixture.pointer("/source/sha256").and_then(Value::as_str),
            Some("4f9130b7db711de4aae861dbe47790a5a105203ffcdd98ccf5a8286a09adca62")
        );
        assert_eq!(
            fixture
                .pointer("/source/byte_length")
                .and_then(Value::as_u64),
            Some(1411)
        );
        assert_eq!(descriptor.class_type, CLASS_TYPE);
        assert_eq!(descriptor.implementation_version, IMPLEMENTATION_VERSION);
        assert_eq!(
            descriptor.schema_version,
            NATIVE_NODE_CONTRACT_SCHEMA_VERSION
        );
        assert_eq!(descriptor.inputs.len(), 1);
        let input = descriptor.inputs.first().ok_or("model input is absent")?;
        assert_eq!(input.name, "model");
        assert!(input.required);
        assert!(!input.hidden);
        assert!(!input.lazy);
        assert_eq!(input.cardinality, NativePortCardinality::Scalar);
        assert!(!input.allows_literal);
        assert_eq!(descriptor.outputs.len(), 1);
        let output = descriptor.outputs.first().ok_or("model output is absent")?;
        assert_eq!(output.name, "model");
        assert!(!output.is_list);
        assert!(descriptor.dynamic_inputs.is_empty());
        assert!(!descriptor.output_node);
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        assert_eq!(presentation.display_name, CLASS_TYPE);
        assert!(presentation.category.is_empty());
        assert_eq!(presentation.description, DESCRIPTION);
        assert_eq!(presentation.output_names, ["model"]);
        assert!(presentation.is_deprecated);
        assert!(!presentation.is_experimental);
        assert_eq!(node.class_type(), descriptor.class_type);
        assert_eq!(
            node.implementation_version(),
            descriptor.implementation_version
        );
        let binding = native_node_bindings()?
            .into_iter()
            .next()
            .ok_or("wanBlockSwap binding is absent")?;
        binding.validate()?;
        NodeRegistry::built_in()?.validate_native_binding(&binding)?;
        Ok(())
    }

    #[test]
    fn identity_success_cache_change_and_effects_are_exact() -> Result<(), Box<dyn Error>> {
        let (_, _, node) = executable()?;
        let attempt_id = AttemptId(Uuid::from_u128(0x75702));
        let store = TestStore::new(0x75703, 0x75704, attempt_id)?;
        let handle = store.handle()?;
        let values = inputs(handle.clone());
        let node_context = context(store.clone(), CancellationToken::default())?;
        assert!(
            node.demanded_lazy_inputs(&node_context, &values)?
                .is_empty()
        );
        assert_eq!(node.cache_change_token(&values)?, CACHE_CHANGE_TOKEN);
        assert_eq!(
            node.cache_dependencies(&node_context, &values)?,
            NativeCacheDependencies::default()
        );
        let outcome = futures::executor::block_on(node.execute(node_context, values))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("wanBlockSwap did not produce values".into());
        };
        assert_eq!(outputs, [NativeValue::Handle { value: handle }]);
        assert!(ui.is_none());
        assert!(effects.is_empty());
        assert_eq!(store.resolve_count.load(Ordering::Acquire), 1);
        assert_eq!(store.publish_count.load(Ordering::Acquire), 0);
        assert_eq!(store.revoke_count.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn boundary_validation_failure_and_cancellation_publish_nothing() -> Result<(), Box<dyn Error>>
    {
        let (_, _, node) = executable()?;
        let attempt_id = AttemptId(Uuid::from_u128(0x75705));
        let store = TestStore::new(0x75706, 0x75707, attempt_id)?;
        let invalid_inputs = [
            BTreeMap::new(),
            BTreeMap::from([(
                "model".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String("not-a-model".to_owned()),
                },
            )]),
            BTreeMap::from([("model".to_owned(), NativeValue::List { values: Vec::new() })]),
            BTreeMap::from([
                (
                    "model".to_owned(),
                    NativeValue::Handle {
                        value: store.handle()?,
                    },
                ),
                (
                    "extra".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Null,
                    },
                ),
            ]),
        ];
        for values in invalid_inputs {
            let error = futures::executor::block_on(node.execute(
                context(store.clone(), CancellationToken::default())?,
                values,
            ))
            .expect_err("invalid wanBlockSwap input must fail");
            assert_eq!(error.code, "invalid_node_inputs");
        }

        let wrong_type = NativeOpaqueHandle::new(
            NativeHandleType::new(NativeHandleKind::Image, "IMAGE")?,
            store.identity,
            store.identifier.clone(),
            1,
            Some(store.digest.clone()),
        )?;
        let error = futures::executor::block_on(node.execute(
            context(store.clone(), CancellationToken::default())?,
            inputs(wrong_type),
        ))
        .expect_err("wrong handle type must fail");
        assert_eq!(error.code, "invalid_node_inputs");

        let missing = NativeOpaqueHandle::new(
            model_type()?,
            store.identity,
            "missing-model",
            1,
            Some(store.digest.clone()),
        )?;
        let error = futures::executor::block_on(node.execute(
            context(store.clone(), CancellationToken::default())?,
            inputs(missing),
        ))
        .expect_err("missing model handle must fail");
        assert_eq!(error.code, "invalid_model_handle");

        let forged_handles = [
            NativeOpaqueHandle::new(
                model_type()?,
                NativeHandleStoreIdentity::new(Uuid::from_u128(0x7570c), Uuid::from_u128(0x75707))?,
                store.identifier.clone(),
                1,
                Some(store.digest.clone()),
            )?,
            NativeOpaqueHandle::new(
                model_type()?,
                store.identity,
                store.identifier.clone(),
                1,
                Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned()),
            )?,
        ];
        for forged_handle in forged_handles {
            let error = futures::executor::block_on(node.execute(
                context(store.clone(), CancellationToken::default())?,
                inputs(forged_handle),
            ))
            .expect_err("forged model handle must fail");
            assert_eq!(error.code, "invalid_model_handle");
        }

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = futures::executor::block_on(node.execute(
            context(store.clone(), cancellation)?,
            inputs(store.handle()?),
        ))
        .expect_err("pre-cancelled execution must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);

        store.cancel_after_resolve.store(true, Ordering::Release);
        let error = futures::executor::block_on(node.execute(
            context(store.clone(), CancellationToken::default())?,
            inputs(store.handle()?),
        ))
        .expect_err("late cancellation must suppress output");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(store.publish_count.load(Ordering::Acquire), 0);
        assert_eq!(store.revoke_count.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn persistence_and_worker_restart_recovery_are_lossless() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let persistence = fixture
            .pointer("/behavior/persistence")
            .cloned()
            .ok_or("fixture persistence case is absent")?;
        let encoded = serde_json::to_vec(&persistence)?;
        assert_eq!(serde_json::from_slice::<Value>(&encoded)?, persistence);
        assert_eq!(
            persistence.get("class_type").and_then(Value::as_str),
            Some(CLASS_TYPE)
        );
        assert_eq!(persistence.get("inputs"), Some(&json!({ "model": [7, 0] })));
        assert_eq!(
            persistence.get("unknown_data"),
            Some(&json!({
                "source_extension": "wanBlockSwap",
                "preserve": true
            }))
        );

        let (_, _, node) = executable()?;
        let attempt_id = AttemptId(Uuid::from_u128(0x75708));
        let before_restart = TestStore::new(0x75709, 0x7570a, attempt_id)?;
        let stale_handle = before_restart.handle()?;
        let after_restart = TestStore::new(0x75709, 0x7570b, attempt_id)?;
        let error = futures::executor::block_on(node.execute(
            context(after_restart.clone(), CancellationToken::default())?,
            inputs(stale_handle),
        ))
        .expect_err("stale worker generation must fail closed");
        assert_eq!(error.code, "invalid_model_handle");

        let fresh_handle = after_restart.handle()?;
        let outcome = futures::executor::block_on(node.execute(
            context(after_restart.clone(), CancellationToken::default())?,
            inputs(fresh_handle.clone()),
        ))?;
        let NativeNodeOutcome::Values { outputs, .. } = outcome else {
            return Err("recovered wanBlockSwap execution did not produce values".into());
        };
        assert_eq!(
            outputs,
            [NativeValue::Handle {
                value: fresh_handle
            }]
        );
        assert_eq!(after_restart.publish_count.load(Ordering::Acquire), 0);
        Ok(())
    }
}
