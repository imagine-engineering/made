use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeLtxvPreprocessServiceError, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality, NativePrimitive,
    NativePrimitiveType, NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType,
    built_in_source_schema,
};
use sim_tensor::{NativeTensorPayload, NativeTensorRole};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["LTXVPreprocess"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CLASS_TYPE: &str = "LTXVPreprocess";
const FEATURE_ID: &str = "COMFY-NODE-0372";
const IMPLEMENTATION_VERSION: &str = "source-d14fe068-v1";
const CACHE_CHANGE_TOKEN: &str = "source-d14fe068-image-compression-service-v1";
const SERVICE_ARTIFACT: &str = "ltxv-preprocess-service";

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    Ok(vec![native_node_binding()?])
}

fn native_node_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    let image_type = image_type()?;
    let source_schema = built_in_source_schema(CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["image".to_owned(), "img_compression".to_owned()],
            &[],
            &["output_image".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: CLASS_TYPE.to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![
                NativeInputDescriptor {
                    name: "image".to_owned(),
                    accepted_types: NativeTypeUnion::new([NativeValueType::Handle(
                        image_type.clone(),
                    )])?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: false,
                },
                NativeInputDescriptor {
                    name: "img_compression".to_owned(),
                    accepted_types: NativeTypeUnion::new([NativeValueType::Primitive(
                        NativePrimitiveType::Integer,
                    )])?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: true,
                },
            ],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "output_image".to_owned(),
                produced_type: NativeValueType::Handle(image_type),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "LTXV Preprocess".to_owned(),
            category: "video/preprocessors".to_owned(),
            description: "Apply LTXV-compatible H.264 compression preprocessing to an image batch."
                .to_owned(),
            output_names: vec!["output_image".to_owned()],
            search_aliases: vec!["ltxv preprocess".to_owned()],
            is_deprecated: false,
            is_experimental: true,
        },
        node: Arc::new(LtxvPreprocessNode),
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

#[derive(Debug)]
struct LtxvPreprocessNode;

impl NativeNode for LtxvPreprocessNode {
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
        parse_inputs(available_inputs)?;
        Ok(BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        parse_inputs(inputs)?;
        Ok(CACHE_CHANGE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context)?;
        parse_inputs(inputs)?;
        let service = context
            .ltxv_preprocess_service()
            .map_err(service_failure)?;
        Ok(NativeCacheDependencies {
            artifact_digests: BTreeMap::from([(
                SERVICE_ARTIFACT.to_owned(),
                service.identity().configuration_sha256().to_owned(),
            )]),
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
            let (input_handle, compression) = parse_inputs(&inputs)?;
            let expected_type = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
            let resolved = context
                .handle_store()
                .resolve(input_handle, &expected_type, &context.cancellation)
                .map_err(handle_failure)?;
            let NativeStoredPayload::Tensor(input_payload) = resolved.as_ref() else {
                return Err(invalid_payload(
                    "resolved IMAGE handle does not contain a tensor payload",
                ));
            };
            if input_payload.role() != NativeTensorRole::Image {
                return Err(invalid_payload("resolved IMAGE handle has the wrong role"));
            }
            let input_image = input_payload.image().ok_or_else(|| {
                invalid_payload("resolved IMAGE handle has no canonical ImageTensor")
            })?;
            let compute = context
                .compute_session()
                .map_err(|error| compute_failure(error.to_string()))?;
            let execution_context = compute
                .execution_context(&context)
                .map_err(|error| compute_failure(error.to_string()))?;
            let service = context
                .ltxv_preprocess_service()
                .map_err(service_failure)?;
            let output_image = service
                .preprocess_image(input_image, compression, &execution_context)
                .await
                .map_err(service_failure)?;
            check_cancellation(&context)?;
            let output_payload =
                NativeTensorPayload::from_image(NativeTensorRole::Image, output_image)
                    .map_err(|error| invalid_payload(error.to_string()))?;
            let output_handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Tensor(Arc::new(output_payload)),
                    &context.cancellation,
                )
                .map_err(handle_failure)?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle {
                    value: output_handle,
                }],
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

fn parse_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(&NativeOpaqueHandle, u8), NativeNodeFailure> {
    if inputs.len() != 2 {
        return Err(invalid_inputs(
            "LTXVPreprocess requires exactly image and img_compression inputs",
        ));
    }
    let Some(NativeValue::Handle { value: image }) = inputs.get("image") else {
        return Err(invalid_inputs(
            "LTXVPreprocess image input must be an IMAGE handle",
        ));
    };
    if image.handle_type().kind != NativeHandleKind::Image || image.handle_type().type_id != "IMAGE"
    {
        return Err(invalid_inputs(
            "LTXVPreprocess image input must be an exact IMAGE handle",
        ));
    }
    let compression = match inputs.get("img_compression") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u8::try_from(*value).ok(),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => u8::try_from(*value).ok(),
        _ => None,
    }
    .filter(|value| *value <= 100)
    .ok_or_else(|| {
        invalid_inputs("LTXVPreprocess img_compression must be an integer from 0 through 100")
    })?;
    Ok((image, compression))
}

fn check_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure())
}

fn service_failure(error: NativeLtxvPreprocessServiceError) -> NativeNodeFailure {
    match error {
        NativeLtxvPreprocessServiceError::Cancelled => interrupted_failure(),
        NativeLtxvPreprocessServiceError::Unavailable => NativeNodeFailure {
            code: "ltxv_preprocess_unavailable".to_owned(),
            message: "LTXVPreprocess service is unavailable".to_owned(),
            kind: NativeNodeFailureKind::Failure,
            retryable: true,
        },
        NativeLtxvPreprocessServiceError::Busy => NativeNodeFailure {
            code: "ltxv_preprocess_busy".to_owned(),
            message: "LTXVPreprocess service is busy".to_owned(),
            kind: NativeNodeFailureKind::Failure,
            retryable: true,
        },
        NativeLtxvPreprocessServiceError::ResourceExhausted => NativeNodeFailure {
            code: "ltxv_preprocess_resource_exhausted".to_owned(),
            message: "LTXVPreprocess exhausted its reviewed resources".to_owned(),
            kind: NativeNodeFailureKind::Failure,
            retryable: true,
        },
        NativeLtxvPreprocessServiceError::InvalidRequest => NativeNodeFailure {
            code: "ltxv_preprocess_invalid_request".to_owned(),
            message: "LTXVPreprocess rejected the request".to_owned(),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
        NativeLtxvPreprocessServiceError::Execution(message) => NativeNodeFailure {
            code: "ltxv_preprocess_failed".to_owned(),
            message: format!("LTXVPreprocess failed: {message}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
    }
}

fn handle_failure(error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure()
    } else {
        NativeNodeFailure {
            code: "invalid_image_handle".to_owned(),
            message: format!("LTXVPreprocess IMAGE handle is not available: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(message: String) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "native_compute_unavailable".to_owned(),
        message: format!("LTXVPreprocess compute session is unavailable: {message}"),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
    }
}

fn invalid_payload(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_image_payload".to_owned(),
        message: format!("LTXVPreprocess: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
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
        message: "LTXVPreprocess execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativeLtxvPreprocessService,
        NativeLtxvPreprocessServiceIdentity, NativeNodeComputeSession, NativeNodeServiceIdentity,
        NativeNodeServices, NativeResolvedPayload, NativeResolvedPayloadRetention,
    };
    use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext, ImageTensor, StreamId};
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::Value;
    use std::{
        error::Error,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, AtomicU8, Ordering},
        },
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/video-preprocessors-comfy-node-0372/fixture.json"
    ));

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(attempt_id: AttemptId) -> Result<Arc<Self>, Box<dyn Error>> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(0x37201),
                    Uuid::from_u128(0x37202),
                )?,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                values: Mutex::new(BTreeMap::new()),
            }))
        }

        fn count(&self) -> Result<usize, NativeHandleStoreError> {
            self.values
                .lock()
                .map(|values| values.len())
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned()))
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
            if handle.store_identity() != self.identity {
                return Err(NativeHandleStoreError::WrongStore);
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
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
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
            let identifier = format!(
                "ltxv-image-{}",
                self.next_identifier.fetch_add(1, Ordering::AcqRel)
            );
            self.values
                .lock()
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
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
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
                .remove(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            Ok(())
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(u8)]
    enum ServiceMode {
        Success = 0,
        Busy = 1,
        ResourceExhausted = 2,
        CancelAfterOutput = 3,
    }

    #[derive(Debug)]
    struct TestService {
        identity: NativeLtxvPreprocessServiceIdentity,
        mode: AtomicU8,
        calls: AtomicU64,
        compression: AtomicU64,
    }

    impl TestService {
        fn new() -> Result<Arc<Self>, NativeNodeContractError> {
            Ok(Arc::new(Self {
                identity: NativeLtxvPreprocessServiceIdentity::checked("a".repeat(64))?,
                mode: AtomicU8::new(ServiceMode::Success as u8),
                calls: AtomicU64::new(0),
                compression: AtomicU64::new(0),
            }))
        }

        fn set_mode(&self, mode: ServiceMode) {
            self.mode.store(mode as u8, Ordering::Release);
        }
    }

    impl NativeLtxvPreprocessService for TestService {
        fn identity(&self) -> &NativeLtxvPreprocessServiceIdentity {
            &self.identity
        }

        fn preprocess_image(
            &self,
            image: &ImageTensor,
            compression: u8,
            context: &ExecutionContext<'_>,
        ) -> BoxFuture<'static, Result<ImageTensor, NativeLtxvPreprocessServiceError>> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            self.compression
                .store(u64::from(compression), Ordering::Release);
            let output = image.clone();
            let result = match self.mode.load(Ordering::Acquire) {
                value if value == ServiceMode::Success as u8 => Ok(output),
                value if value == ServiceMode::Busy as u8 => {
                    Err(NativeLtxvPreprocessServiceError::Busy)
                }
                value if value == ServiceMode::ResourceExhausted as u8 => {
                    Err(NativeLtxvPreprocessServiceError::ResourceExhausted)
                }
                value if value == ServiceMode::CancelAfterOutput as u8 => {
                    context.cancellation.cancel();
                    Ok(output)
                }
                _ => Err(NativeLtxvPreprocessServiceError::Execution(
                    "invalid test mode".to_owned(),
                )),
            };
            Box::pin(futures::future::ready(result))
        }
    }

    struct Harness {
        store: Arc<TestStore>,
        backend: Arc<sim_tensor::CpuBackend>,
        workspace: CpuWorkspaceAuthority,
        attempt_id: AttemptId,
        node_id: NodeId,
        service: Arc<TestService>,
    }

    impl Harness {
        fn new() -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x37203));
            let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
            Ok(Self {
                store: TestStore::new(attempt_id)?,
                backend: Arc::new(backend),
                workspace,
                attempt_id,
                node_id: NodeId("ltxv-preprocess-test".to_owned()),
                service: TestService::new()?,
            })
        }

        fn image_handle(&self) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            let cancellation = CancellationToken::default();
            let context = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(0)?,
                &cancellation,
            );
            let image = ImageTensor::from_f32(
                &self.backend,
                &context,
                1,
                1,
                1,
                3,
                &[0.25, 0.5, 0.75],
            )?;
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
            let scratch = self.workspace.authorize_workspace(1024 * 1024)?;
            let compute = NativeNodeComputeSession::checked(
                NativeNodeServiceIdentity::checked(
                    Uuid::from_u128(0x37204),
                    self.attempt_id,
                    self.node_id.clone(),
                )?,
                self.backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            let services = NativeNodeServices::checked(None, None, Some(compute))?
                .with_ltxv_preprocess(self.service.clone())?;
            Ok(NativeNodeContext::new_with_services(
                PromptId(Uuid::from_u128(0x37205)),
                self.attempt_id,
                self.node_id.clone(),
                cancellation,
                scratch,
                self.store.clone(),
                services,
            )?)
        }

        fn inputs(
            &self,
            image: NativeOpaqueHandle,
            compression: NativePrimitive,
        ) -> BTreeMap<String, NativeValue> {
            BTreeMap::from([
                ("image".to_owned(), NativeValue::Handle { value: image }),
                (
                    "img_compression".to_owned(),
                    NativeValue::Primitive { value: compression },
                ),
            ])
        }
    }

    fn executable() -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable { node, .. } => Some(node),
                _ => None,
            })
            .ok_or_else(|| "LTXVPreprocess executable binding is absent".into())
    }

    #[test]
    fn ltxv_preprocess_descriptor_and_fixture_match_source() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(fixture["feature_id"], FEATURE_ID);
        assert_eq!(fixture["source"]["sha256"], "d3c0a82b42808216b08fe9ca97e055f2a3793083755168a292cf8c076539d12d");
        let binding = native_node_binding()?;
        binding.validate()?;
        let descriptor = binding.descriptor();
        assert_eq!(descriptor.class_type, CLASS_TYPE);
        assert_eq!(descriptor.inputs[0].name, "image");
        assert_eq!(descriptor.inputs[1].name, "img_compression");
        assert_eq!(descriptor.outputs[0].name, "output_image");
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        let schema = descriptor.source_schema.as_ref().ok_or("missing schema")?;
        assert_eq!(
            schema.inputs[1].default,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 35 })
        );
        assert_eq!(
            schema.inputs[1].minimum,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 0 })
        );
        assert_eq!(
            schema.inputs[1].maximum,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 100 })
        );
        Ok(())
    }

    #[test]
    fn ltxv_preprocess_delegates_once_and_publishes_a_fresh_handle()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let input = harness.image_handle()?;
        let inputs = harness.inputs(input.clone(), NativePrimitive::Integer(35));
        let node = executable()?;
        let context = harness.context(CancellationToken::default())?;
        assert_eq!(
            node.cache_dependencies(&context, &inputs)?
                .artifact_digests
                .get(SERVICE_ARTIFACT)
                .map(String::as_str),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        let outcome = futures::executor::block_on(node.execute(context, inputs))?;
        let NativeNodeOutcome::Values { outputs, ui, effects } = outcome else {
            return Err("LTXVPreprocess did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        let Some(NativeValue::Handle { value: output }) = outputs.first() else {
            return Err("LTXVPreprocess output handle is absent".into());
        };
        assert_ne!(output.identifier(), input.identifier());
        assert_eq!(harness.service.calls.load(Ordering::Acquire), 1);
        assert_eq!(harness.service.compression.load(Ordering::Acquire), 35);
        assert_eq!(harness.store.count()?, 2);
        Ok(())
    }

    #[test]
    fn ltxv_preprocess_rejects_bounds_and_maps_service_failures_atomically()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let input = harness.image_handle()?;
        let node = executable()?;
        for compression in [NativePrimitive::Integer(-1), NativePrimitive::UnsignedInteger(101)] {
            let error = futures::executor::block_on(node.execute(
                harness.context(CancellationToken::default())?,
                harness.inputs(input.clone(), compression),
            ))
            .expect_err("invalid compression must fail");
            assert_eq!(error.code, "invalid_node_inputs");
        }
        assert_eq!(harness.service.calls.load(Ordering::Acquire), 0);
        for (mode, code) in [
            (ServiceMode::Busy, "ltxv_preprocess_busy"),
            (
                ServiceMode::ResourceExhausted,
                "ltxv_preprocess_resource_exhausted",
            ),
        ] {
            harness.service.set_mode(mode);
            let error = futures::executor::block_on(node.execute(
                harness.context(CancellationToken::default())?,
                harness.inputs(input.clone(), NativePrimitive::Integer(100)),
            ))
            .expect_err("service failure must not publish");
            assert_eq!(error.code, code);
            assert!(error.retryable);
            assert_eq!(harness.store.count()?, 1);
        }
        harness.service.set_mode(ServiceMode::CancelAfterOutput);
        let error = futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            harness.inputs(input.clone(), NativePrimitive::Integer(0)),
        ))
        .expect_err("late cancellation must discard output");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.count()?, 1);

        harness.service.set_mode(ServiceMode::Success);
        futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            harness.inputs(input, NativePrimitive::UnsignedInteger(1)),
        ))?;
        assert_eq!(harness.store.count()?, 2);
        Ok(())
    }
}
