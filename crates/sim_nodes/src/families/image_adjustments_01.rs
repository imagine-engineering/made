use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativeStoredPayload, NativeTypeUnion,
    NativeValue, NativeValueType, built_in_source_schema,
};
use sim_tensor::{
    BinaryOperation, ImageTensor, NativeTensorPayload, NativeTensorRole, Scalar, ScalarSide,
    Tensor, TensorBackend, TensorError,
};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["AdjustBrightness", "AdjustContrast"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const SOURCE_VERSION: &str = "source-3b27465f-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Adjustment {
    Brightness,
    Contrast,
}

impl Adjustment {
    const fn feature_id(self) -> &'static str {
        match self {
            Self::Brightness => "COMFY-NODE-0004",
            Self::Contrast => "COMFY-NODE-0005",
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::Brightness => "AdjustBrightness",
            Self::Contrast => "AdjustContrast",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::Brightness => "Adjust Brightness",
            Self::Contrast => "Adjust Contrast",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Brightness => "Adjust the brightness of an image.",
            Self::Contrast => "Adjust the contrast of an image.",
        }
    }

    const fn search_alias(self) -> &'static str {
        match self {
            Self::Brightness => "brightness",
            Self::Contrast => "contrast",
        }
    }

    fn implementation_version(self) -> String {
        format!("{}-{}", SOURCE_VERSION, self.class_type())
    }

    fn cache_change_token(self) -> String {
        format!(
            "{}-input-image-factor-identity",
            self.implementation_version()
        )
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [Adjustment::Brightness, Adjustment::Contrast]
        .into_iter()
        .map(native_node_binding)
        .collect()
}

fn native_node_binding(
    adjustment: Adjustment,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let image_type = image_type()?;
    let source_schema = built_in_source_schema(adjustment.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["images".to_owned(), "factor".to_owned()],
            &[],
            &["images".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let implementation_version = adjustment.implementation_version();
    Ok(NativeNodeBinding::Executable {
        feature_id: adjustment.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: adjustment.class_type().to_owned(),
            implementation_version: implementation_version.clone(),
            source_schema: Some(source_schema),
            inputs: vec![
                NativeInputDescriptor {
                    name: "images".to_owned(),
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
                    name: "factor".to_owned(),
                    accepted_types: NativeTypeUnion::new([NativeValueType::Primitive(
                        crate::NativePrimitiveType::Number,
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
                name: "images".to_owned(),
                produced_type: NativeValueType::Handle(image_type),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: adjustment.display_name().to_owned(),
            category: "image/adjustments".to_owned(),
            description: adjustment.description().to_owned(),
            output_names: vec!["images".to_owned()],
            search_aliases: vec![adjustment.search_alias().to_owned()],
            is_deprecated: false,
            is_experimental: true,
        },
        node: Arc::new(ImageAdjustmentNode {
            adjustment,
            implementation_version,
        }),
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

#[derive(Debug)]
struct ImageAdjustmentNode {
    adjustment: Adjustment,
    implementation_version: String,
}

impl NativeNode for ImageAdjustmentNode {
    fn class_type(&self) -> &str {
        self.adjustment.class_type()
    }

    fn implementation_version(&self) -> &str {
        &self.implementation_version
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context, self.adjustment)?;
        parse_inputs(available_inputs, self.adjustment)?;
        Ok(BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        parse_inputs(inputs, self.adjustment)?;
        Ok(self.adjustment.cache_change_token())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, self.adjustment)?;
        parse_inputs(inputs, self.adjustment)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, self.adjustment)?;
            let (input_handle, factor) = parse_inputs(&inputs, self.adjustment)?;
            let expected_type = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
            let resolved = context
                .handle_store()
                .resolve(input_handle, &expected_type, &context.cancellation)
                .map_err(|error| handle_failure(self.adjustment, error))?;
            let NativeStoredPayload::Tensor(input_payload) = resolved.as_ref() else {
                return Err(image_payload_failure(
                    self.adjustment,
                    "resolved IMAGE handle does not contain a tensor payload",
                ));
            };
            if input_payload.role() != NativeTensorRole::Image {
                return Err(image_payload_failure(
                    self.adjustment,
                    "resolved IMAGE handle has the wrong tensor role",
                ));
            }
            let input_image = input_payload.image().ok_or_else(|| {
                image_payload_failure(
                    self.adjustment,
                    "resolved IMAGE handle has no canonical ImageTensor",
                )
            })?;
            let compute = context
                .compute_session()
                .map_err(|error| NativeNodeFailure {
                    code: "native_compute_unavailable".to_owned(),
                    message: format!(
                        "{} compute session is unavailable: {error}",
                        self.class_type()
                    ),
                    kind: NativeNodeFailureKind::Failure,
                    retryable: true,
                })?;
            let execution_context =
                compute
                    .execution_context(&context)
                    .map_err(|error| NativeNodeFailure {
                        code: "native_compute_unavailable".to_owned(),
                        message: format!(
                            "{} compute session is incompatible: {error}",
                            self.class_type()
                        ),
                        kind: NativeNodeFailureKind::Failure,
                        retryable: true,
                    })?;
            let output_tensor = adjust_image(
                self.adjustment,
                input_image.tensor(),
                factor,
                compute.backend(),
                &execution_context,
            )
            .map_err(|error| tensor_failure(self.adjustment, error))?;
            let output_image = ImageTensor::from_tensor(output_tensor)
                .map_err(|error| tensor_failure(self.adjustment, error))?;
            let output_payload =
                NativeTensorPayload::from_image(NativeTensorRole::Image, output_image)
                    .map_err(|error| image_payload_failure(self.adjustment, error.to_string()))?;
            check_cancellation(&context, self.adjustment)?;
            let output_handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Tensor(Arc::new(output_payload)),
                    &context.cancellation,
                )
                .map_err(|error| handle_failure(self.adjustment, error))?;
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
    adjustment: Adjustment,
) -> Result<(&NativeOpaqueHandle, f32), NativeNodeFailure> {
    if inputs.len() != 2 {
        return Err(invalid_inputs(format!(
            "{} requires exactly images and factor inputs",
            adjustment.class_type()
        )));
    }
    let Some(NativeValue::Handle { value: image }) = inputs.get("images") else {
        return Err(invalid_inputs(format!(
            "{} images input must be an IMAGE handle",
            adjustment.class_type()
        )));
    };
    if image.handle_type().kind != NativeHandleKind::Image || image.handle_type().type_id != "IMAGE"
    {
        return Err(invalid_inputs(format!(
            "{} images input must be an exact IMAGE handle",
            adjustment.class_type()
        )));
    }
    let Some(NativeValue::Primitive {
        value: crate::NativePrimitive::Number(factor),
    }) = inputs.get("factor")
    else {
        return Err(invalid_inputs(format!(
            "{} factor input must be a FLOAT",
            adjustment.class_type()
        )));
    };
    if !factor.is_finite() || !(0.0..=2.0).contains(factor) {
        return Err(invalid_inputs(format!(
            "{} factor must be finite and within 0.0 through 2.0",
            adjustment.class_type()
        )));
    }
    Ok((image, *factor as f32))
}

fn adjust_image(
    adjustment: Adjustment,
    input: &Tensor,
    factor: f32,
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, TensorError> {
    let descriptor = input.descriptor().clone();
    let tensor = match adjustment {
        Adjustment::Brightness => apply_scalar(
            backend,
            BinaryOperation::Multiply,
            input,
            factor,
            descriptor.clone(),
            context,
        )?,
        Adjustment::Contrast => {
            let tensor = apply_scalar(
                backend,
                BinaryOperation::Subtract,
                input,
                0.5,
                descriptor.clone(),
                context,
            )?;
            let tensor = apply_scalar(
                backend,
                BinaryOperation::Multiply,
                &tensor,
                factor,
                descriptor.clone(),
                context,
            )?;
            apply_scalar(
                backend,
                BinaryOperation::Add,
                &tensor,
                0.5,
                descriptor.clone(),
                context,
            )?
        }
    };
    let tensor = apply_scalar(
        backend,
        BinaryOperation::Maximum,
        &tensor,
        0.0,
        descriptor.clone(),
        context,
    )?;
    apply_scalar(
        backend,
        BinaryOperation::Minimum,
        &tensor,
        1.0,
        descriptor,
        context,
    )
}

fn apply_scalar(
    backend: &sim_tensor::CpuBackend,
    operation: BinaryOperation,
    input: &Tensor,
    scalar: f32,
    output: sim_tensor::TensorDescriptor,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, TensorError> {
    backend
        .binary_scalar(
            operation,
            input,
            Scalar::Float(f64::from(scalar)),
            ScalarSide::Right,
            output,
            context,
        )
        .map(|(tensor, _event)| tensor)
}

fn check_cancellation(
    context: &NativeNodeContext,
    adjustment: Adjustment,
) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(adjustment))
}

fn handle_failure(adjustment: Adjustment, error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(adjustment)
    } else {
        NativeNodeFailure {
            code: "invalid_image_handle".to_owned(),
            message: format!(
                "{} IMAGE handle is not available: {error}",
                adjustment.class_type()
            ),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn tensor_failure(adjustment: Adjustment, error: TensorError) -> NativeNodeFailure {
    if matches!(error, TensorError::Cancelled) {
        interrupted_failure(adjustment)
    } else {
        NativeNodeFailure {
            code: "image_adjustment_failed".to_owned(),
            message: format!(
                "{} native image adjustment failed: {error}",
                adjustment.class_type()
            ),
            kind: NativeNodeFailureKind::Failure,
            retryable: matches!(
                error,
                TensorError::AllocationFailed { .. }
                    | TensorError::DeviceLost { .. }
                    | TensorError::WorkspaceAuthorizationExceeded { .. }
            ),
        }
    }
}

fn image_payload_failure(adjustment: Adjustment, message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_image_payload".to_owned(),
        message: format!("{}: {}", adjustment.class_type(), message.into()),
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

fn interrupted_failure(adjustment: Adjustment) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: format!("{} execution was interrupted", adjustment.class_type()),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativeNodeComputeSession, NativeNodeServices,
        NativePrimitive, NativeResolvedPayload, NativeResolvedPayloadRetention,
    };
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::{Value, json};
    use std::{
        error::Error,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-adjustments-comfy-node-0004/fixture.json"
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
        fn new(
            store_id: u128,
            generation_id: u128,
            attempt_id: AttemptId,
        ) -> Result<Arc<Self>, Box<dyn Error>> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(store_id),
                    Uuid::from_u128(generation_id),
                )?,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                values: Mutex::new(BTreeMap::new()),
            }))
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
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
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
            let identifier = format!(
                "image-{}",
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

    struct Harness {
        store: Arc<TestStore>,
        backend: Arc<sim_tensor::CpuBackend>,
        workspace: CpuWorkspaceAuthority,
        attempt_id: AttemptId,
        node_id: NodeId,
    }

    impl Harness {
        fn new(store_id: u128, generation_id: u128) -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x38901));
            let (backend, workspace) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
            Ok(Self {
                store: TestStore::new(store_id, generation_id, attempt_id)?,
                backend: Arc::new(backend),
                workspace,
                attempt_id,
                node_id: NodeId("image-adjustments-test".to_owned()),
            })
        }

        fn image_handle(&self, values: &[f32]) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            let cancellation = CancellationToken::default();
            let context = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(0)?,
                &cancellation,
            );
            let image = ImageTensor::from_f32(&self.backend, &context, 1, 1, 2, 3, values)?;
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
            let scratch = self.workspace.authorize_workspace(0)?;
            let identity = crate::NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x38902),
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
                PromptId(Uuid::from_u128(0x38903)),
                self.attempt_id,
                self.node_id.clone(),
                cancellation,
                scratch,
                self.store.clone(),
                NativeNodeServices::checked(None, None, Some(compute))?,
            )?)
        }

        fn values(&self, handle: NativeOpaqueHandle, factor: f64) -> BTreeMap<String, NativeValue> {
            BTreeMap::from([
                ("images".to_owned(), NativeValue::Handle { value: handle }),
                (
                    "factor".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(factor),
                    },
                ),
            ])
        }

        fn output_values(&self, outcome: NativeNodeOutcome) -> Result<Vec<f32>, Box<dyn Error>> {
            let NativeNodeOutcome::Values {
                outputs,
                ui,
                effects,
            } = outcome
            else {
                return Err("image adjustment did not return values".into());
            };
            assert!(ui.is_none());
            assert!(effects.is_empty());
            let Some(NativeValue::Handle { value: handle }) = outputs.first() else {
                return Err("image adjustment output handle is absent".into());
            };
            let resolved =
                self.store
                    .resolve(handle, &image_type()?, &CancellationToken::default())?;
            let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
                return Err("image adjustment output payload is not a tensor".into());
            };
            Ok(payload
                .image()
                .ok_or("image adjustment output has no ImageTensor")?
                .as_f32_slice()?
                .to_vec())
        }
    }

    fn executable(
        adjustment: Adjustment,
    ) -> Result<
        (
            NativeNodeDescriptor,
            NativeNodePresentation,
            Arc<dyn NativeNode>,
        ),
        Box<dyn Error>,
    > {
        let binding = native_node_bindings()?
            .into_iter()
            .find(|binding| binding.descriptor().class_type == adjustment.class_type())
            .ok_or("image adjustment binding is absent")?;
        match binding {
            NativeNodeBinding::Executable {
                descriptor,
                presentation,
                node,
                ..
            } => Ok((descriptor, presentation, node)),
            _ => Err("image adjustment binding is not executable".into()),
        }
    }

    #[test]
    fn source_fixture_and_schema_are_exact() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture.pointer("/source/sha256").and_then(Value::as_str),
            Some("3b27465fec391509083bd1837895c09abc489c04d81afae5ffe631abd6a4e772")
        );
        assert_eq!(
            fixture
                .pointer("/source/byte_length")
                .and_then(Value::as_u64),
            Some(59648)
        );
        for adjustment in [Adjustment::Brightness, Adjustment::Contrast] {
            let (descriptor, presentation, node) = executable(adjustment)?;
            assert_eq!(descriptor.class_type, adjustment.class_type());
            assert_eq!(
                descriptor.schema_version,
                NATIVE_NODE_CONTRACT_SCHEMA_VERSION
            );
            assert_eq!(descriptor.inputs.len(), 2);
            assert_eq!(descriptor.inputs[0].name, "images");
            assert_eq!(descriptor.inputs[1].name, "factor");
            assert_eq!(descriptor.outputs.len(), 1);
            assert_eq!(descriptor.outputs[0].name, "images");
            assert!(!descriptor.outputs[0].is_list);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            let schema = descriptor
                .source_schema
                .as_ref()
                .ok_or("source schema is absent")?;
            assert_eq!(
                schema.inputs[1].minimum,
                Some(crate::NativeSchemaValue::FiniteDecimal {
                    value: "0.0".to_owned()
                })
            );
            assert_eq!(
                schema.inputs[1].maximum,
                Some(crate::NativeSchemaValue::FiniteDecimal {
                    value: "2.0".to_owned()
                })
            );
            assert_eq!(presentation.display_name, adjustment.display_name());
            assert_eq!(presentation.category, "image/adjustments");
            assert_eq!(presentation.description, adjustment.description());
            assert_eq!(presentation.search_aliases, [adjustment.search_alias()]);
            assert!(presentation.is_experimental);
            assert_eq!(node.class_type(), adjustment.class_type());
            assert_eq!(
                node.implementation_version(),
                descriptor.implementation_version
            );
            native_node_binding(adjustment)?.validate()?;
        }
        Ok(())
    }

    #[test]
    fn brightness_and_contrast_match_source_equations_and_boundaries() -> Result<(), Box<dyn Error>>
    {
        let input = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.25];
        let cases = [
            (
                Adjustment::Brightness,
                1.5,
                vec![0.0, 0.0, 0.375, 0.75, 1.0, 1.0],
            ),
            (Adjustment::Brightness, 0.0, vec![0.0; 6]),
            (
                Adjustment::Contrast,
                2.0,
                vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            ),
            (Adjustment::Contrast, 0.0, vec![0.5; 6]),
        ];
        for (index, (adjustment, factor, expected)) in cases.into_iter().enumerate() {
            let harness = Harness::new(0x38910 + index as u128, 0x38920 + index as u128)?;
            let handle = harness.image_handle(&input)?;
            let (_, _, node) = executable(adjustment)?;
            let values = harness.values(handle, factor);
            assert!(
                node.demanded_lazy_inputs(
                    &harness.context(CancellationToken::default())?,
                    &values
                )?
                .is_empty()
            );
            assert_eq!(
                node.cache_change_token(&values)?,
                adjustment.cache_change_token()
            );
            assert_eq!(
                node.cache_dependencies(&harness.context(CancellationToken::default())?, &values)?,
                NativeCacheDependencies::default()
            );
            let outcome = futures::executor::block_on(
                node.execute(harness.context(CancellationToken::default())?, values),
            )?;
            assert_eq!(harness.output_values(outcome)?, expected);
        }
        Ok(())
    }

    #[test]
    fn validation_cancellation_and_stale_handles_fail_without_output() -> Result<(), Box<dyn Error>>
    {
        let harness = Harness::new(0x38930, 0x38931)?;
        let handle = harness.image_handle(&[0.0, 0.25, 0.5, 0.75, 1.0, 1.25])?;
        let (_, _, node) = executable(Adjustment::Brightness)?;
        for factor in [-0.01, 2.01, f64::NAN, f64::INFINITY] {
            let error = futures::executor::block_on(node.execute(
                harness.context(CancellationToken::default())?,
                harness.values(handle.clone(), factor),
            ))
            .expect_err("invalid factor must fail");
            assert_eq!(error.code, "invalid_node_inputs");
        }
        let error = futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            BTreeMap::from([(
                "images".to_owned(),
                NativeValue::Handle {
                    value: handle.clone(),
                },
            )]),
        ))
        .expect_err("missing factor must fail");
        assert_eq!(error.code, "invalid_node_inputs");
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = futures::executor::block_on(node.execute(
            harness.context(cancellation)?,
            harness.values(handle.clone(), 1.0),
        ))
        .expect_err("pre-cancelled execution must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);

        let restarted = Harness::new(0x38930, 0x38932)?;
        let error = futures::executor::block_on(node.execute(
            restarted.context(CancellationToken::default())?,
            restarted.values(handle, 1.0),
        ))
        .expect_err("stale worker generation must fail");
        assert_eq!(error.code, "invalid_image_handle");
        Ok(())
    }

    #[test]
    fn persistence_fixture_is_lossless_and_recovery_succeeds_with_fresh_input()
    -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let persistence = fixture
            .pointer("/behavior/persistence")
            .cloned()
            .ok_or("persistence fixture is absent")?;
        assert_eq!(
            serde_json::from_slice::<Value>(&serde_json::to_vec(&persistence)?)?,
            persistence
        );
        assert_eq!(
            persistence.get("nodes"),
            Some(&json!([
                {
                    "class_type": "AdjustBrightness",
                    "inputs": { "images": [7, 0], "factor": 1.5 },
                    "unknown_data": { "source_extension": "nodes_dataset", "preserve": true }
                },
                {
                    "class_type": "AdjustContrast",
                    "inputs": { "images": [8, 0], "factor": 2.0 },
                    "unknown_data": { "source_extension": "nodes_dataset", "preserve": true }
                }
            ]))
        );

        let harness = Harness::new(0x38940, 0x38941)?;
        let handle = harness.image_handle(&[0.0, 0.25, 0.5, 0.75, 1.0, 1.25])?;
        let (_, _, node) = executable(Adjustment::Contrast)?;
        let outcome = futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            harness.values(handle, 1.0),
        ))?;
        assert_eq!(
            harness.output_values(outcome)?,
            vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
        );
        Ok(())
    }
}
