use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativePrimitiveType,
    NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_tensor::{
    DType, DeviceId, NativeTensorPayload, NativeTensorRole, Tensor, TensorDescriptor,
    generated_neural_network_module_03::max_pool_2d_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["SolidMask", "ThresholdMask", "VOIDQuadmaskPreprocess"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const SOLID_FEATURE_ID: &str = "COMFY-NODE-0625";
const SOLID_CLASS_TYPE: &str = "SolidMask";
const SOLID_IMPLEMENTATION_VERSION: &str = "source-9ff6c44f-v1";
const SOLID_CACHE_TOKEN: &str = "solid-mask-source-9ff6c44f-v1";
const THRESHOLD_FEATURE_ID: &str = "COMFY-NODE-0675";
const THRESHOLD_CLASS_TYPE: &str = "ThresholdMask";
const THRESHOLD_IMPLEMENTATION_VERSION: &str = "source-9ff6c44f-v1";
const THRESHOLD_CACHE_TOKEN: &str = "threshold-mask-source-9ff6c44f-v1";
const VOID_FEATURE_ID: &str = "COMFY-NODE-0742";
const VOID_CLASS_TYPE: &str = "VOIDQuadmaskPreprocess";
const VOID_IMPLEMENTATION_VERSION: &str = "source-b80cd0b5-v1";
const VOID_CACHE_TOKEN: &str = "void-quadmask-preprocess-source-b80cd0b5-v1";
const MAX_RESOLUTION: u64 = 16_384;

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    let mask_type = mask_type()?;
    let solid_source = built_in_source_schema(SOLID_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["value".to_owned(), "width".to_owned(), "height".to_owned()],
            &[],
            &["mask".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let threshold_source = built_in_source_schema(THRESHOLD_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["mask".to_owned(), "value".to_owned()],
            &[],
            &["mask".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let void_source = built_in_source_schema(VOID_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["mask".to_owned(), "dilate_width".to_owned()],
            &[],
            &["quadmask".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;

    Ok(vec![
        executable_binding(
            SOLID_FEATURE_ID,
            SOLID_CLASS_TYPE,
            SOLID_IMPLEMENTATION_VERSION,
            solid_source,
            vec![
                input(
                    "value",
                    NativeValueType::Primitive(NativePrimitiveType::Number),
                    true,
                )?,
                input(
                    "width",
                    NativeValueType::Primitive(NativePrimitiveType::Integer),
                    true,
                )?,
                input(
                    "height",
                    NativeValueType::Primitive(NativePrimitiveType::Integer),
                    true,
                )?,
            ],
            "Create Solid Mask",
            Vec::new(),
            Arc::new(SolidMask),
            mask_type.clone(),
        ),
        executable_binding(
            THRESHOLD_FEATURE_ID,
            THRESHOLD_CLASS_TYPE,
            THRESHOLD_IMPLEMENTATION_VERSION,
            threshold_source,
            vec![
                input("mask", NativeValueType::Handle(mask_type.clone()), false)?,
                input(
                    "value",
                    NativeValueType::Primitive(NativePrimitiveType::Number),
                    true,
                )?,
            ],
            "Threshold Mask",
            vec!["binary mask".to_owned()],
            Arc::new(ThresholdMask),
            mask_type.clone(),
        ),
        executable_binding(
            VOID_FEATURE_ID,
            VOID_CLASS_TYPE,
            VOID_IMPLEMENTATION_VERSION,
            void_source,
            vec![
                input("mask", NativeValueType::Handle(mask_type.clone()), false)?,
                input(
                    "dilate_width",
                    NativeValueType::Primitive(NativePrimitiveType::Integer),
                    true,
                )?,
            ],
            "VOID Quadmask Preprocessor",
            Vec::new(),
            Arc::new(VoidQuadmaskPreprocess),
            mask_type,
        ),
    ])
}

#[allow(clippy::too_many_arguments)]
fn executable_binding(
    feature_id: &str,
    class_type: &str,
    implementation_version: &str,
    source_schema: crate::NativeDescriptorSchemaMetadata,
    inputs: Vec<NativeInputDescriptor>,
    display_name: &str,
    search_aliases: Vec<String>,
    node: Arc<dyn NativeNode>,
    mask_type: NativeHandleType,
) -> NativeNodeBinding {
    NativeNodeBinding::Executable {
        feature_id: feature_id.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: implementation_version.to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: if class_type == VOID_CLASS_TYPE {
                    "quadmask".to_owned()
                } else {
                    "mask".to_owned()
                },
                produced_type: NativeValueType::Handle(mask_type),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: display_name.to_owned(),
            category: "image/mask".to_owned(),
            description: String::new(),
            output_names: vec![if class_type == VOID_CLASS_TYPE {
                "quadmask".to_owned()
            } else {
                "mask".to_owned()
            }],
            search_aliases,
            is_deprecated: false,
            is_experimental: false,
        },
        node,
    }
}

fn input(
    name: &str,
    value_type: NativeValueType,
    allows_literal: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([value_type])?,
        required: true,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal,
    })
}

fn mask_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Mask, "MASK")
}

#[derive(Debug)]
struct SolidMask;

impl NativeNode for SolidMask {
    fn class_type(&self) -> &str {
        SOLID_CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        SOLID_IMPLEMENTATION_VERSION
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        solid_inputs(inputs)?;
        Ok(SOLID_CACHE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, SOLID_CLASS_TYPE)?;
        solid_inputs(inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, SOLID_CLASS_TYPE)?;
            let (value, width, height) = solid_inputs(&inputs)?;
            let element_count = width
                .checked_mul(height)
                .ok_or_else(|| invalid_inputs("solid mask dimensions overflowed"))?;
            let element_count = usize::try_from(element_count)
                .map_err(|_| invalid_inputs("solid mask dimensions are too large"))?;
            let values = vec![value; element_count];
            publish_values(&context, SOLID_CLASS_TYPE, vec![1, height, width], &values)
        })
    }
}

#[derive(Debug)]
struct ThresholdMask;

impl NativeNode for ThresholdMask {
    fn class_type(&self) -> &str {
        THRESHOLD_CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        THRESHOLD_IMPLEMENTATION_VERSION
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        threshold_inputs(inputs)?;
        Ok(THRESHOLD_CACHE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, THRESHOLD_CLASS_TYPE)?;
        threshold_inputs(inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, THRESHOLD_CLASS_TYPE)?;
            let (handle, threshold) = threshold_inputs(&inputs)?;
            let resolved = resolve_mask(&context, &handle, THRESHOLD_CLASS_TYPE)?;
            let payload = require_mask_payload(&resolved)?;
            let mut values = tensor_f32_values(payload.tensor(), &context, THRESHOLD_CLASS_TYPE)?;
            for (index, value) in values.iter_mut().enumerate() {
                periodic_cancellation(&context, THRESHOLD_CLASS_TYPE, index)?;
                *value = if *value > threshold { 1.0 } else { 0.0 };
            }
            let shape = payload.tensor().descriptor().shape().to_vec();
            drop(resolved);
            publish_values(&context, THRESHOLD_CLASS_TYPE, shape, &values)
        })
    }
}

#[derive(Debug)]
struct VoidQuadmaskPreprocess;

impl NativeNode for VoidQuadmaskPreprocess {
    fn class_type(&self) -> &str {
        VOID_CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        VOID_IMPLEMENTATION_VERSION
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        void_inputs(inputs)?;
        Ok(VOID_CACHE_TOKEN.to_owned())
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, VOID_CLASS_TYPE)?;
        void_inputs(inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, VOID_CLASS_TYPE)?;
            let (handle, dilate_width) = void_inputs(&inputs)?;
            let resolved = resolve_mask(&context, &handle, VOID_CLASS_TYPE)?;
            let payload = require_mask_payload(&resolved)?;
            let shape = payload.tensor().descriptor().shape().to_vec();
            let values = tensor_f32_values(payload.tensor(), &context, VOID_CLASS_TYPE)?;
            let values = void_quadmask_values(&context, &shape, values, dilate_width)?;
            drop(resolved);
            publish_values(&context, VOID_CLASS_TYPE, shape, &values)
        })
    }
}

fn solid_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(f32, u64, u64), NativeNodeFailure> {
    if inputs.len() != 3 {
        return Err(invalid_inputs(
            "SolidMask requires exactly value, width, and height",
        ));
    }
    Ok((
        bounded_number(inputs.get("value"), "value", 0.0, 1.0)?,
        bounded_integer(inputs.get("width"), "width", 1, MAX_RESOLUTION)?,
        bounded_integer(inputs.get("height"), "height", 1, MAX_RESOLUTION)?,
    ))
}

fn threshold_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(NativeOpaqueHandle, f32), NativeNodeFailure> {
    if inputs.len() != 2 {
        return Err(invalid_inputs(
            "ThresholdMask requires exactly mask and value",
        ));
    }
    Ok((
        exact_mask_handle(inputs.get("mask"), "mask")?.clone(),
        bounded_number(inputs.get("value"), "value", 0.0, 1.0)?,
    ))
}

fn void_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(NativeOpaqueHandle, u64), NativeNodeFailure> {
    if inputs.len() != 2 {
        return Err(invalid_inputs(
            "VOIDQuadmaskPreprocess requires exactly mask and dilate_width",
        ));
    }
    Ok((
        exact_mask_handle(inputs.get("mask"), "mask")?.clone(),
        bounded_integer(inputs.get("dilate_width"), "dilate_width", 0, 50)?,
    ))
}

fn exact_mask_handle<'a>(
    value: Option<&'a NativeValue>,
    name: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(invalid_inputs(format!("{name} must be a MASK handle")));
    };
    if value.handle_type().kind != NativeHandleKind::Mask || value.handle_type().type_id != "MASK" {
        return Err(invalid_inputs(format!(
            "{name} must be an exact MASK handle"
        )));
    }
    Ok(value)
}

fn bounded_number(
    value: Option<&NativeValue>,
    name: &str,
    minimum: f64,
    maximum: f64,
) -> Result<f32, NativeNodeFailure> {
    let Some(NativeValue::Primitive {
        value: NativePrimitive::Number(value),
    }) = value
    else {
        return Err(invalid_inputs(format!("{name} must be a FLOAT")));
    };
    if !value.is_finite() || !(minimum..=maximum).contains(value) {
        return Err(invalid_inputs(format!(
            "{name} must be finite and between {minimum} and {maximum}"
        )));
    }
    Ok(*value as f32)
}

fn bounded_integer(
    value: Option<&NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let integer = match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => *value,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value)
            .map_err(|_| invalid_inputs(format!("{name} must be non-negative")))?,
        _ => return Err(invalid_inputs(format!("{name} must be an INT"))),
    };
    if !(minimum..=maximum).contains(&integer) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(integer)
}

fn resolve_mask(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
    class_type: &str,
) -> Result<crate::NativeResolvedPayload, NativeNodeFailure> {
    let expected_type = mask_type().map_err(|error| invalid_inputs(error.to_string()))?;
    context
        .handle_store()
        .resolve(handle, &expected_type, &context.cancellation)
        .map_err(|error| handle_failure(error, class_type))
}

fn require_mask_payload(
    payload: &NativeStoredPayload,
) -> Result<&NativeTensorPayload, NativeNodeFailure> {
    let NativeStoredPayload::Tensor(payload) = payload else {
        return Err(native_failure(
            "MASK handle did not resolve to tensor storage",
        ));
    };
    if payload.role() != NativeTensorRole::Mask {
        return Err(native_failure(
            "MASK handle resolved to the wrong tensor role",
        ));
    }
    Ok(payload)
}

fn tensor_f32_values(
    tensor: &Tensor,
    context: &NativeNodeContext,
    class_type: &str,
) -> Result<Vec<f32>, NativeNodeFailure> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(native_failure("MASK tensor storage must be F32"));
    }
    let count = tensor
        .descriptor()
        .element_count()
        .map_err(native_failure)?;
    let count = usize::try_from(count).map_err(|_| native_failure("MASK is too large"))?;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        periodic_cancellation(context, class_type, index)?;
        let bytes = tensor
            .linear_element_bytes(u64::try_from(index).map_err(native_failure)?)
            .map_err(native_failure)?;
        let bytes: [u8; 4] = bytes
            .try_into()
            .map_err(|_| native_failure("MASK tensor contained an invalid F32 element"))?;
        values.push(f32::from_ne_bytes(bytes));
    }
    Ok(values)
}

fn void_quadmask_values(
    context: &NativeNodeContext,
    shape: &[u64],
    mut values: Vec<f32>,
    dilate_width: u64,
) -> Result<Vec<f32>, NativeNodeFailure> {
    if values.is_empty() {
        return Err(native_failure(
            "VOID quadmask global maximum is undefined for an empty MASK",
        ));
    }
    let maximum = if values.iter().any(|value| value.is_nan()) {
        f32::NAN
    } else {
        values.iter().copied().reduce(f32::max).unwrap_or(f32::NAN)
    };
    if maximum <= 1.0 {
        for (index, value) in values.iter_mut().enumerate() {
            periodic_cancellation(context, VOID_CLASS_TYPE, index)?;
            *value *= 255.0;
        }
    }

    if dilate_width > 0 {
        let [batch, height, width] = shape else {
            return Err(native_failure("VOID quadmask input must have rank three"));
        };
        let mut binary = Vec::with_capacity(values.len());
        for (index, value) in values.iter().enumerate() {
            periodic_cancellation(context, VOID_CLASS_TYPE, index)?;
            binary.push(if *value < 128.0 { 1.0 } else { 0.0 });
        }
        let kernel = dilate_width
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_inputs("dilate_width overflowed"))?;
        let kernel = usize::try_from(kernel).map_err(native_failure)?;
        let padding = usize::try_from(dilate_width).map_err(native_failure)?;
        let pooled = max_pool_2d_with_context_exact_native(
            &binary,
            &[
                usize::try_from(*batch).map_err(native_failure)?,
                1,
                usize::try_from(*height).map_err(native_failure)?,
                usize::try_from(*width).map_err(native_failure)?,
            ],
            [kernel, kernel],
            [1, 1],
            [padding, padding],
            [1, 1],
            false,
            DeviceId::CPU,
            &context
                .compute_session()
                .map_err(compute_failure)?
                .execution_context(context)
                .map_err(compute_failure)?,
        )
        .map_err(native_failure)?;
        if pooled.values.len() != values.len() {
            return Err(native_failure(
                "VOID quadmask dilation changed tensor shape",
            ));
        }
        for (index, (value, dilated)) in values.iter_mut().zip(pooled.values).enumerate() {
            periodic_cancellation(context, VOID_CLASS_TYPE, index)?;
            if dilated > 0.5 {
                *value = 0.0;
            }
        }
    }

    for (index, value) in values.iter_mut().enumerate() {
        periodic_cancellation(context, VOID_CLASS_TYPE, index)?;
        let quantized = if *value <= 31.0 {
            0.0
        } else if *value <= 95.0 {
            63.0
        } else if *value <= 191.0 {
            127.0
        } else if *value > 191.0 {
            255.0
        } else {
            *value
        };
        *value = (255.0 - quantized) / 255.0;
    }
    Ok(values)
}

fn publish_values(
    context: &NativeNodeContext,
    class_type: &str,
    shape: Vec<u64>,
    values: &[f32],
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check_cancellation(context, class_type)?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let descriptor =
        TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, execution.stream)
            .map_err(native_failure)?;
    let tensor = compute
        .backend()
        .upload_f32(descriptor, values, &execution)
        .map(|(tensor, _)| tensor)
        .map_err(native_failure)?;
    let payload =
        NativeTensorPayload::from_tensor(NativeTensorRole::Mask, tensor).map_err(native_failure)?;
    let handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(|error| handle_failure(error, class_type))?;
    values_outcome(handle)
}

fn values_outcome(handle: NativeOpaqueHandle) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let outcome = NativeNodeOutcome::Values {
        outputs: vec![NativeValue::Handle { value: handle }],
        ui: None,
        effects: Vec::new(),
    };
    outcome
        .validate()
        .map_err(|error| invalid_inputs(error.to_string()))?;
    Ok(outcome)
}

fn periodic_cancellation(
    context: &NativeNodeContext,
    class_type: &str,
    index: usize,
) -> Result<(), NativeNodeFailure> {
    if index.is_multiple_of(4_096) {
        check_cancellation(context, class_type)?;
    }
    Ok(())
}

fn check_cancellation(
    context: &NativeNodeContext,
    class_type: &str,
) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(class_type))
}

fn handle_failure(error: NativeHandleStoreError, class_type: &str) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(class_type)
    } else {
        NativeNodeFailure {
            code: "invalid_mask_handle".to_owned(),
            message: format!("{class_type} MASK handle is not available: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(error: NativeNodeContractError) -> NativeNodeFailure {
    native_failure(error)
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn native_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("cancelled") {
        return NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message,
            kind: NativeNodeFailureKind::Interrupted,
            retryable: false,
        };
    }
    NativeNodeFailure {
        code: "native_image_mask_failed".to_owned(),
        message,
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure(class_type: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: format!("{class_type} execution was interrupted"),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativeNodeComputeSession,
        NativeNodeServiceIdentity, NativeNodeServices, NativeResolvedPayload,
        NativeResolvedPayloadRetention,
    };
    use sim_tensor::{CpuBackend, CpuWorkspaceAuthority, StreamId};
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::Value;
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-mask-comfy-node-0625/fixture.json"
    ));

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_generation: AtomicU64,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(attempt_id: AttemptId) -> Result<Arc<Self>, NativeNodeContractError> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(0x40301),
                    Uuid::from_u128(0x40302),
                )?,
                attempt_id,
                next_generation: AtomicU64::new(1),
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
            if handle.store_identity() != self.identity {
                return Err(NativeHandleStoreError::WrongStore);
            }
            if handle.handle_type() != expected_type {
                return Err(NativeHandleStoreError::WrongType {
                    expected: expected_type.type_id.clone(),
                    actual: handle.handle_type().type_id.clone(),
                });
            }
            let values = self.values.lock().map_err(|_| {
                NativeHandleStoreError::Rejected("test store is poisoned".to_owned())
            })?;
            let payload = values
                .get(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            if payload.digest_sha256() != handle.digest_sha256().unwrap_or_default() {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            Ok(NativeResolvedPayload::checked(
                payload.clone(),
                Arc::new(TestRetention),
            )?)
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
            let generation = self.next_generation.fetch_add(1, Ordering::AcqRel);
            let identifier = format!("mask-{generation}");
            let handle = NativeOpaqueHandle::new(
                handle_type,
                self.identity,
                identifier.clone(),
                generation,
                Some(digest),
            )?;
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .insert(identifier, Arc::new(payload));
            Ok(handle)
        }

        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            let removed = self
                .values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .remove(handle.identifier());
            if removed.is_none() {
                return Err(NativeHandleStoreError::Missing(
                    handle.identifier().to_owned(),
                ));
            }
            Ok(())
        }
    }

    struct Harness {
        backend: Arc<CpuBackend>,
        store: Arc<TestStore>,
        context: NativeNodeContext,
    }

    impl Harness {
        fn new(cancellation: CancellationToken) -> Result<Self, Box<dyn std::error::Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x40303));
            let node_id = NodeId("image-mask-test".to_owned());
            let store = TestStore::new(attempt_id)?;
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
            let backend = Arc::new(backend);
            let scratch = authority.authorize_workspace(16 * 1024 * 1024)?;
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x40304),
                attempt_id,
                node_id.clone(),
            )?;
            let compute = NativeNodeComputeSession::checked(
                identity,
                backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            let context = NativeNodeContext::new_with_services(
                PromptId(Uuid::from_u128(0x40305)),
                attempt_id,
                node_id,
                cancellation,
                scratch,
                store.clone(),
                NativeNodeServices::checked(None, None, Some(compute))?,
            )?;
            Ok(Self {
                backend,
                store,
                context,
            })
        }

        fn publish_mask(
            &self,
            shape: Vec<u64>,
            values: &[f32],
        ) -> Result<NativeOpaqueHandle, Box<dyn std::error::Error>> {
            let compute = self.context.compute_session()?;
            let execution = compute.execution_context(&self.context)?;
            let descriptor =
                TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, execution.stream)?;
            let tensor = self.backend.upload_f32(descriptor, values, &execution)?.0;
            let payload = NativeTensorPayload::from_tensor(NativeTensorRole::Mask, tensor)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(payload)),
                &CancellationToken::default(),
            )?)
        }

        fn output_values(
            &self,
            outcome: NativeNodeOutcome,
        ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let NativeNodeOutcome::Values { outputs, .. } = outcome else {
                return Err("node did not return values".into());
            };
            let Some(NativeValue::Handle { value }) = outputs.first() else {
                return Err("node did not return a MASK handle".into());
            };
            let payload =
                self.store
                    .resolve(value, &mask_type()?, &CancellationToken::default())?;
            let tensor = require_mask_payload(&payload)?.tensor();
            Ok(tensor_f32_values(tensor, &self.context, "test")?)
        }
    }

    fn node(class_type: &str) -> Result<Arc<dyn NativeNode>, Box<dyn std::error::Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == class_type => Some(node),
                _ => None,
            })
            .ok_or_else(|| format!("{class_type} binding is absent").into())
    }

    #[test]
    fn descriptors_and_fixture_preserve_source_contracts() -> Result<(), Box<dyn std::error::Error>>
    {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 3);
        for binding in bindings {
            let NativeNodeBinding::Executable { descriptor, .. } = binding else {
                return Err("binding was not executable".into());
            };
            descriptor.validate_exact_schema_v2()?;
        }
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture["stable_task_id"],
            "comfy-parity-native-nodes-image-mask-comfy-node-0625"
        );
        Ok(())
    }

    #[test]
    fn solid_and_threshold_execute_exact_source_boundaries()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new(CancellationToken::default())?;
        let solid = futures::executor::block_on(node(SOLID_CLASS_TYPE)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                (
                    "value".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(0.5),
                    },
                ),
                (
                    "width".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(3),
                    },
                ),
                (
                    "height".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(2),
                    },
                ),
            ]),
        ))?;
        assert_eq!(harness.output_values(solid)?, vec![0.5; 6]);

        let mask = harness.publish_mask(vec![1, 1, 4], &[0.49, 0.5, 0.51, f32::NAN])?;
        let threshold = futures::executor::block_on(node(THRESHOLD_CLASS_TYPE)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("mask".to_owned(), NativeValue::Handle { value: mask }),
                (
                    "value".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(0.5),
                    },
                ),
            ]),
        ))?;
        assert_eq!(harness.output_values(threshold)?, vec![0.0, 0.0, 1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn void_quadmask_quantizes_scales_and_dilates_exactly() -> Result<(), Box<dyn std::error::Error>>
    {
        let harness = Harness::new(CancellationToken::default())?;
        let mask = harness.publish_mask(
            vec![1, 3, 5],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )?;
        let outcome = futures::executor::block_on(node(VOID_CLASS_TYPE)?.execute(
            harness.context.clone(),
            BTreeMap::from([
                ("mask".to_owned(), NativeValue::Handle { value: mask }),
                (
                    "dilate_width".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::UnsignedInteger(1),
                    },
                ),
            ]),
        ))?;
        assert_eq!(
            harness.output_values(outcome)?,
            vec![
                0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0
            ]
        );

        let quantized = void_quadmask_values(
            &harness.context,
            &[1, 1, 5],
            vec![0.0, 32.0, 96.0, 192.0, f32::NAN],
            0,
        )?;
        assert_eq!(&quantized[..4], &[1.0, 192.0 / 255.0, 128.0 / 255.0, 0.0]);
        assert!(quantized[4].is_nan());
        Ok(())
    }

    #[test]
    fn bounds_and_cancellation_fail_closed() -> Result<(), Box<dyn std::error::Error>> {
        assert!(bounded_integer(None, "width", 1, MAX_RESOLUTION).is_err());
        assert!(
            bounded_integer(
                Some(&NativeValue::Primitive {
                    value: NativePrimitive::UnsignedInteger(51),
                }),
                "dilate_width",
                0,
                50,
            )
            .is_err()
        );
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let harness = Harness::new(cancellation)?;
        let failure = futures::executor::block_on(
            node(SOLID_CLASS_TYPE)?.execute(harness.context, BTreeMap::new()),
        )
        .expect_err("cancelled execution must fail before parsing inputs");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);
        Ok(())
    }
}
