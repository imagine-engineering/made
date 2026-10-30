use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOutputDescriptor,
    NativePortCardinality, NativePrimitive, NativePrimitiveType, NativeResolvedPayload,
    NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_tensor::{
    ImageTensor, Layout, MemoryFormatReference, NativeTensorPayload, NativeTensorRole, Scalar,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_05::div_with_context_exact_native,
    generated_elementwise_or_runtime_operation_18::sub_method_with_context_exact_native,
    generated_external_tensor_kernel_02::{
        rgb_to_ycbcr_with_context_exact_native, ycbcr_to_rgb_with_context_exact_native,
    },
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_reduction_01::tensor_mean_with_context_exact_native,
    generated_shape_layout_transform_01::tensor_expand_as_exact_native,
    generated_shape_layout_transform_02::{
        torch_cat_with_context_exact_native, torch_movedim_exact_native,
    },
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["ImageRGBToYUV", "ImageYUVToRGB", "NormalizeImages"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "image/color";

#[derive(Clone, Copy, Debug)]
enum ImageColorKind {
    RgbToYuv,
    YuvToRgb,
    Normalize,
}

impl ImageColorKind {
    const fn class_type(self) -> &'static str {
        match self {
            Self::RgbToYuv => "ImageRGBToYUV",
            Self::YuvToRgb => "ImageYUVToRGB",
            Self::Normalize => "NormalizeImages",
        }
    }

    const fn feature_id(self) -> &'static str {
        match self {
            Self::RgbToYuv => "COMFY-NODE-0260",
            Self::YuvToRgb => "COMFY-NODE-0270",
            Self::Normalize => "COMFY-NODE-0456",
        }
    }

    const fn implementation_version(self) -> &'static str {
        match self {
            Self::RgbToYuv | Self::YuvToRgb => "source-2638e6d5-v1",
            Self::Normalize => "source-3b27465f-v1",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::RgbToYuv => "Image RGB to YUV",
            Self::YuvToRgb => "Image YUV to RGB",
            Self::Normalize => "Normalize Image Colors",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::RgbToYuv | Self::YuvToRgb => "",
            Self::Normalize => "Normalize images using mean and standard deviation.",
        }
    }

    fn input_names(self) -> &'static [&'static str] {
        match self {
            Self::RgbToYuv => &["image"],
            Self::YuvToRgb => &["Y", "U", "V"],
            Self::Normalize => &["images", "mean", "std"],
        }
    }

    fn output_names(self) -> &'static [&'static str] {
        match self {
            Self::RgbToYuv => &["Y", "U", "V"],
            Self::YuvToRgb => &["image"],
            Self::Normalize => &["images"],
        }
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        ImageColorKind::RgbToYuv,
        ImageColorKind::YuvToRgb,
        ImageColorKind::Normalize,
    ]
    .into_iter()
    .map(native_binding)
    .collect()
}

fn native_binding(kind: ImageColorKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let class_type = kind.class_type();
    let input_names = kind
        .input_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let output_names = kind
        .output_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let source_schema = built_in_source_schema(class_type)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(&input_names, &[], &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let image_type = image_type()?;
    let inputs = kind
        .input_names()
        .iter()
        .map(|name| {
            let accepted_types = if matches!(*name, "mean" | "std") {
                NativeTypeUnion::new([NativeValueType::Primitive(NativePrimitiveType::Number)])
            } else {
                NativeTypeUnion::new([NativeValueType::Handle(image_type.clone())])
            }?;
            Ok(NativeInputDescriptor {
                name: (*name).to_owned(),
                accepted_types,
                required: true,
                hidden: false,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: matches!(*name, "mean" | "std"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = kind
        .output_names()
        .iter()
        .map(|name| NativeOutputDescriptor {
            name: (*name).to_owned(),
            produced_type: NativeValueType::Handle(image_type.clone()),
            is_list: false,
        })
        .collect();
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: class_type.to_owned(),
            implementation_version: kind.implementation_version().to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs: Vec::new(),
            outputs,
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: kind.display_name().to_owned(),
            category: CATEGORY.to_owned(),
            description: kind.description().to_owned(),
            output_names,
            search_aliases: match kind {
                ImageColorKind::RgbToYuv | ImageColorKind::YuvToRgb => {
                    vec!["color space conversion".to_owned()]
                }
                ImageColorKind::Normalize => {
                    vec!["normalize".to_owned(), "normalize colors".to_owned()]
                }
            },
            is_deprecated: false,
            is_experimental: matches!(kind, ImageColorKind::Normalize),
        },
        node: Arc::new(ImageColorNode { kind }),
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

#[derive(Debug)]
struct ImageColorNode {
    kind: ImageColorKind,
}

impl NativeNode for ImageColorNode {
    fn class_type(&self) -> &str {
        self.kind.class_type()
    }

    fn implementation_version(&self) -> &str {
        self.kind.implementation_version()
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_inputs(self.kind, inputs)?;
        Ok(format!(
            "{}-{}",
            self.kind.class_type(),
            self.kind.implementation_version()
        ))
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context)?;
        validate_inputs(self.kind, inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context)?;
            validate_inputs(self.kind, &inputs)?;
            let compute = context
                .compute_session()
                .map_err(|error| compute_failure(error.to_string()))?;
            let tensor_context = compute
                .execution_context(&context)
                .map_err(|error| compute_failure(error.to_string()))?;
            let outputs = match self.kind {
                ImageColorKind::RgbToYuv => {
                    let image = resolve_image(&context, &inputs, "image")?;
                    let channels = rgb_to_yuv(compute.backend(), &image, &tensor_context)
                        .map_err(tensor_failure)?;
                    channels
                        .into_iter()
                        .map(|image| publish_image(&context, image))
                        .collect::<Result<Vec<_>, _>>()?
                }
                ImageColorKind::YuvToRgb => {
                    let y = resolve_image(&context, &inputs, "Y")?;
                    let u = resolve_image(&context, &inputs, "U")?;
                    let v = resolve_image(&context, &inputs, "V")?;
                    let image = yuv_to_rgb(compute.backend(), &y, &u, &v, &tensor_context)
                        .map_err(tensor_failure)?;
                    vec![publish_image(&context, image)?]
                }
                ImageColorKind::Normalize => {
                    let image = resolve_image(&context, &inputs, "images")?;
                    let mean = required_number(&inputs, "mean", 0.0, 1.0)?;
                    let standard_deviation = required_number(&inputs, "std", 0.001, 1.0)?;
                    let normalized = normalize(
                        compute.backend(),
                        &image,
                        mean,
                        standard_deviation,
                        &tensor_context,
                    )
                    .map_err(tensor_failure)?;
                    vec![publish_image(&context, normalized)?]
                }
            };
            tensor_context
                .check()
                .map_err(|error| tensor_failure(error.to_string()))?;
            let outcome = NativeNodeOutcome::Values {
                outputs,
                ui: None,
                effects: Vec::new(),
            };
            outcome
                .validate()
                .map_err(|error| invalid_inputs(error.to_string()))?;
            Ok(outcome)
        })
    }
}

fn validate_inputs(
    kind: ImageColorKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    if inputs.len() != kind.input_names().len() {
        return Err(invalid_inputs(format!(
            "{} requires exactly {} inputs",
            kind.class_type(),
            kind.input_names().len()
        )));
    }
    for name in kind.input_names() {
        if matches!(*name, "mean" | "std") {
            let (minimum, maximum) = if *name == "mean" {
                (0.0, 1.0)
            } else {
                (0.001, 1.0)
            };
            required_number(inputs, name, minimum, maximum)?;
        } else {
            required_image_handle(inputs, name)?;
        }
    }
    Ok(())
}

fn required_image_handle<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<&'a crate::NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = inputs.get(name) else {
        return Err(invalid_inputs(format!(
            "{name} must be an exact IMAGE handle"
        )));
    };
    if value.handle_type().kind != NativeHandleKind::Image || value.handle_type().type_id != "IMAGE"
    {
        return Err(invalid_inputs(format!(
            "{name} must be an exact IMAGE handle"
        )));
    }
    Ok(value)
}

fn required_number(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: f64,
    maximum: f64,
) -> Result<f32, NativeNodeFailure> {
    let Some(NativeValue::Primitive {
        value: NativePrimitive::Number(value),
    }) = inputs.get(name)
    else {
        return Err(invalid_inputs(format!("{name} must be a finite FLOAT")));
    };
    if !value.is_finite() || *value < minimum || *value > maximum {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(*value as f32)
}

struct ResolvedImage {
    image: ImageTensor,
    _resolved: NativeResolvedPayload,
}

impl std::ops::Deref for ResolvedImage {
    type Target = ImageTensor;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

fn resolve_image(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<ResolvedImage, NativeNodeFailure> {
    let handle = required_image_handle(inputs, name)?;
    let expected_type = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected_type, &context.cancellation)
        .map_err(|error| handle_failure(name, error))?;
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(invalid_inputs(format!(
            "{name} did not resolve to a native tensor payload"
        )));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(invalid_inputs(format!(
            "{name} did not resolve to an IMAGE payload"
        )));
    }
    let image = payload
        .image()
        .ok_or_else(|| invalid_inputs(format!("{name} did not resolve to an IMAGE tensor")))?;
    Ok(ResolvedImage {
        image: image.clone(),
        _resolved: resolved,
    })
}

fn publish_image(
    context: &NativeNodeContext,
    image: ImageTensor,
) -> Result<NativeValue, NativeNodeFailure> {
    let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
        .map_err(|error| tensor_failure(error.to_string()))?;
    let handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(|error| handle_failure("output", error))?;
    Ok(NativeValue::Handle { value: handle })
}

fn rgb_to_yuv(
    backend: &sim_tensor::CpuBackend,
    image: &ImageTensor,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<[ImageTensor; 3], String> {
    let bchw = torch_movedim_exact_native(image.tensor(), &[-1], &[1], context.cancellation)
        .map_err(|error| error.to_string())?;
    let ycbcr = rgb_to_ycbcr_with_context_exact_native(backend, &bchw, context)
        .map_err(|error| error.to_string())?;
    let bhwc = torch_movedim_exact_native(&ycbcr, &[1], &[-1], context.cancellation)
        .map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(3);
    for channel in 0..3_i64 {
        context.check().map_err(|error| error.to_string())?;
        let narrowed = narrow_method_exact_native(&bhwc, 3, channel, 1, context.cancellation)
            .map_err(|error| error.to_string())?;
        let expanded =
            tensor_expand_as_exact_native(&narrowed, image.tensor(), context.cancellation)
                .map_err(|error| error.to_string())?;
        let contiguous = contiguous_with_context_exact_native(
            backend,
            &expanded,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        )
        .map_err(|error| error.to_string())?;
        output.push(ImageTensor::from_tensor(contiguous).map_err(|error| error.to_string())?);
    }
    output
        .try_into()
        .map_err(|_| "RGB-to-YUV did not produce exactly three channels".to_owned())
}

fn yuv_to_rgb(
    backend: &sim_tensor::CpuBackend,
    y: &ImageTensor,
    u: &ImageTensor,
    v: &ImageTensor,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, String> {
    let channels = [y, u, v]
        .into_iter()
        .map(|image| {
            tensor_mean_with_context_exact_native(
                backend,
                image.tensor(),
                Some(&[-1]),
                true,
                None,
                context,
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bhwc = torch_cat_with_context_exact_native(backend, &channels, -1, context)
        .map_err(|error| error.to_string())?;
    let bchw = torch_movedim_exact_native(&bhwc, &[-1], &[1], context.cancellation)
        .map_err(|error| error.to_string())?;
    let rgb = ycbcr_to_rgb_with_context_exact_native(backend, &bchw, context)
        .map_err(|error| error.to_string())?;
    let bhwc = torch_movedim_exact_native(&rgb, &[1], &[-1], context.cancellation)
        .map_err(|error| error.to_string())?;
    let contiguous = contiguous_with_context_exact_native(
        backend,
        &bhwc,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )
    .map_err(|error| error.to_string())?;
    ImageTensor::from_tensor(contiguous).map_err(|error| error.to_string())
}

fn normalize(
    backend: &sim_tensor::CpuBackend,
    image: &ImageTensor,
    mean: f32,
    standard_deviation: f32,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, String> {
    let centered = sub_method_with_context_exact_native(
        backend,
        image.tensor(),
        ElementwiseOperand::Scalar(Scalar::Float(f64::from(mean))),
        1.0,
        context,
    )
    .map_err(|error| error.to_string())?;
    let normalized = div_with_context_exact_native(
        backend,
        &centered,
        ElementwiseOperand::Scalar(Scalar::Float(f64::from(standard_deviation))),
        context,
    )
    .map_err(|error| error.to_string())?;
    ImageTensor::from_tensor(normalized).map_err(|error| error.to_string())
}

fn check_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(context.node_id.0.as_str()))
}

fn handle_failure(name: &str, error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(name)
    } else {
        NativeNodeFailure {
            code: "invalid_image_handle".to_owned(),
            message: format!("{name} IMAGE handle is unavailable: {error}"),
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

fn tensor_failure(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "image_color_compute_failed".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn compute_failure(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "image_color_compute_unavailable".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure(subject: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: format!("{subject} image-color execution was interrupted"),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeHandleStore, NativeHandleStoreIdentity, NativeNodeComputeSession,
        NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle,
        NativeResolvedPayloadRetention, NodeRegistry, native_image_descriptors,
    };
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::Value;
    use std::{
        error::Error,
        sync::{
            Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-color-comfy-node-0254/fixture.json"
    ));

    fn fixture_values(fixture: &Value, pointer: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let values = fixture
            .pointer(pointer)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("fixture array `{pointer}` is absent"))?;
        values
            .iter()
            .map(|value| -> Result<f32, Box<dyn Error>> {
                Ok(value
                    .as_f64()
                    .map(|value| value as f32)
                    .ok_or_else(|| format!("fixture array `{pointer}` contains a non-number"))?)
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[derive(Debug)]
    struct TestResolvedPayloadRetention;

    impl NativeResolvedPayloadRetention for TestResolvedPayloadRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        payloads: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(store_id: u128, attempt_id: AttemptId) -> Result<Arc<Self>, Box<dyn Error>> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(store_id),
                    Uuid::from_u128(store_id + 1),
                )?,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                payloads: Mutex::new(BTreeMap::new()),
            }))
        }

        fn stored_payload(
            &self,
            handle: &NativeOpaqueHandle,
        ) -> Result<Arc<NativeStoredPayload>, Box<dyn Error>> {
            self.payloads
                .lock()
                .map_err(|_| "test payload store is poisoned")?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| "published test payload is absent".into())
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
                .payloads
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            if handle.generation() != 1 {
                return Err(NativeHandleStoreError::Missing(
                    handle.identifier().to_owned(),
                ));
            }
            if handle.digest_sha256() != Some(payload.digest_sha256().as_str()) {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            NativeResolvedPayload::checked(payload, Arc::new(TestResolvedPayloadRetention))
                .map_err(NativeHandleStoreError::from)
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
            self.payloads
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .insert(identifier.clone(), Arc::new(payload));
            NativeOpaqueHandle::new(handle_type, self.identity, identifier, 1, Some(digest))
                .map_err(NativeHandleStoreError::InvalidHandle)
        }

        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            self.payloads
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store is poisoned".to_owned()))?
                .remove(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            Ok(())
        }
    }

    fn executable(class_type: &str) -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == class_type => Some(node),
                _ => None,
            })
            .ok_or_else(|| format!("{class_type} executable binding is absent").into())
    }

    fn node_context(
        store: Arc<TestStore>,
        cancellation: CancellationToken,
        memory_limit: u64,
    ) -> Result<(NativeNodeContext, Arc<sim_tensor::CpuBackend>), Box<dyn Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
        let backend = Arc::new(backend);
        let scratch = workspace.authorize_workspace(memory_limit)?;
        let node_id = NodeId("image-color-test".to_owned());
        let service_identity = NativeNodeServiceIdentity::checked(
            Uuid::from_u128(0x39320),
            store.attempt_id,
            node_id.clone(),
        )?;
        let compute = NativeNodeComputeSession::checked(
            service_identity,
            backend.clone(),
            StreamId::DEFAULT,
            &scratch,
        )?;
        let context = NativeNodeContext::new_with_services(
            PromptId(Uuid::from_u128(0x39321)),
            store.attempt_id,
            node_id,
            cancellation,
            scratch,
            store,
            NativeNodeServices::checked(None, None, Some(compute))?,
        )?;
        Ok((context, backend))
    }

    fn image_handle(
        store: &TestStore,
        image: ImageTensor,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
        let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
            .map_err(|error| NativeHandleStoreError::Rejected(error.to_string()))?;
        store.publish(NativeStoredPayload::Tensor(Arc::new(payload)), cancellation)
    }

    #[test]
    fn source_fixture_schemas_and_early_invert_owner_are_exact() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture.pointer("/stable_task_id").and_then(Value::as_str),
            Some("comfy-parity-native-nodes-image-color-comfy-node-0254")
        );
        assert_eq!(
            fixture.pointer("/sources/0/sha256").and_then(Value::as_str),
            Some("b8dfdde1de8975be762b085048143cc2dda8fc9202695e460ecc2c8dfe44bc4b")
        );
        assert_eq!(
            fixture.pointer("/sources/1/sha256").and_then(Value::as_str),
            Some("2638e6d5b1a2c6a3257c2a15f1ee53215712890cc60674db5a2908e2eb505344")
        );
        assert_eq!(
            fixture.pointer("/sources/2/sha256").and_then(Value::as_str),
            Some("3b27465fec391509083bd1837895c09abc489c04d81afae5ffe631abd6a4e772")
        );

        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), NODE_DESCRIPTOR_IDS.len());
        let registry = NodeRegistry::built_in()?;
        for (binding, identifier) in bindings.iter().zip(NODE_DESCRIPTOR_IDS) {
            assert_eq!(binding.descriptor().class_type, *identifier);
            binding.validate()?;
            registry.validate_native_binding(binding)?;
        }

        let normalize = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == "NormalizeImages")
            .ok_or("NormalizeImages binding is absent")?;
        assert_eq!(normalize.presentation().display_name, "Normalize Image Colors");
        assert_eq!(normalize.presentation().category, CATEGORY);
        assert_eq!(
            normalize.presentation().search_aliases,
            ["normalize", "normalize colors"]
        );
        assert!(normalize.presentation().is_experimental);
        assert_eq!(normalize.descriptor().inputs.len(), 3);
        assert_eq!(normalize.descriptor().outputs.len(), 1);

        assert!(!NODE_DESCRIPTOR_IDS.contains(&"ImageInvert"));
        let invert = native_image_descriptors()?
            .iter()
            .find(|descriptor| descriptor.class_type == "ImageInvert")
            .ok_or("early-slice ImageInvert descriptor is absent")?;
        assert_eq!(invert.display_name, "Invert Image Colors");
        assert_eq!(invert.search_aliases, ["reverse colors"]);
        Ok(())
    }

    #[test]
    fn source_derived_image_color_results_are_exact() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let cancellation = CancellationToken::default();
        let memory_limit = 64 * 1024 * 1024;
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(memory_limit)?;
        let scratch = workspace.authorize_workspace(memory_limit)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let rgb_values = fixture_values(&fixture, "/image/rgb")?;
        let image = ImageTensor::from_f32(&backend, &context, 1, 1, 2, 3, &rgb_values)?;

        let inverted = image.invert(&backend, &context)?;
        assert_close(
            inverted.as_f32_slice()?,
            &fixture_values(&fixture, "/image/inverted")?,
            f32::EPSILON,
        );

        let [y, u, v] = rgb_to_yuv(&backend, &image, &context)?;
        assert_close(
            y.as_f32_slice()?,
            &fixture_values(&fixture, "/image/y")?,
            1.0e-6,
        );
        assert_close(
            u.as_f32_slice()?,
            &fixture_values(&fixture, "/image/u")?,
            1.0e-6,
        );
        assert_close(
            v.as_f32_slice()?,
            &fixture_values(&fixture, "/image/v")?,
            1.0e-6,
        );
        let recovered = yuv_to_rgb(&backend, &y, &u, &v, &context)?;
        assert_close(recovered.as_f32_slice()?, &rgb_values, 5.0e-4);

        let normalize_input = fixture_values(&fixture, "/normalize/input")?;
        let normalize_image =
            ImageTensor::from_f32(&backend, &context, 1, 1, 2, 3, &normalize_input)?;
        let normalized = normalize(&backend, &normalize_image, 0.5, 0.25, &context)?;
        assert_close(
            normalized.as_f32_slice()?,
            &fixture_values(&fixture, "/normalize/expected")?,
            f32::EPSILON,
        );
        context.check()?;
        Ok(())
    }

    #[test]
    fn validation_and_cancellation_fail_closed() -> Result<(), Box<dyn Error>> {
        let valid = BTreeMap::from([
            (
                "mean".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(0.5),
                },
            ),
            (
                "std".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(0.5),
                },
            ),
        ]);
        assert_eq!(required_number(&valid, "mean", 0.0, 1.0)?, 0.5);
        assert_eq!(required_number(&valid, "std", 0.001, 1.0)?, 0.5);
        for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let values = BTreeMap::from([(
                "mean".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(invalid),
                },
            )]);
            assert!(required_number(&values, "mean", 0.0, 1.0).is_err());
        }
        let values = BTreeMap::from([(
            "std".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Number(0.0),
            },
        )]);
        assert!(required_number(&values, "std", 0.001, 1.0).is_err());

        let cancellation = CancellationToken::default();
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(4096)?;
        let scratch = workspace.authorize_workspace(4096)?;
        let image_context = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let image = ImageTensor::from_f32(&backend, &image_context, 1, 1, 1, 3, &[0.0; 3])?;
        cancellation.cancel();
        assert!(rgb_to_yuv(&backend, &image, &image_context).is_err());
        Ok(())
    }

    #[test]
    fn opaque_handle_execution_publication_recovery_and_cancellation_are_exact()
    -> Result<(), Box<dyn Error>> {
        let attempt_id = AttemptId(Uuid::from_u128(0x39310));
        let store = TestStore::new(0x39311, attempt_id)?;
        let cancellation = CancellationToken::default();
        let (context, backend) =
            node_context(store.clone(), cancellation.clone(), 64 * 1024 * 1024)?;
        let tensor_context = context.compute_session()?.execution_context(&context)?;
        let image = ImageTensor::from_f32(&backend, &tensor_context, 1, 1, 1, 3, &[1.0, 0.0, 0.0])?;
        let input_handle = image_handle(&store, image, &cancellation)?;
        let inputs = BTreeMap::from([(
            "image".to_owned(),
            NativeValue::Handle {
                value: input_handle.clone(),
            },
        )]);
        let outcome =
            futures::executor::block_on(executable("ImageRGBToYUV")?.execute(context, inputs))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("ImageRGBToYUV did not produce values".into());
        };
        assert_eq!(outputs.len(), 3);
        assert!(ui.is_none());
        assert!(effects.is_empty());
        for (output, expected) in
            outputs
                .iter()
                .zip([[0.299_f32; 3], [0.331364_f32; 3], [0.999813_f32; 3]])
        {
            let NativeValue::Handle { value } = output else {
                return Err("ImageRGBToYUV output is not an IMAGE handle".into());
            };
            let payload = store.stored_payload(value)?;
            let NativeStoredPayload::Tensor(payload) = payload.as_ref() else {
                return Err("ImageRGBToYUV output payload is not a tensor".into());
            };
            assert_close(
                payload
                    .image()
                    .ok_or("output IMAGE payload is absent")?
                    .as_f32_slice()?,
                &expected,
                1.0e-6,
            );
        }

        let fresh_store = TestStore::new(0x39313, attempt_id)?;
        let (fresh_context, _) =
            node_context(fresh_store, CancellationToken::default(), 64 * 1024 * 1024)?;
        let stale_inputs = BTreeMap::from([(
            "image".to_owned(),
            NativeValue::Handle {
                value: input_handle,
            },
        )]);
        let stale_error = futures::executor::block_on(
            executable("ImageRGBToYUV")?.execute(fresh_context, stale_inputs),
        )
        .expect_err("stale worker handle unexpectedly succeeded");
        assert_eq!(stale_error.code, "invalid_image_handle");

        let cancelled_store = TestStore::new(0x39315, attempt_id)?;
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let (cancelled_context, _) = node_context(cancelled_store, cancelled, 64 * 1024 * 1024)?;
        let error = futures::executor::block_on(
            executable("ImageRGBToYUV")?.execute(cancelled_context, BTreeMap::new()),
        )
        .expect_err("cancelled execution unexpectedly succeeded");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        Ok(())
    }
}
