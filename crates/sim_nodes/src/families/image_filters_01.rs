use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeInputRequirement, NativeNode, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality, NativePrimitive,
    NativeStoredPayload, NativeValue, built_in_source_schema,
    native_value_type_for_output_schema, native_value_types_for_input_schema,
};
use sim_media::image_quantization::{
    NativeImageDither, NativeImageQuantizationError, quantize_image_tensor,
};
use sim_tensor::{
    DType, DeviceId, ImageTensor, Layout, NativeTensorPayload, NativeTensorRole, ResizeCrop,
    ResizeMode, RetryRngPolicy, RngAlgorithm, RngProfileVersion, RngStreamAddress,
    TensorDescriptor, ViewAccess,
    generated_external_tensor_kernel_01::{
        NativeMorphologyOperation, native_morphology_with_context_exact,
    },
    generated_external_tensor_kernel_02::{
        canny_with_context_exact_native, rgb_to_lab_with_context_exact_native,
    },
    generated_external_tensor_kernel_03::lab_to_rgb_with_context_exact_native,
    generated_linear_algebra_01::eigh_with_context_exact_native,
    generated_native_diffusion::tensor_to_f32,
    generated_random_number_generation_01::manual_seed_exact_native,
    generated_random_number_generation_02::randn_with_context_exact_native,
    generated_shape_layout_transform_03::{
        FunctionalPadMode, functional_pad_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        ConvolutionConfiguration, conv_2d_tensor_with_context_exact_native,
    },
};
use futures::future::BoxFuture;
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "Canny",
    "ColorTransfer",
    "ImageAddNoise",
    "ImageBlend",
    "ImageBlur",
    "ImageQuantize",
    "ImageSharpen",
    "Morphology",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "image/filters";
const IMPLEMENTATION_VERSION: &str = "source-5efd352d-96ec39e8-a57638bf-2638e6d5-v1";
const BLEND_MODES: &[&str] = &[
    "normal",
    "multiply",
    "screen",
    "overlay",
    "soft_light",
    "difference",
];
const COLOR_METHODS: &[&str] = &["reinhard_lab", "mkl_lab", "histogram"];
const SOURCE_STATS: &[&str] = &["per_frame", "uniform", "target_frame"];
const DITHERS: &[&str] = &[
    "none",
    "floyd-steinberg",
    "bayer-2",
    "bayer-4",
    "bayer-8",
    "bayer-16",
];
const MORPHOLOGY_OPERATIONS: &[&str] = &[
    "erode",
    "dilate",
    "open",
    "close",
    "gradient",
    "bottom_hat",
    "top_hat",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FilterKind {
    Canny,
    ColorTransfer,
    AddNoise,
    Blend,
    Blur,
    Quantize,
    Sharpen,
    Morphology,
}

impl FilterKind {
    const ALL: [Self; 8] = [
        Self::Canny,
        Self::ColorTransfer,
        Self::AddNoise,
        Self::Blend,
        Self::Blur,
        Self::Quantize,
        Self::Sharpen,
        Self::Morphology,
    ];

    const fn feature_id(self) -> &'static str {
        match self {
            Self::Canny => "COMFY-NODE-0045",
            Self::ColorTransfer => "COMFY-NODE-0078",
            Self::AddNoise => "COMFY-NODE-0240",
            Self::Blend => "COMFY-NODE-0242",
            Self::Blur => "COMFY-NODE-0243",
            Self::Quantize => "COMFY-NODE-0259",
            Self::Sharpen => "COMFY-NODE-0266",
            Self::Morphology => "COMFY-NODE-0453",
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::Canny => "Canny",
            Self::ColorTransfer => "ColorTransfer",
            Self::AddNoise => "ImageAddNoise",
            Self::Blend => "ImageBlend",
            Self::Blur => "ImageBlur",
            Self::Quantize => "ImageQuantize",
            Self::Sharpen => "ImageSharpen",
            Self::Morphology => "Morphology",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::Canny => "Detect Edges (Canny)",
            Self::ColorTransfer => "Transfer Color",
            Self::AddNoise => "Add Noise to Image",
            Self::Blend => "Blend Images",
            Self::Blur => "Blur Image",
            Self::Quantize => "Quantize Image",
            Self::Sharpen => "Sharpen Image",
            Self::Morphology => "Apply Morphology",
        }
    }

    const fn input_names(self) -> &'static [&'static str] {
        match self {
            Self::Canny => &["image", "low_threshold", "high_threshold"],
            Self::ColorTransfer => &[
                "image_target",
                "image_ref",
                "method",
                "source_stats",
                "strength",
            ],
            Self::AddNoise => &["image", "seed", "strength"],
            Self::Blend => &["image1", "image2", "blend_factor", "blend_mode"],
            Self::Blur => &["image", "blur_radius", "sigma"],
            Self::Quantize => &["image", "colors", "dither"],
            Self::Sharpen => &["image", "sharpen_radius", "sigma", "alpha"],
            Self::Morphology => &["image", "operation", "kernel_size"],
        }
    }

    const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Canny => &["edge detection", "outline", "contour detection", "line art"],
            Self::ColorTransfer => &[
                "color match",
                "color grading",
                "color correction",
                "match colors",
                "color transform",
                "mkl",
                "reinhard",
                "histogram",
            ],
            Self::AddNoise => &["film grain"],
            Self::Blend => &["mix images"],
            Self::Morphology => &["erode", "dilate"],
            _ => &[],
        }
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    FilterKind::ALL
        .into_iter()
        .map(native_node_binding)
        .collect()
}

fn native_node_binding(kind: FilterKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let input_names = owned_names(kind.input_names());
    let output_names = vec!["image".to_owned()];
    let source_schema = catalog_schema
        .bind_execution_ports(&input_names, &[], &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog_schema
        .inputs
        .iter()
        .map(source_input_descriptor)
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = source_schema
        .outputs
        .iter()
        .map(|output| {
            Ok(NativeOutputDescriptor {
                name: output.name.clone(),
                produced_type: native_value_type_for_output_schema(output).map_err(|error| {
                    NativeNodeContractError::InvalidSourceSchema(error.to_string())
                })?,
                is_list: false,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
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
            description: if kind == FilterKind::ColorTransfer {
                "Match the colors of one image to another using various algorithms.".to_owned()
            } else {
                String::new()
            },
            output_names,
            search_aliases: owned_names(kind.aliases()),
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(ImageFilterNode { kind }),
    })
}

fn source_input_descriptor(
    input: &crate::CatalogNodeInputSchemaMetadata,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    let accepts_image = input
        .schema
        .source_type_names
        .iter()
        .any(|name| name == "IMAGE");
    Ok(NativeInputDescriptor {
        name: input.schema.name.clone(),
        accepted_types: native_value_types_for_input_schema(&input.schema)
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?,
        required: input.requirement == NativeInputRequirement::Required,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: !accepts_image,
    })
}

fn owned_names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

#[derive(Debug)]
struct ImageFilterNode {
    kind: FilterKind,
}

impl NativeNode for ImageFilterNode {
    fn class_type(&self) -> &str {
        self.kind.class_type()
    }

    fn implementation_version(&self) -> &str {
        IMPLEMENTATION_VERSION
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_inputs(self.kind, inputs)?;
        Ok(format!(
            "{}-{IMPLEMENTATION_VERSION}",
            self.kind.class_type()
        ))
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, self.kind)?;
        validate_inputs(self.kind, inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, self.kind)?;
            validate_inputs(self.kind, &inputs)?;
            let compute = context.compute_session().map_err(compute_failure)?;
            let execution = compute.execution_context(&context).map_err(compute_failure)?;
            let backend = compute.backend();
            let output = match self.kind {
                FilterKind::Canny => canny_image(
                    &resolve_image(&context, &inputs, "image", self.kind)?,
                    required_number(&inputs, "low_threshold", 0.01, 0.99)?,
                    required_number(&inputs, "high_threshold", 0.01, 0.99)?,
                    backend,
                    &execution,
                    self.kind,
                )?,
                FilterKind::ColorTransfer => color_transfer(
                    &resolve_image(&context, &inputs, "image_target", self.kind)?,
                    &resolve_image(&context, &inputs, "image_ref", self.kind)?,
                    required_combo(&inputs, "method", COLOR_METHODS)?,
                    required_source_stats(&inputs)?,
                    required_number(&inputs, "strength", 0.0, 10.0)?,
                    backend,
                    &execution,
                    &context,
                    self.kind,
                )?,
                FilterKind::AddNoise => add_noise(
                    &resolve_image(&context, &inputs, "image", self.kind)?,
                    required_unsigned(&inputs, "seed", 0, u64::MAX)?,
                    required_number(&inputs, "strength", 0.0, 1.0)?,
                    backend,
                    &execution,
                    &context,
                    self.kind,
                )?,
                FilterKind::Blend => blend_images(
                    &resolve_image(&context, &inputs, "image1", self.kind)?,
                    &resolve_image(&context, &inputs, "image2", self.kind)?,
                    required_number(&inputs, "blend_factor", 0.0, 1.0)?,
                    required_combo(&inputs, "blend_mode", BLEND_MODES)?,
                    backend,
                    &execution,
                    &context,
                    self.kind,
                )?,
                FilterKind::Blur => convolve_filter(
                    &resolve_image(&context, &inputs, "image", self.kind)?,
                    required_unsigned(&inputs, "blur_radius", 1, 31)?,
                    required_number(&inputs, "sigma", 0.1, 10.0)?,
                    None,
                    backend,
                    &execution,
                    &context,
                    self.kind,
                )?,
                FilterKind::Quantize => {
                    let image = resolve_image(&context, &inputs, "image", self.kind)?;
                    quantize_image_tensor(
                        &image,
                        u16::try_from(required_unsigned(&inputs, "colors", 1, 256)?)
                            .map_err(|error| native_failure(self.kind, error))?,
                        parse_dither(required_combo(&inputs, "dither", DITHERS)?)?,
                        backend,
                        &execution,
                    )
                    .map_err(|error| quantization_failure(self.kind, error))?
                }
                FilterKind::Sharpen => convolve_filter(
                    &resolve_image(&context, &inputs, "image", self.kind)?,
                    required_unsigned(&inputs, "sharpen_radius", 1, 31)?,
                    required_number(&inputs, "sigma", 0.1, 10.0)?,
                    Some(required_number(&inputs, "alpha", 0.0, 5.0)?),
                    backend,
                    &execution,
                    &context,
                    self.kind,
                )?,
                FilterKind::Morphology => morphology_image(
                    &resolve_image(&context, &inputs, "image", self.kind)?,
                    required_combo(&inputs, "operation", MORPHOLOGY_OPERATIONS)?,
                    required_unsigned(&inputs, "kernel_size", 3, 999)?,
                    backend,
                    &execution,
                    self.kind,
                )?,
            };
            publish_image(&context, output, self.kind)
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceStats {
    PerFrame,
    Uniform,
    TargetFrame(usize),
}

fn validate_inputs(
    kind: FilterKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    if inputs.len() != kind.input_names().len()
        || inputs
            .keys()
            .any(|name| !kind.input_names().contains(&name.as_str()))
    {
        return Err(invalid_inputs(format!(
            "{} requires exactly its declared inputs",
            kind.class_type()
        )));
    }
    match kind {
        FilterKind::Canny => {
            exact_image_handle(inputs.get("image"), "image")?;
            let low = required_number(inputs, "low_threshold", 0.01, 0.99)?;
            let high = required_number(inputs, "high_threshold", 0.01, 0.99)?;
            if low > high {
                return Err(invalid_inputs("low_threshold must not exceed high_threshold"));
            }
        }
        FilterKind::ColorTransfer => {
            exact_image_handle(inputs.get("image_target"), "image_target")?;
            exact_image_handle(inputs.get("image_ref"), "image_ref")?;
            required_combo(inputs, "method", COLOR_METHODS)?;
            required_source_stats(inputs)?;
            required_number(inputs, "strength", 0.0, 10.0)?;
        }
        FilterKind::AddNoise => {
            exact_image_handle(inputs.get("image"), "image")?;
            required_unsigned(inputs, "seed", 0, u64::MAX)?;
            required_number(inputs, "strength", 0.0, 1.0)?;
        }
        FilterKind::Blend => {
            exact_image_handle(inputs.get("image1"), "image1")?;
            exact_image_handle(inputs.get("image2"), "image2")?;
            required_number(inputs, "blend_factor", 0.0, 1.0)?;
            required_combo(inputs, "blend_mode", BLEND_MODES)?;
        }
        FilterKind::Blur => {
            exact_image_handle(inputs.get("image"), "image")?;
            required_unsigned(inputs, "blur_radius", 1, 31)?;
            required_number(inputs, "sigma", 0.1, 10.0)?;
        }
        FilterKind::Quantize => {
            exact_image_handle(inputs.get("image"), "image")?;
            required_unsigned(inputs, "colors", 1, 256)?;
            required_combo(inputs, "dither", DITHERS)?;
        }
        FilterKind::Sharpen => {
            exact_image_handle(inputs.get("image"), "image")?;
            required_unsigned(inputs, "sharpen_radius", 1, 31)?;
            required_number(inputs, "sigma", 0.1, 10.0)?;
            required_number(inputs, "alpha", 0.0, 5.0)?;
        }
        FilterKind::Morphology => {
            exact_image_handle(inputs.get("image"), "image")?;
            required_combo(inputs, "operation", MORPHOLOGY_OPERATIONS)?;
            required_unsigned(inputs, "kernel_size", 3, 999)?;
        }
    }
    Ok(())
}

fn exact_image_handle<'a>(
    value: Option<&'a NativeValue>,
    name: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(invalid_inputs(format!("{name} must be an exact IMAGE handle")));
    };
    if value.handle_type().kind != NativeHandleKind::Image
        || value.handle_type().type_id != "IMAGE"
    {
        return Err(invalid_inputs(format!("{name} must be an exact IMAGE handle")));
    }
    Ok(value)
}

fn required_number(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: f32,
    maximum: f32,
) -> Result<f32, NativeNodeFailure> {
    let value = match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => *value,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => *value as f64,
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => *value as f64,
        _ => return Err(invalid_inputs(format!("{name} must be a FLOAT"))),
    };
    if !value.is_finite() || !(f64::from(minimum)..=f64::from(maximum)).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be finite and between {minimum} and {maximum}"
        )));
    }
    Ok(value as f32)
}

fn required_unsigned(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => *value,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value)
            .map_err(|_| invalid_inputs(format!("{name} must be non-negative")))?,
        _ => return Err(invalid_inputs(format!("{name} must be an INT"))),
    };
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn required_combo<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
    choices: &[&str],
) -> Result<&'a str, NativeNodeFailure> {
    let Some(NativeValue::PreservedUnknown {
        type_name,
        value: Value::String(value),
    }) = inputs.get(name)
    else {
        return Err(invalid_inputs(format!("{name} must be a COMBO value")));
    };
    if type_name != "COMBO" || !choices.contains(&value.as_str()) {
        return Err(invalid_inputs(format!("unsupported {name} value {value}")));
    }
    Ok(value)
}

fn required_source_stats(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<SourceStats, NativeNodeFailure> {
    let Some(NativeValue::PreservedUnknown { type_name, value }) = inputs.get("source_stats")
    else {
        return Err(invalid_inputs(
            "source_stats must be a COMFY_DYNAMICCOMBO_V3 value",
        ));
    };
    if type_name != "COMFY_DYNAMICCOMBO_V3" {
        return Err(invalid_inputs(
            "source_stats must be a COMFY_DYNAMICCOMBO_V3 value",
        ));
    }
    let object = value
        .as_object()
        .ok_or_else(|| invalid_inputs("source_stats must be an object"))?;
    let mode = object
        .get("source_stats")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_inputs("source_stats object is missing source_stats"))?;
    if !SOURCE_STATS.contains(&mode) {
        return Err(invalid_inputs(format!("unsupported source_stats value {mode}")));
    }
    match mode {
        "per_frame" => Ok(SourceStats::PerFrame),
        "uniform" => Ok(SourceStats::Uniform),
        "target_frame" => {
            let index = object.get("target_index").and_then(Value::as_u64).unwrap_or(0);
            if index > 10_000 {
                return Err(invalid_inputs("target_index must be between 0 and 10000"));
            }
            Ok(SourceStats::TargetFrame(index as usize))
        }
        _ => Err(invalid_inputs("unsupported source_stats value")),
    }
}

fn resolve_image(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let handle = exact_image_handle(inputs.get(name), name)?;
    let expected = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected, &context.cancellation)
        .map_err(|error| handle_failure(error, kind))?;
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(native_failure(kind, "IMAGE handle did not resolve to a tensor"));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(native_failure(kind, "IMAGE tensor role changed"));
    }
    payload
        .image()
        .cloned()
        .ok_or_else(|| native_failure(kind, "IMAGE handle has no canonical ImageTensor"))
}

fn publish_image(
    context: &NativeNodeContext,
    image: ImageTensor,
    kind: FilterKind,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check_cancellation(context, kind)?;
    let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
        .map_err(|error| native_failure(kind, error))?;
    let handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(|error| handle_failure(error, kind))?;
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

fn canny_image(
    image: &ImageTensor,
    low: f32,
    high: f32,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let nchw = bhwc_as_nchw(image, kind)?;
    let output = canny_with_context_exact_native(backend, &nchw, low, high, execution)
        .map_err(|error| native_failure(kind, error))?;
    let edges = tensor_to_f32(backend, output.edges(), execution)
        .map_err(|error| native_failure(kind, error))?;
    let pixel_count = checked_count(&[batch, height, width], kind)?;
    if edges.len() != pixel_count || !matches!(channels, 1 | 3) {
        return Err(native_failure(kind, "canonical Canny output shape changed"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(
            pixel_count
                .checked_mul(3)
                .ok_or_else(|| native_failure(kind, "Canny output size overflowed"))?,
        )
        .map_err(|error| native_failure(kind, error))?;
    for (index, edge) in edges.iter().copied().enumerate() {
        execution_periodic_cancellation(execution, kind, index)?;
        values.extend_from_slice(&[edge, edge, edge]);
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, 3, &values)
        .map_err(|error| native_failure(kind, error))
}

fn morphology_image(
    image: &ImageTensor,
    operation: &str,
    kernel_size: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let kernel_count = checked_count(&[kernel_size, kernel_size], kind)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![kernel_size, kernel_size],
        DType::F32,
        DeviceId::CPU,
        execution.stream,
    )
    .map_err(|error| native_failure(kind, error))?;
    let mut kernel_values = Vec::new();
    kernel_values
        .try_reserve_exact(kernel_count)
        .map_err(|error| native_failure(kind, error))?;
    kernel_values.resize(kernel_count, 1.0);
    let kernel = backend
        .upload_f32(descriptor, &kernel_values, execution)
        .map_err(|error| native_failure(kind, error))?
        .0;
    let operation = match operation {
        "erode" => NativeMorphologyOperation::Erosion,
        "dilate" => NativeMorphologyOperation::Dilation,
        "open" => NativeMorphologyOperation::Opening,
        "close" => NativeMorphologyOperation::Closing,
        "gradient" => NativeMorphologyOperation::Gradient,
        "bottom_hat" => NativeMorphologyOperation::BottomHat,
        "top_hat" => NativeMorphologyOperation::TopHat,
        _ => return Err(invalid_inputs("unsupported morphology operation")),
    };
    let output = native_morphology_with_context_exact(
        backend,
        &bhwc_as_nchw(image, kind)?,
        &kernel,
        operation,
        execution,
    )
    .map_err(|error| native_failure(kind, error))?;
    nchw_to_image(
        output, batch, height, width, channels, backend, execution, kind,
    )
}

fn bhwc_as_nchw(
    image: &ImageTensor,
    kind: FilterKind,
) -> Result<sim_tensor::Tensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    image
        .tensor()
        .view(
            TensorDescriptor::channels_last(
                vec![batch, channels, height, width],
                DType::F32,
                DeviceId::CPU,
                image.tensor().descriptor().stream(),
            )
            .map_err(|error| native_failure(kind, error))?,
            ViewAccess::ReadOnly,
        )
        .map_err(|error| native_failure(kind, error))
}

fn nchw_to_image(
    tensor: sim_tensor::Tensor,
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let planar = tensor_to_f32(backend, &tensor, execution)
        .map_err(|error| native_failure(kind, error))?;
    let pixels_per_frame = checked_count(&[height, width], kind)?;
    let expected = checked_count(&[batch, channels, height, width], kind)?;
    if planar.len() != expected {
        return Err(native_failure(kind, "NCHW tensor storage size changed"));
    }
    let mut interleaved = Vec::new();
    interleaved
        .try_reserve_exact(expected)
        .map_err(|error| native_failure(kind, error))?;
    for batch_index in 0..batch as usize {
        let frame_start = batch_index
            .checked_mul(pixels_per_frame)
            .and_then(|value| value.checked_mul(channels as usize))
            .ok_or_else(|| native_failure(kind, "NCHW frame offset overflowed"))?;
        for pixel in 0..pixels_per_frame {
            execution_periodic_cancellation(execution, kind, interleaved.len())?;
            for channel in 0..channels as usize {
                interleaved.push(planar[frame_start + channel * pixels_per_frame + pixel]);
            }
        }
    }
    ImageTensor::from_f32(
        backend,
        execution,
        batch,
        height,
        width,
        channels,
        &interleaved,
    )
    .map_err(|error| native_failure(kind, error))
}

fn add_noise(
    image: &ImageTensor,
    seed: u64,
    strength: f32,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    context: &NativeNodeContext,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let address = RngStreamAddress::new(
        "sim-source-manual-seed",
        "image-add-noise",
        "ImageAddNoise",
        0,
        "image-add-noise",
        0,
        0,
        RetryRngPolicy::Replay,
    )
    .map_err(|error| native_failure(kind, error))?;
    let generator = manual_seed_exact_native(
        RngProfileVersion::V1,
        RngAlgorithm::Mt19937,
        i128::from(seed),
        address,
        &context.cancellation,
    )
    .map_err(|error| native_failure(kind, error))?;
    let normal = randn_with_context_exact_native(
        backend,
        &[batch, height, width, channels],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        generator
            .begin(None)
            .map_err(|error| native_failure(kind, error))?,
        execution,
    )
    .map_err(|error| native_failure(kind, error))?;
    let noise = tensor_to_f32(backend, &normal.tensor, execution)
        .map_err(|error| native_failure(kind, error))?;
    let source = image
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    if source.len() != noise.len() {
        return Err(native_failure(kind, "random tensor shape changed"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len())
        .map_err(|error| native_failure(kind, error))?;
    for (index, (source, noise)) in source.iter().zip(noise.iter()).enumerate() {
        periodic_cancellation(context, kind, index)?;
        values.push((source + strength * noise).clamp(0.0, 1.0));
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, channels, &values)
        .map_err(|error| native_failure(kind, error))
}

fn blend_images(
    first: &ImageTensor,
    second: &ImageTensor,
    factor: f32,
    mode: &str,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    context: &NativeNodeContext,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (first_batch, height, width, channels) = first
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let second = alpha_fix(second, channels, backend, execution, kind)?;
    let (second_batch, second_height, second_width, _) = second
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let second = if second_height != height || second_width != width {
        second
            .resize(
                width,
                height,
                ResizeMode::Bicubic,
                ResizeCrop::Center,
                backend,
                execution,
            )
            .map_err(|error| native_failure(kind, error))?
    } else {
        second
    };
    let output_batch = if first_batch == second_batch {
        first_batch
    } else if first_batch == 1 {
        second_batch
    } else if second_batch == 1 {
        first_batch
    } else {
        return Err(native_failure(kind, "image batches cannot be broadcast"));
    };
    let first_values = first
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let second_values = second
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let frame_count = checked_count(&[height, width, channels], kind)?;
    let output_count = checked_count(&[output_batch, height, width, channels], kind)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|error| native_failure(kind, error))?;
    for batch in 0..output_batch {
        let first_frame = usize::try_from(if first_batch == 1 { 0 } else { batch })
            .ok()
            .and_then(|value| value.checked_mul(frame_count))
            .ok_or_else(|| native_failure(kind, "first image frame offset overflowed"))?;
        let second_frame = usize::try_from(if second_batch == 1 { 0 } else { batch })
            .ok()
            .and_then(|value| value.checked_mul(frame_count))
            .ok_or_else(|| native_failure(kind, "second image frame offset overflowed"))?;
        for offset in 0..frame_count {
            periodic_cancellation(context, kind, values.len())?;
            let first_value = first_values[first_frame + offset];
            let second_value = second_values[second_frame + offset];
            let blended = match mode {
                "normal" => second_value,
                "multiply" => first_value * second_value,
                "screen" => 1.0 - (1.0 - first_value) * (1.0 - second_value),
                "overlay" if first_value <= 0.5 => 2.0 * first_value * second_value,
                "overlay" => 1.0 - 2.0 * (1.0 - first_value) * (1.0 - second_value),
                "soft_light" if second_value <= 0.5 => {
                    first_value
                        - (1.0 - 2.0 * second_value) * first_value * (1.0 - first_value)
                }
                "soft_light" => {
                    let curve = if first_value <= 0.25 {
                        ((16.0 * first_value - 12.0) * first_value + 4.0) * first_value
                    } else {
                        first_value.sqrt()
                    };
                    first_value + (2.0 * second_value - 1.0) * (curve - first_value)
                }
                "difference" => first_value - second_value,
                _ => return Err(invalid_inputs("unsupported blend mode")),
            };
            values.push((first_value * (1.0 - factor) + blended * factor).clamp(0.0, 1.0));
        }
    }
    ImageTensor::from_f32(
        backend,
        execution,
        output_batch,
        height,
        width,
        channels,
        &values,
    )
    .map_err(|error| native_failure(kind, error))
}

fn alpha_fix(
    image: &ImageTensor,
    channels: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, source_channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    if source_channels == channels {
        return Ok(image.clone());
    }
    if !matches!(channels, 1 | 3 | 4) {
        return Err(native_failure(kind, "unsupported destination channel count"));
    }
    let source = image
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let pixels = checked_count(&[batch, height, width], kind)?;
    let source_channels = usize::try_from(source_channels)
        .map_err(|error| native_failure(kind, error))?;
    let channels = usize::try_from(channels).map_err(|error| native_failure(kind, error))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(
            pixels
                .checked_mul(channels)
                .ok_or_else(|| native_failure(kind, "alpha-fix output overflowed"))?,
        )
        .map_err(|error| native_failure(kind, error))?;
    for pixel in 0..pixels {
        execution_periodic_cancellation(execution, kind, pixel)?;
        let start = pixel
            .checked_mul(source_channels)
            .ok_or_else(|| native_failure(kind, "alpha-fix offset overflowed"))?;
        let available = channels.min(source_channels);
        values.extend_from_slice(
            source
                .get(start..start + available)
                .ok_or_else(|| native_failure(kind, "alpha-fix storage ended early"))?,
        );
        if channels > source_channels {
            values.extend(std::iter::repeat_n(0.0, channels - source_channels));
            if let Some(alpha) = values.last_mut() {
                *alpha = 1.0;
            }
        }
    }
    ImageTensor::from_f32(
        backend,
        execution,
        batch,
        height,
        width,
        channels as u64,
        &values,
    )
    .map_err(|error| native_failure(kind, error))
}

#[allow(clippy::too_many_arguments)]
fn convolve_filter(
    image: &ImageTensor,
    radius: u64,
    sigma: f32,
    sharpen_alpha: Option<f32>,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    _context: &NativeNodeContext,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    if radius >= height || radius >= width {
        return Err(native_failure(
            kind,
            "reflect padding requires radius smaller than both image dimensions",
        ));
    }
    let kernel_size = radius
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| native_failure(kind, "kernel size overflowed"))?;
    let mut kernel = gaussian_kernel(kernel_size, sigma, kind)?;
    if let Some(alpha) = sharpen_alpha {
        for value in &mut kernel {
            *value *= -(alpha * 10.0);
        }
        let center = kernel.len() / 2;
        let sum = kernel.iter().sum::<f32>();
        kernel[center] = kernel[center] - sum + 1.0;
    }
    let radius = i64::try_from(radius).map_err(|error| native_failure(kind, error))?;
    let padded = functional_pad_with_context_exact_native(
        backend,
        &bhwc_as_nchw(image, kind)?,
        &[radius, radius, radius, radius],
        FunctionalPadMode::Reflect,
        None,
        execution,
    )
    .map_err(|error| native_failure(kind, error))?;
    let channels_usize = usize::try_from(channels).map_err(|error| native_failure(kind, error))?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(
            kernel
                .len()
                .checked_mul(channels_usize)
                .ok_or_else(|| native_failure(kind, "convolution weights overflowed"))?,
        )
        .map_err(|error| native_failure(kind, error))?;
    for _ in 0..channels_usize {
        weights.extend_from_slice(&kernel);
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![channels, 1, kernel_size, kernel_size],
        DType::F32,
        DeviceId::CPU,
        execution.stream,
    )
    .map_err(|error| native_failure(kind, error))?;
    let weight = backend
        .upload_f32(descriptor, &weights, execution)
        .map_err(|error| native_failure(kind, error))?
        .0;
    let output = conv_2d_tensor_with_context_exact_native(
        backend,
        &padded,
        &weight,
        None,
        &ConvolutionConfiguration {
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: channels_usize,
            output_padding: vec![0, 0],
        },
        execution,
    )
    .map_err(|error| native_failure(kind, error))?;
    let output = nchw_to_image(
        output, batch, height, width, channels, backend, execution, kind,
    )?;
    if sharpen_alpha.is_none() {
        return Ok(output);
    }
    let source = output
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len())
        .map_err(|error| native_failure(kind, error))?;
    values.extend(source.iter().map(|value| value.clamp(0.0, 1.0)));
    ImageTensor::from_f32(backend, execution, batch, height, width, channels, &values)
        .map_err(|error| native_failure(kind, error))
}

fn gaussian_kernel(size: u64, sigma: f32, kind: FilterKind) -> Result<Vec<f32>, NativeNodeFailure> {
    let count = checked_count(&[size, size], kind)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| native_failure(kind, error))?;
    let denominator = (size - 1) as f32;
    for y in 0..size {
        let normalized_y = -1.0 + 2.0 * y as f32 / denominator;
        for x in 0..size {
            let normalized_x = -1.0 + 2.0 * x as f32 / denominator;
            values.push(
                (-(normalized_x * normalized_x + normalized_y * normalized_y)
                    / (2.0 * sigma * sigma))
                    .exp(),
            );
        }
    }
    let sum = values.iter().sum::<f32>();
    if !sum.is_finite() || sum == 0.0 {
        return Err(native_failure(kind, "Gaussian kernel normalization failed"));
    }
    for value in &mut values {
        *value /= sum;
    }
    Ok(values)
}

fn parse_dither(value: &str) -> Result<NativeImageDither, NativeNodeFailure> {
    match value {
        "none" => Ok(NativeImageDither::None),
        "floyd-steinberg" => Ok(NativeImageDither::FloydSteinberg),
        "bayer-2" => Ok(NativeImageDither::Bayer2),
        "bayer-4" => Ok(NativeImageDither::Bayer4),
        "bayer-8" => Ok(NativeImageDither::Bayer8),
        "bayer-16" => Ok(NativeImageDither::Bayer16),
        _ => Err(invalid_inputs("unsupported dither value")),
    }
}

#[allow(clippy::too_many_arguments)]
fn color_transfer(
    target: &ImageTensor,
    reference: &ImageTensor,
    method: &str,
    source_stats: SourceStats,
    strength: f32,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    context: &NativeNodeContext,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = target
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let (reference_batch, reference_height, reference_width, reference_channels) = reference
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    if channels != 3 || reference_channels != 3 || batch == 0 || reference_batch == 0 {
        return Err(native_failure(kind, "ColorTransfer requires nonempty RGB image batches"));
    }
    if strength == 0.0 {
        return Ok(target.clone());
    }
    if method == "histogram" {
        return histogram_transfer(
            target,
            reference,
            source_stats,
            strength,
            backend,
            execution,
            context,
            kind,
        );
    }
    let target_lab = lab_frames(target, backend, execution, kind)?;
    let reference_lab = lab_frames(reference, backend, execution, kind)?;
    let target_pixels = checked_count(&[height, width], kind)?;
    let reference_pixels = checked_count(&[reference_height, reference_width], kind)?;
    let reinhard = method == "reinhard_lab";
    let pooled_reference = if reference_batch == 1
        || matches!(source_stats, SourceStats::Uniform | SourceStats::TargetFrame(_))
    {
        Some(stats(
            &reference_lab,
            reference_pixels,
            reinhard,
            execution,
            kind,
        )?)
    } else {
        None
    };
    let shared_transform = match source_stats {
        SourceStats::Uniform => Some(build_lab_transform(
            stats(&target_lab, target_pixels, reinhard, execution, kind)?,
            pooled_reference
                .clone()
                .ok_or_else(|| native_failure(kind, "reference statistics are unavailable"))?,
            reinhard,
            backend,
            execution,
            kind,
        )?),
        SourceStats::TargetFrame(index) => {
            let index = index.min(batch.saturating_sub(1) as usize);
            Some(build_lab_transform(
                frame_stats(&target_lab[index], reinhard, execution, kind)?,
                pooled_reference
                    .clone()
                    .ok_or_else(|| native_failure(kind, "reference statistics are unavailable"))?,
                reinhard,
                backend,
                execution,
                kind,
            )?)
        }
        SourceStats::PerFrame => None,
    };
    let mut output = Vec::new();
    output
        .try_reserve_exact(checked_count(&[batch, height, width, channels], kind)?)
        .map_err(|error| native_failure(kind, error))?;
    for frame_index in 0..batch as usize {
        check_cancellation(context, kind)?;
        let transform = if let Some(transform) = shared_transform.clone() {
            transform
        } else {
            let reference_stats = if let Some(stats) = pooled_reference.clone() {
                stats
            } else {
                let reference_index = frame_index.min(reference_batch as usize - 1);
                frame_stats(&reference_lab[reference_index], reinhard, execution, kind)?
            };
            build_lab_transform(
                frame_stats(&target_lab[frame_index], reinhard, execution, kind)?,
                reference_stats,
                reinhard,
                backend,
                execution,
                kind,
            )?
        };
        let corrected = apply_lab_transform(
            &target_lab[frame_index],
            transform,
            execution,
            kind,
        )?;
        let mut blended = Vec::new();
        blended
            .try_reserve_exact(corrected.len())
            .map_err(|error| native_failure(kind, error))?;
        for (source, corrected) in target_lab[frame_index].iter().zip(corrected.iter()) {
            blended.push(source + strength * (corrected - source));
        }
        let rgb = lab_frame_to_rgb(&blended, height, width, backend, execution, kind)?;
        output.extend(rgb.into_iter().map(|value| value.clamp(0.0, 1.0)));
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, 3, &output)
        .map_err(|error| native_failure(kind, error))
}

#[derive(Clone, Debug)]
struct ColorStats {
    mean: [f32; 3],
    spread: [[f32; 3]; 3],
}

#[derive(Clone, Debug)]
struct LabTransform {
    matrix: [[f32; 3]; 3],
    offset: [f32; 3],
}

fn lab_frames(
    image: &ImageTensor,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<Vec<Vec<f32>>, NativeNodeFailure> {
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    if channels != 3 {
        return Err(native_failure(kind, "Lab conversion requires RGB input"));
    }
    let source = image
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let frame_values = checked_count(&[height, width, channels], kind)?;
    let mut frames = Vec::new();
    frames
        .try_reserve_exact(batch as usize)
        .map_err(|error| native_failure(kind, error))?;
    for frame in 0..batch as usize {
        let start = frame
            .checked_mul(frame_values)
            .ok_or_else(|| native_failure(kind, "Lab frame offset overflowed"))?;
        let bhwc = ImageTensor::from_f32(
            backend,
            execution,
            1,
            height,
            width,
            3,
            &source[start..start + frame_values],
        )
        .map_err(|error| native_failure(kind, error))?;
        let lab = rgb_to_lab_with_context_exact_native(
            backend,
            &bhwc_as_nchw(&bhwc, kind)?,
            execution,
        )
        .map_err(|error| native_failure(kind, error))?;
        let lab = tensor_to_f32(backend, &lab, execution)
            .map_err(|error| native_failure(kind, error))?;
        let mut frame_values = Vec::new();
        frame_values
            .try_reserve_exact(lab.len())
            .map_err(|error| native_failure(kind, error))?;
        frame_values.extend_from_slice(&lab);
        frames.push(frame_values);
    }
    Ok(frames)
}

fn lab_frame_to_rgb(
    lab: &[f32],
    height: u64,
    width: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<Vec<f32>, NativeNodeFailure> {
    let descriptor = TensorDescriptor::contiguous(
        vec![1, 3, height, width],
        DType::F32,
        DeviceId::CPU,
        execution.stream,
    )
    .map_err(|error| native_failure(kind, error))?;
    let tensor = backend
        .upload_f32(descriptor, lab, execution)
        .map_err(|error| native_failure(kind, error))?
        .0;
    let rgb = lab_to_rgb_with_context_exact_native(backend, &tensor, execution)
        .map_err(|error| native_failure(kind, error))?;
    let planar = tensor_to_f32(backend, &rgb, execution)
        .map_err(|error| native_failure(kind, error))?;
    planar_to_interleaved(&planar, height, width, execution, kind)
}

fn frame_stats(
    frame: &[f32],
    reinhard: bool,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<ColorStats, NativeNodeFailure> {
    stats(
        std::slice::from_ref(&frame),
        frame.len() / 3,
        reinhard,
        execution,
        kind,
    )
}

fn stats<T: AsRef<[f32]>>(
    frames: &[T],
    pixels_per_frame: usize,
    reinhard: bool,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<ColorStats, NativeNodeFailure> {
    if frames.is_empty() || pixels_per_frame == 0 {
        return Err(native_failure(kind, "color statistics require pixels"));
    }
    let mut mean = [0.0_f32; 3];
    for frame in frames {
        let frame = frame.as_ref();
        if frame.len() != pixels_per_frame * 3 {
            return Err(native_failure(kind, "Lab frame size changed"));
        }
        for channel in 0..3 {
            let mut sum = 0.0_f32;
            for (index, value) in frame
                [channel * pixels_per_frame..(channel + 1) * pixels_per_frame]
                .iter()
                .copied()
                .enumerate()
            {
                execution_periodic_cancellation(execution, kind, index)?;
                sum += value;
            }
            mean[channel] += sum / pixels_per_frame as f32;
        }
    }
    for value in &mut mean {
        *value /= frames.len() as f32;
    }
    let mut spread = [[0.0_f32; 3]; 3];
    for frame in frames {
        let frame = frame.as_ref();
        for pixel in 0..pixels_per_frame {
            execution_periodic_cancellation(execution, kind, pixel)?;
            let centered = std::array::from_fn::<_, 3, _>(|channel| {
                frame[channel * pixels_per_frame + pixel] - mean[channel]
            });
            if reinhard {
                for channel in 0..3 {
                    spread[channel][channel] += centered[channel] * centered[channel]
                        / pixels_per_frame as f32;
                }
            } else {
                for row in 0..3 {
                    for column in 0..3 {
                        spread[row][column] += centered[row] * centered[column]
                            / pixels_per_frame as f32;
                    }
                }
            }
        }
    }
    for row in &mut spread {
        for value in row {
            *value /= frames.len() as f32;
        }
    }
    if reinhard {
        for channel in 0..3 {
            spread[channel][channel] = spread[channel][channel].sqrt().max(1e-6);
        }
    }
    Ok(ColorStats { mean, spread })
}

fn build_lab_transform(
    source: ColorStats,
    reference: ColorStats,
    reinhard: bool,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<LabTransform, NativeNodeFailure> {
    let matrix = if reinhard {
        std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                if row == column {
                    reference.spread[row][row] / source.spread[row][row]
                } else {
                    0.0
                }
            })
        })
    } else {
        mkl_matrix(source.spread, reference.spread, backend, execution, kind)?
    };
    let transformed_mean = matrix_vector(matrix, source.mean);
    let offset = std::array::from_fn(|index| reference.mean[index] - transformed_mean[index]);
    Ok(LabTransform { matrix, offset })
}

fn mkl_matrix(
    source: [[f32; 3]; 3],
    reference: [[f32; 3]; 3],
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<[[f32; 3]; 3], NativeNodeFailure> {
    let (source_values, vectors) = eigen_matrix(source, backend, execution, kind)?;
    let source_sqrt = std::array::from_fn(|index| source_values[index].max(0.0).sqrt().max(1e-6));
    let scaled = scale_columns(vectors, source_sqrt);
    let middle = matrix_multiply(matrix_multiply(transpose(scaled), reference), scaled);
    let (middle_values, middle_vectors) = eigen_matrix(middle, backend, execution, kind)?;
    let middle_sqrt = std::array::from_fn(|index| middle_values[index].max(0.0).sqrt());
    let middle_half = matrix_multiply(
        scale_columns(middle_vectors, middle_sqrt),
        transpose(middle_vectors),
    );
    let inverse = scale_columns(vectors, source_sqrt.map(|value| 1.0 / value));
    Ok(matrix_multiply(matrix_multiply(inverse, middle_half), transpose(inverse)))
}

fn eigen_matrix(
    matrix: [[f32; 3]; 3],
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<([f32; 3], [[f32; 3]; 3]), NativeNodeFailure> {
    let descriptor = TensorDescriptor::contiguous(
        vec![3, 3],
        DType::F32,
        DeviceId::CPU,
        execution.stream,
    )
    .map_err(|error| native_failure(kind, error))?;
    let input = backend
        .upload_f32(
            descriptor,
            &matrix.into_iter().flatten().collect::<Vec<_>>(),
            execution,
        )
        .map_err(|error| native_failure(kind, error))?
        .0;
    let output = eigh_with_context_exact_native(backend, &input, false, execution)
        .map_err(|error| native_failure(kind, error))?;
    let eigenvalues = tensor_to_f32(backend, &output.eigenvalues, execution)
        .map_err(|error| native_failure(kind, error))?;
    let eigenvectors = tensor_to_f32(backend, &output.eigenvectors, execution)
        .map_err(|error| native_failure(kind, error))?;
    let eigenvalues: [f32; 3] = (&*eigenvalues)
        .try_into()
        .map_err(|_| native_failure(kind, "eigh eigenvalue shape changed"))?;
    Ok((eigenvalues, matrix_from_slice(&eigenvectors)))
}

fn matrix_from_slice(values: &[f32]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| values[row * 3 + column]))
}

fn scale_columns(matrix: [[f32; 3]; 3], scale: [f32; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[row][column] * scale[column]))
}

fn transpose(matrix: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn matrix_multiply(left: [[f32; 3]; 3], right: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3).map(|inner| left[row][inner] * right[inner][column]).sum()
        })
    })
}

fn matrix_vector(matrix: [[f32; 3]; 3], value: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|row| (0..3).map(|column| matrix[row][column] * value[column]).sum())
}

fn apply_lab_transform(
    frame: &[f32],
    transform: LabTransform,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<Vec<f32>, NativeNodeFailure> {
    let pixels = frame.len() / 3;
    let mut output = Vec::new();
    output
        .try_reserve_exact(frame.len())
        .map_err(|error| native_failure(kind, error))?;
    output.resize(frame.len(), 0.0);
    for pixel in 0..pixels {
        execution_periodic_cancellation(execution, kind, pixel)?;
        let value = std::array::from_fn(|channel| frame[channel * pixels + pixel]);
        let transformed = matrix_vector(transform.matrix, value);
        for channel in 0..3 {
            output[channel * pixels + pixel] = transformed[channel] + transform.offset[channel];
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn histogram_transfer(
    target: &ImageTensor,
    reference: &ImageTensor,
    source_stats: SourceStats,
    strength: f32,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    context: &NativeNodeContext,
    kind: FilterKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = target
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let (reference_batch, reference_height, reference_width, _) = reference
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let target_values = target
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let reference_values = reference
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let target_frame = checked_count(&[height, width, channels], kind)?;
    let reference_frame = checked_count(&[reference_height, reference_width, channels], kind)?;
    let reference_cdf = pooled_cdf(reference_values, reference_frame, execution, kind)?;
    let shared_lut = match source_stats {
        SourceStats::Uniform => Some(lut_from_cdfs(
            pooled_cdf(target_values, target_frame, execution, kind)?,
            reference_cdf,
        )),
        SourceStats::TargetFrame(index) => {
            let index = index.min(batch as usize - 1);
            let start = index * target_frame;
            Some(lut_from_cdfs(
                pooled_cdf(
                    &target_values[start..start + target_frame],
                    target_frame,
                    execution,
                    kind,
                )?,
                reference_cdf,
            ))
        }
        SourceStats::PerFrame => None,
    };
    let mut output = Vec::new();
    output
        .try_reserve_exact(target_values.len())
        .map_err(|error| native_failure(kind, error))?;
    let pixels = checked_count(&[height, width], kind)?;
    for frame in 0..batch as usize {
        check_cancellation(context, kind)?;
        let start = frame * target_frame;
        let frame_values = &target_values[start..start + target_frame];
        let lut = if let Some(lut) = shared_lut {
            lut
        } else {
            let reference_index = frame.min(reference_batch as usize - 1);
            let reference_start = reference_index * reference_frame;
            lut_from_cdfs(
                pooled_cdf(frame_values, target_frame, execution, kind)?,
                pooled_cdf(
                    &reference_values[reference_start..reference_start + reference_frame],
                    reference_frame,
                    execution,
                    kind,
                )?,
            )
        };
        for pixel in 0..pixels {
            for channel in 0..3 {
                let source = frame_values[pixel * 3 + channel];
                let bin = ((source * 255.0) as i64).clamp(0, 255) as usize;
                let matched = lut[channel][bin];
                output.push((source + strength * (matched - source)).clamp(0.0, 1.0));
            }
        }
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, 3, &output)
        .map_err(|error| native_failure(kind, error))
}

fn pooled_cdf(
    values: &[f32],
    frame_values: usize,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<[[f32; 256]; 3], NativeNodeFailure> {
    if frame_values == 0
        || !frame_values.is_multiple_of(3)
        || !values.len().is_multiple_of(frame_values)
    {
        return Err(native_failure(kind, "histogram frame shape is invalid"));
    }
    let mut histogram = [[0_u64; 256]; 3];
    for (index, pixel) in values.chunks_exact(3).enumerate() {
        execution_periodic_cancellation(execution, kind, index)?;
        for channel in 0..3 {
            let bin = ((pixel[channel] * 255.0) as i64).clamp(0, 255) as usize;
            histogram[channel][bin] += 1;
        }
    }
    let total = (values.len() / 3) as f32;
    Ok(std::array::from_fn(|channel| {
        let mut cumulative = 0_u64;
        std::array::from_fn(|bin| {
            cumulative += histogram[channel][bin];
            cumulative as f32 / total
        })
    }))
}

fn lut_from_cdfs(
    source: [[f32; 256]; 3],
    reference: [[f32; 256]; 3],
) -> [[f32; 256]; 3] {
    std::array::from_fn(|channel| {
        std::array::from_fn(|bin| {
            let selected = reference[channel]
                .partition_point(|value| *value < source[channel][bin])
                .min(255);
            selected as f32 / 255.0
        })
    })
}

fn planar_to_interleaved(
    planar: &[f32],
    height: u64,
    width: u64,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
) -> Result<Vec<f32>, NativeNodeFailure> {
    let pixels = checked_count(&[height, width], kind)?;
    if planar.len() != pixels * 3 {
        return Err(native_failure(kind, "planar RGB storage size changed"));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(planar.len())
        .map_err(|error| native_failure(kind, error))?;
    for pixel in 0..pixels {
        execution_periodic_cancellation(execution, kind, pixel)?;
        for channel in 0..3 {
            output.push(planar[channel * pixels + pixel]);
        }
    }
    Ok(output)
}

fn checked_count(dimensions: &[u64], kind: FilterKind) -> Result<usize, NativeNodeFailure> {
    dimensions
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| native_failure(kind, "tensor element count overflowed"))
}

fn periodic_cancellation(
    context: &NativeNodeContext,
    kind: FilterKind,
    index: usize,
) -> Result<(), NativeNodeFailure> {
    if index & 0x3fff == 0 {
        check_cancellation(context, kind)?;
    }
    Ok(())
}

fn execution_periodic_cancellation(
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: FilterKind,
    index: usize,
) -> Result<(), NativeNodeFailure> {
    if index & 0x3fff == 0 {
        execution
            .cancellation
            .check()
            .map_err(|_| interrupted_failure(kind))?;
    }
    Ok(())
}

fn check_cancellation(
    context: &NativeNodeContext,
    kind: FilterKind,
) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(kind))
}

fn handle_failure(error: NativeHandleStoreError, kind: FilterKind) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(kind)
    } else {
        NativeNodeFailure {
            code: "invalid_native_handle".to_owned(),
            message: format!("{} IMAGE handle is unavailable: {error}", kind.class_type()),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(error: NativeNodeContractError) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "native_compute_unavailable".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
    }
}

fn quantization_failure(
    kind: FilterKind,
    error: NativeImageQuantizationError,
) -> NativeNodeFailure {
    if matches!(error, NativeImageQuantizationError::Cancelled) {
        interrupted_failure(kind)
    } else {
        native_failure(kind, error)
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

fn native_failure(kind: FilterKind, error: impl std::fmt::Display) -> NativeNodeFailure {
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("cancelled") {
        return interrupted_failure(kind);
    }
    NativeNodeFailure {
        code: "native_image_filter_failed".to_owned(),
        message: format!("{} failed: {message}", kind.class_type()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure(kind: FilterKind) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: format!("{} execution was interrupted", kind.class_type()),
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
    use std::sync::{Mutex, atomic::{AtomicU64, Ordering}};
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-filters-comfy-node-0045/fixture.json"
    ));

    #[test]
    fn fixture_and_descriptor_set_cover_only_the_assigned_rows() {
        let fixture: Value = serde_json::from_str(FIXTURE).expect("fixture must parse");
        assert_eq!(fixture["stable_task_id"], "comfy-parity-native-nodes-image-filters-comfy-node-0045");
        let nodes = fixture["nodes"].as_array().expect("fixture nodes");
        assert_eq!(nodes.len(), 8);
        assert_eq!(NODE_DESCRIPTOR_IDS.len(), 8);
        let bindings = native_node_bindings().expect("bindings");
        assert_eq!(bindings.len(), 8);
        for binding in &bindings {
            let NativeNodeBinding::Executable { descriptor, .. } = binding else {
                panic!("assigned image filter binding must be executable");
            };
            descriptor
                .validate_exact_schema_v2()
                .expect("descriptor schema must remain exact");
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert!(!descriptor.output_node);
            assert!(descriptor.inputs.iter().all(|input| !input.lazy));
            assert!(descriptor.outputs.iter().all(|output| !output.is_list));
        }
        let persisted = BTreeMap::from([
            dynamic_stats("target_frame", Some(7)),
            number("strength", 0.75),
        ]);
        let encoded = serde_json::to_vec(&persisted).expect("inputs serialize");
        assert_eq!(
            serde_json::from_slice::<BTreeMap<String, NativeValue>>(&encoded)
                .expect("inputs deserialize"),
            persisted,
        );
    }

    #[test]
    fn gaussian_blend_and_histogram_primitives_match_source_boundaries() {
        let kernel = gaussian_kernel(3, 1.0, FilterKind::Blur).expect("kernel");
        assert!((kernel.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let source = std::array::from_fn(|channel| {
            std::array::from_fn(|bin| if channel == 0 { bin as f32 / 255.0 } else { 1.0 })
        });
        let lut = lut_from_cdfs(source, source);
        assert_eq!(lut[0][0], 0.0);
        assert_eq!(lut[0][255], 1.0);
    }

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
        fn new(seed: u128, attempt_id: AttemptId) -> Result<Arc<Self>, NativeNodeContractError> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(seed),
                    Uuid::from_u128(seed + 1),
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
            let identifier = format!("image-filter-{generation}");
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
                return Err(NativeHandleStoreError::Missing(handle.identifier().to_owned()));
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
        fn new(seed: u128, cancellation: CancellationToken) -> Result<Self, Box<dyn std::error::Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(seed + 1));
            let node_id = NodeId(format!("image-filter-{seed}"));
            let store = TestStore::new(seed + 2, attempt_id)?;
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
            let backend = Arc::new(backend);
            let scratch = authority.authorize_workspace(64 * 1024 * 1024)?;
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(seed + 4),
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
                PromptId(Uuid::from_u128(seed + 5)),
                attempt_id,
                node_id,
                cancellation,
                scratch,
                store.clone(),
                NativeNodeServices::checked(None, None, Some(compute))?,
            )?;
            Ok(Self { backend, store, context })
        }

        fn publish_image(
            &self,
            dimensions: [u64; 4],
            values: &[f32],
        ) -> Result<NativeOpaqueHandle, Box<dyn std::error::Error>> {
            let execution = self.context.compute_session()?.execution_context(&self.context)?;
            let image = ImageTensor::from_f32(
                &self.backend,
                &execution,
                dimensions[0],
                dimensions[1],
                dimensions[2],
                dimensions[3],
                values,
            )?;
            let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(payload)),
                &CancellationToken::default(),
            )?)
        }

        fn output_image(
            &self,
            outcome: NativeNodeOutcome,
        ) -> Result<ImageTensor, Box<dyn std::error::Error>> {
            let NativeNodeOutcome::Values { outputs, effects, .. } = outcome else {
                return Err("node did not return values".into());
            };
            assert!(effects.is_empty());
            let Some(NativeValue::Handle { value }) = outputs.first() else {
                return Err("node did not return an IMAGE handle".into());
            };
            let resolved = self.store.resolve(
                value,
                &image_type()?,
                &CancellationToken::default(),
            )?;
            let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
                return Err("IMAGE output did not contain a tensor".into());
            };
            payload.image().cloned().ok_or_else(|| "IMAGE output lost its canonical image".into())
        }
    }

    fn node(class_type: &str) -> Result<Arc<dyn NativeNode>, Box<dyn std::error::Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable { descriptor, node, .. }
                    if descriptor.class_type == class_type => Some(node),
                _ => None,
            })
            .ok_or_else(|| format!("{class_type} executable binding is absent").into())
    }

    fn handle(name: &str, value: NativeOpaqueHandle) -> (String, NativeValue) {
        (name.to_owned(), NativeValue::Handle { value })
    }

    fn number(name: &str, value: f64) -> (String, NativeValue) {
        (name.to_owned(), NativeValue::Primitive { value: NativePrimitive::Number(value) })
    }

    fn integer(name: &str, value: u64) -> (String, NativeValue) {
        (
            name.to_owned(),
            NativeValue::Primitive { value: NativePrimitive::UnsignedInteger(value) },
        )
    }

    fn combo(name: &str, value: &str) -> (String, NativeValue) {
        (
            name.to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "COMBO".to_owned(),
                value: Value::String(value.to_owned()),
            },
        )
    }

    fn dynamic_stats(value: &str, target_index: Option<u64>) -> (String, NativeValue) {
        let mut fields = serde_json::Map::from_iter([(
            "source_stats".to_owned(),
            Value::String(value.to_owned()),
        )]);
        if let Some(target_index) = target_index {
            fields.insert("target_index".to_owned(), Value::from(target_index));
        }
        (
            "source_stats".to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "COMFY_DYNAMICCOMBO_V3".to_owned(),
                value: Value::Object(fields),
            },
        )
    }

    #[test]
    fn every_assigned_filter_executes_and_publishes_a_canonical_image()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new(0x45, CancellationToken::default())?;
        let values = (0..48).map(|index| index as f32 / 47.0).collect::<Vec<_>>();
        let image = harness.publish_image([1, 4, 4, 3], &values)?;
        let reference_values = values.iter().rev().copied().collect::<Vec<_>>();
        let reference = harness.publish_image([1, 4, 4, 3], &reference_values)?;
        let cases = [
            (
                "Canny",
                BTreeMap::from([
                    handle("image", image.clone()),
                    number("low_threshold", 0.4),
                    number("high_threshold", 0.8),
                ]),
            ),
            (
                "ColorTransfer",
                BTreeMap::from([
                    handle("image_target", image.clone()),
                    handle("image_ref", reference.clone()),
                    combo("method", "reinhard_lab"),
                    dynamic_stats("per_frame", None),
                    number("strength", 1.0),
                ]),
            ),
            (
                "ColorTransfer",
                BTreeMap::from([
                    handle("image_target", image.clone()),
                    handle("image_ref", reference.clone()),
                    combo("method", "mkl_lab"),
                    dynamic_stats("uniform", None),
                    number("strength", 0.5),
                ]),
            ),
            (
                "ColorTransfer",
                BTreeMap::from([
                    handle("image_target", image.clone()),
                    handle("image_ref", reference),
                    combo("method", "histogram"),
                    dynamic_stats("target_frame", Some(10_000)),
                    number("strength", 1.0),
                ]),
            ),
            (
                "ImageAddNoise",
                BTreeMap::from([
                    handle("image", image.clone()),
                    integer("seed", 7),
                    number("strength", 0.25),
                ]),
            ),
            (
                "ImageBlend",
                BTreeMap::from([
                    handle("image1", image.clone()),
                    handle("image2", image.clone()),
                    number("blend_factor", 0.5),
                    combo("blend_mode", "soft_light"),
                ]),
            ),
            (
                "ImageBlur",
                BTreeMap::from([
                    handle("image", image.clone()),
                    integer("blur_radius", 1),
                    number("sigma", 1.0),
                ]),
            ),
            (
                "ImageQuantize",
                BTreeMap::from([
                    handle("image", image.clone()),
                    integer("colors", 8),
                    combo("dither", "bayer-2"),
                ]),
            ),
            (
                "ImageSharpen",
                BTreeMap::from([
                    handle("image", image.clone()),
                    integer("sharpen_radius", 1),
                    number("sigma", 1.0),
                    number("alpha", 1.0),
                ]),
            ),
            (
                "Morphology",
                BTreeMap::from([
                    handle("image", image),
                    combo("operation", "gradient"),
                    integer("kernel_size", 3),
                ]),
            ),
        ];
        for (class_type, inputs) in cases {
            let outcome = futures::executor::block_on(node(class_type)?.execute(
                harness.context.clone(),
                inputs,
            ))?;
            let output = harness.output_image(outcome)?;
            assert_eq!(output.dimensions()?, (1, 4, 4, 3));
            assert_eq!(output.as_f32_slice()?.len(), 48);
        }
        Ok(())
    }

    #[test]
    fn cancellation_and_invalid_values_fail_before_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let harness = Harness::new(0x99, cancellation.clone())?;
        let image = harness.publish_image([1, 3, 3, 3], &[0.5; 27])?;
        cancellation.cancel();
        let error = futures::executor::block_on(node("ImageBlur")?.execute(
            harness.context,
            BTreeMap::from([
                handle("image", image),
                integer("blur_radius", 1),
                number("sigma", 1.0),
            ]),
        ))
        .expect_err("cancelled execution must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(error.code, "execution_interrupted");
        assert!(required_number(
            &BTreeMap::from([number("strength", f64::NAN)]),
            "strength",
            0.0,
            1.0,
        )
        .is_err());
        Ok(())
    }
}
