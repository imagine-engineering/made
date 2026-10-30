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
    ImageTensor, NativeTensorPayload, NativeTensorRole, ResizeCrop, ResizeMode, ViewAccess,
};
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "ImageCompositeMasked",
    "JoinImageWithAlpha",
    "PorterDuffImageComposite",
    "SplitImageWithAlpha",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "image/compositing";
const IMPLEMENTATION_VERSION: &str = "source-3b27465f-e001e296-v1";
const MAX_RESOLUTION: i64 = 16_384;
const PORTER_DUFF_MODES: &[&str] = &[
    "ADD", "CLEAR", "DARKEN", "DST", "DST_ATOP", "DST_IN", "DST_OUT", "DST_OVER", "LIGHTEN",
    "MULTIPLY", "OVERLAY", "SCREEN", "SRC", "SRC_ATOP", "SRC_IN", "SRC_OUT", "SRC_OVER", "XOR",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompositingKind {
    CompositeMasked,
    JoinAlpha,
    PorterDuff,
    SplitAlpha,
}

impl CompositingKind {
    const fn feature_id(self) -> &'static str {
        match self {
            Self::CompositeMasked => "COMFY-NODE-0246",
            Self::JoinAlpha => "COMFY-NODE-0275",
            Self::PorterDuff => "COMFY-NODE-0486",
            Self::SplitAlpha => "COMFY-NODE-0632",
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::CompositeMasked => "ImageCompositeMasked",
            Self::JoinAlpha => "JoinImageWithAlpha",
            Self::PorterDuff => "PorterDuffImageComposite",
            Self::SplitAlpha => "SplitImageWithAlpha",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::CompositeMasked => "Image Composite Masked",
            Self::JoinAlpha => "Join Image with Alpha",
            Self::PorterDuff => "Porter-Duff Image Composite",
            Self::SplitAlpha => "Split Image with Alpha",
        }
    }

    const fn input_names(self) -> &'static [&'static str] {
        match self {
            Self::CompositeMasked => &["destination", "source", "x", "y", "resize_source"],
            Self::JoinAlpha => &["image", "alpha"],
            Self::PorterDuff => &[
                "source",
                "source_alpha",
                "destination",
                "destination_alpha",
                "mode",
            ],
            Self::SplitAlpha => &["image"],
        }
    }

    const fn optional_input_names(self) -> &'static [&'static str] {
        match self {
            Self::CompositeMasked => &["mask"],
            Self::JoinAlpha | Self::PorterDuff | Self::SplitAlpha => &[],
        }
    }

    const fn output_names(self) -> &'static [&'static str] {
        match self {
            Self::CompositeMasked | Self::JoinAlpha => &["image"],
            Self::PorterDuff | Self::SplitAlpha => &["image", "mask"],
        }
    }

    const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::CompositeMasked => &[
                "overlay image",
                "layer image",
                "paste image",
                "image composition",
            ],
            Self::JoinAlpha => &["add transparency", "apply alpha", "composite alpha", "RGBA"],
            Self::PorterDuff => &[
                "alpha composite",
                "blend modes",
                "layer blend",
                "transparency blend",
            ],
            Self::SplitAlpha => &["extract alpha", "separate transparency", "remove alpha"],
        }
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [
        CompositingKind::CompositeMasked,
        CompositingKind::JoinAlpha,
        CompositingKind::PorterDuff,
        CompositingKind::SplitAlpha,
    ]
    .into_iter()
    .map(native_node_binding)
    .collect()
}

fn native_node_binding(
    kind: CompositingKind,
) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let mut source_input_names = owned_names(kind.input_names());
    source_input_names.extend(owned_names(kind.optional_input_names()));
    let output_names = owned_names(kind.output_names());
    let source_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(&source_input_names, &[], &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = kind
        .input_names()
        .iter()
        .map(|name| input_descriptor(kind, name, true))
        .chain(
            kind.optional_input_names()
                .iter()
                .map(|name| input_descriptor(kind, name, false)),
        )
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = kind
        .output_names()
        .iter()
        .map(|name| {
            Ok(NativeOutputDescriptor {
                name: (*name).to_owned(),
                produced_type: NativeValueType::Handle(if *name == "mask" {
                    mask_type()
                } else {
                    image_type()
                }?),
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
            description: String::new(),
            output_names,
            search_aliases: owned_names(kind.aliases()),
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(ImageCompositingNode { kind }),
    })
}

fn owned_names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

fn input_descriptor(
    kind: CompositingKind,
    name: &str,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    let value_type = match name {
        "x" | "y" => NativeValueType::Primitive(NativePrimitiveType::Integer),
        "resize_source" => NativeValueType::Primitive(NativePrimitiveType::Boolean),
        "mode" => NativeValueType::NamedPreservedUnknown("COMBO".to_owned()),
        "mask" | "alpha" | "source_alpha" | "destination_alpha" => {
            NativeValueType::Handle(mask_type()?)
        }
        "destination" | "source" | "image" => NativeValueType::Handle(image_type()?),
        _ => {
            return Err(NativeNodeContractError::InvalidSourceSchema(format!(
                "{} has an unsupported input {name}",
                kind.class_type()
            )));
        }
    };
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([value_type])?,
        required,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: matches!(name, "x" | "y" | "resize_source" | "mode"),
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

fn mask_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Mask, "MASK")
}

#[derive(Debug)]
struct ImageCompositingNode {
    kind: CompositingKind,
}

impl NativeNode for ImageCompositingNode {
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
            let compute = context
                .compute_session()
                .map_err(|error| compute_failure(self.kind, error.to_string()))?;
            let tensor_context = compute
                .execution_context(&context)
                .map_err(|error| compute_failure(self.kind, error.to_string()))?;

            let outputs = match self.kind {
                CompositingKind::CompositeMasked => {
                    let destination = resolve_image(&context, &inputs, "destination")?;
                    let source = resolve_image(&context, &inputs, "source")?;
                    let mask = inputs
                        .contains_key("mask")
                        .then(|| resolve_mask(&context, &inputs, "mask"))
                        .transpose()?;
                    let output = composite_masked(
                        &destination,
                        &source,
                        mask.as_deref(),
                        required_integer(&inputs, "x")?,
                        required_integer(&inputs, "y")?,
                        required_boolean(&inputs, "resize_source")?,
                        compute.backend(),
                        &tensor_context,
                    )
                    .map_err(|error| tensor_failure(self.kind, error))?;
                    vec![publish_image(&context, output, self.kind)?]
                }
                CompositingKind::JoinAlpha => {
                    let image = resolve_image(&context, &inputs, "image")?;
                    let alpha = resolve_mask(&context, &inputs, "alpha")?;
                    let output = join_alpha(&image, &alpha, compute.backend(), &tensor_context)
                        .map_err(|error| tensor_failure(self.kind, error))?;
                    vec![publish_image(&context, output, self.kind)?]
                }
                CompositingKind::PorterDuff => {
                    let source = resolve_image(&context, &inputs, "source")?;
                    let source_alpha = resolve_mask(&context, &inputs, "source_alpha")?;
                    let destination = resolve_image(&context, &inputs, "destination")?;
                    let destination_alpha = resolve_mask(&context, &inputs, "destination_alpha")?;
                    let (image, mask) = porter_duff(
                        &source,
                        &source_alpha,
                        &destination,
                        &destination_alpha,
                        required_mode(&inputs)?,
                        compute.backend(),
                        &tensor_context,
                    )
                    .map_err(|error| tensor_failure(self.kind, error))?;
                    vec![
                        publish_image(&context, image, self.kind)?,
                        publish_mask(&context, mask, self.kind)?,
                    ]
                }
                CompositingKind::SplitAlpha => {
                    let image = resolve_image(&context, &inputs, "image")?;
                    let (image, mask) = split_alpha(&image, compute.backend(), &tensor_context)
                        .map_err(|error| tensor_failure(self.kind, error))?;
                    vec![
                        publish_image(&context, image, self.kind)?,
                        publish_mask(&context, mask, self.kind)?,
                    ]
                }
            };
            tensor_context
                .check()
                .map_err(|error| tensor_failure(self.kind, error.to_string()))?;
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
    kind: CompositingKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    let expected = kind.input_names().len();
    let maximum = expected + kind.optional_input_names().len();
    if inputs.len() < expected || inputs.len() > maximum {
        return Err(invalid_inputs(format!(
            "{} requires {} inputs{}",
            kind.class_type(),
            expected,
            if maximum == expected {
                String::new()
            } else {
                format!(" with at most {} optional input", maximum - expected)
            }
        )));
    }
    for name in inputs.keys() {
        if !kind.input_names().contains(&name.as_str())
            && !kind.optional_input_names().contains(&name.as_str())
        {
            return Err(invalid_inputs(format!(
                "{} received unknown input {name}",
                kind.class_type()
            )));
        }
    }
    for name in kind.input_names() {
        validate_input(name, inputs)?;
    }
    for name in kind.optional_input_names() {
        if inputs.contains_key(*name) {
            validate_input(name, inputs)?;
        }
    }
    Ok(())
}

fn validate_input(
    name: &str,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    match name {
        "x" | "y" => {
            required_integer(inputs, name)?;
        }
        "resize_source" => {
            required_boolean(inputs, name)?;
        }
        "mode" => {
            required_mode(inputs)?;
        }
        "mask" | "alpha" | "source_alpha" | "destination_alpha" => {
            required_handle(inputs, name, NativeHandleKind::Mask, "MASK")?;
        }
        "destination" | "source" | "image" => {
            required_handle(inputs, name, NativeHandleKind::Image, "IMAGE")?;
        }
        _ => return Err(invalid_inputs(format!("unsupported input {name}"))),
    }
    Ok(())
}

fn required_handle<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
    kind: NativeHandleKind,
    type_id: &str,
) -> Result<&'a crate::NativeOpaqueHandle, NativeNodeFailure> {
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

fn required_integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<u64, NativeNodeFailure> {
    let Some(NativeValue::Primitive {
        value: NativePrimitive::Integer(value),
    }) = inputs.get(name)
    else {
        return Err(invalid_inputs(format!("{name} must be an INT")));
    };
    if !(0..=MAX_RESOLUTION).contains(value) {
        return Err(invalid_inputs(format!(
            "{name} must be within 0 through {MAX_RESOLUTION}"
        )));
    }
    u64::try_from(*value).map_err(|_| invalid_inputs(format!("{name} is out of range")))
}

fn required_boolean(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<bool, NativeNodeFailure> {
    let Some(NativeValue::Primitive {
        value: NativePrimitive::Boolean(value),
    }) = inputs.get(name)
    else {
        return Err(invalid_inputs(format!("{name} must be a BOOLEAN")));
    };
    Ok(*value)
}

fn required_mode(inputs: &BTreeMap<String, NativeValue>) -> Result<&str, NativeNodeFailure> {
    let Some(NativeValue::PreservedUnknown {
        type_name,
        value: serde_json::Value::String(mode),
    }) = inputs.get("mode")
    else {
        return Err(invalid_inputs("mode must be a Porter-Duff mode"));
    };
    if type_name != "COMBO" || !PORTER_DUFF_MODES.contains(&mode.as_str()) {
        return Err(invalid_inputs(format!(
            "unsupported Porter-Duff mode {mode}"
        )));
    }
    Ok(mode)
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
    let handle = required_handle(inputs, name, NativeHandleKind::Image, "IMAGE")?;
    let expected = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected, &context.cancellation)
        .map_err(|error| handle_failure(name, error))?;
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(invalid_inputs(format!(
            "{name} did not resolve to a tensor payload"
        )));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(invalid_inputs(format!(
            "{name} did not resolve to an IMAGE payload"
        )));
    }
    let image = payload
        .image()
        .ok_or_else(|| invalid_inputs(format!("{name} has no canonical IMAGE tensor")))?;
    Ok(ResolvedImage {
        image: image.clone(),
        _resolved: resolved,
    })
}

fn resolve_mask(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<ResolvedImage, NativeNodeFailure> {
    let handle = required_handle(inputs, name, NativeHandleKind::Mask, "MASK")?;
    let expected = mask_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected, &context.cancellation)
        .map_err(|error| handle_failure(name, error))?;
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(invalid_inputs(format!(
            "{name} did not resolve to a tensor payload"
        )));
    };
    if payload.role() != NativeTensorRole::Mask {
        return Err(invalid_inputs(format!(
            "{name} did not resolve to a MASK payload"
        )));
    }
    let tensor = payload.tensor();
    let [batch, height, width] = tensor.descriptor().shape() else {
        return Err(invalid_inputs(format!(
            "{name} MASK tensor must have BHW shape"
        )));
    };
    let descriptor = tensor
        .descriptor()
        .reshaped_view(vec![*batch, *height, *width, 1])
        .map_err(|error| invalid_inputs(error.to_string()))?;
    let image = tensor
        .view(descriptor, ViewAccess::ReadOnly)
        .and_then(ImageTensor::from_tensor)
        .map_err(|error| invalid_inputs(format!("{name} MASK tensor is invalid: {error}")))?;
    Ok(ResolvedImage {
        image,
        _resolved: resolved,
    })
}

fn publish_image(
    context: &NativeNodeContext,
    image: ImageTensor,
    kind: CompositingKind,
) -> Result<NativeValue, NativeNodeFailure> {
    publish_tensor(context, image, NativeTensorRole::Image, kind)
}

fn publish_mask(
    context: &NativeNodeContext,
    mask: ImageTensor,
    kind: CompositingKind,
) -> Result<NativeValue, NativeNodeFailure> {
    publish_tensor(context, mask, NativeTensorRole::Mask, kind)
}

fn publish_tensor(
    context: &NativeNodeContext,
    image: ImageTensor,
    role: NativeTensorRole,
    kind: CompositingKind,
) -> Result<NativeValue, NativeNodeFailure> {
    let payload = NativeTensorPayload::from_image(role, image)
        .map_err(|error| tensor_failure(kind, error.to_string()))?;
    let handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(|error| handle_failure("output", error))?;
    Ok(NativeValue::Handle { value: handle })
}

fn composite_masked(
    destination: &ImageTensor,
    source: &ImageTensor,
    mask: Option<&ImageTensor>,
    x: u64,
    y: u64,
    resize_source: bool,
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, String> {
    let (destination_batch, destination_height, destination_width, destination_channels) =
        destination
            .dimensions()
            .map_err(|error| error.to_string())?;
    let (source_batch, source_height, source_width, source_channels) =
        source.dimensions().map_err(|error| error.to_string())?;
    let fixed_channels = if destination_channels < source_channels {
        destination_channels
    } else if destination_channels > source_channels {
        source_channels
            .checked_add(1)
            .ok_or_else(|| "source channel count overflowed".to_owned())?
    } else {
        source_channels
    };
    if fixed_channels != destination_channels {
        return Err(format!(
            "source alpha fix produced {fixed_channels} channels for a {destination_channels}-channel destination"
        ));
    }
    let fixed_values = alpha_fix_values(source, fixed_channels)?;
    let mut fixed_source = ImageTensor::from_f32(
        backend,
        context,
        source_batch,
        source_height,
        source_width,
        fixed_channels,
        &fixed_values,
    )
    .map_err(|error| error.to_string())?;
    if resize_source {
        fixed_source = fixed_source
            .resize(
                destination_width,
                destination_height,
                ResizeMode::Bilinear,
                ResizeCrop::Disabled,
                backend,
                context,
            )
            .map_err(|error| error.to_string())?;
    }
    let (_, source_height, source_width, source_channels) = fixed_source
        .dimensions()
        .map_err(|error| error.to_string())?;
    let source_values = repeat_batches(&fixed_source, destination_batch)?;
    let mask_values = match mask {
        Some(mask) => {
            let resized = mask
                .resize(
                    source_width,
                    source_height,
                    ResizeMode::Bilinear,
                    ResizeCrop::Disabled,
                    backend,
                    context,
                )
                .map_err(|error| error.to_string())?;
            repeat_batches(&resized, destination_batch)?
        }
        None => vec![1.0; checked_elements(destination_batch, source_height, source_width, 1)?],
    };
    let mut output = destination
        .as_f32_slice()
        .map_err(|error| error.to_string())?
        .to_vec();
    let x = x.min(destination_width);
    let y = y.min(destination_height);
    let copy_width = source_width.min(destination_width.saturating_sub(x));
    let copy_height = source_height.min(destination_height.saturating_sub(y));
    for batch in 0..destination_batch {
        context.check().map_err(|error| error.to_string())?;
        for row in 0..copy_height {
            for column in 0..copy_width {
                let mask_index = offset(batch, row, column, 0, source_height, source_width, 1)?;
                let mask_value = *mask_values
                    .get(mask_index)
                    .ok_or_else(|| "composite mask indexing exceeded storage".to_owned())?;
                for channel in 0..source_channels {
                    let source_index = offset(
                        batch,
                        row,
                        column,
                        channel,
                        source_height,
                        source_width,
                        source_channels,
                    )?;
                    let destination_index = offset(
                        batch,
                        y + row,
                        x + column,
                        channel,
                        destination_height,
                        destination_width,
                        destination_channels,
                    )?;
                    let source_value = *source_values
                        .get(source_index)
                        .ok_or_else(|| "composite source indexing exceeded storage".to_owned())?;
                    let destination_value = output.get_mut(destination_index).ok_or_else(|| {
                        "composite destination indexing exceeded storage".to_owned()
                    })?;
                    *destination_value =
                        mask_value * source_value + (1.0 - mask_value) * *destination_value;
                }
            }
        }
    }
    ImageTensor::from_f32(
        backend,
        context,
        destination_batch,
        destination_height,
        destination_width,
        destination_channels,
        &output,
    )
    .map_err(|error| error.to_string())
}

fn alpha_fix_values(source: &ImageTensor, output_channels: u64) -> Result<Vec<f32>, String> {
    let (batch, height, width, source_channels) =
        source.dimensions().map_err(|error| error.to_string())?;
    let pixel_count = checked_elements(batch, height, width, 1)?;
    let capacity = checked_elements(batch, height, width, output_channels)?;
    let source_values = source.as_f32_slice().map_err(|error| error.to_string())?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|error| format!("source alpha-fix allocation failed: {error}"))?;
    let source_channels_usize = usize::try_from(source_channels)
        .map_err(|_| "source channel count is too large".to_owned())?;
    let output_channels_usize = usize::try_from(output_channels)
        .map_err(|_| "output channel count is too large".to_owned())?;
    for pixel in 0..pixel_count {
        let start = pixel
            .checked_mul(source_channels_usize)
            .ok_or_else(|| "source pixel offset overflowed".to_owned())?;
        let end = start
            .checked_add(source_channels_usize)
            .ok_or_else(|| "source pixel end overflowed".to_owned())?;
        let channels = source_values
            .get(start..end)
            .ok_or_else(|| "source alpha-fix indexing exceeded storage".to_owned())?;
        let copied = output_channels_usize.min(source_channels_usize);
        output.extend_from_slice(&channels[..copied]);
        if output_channels_usize > source_channels_usize {
            output.push(1.0);
        }
    }
    Ok(output)
}

fn join_alpha(
    image: &ImageTensor,
    alpha: &ImageTensor,
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, String> {
    let (image_batch, height, width, image_channels) =
        image.dimensions().map_err(|error| error.to_string())?;
    let (alpha_batch, _, _, _) = alpha.dimensions().map_err(|error| error.to_string())?;
    let batch = image_batch.max(alpha_batch);
    let alpha = alpha
        .resize(
            width,
            height,
            ResizeMode::Bilinear,
            ResizeCrop::Disabled,
            backend,
            context,
        )
        .map_err(|error| error.to_string())?;
    let image_values = repeat_batches(image, batch)?;
    let alpha_values = repeat_batches(&alpha, batch)?;
    let rgb_channels = image_channels.min(3);
    let output_channels = rgb_channels
        .checked_add(1)
        .ok_or_else(|| "joined image channel count overflowed".to_owned())?;
    if !matches!(output_channels, 1 | 3 | 4) {
        return Err(format!(
            "joining alpha to a {image_channels}-channel IMAGE produces unsupported {output_channels}-channel output"
        ));
    }
    let pixel_count = checked_elements(batch, height, width, 1)?;
    let capacity = checked_elements(batch, height, width, output_channels)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|error| format!("join-alpha allocation failed: {error}"))?;
    let image_channels_usize = usize::try_from(image_channels)
        .map_err(|_| "image channel count is too large".to_owned())?;
    let rgb_channels_usize =
        usize::try_from(rgb_channels).map_err(|_| "RGB channel count is too large".to_owned())?;
    for pixel in 0..pixel_count {
        context.check().map_err(|error| error.to_string())?;
        let image_start = pixel
            .checked_mul(image_channels_usize)
            .ok_or_else(|| "join-alpha image offset overflowed".to_owned())?;
        let image_end = image_start
            .checked_add(rgb_channels_usize)
            .ok_or_else(|| "join-alpha image end overflowed".to_owned())?;
        output.extend_from_slice(
            image_values
                .get(image_start..image_end)
                .ok_or_else(|| "join-alpha image indexing exceeded storage".to_owned())?,
        );
        let alpha_value = alpha_values
            .get(pixel)
            .ok_or_else(|| "join-alpha mask indexing exceeded storage".to_owned())?;
        output.push(1.0 - alpha_value);
    }
    ImageTensor::from_f32(
        backend,
        context,
        batch,
        height,
        width,
        output_channels,
        &output,
    )
    .map_err(|error| error.to_string())
}

fn split_alpha(
    image: &ImageTensor,
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<(ImageTensor, ImageTensor), String> {
    let (batch, height, width, channels) = image.dimensions().map_err(|error| error.to_string())?;
    let output_channels = channels.min(3);
    let pixel_count = checked_elements(batch, height, width, 1)?;
    let image_capacity = checked_elements(batch, height, width, output_channels)?;
    let values = image.as_f32_slice().map_err(|error| error.to_string())?;
    let channels_usize =
        usize::try_from(channels).map_err(|_| "image channel count is too large".to_owned())?;
    let output_channels_usize = usize::try_from(output_channels)
        .map_err(|_| "output channel count is too large".to_owned())?;
    let mut image_values = Vec::new();
    image_values
        .try_reserve_exact(image_capacity)
        .map_err(|error| format!("split image allocation failed: {error}"))?;
    let mut mask_values = Vec::new();
    mask_values
        .try_reserve_exact(pixel_count)
        .map_err(|error| format!("split mask allocation failed: {error}"))?;
    for pixel in 0..pixel_count {
        context.check().map_err(|error| error.to_string())?;
        let start = pixel
            .checked_mul(channels_usize)
            .ok_or_else(|| "split image offset overflowed".to_owned())?;
        let end = start
            .checked_add(channels_usize)
            .ok_or_else(|| "split image end overflowed".to_owned())?;
        let pixel_values = values
            .get(start..end)
            .ok_or_else(|| "split image indexing exceeded storage".to_owned())?;
        image_values.extend_from_slice(&pixel_values[..output_channels_usize]);
        let alpha = pixel_values.get(3).copied().unwrap_or(1.0);
        mask_values.push(1.0 - alpha);
    }
    let image = ImageTensor::from_f32(
        backend,
        context,
        batch,
        height,
        width,
        output_channels,
        &image_values,
    )
    .map_err(|error| error.to_string())?;
    let mask = ImageTensor::from_f32(backend, context, batch, height, width, 1, &mask_values)
        .map_err(|error| error.to_string())?;
    Ok((image, mask))
}

#[derive(Clone, Copy, Debug)]
struct PremultipliedPixel {
    source_alpha: f32,
    destination_alpha: f32,
    source: f32,
    destination: f32,
}

fn porter_duff(
    source: &ImageTensor,
    source_alpha: &ImageTensor,
    destination: &ImageTensor,
    destination_alpha: &ImageTensor,
    mode: &str,
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<(ImageTensor, ImageTensor), String> {
    let (destination_batch, height, width, destination_channels) = destination
        .dimensions()
        .map_err(|error| error.to_string())?;
    let (source_batch, source_height, source_width, source_channels) =
        source.dimensions().map_err(|error| error.to_string())?;
    if source_channels != destination_channels {
        return Err(format!(
            "source and destination channel counts differ: {source_channels} != {destination_channels}"
        ));
    }
    let destination_alpha = destination_alpha
        .resize(
            width,
            height,
            ResizeMode::Bicubic,
            ResizeCrop::Center,
            backend,
            context,
        )
        .map_err(|error| error.to_string())?;
    let source =
        if (source_height, source_width) != (height, width) || source_batch != destination_batch {
            source
                .resize(
                    width,
                    height,
                    ResizeMode::Bicubic,
                    ResizeCrop::Center,
                    backend,
                    context,
                )
                .map_err(|error| error.to_string())?
        } else {
            source.clone()
        };
    let (_, alpha_height, alpha_width, _) = destination_alpha
        .dimensions()
        .map_err(|error| error.to_string())?;
    let source_alpha = source_alpha
        .resize(
            alpha_width,
            alpha_height,
            ResizeMode::Bicubic,
            ResizeCrop::Center,
            backend,
            context,
        )
        .map_err(|error| error.to_string())?;
    let source_alpha_batch = source_alpha
        .dimensions()
        .map_err(|error| error.to_string())?
        .0;
    let destination_alpha_batch = destination_alpha
        .dimensions()
        .map_err(|error| error.to_string())?
        .0;
    let batch = source_batch
        .min(source_alpha_batch)
        .min(destination_batch)
        .min(destination_alpha_batch);
    let source_values = source.as_f32_slice().map_err(|error| error.to_string())?;
    let source_alpha_values = source_alpha
        .as_f32_slice()
        .map_err(|error| error.to_string())?;
    let destination_values = destination
        .as_f32_slice()
        .map_err(|error| error.to_string())?;
    let destination_alpha_values = destination_alpha
        .as_f32_slice()
        .map_err(|error| error.to_string())?;
    let pixel_count = checked_elements(batch, height, width, 1)?;
    let image_capacity = checked_elements(batch, height, width, destination_channels)?;
    let channels_usize = usize::try_from(destination_channels)
        .map_err(|_| "image channel count is too large".to_owned())?;
    let mut image_values = Vec::new();
    image_values
        .try_reserve_exact(image_capacity)
        .map_err(|error| format!("Porter-Duff image allocation failed: {error}"))?;
    let mut mask_values = Vec::new();
    mask_values
        .try_reserve_exact(pixel_count)
        .map_err(|error| format!("Porter-Duff mask allocation failed: {error}"))?;
    for pixel in 0..pixel_count {
        context.check().map_err(|error| error.to_string())?;
        let source_alpha_value = 1.0
            - source_alpha_values
                .get(pixel)
                .copied()
                .ok_or_else(|| "source alpha indexing exceeded storage".to_owned())?;
        let destination_alpha_value = 1.0
            - destination_alpha_values
                .get(pixel)
                .copied()
                .ok_or_else(|| "destination alpha indexing exceeded storage".to_owned())?;
        let alpha = porter_duff_alpha(mode, source_alpha_value, destination_alpha_value)?;
        mask_values.push(1.0 - alpha);
        let start = pixel
            .checked_mul(channels_usize)
            .ok_or_else(|| "Porter-Duff image offset overflowed".to_owned())?;
        let end = start
            .checked_add(channels_usize)
            .ok_or_else(|| "Porter-Duff image end overflowed".to_owned())?;
        let source_pixel = source_values
            .get(start..end)
            .ok_or_else(|| "source image indexing exceeded storage".to_owned())?;
        let destination_pixel = destination_values
            .get(start..end)
            .ok_or_else(|| "destination image indexing exceeded storage".to_owned())?;
        for (&source_value, &destination_value) in source_pixel.iter().zip(destination_pixel.iter())
        {
            let premultiplied = PremultipliedPixel {
                source_alpha: source_alpha_value,
                destination_alpha: destination_alpha_value,
                source: source_value * source_alpha_value,
                destination: destination_value * destination_alpha_value,
            };
            let value = porter_duff_channel(mode, premultiplied)?;
            image_values.push(if alpha > 1.0e-5 {
                (value / alpha).clamp(0.0, 1.0)
            } else {
                0.0
            });
        }
    }
    let image = ImageTensor::from_f32(
        backend,
        context,
        batch,
        height,
        width,
        destination_channels,
        &image_values,
    )
    .map_err(|error| error.to_string())?;
    let mask = ImageTensor::from_f32(backend, context, batch, height, width, 1, &mask_values)
        .map_err(|error| error.to_string())?;
    Ok((image, mask))
}

fn porter_duff_alpha(mode: &str, source: f32, destination: f32) -> Result<f32, String> {
    let alpha = match mode {
        "ADD" => (source + destination).clamp(0.0, 1.0),
        "CLEAR" => 0.0,
        "DARKEN" | "LIGHTEN" | "OVERLAY" | "SCREEN" => source + destination - source * destination,
        "DST" => destination,
        "DST_ATOP" => source,
        "DST_IN" | "MULTIPLY" | "SRC_IN" => source * destination,
        "DST_OUT" => (1.0 - source) * destination,
        "DST_OVER" => destination + (1.0 - destination) * source,
        "SRC" => source,
        "SRC_ATOP" => destination,
        "SRC_OUT" => (1.0 - destination) * source,
        "SRC_OVER" => source + (1.0 - source) * destination,
        "XOR" => (1.0 - destination) * source + (1.0 - source) * destination,
        _ => return Err(format!("unsupported Porter-Duff mode {mode}")),
    };
    Ok(alpha)
}

fn porter_duff_channel(mode: &str, pixel: PremultipliedPixel) -> Result<f32, String> {
    let source_alpha = pixel.source_alpha;
    let destination_alpha = pixel.destination_alpha;
    let source = pixel.source;
    let destination = pixel.destination;
    let value = match mode {
        "ADD" => (source + destination).clamp(0.0, 1.0),
        "CLEAR" => 0.0,
        "DARKEN" => {
            (1.0 - destination_alpha) * source
                + (1.0 - source_alpha) * destination
                + torch_min(source, destination)
        }
        "DST" => destination,
        "DST_ATOP" => source_alpha * destination + (1.0 - destination_alpha) * source,
        "DST_IN" => destination * source_alpha,
        "DST_OUT" => (1.0 - source_alpha) * destination,
        "DST_OVER" => destination + (1.0 - destination_alpha) * source,
        "LIGHTEN" => {
            (1.0 - destination_alpha) * source
                + (1.0 - source_alpha) * destination
                + torch_max(source, destination)
        }
        "MULTIPLY" => source * destination,
        "OVERLAY" => {
            if 2.0 * destination < destination_alpha {
                2.0 * source * destination
            } else {
                source_alpha * destination_alpha
                    - 2.0 * (destination_alpha - source) * (source_alpha - destination)
            }
        }
        "SCREEN" => source + destination - source * destination,
        "SRC" => source,
        "SRC_ATOP" => destination_alpha * source + (1.0 - source_alpha) * destination,
        "SRC_IN" => source * destination_alpha,
        "SRC_OUT" => (1.0 - destination_alpha) * source,
        "SRC_OVER" => source + (1.0 - source_alpha) * destination,
        "XOR" => (1.0 - destination_alpha) * source + (1.0 - source_alpha) * destination,
        _ => return Err(format!("unsupported Porter-Duff mode {mode}")),
    };
    Ok(value)
}

fn torch_min(left: f32, right: f32) -> f32 {
    if left.is_nan() || right.is_nan() {
        f32::NAN
    } else {
        left.min(right)
    }
}

fn torch_max(left: f32, right: f32) -> f32 {
    if left.is_nan() || right.is_nan() {
        f32::NAN
    } else {
        left.max(right)
    }
}

fn repeat_batches(image: &ImageTensor, target_batch: u64) -> Result<Vec<f32>, String> {
    let (batch, height, width, channels) = image.dimensions().map_err(|error| error.to_string())?;
    let batch_elements = checked_elements(1, height, width, channels)?;
    let target_elements = checked_elements(target_batch, height, width, channels)?;
    let values = image.as_f32_slice().map_err(|error| error.to_string())?;
    if target_batch > 0 && batch == 0 {
        return Err("cannot repeat an empty image batch to a non-empty batch".to_owned());
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(target_elements)
        .map_err(|error| format!("batch-repeat allocation failed: {error}"))?;
    for output_batch in 0..target_batch {
        let source_batch = output_batch % batch;
        let source_batch_usize = usize::try_from(source_batch)
            .map_err(|_| "source batch index is too large".to_owned())?;
        let start = source_batch_usize
            .checked_mul(batch_elements)
            .ok_or_else(|| "batch-repeat offset overflowed".to_owned())?;
        let end = start
            .checked_add(batch_elements)
            .ok_or_else(|| "batch-repeat end overflowed".to_owned())?;
        output.extend_from_slice(
            values
                .get(start..end)
                .ok_or_else(|| "batch-repeat indexing exceeded storage".to_owned())?,
        );
    }
    Ok(output)
}

fn checked_elements(batch: u64, height: u64, width: u64, channels: u64) -> Result<usize, String> {
    let count = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| "image element count overflowed".to_owned())?;
    usize::try_from(count).map_err(|_| "image element count exceeds this platform".to_owned())
}

fn offset(
    batch: u64,
    row: u64,
    column: u64,
    channel: u64,
    height: u64,
    width: u64,
    channels: u64,
) -> Result<usize, String> {
    let value = batch
        .checked_mul(height)
        .and_then(|value| value.checked_add(row))
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_add(column))
        .and_then(|value| value.checked_mul(channels))
        .and_then(|value| value.checked_add(channel))
        .ok_or_else(|| "image offset overflowed".to_owned())?;
    usize::try_from(value).map_err(|_| "image offset exceeds this platform".to_owned())
}

fn check_cancellation(
    context: &NativeNodeContext,
    kind: CompositingKind,
) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(kind))
}

fn handle_failure(name: &str, error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message: format!("{name} image-compositing execution was interrupted"),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: false,
        }
    } else {
        NativeNodeFailure {
            code: "invalid_image_handle".to_owned(),
            message: format!("{name} handle is unavailable: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(kind: CompositingKind, message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "image_compositing_compute_unavailable".to_owned(),
        message: format!("{}: {}", kind.class_type(), message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
    }
}

fn tensor_failure(kind: CompositingKind, message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "image_compositing_failed".to_owned(),
        message: format!("{}: {}", kind.class_type(), message.into()),
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

fn interrupted_failure(kind: CompositingKind) -> NativeNodeFailure {
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
    use sim_tensor::{CpuWorkspaceAuthority, ExecutionContext, StreamId, TensorError};
    use sim_types::CancellationToken;
    use serde_json::Value;
    use std::error::Error;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-compositing-comfy-node-0246/fixture.json"
    ));

    fn tensor_context<'a>(
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, TensorError> {
        Ok(ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation,
        })
    }

    fn fixture_values(fixture: &Value, pointer: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        fixture
            .pointer(pointer)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("fixture array {pointer} is missing").into())
            .and_then(|values| {
                values
                    .iter()
                    .map(|value| {
                        value.as_f64().map(|value| value as f32).ok_or_else(|| {
                            format!("fixture value in {pointer} is not numeric").into()
                        })
                    })
                    .collect()
            })
    }

    fn image(
        backend: &sim_tensor::CpuBackend,
        context: &ExecutionContext<'_>,
        shape: [u64; 4],
        values: &[f32],
    ) -> Result<ImageTensor, TensorError> {
        ImageTensor::from_f32(
            backend, context, shape[0], shape[1], shape[2], shape[3], values,
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "value {index} differs: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn source_descriptors_match_the_four_assigned_rows() -> Result<(), Box<dyn Error>> {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 4);
        assert_eq!(
            bindings
                .iter()
                .map(NativeNodeBinding::feature_id)
                .collect::<Vec<_>>(),
            [
                "COMFY-NODE-0246",
                "COMFY-NODE-0275",
                "COMFY-NODE-0486",
                "COMFY-NODE-0632",
            ]
        );
        for binding in bindings {
            let NativeNodeBinding::Executable {
                descriptor,
                presentation,
                ..
            } = binding
            else {
                return Err("assigned image-compositing row is not executable".into());
            };
            assert_eq!(presentation.category, CATEGORY);
            assert!(!presentation.is_deprecated);
            assert!(!presentation.is_experimental);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert!(descriptor.source_schema.is_some());
        }
        Ok(())
    }

    #[test]
    fn join_and_split_preserve_sim_inverted_alpha_semantics() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = tensor_context(&authority, &cancellation)?;
        let rgb = image(
            &backend,
            &context,
            [1, 1, 2, 3],
            &fixture_values(&fixture, "/join/input_image")?,
        )?;
        let mask = image(
            &backend,
            &context,
            [1, 1, 2, 1],
            &fixture_values(&fixture, "/join/input_mask")?,
        )?;
        let joined = join_alpha(&rgb, &mask, &backend, &context)?;
        assert_close(
            joined.as_f32_slice()?,
            &fixture_values(&fixture, "/join/output_rgba")?,
        );
        let (split_image, split_mask) = split_alpha(&joined, &backend, &context)?;
        assert_close(split_image.as_f32_slice()?, rgb.as_f32_slice()?);
        assert_close(split_mask.as_f32_slice()?, mask.as_f32_slice()?);
        Ok(())
    }

    #[test]
    fn masked_composite_clips_offsets_and_broadcasts_mask() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = tensor_context(&authority, &cancellation)?;
        let destination = image(
            &backend,
            &context,
            [1, 2, 2, 3],
            &fixture_values(&fixture, "/composite/destination")?,
        )?;
        let source = image(
            &backend,
            &context,
            [1, 2, 2, 3],
            &fixture_values(&fixture, "/composite/source")?,
        )?;
        let mask = image(
            &backend,
            &context,
            [1, 2, 2, 1],
            &fixture_values(&fixture, "/composite/mask")?,
        )?;
        let output = composite_masked(
            &destination,
            &source,
            Some(&mask),
            1,
            0,
            false,
            &backend,
            &context,
        )?;
        assert_close(
            output.as_f32_slice()?,
            &fixture_values(&fixture, "/composite/output")?,
        );
        Ok(())
    }

    #[test]
    fn porter_duff_modes_match_premultiplied_source_oracle() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let source = fixture_values(&fixture, "/porter/source")?[0];
        let destination = fixture_values(&fixture, "/porter/destination")?[0];
        let source_alpha = fixture_values(&fixture, "/porter/source_alpha")?[0];
        let destination_alpha = fixture_values(&fixture, "/porter/destination_alpha")?[0];
        let pixel = PremultipliedPixel {
            source_alpha,
            destination_alpha,
            source: source * source_alpha,
            destination: destination * destination_alpha,
        };
        for mode in PORTER_DUFF_MODES {
            let alpha = porter_duff_alpha(mode, source_alpha, destination_alpha)?;
            let premultiplied = porter_duff_channel(mode, pixel)?;
            let output = if alpha > 1.0e-5 {
                (premultiplied / alpha).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let expected = fixture
                .pointer(&format!("/porter/outputs/{mode}"))
                .and_then(Value::as_array)
                .ok_or_else(|| format!("missing Porter-Duff fixture for {mode}"))?;
            let expected_output = expected[0]
                .as_f64()
                .ok_or("Porter-Duff image fixture is not numeric")?
                as f32;
            let expected_mask = expected[1]
                .as_f64()
                .ok_or("Porter-Duff mask fixture is not numeric")?
                as f32;
            assert_close(&[output, 1.0 - alpha], &[expected_output, expected_mask]);
        }
        Ok(())
    }
}
