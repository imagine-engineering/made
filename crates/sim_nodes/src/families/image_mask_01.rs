use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeHandleKind, NativeHandleStoreError,
    NativeHandleType, NativeInputDescriptor, NativeInputRequirement, NativeInputSchemaMetadata,
    NativeNode, NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext,
    NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor,
    NativePortCardinality, NativePrimitive, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
    native_value_type_for_output_schema, native_value_types_for_input_schema,
};
use sim_tensor::{
    DType, DeviceId, ImageTensor, NativeTensorPayload, NativeTensorRole, ResizeCrop, ResizeMode,
    TensorDescriptor, generated_native_diffusion::tensor_to_f32,
};
use futures::future::BoxFuture;
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "BatchMasksNode",
    "CropMask",
    "FeatherMask",
    "GrowMask",
    "ImageColorToMask",
    "ImageToMask",
    "InvertMask",
    "MaskComposite",
    "MaskPreview",
    "MaskToImage",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "image/mask";
const IMPLEMENTATION_VERSION: &str = "source-9ff6c44f-96ec39e8-v1";
const MAX_RESOLUTION: i64 = 16_384;
const MAX_AUTOGROW_INPUTS: usize = 50;
const MAX_MORPHOLOGY_VISITS: u64 = 100_000_000;
const COMPOSITE_OPERATIONS: &[&str] = &["multiply", "add", "subtract", "and", "or", "xor"];
const IMAGE_CHANNELS: &[&str] = &["red", "green", "blue", "alpha"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MaskKind {
    Batch,
    Crop,
    Feather,
    Grow,
    ImageColorToMask,
    ImageToMask,
    Invert,
    Composite,
    Preview,
    ToImage,
}

impl MaskKind {
    const ALL: [Self; 10] = [
        Self::Batch,
        Self::Crop,
        Self::Feather,
        Self::Grow,
        Self::ImageColorToMask,
        Self::ImageToMask,
        Self::Invert,
        Self::Composite,
        Self::Preview,
        Self::ToImage,
    ];

    const fn feature_id(self) -> &'static str {
        match self {
            Self::Batch => "COMFY-NODE-0019",
            Self::Crop => "COMFY-NODE-0126",
            Self::Feather => "COMFY-NODE-0171",
            Self::Grow => "COMFY-NODE-0219",
            Self::ImageColorToMask => "COMFY-NODE-0244",
            Self::ImageToMask => "COMFY-NODE-0268",
            Self::Invert => "COMFY-NODE-0273",
            Self::Composite => "COMFY-NODE-0399",
            Self::Preview => "COMFY-NODE-0400",
            Self::ToImage => "COMFY-NODE-0401",
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::Batch => "BatchMasksNode",
            Self::Crop => "CropMask",
            Self::Feather => "FeatherMask",
            Self::Grow => "GrowMask",
            Self::ImageColorToMask => "ImageColorToMask",
            Self::ImageToMask => "ImageToMask",
            Self::Invert => "InvertMask",
            Self::Composite => "MaskComposite",
            Self::Preview => "MaskPreview",
            Self::ToImage => "MaskToImage",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::Batch => "Batch Masks",
            Self::Crop => "Crop Mask",
            Self::Feather => "Feather Mask",
            Self::Grow => "Grow Mask",
            Self::ImageColorToMask => "Convert Image Color to Mask",
            Self::ImageToMask => "Convert Image to Mask",
            Self::Invert => "Invert Mask",
            Self::Composite => "Combine Masks",
            Self::Preview => "Preview Mask",
            Self::ToImage => "Convert Mask to Image",
        }
    }

    const fn input_names(self) -> &'static [&'static str] {
        match self {
            Self::Batch => &[],
            Self::Crop => &["mask", "x", "y", "width", "height"],
            Self::Feather => &["mask", "left", "top", "right", "bottom"],
            Self::Grow => &["mask", "expand", "tapered_corners"],
            Self::ImageColorToMask => &["image", "color"],
            Self::ImageToMask => &["image", "channel"],
            Self::Invert | Self::Preview | Self::ToImage => &["mask"],
            Self::Composite => &["destination", "source", "x", "y", "operation"],
        }
    }

    const fn output_name(self) -> Option<&'static str> {
        match self {
            Self::Preview => None,
            Self::ToImage => Some("image"),
            _ => Some("mask"),
        }
    }

    const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Batch => &["combine masks", "stack masks", "merge masks"],
            Self::Crop => &["cut mask", "extract mask region", "mask slice"],
            Self::Feather => &["soft edge mask", "blur mask edges", "gradient mask edge"],
            Self::Grow => &["expand mask", "shrink mask"],
            Self::ImageColorToMask => &["color keying", "chroma key"],
            Self::ImageToMask => &["extract channel", "channel to mask"],
            Self::Invert => &["reverse mask", "flip mask"],
            Self::Composite => &["combine masks", "blend masks", "layer masks", "masks composition"],
            Self::Preview => &["show mask", "view mask", "inspect mask", "debug mask"],
            Self::ToImage => &["convert mask"],
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Preview => "Saves the input images to your ComfyUI output directory.",
            _ => "",
        }
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    MaskKind::ALL.into_iter().map(native_node_binding).collect()
}

fn native_node_binding(kind: MaskKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let dynamic_schema = catalog_schema.dynamic_inputs.clone();
    let input_names = owned_names(kind.input_names());
    let output_names = kind.output_name().into_iter().map(str::to_owned).collect::<Vec<_>>();
    let mut source_schema = catalog_schema
        .bind_execution_ports(&input_names, &dynamic_schema, &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let mut inputs = catalog_schema
        .inputs
        .iter()
        .map(|input| source_input_descriptor(input, false))
        .collect::<Result<Vec<_>, _>>()?;
    if kind == MaskKind::Preview {
        source_schema.inputs.extend([
            NativeInputSchemaMetadata::compatibility("prompt", "PROMPT"),
            NativeInputSchemaMetadata::compatibility("extra_pnginfo", "EXTRA_PNGINFO"),
        ]);
        inputs.extend([
            hidden_input("prompt", "PROMPT")?,
            hidden_input("extra_pnginfo", "EXTRA_PNGINFO")?,
        ]);
    }
    let dynamic_inputs = dynamic_schema
        .iter()
        .map(|dynamic| {
            Ok(NativeDynamicInputDescriptor {
                name_template: dynamic.identity.clone(),
                start_index: dynamic.start_index,
                minimum_count: dynamic.minimum_count,
                maximum_count: dynamic.maximum_count,
                input: NativeInputDescriptor {
                    name: dynamic.input.name.clone(),
                    accepted_types: native_value_types_for_input_schema(&dynamic.input).map_err(
                        |error| NativeNodeContractError::InvalidSourceSchema(error.to_string()),
                    )?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: false,
                },
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
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
    let output_node = kind == MaskKind::Preview;
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs,
            outputs,
            output_node,
            effect: if output_node {
                NativeEffectClass::WritesArtifact
            } else {
                NativeEffectClass::Pure
            },
            cache: if output_node {
                NativeCachePolicy::Never
            } else {
                NativeCachePolicy::InputIdentity
            },
        },
        presentation: NativeNodePresentation {
            display_name: kind.display_name().to_owned(),
            category: CATEGORY.to_owned(),
            description: kind.description().to_owned(),
            output_names,
            search_aliases: owned_names(kind.aliases()),
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(ImageMaskNode { kind }),
    })
}

fn source_input_descriptor(
    input: &crate::CatalogNodeInputSchemaMetadata,
    hidden: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    let accepts_handles = input
        .schema
        .source_type_names
        .iter()
        .any(|name| matches!(name.as_str(), "IMAGE" | "MASK"));
    Ok(NativeInputDescriptor {
        name: input.schema.name.clone(),
        accepted_types: native_value_types_for_input_schema(&input.schema)
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?,
        required: input.requirement == NativeInputRequirement::Required,
        hidden,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: !accepts_handles,
    })
}

fn hidden_input(
    name: &str,
    type_name: &str,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([NativeValueType::NamedPreservedUnknown(
            type_name.to_owned(),
        )])?,
        required: false,
        hidden: true,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: true,
    })
}

fn owned_names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

fn mask_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Mask, "MASK")
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

#[derive(Debug)]
struct ImageMaskNode {
    kind: MaskKind,
}

impl NativeNode for ImageMaskNode {
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
            let execution = compute
                .execution_context(&context)
                .map_err(compute_failure)?;
            let result = match self.kind {
                MaskKind::Batch => {
                    let handles = dynamic_mask_handles(&inputs)?;
                    let masks = handles
                        .iter()
                        .map(|handle| resolve_mask(&context, handle, self.kind))
                        .collect::<Result<Vec<_>, _>>()?;
                    let output = batch_masks(
                        &masks,
                        compute.backend(),
                        &execution,
                        &context,
                        self.kind,
                    )?;
                    publish_mask_outcome(&context, output, self.kind)
                }
                MaskKind::Crop => {
                    let mask = resolve_named_mask(&context, &inputs, "mask", self.kind)?;
                    let output = crop_mask(
                        &mask,
                        required_unsigned(&inputs, "x", 0, MAX_RESOLUTION as u64)?,
                        required_unsigned(&inputs, "y", 0, MAX_RESOLUTION as u64)?,
                        required_unsigned(&inputs, "width", 1, MAX_RESOLUTION as u64)?,
                        required_unsigned(&inputs, "height", 1, MAX_RESOLUTION as u64)?,
                        &context,
                        self.kind,
                    )?;
                    publish_mask_outcome(&context, output, self.kind)
                }
                MaskKind::Feather => {
                    let mask = resolve_named_mask(&context, &inputs, "mask", self.kind)?;
                    let output = feather_mask(
                        mask,
                        [
                            required_unsigned(&inputs, "left", 0, MAX_RESOLUTION as u64)?,
                            required_unsigned(&inputs, "top", 0, MAX_RESOLUTION as u64)?,
                            required_unsigned(&inputs, "right", 0, MAX_RESOLUTION as u64)?,
                            required_unsigned(&inputs, "bottom", 0, MAX_RESOLUTION as u64)?,
                        ],
                        &context,
                        self.kind,
                    )?;
                    publish_mask_outcome(&context, output, self.kind)
                }
                MaskKind::Grow => {
                    let mask = resolve_named_mask(&context, &inputs, "mask", self.kind)?;
                    let output = grow_mask(
                        mask,
                        required_signed(&inputs, "expand", -MAX_RESOLUTION, MAX_RESOLUTION)?,
                        required_boolean(&inputs, "tapered_corners")?,
                        &context,
                        self.kind,
                    )?;
                    publish_mask_outcome(&context, output, self.kind)
                }
                MaskKind::ImageColorToMask => {
                    let image = resolve_named_image(&context, &inputs, "image", self.kind)?;
                    let color = required_unsigned(&inputs, "color", 0, 0xFF_FFFF)?;
                    let output = image_color_to_mask(&image, color, &context, self.kind)?;
                    publish_mask_outcome(&context, output, self.kind)
                }
                MaskKind::ImageToMask => {
                    let image = resolve_named_image(&context, &inputs, "image", self.kind)?;
                    let channel = required_combo(&inputs, "channel", IMAGE_CHANNELS)?;
                    let output = image_to_mask(&image, channel, &context, self.kind)?;
                    publish_mask_outcome(&context, output, self.kind)
                }
                MaskKind::Invert => {
                    let mut mask = resolve_named_mask(&context, &inputs, "mask", self.kind)?;
                    for (index, value) in mask.values.iter_mut().enumerate() {
                        periodic_cancellation(&context, self.kind, index)?;
                        *value = 1.0 - *value;
                    }
                    publish_mask_outcome(&context, mask, self.kind)
                }
                MaskKind::Composite => {
                    let destination =
                        resolve_named_mask(&context, &inputs, "destination", self.kind)?;
                    let source = resolve_named_mask(&context, &inputs, "source", self.kind)?;
                    let output = composite_masks(
                        destination,
                        &source,
                        required_unsigned(&inputs, "x", 0, MAX_RESOLUTION as u64)?,
                        required_unsigned(&inputs, "y", 0, MAX_RESOLUTION as u64)?,
                        required_combo(&inputs, "operation", COMPOSITE_OPERATIONS)?,
                        &context,
                        self.kind,
                    )?;
                    publish_mask_outcome(&context, output, self.kind)
                }
                MaskKind::Preview => {
                    let mask = resolve_named_mask(&context, &inputs, "mask", self.kind)?;
                    let image = mask_to_image(&mask, compute.backend(), &execution, self.kind)?;
                    let preview = context
                        .prepare_image_preview(&image, "ComfyUI")
                        .map_err(|error| preview_failure(&context, error))?;
                    let (effects, ui) = preview.into_parts();
                    let outcome = NativeNodeOutcome::Values {
                        outputs: Vec::new(),
                        ui: Some(ui),
                        effects,
                    };
                    outcome
                        .validate()
                        .map_err(|error| invalid_inputs(error.to_string()))?;
                    Ok(outcome)
                }
                MaskKind::ToImage => {
                    let mask = resolve_named_mask(&context, &inputs, "mask", self.kind)?;
                    let image = mask_to_image(&mask, compute.backend(), &execution, self.kind)?;
                    publish_image_outcome(&context, image, self.kind)
                }
            }?;
            Ok(result)
        })
    }
}

#[derive(Clone, Debug)]
struct MaskData {
    batch: u64,
    height: u64,
    width: u64,
    values: Vec<f32>,
}

fn validate_inputs(
    kind: MaskKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    if kind == MaskKind::Batch {
        dynamic_mask_handles(inputs)?;
        return Ok(());
    }
    let optional = if kind == MaskKind::Preview {
        &["prompt", "extra_pnginfo"][..]
    } else {
        &[][..]
    };
    if inputs.len() < kind.input_names().len()
        || inputs.len() > kind.input_names().len() + optional.len()
    {
        return Err(invalid_inputs(format!(
            "{} requires exactly its declared inputs",
            kind.class_type()
        )));
    }
    for name in inputs.keys() {
        if !kind.input_names().contains(&name.as_str()) && !optional.contains(&name.as_str()) {
            return Err(invalid_inputs(format!(
                "{} received unknown input {name}",
                kind.class_type()
            )));
        }
    }
    for name in kind.input_names() {
        validate_named_input(inputs, name)?;
    }
    for (name, type_name) in [("prompt", "PROMPT"), ("extra_pnginfo", "EXTRA_PNGINFO")] {
        if let Some(value) = inputs.get(name) {
            let NativeValue::PreservedUnknown {
                type_name: actual, ..
            } = value
            else {
                return Err(invalid_inputs(format!("{name} must be {type_name}")));
            };
            if actual != type_name {
                return Err(invalid_inputs(format!("{name} must be {type_name}")));
            }
        }
    }
    Ok(())
}

fn validate_named_input(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<(), NativeNodeFailure> {
    match name {
        "mask" | "destination" | "source" => {
            exact_handle(inputs.get(name), name, NativeHandleKind::Mask, "MASK")?;
        }
        "image" => {
            exact_handle(inputs.get(name), name, NativeHandleKind::Image, "IMAGE")?;
        }
        "x" | "y" | "left" | "top" | "right" | "bottom" => {
            required_unsigned(inputs, name, 0, MAX_RESOLUTION as u64)?;
        }
        "width" | "height" => {
            required_unsigned(inputs, name, 1, MAX_RESOLUTION as u64)?;
        }
        "expand" => {
            required_signed(inputs, name, -MAX_RESOLUTION, MAX_RESOLUTION)?;
        }
        "color" => {
            required_unsigned(inputs, name, 0, 0xFF_FFFF)?;
        }
        "tapered_corners" => {
            required_boolean(inputs, name)?;
        }
        "channel" => {
            required_combo(inputs, name, IMAGE_CHANNELS)?;
        }
        "operation" => {
            required_combo(inputs, name, COMPOSITE_OPERATIONS)?;
        }
        _ => return Err(invalid_inputs(format!("unsupported input {name}"))),
    }
    Ok(())
}

fn dynamic_mask_handles(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<Vec<NativeOpaqueHandle>, NativeNodeFailure> {
    if inputs.is_empty() || inputs.len() > MAX_AUTOGROW_INPUTS {
        return Err(invalid_inputs(
            "BatchMasksNode requires between one and fifty mask inputs",
        ));
    }
    let mut indexed = inputs
        .iter()
        .map(|(name, value)| {
            let index = name
                .strip_prefix("mask")
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|index| (1..=MAX_AUTOGROW_INPUTS).contains(index))
                .ok_or_else(|| invalid_inputs(format!("invalid autogrow input {name}")))?;
            let handle = exact_handle(Some(value), name, NativeHandleKind::Mask, "MASK")?;
            Ok((index, handle.clone()))
        })
        .collect::<Result<Vec<_>, NativeNodeFailure>>()?;
    indexed.sort_by_key(|(index, _)| *index);
    if indexed.windows(2).any(|window| window[0].0 == window[1].0) {
        return Err(invalid_inputs("duplicate BatchMasksNode input index"));
    }
    Ok(indexed.into_iter().map(|(_, handle)| handle).collect())
}

fn exact_handle<'a>(
    value: Option<&'a NativeValue>,
    name: &str,
    kind: NativeHandleKind,
    type_name: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(invalid_inputs(format!(
            "{name} must be an exact {type_name} handle"
        )));
    };
    if value.handle_type().kind != kind || value.handle_type().type_id != type_name {
        return Err(invalid_inputs(format!(
            "{name} must be an exact {type_name} handle"
        )));
    }
    Ok(value)
}

fn primitive_integer(value: Option<&NativeValue>, name: &str) -> Result<i64, NativeNodeFailure> {
    match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => i64::try_from(*value).map_err(|_| invalid_inputs(format!("{name} is too large"))),
        _ => Err(invalid_inputs(format!("{name} must be an INT"))),
    }
}

fn required_unsigned(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = primitive_integer(inputs.get(name), name)?;
    let value = u64::try_from(value)
        .map_err(|_| invalid_inputs(format!("{name} must be non-negative")))?;
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn required_signed(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: i64,
    maximum: i64,
) -> Result<i64, NativeNodeFailure> {
    let value = primitive_integer(inputs.get(name), name)?;
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
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

fn resolve_named_mask(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let handle = exact_handle(inputs.get(name), name, NativeHandleKind::Mask, "MASK")?;
    resolve_mask(context, handle, kind)
}

fn resolve_mask(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let expected = mask_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected, &context.cancellation)
        .map_err(|error| handle_failure(error, kind, "MASK"))?;
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(native_failure(kind, "MASK handle did not resolve to a tensor"));
    };
    if payload.role() != NativeTensorRole::Mask {
        return Err(native_failure(kind, "MASK tensor role changed"));
    }
    let [batch, height, width] = payload.tensor().descriptor().shape() else {
        return Err(native_failure(kind, "MASK tensor must have rank three"));
    };
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let values = tensor_to_f32(compute.backend(), payload.tensor(), &execution)
        .map_err(|error| native_failure(kind, error))?;
    Ok(MaskData {
        batch: *batch,
        height: *height,
        width: *width,
        values: values.to_vec(),
    })
}

fn resolve_named_image(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    kind: MaskKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let handle = exact_handle(inputs.get(name), name, NativeHandleKind::Image, "IMAGE")?;
    let expected = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected, &context.cancellation)
        .map_err(|error| handle_failure(error, kind, "IMAGE"))?;
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

fn batch_masks(
    masks: &[MaskData],
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let first = masks
        .first()
        .ok_or_else(|| invalid_inputs("BatchMasksNode requires at least one MASK"))?;
    let total_batch = masks.iter().try_fold(0_u64, |batch, mask| {
        batch.checked_add(mask.batch)
    })
    .ok_or_else(|| invalid_inputs("batched MASK count overflowed"))?;
    let output_count = checked_usize(total_batch, first.height, first.width)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_count)
        .map_err(|error| native_failure(kind, error))?;
    for mask in masks {
        check_cancellation(context, kind)?;
        if mask.height == first.height && mask.width == first.width {
            values.extend_from_slice(&mask.values);
            continue;
        }
        if first.height == 0 || first.width == 0 || mask.height == 0 || mask.width == 0 {
            return Err(native_failure(kind, "cannot resize a MASK with an empty spatial axis"));
        }
        let image = ImageTensor::from_f32(
            backend,
            execution,
            mask.batch,
            mask.height,
            mask.width,
            1,
            &mask.values,
        )
        .map_err(|error| native_failure(kind, error))?;
        let resized = image
            .resize(
                first.width,
                first.height,
                ResizeMode::Bilinear,
                ResizeCrop::Center,
                backend,
                execution,
            )
            .map_err(|error| native_failure(kind, error))?;
        values.extend_from_slice(
            resized
                .as_f32_slice()
                .map_err(|error| native_failure(kind, error))?,
        );
    }
    Ok(MaskData {
        batch: total_batch,
        height: first.height,
        width: first.width,
        values,
    })
}

fn crop_mask(
    mask: &MaskData,
    x: u64,
    y: u64,
    width: u64,
    height: u64,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let output_width = width.min(mask.width.saturating_sub(x));
    let output_height = height.min(mask.height.saturating_sub(y));
    let mut values = Vec::new();
    for batch in 0..mask.batch {
        for output_y in 0..output_height {
            for output_x in 0..output_width {
                let index = mask_index(mask, batch, y + output_y, x + output_x)?;
                periodic_cancellation(context, kind, values.len())?;
                values.push(mask.values[index]);
            }
        }
    }
    Ok(MaskData {
        batch: mask.batch,
        height: output_height,
        width: output_width,
        values,
    })
}

fn feather_mask(
    mut mask: MaskData,
    [left, top, right, bottom]: [u64; 4],
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let left = left.min(mask.width);
    let right = right.min(mask.width);
    let top = top.min(mask.height);
    let bottom = bottom.min(mask.height);
    let width = mask.width;
    let height = mask.height;
    for x in 0..left {
        multiply_mask_column(&mut mask, x, (x + 1) as f32 / left as f32, context, kind)?;
    }
    for x in 0..right {
        multiply_mask_column(
            &mut mask,
            width - x - 1,
            (x + 1) as f32 / right as f32,
            context,
            kind,
        )?;
    }
    for y in 0..top {
        multiply_mask_row(&mut mask, y, (y + 1) as f32 / top as f32, context, kind)?;
    }
    for y in 0..bottom {
        multiply_mask_row(
            &mut mask,
            height - y - 1,
            (y + 1) as f32 / bottom as f32,
            context,
            kind,
        )?;
    }
    Ok(mask)
}

fn multiply_mask_column(
    mask: &mut MaskData,
    x: u64,
    factor: f32,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<(), NativeNodeFailure> {
    for batch in 0..mask.batch {
        for y in 0..mask.height {
            let index = mask_index(mask, batch, y, x)?;
            periodic_cancellation(context, kind, index)?;
            mask.values[index] *= factor;
        }
    }
    Ok(())
}

fn multiply_mask_row(
    mask: &mut MaskData,
    y: u64,
    factor: f32,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<(), NativeNodeFailure> {
    for batch in 0..mask.batch {
        for x in 0..mask.width {
            let index = mask_index(mask, batch, y, x)?;
            periodic_cancellation(context, kind, index)?;
            mask.values[index] *= factor;
        }
    }
    Ok(())
}

fn grow_mask(
    mut mask: MaskData,
    expand: i64,
    tapered_corners: bool,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let iterations = expand.unsigned_abs();
    let neighbors = if tapered_corners { 5 } else { 9 };
    let visits = u64::try_from(mask.values.len())
        .ok()
        .and_then(|count| count.checked_mul(iterations))
        .and_then(|count| count.checked_mul(neighbors))
        .ok_or_else(|| native_failure(kind, "GrowMask operation count overflowed"))?;
    if visits > MAX_MORPHOLOGY_VISITS {
        return Err(native_failure(kind, "GrowMask exceeds the bounded native operation limit"));
    }
    let mut output = vec![0.0_f32; mask.values.len()];
    for iteration in 0..iterations {
        check_cancellation(context, kind)?;
        for batch in 0..mask.batch {
            for y in 0..mask.height {
                for x in 0..mask.width {
                    let index = mask_index(&mask, batch, y, x)?;
                    let mut selected = mask.values[index];
                    for y_offset in -1_i64..=1 {
                        for x_offset in -1_i64..=1 {
                            if tapered_corners && x_offset != 0 && y_offset != 0 {
                                continue;
                            }
                            let neighbor_y = reflected_neighbor(y, y_offset, mask.height);
                            let neighbor_x = reflected_neighbor(x, x_offset, mask.width);
                            let neighbor = mask.values[mask_index(
                                &mask,
                                batch,
                                neighbor_y,
                                neighbor_x,
                            )?];
                            if (expand < 0 && neighbor < selected)
                                || (expand > 0 && neighbor > selected)
                            {
                                selected = neighbor;
                            }
                        }
                    }
                    periodic_cancellation(context, kind, index)?;
                    output[index] = selected;
                }
            }
        }
        std::mem::swap(&mut mask.values, &mut output);
        if iteration + 1 < iterations {
            output.fill(0.0);
        }
    }
    Ok(mask)
}

fn reflected_neighbor(position: u64, offset: i64, extent: u64) -> u64 {
    if extent == 0 {
        return 0;
    }
    match offset {
        -1 => position.saturating_sub(1),
        1 => position.saturating_add(1).min(extent - 1),
        _ => position,
    }
}

fn image_color_to_mask(
    image: &ImageTensor,
    color: u64,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    if channels < 3 {
        return Err(native_failure(kind, "ImageColorToMask requires at least three channels"));
    }
    let pixels = image
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let pixel_count = checked_usize(batch, height, width)?;
    let channels = usize::try_from(channels).map_err(|_| native_failure(kind, "channel overflow"))?;
    let mut values = Vec::with_capacity(pixel_count);
    for pixel in 0..pixel_count {
        periodic_cancellation(context, kind, pixel)?;
        let start = pixel
            .checked_mul(channels)
            .ok_or_else(|| native_failure(kind, "pixel offset overflowed"))?;
        let rgb = pixels
            .get(start..start + 3)
            .ok_or_else(|| native_failure(kind, "IMAGE storage ended before RGB data"))?;
        let key = rgb.iter().try_fold(0_u64, |key, component| {
            let component = if component.is_finite() {
                (component.clamp(0.0, 1.0) * 255.0).round_ties_even() as u64
            } else {
                256
            };
            key.checked_mul(256)?.checked_add(component)
        });
        values.push(if key == Some(color) { 1.0 } else { 0.0 });
    }
    Ok(MaskData {
        batch,
        height,
        width,
        values,
    })
}

fn image_to_mask(
    image: &ImageTensor,
    channel: &str,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    let channel_index = IMAGE_CHANNELS
        .iter()
        .position(|candidate| *candidate == channel)
        .ok_or_else(|| invalid_inputs(format!("unsupported image channel {channel}")))?;
    let (batch, height, width, channels) = image
        .dimensions()
        .map_err(|error| native_failure(kind, error))?;
    let channels = usize::try_from(channels).map_err(|_| native_failure(kind, "channel overflow"))?;
    if channel_index >= channels {
        return Err(native_failure(kind, format!("IMAGE has no {channel} channel")));
    }
    let pixels = image
        .as_f32_slice()
        .map_err(|error| native_failure(kind, error))?;
    let pixel_count = checked_usize(batch, height, width)?;
    let mut values = Vec::with_capacity(pixel_count);
    for pixel in 0..pixel_count {
        periodic_cancellation(context, kind, pixel)?;
        let index = pixel
            .checked_mul(channels)
            .and_then(|offset| offset.checked_add(channel_index))
            .ok_or_else(|| native_failure(kind, "channel offset overflowed"))?;
        values.push(
            *pixels
                .get(index)
                .ok_or_else(|| native_failure(kind, "IMAGE storage ended before channel data"))?,
        );
    }
    Ok(MaskData {
        batch,
        height,
        width,
        values,
    })
}

fn composite_masks(
    mut destination: MaskData,
    source: &MaskData,
    x: u64,
    y: u64,
    operation: &str,
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<MaskData, NativeNodeFailure> {
    if source.batch != destination.batch && source.batch != 1 {
        return Err(native_failure(kind, "source MASK batch cannot broadcast to destination"));
    }
    if x > destination.width || y > destination.height {
        return Err(native_failure(kind, "mask composite offset exceeds destination bounds"));
    }
    let visible_width = source.width.min(destination.width - x);
    let visible_height = source.height.min(destination.height - y);
    for batch in 0..destination.batch {
        let source_batch = if source.batch == 1 { 0 } else { batch };
        for source_y in 0..visible_height {
            for source_x in 0..visible_width {
                let destination_index =
                    mask_index(&destination, batch, y + source_y, x + source_x)?;
                let source_index = mask_index(source, source_batch, source_y, source_x)?;
                periodic_cancellation(context, kind, destination_index)?;
                let destination_value = destination.values[destination_index];
                let source_value = source.values[source_index];
                destination.values[destination_index] = match operation {
                    "multiply" => destination_value * source_value,
                    "add" => destination_value + source_value,
                    "subtract" => destination_value - source_value,
                    "and" => (bool_float(destination_value) && bool_float(source_value)).into_float(),
                    "or" => (bool_float(destination_value) || bool_float(source_value)).into_float(),
                    "xor" => (bool_float(destination_value) ^ bool_float(source_value)).into_float(),
                    _ => return Err(invalid_inputs(format!("unsupported operation {operation}"))),
                };
            }
        }
    }
    for (index, value) in destination.values.iter_mut().enumerate() {
        periodic_cancellation(context, kind, index)?;
        *value = value.clamp(0.0, 1.0);
    }
    Ok(destination)
}

trait IntoFloat {
    fn into_float(self) -> f32;
}

impl IntoFloat for bool {
    fn into_float(self) -> f32 {
        if self { 1.0 } else { 0.0 }
    }
}

fn bool_float(value: f32) -> bool {
    value.round_ties_even() != 0.0
}

fn mask_to_image(
    mask: &MaskData,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    kind: MaskKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let pixel_count = checked_usize(mask.batch, mask.height, mask.width)?;
    let output_count = pixel_count
        .checked_mul(3)
        .ok_or_else(|| invalid_inputs("MaskToImage output size overflowed"))?;
    let mut values = Vec::with_capacity(output_count);
    for value in &mask.values {
        values.extend_from_slice(&[*value; 3]);
    }
    ImageTensor::from_f32(
        backend,
        execution,
        mask.batch,
        mask.height,
        mask.width,
        3,
        &values,
    )
    .map_err(|error| native_failure(kind, error))
}

fn checked_usize(batch: u64, height: u64, width: u64) -> Result<usize, NativeNodeFailure> {
    batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid_inputs("tensor element count overflowed"))
}

fn mask_index(
    mask: &MaskData,
    batch: u64,
    y: u64,
    x: u64,
) -> Result<usize, NativeNodeFailure> {
    batch
        .checked_mul(mask.height)
        .and_then(|value| value.checked_add(y))
        .and_then(|value| value.checked_mul(mask.width))
        .and_then(|value| value.checked_add(x))
        .and_then(|value| usize::try_from(value).ok())
        .filter(|index| *index < mask.values.len())
        .ok_or_else(|| native_failure(MaskKind::Composite, "MASK index exceeded storage"))
}

fn publish_mask_outcome(
    context: &NativeNodeContext,
    mask: MaskData,
    kind: MaskKind,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute.execution_context(context).map_err(compute_failure)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![mask.batch, mask.height, mask.width],
        DType::F32,
        DeviceId::CPU,
        execution.stream,
    )
    .map_err(|error| native_failure(kind, error))?;
    let tensor = compute
        .backend()
        .upload_f32(descriptor, &mask.values, &execution)
        .map(|(tensor, _)| tensor)
        .map_err(|error| native_failure(kind, error))?;
    let payload = NativeTensorPayload::from_tensor(NativeTensorRole::Mask, tensor)
        .map_err(|error| native_failure(kind, error))?;
    publish_tensor_outcome(context, payload, kind)
}

fn publish_image_outcome(
    context: &NativeNodeContext,
    image: ImageTensor,
    kind: MaskKind,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
        .map_err(|error| native_failure(kind, error))?;
    publish_tensor_outcome(context, payload, kind)
}

fn publish_tensor_outcome(
    context: &NativeNodeContext,
    payload: NativeTensorPayload,
    kind: MaskKind,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check_cancellation(context, kind)?;
    let handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(|error| handle_failure(error, kind, "output"))?;
    let outcome = NativeNodeOutcome::Values {
        outputs: vec![NativeValue::Handle { value: handle.clone() }],
        ui: None,
        effects: Vec::new(),
    };
    if let Err(error) = outcome.validate() {
        let cleanup = sim_types::CancellationToken::default();
        context
            .handle_store()
            .revoke(&handle, &cleanup)
            .map_err(|revoke_error| native_failure(kind, revoke_error))?;
        return Err(invalid_inputs(error.to_string()));
    }
    Ok(outcome)
}

fn periodic_cancellation(
    context: &NativeNodeContext,
    kind: MaskKind,
    index: usize,
) -> Result<(), NativeNodeFailure> {
    if index.is_multiple_of(4_096) {
        check_cancellation(context, kind)?;
    }
    Ok(())
}

fn check_cancellation(
    context: &NativeNodeContext,
    kind: MaskKind,
) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure(kind))
}

fn handle_failure(
    error: NativeHandleStoreError,
    kind: MaskKind,
    boundary: &str,
) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(kind)
    } else {
        NativeNodeFailure {
            code: "invalid_native_handle".to_owned(),
            message: format!("{} {boundary} handle is unavailable: {error}", kind.class_type()),
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

fn preview_failure(
    context: &NativeNodeContext,
    error: crate::NativeImagePreviewError,
) -> NativeNodeFailure {
    if context.cancellation.is_cancelled()
        || matches!(
            error,
            crate::NativeImagePreviewError::Effect(crate::NativeEffectServiceError::Cancelled)
        )
    {
        interrupted_failure(MaskKind::Preview)
    } else {
        NativeNodeFailure {
            code: "native_mask_preview_failed".to_owned(),
            message: error.to_string(),
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

fn native_failure(kind: MaskKind, error: impl std::fmt::Display) -> NativeNodeFailure {
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("cancelled") {
        return interrupted_failure(kind);
    }
    NativeNodeFailure {
        code: "native_image_mask_failed".to_owned(),
        message: format!("{} failed: {message}", kind.class_type()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure(kind: MaskKind) -> NativeNodeFailure {
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
        NativeEffectServiceError, NativeHandleStore, NativeHandleStoreIdentity,
        NativeNodeComputeSession, NativeNodeServiceIdentity, NativeNodeServices,
        NativeOutputEffectRequest, NativePreparedEffectKind, NativePreparedEffectRequest,
        NativePreparedEffectService, NativeResolvedPayload, NativeResolvedPayloadRetention,
    };
    use sim_tensor::{CpuBackend, CpuWorkspaceAuthority, StreamId};
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::Value;
    use std::{
        fmt,
        sync::{
            Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-mask-comfy-node-0019/fixture.json"
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
            let identifier = format!("image-mask-{generation}");
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

    struct TestEffects {
        identity: NativeNodeServiceIdentity,
        fail_on_ordinal: Option<u64>,
        next_ordinal: AtomicU64,
        prepared: Mutex<Vec<NativePreparedEffectRequest>>,
        rolled_back: Mutex<Vec<Uuid>>,
    }

    impl fmt::Debug for TestEffects {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("TestEffects")
                .field("identity", &self.identity)
                .finish_non_exhaustive()
        }
    }

    impl NativePreparedEffectService for TestEffects {
        fn identity(&self) -> &NativeNodeServiceIdentity {
            &self.identity
        }

        fn maximum_output_bytes(&self) -> u64 {
            1024 * 1024
        }

        fn prepare_output(
            &self,
            request: NativeOutputEffectRequest,
            cancellation: &CancellationToken,
        ) -> Result<NativePreparedEffectRequest, NativeEffectServiceError> {
            cancellation
                .check()
                .map_err(|_| NativeEffectServiceError::Cancelled)?;
            let ordinal = self.next_ordinal.fetch_add(1, Ordering::AcqRel);
            if self.fail_on_ordinal == Some(ordinal) {
                return Err(NativeEffectServiceError::Rejected);
            }
            let ticket = NativePreparedEffectRequest::checked(
                self.identity.service_id(),
                Uuid::from_u128(0x504_100 + u128::from(ordinal)),
                NativePreparedEffectKind::Output,
                request.request_digest_sha256(),
            )
            .map_err(|_| NativeEffectServiceError::Rejected)?;
            self.prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .push(ticket.clone());
            Ok(ticket)
        }

        fn rollback_prepared(
            &self,
            request: &NativePreparedEffectRequest,
        ) -> Result<(), NativeEffectServiceError> {
            let mut prepared = self
                .prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?;
            let index = prepared
                .iter()
                .position(|ticket| ticket == request)
                .ok_or(NativeEffectServiceError::InvalidTicket)?;
            let ticket = prepared.remove(index);
            self.rolled_back
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .push(ticket.transaction_id());
            Ok(())
        }

        fn rollback_all_prepared(&self) -> Result<(), NativeEffectServiceError> {
            let prepared = std::mem::take(
                &mut *self
                    .prepared
                    .lock()
                    .map_err(|_| NativeEffectServiceError::Rejected)?,
            );
            self.rolled_back
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .extend(prepared.into_iter().map(|ticket| ticket.transaction_id()));
            Ok(())
        }
    }

    struct Harness {
        backend: Arc<CpuBackend>,
        store: Arc<TestStore>,
        effects: Option<Arc<TestEffects>>,
        context: NativeNodeContext,
    }

    impl Harness {
        fn new(
            seed: u128,
            cancellation: CancellationToken,
            fail_on_effect: Option<u64>,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(seed + 2));
            let node_id = NodeId(format!("image-mask-{seed}"));
            let store = TestStore::new(seed + 3, attempt_id)?;
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
            let backend = Arc::new(backend);
            let scratch = authority.authorize_workspace(32 * 1024 * 1024)?;
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(seed + 5),
                attempt_id,
                node_id.clone(),
            )?;
            let compute = NativeNodeComputeSession::checked(
                identity.clone(),
                backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            let effects = fail_on_effect.map(|fail_on_ordinal| {
                Arc::new(TestEffects {
                    identity: identity.clone(),
                    fail_on_ordinal: (fail_on_ordinal != u64::MAX).then_some(fail_on_ordinal),
                    next_ordinal: AtomicU64::new(0),
                    prepared: Mutex::new(Vec::new()),
                    rolled_back: Mutex::new(Vec::new()),
                })
            });
            let effects_service = effects
                .as_ref()
                .map(|effects| effects.clone() as Arc<dyn NativePreparedEffectService>);
            let context = NativeNodeContext::new_with_services(
                PromptId(Uuid::from_u128(seed + 6)),
                attempt_id,
                node_id,
                cancellation,
                scratch,
                store.clone(),
                NativeNodeServices::checked(None, effects_service, Some(compute))?,
            )?;
            Ok(Self {
                backend,
                store,
                effects,
                context,
            })
        }

        fn with_effects(seed: u128) -> Result<Self, Box<dyn std::error::Error>> {
            Self::new(seed, CancellationToken::default(), Some(u64::MAX))
        }

        fn publish_mask(
            &self,
            shape: [u64; 3],
            values: &[f32],
        ) -> Result<NativeOpaqueHandle, Box<dyn std::error::Error>> {
            let execution = self.context.compute_session()?.execution_context(&self.context)?;
            let descriptor = TensorDescriptor::contiguous(
                shape.to_vec(),
                DType::F32,
                DeviceId::CPU,
                execution.stream,
            )?;
            let tensor = self.backend.upload_f32(descriptor, values, &execution)?.0;
            let payload = NativeTensorPayload::from_tensor(NativeTensorRole::Mask, tensor)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(payload)),
                &CancellationToken::default(),
            )?)
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

        fn output_tensor(
            &self,
            outcome: NativeNodeOutcome,
            expected_type: NativeHandleType,
        ) -> Result<(Vec<u64>, Vec<f32>), Box<dyn std::error::Error>> {
            let NativeNodeOutcome::Values { outputs, .. } = outcome else {
                return Err("node did not return values".into());
            };
            let Some(NativeValue::Handle { value }) = outputs.first() else {
                return Err("node did not return a tensor handle".into());
            };
            let resolved = self.store.resolve(
                value,
                &expected_type,
                &CancellationToken::default(),
            )?;
            let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
                return Err("output handle did not contain a tensor".into());
            };
            let execution = self.context.compute_session()?.execution_context(&self.context)?;
            let values = tensor_to_f32(&self.backend, payload.tensor(), &execution)?.to_vec();
            Ok((payload.tensor().descriptor().shape().to_vec(), values))
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

    fn handle_input(name: &str, handle: NativeOpaqueHandle) -> (String, NativeValue) {
        (name.to_owned(), NativeValue::Handle { value: handle })
    }

    fn integer_input(name: &str, value: u64) -> (String, NativeValue) {
        (
            name.to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(value),
            },
        )
    }

    fn signed_input(name: &str, value: i64) -> (String, NativeValue) {
        (
            name.to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Integer(value),
            },
        )
    }

    fn boolean_input(name: &str, value: bool) -> (String, NativeValue) {
        (
            name.to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(value),
            },
        )
    }

    fn combo_input(name: &str, value: &str) -> (String, NativeValue) {
        (
            name.to_owned(),
            NativeValue::PreservedUnknown {
                type_name: "COMBO".to_owned(),
                value: Value::String(value.to_owned()),
            },
        )
    }

    #[test]
    fn descriptors_fixture_and_persistence_preserve_all_source_contracts()
    -> Result<(), Box<dyn std::error::Error>> {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 10);
        for binding in &bindings {
            let NativeNodeBinding::Executable { descriptor, .. } = binding else {
                return Err("binding was not executable".into());
            };
            descriptor.validate_exact_schema_v2()?;
        }
        let batch = bindings
            .iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable { descriptor, .. }
                    if descriptor.class_type == "BatchMasksNode" =>
                {
                    Some(descriptor)
                }
                _ => None,
            })
            .ok_or("BatchMasksNode descriptor is absent")?;
        assert_eq!(batch.dynamic_inputs.len(), 1);
        assert_eq!(batch.dynamic_inputs[0].name_template, "mask{index}");
        assert_eq!(batch.dynamic_inputs[0].start_index, 1);
        assert_eq!(batch.dynamic_inputs[0].minimum_count, 1);
        assert_eq!(batch.dynamic_inputs[0].maximum_count, 50);

        let preview = bindings
            .iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable { descriptor, .. }
                    if descriptor.class_type == "MaskPreview" =>
                {
                    Some(descriptor)
                }
                _ => None,
            })
            .ok_or("MaskPreview descriptor is absent")?;
        assert!(preview.output_node);
        assert_eq!(preview.effect, NativeEffectClass::WritesArtifact);
        assert_eq!(preview.cache, NativeCachePolicy::Never);
        assert!(preview.outputs.is_empty());
        assert_eq!(preview.inputs.iter().filter(|input| input.hidden).count(), 2);

        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture["stable_task_id"],
            "comfy-parity-native-nodes-image-mask-comfy-node-0019"
        );
        assert_eq!(fixture["nodes"].as_array().map(Vec::len), Some(10));
        let encoded = serde_json::to_vec(&fixture)?;
        assert_eq!(serde_json::from_slice::<Value>(&encoded)?, fixture);

        let persisted_inputs = BTreeMap::from([
            integer_input("x", 7),
            integer_input("y", 9),
            combo_input("operation", "xor"),
        ]);
        let encoded = serde_json::to_vec(&persisted_inputs)?;
        assert_eq!(
            serde_json::from_slice::<BTreeMap<String, NativeValue>>(&encoded)?,
            persisted_inputs
        );
        Ok(())
    }

    #[test]
    fn mask_arithmetic_crop_feather_and_grow_match_source_boundaries()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new(0x504_200, CancellationToken::default(), None)?;
        let mask = MaskData {
            batch: 1,
            height: 3,
            width: 4,
            values: (0..12).map(|value| value as f32).collect(),
        };
        let cropped = crop_mask(&mask, 2, 1, 8, 8, &harness.context, MaskKind::Crop)?;
        assert_eq!((cropped.batch, cropped.height, cropped.width), (1, 2, 2));
        assert_eq!(cropped.values, vec![6.0, 7.0, 10.0, 11.0]);
        let empty = crop_mask(&mask, 9, 9, 2, 2, &harness.context, MaskKind::Crop)?;
        assert_eq!((empty.height, empty.width), (0, 0));
        assert!(empty.values.is_empty());

        let feathered = feather_mask(
            MaskData {
                batch: 1,
                height: 3,
                width: 3,
                values: vec![1.0; 9],
            },
            [2, 2, 0, 0],
            &harness.context,
            MaskKind::Feather,
        )?;
        assert_eq!(
            feathered.values,
            vec![0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0]
        );

        let center = MaskData {
            batch: 1,
            height: 3,
            width: 3,
            values: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        };
        let tapered = grow_mask(
            center.clone(),
            1,
            true,
            &harness.context,
            MaskKind::Grow,
        )?;
        assert_eq!(
            tapered.values,
            vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        );
        let square = grow_mask(center, 1, false, &harness.context, MaskKind::Grow)?;
        assert_eq!(square.values, vec![1.0; 9]);
        let eroded = grow_mask(
            MaskData {
                batch: 1,
                height: 3,
                width: 3,
                values: vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            },
            -1,
            false,
            &harness.context,
            MaskKind::Grow,
        )?;
        assert_eq!(eroded.values, vec![0.0; 9]);

        let destination = MaskData {
            batch: 1,
            height: 1,
            width: 4,
            values: vec![-1.0, 0.5, 0.75, 2.0],
        };
        let source = MaskData {
            batch: 1,
            height: 1,
            width: 3,
            values: vec![0.5, 0.5, 0.5],
        };
        let added = composite_masks(
            destination.clone(),
            &source,
            1,
            0,
            "add",
            &harness.context,
            MaskKind::Composite,
        )?;
        assert_eq!(added.values, vec![0.0, 1.0, 1.0, 1.0]);
        let xor = composite_masks(
            destination,
            &source,
            1,
            0,
            "xor",
            &harness.context,
            MaskKind::Composite,
        )?;
        assert_eq!(xor.values, vec![0.0, 0.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn executable_conversions_batching_and_inversion_are_typed_and_exact()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::new(0x504_300, CancellationToken::default(), None)?;
        let image = harness.publish_image(
            [1, 1, 2, 4],
            &[0.5 / 255.0, 1.5 / 255.0, 2.5 / 255.0, 0.25, 1.0, 0.0, 0.5, 0.75],
        )?;
        let color = futures::executor::block_on(node("ImageColorToMask")?.execute(
            harness.context.clone(),
            BTreeMap::from([
                handle_input("image", image.clone()),
                integer_input("color", 0x0002_02),
            ]),
        ))?;
        assert_eq!(harness.output_tensor(color, mask_type()?)?.1, vec![1.0, 0.0]);

        let alpha = futures::executor::block_on(node("ImageToMask")?.execute(
            harness.context.clone(),
            BTreeMap::from([
                handle_input("image", image),
                combo_input("channel", "alpha"),
            ]),
        ))?;
        assert_eq!(
            harness.output_tensor(alpha, mask_type()?)?,
            (vec![1, 1, 2], vec![0.25, 0.75])
        );

        let mask = harness.publish_mask([1, 1, 2], &[-0.25, 1.5])?;
        let inverted = futures::executor::block_on(node("InvertMask")?.execute(
            harness.context.clone(),
            BTreeMap::from([handle_input("mask", mask.clone())]),
        ))?;
        assert_eq!(
            harness.output_tensor(inverted, mask_type()?)?.1,
            vec![1.25, -0.5]
        );
        let image = futures::executor::block_on(node("MaskToImage")?.execute(
            harness.context.clone(),
            BTreeMap::from([handle_input("mask", mask)]),
        ))?;
        assert_eq!(
            harness.output_tensor(image, image_type()?)?,
            (vec![1, 1, 2, 3], vec![-0.25, -0.25, -0.25, 1.5, 1.5, 1.5])
        );

        let first = harness.publish_mask([1, 1, 2], &[1.0, 2.0])?;
        let second = harness.publish_mask([2, 1, 2], &[3.0, 4.0, 5.0, 6.0])?;
        let batched = futures::executor::block_on(node("BatchMasksNode")?.execute(
            harness.context.clone(),
            BTreeMap::from([
                handle_input("mask2", second),
                handle_input("mask1", first),
            ]),
        ))?;
        assert_eq!(
            harness.output_tensor(batched, mask_type()?)?,
            (vec![3, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );

        let rgb = harness.publish_image([1, 1, 1, 3], &[0.0, 0.0, 0.0])?;
        let failure = futures::executor::block_on(node("ImageToMask")?.execute(
            harness.context,
            BTreeMap::from([
                handle_input("image", rgb),
                combo_input("channel", "alpha"),
            ]),
        ))
        .expect_err("a missing alpha channel must fail");
        assert_eq!(failure.code, "native_image_mask_failed");
        Ok(())
    }

    #[test]
    fn preview_prepares_effects_and_rolls_back_partial_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let harness = Harness::with_effects(0x504_400)?;
        let mask = harness.publish_mask([1, 1, 2], &[0.0, 1.0])?;
        let outcome = futures::executor::block_on(node("MaskPreview")?.execute(
            harness.context.clone(),
            BTreeMap::from([handle_input("mask", mask)]),
        ))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("preview did not return values".into());
        };
        assert!(outputs.is_empty());
        assert_eq!(effects.len(), 1);
        assert_eq!(ui.as_ref().and_then(|value| value["images"].as_array()).map(Vec::len), Some(1));
        assert_eq!(
            harness
                .effects
                .as_ref()
                .ok_or("effect service is absent")?
                .prepared
                .lock()
                .map_err(|_| "effect prepared state is poisoned")?
                .len(),
            1
        );

        let failing = Harness::new(0x504_500, CancellationToken::default(), Some(1))?;
        let mask = failing.publish_mask([2, 1, 1], &[0.0, 1.0])?;
        let failure = futures::executor::block_on(node("MaskPreview")?.execute(
            failing.context.clone(),
            BTreeMap::from([handle_input("mask", mask)]),
        ))
        .expect_err("the second prepared effect must fail");
        assert_eq!(failure.code, "native_mask_preview_failed");
        let effects = failing.effects.as_ref().ok_or("effect service is absent")?;
        assert!(
            effects
                .prepared
                .lock()
                .map_err(|_| "effect prepared state is poisoned")?
                .is_empty()
        );
        assert_eq!(
            effects
                .rolled_back
                .lock()
                .map_err(|_| "effect rollback state is poisoned")?
                .len(),
            1
        );
        Ok(())
    }

    #[test]
    fn cancellation_validation_and_stale_handle_recovery_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let cancelled = Harness::new(0x504_600, cancellation, None)?;
        let failure = futures::executor::block_on(
            node("InvertMask")?.execute(cancelled.context, BTreeMap::new()),
        )
        .expect_err("cancelled execution must stop before input validation");
        assert_eq!(failure.kind, NativeNodeFailureKind::Interrupted);

        let old = Harness::new(0x504_700, CancellationToken::default(), None)?;
        let stale = old.publish_mask([1, 1, 1], &[0.25])?;
        let fresh = Harness::new(0x504_800, CancellationToken::default(), None)?;
        let failure = futures::executor::block_on(node("InvertMask")?.execute(
            fresh.context.clone(),
            BTreeMap::from([handle_input("mask", stale)]),
        ))
        .expect_err("a foreign attempt handle must fail");
        assert_eq!(failure.code, "invalid_native_handle");

        let mask = fresh.publish_mask([1, 1, 1], &[0.25])?;
        let first = futures::executor::block_on(node("InvertMask")?.execute(
            fresh.context.clone(),
            BTreeMap::from([handle_input("mask", mask.clone())]),
        ))?;
        let second = futures::executor::block_on(node("InvertMask")?.execute(
            fresh.context.clone(),
            BTreeMap::from([handle_input("mask", mask)]),
        ))?;
        assert_eq!(
            fresh.output_tensor(first, mask_type()?)?,
            fresh.output_tensor(second, mask_type()?)?
        );

        assert!(dynamic_mask_handles(&BTreeMap::new()).is_err());
        assert!(
            validate_inputs(
                MaskKind::Grow,
                &BTreeMap::from([
                    handle_input(
                        "mask",
                        fresh.publish_mask([1, 1, 1], &[1.0])?,
                    ),
                    signed_input("expand", MAX_RESOLUTION + 1),
                    boolean_input("tapered_corners", true),
                ]),
            )
            .is_err()
        );
        Ok(())
    }
}
