use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeEffectServiceError, NativeHandleKind, NativeHandleStoreError,
    NativeHandleType, NativeImagePreviewError, NativeInputDescriptor, NativeNode,
    NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError,
    NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality,
    NativePreparedEffectRequest, NativePrimitive, NativePrimitiveType, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_media::{NativeBoundingBox, NativeBoundingBoxPayload};
use sim_tensor::{
    ImageTensor, NativeTensorPayload, NativeTensorRole, NumpyRandomState, ResizeCrop, ResizeMode,
    RngError, TensorError,
};
use futures::future::BoxFuture;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "CenterCropImages",
    "CropByBBoxes",
    "ImageCrop",
    "ImageCropV2",
    "ImageFlip",
    "ImagePadForOutpaint",
    "ImageRotate",
    "ImageStitch",
    "RandomCropImages",
    "ResizeAndPadImage",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const MAX_RESOLUTION: u64 = 16_384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransformKind {
    CenterCrop,
    CropByBBoxes,
    Crop,
    CropV2,
    Flip,
    PadForOutpaint,
    Rotate,
    Stitch,
    RandomCrop,
    ResizeAndPad,
}

impl TransformKind {
    const fn feature_id(self) -> &'static str {
        match self {
            Self::CenterCrop => "COMFY-NODE-0047",
            Self::CropByBBoxes => "COMFY-NODE-0125",
            Self::Crop => "COMFY-NODE-0247",
            Self::CropV2 => "COMFY-NODE-0248",
            Self::Flip => "COMFY-NODE-0250",
            Self::PadForOutpaint => "COMFY-NODE-0258",
            Self::Rotate => "COMFY-NODE-0261",
            Self::Stitch => "COMFY-NODE-0267",
            Self::RandomCrop => "COMFY-NODE-0504",
            Self::ResizeAndPad => "COMFY-NODE-0540",
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::CenterCrop => "CenterCropImages",
            Self::CropByBBoxes => "CropByBBoxes",
            Self::Crop => "ImageCrop",
            Self::CropV2 => "ImageCropV2",
            Self::Flip => "ImageFlip",
            Self::PadForOutpaint => "ImagePadForOutpaint",
            Self::Rotate => "ImageRotate",
            Self::Stitch => "ImageStitch",
            Self::RandomCrop => "RandomCropImages",
            Self::ResizeAndPad => "ResizeAndPadImage",
        }
    }

    const fn implementation_version(self) -> &'static str {
        match self {
            Self::CenterCrop | Self::RandomCrop => "source-3b27465f-v1",
            Self::CropByBBoxes => "source-d9b38524-v1",
            Self::PadForOutpaint => "source-b8dfdde1-v1",
            Self::Crop
            | Self::CropV2
            | Self::Flip
            | Self::Rotate
            | Self::Stitch
            | Self::ResizeAndPad => "source-a57638bf-v1",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::CenterCrop => "Crop Image (Center)",
            Self::CropByBBoxes => "Crop By Bounding Boxes",
            Self::Crop => "Crop Image (DEPRECATED)",
            Self::CropV2 => "Crop Image",
            Self::Flip => "Flip Image",
            Self::PadForOutpaint => "Pad Image for Outpainting",
            Self::Rotate => "Rotate Image",
            Self::Stitch => "Stitch Images",
            Self::RandomCrop => "Crop Image (Random)",
            Self::ResizeAndPad => "Resize And Pad Image",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::CenterCrop => "Center crop an image to the specified dimensions.",
            Self::CropByBBoxes => {
                "Crop and resize regions from the input image batch based on provided bounding boxes."
            }
            Self::CropV2 => "Crop an image to the specified dimensions.",
            Self::Stitch => {
                "Stitches image2 to image1 in the specified direction.\nIf image2 is not provided, returns image1 unchanged.\nOptional spacing can be added between images."
            }
            Self::RandomCrop => "Randomly crop an image to the specified dimensions.",
            _ => "",
        }
    }

    fn aliases(self) -> Vec<String> {
        let values: &[&str] = match self {
            Self::CenterCrop | Self::RandomCrop => &["crop", "cut", "trim"],
            Self::CropByBBoxes => &["crop", "face crop", "bbox crop", "pose", "bounding box"],
            Self::Crop => &["trim"],
            Self::CropV2 => &["crop", "cut", "trim"],
            Self::Flip => &["mirror", "reflect"],
            Self::PadForOutpaint => &["extend canvas", "expand image"],
            Self::Rotate => &["turn", "flip orientation"],
            Self::Stitch => &[
                "combine images",
                "join images",
                "concatenate images",
                "side by side",
            ],
            Self::ResizeAndPad => &["fit to size"],
        };
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn input_names(self) -> Vec<String> {
        let values: &[&str] = match self {
            Self::CenterCrop => &["images", "width", "height"],
            Self::CropByBBoxes => &[
                "image",
                "bboxes",
                "output_width",
                "output_height",
                "padding",
                "keep_aspect",
            ],
            Self::Crop => &["image", "width", "height", "x", "y"],
            Self::CropV2 => &["image", "crop_region"],
            Self::Flip => &["image", "flip_method"],
            Self::PadForOutpaint => &["image", "left", "top", "right", "bottom", "feathering"],
            Self::Rotate => &["image", "rotation"],
            Self::Stitch => &[
                "image1",
                "direction",
                "match_image_size",
                "spacing_width",
                "spacing_color",
                "image2",
            ],
            Self::RandomCrop => &["images", "width", "height", "seed"],
            Self::ResizeAndPad => &[
                "image",
                "target_width",
                "target_height",
                "padding_color",
                "interpolation",
            ],
        };
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn output_names(self) -> Vec<String> {
        if self == Self::PadForOutpaint {
            vec!["image".to_owned(), "mask".to_owned()]
        } else if matches!(self, Self::CenterCrop | Self::RandomCrop) {
            vec!["images".to_owned()]
        } else {
            vec!["image".to_owned()]
        }
    }
}

const ALL_KINDS: [TransformKind; 10] = [
    TransformKind::CenterCrop,
    TransformKind::CropByBBoxes,
    TransformKind::Crop,
    TransformKind::CropV2,
    TransformKind::Flip,
    TransformKind::PadForOutpaint,
    TransformKind::Rotate,
    TransformKind::Stitch,
    TransformKind::RandomCrop,
    TransformKind::ResizeAndPad,
];

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    ALL_KINDS.into_iter().map(native_node_binding).collect()
}

fn native_node_binding(kind: TransformKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let input_names = kind.input_names();
    let output_names = kind.output_names();
    let source_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(&input_names, &[], &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: kind.implementation_version().to_owned(),
            source_schema: Some(source_schema),
            inputs: input_descriptors(kind)?,
            dynamic_inputs: Vec::new(),
            outputs: output_descriptors(kind)?,
            output_node: false,
            effect: if kind == TransformKind::CropV2 {
                NativeEffectClass::WritesArtifact
            } else {
                NativeEffectClass::Pure
            },
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: kind.display_name().to_owned(),
            category: "image/transform".to_owned(),
            description: kind.description().to_owned(),
            output_names,
            search_aliases: kind.aliases(),
            is_deprecated: kind == TransformKind::Crop,
            is_experimental: matches!(kind, TransformKind::CenterCrop | TransformKind::RandomCrop),
        },
        node: Arc::new(ImageTransformNode { kind }),
    })
}

fn input_descriptors(
    kind: TransformKind,
) -> Result<Vec<NativeInputDescriptor>, NativeNodeContractError> {
    let image = image_type()?;
    let bounding_box = bounding_box_type()?;
    let descriptors = match kind {
        TransformKind::CenterCrop => vec![
            handle_input("images", image, true)?,
            integer_input("width", true)?,
            integer_input("height", true)?,
        ],
        TransformKind::CropByBBoxes => vec![
            handle_input("image", image, true)?,
            handle_input("bboxes", bounding_box, true)?,
            integer_input("output_width", true)?,
            integer_input("output_height", true)?,
            integer_input("padding", true)?,
            string_input("keep_aspect", true)?,
        ],
        TransformKind::Crop => vec![
            handle_input("image", image, true)?,
            integer_input("width", true)?,
            integer_input("height", true)?,
            integer_input("x", true)?,
            integer_input("y", true)?,
        ],
        TransformKind::CropV2 => vec![
            handle_input("image", image, true)?,
            handle_input("crop_region", bounding_box, true)?,
        ],
        TransformKind::Flip => vec![
            handle_input("image", image, true)?,
            string_input("flip_method", true)?,
        ],
        TransformKind::PadForOutpaint => vec![
            handle_input("image", image, true)?,
            integer_input("left", true)?,
            integer_input("top", true)?,
            integer_input("right", true)?,
            integer_input("bottom", true)?,
            integer_input("feathering", true)?,
        ],
        TransformKind::Rotate => vec![
            handle_input("image", image, true)?,
            string_input("rotation", true)?,
        ],
        TransformKind::Stitch => vec![
            handle_input("image1", image.clone(), true)?,
            string_input("direction", true)?,
            boolean_input("match_image_size", true)?,
            integer_input("spacing_width", true)?,
            string_input("spacing_color", true)?,
            handle_input("image2", image, false)?,
        ],
        TransformKind::RandomCrop => vec![
            handle_input("images", image, true)?,
            integer_input("width", true)?,
            integer_input("height", true)?,
            integer_input("seed", true)?,
        ],
        TransformKind::ResizeAndPad => vec![
            handle_input("image", image, true)?,
            integer_input("target_width", true)?,
            integer_input("target_height", true)?,
            string_input("padding_color", true)?,
            string_input("interpolation", true)?,
        ],
    };
    Ok(descriptors)
}

fn output_descriptors(
    kind: TransformKind,
) -> Result<Vec<NativeOutputDescriptor>, NativeNodeContractError> {
    let image = NativeValueType::Handle(image_type()?);
    if kind == TransformKind::PadForOutpaint {
        Ok(vec![
            NativeOutputDescriptor {
                name: "image".to_owned(),
                produced_type: image,
                is_list: false,
            },
            NativeOutputDescriptor {
                name: "mask".to_owned(),
                produced_type: NativeValueType::Handle(mask_type()?),
                is_list: false,
            },
        ])
    } else {
        Ok(vec![NativeOutputDescriptor {
            name: if matches!(kind, TransformKind::CenterCrop | TransformKind::RandomCrop) {
                "images".to_owned()
            } else {
                "image".to_owned()
            },
            produced_type: image,
            is_list: false,
        }])
    }
}

fn handle_input(
    name: &str,
    handle_type: NativeHandleType,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([NativeValueType::Handle(handle_type)])?,
        required,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: false,
    })
}

fn primitive_input(
    name: &str,
    primitive_type: NativePrimitiveType,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([NativeValueType::Primitive(primitive_type)])?,
        required,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: true,
    })
}

fn integer_input(
    name: &str,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    primitive_input(name, NativePrimitiveType::Integer, required)
}

fn string_input(
    name: &str,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    primitive_input(name, NativePrimitiveType::String, required)
}

fn boolean_input(
    name: &str,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    primitive_input(name, NativePrimitiveType::Boolean, required)
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

fn mask_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Mask, "MASK")
}

fn bounding_box_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::StructuredCompute, "BOUNDING_BOX")
}

#[derive(Debug)]
struct ImageTransformNode {
    kind: TransformKind,
}

impl NativeNode for ImageTransformNode {
    fn class_type(&self) -> &str {
        self.kind.class_type()
    }

    fn implementation_version(&self) -> &str {
        self.kind.implementation_version()
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<std::collections::BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context, self.kind.class_type())?;
        validate_inputs(self.kind, available_inputs)?;
        Ok(std::collections::BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        validate_inputs(self.kind, inputs)?;
        Ok(format!(
            "{}-{}-input-identity",
            self.kind.class_type(),
            self.kind.implementation_version()
        ))
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, self.kind.class_type())?;
        validate_inputs(self.kind, inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, self.kind.class_type())?;
            validate_inputs(self.kind, &inputs)?;
            match self.kind {
                TransformKind::CenterCrop => execute_center_crop(&context, &inputs),
                TransformKind::CropByBBoxes => execute_crop_by_bboxes(&context, &inputs),
                TransformKind::Crop => execute_crop(&context, &inputs, false),
                TransformKind::CropV2 => execute_crop(&context, &inputs, true),
                TransformKind::Flip => execute_flip(&context, &inputs),
                TransformKind::PadForOutpaint => execute_pad_for_outpaint(&context, &inputs),
                TransformKind::Rotate => execute_rotate(&context, &inputs),
                TransformKind::Stitch => execute_stitch(&context, &inputs),
                TransformKind::RandomCrop => execute_random_crop(&context, &inputs),
                TransformKind::ResizeAndPad => execute_resize_and_pad(&context, &inputs),
            }
        })
    }
}

fn execute_center_crop(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image = resolve_image(
        context,
        handle(inputs, "images")?,
        TransformKind::CenterCrop,
    )?;
    let width = bounded_u64(inputs, "width", 1, 8_192)?;
    let height = bounded_u64(inputs, "height", 1, 8_192)?;
    let (batch, image_height, image_width, _) = image.dimensions().map_err(native_failure)?;
    if batch != 1 {
        return Err(invalid_inputs(
            "CenterCropImages requires exactly one image",
        ));
    }
    let left = image_width.saturating_sub(width) / 2;
    let top = image_height.saturating_sub(height) / 2;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = image
        .source_compatible_u8_crop(
            left,
            top,
            width.min(image_width),
            height.min(image_height),
            compute.backend(),
            &execution,
        )
        .map_err(tensor_failure)?;
    publish_images(
        context,
        vec![output],
        None,
        Vec::new(),
        TransformKind::CenterCrop,
    )
}

fn execute_random_crop(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image = resolve_image(
        context,
        handle(inputs, "images")?,
        TransformKind::RandomCrop,
    )?;
    let width = bounded_u64(inputs, "width", 1, 8_192)?;
    let height = bounded_u64(inputs, "height", 1, 8_192)?;
    let seed = integer(inputs, "seed")?;
    let (batch, image_height, image_width, _) = image.dimensions().map_err(native_failure)?;
    if batch != 1 {
        return Err(invalid_inputs(
            "RandomCropImages requires exactly one image",
        ));
    }
    let mut random = NumpyRandomState::from_seed(seed);
    let max_left = image_width.saturating_sub(width);
    let max_top = image_height.saturating_sub(height);
    let left = if max_left == 0 {
        0
    } else {
        u64::from(
            random
                .randint(
                    0,
                    u32::try_from(max_left + 1)
                        .map_err(|_| invalid_inputs("random crop width exceeds NumPy range"))?,
                    &context.cancellation,
                )
                .map_err(rng_failure)?,
        )
    };
    let top = if max_top == 0 {
        0
    } else {
        u64::from(
            random
                .randint(
                    0,
                    u32::try_from(max_top + 1)
                        .map_err(|_| invalid_inputs("random crop height exceeds NumPy range"))?,
                    &context.cancellation,
                )
                .map_err(rng_failure)?,
        )
    };
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = image
        .source_compatible_u8_crop(
            left,
            top,
            width.min(image_width),
            height.min(image_height),
            compute.backend(),
            &execution,
        )
        .map_err(tensor_failure)?;
    publish_images(
        context,
        vec![output],
        None,
        Vec::new(),
        TransformKind::RandomCrop,
    )
}

fn execute_crop(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
    preview: bool,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let kind = if preview {
        TransformKind::CropV2
    } else {
        TransformKind::Crop
    };
    let image = resolve_image(context, handle(inputs, "image")?, kind)?;
    let (x, y, width, height) = if preview {
        let boxes = resolve_bounding_boxes(context, handle(inputs, "crop_region")?, kind)?;
        let bounding_box = boxes
            .frames()
            .first()
            .and_then(|frame| frame.first())
            .ok_or_else(|| invalid_inputs("crop_region must contain one bounding box"))?;
        (
            integral_u64(bounding_box.x(), "crop x")?,
            integral_u64(bounding_box.y(), "crop y")?,
            integral_u64(bounding_box.width(), "crop width")?,
            integral_u64(bounding_box.height(), "crop height")?,
        )
    } else {
        (
            bounded_u64(inputs, "x", 0, MAX_RESOLUTION)?,
            bounded_u64(inputs, "y", 0, MAX_RESOLUTION)?,
            bounded_u64(inputs, "width", 1, MAX_RESOLUTION)?,
            bounded_u64(inputs, "height", 1, MAX_RESOLUTION)?,
        )
    };
    let (_, image_height, image_width, _) = image.dimensions().map_err(native_failure)?;
    let x = x.min(image_width.saturating_sub(1));
    let y = y.min(image_height.saturating_sub(1));
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = crop_f32(&image, x, y, width, height, compute.backend(), &execution)?;
    if preview {
        let prepared = context
            .prepare_image_preview(&output, "ImageCropV2")
            .map_err(|error| image_preview_failure(kind, error))?;
        let (effects, ui) = prepared.into_parts();
        publish_images(context, vec![output], Some(ui), effects, kind)
    } else {
        publish_images(context, vec![output], None, Vec::new(), kind)
    }
}

fn execute_flip(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image = resolve_image(context, handle(inputs, "image")?, TransformKind::Flip)?;
    let method = string(inputs, "flip_method")?;
    let vertical = match method {
        "x-axis: vertically" => true,
        "y-axis: horizontally" => false,
        _ => return Err(invalid_inputs("flip_method is not supported")),
    };
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = remap_image(
        &image,
        compute.backend(),
        &execution,
        |y, x, height, width| {
            if vertical {
                (height - 1 - y, x)
            } else {
                (y, width - 1 - x)
            }
        },
    )?;
    publish_images(context, vec![output], None, Vec::new(), TransformKind::Flip)
}

fn execute_rotate(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image = resolve_image(context, handle(inputs, "image")?, TransformKind::Rotate)?;
    let rotation = string(inputs, "rotation")?;
    let turns = match rotation {
        "none" => 0,
        "90 degrees" => 1,
        "180 degrees" => 2,
        "270 degrees" => 3,
        _ => return Err(invalid_inputs("rotation is not supported")),
    };
    if turns == 0 {
        return values_outcome(vec![NativeValue::Handle {
            value: handle(inputs, "image")?.clone(),
        }]);
    }
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = rotate_image(&image, turns, compute.backend(), &execution)?;
    publish_images(
        context,
        vec![output],
        None,
        Vec::new(),
        TransformKind::Rotate,
    )
}

fn execute_resize_and_pad(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image = resolve_image(
        context,
        handle(inputs, "image")?,
        TransformKind::ResizeAndPad,
    )?;
    let target_width = bounded_u64(inputs, "target_width", 1, MAX_RESOLUTION)?;
    let target_height = bounded_u64(inputs, "target_height", 1, MAX_RESOLUTION)?;
    let fill = match string(inputs, "padding_color")? {
        "black" => 0.0,
        "white" => 1.0,
        _ => return Err(invalid_inputs("padding_color is not supported")),
    };
    let mode = resize_mode(string(inputs, "interpolation")?)?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = resize_and_pad(
        &image,
        target_width,
        target_height,
        fill,
        mode,
        compute.backend(),
        &execution,
    )?;
    publish_images(
        context,
        vec![output],
        None,
        Vec::new(),
        TransformKind::ResizeAndPad,
    )
}

fn execute_pad_for_outpaint(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image = resolve_image(
        context,
        handle(inputs, "image")?,
        TransformKind::PadForOutpaint,
    )?;
    let left = bounded_u64(inputs, "left", 0, MAX_RESOLUTION)?;
    let top = bounded_u64(inputs, "top", 0, MAX_RESOLUTION)?;
    let right = bounded_u64(inputs, "right", 0, MAX_RESOLUTION)?;
    let bottom = bounded_u64(inputs, "bottom", 0, MAX_RESOLUTION)?;
    let feathering = bounded_u64(inputs, "feathering", 0, MAX_RESOLUTION)?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let (output, mask) = pad_for_outpaint(
        &image,
        left,
        top,
        right,
        bottom,
        feathering,
        compute.backend(),
        &execution,
    )?;
    publish_image_and_mask(context, output, mask)
}

fn execute_stitch(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let first_handle = handle(inputs, "image1")?;
    if inputs.get("image2").is_none() {
        return values_outcome(vec![NativeValue::Handle {
            value: first_handle.clone(),
        }]);
    }
    let first = resolve_image(context, first_handle, TransformKind::Stitch)?;
    let second = resolve_image(context, handle(inputs, "image2")?, TransformKind::Stitch)?;
    let direction = string(inputs, "direction")?;
    let match_size = boolean(inputs, "match_image_size")?;
    let spacing_width = bounded_u64(inputs, "spacing_width", 0, 1_024)?;
    let spacing_color = spacing_color(string(inputs, "spacing_color")?)?;
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = stitch_images(
        &first,
        &second,
        direction,
        match_size,
        spacing_width,
        spacing_color,
        compute.backend(),
        &execution,
    )?;
    publish_images(
        context,
        vec![output],
        None,
        Vec::new(),
        TransformKind::Stitch,
    )
}

fn execute_crop_by_bboxes(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image_handle = handle(inputs, "image")?;
    let image = resolve_image(context, image_handle, TransformKind::CropByBBoxes)?;
    let boxes = resolve_bounding_boxes(
        context,
        handle(inputs, "bboxes")?,
        TransformKind::CropByBBoxes,
    )?;
    if boxes.frames().is_empty() {
        return values_outcome(vec![NativeValue::Handle {
            value: image_handle.clone(),
        }]);
    }
    let output_width = bounded_u64(inputs, "output_width", 64, 4_096)?;
    let output_height = bounded_u64(inputs, "output_height", 64, 4_096)?;
    let padding = bounded_u64(inputs, "padding", 0, 1_024)?;
    let keep_aspect = string(inputs, "keep_aspect")?;
    if !matches!(keep_aspect, "stretch" | "pad") {
        return Err(invalid_inputs("keep_aspect is not supported"));
    }
    let compute = context.compute_session().map_err(compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(compute_failure)?;
    let output = crop_by_bboxes(
        &image,
        &boxes,
        output_width,
        output_height,
        padding,
        keep_aspect == "pad",
        compute.backend(),
        &execution,
    )?;
    let Some(output) = output else {
        return values_outcome(vec![NativeValue::Handle {
            value: image_handle.clone(),
        }]);
    };
    publish_images(
        context,
        vec![output],
        None,
        Vec::new(),
        TransformKind::CropByBBoxes,
    )
}

fn crop_f32(
    image: &ImageTensor,
    left: u64,
    top: u64,
    width: u64,
    height: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, input_height, input_width, channels) =
        image.dimensions().map_err(native_failure)?;
    if input_width == 0 || input_height == 0 || width == 0 || height == 0 {
        return Err(invalid_inputs("crop dimensions must be non-zero"));
    }
    let output_width = width.min(input_width - left);
    let output_height = height.min(input_height - top);
    let capacity = element_count(batch, output_height, output_width, channels)?;
    let source = image.as_f32_slice().map_err(native_failure)?;
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(native_failure)?;
    for batch_index in 0..batch {
        for y in 0..output_height {
            execution.check().map_err(tensor_failure)?;
            for x in 0..output_width {
                for channel in 0..channels {
                    values.push(value_at(
                        source,
                        batch_index,
                        top + y,
                        left + x,
                        channel,
                        input_height,
                        input_width,
                        channels,
                    )?);
                }
            }
        }
    }
    ImageTensor::from_f32(
        backend,
        execution,
        batch,
        output_height,
        output_width,
        channels,
        &values,
    )
    .map_err(native_failure)
}

fn remap_image(
    image: &ImageTensor,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
    map: impl Fn(u64, u64, u64, u64) -> (u64, u64),
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    let source = image.as_f32_slice().map_err(native_failure)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count(batch, height, width, channels)?)
        .map_err(native_failure)?;
    for batch_index in 0..batch {
        for y in 0..height {
            execution.check().map_err(tensor_failure)?;
            for x in 0..width {
                let (source_y, source_x) = map(y, x, height, width);
                for channel in 0..channels {
                    values.push(value_at(
                        source,
                        batch_index,
                        source_y,
                        source_x,
                        channel,
                        height,
                        width,
                        channels,
                    )?);
                }
            }
        }
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, channels, &values)
        .map_err(native_failure)
}

fn rotate_image(
    image: &ImageTensor,
    turns: u8,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    let (output_height, output_width) = if turns.is_multiple_of(2) {
        (height, width)
    } else {
        (width, height)
    };
    let source = image.as_f32_slice().map_err(native_failure)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count(batch, output_height, output_width, channels)?)
        .map_err(native_failure)?;
    for batch_index in 0..batch {
        for y in 0..output_height {
            execution.check().map_err(tensor_failure)?;
            for x in 0..output_width {
                let (source_y, source_x) = match turns {
                    1 => (x, width - 1 - y),
                    2 => (height - 1 - y, width - 1 - x),
                    3 => (height - 1 - x, y),
                    _ => return Err(invalid_inputs("rotation turn count is invalid")),
                };
                for channel in 0..channels {
                    values.push(value_at(
                        source,
                        batch_index,
                        source_y,
                        source_x,
                        channel,
                        height,
                        width,
                        channels,
                    )?);
                }
            }
        }
    }
    ImageTensor::from_f32(
        backend,
        execution,
        batch,
        output_height,
        output_width,
        channels,
        &values,
    )
    .map_err(native_failure)
}

fn resize_and_pad(
    image: &ImageTensor,
    target_width: u64,
    target_height: u64,
    fill: f32,
    mode: ResizeMode,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    let (new_width, new_height) =
        fit_dimensions(width, height, target_width, target_height, false)?;
    let resized = image
        .resize(
            new_width,
            new_height,
            mode,
            ResizeCrop::Disabled,
            backend,
            execution,
        )
        .map_err(tensor_failure)?;
    let mut values = filled_values(
        element_count(batch, target_height, target_width, channels)?,
        fill,
    )?;
    let x_offset = (target_width - new_width) / 2;
    let y_offset = (target_height - new_height) / 2;
    paste_image(
        &mut values,
        batch,
        target_height,
        target_width,
        channels,
        resized.as_f32_slice().map_err(native_failure)?,
        batch,
        new_height,
        new_width,
        channels,
        0,
        y_offset,
        x_offset,
        execution,
    )?;
    ImageTensor::from_f32(
        backend,
        execution,
        batch,
        target_height,
        target_width,
        channels,
        &values,
    )
    .map_err(tensor_failure)
}

#[allow(clippy::too_many_arguments)]
fn pad_for_outpaint(
    image: &ImageTensor,
    left: u64,
    top: u64,
    right: u64,
    bottom: u64,
    feathering: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<(ImageTensor, sim_tensor::Tensor), NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    let output_width = width
        .checked_add(left)
        .and_then(|value| value.checked_add(right))
        .ok_or_else(|| invalid_inputs("outpaint width overflowed"))?;
    let output_height = height
        .checked_add(top)
        .and_then(|value| value.checked_add(bottom))
        .ok_or_else(|| invalid_inputs("outpaint height overflowed"))?;
    let mut image_values = filled_values(
        element_count(batch, output_height, output_width, channels)?,
        0.5,
    )?;
    paste_image(
        &mut image_values,
        batch,
        output_height,
        output_width,
        channels,
        image.as_f32_slice().map_err(native_failure)?,
        batch,
        height,
        width,
        channels,
        0,
        top,
        left,
        execution,
    )?;
    let mut mask_values = filled_values(element_count(1, output_height, output_width, 1)?, 1.0)?;
    for y in 0..height {
        execution.check().map_err(tensor_failure)?;
        for x in 0..width {
            let value = if feathering > 0
                && feathering.saturating_mul(2) < height
                && feathering.saturating_mul(2) < width
            {
                let distance_top = if top != 0 { y } else { height };
                let distance_bottom = if bottom != 0 { height - y } else { height };
                let distance_left = if left != 0 { x } else { width };
                let distance_right = if right != 0 { width - x } else { width };
                let distance = distance_top
                    .min(distance_bottom)
                    .min(distance_left)
                    .min(distance_right);
                if distance < feathering {
                    let ratio = (feathering - distance) as f32 / feathering as f32;
                    ratio * ratio
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let index = image_offset(0, top + y, left + x, 0, output_height, output_width, 1)?;
            *mask_values
                .get_mut(index)
                .ok_or_else(|| native_failure("outpaint mask index exceeded storage"))? = value;
        }
    }
    let output = ImageTensor::from_f32(
        backend,
        execution,
        batch,
        output_height,
        output_width,
        channels,
        &image_values,
    )
    .map_err(tensor_failure)?;
    let descriptor = sim_tensor::TensorDescriptor::contiguous(
        vec![1, output_height, output_width],
        sim_tensor::DType::F32,
        sim_tensor::DeviceId::CPU,
        image.tensor().descriptor().stream(),
    )
    .map_err(native_failure)?;
    let (mask, _) = backend
        .upload_f32(descriptor, &mask_values, execution)
        .map_err(tensor_failure)?;
    Ok((output, mask))
}

#[allow(clippy::too_many_arguments)]
fn stitch_images(
    first: &ImageTensor,
    second: &ImageTensor,
    direction: &str,
    match_size: bool,
    spacing_width: u64,
    spacing_color: [f32; 3],
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    if !matches!(direction, "right" | "down" | "left" | "up") {
        return Err(invalid_inputs("direction is not supported"));
    }
    let (first_batch, first_height, first_width, first_channels) =
        first.dimensions().map_err(native_failure)?;
    let (second_batch, _, _, second_channels) = second.dimensions().map_err(native_failure)?;
    if first_batch == 0 || second_batch == 0 {
        return Err(invalid_inputs(
            "ImageStitch requires non-empty image batches",
        ));
    }
    let batch = first_batch.max(second_batch);
    let channels = first_channels.max(second_channels);
    let first = expand_batch_and_channels(first, batch, channels, backend, execution)?;
    let mut second = expand_batch_and_channels(second, batch, channels, backend, execution)?;
    if match_size {
        let (_, current_height, current_width, _) = second.dimensions().map_err(native_failure)?;
        let (target_width, target_height) = if matches!(direction, "left" | "right") {
            (
                ((first_height as f64) * (current_width as f64 / current_height as f64)) as u64,
                first_height,
            )
        } else {
            (
                first_width,
                ((first_width as f64) / (current_width as f64 / current_height as f64)) as u64,
            )
        };
        second = second
            .resize(
                target_width,
                target_height,
                ResizeMode::Lanczos,
                ResizeCrop::Disabled,
                backend,
                execution,
            )
            .map_err(tensor_failure)?;
    }
    let (_, second_height, second_width, _) = second.dimensions().map_err(native_failure)?;
    let horizontal = matches!(direction, "left" | "right");
    let aligned_height = if horizontal {
        first_height.max(second_height)
    } else {
        first_height
    };
    let aligned_width = if horizontal {
        first_width
    } else {
        first_width.max(second_width)
    };
    let first = if match_size {
        first
    } else {
        center_pad(
            &first,
            aligned_width,
            aligned_height,
            scalar_pad_color(spacing_color),
            backend,
            execution,
        )?
    };
    let second = if match_size {
        second
    } else {
        center_pad(
            &second,
            if horizontal {
                second_width
            } else {
                aligned_width
            },
            if horizontal {
                aligned_height
            } else {
                second_height
            },
            scalar_pad_color(spacing_color),
            backend,
            execution,
        )?
    };
    let (_, first_height, first_width, channels) = first.dimensions().map_err(native_failure)?;
    let (_, second_height, second_width, _) = second.dimensions().map_err(native_failure)?;
    let spacing_width = spacing_width + spacing_width % 2;
    let output_width = if horizontal {
        first_width
            .checked_add(second_width)
            .and_then(|value| value.checked_add(spacing_width))
            .ok_or_else(|| invalid_inputs("stitched image width overflowed"))?
    } else {
        first_width.max(second_width)
    };
    let output_height = if horizontal {
        first_height.max(second_height)
    } else {
        first_height
            .checked_add(second_height)
            .and_then(|value| value.checked_add(spacing_width))
            .ok_or_else(|| invalid_inputs("stitched image height overflowed"))?
    };
    let mut output = filled_values(
        element_count(batch, output_height, output_width, channels)?,
        0.0,
    )?;
    let (leading, trailing) = if matches!(direction, "left" | "up") {
        (&second, &first)
    } else {
        (&first, &second)
    };
    let (_, leading_height, leading_width, _) = leading.dimensions().map_err(native_failure)?;
    paste_image(
        &mut output,
        batch,
        output_height,
        output_width,
        channels,
        leading.as_f32_slice().map_err(native_failure)?,
        batch,
        leading_height,
        leading_width,
        channels,
        0,
        0,
        0,
        execution,
    )?;
    if spacing_width > 0 {
        fill_spacing(
            &mut output,
            batch,
            output_height,
            output_width,
            channels,
            horizontal,
            if horizontal {
                leading_width
            } else {
                leading_height
            },
            spacing_width,
            spacing_color,
            execution,
        )?;
    }
    let (_, trailing_height, trailing_width, _) = trailing.dimensions().map_err(native_failure)?;
    let (trailing_y, trailing_x) = if horizontal {
        (0, leading_width + spacing_width)
    } else {
        (leading_height + spacing_width, 0)
    };
    paste_image(
        &mut output,
        batch,
        output_height,
        output_width,
        channels,
        trailing.as_f32_slice().map_err(native_failure)?,
        batch,
        trailing_height,
        trailing_width,
        channels,
        0,
        trailing_y,
        trailing_x,
        execution,
    )?;
    ImageTensor::from_f32(
        backend,
        execution,
        batch,
        output_height,
        output_width,
        channels,
        &output,
    )
    .map_err(native_failure)
}

#[allow(clippy::too_many_arguments)]
fn crop_by_bboxes(
    image: &ImageTensor,
    boxes: &NativeBoundingBoxPayload,
    output_width: u64,
    output_height: u64,
    padding: u64,
    keep_aspect: bool,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<Option<ImageTensor>, NativeNodeFailure> {
    let (batch, image_height, image_width, channels) =
        image.dimensions().map_err(native_failure)?;
    let frames = boxes.frames();
    let mut crops = Vec::new();
    for frame_index in 0..batch {
        execution.check().map_err(tensor_failure)?;
        let frame = frames
            .get(
                usize::try_from(frame_index)
                    .map_err(native_failure)?
                    .min(frames.len() - 1),
            )
            .ok_or_else(|| native_failure("bounding-box frame index exceeded storage"))?;
        if frame.is_empty() {
            continue;
        }
        let (x1, y1, x2, y2) = bounding_union(frame)?;
        let padding = i64::try_from(padding)
            .map_err(|_| invalid_inputs("bounding-box padding exceeds the native range"))?;
        let image_width_i64 = i64::try_from(image_width)
            .map_err(|_| invalid_inputs("IMAGE width exceeds the native bounding-box range"))?;
        let image_height_i64 = i64::try_from(image_height)
            .map_err(|_| invalid_inputs("IMAGE height exceeds the native bounding-box range"))?;
        let mut x1 = u64::try_from(x1.saturating_sub(padding).max(0)).map_err(native_failure)?;
        let mut y1 = u64::try_from(y1.saturating_sub(padding).max(0)).map_err(native_failure)?;
        let mut x2 = u64::try_from(x2.saturating_add(padding).clamp(0, image_width_i64))
            .map_err(native_failure)?;
        let mut y2 = u64::try_from(y2.saturating_add(padding).clamp(0, image_height_i64))
            .map_err(native_failure)?;
        if x2 <= x1 || y2 <= y1 {
            let fallback_size = ((image_height.min(image_width) as f64) * 0.3) as u64;
            x1 = image_width.saturating_sub(fallback_size) / 2;
            y1 = ((image_height as f64) * 0.1) as u64;
            x2 = x1.saturating_add(fallback_size).min(image_width);
            y2 = y1.saturating_add(fallback_size).min(image_height);
            if x2 <= x1 || y2 <= y1 {
                crops.push(
                    ImageTensor::from_f32(
                        backend,
                        execution,
                        1,
                        output_height,
                        output_width,
                        channels,
                        &filled_values(
                            element_count(1, output_height, output_width, channels)?,
                            0.0,
                        )?,
                    )
                    .map_err(native_failure)?,
                );
                continue;
            }
        }
        let frame_image = crop_f32(image, x1, y1, x2 - x1, y2 - y1, backend, execution)?;
        let frame_image = select_batch(&frame_image, frame_index, backend, execution)?;
        let resized = if keep_aspect {
            let (scaled_width, scaled_height) =
                fit_dimensions(x2 - x1, y2 - y1, output_width, output_height, true)?;
            let scaled = frame_image
                .resize(
                    scaled_width,
                    scaled_height,
                    ResizeMode::Area,
                    ResizeCrop::Disabled,
                    backend,
                    execution,
                )
                .map_err(tensor_failure)?;
            center_pad(
                &scaled,
                output_width,
                output_height,
                0.0,
                backend,
                execution,
            )?
        } else {
            frame_image
                .resize(
                    output_width,
                    output_height,
                    ResizeMode::Area,
                    ResizeCrop::Disabled,
                    backend,
                    execution,
                )
                .map_err(tensor_failure)?
        };
        crops.push(resized);
    }
    if crops.is_empty() {
        return Ok(None);
    }
    concatenate_batches(&crops, backend, execution).map(Some)
}

fn select_batch(
    image: &ImageTensor,
    index: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    if index >= batch {
        return Err(native_failure("batch selection exceeded IMAGE storage"));
    }
    let frame_elements = element_count(1, height, width, channels)?;
    let start = usize::try_from(index)
        .map_err(native_failure)?
        .checked_mul(frame_elements)
        .ok_or_else(|| native_failure("batch selection overflowed"))?;
    let end = start
        .checked_add(frame_elements)
        .ok_or_else(|| native_failure("batch selection overflowed"))?;
    let values = image
        .as_f32_slice()
        .map_err(native_failure)?
        .get(start..end)
        .ok_or_else(|| native_failure("batch selection exceeded IMAGE storage"))?;
    ImageTensor::from_f32(backend, execution, 1, height, width, channels, values)
        .map_err(native_failure)
}

fn concatenate_batches(
    images: &[ImageTensor],
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let first = images
        .first()
        .ok_or_else(|| invalid_inputs("cannot concatenate an empty crop list"))?;
    let (_, height, width, channels) = first.dimensions().map_err(native_failure)?;
    let mut batch = 0_u64;
    let mut values = Vec::new();
    for image in images {
        execution.check().map_err(tensor_failure)?;
        let (image_batch, image_height, image_width, image_channels) =
            image.dimensions().map_err(native_failure)?;
        if (image_height, image_width, image_channels) != (height, width, channels) {
            return Err(native_failure(
                "crop dimensions changed before concatenation",
            ));
        }
        batch = batch
            .checked_add(image_batch)
            .ok_or_else(|| native_failure("crop batch size overflowed"))?;
        let image_values = image.as_f32_slice().map_err(native_failure)?;
        values
            .try_reserve(image_values.len())
            .map_err(native_failure)?;
        values.extend_from_slice(image_values);
    }
    ImageTensor::from_f32(backend, execution, batch, height, width, channels, &values)
        .map_err(native_failure)
}

fn bounding_union(frame: &[NativeBoundingBox]) -> Result<(i64, i64, i64, i64), NativeNodeFailure> {
    let first = frame
        .first()
        .ok_or_else(|| invalid_inputs("bounding-box frame is empty"))?;
    let mut x1 = truncating_i64(first.x(), "bounding-box x")?;
    let mut y1 = truncating_i64(first.y(), "bounding-box y")?;
    let mut x2 = truncating_i64(first.x() + first.width(), "bounding-box right")?;
    let mut y2 = truncating_i64(first.y() + first.height(), "bounding-box bottom")?;
    for bounding_box in &frame[1..] {
        x1 = x1.min(truncating_i64(bounding_box.x(), "bounding-box x")?);
        y1 = y1.min(truncating_i64(bounding_box.y(), "bounding-box y")?);
        x2 = x2.max(truncating_i64(
            bounding_box.x() + bounding_box.width(),
            "bounding-box right",
        )?);
        y2 = y2.max(truncating_i64(
            bounding_box.y() + bounding_box.height(),
            "bounding-box bottom",
        )?);
    }
    Ok((x1, y1, x2, y2))
}

fn expand_batch_and_channels(
    image: &ImageTensor,
    target_batch: u64,
    target_channels: u64,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    if batch == target_batch && channels == target_channels {
        return Ok(image.clone());
    }
    let source = image.as_f32_slice().map_err(native_failure)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count(target_batch, height, width, target_channels)?)
        .map_err(native_failure)?;
    for target_batch_index in 0..target_batch {
        execution.check().map_err(tensor_failure)?;
        let source_batch = target_batch_index.min(batch - 1);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..target_channels {
                    values.push(if channel < channels {
                        value_at(source, source_batch, y, x, channel, height, width, channels)?
                    } else {
                        1.0
                    });
                }
            }
        }
    }
    ImageTensor::from_f32(
        backend,
        execution,
        target_batch,
        height,
        width,
        target_channels,
        &values,
    )
    .map_err(native_failure)
}

fn center_pad(
    image: &ImageTensor,
    target_width: u64,
    target_height: u64,
    fill: f32,
    backend: &sim_tensor::CpuBackend,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, height, width, channels) = image.dimensions().map_err(native_failure)?;
    if width == target_width && height == target_height {
        return Ok(image.clone());
    }
    if width > target_width || height > target_height {
        return Err(native_failure("center padding cannot shrink an image"));
    }
    let mut values = filled_values(
        element_count(batch, target_height, target_width, channels)?,
        fill,
    )?;
    paste_image(
        &mut values,
        batch,
        target_height,
        target_width,
        channels,
        image.as_f32_slice().map_err(native_failure)?,
        batch,
        height,
        width,
        channels,
        0,
        (target_height - height) / 2,
        (target_width - width) / 2,
        execution,
    )?;
    ImageTensor::from_f32(
        backend,
        execution,
        batch,
        target_height,
        target_width,
        channels,
        &values,
    )
    .map_err(native_failure)
}

#[allow(clippy::too_many_arguments)]
fn paste_image(
    output: &mut [f32],
    output_batch: u64,
    output_height: u64,
    output_width: u64,
    output_channels: u64,
    source: &[f32],
    source_batch: u64,
    source_height: u64,
    source_width: u64,
    source_channels: u64,
    batch_offset: u64,
    y_offset: u64,
    x_offset: u64,
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<(), NativeNodeFailure> {
    if batch_offset.checked_add(source_batch).is_none_or(|end| end > output_batch)
        || y_offset.checked_add(source_height).is_none_or(|end| end > output_height)
        || x_offset.checked_add(source_width).is_none_or(|end| end > output_width)
        || source_channels != output_channels
    {
        return Err(native_failure("image paste exceeds destination bounds"));
    }
    for batch in 0..source_batch {
        for y in 0..source_height {
            execution.check().map_err(tensor_failure)?;
            for x in 0..source_width {
                for channel in 0..source_channels {
                    let value = value_at(
                        source,
                        batch,
                        y,
                        x,
                        channel,
                        source_height,
                        source_width,
                        source_channels,
                    )?;
                    let index = image_offset(
                        batch_offset + batch,
                        y_offset + y,
                        x_offset + x,
                        channel,
                        output_height,
                        output_width,
                        output_channels,
                    )?;
                    *output
                        .get_mut(index)
                        .ok_or_else(|| native_failure("image paste index exceeded storage"))? =
                        value;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fill_spacing(
    output: &mut [f32],
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    horizontal: bool,
    offset: u64,
    spacing_width: u64,
    color: [f32; 3],
    execution: &sim_tensor::ExecutionContext<'_>,
) -> Result<(), NativeNodeFailure> {
    let spacing_end = offset
        .checked_add(spacing_width)
        .ok_or_else(|| native_failure("spacing range overflowed"))?;
    for batch_index in 0..batch {
        let (y_range, x_range) = if horizontal {
            (0..height, offset..spacing_end)
        } else {
            (offset..spacing_end, 0..width)
        };
        for y in y_range {
            execution.check().map_err(tensor_failure)?;
            for x in x_range.clone() {
                for channel in 0..channels {
                    let value = if channel < 3 {
                        color[usize::try_from(channel).map_err(native_failure)?]
                    } else if channel == 3 {
                        1.0
                    } else {
                        0.0
                    };
                    let index = image_offset(batch_index, y, x, channel, height, width, channels)?;
                    *output
                        .get_mut(index)
                        .ok_or_else(|| native_failure("spacing index exceeded storage"))? = value;
                }
            }
        }
    }
    Ok(())
}

fn fit_dimensions(
    width: u64,
    height: u64,
    target_width: u64,
    target_height: u64,
    round: bool,
) -> Result<(u64, u64), NativeNodeFailure> {
    if width == 0 || height == 0 {
        return Err(invalid_inputs(
            "cannot fit an image with a zero spatial dimension",
        ));
    }
    let scale = (target_width as f64 / width as f64).min(target_height as f64 / height as f64);
    let scaled_width = width as f64 * scale;
    let scaled_height = height as f64 * scale;
    let new_width = if round {
        scaled_width.round_ties_even() as u64
    } else {
        scaled_width as u64
    };
    let new_height = if round {
        scaled_height.round_ties_even() as u64
    } else {
        scaled_height as u64
    };
    if new_width == 0 || new_height == 0 {
        return Err(invalid_inputs(
            "fitted image dimensions truncate to zero",
        ));
    }
    Ok((new_width, new_height))
}

fn resize_mode(value: &str) -> Result<ResizeMode, NativeNodeFailure> {
    match value {
        "area" => Ok(ResizeMode::Area),
        "bicubic" => Ok(ResizeMode::Bicubic),
        "nearest-exact" => Ok(ResizeMode::NearestExact),
        "bilinear" => Ok(ResizeMode::Bilinear),
        "lanczos" => Ok(ResizeMode::Lanczos),
        _ => Err(invalid_inputs("interpolation is not supported")),
    }
}

fn spacing_color(value: &str) -> Result<[f32; 3], NativeNodeFailure> {
    match value {
        "white" => Ok([1.0, 1.0, 1.0]),
        "black" => Ok([0.0, 0.0, 0.0]),
        "red" => Ok([1.0, 0.0, 0.0]),
        "green" => Ok([0.0, 1.0, 0.0]),
        "blue" => Ok([0.0, 0.0, 1.0]),
        _ => Err(invalid_inputs("spacing_color is not supported")),
    }
}

fn scalar_pad_color(color: [f32; 3]) -> f32 {
    if color == [1.0, 1.0, 1.0] { 1.0 } else { 0.0 }
}

fn resolve_image(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
    kind: TransformKind,
) -> Result<ImageTensor, NativeNodeFailure> {
    let expected = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected, &context.cancellation)
        .map_err(|error| handle_failure(kind, error))?;
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(native_failure(
            "IMAGE handle did not resolve to tensor storage",
        ));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(native_failure(
            "IMAGE handle resolved to the wrong tensor role",
        ));
    }
    payload
        .image()
        .cloned()
        .ok_or_else(|| native_failure("IMAGE handle did not resolve to canonical image storage"))
}

fn resolve_bounding_boxes(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
    kind: TransformKind,
) -> Result<Arc<NativeBoundingBoxPayload>, NativeNodeFailure> {
    let expected = bounding_box_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let resolved = context
        .handle_store()
        .resolve(handle, &expected, &context.cancellation)
        .map_err(|error| handle_failure(kind, error))?;
    let NativeStoredPayload::BoundingBox(payload) = resolved.as_ref() else {
        return Err(native_failure(
            "BOUNDING_BOX handle did not resolve to bounding-box storage",
        ));
    };
    Ok(payload.clone())
}

fn publish_images(
    context: &NativeNodeContext,
    images: Vec<ImageTensor>,
    ui: Option<serde_json::Value>,
    effects: Vec<NativePreparedEffectRequest>,
    kind: TransformKind,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let mut published = Vec::new();
    for image in images {
        let payload = match NativeTensorPayload::from_image(NativeTensorRole::Image, image) {
            Ok(payload) => payload,
            Err(error) => {
                rollback_publication(context, &published, &effects, kind)?;
                return Err(native_failure(error));
            }
        };
        match context.handle_store().publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        ) {
            Ok(handle) => published.push(handle),
            Err(error) => {
                rollback_publication(context, &published, &effects, kind)?;
                return Err(handle_failure(kind, error));
            }
        }
    }
    if let Err(failure) = check_cancellation(context, kind.class_type()) {
        rollback_publication(context, &published, &effects, kind)?;
        return Err(failure);
    }
    let outputs = published
        .iter()
        .cloned()
        .map(|value| NativeValue::Handle { value })
        .collect();
    let outcome = NativeNodeOutcome::Values {
        outputs,
        ui,
        effects: effects.clone(),
    };
    if let Err(error) = outcome.validate() {
        rollback_publication(context, &published, &effects, kind)?;
        return Err(native_failure(error));
    }
    Ok(outcome)
}

fn publish_image_and_mask(
    context: &NativeNodeContext,
    image: ImageTensor,
    mask: sim_tensor::Tensor,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let image_payload =
        NativeTensorPayload::from_image(NativeTensorRole::Image, image).map_err(native_failure)?;
    let mask_payload =
        NativeTensorPayload::from_tensor(NativeTensorRole::Mask, mask).map_err(native_failure)?;
    let image_handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(image_payload)),
            &context.cancellation,
        )
        .map_err(|error| handle_failure(TransformKind::PadForOutpaint, error))?;
    let mask_handle = match context.handle_store().publish(
        NativeStoredPayload::Tensor(Arc::new(mask_payload)),
        &context.cancellation,
    ) {
        Ok(handle) => handle,
        Err(error) => {
            rollback_handles(context, &[image_handle], TransformKind::PadForOutpaint)?;
            return Err(handle_failure(TransformKind::PadForOutpaint, error));
        }
    };
    if let Err(failure) = check_cancellation(context, TransformKind::PadForOutpaint.class_type()) {
        rollback_handles(
            context,
            &[image_handle, mask_handle],
            TransformKind::PadForOutpaint,
        )?;
        return Err(failure);
    }
    let outcome = values_outcome(vec![
        NativeValue::Handle {
            value: image_handle.clone(),
        },
        NativeValue::Handle {
            value: mask_handle.clone(),
        },
    ]);
    if outcome.is_err() {
        rollback_handles(
            context,
            &[image_handle, mask_handle],
            TransformKind::PadForOutpaint,
        )?;
    }
    outcome
}

fn rollback_handles(
    context: &NativeNodeContext,
    handles: &[NativeOpaqueHandle],
    kind: TransformKind,
) -> Result<(), NativeNodeFailure> {
    for handle in handles.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &sim_types::CancellationToken::default())
            .map_err(|error| NativeNodeFailure {
                kind: NativeNodeFailureKind::Failure,
                code: "image_transform_rollback_failed".to_owned(),
                message: format!(
                    "{} could not revoke a partial output: {error}",
                    kind.class_type()
                ),
                retryable: false,
            })?;
    }
    Ok(())
}

fn rollback_publication(
    context: &NativeNodeContext,
    handles: &[NativeOpaqueHandle],
    effects: &[NativePreparedEffectRequest],
    kind: TransformKind,
) -> Result<(), NativeNodeFailure> {
    let handles_result = rollback_handles(context, handles, kind);
    let effects_result = rollback_effects(context, effects, kind);
    handles_result?;
    effects_result
}

fn rollback_effects(
    context: &NativeNodeContext,
    effects: &[NativePreparedEffectRequest],
    kind: TransformKind,
) -> Result<(), NativeNodeFailure> {
    if effects.is_empty() {
        return Ok(());
    }
    let service = context
        .prepared_effects()
        .map_err(|error| prepared_effect_failure(kind, error))?;
    for effect in effects.iter().rev() {
        service
            .rollback_prepared(effect)
            .map_err(|error| prepared_effect_failure(kind, error))?;
    }
    Ok(())
}

fn values_outcome(outputs: Vec<NativeValue>) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    let outcome = NativeNodeOutcome::Values {
        outputs,
        ui: None,
        effects: Vec::new(),
    };
    outcome.validate().map_err(native_failure)?;
    Ok(outcome)
}

fn validate_inputs(
    kind: TransformKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    for descriptor in input_descriptors(kind).map_err(native_failure)? {
        match inputs.get(&descriptor.name) {
            Some(value) if descriptor.accepted_types.accepts(value) => {}
            Some(_) => {
                return Err(invalid_inputs(format!(
                    "{} has the wrong type",
                    descriptor.name
                )));
            }
            None if !descriptor.required => {}
            None => return Err(invalid_inputs(format!("{} is required", descriptor.name))),
        }
    }
    let expected_names = kind.input_names();
    if inputs.keys().any(|name| !expected_names.contains(name)) {
        return Err(invalid_inputs("image transform received unexpected inputs"));
    }
    validate_input_values(kind, inputs)?;
    Ok(())
}

fn validate_input_values(
    kind: TransformKind,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(), NativeNodeFailure> {
    match kind {
        TransformKind::CenterCrop => {
            bounded_u64(inputs, "width", 1, 8_192)?;
            bounded_u64(inputs, "height", 1, 8_192)?;
        }
        TransformKind::CropByBBoxes => {
            bounded_u64(inputs, "output_width", 64, 4_096)?;
            bounded_u64(inputs, "output_height", 64, 4_096)?;
            bounded_u64(inputs, "padding", 0, 1_024)?;
            one_of(inputs, "keep_aspect", &["stretch", "pad"])?;
        }
        TransformKind::Crop => {
            bounded_u64(inputs, "width", 1, MAX_RESOLUTION)?;
            bounded_u64(inputs, "height", 1, MAX_RESOLUTION)?;
            bounded_u64(inputs, "x", 0, MAX_RESOLUTION)?;
            bounded_u64(inputs, "y", 0, MAX_RESOLUTION)?;
        }
        TransformKind::CropV2 => {}
        TransformKind::Flip => {
            one_of(
                inputs,
                "flip_method",
                &["x-axis: vertically", "y-axis: horizontally"],
            )?;
        }
        TransformKind::PadForOutpaint => {
            for name in ["left", "top", "right", "bottom", "feathering"] {
                bounded_u64(inputs, name, 0, MAX_RESOLUTION)?;
            }
        }
        TransformKind::Rotate => {
            one_of(
                inputs,
                "rotation",
                &["none", "90 degrees", "180 degrees", "270 degrees"],
            )?;
        }
        TransformKind::Stitch => {
            one_of(inputs, "direction", &["right", "down", "left", "up"])?;
            boolean(inputs, "match_image_size")?;
            bounded_u64(inputs, "spacing_width", 0, 1_024)?;
            one_of(
                inputs,
                "spacing_color",
                &["white", "black", "red", "green", "blue"],
            )?;
        }
        TransformKind::RandomCrop => {
            bounded_u64(inputs, "width", 1, 8_192)?;
            bounded_u64(inputs, "height", 1, 8_192)?;
            integer(inputs, "seed")?;
        }
        TransformKind::ResizeAndPad => {
            bounded_u64(inputs, "target_width", 1, MAX_RESOLUTION)?;
            bounded_u64(inputs, "target_height", 1, MAX_RESOLUTION)?;
            one_of(inputs, "padding_color", &["white", "black"])?;
            one_of(
                inputs,
                "interpolation",
                &["area", "bicubic", "nearest-exact", "bilinear", "lanczos"],
            )?;
        }
    }
    Ok(())
}

fn one_of<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
    values: &[&str],
) -> Result<&'a str, NativeNodeFailure> {
    let value = string(inputs, name)?;
    if values.contains(&value) {
        Ok(value)
    } else {
        Err(invalid_inputs(format!("{name} is not supported")))
    }
}

fn handle<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Handle { value }) => Ok(value),
        _ => Err(invalid_inputs(format!("{name} must be a handle"))),
    }
}

fn integer(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Ok(*value),
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value)
            .map_err(|_| invalid_inputs(format!("{name} must be non-negative"))),
        _ => Err(invalid_inputs(format!("{name} must be an integer"))),
    }
}

fn bounded_u64(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = integer(inputs, name)?;
    if !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn string<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
) -> Result<&'a str, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => Ok(value),
        _ => Err(invalid_inputs(format!("{name} must be a string"))),
    }
}

fn boolean(inputs: &BTreeMap<String, NativeValue>, name: &str) -> Result<bool, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Boolean(value),
        }) => Ok(*value),
        _ => Err(invalid_inputs(format!("{name} must be a boolean"))),
    }
}

fn truncating_u64(value: f64, name: &str) -> Result<u64, NativeNodeFailure> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        return Err(invalid_inputs(format!(
            "{name} is outside the supported range"
        )));
    }
    Ok(value as u64)
}

fn integral_u64(value: f64, name: &str) -> Result<u64, NativeNodeFailure> {
    if value.fract() != 0.0 {
        return Err(invalid_inputs(format!("{name} must be an integer")));
    }
    truncating_u64(value, name)
}

fn truncating_i64(value: f64, name: &str) -> Result<i64, NativeNodeFailure> {
    if !value.is_finite() {
        return Err(invalid_inputs(format!(
            "{name} is outside the supported range"
        )));
    }
    Ok(value as i64)
}

fn value_at(
    values: &[f32],
    batch: u64,
    y: u64,
    x: u64,
    channel: u64,
    height: u64,
    width: u64,
    channels: u64,
) -> Result<f32, NativeNodeFailure> {
    values
        .get(image_offset(batch, y, x, channel, height, width, channels)?)
        .copied()
        .ok_or_else(|| native_failure("IMAGE index exceeded storage"))
}

fn image_offset(
    batch: u64,
    y: u64,
    x: u64,
    channel: u64,
    height: u64,
    width: u64,
    channels: u64,
) -> Result<usize, NativeNodeFailure> {
    batch
        .checked_mul(height)
        .and_then(|value| value.checked_add(y))
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_add(x))
        .and_then(|value| value.checked_mul(channels))
        .and_then(|value| value.checked_add(channel))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| native_failure("IMAGE index overflowed"))
}

fn element_count(
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
) -> Result<usize, NativeNodeFailure> {
    batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(channels))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| native_failure("IMAGE element count overflowed"))
}

fn filled_values(element_count: usize, value: f32) -> Result<Vec<f32>, NativeNodeFailure> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count)
        .map_err(native_failure)?;
    values.resize(element_count, value);
    Ok(values)
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

fn handle_failure(kind: TransformKind, error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure(kind.class_type())
    } else {
        NativeNodeFailure {
            kind: NativeNodeFailureKind::Failure,
            code: "invalid_image_transform_handle".to_owned(),
            message: format!(
                "{} could not access native payload storage: {error}",
                kind.class_type()
            ),
            retryable: false,
        }
    }
}

fn image_preview_failure(kind: TransformKind, error: NativeImagePreviewError) -> NativeNodeFailure {
    if matches!(
        error,
        NativeImagePreviewError::Effect(NativeEffectServiceError::Cancelled)
    ) {
        return interrupted_failure(kind.class_type());
    }
    NativeNodeFailure {
        kind: NativeNodeFailureKind::Failure,
        code: "image_transform_preview_failed".to_owned(),
        message: format!(
            "{} could not prepare its preview: {error}",
            kind.class_type()
        ),
        retryable: false,
    }
}

fn prepared_effect_failure(
    kind: TransformKind,
    error: NativeEffectServiceError,
) -> NativeNodeFailure {
    if error == NativeEffectServiceError::Cancelled {
        return interrupted_failure(kind.class_type());
    }
    NativeNodeFailure {
        kind: NativeNodeFailureKind::Failure,
        code: "image_transform_preview_failed".to_owned(),
        message: format!(
            "{} could not roll back its preview: {error}",
            kind.class_type()
        ),
        retryable: false,
    }
}

fn compute_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        kind: NativeNodeFailureKind::Failure,
        code: "image_transform_compute_unavailable".to_owned(),
        message: error.to_string(),
        retryable: false,
    }
}

fn tensor_failure(error: TensorError) -> NativeNodeFailure {
    if error == TensorError::Cancelled {
        interrupted_failure("image transform")
    } else {
        native_failure(error)
    }
}

fn rng_failure(error: RngError) -> NativeNodeFailure {
    if error == RngError::Cancelled {
        interrupted_failure("image transform")
    } else {
        native_failure(error)
    }
}

fn native_failure(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        kind: NativeNodeFailureKind::Failure,
        code: "image_transform_failed".to_owned(),
        message: error.to_string(),
        retryable: false,
    }
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        kind: NativeNodeFailureKind::Failure,
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        retryable: false,
    }
}

fn interrupted_failure(class_type: &str) -> NativeNodeFailure {
    NativeNodeFailure {
        kind: NativeNodeFailureKind::Interrupted,
        code: "execution_interrupted".to_owned(),
        message: format!("{class_type} execution was cancelled"),
        retryable: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};
    use serde_json::Value;

    fn fixture() -> Result<Value, serde_json::Error> {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/image-transform-comfy-node-0047/fixture.json"
        )))
    }

    fn image(
        batch: u64,
        height: u64,
        width: u64,
        channels: u64,
        values: &[f32],
    ) -> Result<ImageTensor, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let cancellation = sim_types::CancellationToken::default();
        let scratch = authority.authorize_workspace(16 * 1024 * 1024)?;
        let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        Ok(ImageTensor::from_f32(
            &backend, &execution, batch, height, width, channels, values,
        )?)
    }

    #[test]
    fn descriptors_match_exact_assigned_rows() -> Result<(), Box<dyn std::error::Error>> {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 10);
        let actual = bindings
            .iter()
            .map(|binding| binding.feature_id())
            .collect::<Vec<_>>();
        let expected = ALL_KINDS
            .iter()
            .map(|kind| kind.feature_id())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        for binding in &bindings {
            let NativeNodeBinding::Executable { descriptor, .. } = binding else {
                return Err("image transform binding was not executable".into());
            };
            descriptor.validate_exact_schema_v2()?;
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(
                descriptor.effect,
                if descriptor.class_type == "ImageCropV2" {
                    NativeEffectClass::WritesArtifact
                } else {
                    NativeEffectClass::Pure
                }
            );
            assert!(!descriptor.output_node);
            assert!(descriptor.inputs.iter().all(|input| !input.lazy));
            assert!(descriptor.outputs.iter().all(|output| !output.is_list));
        }
        let fixture = fixture()?;
        assert_eq!(
            fixture["stable_task_id"],
            "comfy-parity-native-nodes-image-transform-comfy-node-0047"
        );
        let nodes = fixture["nodes"]
            .as_array()
            .ok_or("fixture nodes were not an array")?;
        assert_eq!(nodes.len(), 10);
        let expected_hashes = [
            ("COMFY-NODE-0047", "f5258e4d133be01f4db12430bc814bdc25d7b937fede545fa8cdd1852740a190"),
            ("COMFY-NODE-0125", "08f10e4d60924fdc5002b2ea82d180c8dd71a2fcd4cb8b54df611dbdfce3366d"),
            ("COMFY-NODE-0247", "6a2e83dfcd98423d8dc1476248f77284f22dd087638c3a815840cd22042392c7"),
            ("COMFY-NODE-0248", "7bff80a167cec29a7737dc1ef26114d0821c49ef4130cc531c6612a26c3a9ec9"),
            ("COMFY-NODE-0250", "2950156b24562cc8da1f756574b9fe191c717d8edc336b5da0656af48f47dcf0"),
            ("COMFY-NODE-0258", "ed9bce1af997ce50cdea6eb3de0646a734819fbe2e07ca62e2e70db226d0f17b"),
            ("COMFY-NODE-0261", "7fed131da61d51f061bb20166431244379e86922b5daf75c3a6dbcb6a6cfd4bd"),
            ("COMFY-NODE-0267", "124c054e8bed03d8702e4336a20bda8f0c6cfe628f6c83b4d2a9788c2cd9409a"),
            ("COMFY-NODE-0504", "b27066e5da7a4aa80840fa172d2410d5609e92f02bd5244494c467da0ee8a2a6"),
            ("COMFY-NODE-0540", "e4472818dee17ff8bd42e25a996169499dac94f0b0fcee0d81277dada677ffbf"),
        ];
        for (node, (feature_id, definition_hash)) in nodes.iter().zip(expected_hashes) {
            assert_eq!(node["feature_id"], feature_id);
            assert_eq!(node["catalog_definition_sha256"], definition_hash);
        }
        assert_eq!(
            bindings[0].presentation().display_name,
            "Crop Image (Center)"
        );
        assert_eq!(
            bindings[8].presentation().display_name,
            "Crop Image (Random)"
        );
        let encoded = serde_json::to_vec(&fixture)?;
        assert_eq!(serde_json::from_slice::<Value>(&encoded)?, fixture);
        Ok(())
    }

    #[test]
    fn source_crop_quantization_and_numpy_coordinates_match_oracles()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let cancellation = sim_types::CancellationToken::default();
        let scratch = authority.authorize_workspace(16 * 1024 * 1024)?;
        let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let source = image(1, 2, 3, 1, &[0.0, 0.1, 0.5, 0.9, 1.0, 1.2])?;
        let cropped = source.source_compatible_u8_crop(1, 0, 2, 2, &backend, &execution)?;
        assert_eq!(
            cropped.as_f32_slice()?,
            &[25.0 / 255.0, 127.0 / 255.0, 1.0, 50.0 / 255.0]
        );
        let mut random = NumpyRandomState::from_seed(42);
        assert_eq!(random.randint(0, 8, &cancellation)?, 6);
        assert_eq!(random.randint(0, 5, &cancellation)?, 3);
        Ok(())
    }

    #[test]
    fn rotate_flip_pad_and_stitch_preserve_source_ordering()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let cancellation = sim_types::CancellationToken::default();
        let scratch = authority.authorize_workspace(16 * 1024 * 1024)?;
        let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let source = image(1, 2, 3, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let rotated = rotate_image(&source, 1, &backend, &execution)?;
        assert_eq!(rotated.dimensions()?, (1, 3, 2, 1));
        assert_eq!(rotated.as_f32_slice()?, &[3.0, 6.0, 2.0, 5.0, 1.0, 4.0]);
        let flipped = remap_image(&source, &backend, &execution, |y, x, _, width| {
            (y, width - 1 - x)
        })?;
        assert_eq!(flipped.as_f32_slice()?, &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
        let (padded, mask) = pad_for_outpaint(&source, 1, 1, 0, 0, 1, &backend, &execution)?;
        assert_eq!(padded.dimensions()?, (1, 3, 4, 1));
        assert_eq!(mask.descriptor().shape(), &[1, 3, 4]);
        let first = image(1, 1, 1, 3, &[0.1, 0.2, 0.3])?;
        let second = image(1, 1, 1, 4, &[0.4, 0.5, 0.6, 0.7])?;
        let stitched = stitch_images(
            &first,
            &second,
            "right",
            false,
            1,
            [1.0, 0.0, 0.0],
            &backend,
            &execution,
        )?;
        assert_eq!(stitched.dimensions()?, (1, 1, 4, 4));
        assert_eq!(
            stitched.as_f32_slice()?,
            &[
                0.1, 0.2, 0.3, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.4, 0.5, 0.6, 0.7,
            ]
        );
        let bounding_box = NativeBoundingBox::checked(-2.0, -1.0, 3.0, 4.0, None, None)?;
        assert_eq!(bounding_union(&[bounding_box])?, (-2, -1, 1, 3));
        assert_eq!(fit_dimensions(2, 4, 3, 5, true)?, (2, 5));
        assert!(fit_dimensions(16_384, 1, 1, 1, false).is_err());
        assert!(integral_u64(1.5, "crop x").is_err());
        Ok(())
    }

    #[test]
    fn cancellation_and_input_boundaries_fail_closed() -> Result<(), Box<dyn std::error::Error>> {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "width".to_owned(),
            NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(0),
            },
        );
        assert!(bounded_u64(&inputs, "width", 1, 8_192).is_err());
        let cancellation = sim_types::CancellationToken::default();
        cancellation.cancel();
        let mut random = NumpyRandomState::from_seed(1);
        assert!(random.randint(0, 4, &cancellation).is_err());
        Ok(())
    }
}
