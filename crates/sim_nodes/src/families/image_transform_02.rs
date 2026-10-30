use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleStoreError, NativeHandleType,
    NativeInputDescriptor, NativeNode, NativeNodeBinding, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativePortCardinality, NativePrimitive, NativePrimitiveType,
    NativeStoredPayload, NativeStructuredValue, NativeTypeUnion, NativeValue, NativeValueType,
    built_in_source_schema,
};
use sim_tensor::{
    ImageTensor, NativeTensorPayload, NativeTensorRole, ResizeCrop, ResizeMode, ViewAccess,
};
use futures::future::BoxFuture;
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "ResizeImageMaskNode",
    "ResizeImagesByLongerEdge",
    "ResizeImagesByShorterEdge",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const RESIZE_FEATURE_ID: &str = "COMFY-NODE-0541";
const RESIZE_CLASS_TYPE: &str = "ResizeImageMaskNode";
const LONGER_FEATURE_ID: &str = "COMFY-NODE-0542";
const LONGER_CLASS_TYPE: &str = "ResizeImagesByLongerEdge";
const SHORTER_FEATURE_ID: &str = "COMFY-NODE-0543";
const SHORTER_CLASS_TYPE: &str = "ResizeImagesByShorterEdge";
const POST_PROCESSING_SOURCE_VERSION: &str = "source-96ec39e8-v1";
const DATASET_SOURCE_VERSION: &str = "source-3b27465f-v1";
const MAX_RESOLUTION: u64 = 16_384;
const MAX_DEPRECATED_RESOLUTION: u64 = 8_192;

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    Ok(vec![
        resize_image_mask_binding()?,
        deprecated_edge_binding(Edge::Longer)?,
        deprecated_edge_binding(Edge::Shorter)?,
    ])
}

fn resize_image_mask_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    let source_schema = built_in_source_schema(RESIZE_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &[
                "input".to_owned(),
                "resize_type".to_owned(),
                "scale_method".to_owned(),
            ],
            &[],
            &["resized".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: RESIZE_FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: RESIZE_CLASS_TYPE.to_owned(),
            implementation_version: POST_PROCESSING_SOURCE_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![
                NativeInputDescriptor {
                    name: "input".to_owned(),
                    accepted_types: NativeTypeUnion::new([NativeValueType::Any])?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: false,
                },
                NativeInputDescriptor {
                    name: "resize_type".to_owned(),
                    accepted_types: NativeTypeUnion::new([
                        NativeValueType::NamedPreservedUnknown(
                            "COMFY_DYNAMICCOMBO_V3".to_owned(),
                        ),
                    ])?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: true,
                },
                NativeInputDescriptor {
                    name: "scale_method".to_owned(),
                    accepted_types: NativeTypeUnion::new([NativeValueType::Primitive(
                        NativePrimitiveType::String,
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
                name: "resized".to_owned(),
                produced_type: NativeValueType::Any,
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "Resize Image/Mask".to_owned(),
            category: "image/transform".to_owned(),
            description: "Resize an image or mask using various scaling methods.".to_owned(),
            output_names: vec!["resized".to_owned()],
            search_aliases: [
                "resize",
                "resize image",
                "resize mask",
                "scale",
                "scale image",
                "scale mask",
                "image resize",
                "change size",
                "dimensions",
                "shrink",
                "enlarge",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(ResizeImageMask),
    })
}

fn deprecated_edge_binding(edge: Edge) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let image_type = image_type()?;
    let source_schema = built_in_source_schema(edge.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["images".to_owned(), edge.input_name().to_owned()],
            &[],
            &["images".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: edge.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: edge.class_type().to_owned(),
            implementation_version: DATASET_SOURCE_VERSION.to_owned(),
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
                    name: edge.input_name().to_owned(),
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
                name: "images".to_owned(),
                produced_type: NativeValueType::Handle(image_type),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: edge.display_name().to_owned(),
            category: "image/transform".to_owned(),
            description: edge.description().to_owned(),
            output_names: vec!["images".to_owned()],
            search_aliases: Vec::new(),
            is_deprecated: true,
            is_experimental: true,
        },
        node: Arc::new(DeprecatedEdgeResize { edge }),
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

#[derive(Clone, Copy, Debug)]
enum Edge {
    Longer,
    Shorter,
}

impl Edge {
    const fn feature_id(self) -> &'static str {
        match self {
            Self::Longer => LONGER_FEATURE_ID,
            Self::Shorter => SHORTER_FEATURE_ID,
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::Longer => LONGER_CLASS_TYPE,
            Self::Shorter => SHORTER_CLASS_TYPE,
        }
    }

    const fn input_name(self) -> &'static str {
        match self {
            Self::Longer => "longer_edge",
            Self::Shorter => "shorter_edge",
        }
    }

    const fn display_name(self) -> &'static str {
        match self {
            Self::Longer => "Resize Images by Longer Edge (DEPRECATED)",
            Self::Shorter => "Resize Images by Shorter Edge (DEPRECATED)",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Longer => {
                "Resize images so that the longer edge matches the specified dimension while preserving aspect ratio."
            }
            Self::Shorter => {
                "Resize images so that the shorter edge matches the specified dimension while preserving aspect ratio."
            }
        }
    }
}

#[derive(Debug)]
struct ResizeImageMask;

impl NativeNode for ResizeImageMask {
    fn class_type(&self) -> &str {
        RESIZE_CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        POST_PROCESSING_SOURCE_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<std::collections::BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context, RESIZE_CLASS_TYPE)?;
        resize_inputs(available_inputs)?;
        Ok(std::collections::BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        resize_inputs(inputs)?;
        Ok(format!(
            "{POST_PROCESSING_SOURCE_VERSION}-typed-resize-input-identity"
        ))
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, RESIZE_CLASS_TYPE)?;
        resize_inputs(inputs)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, RESIZE_CLASS_TYPE)?;
            let parsed = resize_inputs(&inputs)?;
            let resolved = resolve_tensor(&context, parsed.input, RESIZE_CLASS_TYPE)?;
            let role = resolved_tensor_role(&resolved, parsed.input)?;
            let input_image = image_from_payload(&resolved, role)?;
            let dimensions = input_image
                .dimensions()
                .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
            let plan = resize_plan(&context, &parsed, dimensions)?;
            if matches!(plan, ResizePlan::Identity) {
                return values_outcome(parsed.input.clone());
            }
            let compute = context.compute_session().map_err(compute_failure)?;
            let execution = compute
                .execution_context(&context)
                .map_err(compute_failure)?;
            let output = execute_resize_plan(
                input_image,
                plan,
                parsed.mode,
                compute.backend(),
                &execution,
            )?;
            check_cancellation(&context, RESIZE_CLASS_TYPE)?;
            publish_image(&context, output, role, RESIZE_CLASS_TYPE)
        })
    }
}

#[derive(Debug)]
struct DeprecatedEdgeResize {
    edge: Edge,
}

impl NativeNode for DeprecatedEdgeResize {
    fn class_type(&self) -> &str {
        self.edge.class_type()
    }

    fn implementation_version(&self) -> &str {
        DATASET_SOURCE_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<std::collections::BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context, self.edge.class_type())?;
        edge_inputs(available_inputs, self.edge)?;
        Ok(std::collections::BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        edge_inputs(inputs, self.edge)?;
        Ok(format!(
            "{DATASET_SOURCE_VERSION}-{}-pillow-lanczos",
            self.edge.class_type()
        ))
    }

    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check_cancellation(context, self.edge.class_type())?;
        edge_inputs(inputs, self.edge)?;
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context, self.edge.class_type())?;
            let (handle, requested) = edge_inputs(&inputs, self.edge)?;
            let resolved = resolve_tensor(&context, handle, self.edge.class_type())?;
            let role = resolved_tensor_role(&resolved, handle)?;
            if role != NativeTensorRole::Image {
                return Err(invalid_inputs(format!(
                    "{} images input must resolve to an IMAGE payload",
                    self.edge.class_type()
                )));
            }
            let image = image_from_payload(&resolved, role)?;
            let (batch, height, width, _) = image
                .dimensions()
                .map_err(|error| resize_failure(self.edge.class_type(), error))?;
            if matches!(self.edge, Edge::Shorter) && batch != 1 {
                return Err(invalid_inputs(
                    "ResizeImagesByShorterEdge accepts exactly one image per mapped invocation",
                ));
            }
            let (output_width, output_height) = deprecated_edge_dimensions(
                self.edge,
                width,
                height,
                requested,
            )?;
            let compute = context.compute_session().map_err(compute_failure)?;
            let execution = compute
                .execution_context(&context)
                .map_err(compute_failure)?;
            let output = image
                .resize(
                    output_width,
                    output_height,
                    ResizeMode::Lanczos,
                    ResizeCrop::Disabled,
                    compute.backend(),
                    &execution,
                )
                .map_err(|error| resize_failure(self.edge.class_type(), error))?;
            check_cancellation(&context, self.edge.class_type())?;
            publish_image(
                &context,
                output,
                NativeTensorRole::Image,
                self.edge.class_type(),
            )
        })
    }
}

struct ResizeInputs<'a> {
    input: &'a NativeOpaqueHandle,
    fields: BTreeMap<String, NativeValue>,
    mode: ResizeMode,
}

fn resize_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<ResizeInputs<'_>, NativeNodeFailure> {
    if inputs.len() != 3 {
        return Err(invalid_inputs(
            "ResizeImageMaskNode requires exactly input, resize_type, and scale_method",
        ));
    }
    let input = required_image_or_mask_handle(inputs.get("input"), "input")?;
    let fields = structured_fields(inputs.get("resize_type"))?;
    let mode = resize_mode(inputs.get("scale_method"))?;
    validate_resize_fields(&fields)?;
    Ok(ResizeInputs {
        input,
        fields,
        mode,
    })
}

fn structured_fields(
    value: Option<&NativeValue>,
) -> Result<BTreeMap<String, NativeValue>, NativeNodeFailure> {
    let value = value.ok_or_else(|| invalid_inputs("resize_type is required"))?;
    if let Some(structured) = NativeStructuredValue::from_native_value(value)
        .map_err(|error| invalid_inputs(error.to_string()))?
    {
        if structured.type_name() != "COMFY_DYNAMICCOMBO_V3" {
            return Err(invalid_inputs(
                "resize_type must use the COMFY_DYNAMICCOMBO_V3 structured type",
            ));
        }
        return Ok(structured.fields().clone());
    }
    let NativeValue::PreservedUnknown { type_name, value } = value else {
        return Err(invalid_inputs("resize_type must be a structured value"));
    };
    if type_name != "COMFY_DYNAMICCOMBO_V3" {
        return Err(invalid_inputs(
            "resize_type must use the COMFY_DYNAMICCOMBO_V3 structured type",
        ));
    }
    let Value::Object(fields) = value else {
        return Err(invalid_inputs("resize_type must contain an object"));
    };
    fields
        .iter()
        .map(|(name, value)| Ok((name.clone(), json_native_value(value)?)))
        .collect()
}

fn json_native_value(value: &Value) -> Result<NativeValue, NativeNodeFailure> {
    let value = match value {
        Value::Null => NativePrimitive::Null,
        Value::Bool(value) => NativePrimitive::Boolean(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                NativePrimitive::Integer(value)
            } else if let Some(value) = value.as_u64() {
                NativePrimitive::UnsignedInteger(value)
            } else {
                NativePrimitive::Number(
                    value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| invalid_inputs("resize_type contains an invalid number"))?,
                )
            }
        }
        Value::String(value) => NativePrimitive::String(value.clone()),
        Value::Array(_) | Value::Object(_) => {
            return Err(invalid_inputs(
                "resize_type contains an unsupported nested literal",
            ));
        }
    };
    Ok(NativeValue::Primitive { value })
}

fn validate_resize_fields(fields: &BTreeMap<String, NativeValue>) -> Result<(), NativeNodeFailure> {
    let selector = selector(fields)?;
    let allowed: &[&str] = match selector {
        "scale by multiplier" => &["resize_type", "multiplier"],
        "scale dimensions" => &["resize_type", "width", "height", "crop"],
        "scale longer dimension" => &["resize_type", "longer_size"],
        "scale shorter dimension" => &["resize_type", "shorter_size"],
        "scale width" => &["resize_type", "width"],
        "scale height" => &["resize_type", "height"],
        "scale total pixels" => &["resize_type", "megapixels"],
        "match size" => &["resize_type", "match", "crop"],
        "scale to multiple" => &["resize_type", "multiple"],
        _ => {
            return Err(invalid_inputs(format!(
                "Unsupported resize type: {selector}"
            )));
        }
    };
    if fields.len() != allowed.len() || !allowed.iter().all(|name| fields.contains_key(*name)) {
        return Err(invalid_inputs(format!(
            "resize_type {selector:?} has missing or unexpected fields"
        )));
    }
    match selector {
        "scale by multiplier" => {
            bounded_number(fields.get("multiplier"), "multiplier", 0.01, 8.0)?;
        }
        "scale dimensions" => {
            bounded_unsigned(fields.get("width"), "width", 0, MAX_RESOLUTION)?;
            bounded_unsigned(fields.get("height"), "height", 0, MAX_RESOLUTION)?;
            crop_mode(fields.get("crop"))?;
        }
        "scale longer dimension" => {
            bounded_unsigned(
                fields.get("longer_size"),
                "longer_size",
                0,
                MAX_RESOLUTION,
            )?;
        }
        "scale shorter dimension" => {
            bounded_unsigned(
                fields.get("shorter_size"),
                "shorter_size",
                0,
                MAX_RESOLUTION,
            )?;
        }
        "scale width" => {
            bounded_unsigned(fields.get("width"), "width", 0, MAX_RESOLUTION)?;
        }
        "scale height" => {
            bounded_unsigned(fields.get("height"), "height", 0, MAX_RESOLUTION)?;
        }
        "scale total pixels" => {
            bounded_number(fields.get("megapixels"), "megapixels", 0.01, 16.0)?;
        }
        "match size" => {
            required_image_or_mask_handle(fields.get("match"), "match")?;
            crop_mode(fields.get("crop"))?;
        }
        "scale to multiple" => {
            bounded_unsigned(fields.get("multiple"), "multiple", 1, MAX_RESOLUTION)?;
        }
        _ => return Err(invalid_inputs("resize_type selector is unsupported")),
    }
    Ok(())
}

fn selector(fields: &BTreeMap<String, NativeValue>) -> Result<&str, NativeNodeFailure> {
    match fields.get("resize_type") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => Ok(value),
        _ => Err(invalid_inputs("resize_type selector must be a STRING")),
    }
}

fn resize_mode(value: Option<&NativeValue>) -> Result<ResizeMode, NativeNodeFailure> {
    match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => match value.as_str() {
            "nearest-exact" => Ok(ResizeMode::NearestExact),
            "bilinear" => Ok(ResizeMode::Bilinear),
            "area" => Ok(ResizeMode::Area),
            "bicubic" => Ok(ResizeMode::Bicubic),
            "lanczos" => Ok(ResizeMode::Lanczos),
            _ => Err(invalid_inputs(format!(
                "unsupported scale_method {value:?}"
            ))),
        },
        _ => Err(invalid_inputs("scale_method must be a STRING")),
    }
}

fn crop_mode(value: Option<&NativeValue>) -> Result<ResizeCrop, NativeNodeFailure> {
    match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => match value.as_str() {
            "disabled" => Ok(ResizeCrop::Disabled),
            "center" => Ok(ResizeCrop::Center),
            _ => Err(invalid_inputs(format!("unsupported crop mode {value:?}"))),
        },
        _ => Err(invalid_inputs("crop must be a STRING")),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResizePlan {
    Identity,
    Resize {
        width: u64,
        height: u64,
        crop: ResizeCrop,
    },
    ResizeThenCrop {
        scaled_width: u64,
        scaled_height: u64,
        width: u64,
        height: u64,
    },
}

fn resize_plan(
    context: &NativeNodeContext,
    inputs: &ResizeInputs<'_>,
    dimensions: (u64, u64, u64, u64),
) -> Result<ResizePlan, NativeNodeFailure> {
    let (_, height, width, _) = dimensions;
    let selector = selector(&inputs.fields)?;
    let plan = match selector {
        "scale by multiplier" => {
            let multiplier = bounded_number(
                inputs.fields.get("multiplier"),
                "multiplier",
                0.01,
                8.0,
            )?;
            ResizePlan::Resize {
                width: round_dimension(width as f64 * multiplier, "scaled width")?,
                height: round_dimension(height as f64 * multiplier, "scaled height")?,
                crop: ResizeCrop::Disabled,
            }
        }
        "scale dimensions" => {
            let requested_width = bounded_unsigned(
                inputs.fields.get("width"),
                "width",
                0,
                MAX_RESOLUTION,
            )?;
            let requested_height = bounded_unsigned(
                inputs.fields.get("height"),
                "height",
                0,
                MAX_RESOLUTION,
            )?;
            if requested_width == 0 && requested_height == 0 {
                ResizePlan::Identity
            } else {
                let (output_width, output_height) = proportional_dimensions(
                    width,
                    height,
                    requested_width,
                    requested_height,
                )?;
                ResizePlan::Resize {
                    width: output_width,
                    height: output_height,
                    crop: crop_mode(inputs.fields.get("crop"))?,
                }
            }
        }
        "scale longer dimension" => {
            let size = bounded_unsigned(
                inputs.fields.get("longer_size"),
                "longer_size",
                0,
                MAX_RESOLUTION,
            )?;
            let (width, height) = proportional_edge_dimensions(Edge::Longer, width, height, size)?;
            ResizePlan::Resize {
                width,
                height,
                crop: ResizeCrop::Disabled,
            }
        }
        "scale shorter dimension" => {
            let size = bounded_unsigned(
                inputs.fields.get("shorter_size"),
                "shorter_size",
                0,
                MAX_RESOLUTION,
            )?;
            let (width, height) = proportional_edge_dimensions(Edge::Shorter, width, height, size)?;
            ResizePlan::Resize {
                width,
                height,
                crop: ResizeCrop::Disabled,
            }
        }
        "scale width" => {
            let requested_width = bounded_unsigned(
                inputs.fields.get("width"),
                "width",
                0,
                MAX_RESOLUTION,
            )?;
            if requested_width == 0 {
                ResizePlan::Identity
            } else {
                let (output_width, output_height) =
                    proportional_dimensions(width, height, requested_width, 0)?;
                ResizePlan::Resize {
                    width: output_width,
                    height: output_height,
                    crop: ResizeCrop::Disabled,
                }
            }
        }
        "scale height" => {
            let requested_height = bounded_unsigned(
                inputs.fields.get("height"),
                "height",
                0,
                MAX_RESOLUTION,
            )?;
            if requested_height == 0 {
                ResizePlan::Identity
            } else {
                let (output_width, output_height) =
                    proportional_dimensions(width, height, 0, requested_height)?;
                ResizePlan::Resize {
                    width: output_width,
                    height: output_height,
                    crop: ResizeCrop::Disabled,
                }
            }
        }
        "scale total pixels" => {
            let megapixels = bounded_number(
                inputs.fields.get("megapixels"),
                "megapixels",
                0.01,
                16.0,
            )?;
            let total = (megapixels * 1024.0 * 1024.0).trunc();
            let scale = (total / (width as f64 * height as f64)).sqrt();
            ResizePlan::Resize {
                width: round_dimension(width as f64 * scale, "megapixel width")?,
                height: round_dimension(height as f64 * scale, "megapixel height")?,
                crop: ResizeCrop::Disabled,
            }
        }
        "match size" => {
            let match_handle = required_image_or_mask_handle(inputs.fields.get("match"), "match")?;
            let match_payload = resolve_tensor(context, match_handle, RESIZE_CLASS_TYPE)?;
            let match_role = resolved_tensor_role(&match_payload, match_handle)?;
            let match_image = image_from_payload(&match_payload, match_role)?;
            let (_, height, width, _) = match_image
                .dimensions()
                .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
            ResizePlan::Resize {
                width,
                height,
                crop: crop_mode(inputs.fields.get("crop"))?,
            }
        }
        "scale to multiple" => {
            let multiple = bounded_unsigned(
                inputs.fields.get("multiple"),
                "multiple",
                1,
                MAX_RESOLUTION,
            )?;
            multiple_plan(width, height, multiple)?
        }
        _ => {
            return Err(invalid_inputs(format!(
                "Unsupported resize type: {selector}"
            )));
        }
    };
    Ok(plan)
}

fn proportional_dimensions(
    input_width: u64,
    input_height: u64,
    requested_width: u64,
    requested_height: u64,
) -> Result<(u64, u64), NativeNodeFailure> {
    if input_width == 0 || input_height == 0 {
        return Err(invalid_inputs("input spatial dimensions must be nonzero"));
    }
    let width = if requested_width == 0 {
        round_dimension(
            input_width as f64 * requested_height as f64 / input_height as f64,
            "proportional width",
        )?
        .max(1)
    } else {
        requested_width
    };
    let height = if requested_height == 0 {
        round_dimension(
            input_height as f64 * width as f64 / input_width as f64,
            "proportional height",
        )?
        .max(1)
    } else {
        requested_height
    };
    Ok((width, height))
}

fn proportional_edge_dimensions(
    edge: Edge,
    width: u64,
    height: u64,
    size: u64,
) -> Result<(u64, u64), NativeNodeFailure> {
    if width == 0 || height == 0 {
        return Err(invalid_inputs("input spatial dimensions must be nonzero"));
    }
    let resize_width = match edge {
        Edge::Longer => width > height,
        Edge::Shorter => width < height,
    };
    let resize_height = match edge {
        Edge::Longer => height > width,
        Edge::Shorter => height < width,
    };
    if resize_width {
        Ok((
            size,
            round_dimension(height as f64 / width as f64 * size as f64, "scaled height")?,
        ))
    } else if resize_height {
        Ok((
            round_dimension(width as f64 / height as f64 * size as f64, "scaled width")?,
            size,
        ))
    } else {
        Ok((size, size))
    }
}

fn deprecated_edge_dimensions(
    edge: Edge,
    width: u64,
    height: u64,
    size: u64,
) -> Result<(u64, u64), NativeNodeFailure> {
    if width == 0 || height == 0 {
        return Err(invalid_inputs("input spatial dimensions must be nonzero"));
    }
    let resize_width = match edge {
        Edge::Longer => width > height,
        Edge::Shorter => width < height,
    };
    if resize_width {
        Ok((
            size,
            floor_dimension(height as f64 * (size as f64 / width as f64), "scaled height")?,
        ))
    } else {
        Ok((
            floor_dimension(width as f64 * (size as f64 / height as f64), "scaled width")?,
            size,
        ))
    }
}

fn multiple_plan(
    width: u64,
    height: u64,
    multiple: u64,
) -> Result<ResizePlan, NativeNodeFailure> {
    if multiple <= 1 {
        return Ok(ResizePlan::Identity);
    }
    let target_width = width / multiple * multiple;
    let target_height = height / multiple * multiple;
    if target_width == 0
        || target_height == 0
        || (target_width == width && target_height == height)
    {
        return Ok(ResizePlan::Identity);
    }
    let width_scale = target_width as f64 / width as f64;
    let height_scale = target_height as f64 / height as f64;
    let (scaled_width, scaled_height) = if width_scale >= height_scale {
        (
            target_width,
            ceil_dimension(height as f64 * width_scale, "multiple scaled height")?
                .max(target_height),
        )
    } else {
        (
            ceil_dimension(width as f64 * height_scale, "multiple scaled width")?
                .max(target_width),
            target_height,
        )
    };
    Ok(ResizePlan::ResizeThenCrop {
        scaled_width,
        scaled_height,
        width: target_width,
        height: target_height,
    })
}

fn execute_resize_plan(
    image: ImageTensor,
    plan: ResizePlan,
    mode: ResizeMode,
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    match plan {
        ResizePlan::Identity => Ok(image),
        ResizePlan::Resize {
            width,
            height,
            crop,
        } => image
            .resize(width, height, mode, crop, backend, context)
            .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error)),
        ResizePlan::ResizeThenCrop {
            scaled_width,
            scaled_height,
            width,
            height,
        } => {
            let resized = image
                .resize(
                    scaled_width,
                    scaled_height,
                    mode,
                    ResizeCrop::Disabled,
                    backend,
                    context,
                )
                .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
            center_crop(&resized, width, height, backend, context)
        }
    }
}

fn center_crop(
    image: &ImageTensor,
    width: u64,
    height: u64,
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<ImageTensor, NativeNodeFailure> {
    let (batch, input_height, input_width, channels) = image
        .dimensions()
        .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
    if width == 0 || height == 0 || width > input_width || height > input_height {
        return Err(resize_failure(
            RESIZE_CLASS_TYPE,
            "center crop is outside the resized input",
        ));
    }
    let left = (input_width - width) / 2;
    let top = (input_height - height) / 2;
    let count = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(channels))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| resize_failure(RESIZE_CLASS_TYPE, "crop shape overflowed"))?;
    let source = image
        .as_f32_slice()
        .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
    let mut output = backend
        .workspace_vec(context, count)
        .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
    for batch_index in 0..batch {
        for y in 0..height {
            context
                .check()
                .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
            for x in 0..width {
                for channel in 0..channels {
                    let index = image_offset(
                        batch_index,
                        top + y,
                        left + x,
                        channel,
                        input_height,
                        input_width,
                        channels,
                    )?;
                    output
                        .try_push(*source.get(index).ok_or_else(|| {
                            resize_failure(RESIZE_CLASS_TYPE, "crop index exceeded storage")
                        })?)
                        .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
                }
            }
        }
    }
    ImageTensor::from_f32(backend, context, batch, height, width, channels, &output)
        .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))
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
        .ok_or_else(|| resize_failure(RESIZE_CLASS_TYPE, "image index overflowed"))
}

fn edge_inputs(
    inputs: &BTreeMap<String, NativeValue>,
    edge: Edge,
) -> Result<(&NativeOpaqueHandle, u64), NativeNodeFailure> {
    if inputs.len() != 2 {
        return Err(invalid_inputs(format!(
            "{} requires exactly images and {}",
            edge.class_type(),
            edge.input_name()
        )));
    }
    let handle = required_exact_handle(
        inputs.get("images"),
        "images",
        NativeHandleKind::Image,
        "IMAGE",
    )?;
    let size = bounded_unsigned(
        inputs.get(edge.input_name()),
        edge.input_name(),
        1,
        MAX_DEPRECATED_RESOLUTION,
    )?;
    Ok((handle, size))
}

fn required_image_or_mask_handle<'a>(
    value: Option<&'a NativeValue>,
    name: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(invalid_inputs(format!(
            "{name} must be an IMAGE or MASK handle"
        )));
    };
    if !matches!(
        (value.handle_type().kind, value.handle_type().type_id.as_str()),
        (NativeHandleKind::Image, "IMAGE") | (NativeHandleKind::Mask, "MASK")
    ) {
        return Err(invalid_inputs(format!(
            "{name} must be an exact IMAGE or MASK handle"
        )));
    }
    Ok(value)
}

fn required_exact_handle<'a>(
    value: Option<&'a NativeValue>,
    name: &str,
    kind: NativeHandleKind,
    type_id: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(invalid_inputs(format!("{name} must be a {type_id} handle")));
    };
    if value.handle_type().kind != kind || value.handle_type().type_id != type_id {
        return Err(invalid_inputs(format!(
            "{name} must be an exact {type_id} handle"
        )));
    }
    Ok(value)
}

fn bounded_unsigned(
    value: Option<&NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = match value {
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

fn bounded_number(
    value: Option<&NativeValue>,
    name: &str,
    minimum: f64,
    maximum: f64,
) -> Result<f64, NativeNodeFailure> {
    let value = match value {
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
    if !value.is_finite() || !(minimum..=maximum).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be finite and between {minimum} and {maximum}"
        )));
    }
    Ok(value)
}

fn round_dimension(value: f64, name: &str) -> Result<u64, NativeNodeFailure> {
    checked_dimension(value.round_ties_even(), name)
}

fn floor_dimension(value: f64, name: &str) -> Result<u64, NativeNodeFailure> {
    checked_dimension(value.floor(), name)
}

fn ceil_dimension(value: f64, name: &str) -> Result<u64, NativeNodeFailure> {
    checked_dimension(value.ceil(), name)
}

fn checked_dimension(value: f64, name: &str) -> Result<u64, NativeNodeFailure> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        return Err(invalid_inputs(format!("{name} overflowed")));
    }
    Ok(value as u64)
}

fn resolve_tensor(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
    class_type: &str,
) -> Result<crate::NativeResolvedPayload, NativeNodeFailure> {
    context
        .handle_store()
        .resolve(handle, handle.handle_type(), &context.cancellation)
        .map_err(|error| handle_failure(error, class_type))
}

fn resolved_tensor_role(
    resolved: &NativeStoredPayload,
    handle: &NativeOpaqueHandle,
) -> Result<NativeTensorRole, NativeNodeFailure> {
    let NativeStoredPayload::Tensor(payload) = resolved else {
        return Err(resize_failure(
            RESIZE_CLASS_TYPE,
            "image transform handle did not resolve to tensor storage",
        ));
    };
    let expected = match handle.handle_type().kind {
        NativeHandleKind::Image => NativeTensorRole::Image,
        NativeHandleKind::Mask => NativeTensorRole::Mask,
        _ => {
            return Err(invalid_inputs(
                "image transform handle must be IMAGE or MASK",
            ));
        }
    };
    if payload.role() != expected {
        return Err(resize_failure(
            RESIZE_CLASS_TYPE,
            "image transform handle resolved to the wrong tensor role",
        ));
    }
    Ok(expected)
}

fn image_from_payload(
    resolved: &NativeStoredPayload,
    role: NativeTensorRole,
) -> Result<ImageTensor, NativeNodeFailure> {
    let NativeStoredPayload::Tensor(payload) = resolved else {
        return Err(resize_failure(
            RESIZE_CLASS_TYPE,
            "image transform payload is not a tensor",
        ));
    };
    if role == NativeTensorRole::Image {
        return payload.image().cloned().ok_or_else(|| {
            resize_failure(RESIZE_CLASS_TYPE, "IMAGE payload has no canonical ImageTensor")
        });
    }
    let shape = payload.tensor().descriptor().shape();
    let [batch, height, width] = shape else {
        return Err(resize_failure(
            RESIZE_CLASS_TYPE,
            "MASK payload must have BHW rank three",
        ));
    };
    let descriptor = payload
        .tensor()
        .descriptor()
        .reshaped_view(vec![*batch, *height, *width, 1])
        .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
    let tensor = payload
        .tensor()
        .view(descriptor, ViewAccess::ReadOnly)
        .map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))?;
    ImageTensor::from_tensor(tensor).map_err(|error| resize_failure(RESIZE_CLASS_TYPE, error))
}

fn publish_image(
    context: &NativeNodeContext,
    image: ImageTensor,
    role: NativeTensorRole,
    class_type: &str,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check_cancellation(context, class_type)?;
    let payload = NativeTensorPayload::from_image(role, image)
        .map_err(|error| resize_failure(class_type, error))?;
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
            code: "invalid_image_transform_handle".to_owned(),
            message: format!("{class_type} handle is not available: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn compute_failure(error: NativeNodeContractError) -> NativeNodeFailure {
    resize_failure(RESIZE_CLASS_TYPE, error)
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn resize_failure(class_type: &str, error: impl std::fmt::Display) -> NativeNodeFailure {
    let message = error.to_string();
    if message.to_ascii_lowercase().contains("cancelled") {
        return interrupted_failure(class_type);
    }
    NativeNodeFailure {
        code: "native_image_resize_failed".to_owned(),
        message: format!("{class_type} resize failed: {message}"),
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
        NativeResolvedPayloadRetention, NodeRegistry,
    };
    use sim_tensor::{
        CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, StreamId, TensorDescriptor,
    };
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::{Value, json};
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
        "/../sim_test_support/fixtures/nodes/image-transform-comfy-node-0541/fixture.json"
    ));

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        payloads: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(store: u128, attempt_id: AttemptId) -> Result<Arc<Self>, NativeNodeContractError> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(store),
                    Uuid::from_u128(store + 1),
                )?,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                payloads: Mutex::new(BTreeMap::new()),
            }))
        }

        fn payload(
            &self,
            handle: &NativeOpaqueHandle,
        ) -> Result<Arc<NativeStoredPayload>, Box<dyn Error>> {
            self.payloads
                .lock()
                .map_err(|_| "test payload store is poisoned")?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| "test payload is absent".into())
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
            NativeResolvedPayload::checked(payload, Arc::new(TestRetention))
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
                "resize-{}",
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

    struct Harness {
        backend: Arc<CpuBackend>,
        store: Arc<TestStore>,
        context: NativeNodeContext,
    }

    impl Harness {
        fn new(store: u128, cancellation: CancellationToken) -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x50801));
            let store = TestStore::new(store, attempt_id)?;
            let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
            let backend = Arc::new(backend);
            let scratch = authority.authorize_workspace(64 * 1024 * 1024)?;
            let node_id = NodeId("image-transform-test".to_owned());
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x50802),
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
                PromptId(Uuid::from_u128(0x50803)),
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

        fn publish_image(
            &self,
            batch: u64,
            height: u64,
            width: u64,
            channels: u64,
            values: &[f32],
        ) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            let execution = self.context.compute_session()?.execution_context(&self.context)?;
            let image = ImageTensor::from_f32(
                &self.backend,
                &execution,
                batch,
                height,
                width,
                channels,
                values,
            )?;
            let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(payload)),
                &self.context.cancellation,
            )?)
        }

        fn publish_mask(
            &self,
            height: u64,
            width: u64,
            values: &[f32],
        ) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            let execution = self.context.compute_session()?.execution_context(&self.context)?;
            let descriptor = TensorDescriptor::contiguous(
                vec![1, height, width],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?;
            let (tensor, _) = self.backend.upload_f32(descriptor, values, &execution)?;
            let payload = NativeTensorPayload::from_tensor(NativeTensorRole::Mask, tensor)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(payload)),
                &self.context.cancellation,
            )?)
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

    fn dynamic(fields: impl IntoIterator<Item = (&'static str, NativeValue)>) -> NativeValue {
        NativeStructuredValue::checked(
            "COMFY_DYNAMICCOMBO_V3",
            fields
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect(),
        )
        .expect("valid structured test input")
        .into_native_value()
    }

    fn string(value: &str) -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::String(value.to_owned()),
        }
    }

    fn integer(value: u64) -> NativeValue {
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }
    }

    fn resize_inputs_for(
        input: NativeOpaqueHandle,
        resize_type: NativeValue,
        method: &str,
    ) -> BTreeMap<String, NativeValue> {
        BTreeMap::from([
            ("input".to_owned(), NativeValue::Handle { value: input }),
            ("resize_type".to_owned(), resize_type),
            ("scale_method".to_owned(), string(method)),
        ])
    }

    fn outcome_handle(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("resize did not return values".into());
        };
        if ui.is_some() || !effects.is_empty() || outputs.len() != 1 {
            return Err("resize returned an invalid outcome envelope".into());
        }
        let Some(NativeValue::Handle { value }) = outputs.into_iter().next() else {
            return Err("resize output is not a handle".into());
        };
        Ok(value)
    }

    fn payload_dimensions(
        store: &TestStore,
        handle: &NativeOpaqueHandle,
    ) -> Result<(NativeTensorRole, Vec<u64>), Box<dyn Error>> {
        let payload = store.payload(handle)?;
        let NativeStoredPayload::Tensor(payload) = payload.as_ref() else {
            return Err("resize output is not a tensor payload".into());
        };
        Ok((
            payload.role(),
            payload.tensor().descriptor().shape().to_vec(),
        ))
    }

    #[test]
    fn fixture_and_exact_source_schemas_are_bound_once() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(
            fixture.pointer("/stable_task_id").and_then(Value::as_str),
            Some("comfy-parity-native-nodes-image-transform-comfy-node-0541")
        );
        assert_eq!(
            fixture.pointer("/sources/0/sha256").and_then(Value::as_str),
            Some("96ec39e8d0e9fe9a70b332f97f994d507e1fa223a26699a3c9c9fbeedacf6575")
        );
        assert_eq!(
            fixture.pointer("/sources/1/sha256").and_then(Value::as_str),
            Some("3b27465fec391509083bd1837895c09abc489c04d81afae5ffe631abd6a4e772")
        );
        assert_eq!(
            fixture
                .pointer("/deprecated_edges/presentation/is_deprecated")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            fixture
                .pointer("/deprecated_edges/presentation/is_experimental")
                .and_then(Value::as_bool),
            Some(true)
        );
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), NODE_DESCRIPTOR_IDS.len());
        let registry = NodeRegistry::built_in()?;
        for (binding, class_type) in bindings.iter().zip(NODE_DESCRIPTOR_IDS) {
            assert_eq!(binding.descriptor().class_type, *class_type);
            binding.validate()?;
            registry.validate_native_binding(binding)?;
        }
        for class_type in [LONGER_CLASS_TYPE, SHORTER_CLASS_TYPE] {
            let binding = bindings
                .iter()
                .find(|binding| binding.descriptor().class_type == class_type)
                .ok_or("deprecated edge binding is absent")?;
            assert!(binding.presentation().is_deprecated);
            assert!(binding.presentation().is_experimental);
            assert!(binding.presentation().display_name.ends_with(" (DEPRECATED)"));
        }
        let resize = bindings
            .iter()
            .find(|binding| binding.descriptor().class_type == RESIZE_CLASS_TYPE)
            .ok_or("ResizeImageMaskNode binding is absent")?;
        assert_eq!(resize.descriptor().inputs.len(), 3);
        assert_eq!(resize.descriptor().outputs[0].produced_type, NativeValueType::Any);
        assert!(!resize.presentation().is_experimental);
        let structured = resize
            .descriptor()
            .source_schema
            .as_ref()
            .ok_or("source schema")?
            .inputs
            .iter()
            .find(|input| input.name == "resize_type")
            .ok_or("resize_type schema")?
            .structured_options()?;
        assert_eq!(structured.len(), 9);
        let match_size = structured.iter().find(|option| option.selector == "match size")
            .ok_or("match-size option")?;
        assert_eq!(match_size.fields.iter().find(|field| field.path == ["match"])
            .ok_or("match field")?.schema.source_type_names, ["IMAGE", "MASK"]);
        Ok(())
    }

    #[test]
    fn all_resize_plans_preserve_python_dimension_rules() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new(0x50810, CancellationToken::default())?;
        let input = harness.publish_image(1, 3, 5, 1, &[0.5; 15])?;
        let cases = [
            (dynamic([("resize_type", string("scale by multiplier")), ("multiplier", NativeValue::Primitive { value: NativePrimitive::Number(1.5) })]), (8, 4)),
            (dynamic([("resize_type", string("scale dimensions")), ("width", integer(10)), ("height", integer(0)), ("crop", string("disabled"))]), (10, 6)),
            (dynamic([("resize_type", string("scale longer dimension")), ("longer_size", integer(7))]), (7, 4)),
            (dynamic([("resize_type", string("scale shorter dimension")), ("shorter_size", integer(7))]), (12, 7)),
            (dynamic([("resize_type", string("scale width")), ("width", integer(6))]), (6, 4)),
            (dynamic([("resize_type", string("scale height")), ("height", integer(6))]), (10, 6)),
            (dynamic([("resize_type", string("scale total pixels")), ("megapixels", NativeValue::Primitive { value: NativePrimitive::Number(0.01) })]), (132, 79)),
            (dynamic([("resize_type", string("scale to multiple")), ("multiple", integer(2))]), (4, 2)),
        ];
        for (fields, expected) in cases {
            let inputs = resize_inputs_for(input.clone(), fields, "nearest-exact");
            let parsed = resize_inputs(&inputs)?;
            let plan = resize_plan(&harness.context, &parsed, (1, 3, 5, 1))?;
            let (width, height) = match plan {
                ResizePlan::Resize { width, height, .. } => (width, height),
                ResizePlan::ResizeThenCrop { width, height, .. } => (width, height),
                ResizePlan::Identity => (5, 3),
            };
            assert_eq!((width, height), expected);
        }
        Ok(())
    }

    #[test]
    fn image_mask_match_and_identity_execution_are_exact() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new(0x50820, CancellationToken::default())?;
        let image = harness.publish_image(1, 2, 4, 1, &[0.0, 0.25, 0.5, 1.0, 1.0, 0.5, 0.25, 0.0])?;
        let mask = harness.publish_mask(3, 2, &[0.0; 6])?;
        let match_fields = dynamic([
            ("resize_type", string("match size")),
            ("match", NativeValue::Handle { value: mask.clone() }),
            ("crop", string("center")),
        ]);
        let output = futures::executor::block_on(executable(RESIZE_CLASS_TYPE)?.execute(
            harness.context.clone(),
            resize_inputs_for(image.clone(), match_fields, "bilinear"),
        ))?;
        let output = outcome_handle(output)?;
        assert_eq!(payload_dimensions(&harness.store, &output)?, (NativeTensorRole::Image, vec![1, 3, 2, 1]));

        let mask_fields = dynamic([
            ("resize_type", string("scale dimensions")),
            ("width", integer(4)),
            ("height", integer(6)),
            ("crop", string("disabled")),
        ]);
        let output = futures::executor::block_on(executable(RESIZE_CLASS_TYPE)?.execute(
            harness.context.clone(),
            resize_inputs_for(mask, mask_fields, "nearest-exact"),
        ))?;
        let output = outcome_handle(output)?;
        assert_eq!(payload_dimensions(&harness.store, &output)?, (NativeTensorRole::Mask, vec![1, 6, 4]));

        let identity = dynamic([
            ("resize_type", string("scale dimensions")),
            ("width", integer(0)),
            ("height", integer(0)),
            ("crop", string("center")),
        ]);
        let output = futures::executor::block_on(executable(RESIZE_CLASS_TYPE)?.execute(
            harness.context,
            resize_inputs_for(image.clone(), identity, "lanczos"),
        ))?;
        assert_eq!(outcome_handle(output)?, image);
        Ok(())
    }

    #[test]
    fn deprecated_edges_use_pillow_lanczos_and_source_batch_rules() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new(0x50830, CancellationToken::default())?;
        let image = harness.publish_image(1, 2, 4, 1, &[0.0, 0.25, 0.5, 1.0, 1.0, 0.5, 0.25, 0.0])?;
        for (class_type, name, size, shape) in [
            (LONGER_CLASS_TYPE, "longer_edge", 6, vec![1, 3, 6, 1]),
            (SHORTER_CLASS_TYPE, "shorter_edge", 3, vec![1, 3, 6, 1]),
        ] {
            let output = futures::executor::block_on(executable(class_type)?.execute(
                harness.context.clone(),
                BTreeMap::from([
                    ("images".to_owned(), NativeValue::Handle { value: image.clone() }),
                    (name.to_owned(), integer(size)),
                ]),
            ))?;
            let output = outcome_handle(output)?;
            let payload = harness.store.payload(&output)?;
            let NativeStoredPayload::Tensor(payload) = payload.as_ref() else { return Err("tensor".into()); };
            assert_eq!(payload.tensor().descriptor().shape(), shape);
            assert!(payload.tensor().contiguous_bytes()?.chunks_exact(4).all(|bytes| {
                let value = f32::from_ne_bytes(bytes.try_into().expect("F32 byte width"));
                (value * 255.0).fract() == 0.0
            }));
        }
        Ok(())
    }

    #[test]
    fn invalid_nested_handles_cancellation_and_recovery_fail_closed() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new(0x50840, CancellationToken::default())?;
        let image = harness.publish_image(1, 2, 2, 1, &[0.5; 4])?;
        let fresh = Harness::new(0x50850, CancellationToken::default())?;
        let fields = dynamic([
            ("resize_type", string("match size")),
            ("match", NativeValue::Handle { value: image }),
            ("crop", string("center")),
        ]);
        let error = futures::executor::block_on(executable(RESIZE_CLASS_TYPE)?.execute(
            fresh.context.clone(),
            resize_inputs_for(fresh.publish_image(1, 1, 1, 1, &[0.0])?, fields, "area"),
        )).expect_err("stale nested match handle unexpectedly succeeded");
        assert_eq!(error.code, "invalid_image_transform_handle");

        let cancelled = CancellationToken::default();
        let cancelled_harness = Harness::new(0x50860, cancelled.clone())?;
        let input = cancelled_harness.publish_image(1, 1, 1, 1, &[0.0])?;
        cancelled.cancel();
        let error = futures::executor::block_on(executable(RESIZE_CLASS_TYPE)?.execute(
            cancelled_harness.context,
            resize_inputs_for(input, dynamic([
                ("resize_type", string("scale by multiplier")),
                ("multiplier", NativeValue::Primitive { value: NativePrimitive::Number(2.0) }),
            ]), "bilinear"),
        )).expect_err("cancelled resize unexpectedly succeeded");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);

        let recovered = futures::executor::block_on(executable(RESIZE_CLASS_TYPE)?.execute(
            fresh.context.clone(),
            resize_inputs_for(
                fresh.publish_image(1, 1, 1, 1, &[0.25])?,
                dynamic([
                    ("resize_type", string("scale dimensions")),
                    ("width", integer(2)),
                    ("height", integer(2)),
                    ("crop", string("disabled")),
                ]),
                "nearest-exact",
            ),
        ))?;
        assert_eq!(payload_dimensions(&fresh.store, &outcome_handle(recovered)?)?.1, vec![1, 2, 2, 1]);
        Ok(())
    }

    #[test]
    fn fixture_persistence_and_cache_contract_are_lossless() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        let workflow = fixture.pointer("/persistence/workflow").ok_or("workflow")?;
        assert_eq!(serde_json::from_slice::<Value>(&serde_json::to_vec(workflow)?)?, *workflow);
        assert_eq!(workflow.pointer("/inputs/resize_type/match"), Some(&json!(["reference", 0])));
        assert_eq!(workflow.pointer("/unknown_data/preserve"), Some(&json!(true)));
        let fields = dynamic([
            ("resize_type", string("scale dimensions")),
            ("width", integer(0)),
            ("height", integer(0)),
            ("crop", string("center")),
        ]);
        let harness = Harness::new(0x50870, CancellationToken::default())?;
        let input = harness.publish_image(1, 1, 1, 1, &[0.0])?;
        let inputs = resize_inputs_for(input, fields, "area");
        let node = executable(RESIZE_CLASS_TYPE)?;
        assert_eq!(node.cache_change_token(&inputs)?, node.cache_change_token(&inputs)?);
        assert_eq!(node.cache_dependencies(&harness.context, &inputs)?, NativeCacheDependencies::default());
        Ok(())
    }
}
