use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeDynamicInputDescriptor, NativeEffectClass, NativeEffectServiceError, NativeHandleKind,
    NativeHandleStoreError, NativeHandleType, NativeInputDescriptor, NativeInputRequirement,
    NativeNode, NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext,
    NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind,
    NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor,
    NativePortCardinality, NativePreparedEffectRequest, NativePrimitive, NativeStoredPayload,
    NativeStructuredValue, NativeValue,
    built_in_source_schema, native_value_type_for_output_schema,
    native_value_types_for_input_schema,
};
use sim_tensor::{
    NativeShaderError, NativeShaderRequest, NativeTensorPayload, NativeTensorRole,
    MAX_SHADER_BOOLS, MAX_SHADER_CURVES, MAX_SHADER_FLOATS, MAX_SHADER_IMAGES, MAX_SHADER_INTS,
    MAX_SHADER_OUTPUTS,
};
use sim_types::CancellationToken;
use futures::future::BoxFuture;
use serde_json::Value;
use crate::execution::{NativeShaderPreviewError, NativeShaderServiceError};
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["GLSLShader"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CLASS_TYPE: &str = "GLSLShader";
const FEATURE_ID: &str = "COMFY-NODE-0211";
const IMPLEMENTATION_VERSION: &str = "source-fd472517-v1";
const CACHE_CHANGE_TOKEN: &str = "source-fd472517-glsl-shader-v1";
const CATEGORY: &str = "image/shader";
const MAX_RESOLUTION: u32 = 16_384;
const CURVE_LUT_SAMPLES: usize = 256;
const OUTPUT_NAMES: [&str; MAX_SHADER_OUTPUTS] = ["IMAGE0", "IMAGE1", "IMAGE2", "IMAGE3"];

#[cfg(test)]
const DEFAULT_FRAGMENT_SHADER: &str = "#version 300 es\n\
precision highp float;\n\
\n\
uniform sampler2D u_image0;\n\
uniform vec2 u_resolution;\n\
\n\
in vec2 v_texCoord;\n\
layout(location = 0) out vec4 fragColor0;\n\
\n\
void main() {\n\
    fragColor0 = texture(u_image0, v_texCoord);\n\
}\n";

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    Ok(vec![native_binding()?])
}

fn native_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    let catalog_schema = built_in_source_schema(CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let dynamic_schema = catalog_schema.dynamic_inputs.clone();
    let source_schema = catalog_schema
        .bind_execution_ports(
            &["fragment_shader".to_owned(), "size_mode".to_owned()],
            &dynamic_schema,
            &OUTPUT_NAMES.map(str::to_owned),
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog_schema
        .inputs
        .iter()
        .map(|input| {
            let accepted_types = native_value_types_for_input_schema(&input.schema)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: input.requirement == NativeInputRequirement::Hidden,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal: true,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    let dynamic_inputs = dynamic_schema
        .iter()
        .map(|dynamic| {
            let accepted_types = native_value_types_for_input_schema(&dynamic.input)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let allows_literal = !dynamic
                .input
                .source_type_names
                .iter()
                .any(|source_type| source_type == "IMAGE");
            Ok(NativeDynamicInputDescriptor {
                name_template: dynamic.identity.clone(),
                start_index: dynamic.start_index,
                minimum_count: dynamic.minimum_count,
                maximum_count: dynamic.maximum_count,
                input: NativeInputDescriptor {
                    name: dynamic.input.name.clone(),
                    accepted_types,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal,
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
    Ok(NativeNodeBinding::Executable {
        feature_id: FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: CLASS_TYPE.to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs,
            outputs,
            output_node: false,
            effect: NativeEffectClass::ExclusiveDevice,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "GLSL Shader".to_owned(),
            category: CATEGORY.to_owned(),
            description: "Apply GLSL ES fragment shaders to images. u_resolution (vec2) is always available."
                .to_owned(),
            output_names: OUTPUT_NAMES.map(str::to_owned).to_vec(),
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: true,
        },
        node: Arc::new(GlslShaderNode),
    })
}

#[derive(Debug)]
struct GlslShaderNode;

impl NativeNode for GlslShaderNode {
    fn class_type(&self) -> &str {
        CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        IMPLEMENTATION_VERSION
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
        Ok(NativeCacheDependencies::default())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move { execute_shader_node(&context, &inputs) })
    }
}

struct ShaderInputs<'a> {
    fragment_source: &'a str,
    size: ShaderSize,
    images: Vec<&'a NativeOpaqueHandle>,
    floats: Vec<f32>,
    ints: Vec<i32>,
    bools: Vec<bool>,
    curves: Vec<Vec<f32>>,
}

#[derive(Clone, Copy)]
enum ShaderSize {
    FromInput,
    Custom { width: u32, height: u32 },
}

fn parse_inputs(inputs: &BTreeMap<String, NativeValue>) -> Result<ShaderInputs<'_>, NativeNodeFailure> {
    for value in inputs.values() {
        value
            .validate()
            .map_err(|error| invalid_inputs(error.to_string()))?;
    }
    for name in inputs.keys() {
        if !matches!(name.as_str(), "fragment_shader" | "size_mode")
            && dynamic_index(name, "image", MAX_SHADER_IMAGES).is_none()
            && dynamic_index(name, "u_float", MAX_SHADER_FLOATS).is_none()
            && dynamic_index(name, "u_int", MAX_SHADER_INTS).is_none()
            && dynamic_index(name, "u_bool", MAX_SHADER_BOOLS).is_none()
            && dynamic_index(name, "u_curve", MAX_SHADER_CURVES).is_none()
        {
            return Err(invalid_inputs(format!("GLSLShader received unknown input {name}")));
        }
    }
    let fragment_source = match inputs.get("fragment_shader") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => value.as_str(),
        _ => return Err(invalid_inputs("fragment_shader must be a STRING")),
    };
    let size = parse_size_mode(inputs.get("size_mode"))?;
    let images = dynamic_values(inputs, "image", MAX_SHADER_IMAGES)
        .into_iter()
        .map(|(_, value)| exact_image_handle(value))
        .collect::<Result<Vec<_>, _>>()?;
    if images.is_empty() || !inputs.contains_key("image1") {
        return Err(invalid_inputs("At least one input image is required"));
    }
    let floats = dynamic_values(inputs, "u_float", MAX_SHADER_FLOATS)
        .into_iter()
        .map(|(_, value)| match value {
            NativeValue::Primitive {
                value: NativePrimitive::Number(value),
            } if value.is_finite() => Ok(*value as f32),
            _ => Err(invalid_inputs("shader float uniforms must be finite FLOAT values")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if floats.iter().any(|value| !value.is_finite()) {
        return Err(invalid_inputs("shader float uniforms exceed F32 bounds"));
    }
    let ints = dynamic_values(inputs, "u_int", MAX_SHADER_INTS)
        .into_iter()
        .map(|(_, value)| integer_i32(value, "shader integer uniform"))
        .collect::<Result<Vec<_>, _>>()?;
    let bools = dynamic_values(inputs, "u_bool", MAX_SHADER_BOOLS)
        .into_iter()
        .map(|(_, value)| match value {
            NativeValue::Primitive {
                value: NativePrimitive::Boolean(value),
            } => Ok(*value),
            _ => Err(invalid_inputs("shader boolean uniforms must be BOOLEAN values")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let curves = dynamic_values(inputs, "u_curve", MAX_SHADER_CURVES)
        .into_iter()
        .map(|(_, value)| curve_lut(value))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ShaderInputs {
        fragment_source,
        size,
        images,
        floats,
        ints,
        bools,
        curves,
    })
}

fn dynamic_index(name: &str, prefix: &str, maximum: usize) -> Option<u32> {
    name.strip_prefix(prefix)
        .and_then(|suffix| suffix.parse::<u32>().ok())
        .filter(|index| *index >= 1 && usize::try_from(*index).is_ok_and(|index| index <= maximum))
}

fn dynamic_values<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    prefix: &str,
    maximum: usize,
) -> Vec<(u32, &'a NativeValue)> {
    let mut values = inputs
        .iter()
        .filter_map(|(name, value)| dynamic_index(name, prefix, maximum).map(|index| (index, value)))
        .collect::<Vec<_>>();
    values.sort_unstable_by_key(|(index, _)| *index);
    values
}

fn parse_size_mode(value: Option<&NativeValue>) -> Result<ShaderSize, NativeNodeFailure> {
    let fields = structured_fields(value)?;
    let selector = match fields.get("size_mode") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => value.as_str(),
        _ => return Err(invalid_inputs("size_mode must contain a string selector")),
    };
    match selector {
        "from_input" if fields.len() == 1 => Ok(ShaderSize::FromInput),
        "custom"
            if fields.len() == 3
                && fields.contains_key("width")
                && fields.contains_key("height") =>
        {
            let width = dimension(fields.get("width"), "width")?;
            let height = dimension(fields.get("height"), "height")?;
            Ok(ShaderSize::Custom { width, height })
        }
        "from_input" | "custom" => Err(invalid_inputs(
            "size_mode contains missing or inactive structured fields",
        )),
        _ => Err(invalid_inputs("size_mode selector must be from_input or custom")),
    }
}

fn structured_fields(
    value: Option<&NativeValue>,
) -> Result<BTreeMap<String, NativeValue>, NativeNodeFailure> {
    let value = value.ok_or_else(|| invalid_inputs("size_mode is required"))?;
    if let Some(structured) = NativeStructuredValue::from_native_value(value)
        .map_err(|error| invalid_inputs(error.to_string()))?
    {
        if structured.type_name() != "COMFY_DYNAMICCOMBO_V3" {
            return Err(invalid_inputs(
                "size_mode must use the COMFY_DYNAMICCOMBO_V3 structured type",
            ));
        }
        return Ok(structured.fields().clone());
    }
    let NativeValue::PreservedUnknown { type_name, value } = value else {
        return Err(invalid_inputs("size_mode must be a structured value"));
    };
    if type_name != "COMFY_DYNAMICCOMBO_V3" {
        return Err(invalid_inputs(
            "size_mode must use the COMFY_DYNAMICCOMBO_V3 structured type",
        ));
    }
    let Value::Object(fields) = value else {
        return Err(invalid_inputs("size_mode must contain an object"));
    };
    fields
        .iter()
        .map(|(name, value)| Ok((name.clone(), json_primitive(value)?)))
        .collect()
}

fn json_primitive(value: &Value) -> Result<NativeValue, NativeNodeFailure> {
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
                        .ok_or_else(|| invalid_inputs("size_mode contains an invalid number"))?,
                )
            }
        }
        Value::String(value) => NativePrimitive::String(value.clone()),
        Value::Array(_) | Value::Object(_) => {
            return Err(invalid_inputs("size_mode contains a nested value"));
        }
    };
    Ok(NativeValue::Primitive { value })
}

fn dimension(value: Option<&NativeValue>, name: &str) -> Result<u32, NativeNodeFailure> {
    let value = integer_u64(value, name)?;
    if !(1..=u64::from(MAX_RESOLUTION)).contains(&value) {
        return Err(invalid_inputs(format!(
            "{name} must be between 1 and {MAX_RESOLUTION}"
        )));
    }
    u32::try_from(value).map_err(|_| invalid_inputs(format!("{name} is out of range")))
}

fn integer_u64(value: Option<&NativeValue>, name: &str) -> Result<u64, NativeNodeFailure> {
    match value {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).map_err(|_| invalid_inputs(format!("{name} must be positive"))),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Ok(*value),
        _ => Err(invalid_inputs(format!("{name} must be an INT"))),
    }
}

fn integer_i32(value: &NativeValue, name: &str) -> Result<i32, NativeNodeFailure> {
    match value {
        NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        } => i32::try_from(*value).map_err(|_| invalid_inputs(format!("{name} exceeds I32 bounds"))),
        NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        } => i32::try_from(*value).map_err(|_| invalid_inputs(format!("{name} exceeds I32 bounds"))),
        _ => Err(invalid_inputs(format!("{name} must be an INT"))),
    }
}

fn exact_image_handle(value: &NativeValue) -> Result<&NativeOpaqueHandle, NativeNodeFailure> {
    let NativeValue::Handle { value } = value else {
        return Err(invalid_inputs("shader image inputs must be IMAGE handles"));
    };
    if value.handle_type().kind != NativeHandleKind::Image || value.handle_type().type_id != "IMAGE" {
        return Err(invalid_inputs("shader image inputs must be exact IMAGE handles"));
    }
    Ok(value)
}

fn curve_lut(value: &NativeValue) -> Result<Vec<f32>, NativeNodeFailure> {
    let NativeValue::PreservedUnknown { type_name, value } = value else {
        return Err(invalid_inputs("shader curve inputs must be CURVE values"));
    };
    if type_name != "CURVE" {
        return Err(invalid_inputs("shader curve inputs must use the CURVE type"));
    }
    let (points, interpolation) = match value {
        Value::Object(object) => (
            object
                .get("points")
                .ok_or_else(|| invalid_inputs("CURVE is missing points"))?,
            object.get("interpolation").and_then(Value::as_str),
        ),
        Value::Array(_) => (value, None),
        _ => return Err(invalid_inputs("CURVE must contain points or a curve object")),
    };
    let Value::Array(points) = points else {
        return Err(invalid_inputs("CURVE points must be an array"));
    };
    let mut parsed = Vec::new();
    parsed
        .try_reserve_exact(points.len())
        .map_err(|error| resource_failure(format!("CURVE point allocation failed: {error}")))?;
    for point in points {
        let Value::Array(coordinates) = point else {
            return Err(invalid_inputs("each CURVE point must be an [x, y] pair"));
        };
        let [x, y] = coordinates.as_slice() else {
            return Err(invalid_inputs("each CURVE point must be an [x, y] pair"));
        };
        parsed.push((json_f64(x, "CURVE x")?, json_f64(y, "CURVE y")?));
    }
    parsed.sort_by(|left, right| left.0.total_cmp(&right.0));
    let slopes = if interpolation == Some("linear") {
        None
    } else {
        Some(monotone_slopes(&parsed)?)
    };
    let mut result = Vec::new();
    result
        .try_reserve_exact(CURVE_LUT_SAMPLES)
        .map_err(|error| resource_failure(format!("CURVE LUT allocation failed: {error}")))?;
    for index in 0..CURVE_LUT_SAMPLES {
        let x = index as f64 / (CURVE_LUT_SAMPLES - 1) as f64;
        let value = match &slopes {
            Some(slopes) => monotone_interpolate(&parsed, slopes, x),
            None => linear_interpolate(&parsed, x),
        };
        let value = value as f32;
        if !value.is_finite() {
            return Err(invalid_inputs("CURVE LUT exceeds finite F32 bounds"));
        }
        result.push(value);
    }
    Ok(result)
}

fn json_f64(value: &Value, name: &str) -> Result<f64, NativeNodeFailure> {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid_inputs(format!("{name} must be finite")))
}

fn monotone_slopes(points: &[(f64, f64)]) -> Result<Vec<f64>, NativeNodeFailure> {
    let count = points.len();
    if count < 2 {
        return Ok(vec![0.0; count]);
    }
    let mut deltas = Vec::new();
    deltas
        .try_reserve_exact(count - 1)
        .map_err(|error| resource_failure(format!("CURVE slope allocation failed: {error}")))?;
    for pair in points.windows(2) {
        let delta_x = pair[1].0 - pair[0].0;
        deltas.push(if delta_x == 0.0 {
            0.0
        } else {
            (pair[1].1 - pair[0].1) / delta_x
        });
    }
    let mut slopes = vec![0.0; count];
    slopes[0] = deltas[0];
    slopes[count - 1] = deltas[count - 2];
    for index in 1..count - 1 {
        slopes[index] = if deltas[index - 1] * deltas[index] <= 0.0 {
            0.0
        } else {
            (deltas[index - 1] + deltas[index]) / 2.0
        };
    }
    for index in 0..count - 1 {
        if deltas[index] == 0.0 {
            slopes[index] = 0.0;
            slopes[index + 1] = 0.0;
            continue;
        }
        let alpha = slopes[index] / deltas[index];
        let beta = slopes[index + 1] / deltas[index];
        let sum = alpha * alpha + beta * beta;
        if sum > 9.0 {
            let factor = 3.0 / sum.sqrt();
            slopes[index] = factor * alpha * deltas[index];
            slopes[index + 1] = factor * beta * deltas[index];
        }
    }
    Ok(slopes)
}

fn linear_interpolate(points: &[(f64, f64)], x: f64) -> f64 {
    match points {
        [] => 0.0,
        [point] => point.1,
        _ if x <= points[0].0 => points[0].1,
        _ if x >= points[points.len() - 1].0 => points[points.len() - 1].1,
        _ => {
            let upper = points.partition_point(|point| point.0 <= x).min(points.len() - 1);
            let lower = upper.saturating_sub(1);
            let delta_x = points[upper].0 - points[lower].0;
            if delta_x == 0.0 {
                points[lower].1
            } else {
                let amount = (x - points[lower].0) / delta_x;
                points[lower].1 + amount * (points[upper].1 - points[lower].1)
            }
        }
    }
}

fn monotone_interpolate(points: &[(f64, f64)], slopes: &[f64], x: f64) -> f64 {
    match points {
        [] => 0.0,
        [point] => point.1,
        _ if x <= points[0].0 => points[0].1,
        _ if x >= points[points.len() - 1].0 => points[points.len() - 1].1,
        _ => {
            let upper = points.partition_point(|point| point.0 <= x).clamp(1, points.len() - 1);
            let lower = upper - 1;
            let delta_x = points[upper].0 - points[lower].0;
            if delta_x == 0.0 {
                return points[lower].1;
            }
            let amount = (x - points[lower].0) / delta_x;
            let amount_squared = amount * amount;
            let amount_cubed = amount_squared * amount;
            let h00 = 2.0 * amount_cubed - 3.0 * amount_squared + 1.0;
            let h10 = amount_cubed - 2.0 * amount_squared + amount;
            let h01 = -2.0 * amount_cubed + 3.0 * amount_squared;
            let h11 = amount_cubed - amount_squared;
            h00 * points[lower].1
                + h10 * delta_x * slopes[lower]
                + h01 * points[upper].1
                + h11 * delta_x * slopes[upper]
        }
    }
}

fn execute_shader_node(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check_cancellation(context)?;
    let parsed = parse_inputs(inputs)?;
    let image_type = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
    let mut retained = Vec::new();
    let mut images = Vec::new();
    retained
        .try_reserve_exact(parsed.images.len())
        .map_err(|error| resource_failure(format!("shader handle retention failed: {error}")))?;
    images
        .try_reserve_exact(parsed.images.len())
        .map_err(|error| resource_failure(format!("shader image allocation failed: {error}")))?;
    for handle in parsed.images {
        let resolved = context
            .handle_store()
            .resolve(handle, &image_type, &context.cancellation)
            .map_err(handle_failure)?;
        let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
            return Err(invalid_image("IMAGE handle does not contain a tensor payload"));
        };
        if payload.role() != NativeTensorRole::Image {
            return Err(invalid_image("IMAGE handle has the wrong tensor role"));
        }
        let image = payload
            .image()
            .ok_or_else(|| invalid_image("IMAGE handle has no canonical ImageTensor"))?
            .clone();
        retained.push(resolved);
        images.push(image);
    }
    let (width, height) = match parsed.size {
        ShaderSize::FromInput => {
            let (_, height, width, _) = images
                .first()
                .ok_or_else(|| invalid_inputs("At least one input image is required"))?
                .dimensions()
                .map_err(|error| invalid_image(error.to_string()))?;
            (
                u32::try_from(width).map_err(|_| invalid_inputs("input width exceeds U32 bounds"))?,
                u32::try_from(height)
                    .map_err(|_| invalid_inputs("input height exceeds U32 bounds"))?,
            )
        }
        ShaderSize::Custom { width, height } => (width, height),
    };
    let request = NativeShaderRequest {
        fragment_source: parsed.fragment_source.to_owned(),
        images,
        floats: parsed.floats,
        ints: parsed.ints,
        bools: parsed.bools,
        curves: parsed.curves,
        width,
        height,
    };
    let prepared = context
        .execute_shader_with_previews(&request)
        .map_err(shader_preview_failure)?;
    let (shader, effects, ui) = prepared.into_parts();
    let mut published = Vec::new();
    let completion = (|| {
        let mut payloads = Vec::new();
        payloads
            .try_reserve_exact(shader.outputs.len())
            .map_err(|error| resource_failure(format!("shader output allocation failed: {error}")))?;
        for output in shader.outputs {
            let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, output)
                .map_err(|error| invalid_image(error.to_string()))?;
            payloads.push(NativeStoredPayload::Tensor(Arc::new(payload)));
        }
        published
            .try_reserve_exact(payloads.len())
            .map_err(|error| resource_failure(format!("shader handle allocation failed: {error}")))?;
        for payload in payloads {
            check_cancellation(context)?;
            let handle = context
                .handle_store()
                .publish(payload, &context.cancellation)
                .map_err(handle_failure)?;
            published.push(handle);
        }
        check_cancellation(context)?;
        let outcome = NativeNodeOutcome::Values {
            outputs: published
                .iter()
                .cloned()
                .map(|value| NativeValue::Handle { value })
                .collect(),
            ui: Some(ui),
            effects: effects.clone(),
        };
        outcome
            .validate()
            .map_err(|error| invalid_inputs(error.to_string()))?;
        Ok(outcome)
    })();
    drop(retained);
    match completion {
        Ok(outcome) => Ok(outcome),
        Err(error) => {
            cleanup_failed_execution(context, &published, &effects)?;
            Err(error)
        }
    }
}

fn cleanup_failed_execution(
    context: &NativeNodeContext,
    published: &[NativeOpaqueHandle],
    effects: &[NativePreparedEffectRequest],
) -> Result<(), NativeNodeFailure> {
    let cleanup = CancellationToken::default();
    let mut cleanup_failure = None;
    for handle in published.iter().rev() {
        if let Err(error) = context.handle_store().revoke(handle, &cleanup)
            && cleanup_failure.is_none()
        {
            cleanup_failure = Some(NativeNodeFailure {
                code: "shader_output_rollback_failed".to_owned(),
                message: format!("GLSLShader could not revoke partial output: {error}"),
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            });
        }
    }
    match context.prepared_effects() {
        Ok(service) => {
            for effect in effects.iter().rev() {
                if let Err(error) = service.rollback_prepared(effect)
                    && cleanup_failure.is_none()
                {
                    cleanup_failure = Some(NativeNodeFailure {
                        code: "shader_preview_rollback_failed".to_owned(),
                        message: format!("GLSLShader could not roll back preview: {error}"),
                        kind: NativeNodeFailureKind::Failure,
                        retryable: false,
                    });
                }
            }
        }
        Err(error) if cleanup_failure.is_none() => {
            cleanup_failure = Some(NativeNodeFailure {
                code: "shader_preview_rollback_failed".to_owned(),
                message: format!("GLSLShader preview service disappeared during rollback: {error}"),
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            });
        }
        Err(_) => {}
    }
    cleanup_failure.map_or(Ok(()), Err)
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
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
            code: "invalid_shader_image_handle".to_owned(),
            message: format!("GLSLShader IMAGE handle is unavailable: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn shader_preview_failure(error: NativeShaderPreviewError) -> NativeNodeFailure {
    match error {
        NativeShaderPreviewError::Shader(error) => shader_service_failure(error),
        NativeShaderPreviewError::Preview(error) => NativeNodeFailure {
            code: "shader_preview_failed".to_owned(),
            message: format!("GLSLShader preview preparation failed: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
        NativeShaderPreviewError::Effect(NativeEffectServiceError::Cancelled) => {
            interrupted_failure()
        }
        NativeShaderPreviewError::Effect(error) => NativeNodeFailure {
            code: "shader_preview_effect_failed".to_owned(),
            message: format!("GLSLShader preview effect failed: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: matches!(error, NativeEffectServiceError::Unavailable),
        },
    }
}

fn shader_service_failure(error: NativeShaderServiceError) -> NativeNodeFailure {
    match error {
        NativeShaderServiceError::Shader(NativeShaderError::Cancelled) => interrupted_failure(),
        NativeShaderServiceError::Unavailable | NativeShaderServiceError::Contract(_) => {
            NativeNodeFailure {
                code: "native_shader_service_unavailable".to_owned(),
                message: format!("GLSLShader native service is unavailable: {error}"),
                kind: NativeNodeFailureKind::Failure,
                retryable: true,
            }
        }
        NativeShaderServiceError::Shader(shader_error) => NativeNodeFailure {
            code: "native_shader_execution_failed".to_owned(),
            message: format!("GLSLShader native execution failed: {shader_error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: matches!(
                shader_error,
                NativeShaderError::BackendUnavailable(_) | NativeShaderError::DeviceLost(_)
            ),
        },
        NativeShaderServiceError::InvalidProjection => NativeNodeFailure {
            code: "invalid_shader_result".to_owned(),
            message: "GLSLShader native service returned an invalid result projection".to_owned(),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
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

fn invalid_image(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_shader_image".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn resource_failure(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "shader_resource_exhausted".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
    }
}

fn interrupted_failure() -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: "GLSLShader execution was interrupted".to_owned(),
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
        NativeOutputEffectRequest, NativePreparedEffectKind, NativePreparedEffectService,
        NativeResolvedPayload, NativeResolvedPayloadRetention,
    };
    use sim_tensor::{
        CpuBackend, CpuWorkspaceAuthority, ExecutionContext, ImageTensor, NativeShaderExecutor,
        NativeShaderResult, StreamId,
    };
    use sim_types::{AttemptId, NodeId, PromptId};
    use serde_json::json;
    use std::{
        error::Error,
        sync::{
            Mutex,
            atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
        },
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-shader-comfy-node-0211/fixture.json"
    ));

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        publish_calls: AtomicUsize,
        fail_publish_call: AtomicUsize,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(attempt_id: AttemptId, store: u128, generation: u128) -> Result<Arc<Self>, Box<dyn Error>> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(store),
                    Uuid::from_u128(generation),
                )?,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                publish_calls: AtomicUsize::new(0),
                fail_publish_call: AtomicUsize::new(usize::MAX),
                values: Mutex::new(BTreeMap::new()),
            }))
        }

        fn fail_next_output_at(&self, offset: usize) {
            let current = self.publish_calls.load(AtomicOrdering::Acquire);
            self.fail_publish_call
                .store(current.saturating_add(offset), AtomicOrdering::Release);
        }

        fn value_count(&self) -> Result<usize, Box<dyn Error>> {
            Ok(self
                .values
                .lock()
                .map_err(|_| "test store lock was poisoned")?
                .len())
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
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned()))?
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
            let call = self.publish_calls.fetch_add(1, AtomicOrdering::AcqRel) + 1;
            if call == self.fail_publish_call.load(AtomicOrdering::Acquire) {
                return Err(NativeHandleStoreError::Rejected("injected publish failure".to_owned()));
            }
            payload.validate()?;
            let handle_type = payload.handle_type()?;
            let digest = payload.digest_sha256();
            let identifier = format!("shader-{call}-{}", self.next_identifier.fetch_add(1, AtomicOrdering::AcqRel));
            self.values
                .lock()
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned()))?
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
                .map_err(|_| NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned()))?
                .remove(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            Ok(())
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct CapturedRequest {
        image_count: usize,
        floats: Vec<f32>,
        ints: Vec<i32>,
        bools: Vec<bool>,
        curves: Vec<Vec<f32>>,
        width: u32,
        height: u32,
    }

    #[derive(Debug, Default)]
    struct TestShaderExecutor {
        captured: Mutex<Option<CapturedRequest>>,
    }

    impl NativeShaderExecutor for TestShaderExecutor {
        fn configuration_identity(&self) -> String {
            "test-glsl-shader-v1".to_owned()
        }

        fn execute(
            &self,
            request: &NativeShaderRequest,
            backend: &CpuBackend,
            context: &ExecutionContext<'_>,
        ) -> Result<NativeShaderResult, NativeShaderError> {
            context
                .cancellation
                .check()
                .map_err(|_| NativeShaderError::Cancelled)?;
            let batch = request
                .images
                .first()
                .ok_or_else(|| NativeShaderError::Bounds("missing image".to_owned()))?
                .dimensions()?
                .0;
            *self
                .captured
                .lock()
                .map_err(|_| NativeShaderError::DeviceLost("capture lock poisoned".to_owned()))? =
                Some(CapturedRequest {
                    image_count: request.images.len(),
                    floats: request.floats.clone(),
                    ints: request.ints.clone(),
                    bools: request.bools.clone(),
                    curves: request.curves.clone(),
                    width: request.width,
                    height: request.height,
                });
            let length = usize::try_from(batch)
                .ok()
                .and_then(|batch| batch.checked_mul(request.height as usize))
                .and_then(|length| length.checked_mul(request.width as usize))
                .and_then(|length| length.checked_mul(4))
                .ok_or_else(|| NativeShaderError::Bounds("output length overflowed".to_owned()))?;
            let values = vec![0.5; length];
            let mut outputs = Vec::new();
            for _ in 0..MAX_SHADER_OUTPUTS {
                outputs.push(ImageTensor::from_f32(
                    backend,
                    context,
                    batch,
                    u64::from(request.height),
                    u64::from(request.width),
                    4,
                    &values,
                )?);
            }
            Ok(NativeShaderResult {
                outputs,
                pass_count: 2,
            })
        }
    }

    #[derive(Debug)]
    struct TestEffects {
        identity: NativeNodeServiceIdentity,
        next_transaction: AtomicU64,
        prepared: Mutex<BTreeMap<Uuid, NativePreparedEffectRequest>>,
    }

    impl TestEffects {
        fn new(identity: NativeNodeServiceIdentity) -> Arc<Self> {
            Arc::new(Self {
                identity,
                next_transaction: AtomicU64::new(0x9000),
                prepared: Mutex::new(BTreeMap::new()),
            })
        }

        fn prepared_count(&self) -> Result<usize, Box<dyn Error>> {
            Ok(self
                .prepared
                .lock()
                .map_err(|_| "effect lock was poisoned")?
                .len())
        }
    }

    impl NativePreparedEffectService for TestEffects {
        fn identity(&self) -> &NativeNodeServiceIdentity {
            &self.identity
        }

        fn maximum_output_bytes(&self) -> u64 {
            16 * 1024 * 1024
        }

        fn prepare_output(
            &self,
            request: NativeOutputEffectRequest,
            cancellation: &CancellationToken,
        ) -> Result<NativePreparedEffectRequest, NativeEffectServiceError> {
            cancellation
                .check()
                .map_err(|_| NativeEffectServiceError::Cancelled)?;
            let transaction_id = Uuid::from_u128(u128::from(
                self.next_transaction.fetch_add(1, AtomicOrdering::AcqRel),
            ));
            let prepared = NativePreparedEffectRequest::checked(
                self.identity.service_id(),
                transaction_id,
                NativePreparedEffectKind::Output,
                request.request_digest_sha256(),
            )
            .map_err(|_| NativeEffectServiceError::InvalidRequest)?;
            self.prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .insert(transaction_id, prepared.clone());
            Ok(prepared)
        }

        fn rollback_prepared(
            &self,
            request: &NativePreparedEffectRequest,
        ) -> Result<(), NativeEffectServiceError> {
            self.prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .remove(&request.transaction_id())
                .ok_or(NativeEffectServiceError::InvalidTicket)?;
            Ok(())
        }

        fn rollback_all_prepared(&self) -> Result<(), NativeEffectServiceError> {
            self.prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .clear();
            Ok(())
        }
    }

    struct Harness {
        store: Arc<TestStore>,
        effects: Arc<TestEffects>,
        shader: Arc<TestShaderExecutor>,
        backend: Arc<CpuBackend>,
        workspace: CpuWorkspaceAuthority,
        attempt_id: AttemptId,
        node_id: NodeId,
    }

    impl Harness {
        fn new(store: u128, generation: u128) -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x2101));
            let node_id = NodeId::from("glsl-shader");
            let identity = NativeNodeServiceIdentity::checked(
                Uuid::from_u128(0x2102),
                attempt_id,
                node_id.clone(),
            )?;
            let (backend, workspace) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
            Ok(Self {
                store: TestStore::new(attempt_id, store, generation)?,
                effects: TestEffects::new(identity),
                shader: Arc::new(TestShaderExecutor::default()),
                backend: Arc::new(backend),
                workspace,
                attempt_id,
                node_id,
            })
        }

        fn context(&self, cancellation: CancellationToken) -> Result<NativeNodeContext, Box<dyn Error>> {
            let scratch = self.workspace.authorize_workspace(8 * 1024 * 1024)?;
            let identity = self.effects.identity().clone();
            let compute = NativeNodeComputeSession::checked(
                identity,
                self.backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            let services = NativeNodeServices::checked(
                None,
                Some(self.effects.clone()),
                Some(compute),
            )?
            .with_shader(self.shader.clone());
            Ok(NativeNodeContext::new_with_services(
                PromptId(Uuid::from_u128(0x2103)),
                self.attempt_id,
                self.node_id.clone(),
                cancellation,
                scratch,
                self.store.clone(),
                services,
            )?)
        }

        fn image(&self) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            let cancellation = CancellationToken::default();
            let context = self.context(cancellation.clone())?;
            let execution = context.compute_session()?.execution_context(&context)?;
            let image = ImageTensor::from_f32(
                &self.backend,
                &execution,
                1,
                2,
                3,
                3,
                &[0.25; 18],
            )?;
            let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)?;
            Ok(self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(payload)),
                &cancellation,
            )?)
        }
    }

    fn node() -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        let binding = native_node_bindings()?.into_iter().next().ok_or("missing binding")?;
        match binding {
            NativeNodeBinding::Executable { node, .. } => Ok(node),
            _ => Err("GLSLShader is not executable".into()),
        }
    }

    fn inputs(image: NativeOpaqueHandle) -> BTreeMap<String, NativeValue> {
        BTreeMap::from([
            (
                "fragment_shader".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String(DEFAULT_FRAGMENT_SHADER.to_owned()),
                },
            ),
            (
                "size_mode".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "COMFY_DYNAMICCOMBO_V3".to_owned(),
                    value: json!({"size_mode": "from_input"}),
                },
            ),
            ("image1".to_owned(), NativeValue::Handle { value: image }),
            (
                "u_float2".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(0.75),
                },
            ),
            (
                "u_float10".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(0.25),
                },
            ),
            (
                "u_int1".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Integer(7),
                },
            ),
            (
                "u_bool1".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Boolean(true),
                },
            ),
            (
                "u_curve1".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "CURVE".to_owned(),
                    value: json!({
                        "points": [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
                        "interpolation": "monotone_cubic"
                    }),
                },
            ),
        ])
    }

    #[test]
    fn exact_schema_and_fixture_are_bound() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(fixture["stable_task_id"], "comfy-parity-native-nodes-image-shader-comfy-node-0211");
        assert_eq!(fixture["source"]["sha256"], "fd4725172fe84e5ea3b9274ceddb44fefedaa27815fb6c9bc2689862760bda9c");
        let binding = native_node_bindings()?.into_iter().next().ok_or("missing binding")?;
        binding.validate()?;
        let descriptor = binding.descriptor();
        descriptor.validate_exact_schema_v2()?;
        assert_eq!(descriptor.class_type, CLASS_TYPE);
        assert_eq!(descriptor.dynamic_inputs.len(), 5);
        assert_eq!(descriptor.outputs.len(), MAX_SHADER_OUTPUTS);
        assert_eq!(descriptor.effect, NativeEffectClass::ExclusiveDevice);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        assert!(binding.presentation().is_experimental);
        assert_eq!(binding.presentation().output_names, OUTPUT_NAMES.map(str::to_owned));
        Ok(())
    }

    #[test]
    fn request_projection_delegates_once_and_publishes_four_images() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new(0x2110, 0x2111)?;
        let image = harness.image()?;
        let outcome = futures::executor::block_on(node()?.execute(
            harness.context(CancellationToken::default())?,
            inputs(image),
        ))?;
        let NativeNodeOutcome::Values { outputs, ui, effects } = outcome else {
            return Err("unexpected shader outcome".into());
        };
        assert_eq!(outputs.len(), MAX_SHADER_OUTPUTS);
        assert!(outputs.iter().all(|output| matches!(output, NativeValue::Handle { value } if value.handle_type().type_id == "IMAGE")));
        assert_eq!(effects.len(), 2);
        let ui = ui.ok_or("missing shader UI")?;
        assert_eq!(ui["input_images"].as_array().map(Vec::len), Some(1));
        assert_eq!(ui["images"].as_array().map(Vec::len), Some(1));
        let captured = harness
            .shader
            .captured
            .lock()
            .map_err(|_| "capture lock was poisoned")?
            .clone()
            .ok_or("shader request was not captured")?;
        assert_eq!(captured.image_count, 1);
        assert_eq!(captured.floats, [0.75, 0.25]);
        assert_eq!(captured.ints, [7]);
        assert_eq!(captured.bools, [true]);
        assert_eq!((captured.width, captured.height), (3, 2));
        assert_eq!(captured.curves.len(), 1);
        assert_eq!(captured.curves[0].len(), CURVE_LUT_SAMPLES);
        assert_eq!(captured.curves[0].first().copied(), Some(0.0));
        assert_eq!(captured.curves[0].last().copied(), Some(1.0));
        assert_eq!(harness.store.value_count()?, 5);
        Ok(())
    }

    #[test]
    fn invalid_cancelled_and_partial_publication_paths_are_atomic() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new(0x2120, 0x2121)?;
        let image = harness.image()?;
        let mut missing = inputs(image.clone());
        missing.remove("image1");
        let error = node()?.cache_change_token(&missing).expect_err("missing image must fail");
        assert_eq!(error.code, "invalid_node_inputs");

        let cancellation = CancellationToken::default();
        assert!(cancellation.cancel());
        let error = futures::executor::block_on(node()?.execute(
            harness.context(cancellation)?,
            inputs(image.clone()),
        ))
        .expect_err("cancelled shader must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.value_count()?, 1);
        assert_eq!(harness.effects.prepared_count()?, 0);

        harness.store.fail_next_output_at(3);
        let error = futures::executor::block_on(node()?.execute(
            harness.context(CancellationToken::default())?,
            inputs(image),
        ))
        .expect_err("injected publication failure must fail");
        assert_eq!(error.code, "invalid_shader_image_handle");
        assert_eq!(harness.store.value_count()?, 1);
        assert_eq!(harness.effects.prepared_count()?, 0);
        Ok(())
    }

    #[test]
    fn stale_handle_failure_recovers_with_current_generation() -> Result<(), Box<dyn Error>> {
        let stale = Harness::new(0x2130, 0x2131)?;
        let stale_image = stale.image()?;
        let current = Harness::new(0x2130, 0x2132)?;
        let error = futures::executor::block_on(node()?.execute(
            current.context(CancellationToken::default())?,
            inputs(stale_image),
        ))
        .expect_err("stale image must fail");
        assert_eq!(error.code, "invalid_shader_image_handle");
        assert_eq!(current.store.value_count()?, 0);

        let image = current.image()?;
        let outcome = futures::executor::block_on(node()?.execute(
            current.context(CancellationToken::default())?,
            inputs(image),
        ))?;
        let NativeNodeOutcome::Values { outputs, .. } = outcome else {
            return Err("unexpected recovery outcome".into());
        };
        assert_eq!(outputs.len(), MAX_SHADER_OUTPUTS);
        Ok(())
    }
}
