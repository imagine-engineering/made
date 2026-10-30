use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TransportLimits, TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.openai";
pub const SOURCE_SHA256: &str = "d8d049f380d4e781dd07d3af1d4b67e798f1260d1fa774e362fc415ead22b697";
pub const SORA_SOURCE_SHA256: &str =
    "397c8f803454e98d9723b3985a4df00abcde942030a8a5cd497f3a5f921e20f5";
pub const ALIASES: &[&str] = &[];
pub const MAXIMUM_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
const MAXIMUM_POLLS: u32 = 480;
const MAXIMUM_TOTAL_POLLS: u32 = 4096;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0458",
        node_identifier: "OpenAIChatConfig",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0459",
        node_identifier: "OpenAIChatNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0460",
        node_identifier: "OpenAIDalle2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0461",
        node_identifier: "OpenAIDalle3",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0462",
        node_identifier: "OpenAIGPTImage1",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0463",
        node_identifier: "OpenAIGPTImageNodeV2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0464",
        node_identifier: "OpenAIInputFiles",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0465",
        node_identifier: "OpenAIVideoSora2",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0120",
        method: "POST",
        path: "/proxy/openai/images/edits",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0121",
        method: "POST",
        path: "/proxy/openai/images/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0122",
        method: "POST",
        path: "/proxy/openai/v1/responses",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0123",
        method: "POST",
        path: "/proxy/openai/v1/videos",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0124",
        method: "GET",
        path: "/proxy/openai/v1/videos/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0125",
        method: "GET",
        path: "/proxy/openai/v1/videos/{initial_response.id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0126",
        method: "GET",
        path: "/proxy/openai/v1/videos/{initial_response.id}/content",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0226",
        method: "GET",
        path: "/proxy/openai/v1/responses/{response_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum OpenAiError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    Http {
        status: u16,
        retry_after: Option<String>,
        rate_limited: bool,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for OpenAiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "OpenAi job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "OpenAi HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for OpenAiError {}
impl From<ConformanceError> for OpenAiError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for OpenAiError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, OpenAiError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(OpenAiError::Input("Missing source string field"))
}
fn integer(value: &Value, key: &str) -> Result<i64, OpenAiError> {
    match value.get(key) {
        Some(Value::Number(number)) => number.as_i64().or_else(|| {
            number
                .as_f64()
                .filter(|number| {
                    number.is_finite()
                        && number.fract() == 0.0
                        && *number >= i64::MIN as f64
                        && *number < i64::MAX as f64
                })
                .map(|number| number as i64)
        }),
        Some(Value::String(value)) => python_strip(value).parse().ok(),
        Some(Value::Bool(value)) => Some(i64::from(*value)),
        _ => None,
    }
    .ok_or(OpenAiError::Input("Missing source integer field"))
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageInput {
    pub role: String,
    pub batch: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default = "rgb_channels")]
    pub channels: u32,
    #[serde(default)]
    pub null: bool,
}
fn rgb_channels() -> u32 {
    3
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MaskInput {
    pub batch: u32,
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    pub images: Vec<ImageInput>,
    pub mask: Option<MaskInput>,
    pub advanced_options: Option<Value>,
    pub files: Option<Vec<Value>>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparationPurpose {
    SourcePng,
    OverlayMaskAlpha,
    ZeroRgbMaskAlpha,
    ChatDataUri,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaRequirement {
    pub role: String,
    pub batch_index: u32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub purpose: PreparationPurpose,
    pub field: String,
    pub file_name: String,
    pub total_pixels: u64,
}
#[derive(Clone, Debug)]
pub struct PreparedMedia {
    pub requirement: MediaRequirement,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Debug, Default)]
pub struct PreparedInputs {
    pub media: Vec<PreparedMedia>,
    pub image_contents: Vec<Value>,
    pub file_content: Option<Value>,
}
pub trait OpenAiInputOwner {
    // Encoding/mask math/file reads require the caller's real tensor, Model and
    // filesystem authority. No metadata-only native result can satisfy this.
    fn prepare(
        &mut self,
        _: &Input,
        _: &[MediaRequirement],
    ) -> Result<PreparedInputs, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn prompt(input: &Input) -> Result<&str, OpenAiError> {
    string(&input.settings, "prompt")
}
fn is_local(node: &str) -> bool {
    matches!(node, "OpenAIChatConfig" | "OpenAIInputFiles")
}
fn model_settings(input: &Input) -> Result<&Value, OpenAiError> {
    if input.node == "OpenAIGPTImageNodeV2" {
        input
            .settings
            .get("model")
            .filter(|value| value.is_object())
            .ok_or(OpenAiError::Input("Missing source model controls"))
    } else {
        Ok(&input.settings)
    }
}
fn value_string<'a>(value: &'a Value, key: &str, default: &'a str) -> Result<&'a str, OpenAiError> {
    match value.get(key) {
        None => Ok(default),
        Some(Value::String(value)) => Ok(value),
        _ => Err(OpenAiError::Input("Invalid source string control")),
    }
}
fn value_integer(value: &Value, key: &str, default: i64) -> Result<i64, OpenAiError> {
    if value.get(key).is_none() {
        Ok(default)
    } else {
        integer(value, key)
    }
}
fn image_size(input: &Input) -> Result<String, OpenAiError> {
    let settings = model_settings(input)?;
    let model = value_string(settings, "model", "gpt-image-1")?;
    let size = value_string(settings, "size", "1024x1024")?;
    if size == "Custom" {
        if input.node == "OpenAIGPTImage1" && model != "gpt-image-2" {
            return Err(OpenAiError::Input(
                "Custom source resolution requires GPT Image2",
            ));
        }
        let width = value_integer(settings, "custom_width", 1024)?;
        let height = value_integer(settings, "custom_height", 1024)?;
        if width <= 0
            || height <= 0
            || width % 16 != 0
            || height % 16 != 0
            || width.max(height) > 3840
            || width.max(height) > 3 * width.min(height)
            || !(655360..=8294400).contains(
                &width
                    .checked_mul(height)
                    .ok_or(ConformanceError::BodyLimit)?,
            )
        {
            return Err(OpenAiError::Input("Invalid source custom resolution"));
        }
        return Ok(format!("{width}x{height}"));
    }
    if input.node == "OpenAIGPTImage1"
        && matches!(model, "gpt-image-1" | "gpt-image-1.5")
        && !matches!(size, "auto" | "1024x1024" | "1024x1536" | "1536x1024")
    {
        return Err(OpenAiError::Input("Unsupported source model resolution"));
    }
    Ok(size.into())
}
pub fn media_requirements(input: &Input) -> Result<Vec<MediaRequirement>, OpenAiError> {
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(OpenAiError::Input("Unknown OpenAI node"));
    }
    if !is_local(&input.node) {
        prompt(input)?;
    }
    let images: Vec<_> = input.images.iter().filter(|image| !image.null).collect();
    let image_count = images.iter().try_fold(0u32, |count, image| {
        count
            .checked_add(image.batch)
            .ok_or(ConformanceError::BodyLimit)
    })?;
    if image_count > 128 {
        return Err(ConformanceError::BodyLimit.into());
    }
    if input.node == "OpenAIDalle2" && (images.is_empty() == input.mask.is_some()) {
        return Err(OpenAiError::Input(
            "Dall-E2 source edit requires image and mask",
        ));
    }
    if matches!(
        input.node.as_str(),
        "OpenAIGPTImage1" | "OpenAIGPTImageNodeV2"
    ) {
        if input.mask.is_some() && image_count == 0 {
            return Err(OpenAiError::Input("Mask requires source image"));
        }
        let settings = model_settings(input)?;
        let model = value_string(settings, "model", "gpt-image-1")?;
        image_size(input)?;
        if !matches!(model, "gpt-image-1" | "gpt-image-1.5" | "gpt-image-2") {
            return Err(OpenAiError::Input("Unknown source image model"));
        }
        if input.node == "OpenAIGPTImage1"
            && model == "gpt-image-2"
            && value_string(settings, "background", "opaque")? == "transparent"
        {
            return Err(OpenAiError::Input(
                "Source legacy GPT Image2 forbids transparent background",
            ));
        }
    }
    if input.node == "OpenAIVideoSora2" {
        if value_string(&input.settings, "model", "sora-2")? == "sora-2"
            && !matches!(
                value_string(&input.settings, "size", "1280x720")?,
                "720x1280" | "1280x720"
            )
        {
            return Err(OpenAiError::Input("Unsupported source Sora2 resolution"));
        }
        if !images.is_empty() && image_count != 1 {
            return Err(OpenAiError::Input("Source Sora supports one image"));
        }
    }
    if input.mask.is_some() && image_count != 1 {
        return Err(OpenAiError::Input("Source mask requires one image"));
    }
    if let Some(mask) = &input.mask {
        let image = images
            .first()
            .ok_or(OpenAiError::Input("Missing source mask image"))?;
        if mask.batch != 1 || mask.width != image.width || mask.height != image.height {
            return Err(OpenAiError::Input("Source mask dimensions differ"));
        }
    }
    let mut requirements = vec![];
    if input.node == "OpenAIDalle3" || is_local(&input.node) {
        return Ok(requirements);
    }
    for image in images {
        if image.width == 0
            || image.height == 0
            || !(1..=4).contains(&image.channels)
            || image.role.is_empty()
        {
            return Err(OpenAiError::Input(
                "Invalid canonical source image dimensions",
            ));
        }
        if input.node == "OpenAIDalle2"
            && (image.batch != 1 || image.width == 1 || image.height == 1 || image.channels == 1)
        {
            return Err(OpenAiError::Input("Source Dall-E2 squeeze dimensions"));
        }
        for index in 0..image.batch {
            let ordinal = requirements.len();
            let (field, file_name) = match input.node.as_str() {
                "OpenAIDalle2" => ("image".into(), "image.png".into()),
                "OpenAIVideoSora2" => ("input_reference".into(), "image.png".into()),
                "OpenAIChatNode" => ("input_image".into(), format!("image_{ordinal}.png")),
                _ => (
                    if image_count == 1 {
                        "image".into()
                    } else {
                        "image[]".into()
                    },
                    format!("image_{ordinal}.png"),
                ),
            };
            requirements.push(MediaRequirement {
                role: image.role.clone(),
                batch_index: index,
                width: image.width,
                height: image.height,
                channels: if input.node == "OpenAIDalle2" {
                    4
                } else {
                    image.channels
                },
                purpose: if input.node == "OpenAIChatNode" {
                    PreparationPurpose::ChatDataUri
                } else if input.node == "OpenAIDalle2" {
                    PreparationPurpose::OverlayMaskAlpha
                } else {
                    PreparationPurpose::SourcePng
                },
                field,
                file_name,
                total_pixels: 2048 * 2048,
            });
        }
    }
    if let Some(mask) = &input.mask
        && input.node != "OpenAIDalle2"
    {
        requirements.push(MediaRequirement {
            role: "mask".into(),
            batch_index: 0,
            width: mask.width,
            height: mask.height,
            channels: 4,
            purpose: PreparationPurpose::ZeroRgbMaskAlpha,
            field: "mask".into(),
            file_name: "mask.png".into(),
            total_pixels: 2048 * 2048,
        });
    }
    Ok(requirements)
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct FormField {
    pub name: String,
    pub value: String,
}
#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub path: String,
    pub body: Value,
    pub form_fields: Vec<FormField>,
    pub multipart: bool,
    pub requirements: Vec<MediaRequirement>,
    pub image_model: Option<String>,
    pub estimated_duration_seconds: Option<u64>,
}
fn fields(body: &Value, order: &[&str]) -> Result<Vec<FormField>, OpenAiError> {
    order
        .iter()
        .filter_map(|key| body.get(*key).map(|value| (*key, value)))
        .map(|(name, value)| {
            Ok(FormField {
                name: name.into(),
                value: match value {
                    Value::String(value) => value.clone(),
                    Value::Bool(true) => "True".into(),
                    Value::Bool(false) => "False".into(),
                    Value::Number(value) => value.to_string(),
                    _ => return Err(OpenAiError::Input("Unsupported source multipart scalar")),
                },
            })
        })
        .collect()
}
fn validate_content(value: &Value, kind: &str) -> Result<Value, OpenAiError> {
    let mut result = json!({"type":string(value,"type")?});
    for key in if kind == "image" {
        vec!["detail", "file_id", "image_url"]
    } else {
        vec!["file_data", "file_id", "filename"]
    } {
        match value.get(key) {
            None | Some(Value::Null) => {}
            Some(Value::String(text)) => result[key] = json!(text),
            _ => return Err(OpenAiError::Input("Invalid source input content string")),
        }
    }
    if kind == "image" {
        string(value, "detail")?;
    }
    Ok(result)
}
pub fn local_value(input: &Input, prepared: &PreparedInputs) -> Result<Value, OpenAiError> {
    match input.node.as_str() {
        "OpenAIChatConfig" => {
            let mut value = json!({});
            if input
                .settings
                .get("truncation")
                .is_some_and(|value| !value.is_null())
            {
                value["truncation"] = json!(string(&input.settings, "truncation")?);
            }
            if input
                .settings
                .get("instructions")
                .is_some_and(|value| !value.is_null())
            {
                value["instructions"] = json!(string(&input.settings, "instructions")?);
            }
            if input
                .settings
                .get("max_output_tokens")
                .is_some_and(|value| !value.is_null())
            {
                value["max_output_tokens"] = json!(integer(&input.settings, "max_output_tokens")?);
            }
            Ok(value)
        }
        "OpenAIInputFiles" => {
            let file = prepared
                .file_content
                .as_ref()
                .ok_or(ConformanceError::HostFailure)?;
            let mut values = vec![validate_content(file, "file")?];
            if let Some(chain) = &input.files {
                for file in chain {
                    values.push(validate_content(file, "file")?);
                }
            }
            Ok(json!(values))
        }
        _ => Err(OpenAiError::Input("Node is not a source local helper")),
    }
}
pub fn request_plan(input: &Input, prepared: &PreparedInputs) -> Result<RequestPlan, OpenAiError> {
    let requirements = media_requirements(input)?;
    let settings = model_settings(input)?;
    let mut body = json!({});
    let mut multipart = false;
    let mut image_model = None;
    let mut estimate = None;
    let path = match input.node.as_str() {
        "OpenAIDalle2" | "OpenAIDalle3" | "OpenAIGPTImage1" | "OpenAIGPTImageNodeV2" => {
            let model = match input.node.as_str() {
                "OpenAIDalle2" => "dall-e-2",
                "OpenAIDalle3" => "dall-e-3",
                _ => value_string(settings, "model", "gpt-image-1")?,
            };
            body["model"] = json!(model);
            body["prompt"] = json!(prompt(input)?);
            body["size"] = json!(if model.starts_with("gpt-image") {
                image_size(input)?
            } else {
                value_string(settings, "size", "1024x1024")?.into()
            });
            if input.node != "OpenAIDalle3" {
                body["n"] = json!(value_integer(&input.settings, "n", 1)?);
            }
            if input.node == "OpenAIDalle3" {
                body["quality"] = json!(value_string(settings, "quality", "standard")?);
                body["style"] = json!(value_string(settings, "style", "natural")?);
            }
            if model.starts_with("gpt-image") {
                body["quality"] = json!(value_string(settings, "quality", "low")?);
                body["background"] = json!(value_string(settings, "background", "opaque")?);
                body["moderation"] = json!("low");
                image_model = Some(model.into());
            }
            multipart =
                input.images.iter().any(|image| !image.null) && input.node != "OpenAIDalle3";
            if multipart {
                "/proxy/openai/images/edits"
            } else {
                "/proxy/openai/images/generations"
            }
        }
        "OpenAIChatNode" => {
            let mut content = vec![json!({"text":prompt(input)?,"type":"input_text"})];
            let image_count = requirements.len();
            if prepared.image_contents.len() != image_count {
                return Err(OpenAiError::Input(
                    "Missing canonical chat data URI preparation",
                ));
            }
            for image in &prepared.image_contents {
                content.push(validate_content(image, "image")?);
            }
            if let Some(files) = &input.files {
                for file in files {
                    content.push(validate_content(file, "file")?);
                }
            }
            body = json!({"input":[{"content":content,"role":"user"}],"store":true,"stream":false,"model":value_string(&input.settings,"model","gpt-5")?,"parallel_tool_calls":true});
            if let Some(config) = &input.advanced_options {
                if config.get("model").is_some_and(|value| !value.is_null()) {
                    return Err(OpenAiError::Input("Source duplicate model argument"));
                }
                for key in ["instructions", "truncation"] {
                    if config.get(key).is_some_and(|value| !value.is_null()) {
                        body[key] = json!(string(config, key)?);
                    }
                }
                if config
                    .get("max_output_tokens")
                    .is_some_and(|value| !value.is_null())
                {
                    body["max_output_tokens"] = json!(integer(config, "max_output_tokens")?);
                }
                for (key, maximum) in [("temperature", 2.0), ("top_p", 1.0)] {
                    if config.get(key).is_some_and(|value| !value.is_null()) {
                        let value = config
                            .get(key)
                            .and_then(Value::as_f64)
                            .filter(|value| value.is_finite() && *value >= 0.0 && *value <= maximum)
                            .ok_or(OpenAiError::Input("Source configuration range"))?;
                        body[key] = json!(value);
                    }
                }
            }
            "/proxy/openai/v1/responses"
        }
        "OpenAIVideoSora2" => {
            let model = value_string(&input.settings, "model", "sora-2")?;
            let duration = value_integer(&input.settings, "duration", 8)?;
            body = json!({"prompt":prompt(input)?,"model":model,"seconds":duration.to_string(),"size":value_string(&input.settings,"size","1280x720")?});
            multipart = true;
            estimate = Some(
                u64::try_from(
                    duration
                        .checked_mul(if model == "sora-2" { 45 } else { 90 })
                        .ok_or(ConformanceError::BodyLimit)?
                        / 4,
                )
                .map_err(|_| OpenAiError::Input("Negative source duration estimate"))?,
            );
            "/proxy/openai/v1/videos"
        }
        _ => return Err(OpenAiError::Input("Local helper has no HTTP request")),
    };
    let form_fields = if !multipart {
        vec![]
    } else if input.node == "OpenAIVideoSora2" {
        fields(&body, &["prompt", "model", "seconds", "size"])?
    } else {
        fields(
            &body,
            &[
                "background",
                "model",
                "moderation",
                "n",
                "output_compression",
                "output_format",
                "prompt",
                "quality",
                "size",
            ],
        )?
    };
    Ok(RequestPlan {
        path: path.into(),
        body,
        form_fields,
        multipart,
        requirements,
        image_model,
        estimated_duration_seconds: estimate,
    })
}
pub fn encode_request(
    plan: &RequestPlan,
    prepared: &PreparedInputs,
    boundary: &str,
) -> Result<Vec<u8>, OpenAiError> {
    if !plan.multipart {
        let bytes = serde_json::to_vec(&plan.body)?;
        if bytes.len() > MAXIMUM_REQUEST_BYTES {
            return Err(ConformanceError::BodyLimit.into());
        }
        return Ok(bytes);
    }
    if prepared.media.len() != plan.requirements.len() {
        return Err(OpenAiError::Input(
            "Missing actual canonical source PNG parts",
        ));
    }
    let mut payload_bytes = plan.form_fields.iter().try_fold(0usize, |total, field| {
        total
            .checked_add(field.value.len())
            .ok_or(ConformanceError::BodyLimit)
    })?;
    if payload_bytes > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    for media in &prepared.media {
        payload_bytes = payload_bytes
            .checked_add(media.bytes.len())
            .ok_or(ConformanceError::BodyLimit)?;
        if payload_bytes > MAXIMUM_REQUEST_BYTES {
            return Err(ConformanceError::BodyLimit.into());
        }
    }
    let mut fields: Vec<_> = plan
        .form_fields
        .iter()
        .map(|field| MultipartField {
            name: field.name.clone(),
            file_name: None,
            media_type: Some("text/plain; charset=utf-8".into()),
            bytes: field.value.as_bytes().to_vec(),
        })
        .collect();
    for (requirement, media) in plan.requirements.iter().zip(&prepared.media) {
        if *requirement != media.requirement || media.bytes.is_empty() {
            return Err(OpenAiError::Input("Altered canonical media preparation"));
        }
        fields.push(MultipartField {
            name: requirement.field.clone(),
            file_name: Some(requirement.file_name.clone()),
            media_type: Some("image/png".into()),
            bytes: media.bytes.clone(),
        });
    }
    let limits = TransportLimits {
        maximum_body_bytes: MAXIMUM_REQUEST_BYTES,
        maximum_chunk_bytes: MAXIMUM_CHUNK_BYTES,
        ..TransportLimits::conformance()
    };
    Ok(MultipartField::encode_checked(&fields, boundary, limits)?)
}
pub trait OpenAiTransport: ProviderTransport {
    fn report_image_price(
        &mut self,
        _handle: StreamHandle,
        _price: &ImagePrice,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn begin_response_deadline(
        &mut self,
        _handle: StreamHandle,
        _timeout_milliseconds: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn report_poll_status(
        &mut self,
        _handle: StreamHandle,
        _status: &str,
        _queued: bool,
        _estimated_duration_seconds: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn retry_after_delay(
        &mut self,
        header: &str,
        fallback_milliseconds: u64,
    ) -> Result<u64, ConformanceError>;
    fn retry_request(
        &mut self,
        handle: StreamHandle,
        delay_milliseconds: u64,
    ) -> Result<StreamHandle, ConformanceError>;
    fn delay(&mut self, handle: StreamHandle, milliseconds: u64) -> Result<(), ConformanceError>;
}
#[derive(Default)]
struct Session {
    invocation: Option<u64>,
    handles: BTreeSet<(u64, u32, u32)>,
}
impl Session {
    fn retain(&mut self, handle: StreamHandle) -> Result<(), OpenAiError> {
        if handle.invocation == 0
            || handle.slot == 0
            || handle.generation == 0
            || self
                .invocation
                .is_some_and(|invocation| invocation != handle.invocation)
            || self.handles.len() >= 8192
            || !self
                .handles
                .insert((handle.invocation, handle.slot, handle.generation))
        {
            return Err(ConformanceError::InvalidHandle.into());
        }
        self.invocation = Some(handle.invocation);
        Ok(())
    }
}
fn endpoint(value: &str) -> Result<bool, OpenAiError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(OpenAiError::Response("Invalid OpenAi response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(OpenAiError::Response("Invalid OpenAi response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(OpenAiError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(OpenAiError::Response("Invalid OpenAi endpoint authority"));
    }
    Ok(false)
}

#[derive(Clone, Copy, Debug)]
pub struct CredentialAvailability {
    pub auth_token: bool,
    pub api_key: bool,
}
fn request_head(
    path: &str,
    method: HttpMethod,
    credentials: CredentialAvailability,
    binary: bool,
) -> Result<RequestHead, OpenAiError> {
    let relative = endpoint(path)?;
    let mut headers = vec![Header {
        name: "Accept".into(),
        value: if binary { "*/*" } else { "application/json" }.into(),
    }];
    let secret_id = if relative {
        let (secret, name, value) = if credentials.auth_token {
            (
                "sim_org.auth_token",
                "Authorization",
                "host-capability:auth_token_sim_org",
            )
        } else if credentials.api_key {
            (
                "sim_org.api_key",
                "X-API-KEY",
                "host-capability:api_key_sim_org",
            )
        } else {
            return Err(OpenAiError::Input("Missing Comfy API credentials"));
        };
        headers.push(Header {
            name: name.into(),
            value: value.into(),
        });
        headers.push(Header {
            name: "Comfy-Env".into(),
            value: "host".into(),
        });
        headers.push(Header {
            name: "Comfy-Usage-Source".into(),
            value: "comfyui-api".into(),
        });
        Some(secret.into())
    } else {
        None
    };
    if method == HttpMethod::Post {
        headers.push(Header {
            name: "Content-Type".into(),
            value: "application/json".into(),
        });
    }
    Ok(RequestHead {
        method,
        path: path.into(),
        secret_id,
        headers,
    })
}

#[derive(Clone, Copy)]
struct RetryPolicy {
    ordinary_limit: u32,
    rate_limit: u32,
    backoff: f64,
    download: bool,
    timeout_milliseconds: Option<u64>,
}
const GENERATION_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 3,
    rate_limit: 16,
    backoff: 2.0,
    download: false,
    timeout_milliseconds: Some(3_600_000),
};
const POLL_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 10,
    rate_limit: 16,
    backoff: 1.4,
    download: false,
    timeout_milliseconds: Some(120_000),
};
const DOWNLOAD_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 5,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
    timeout_milliseconds: None,
};
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    session: &mut Session,
    transport: &mut dyn OpenAiTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), OpenAiError> {
    transport.check_cancelled(None)?;
    let handle = transport.start_request(head)?;
    session.retain(handle)?;
    exchange_existing(
        head,
        body,
        maximum_bytes,
        policy,
        handle,
        session,
        transport,
    )
}
fn exchange_existing(
    _head: &RequestHead,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    mut handle: StreamHandle,
    session: &mut Session,
    transport: &mut dyn OpenAiTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), OpenAiError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(OpenAiError::Http {
                status,
                retry_after,
                rate_limited,
            }) => {
                let delay = if (status == 429 || rate_limited)
                    && !policy.download
                    && rate_limit_retries < policy.rate_limit
                {
                    let delay = (1000.0 * policy.backoff.powi(rate_limit_retries as i32))
                        .round()
                        .min(30_000.0) as u64;
                    rate_limit_retries += 1;
                    delay
                } else if (matches!(status, 408 | 500 | 502 | 503 | 504)
                    || policy.download && status == 429)
                    && ordinary_retries < policy.ordinary_limit
                {
                    let delay =
                        (1000.0 * policy.backoff.powi(ordinary_retries as i32)).round() as u64;
                    ordinary_retries += 1;
                    delay
                } else {
                    return Err(OpenAiError::Http {
                        status,
                        retry_after,
                        rate_limited,
                    });
                };
                let delay = if !policy.download {
                    match retry_after {
                        Some(value) => transport.retry_after_delay(&value, delay)?.min(150_000),
                        None => delay,
                    }
                } else {
                    delay
                };
                transport.check_cancelled(Some(handle))?;
                let fresh = transport.retry_request(handle, delay)?;
                session.retain(fresh)?;
                handle = fresh;
            }
            result => return result,
        }
    }
}

fn exchange_on_handle(
    handle: StreamHandle,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn OpenAiTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), OpenAiError> {
    transport.begin_response_deadline(handle, policy.timeout_milliseconds)?;
    if body.is_empty() {
        transport.check_cancelled(Some(handle))?;
        transport.write_request_chunk(handle, 0, &[], true)?;
    } else {
        let count = body.len().div_ceil(MAXIMUM_CHUNK_BYTES);
        for (sequence, chunk) in body.chunks(MAXIMUM_CHUNK_BYTES).enumerate() {
            transport.check_cancelled(Some(handle))?;
            transport.write_request_chunk(handle, sequence as u64, chunk, sequence + 1 == count)?;
        }
    }
    let mut status = None;
    let mut retry_after = None;
    let mut bytes = Vec::new();
    let mut after_sequence = None;
    let mut frame_sequence = 0u64;
    for _ in 0..4096 {
        transport.check_cancelled(Some(handle))?;
        match transport.wait_response(handle, after_sequence, 1000)? {
            ResponseEvent::TimedOut => {}
            ResponseEvent::Cancelled => return Err(ConformanceError::Cancelled.into()),
            ResponseEvent::Head {
                status: response_status,
                headers,
            } if status.is_none() => {
                if !(100..=599).contains(&response_status)
                    || headers.len() > 16
                    || headers
                        .iter()
                        .map(|header| header.name.len() + header.value.len())
                        .sum::<usize>()
                        > 16 * 1024
                    || headers.iter().any(|header| {
                        header.name.len() + header.value.len() > 16 * 1024
                            || header.name.chars().any(char::is_control)
                            || header.value.chars().any(char::is_control)
                    })
                {
                    return Err(ConformanceError::HeaderLimit.into());
                }
                retry_after = headers
                    .iter()
                    .find(|header| header.name.eq_ignore_ascii_case("Retry-After"))
                    .map(|header| header.value.clone());
                status = Some(response_status);
                after_sequence = Some(frame_sequence);
                frame_sequence += 1;
            }
            ResponseEvent::Chunk(chunk) if status.is_some() => {
                if chunk.len() > MAXIMUM_CHUNK_BYTES {
                    return Err(ConformanceError::ChunkLimit.into());
                }
                if bytes
                    .len()
                    .checked_add(chunk.len())
                    .is_none_or(|length| length > maximum_bytes)
                {
                    return Err(ConformanceError::BodyLimit.into());
                }
                bytes.extend(chunk);
                after_sequence = Some(frame_sequence);
                frame_sequence += 1;
            }
            ResponseEvent::Completed(receipt) if status.is_some() => {
                if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                if !status.is_some_and(|status| (200..300).contains(&status)) {
                    return Err(OpenAiError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
                        rate_limited: false,
                    });
                }
                transport.check_cancelled(Some(handle))?;
                return Ok((bytes, receipt, handle));
            }
            _ => return Err(ConformanceError::InvalidResponse.into()),
        }
    }
    Err(ConformanceError::PollLimit.into())
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImageSource {
    InlineBase64 {
        index: usize,
        encoded: String,
    },
    Download {
        index: usize,
        url: String,
        bytes: Vec<u8>,
        handle: StreamHandle,
        receipt: Vec<u8>,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub enum ResultValue {
    LocalModel(Value),
    Text(String),
    Images(Vec<ImageSource>),
    Video {
        bytes: Vec<u8>,
        url: String,
        handle: StreamHandle,
        receipt: Vec<u8>,
    },
}
#[derive(Clone, Debug)]
pub struct InvocationResult {
    pub node: String,
    pub value: ResultValue,
    pub response_bytes: Vec<u8>,
    pub response_receipt: Vec<u8>,
    pub response_handle: Option<StreamHandle>,
    pub cost_receipt: Vec<u8>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum ImagePrice {
    AvailableUsd(f64),
    Unavailable,
}
pub trait OpenAiOutputOwner {
    // Inline base64 must be selected from the verified retained JSON receipt;
    // a download receipt cannot be invented for bytes embedded in that response.
    fn materialize(
        &mut self,
        _: &InvocationResult,
        _: &[ExpectedOutput],
    ) -> Result<Vec<EncodedOutput>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
pub fn image_price(model: &str, response: &Value) -> ImagePrice {
    let rates = match model {
        "gpt-image-1" => (10., 40.),
        "gpt-image-1.5" => (8., 32.),
        "gpt-image-2" => (8., 30.),
        _ => return ImagePrice::Unavailable,
    };
    let Some(usage) = response.get("usage") else {
        return ImagePrice::Unavailable;
    };
    let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) else {
        return ImagePrice::Unavailable;
    };
    let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) else {
        return ImagePrice::Unavailable;
    };
    ImagePrice::AvailableUsd((input as f64 * rates.0 + output as f64 * rates.1) / 1_000_000.)
}
pub fn chat_text(response: &Value) -> Result<String, OpenAiError> {
    let message = response
        .get("output")
        .and_then(Value::as_array)
        .ok_or(OpenAiError::Response("Missing chat outputs"))?
        .iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .ok_or(OpenAiError::Response("Missing chat message"))?;
    let content = message
        .get("content")
        .and_then(Value::as_array)
        .ok_or(OpenAiError::Response("Missing message content"))?;
    for item in content {
        if item.get("type").and_then(Value::as_str) == Some("output_text") {
            return match item.get("text") {
                Some(Value::String(text)) => Ok(text.clone()),
                None | Some(Value::Null) => Ok("None".into()),
                _ => Err(OpenAiError::Response("Invalid source text field")),
            };
        }
    }
    Ok("No text output found in response".into())
}
fn route_selector(response: &Value) -> Result<String, OpenAiError> {
    let id = string(response, "id")?;
    if id.is_empty()
        || id.len() > 1024
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(OpenAiError::Response("Unsafe retained response selector"));
    }
    Ok(id.into())
}
fn poll_status<'a>(
    response: &'a Value,
    retained_id: &str,
    required_id: bool,
) -> Result<&'a str, OpenAiError> {
    match response.get("id") {
        None | Some(Value::Null) if !required_id => {}
        Some(Value::String(id)) if id == retained_id => {}
        _ => return Err(OpenAiError::Response("Retained response identity changed")),
    }
    match response.get("status") {
        None | Some(Value::Null) => Ok(""),
        Some(Value::String(status)) => Ok(status),
        _ => Err(OpenAiError::Response("Invalid source status")),
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Queued,
    Processing,
    Completed,
}
pub fn task_phase(status: &str, chat: bool) -> Result<Phase, OpenAiError> {
    let status = python_strip(status).to_lowercase();
    if (chat && matches!(status.as_str(), "incomplete" | "completed"))
        || (!chat
            && matches!(
                status.as_str(),
                "succeeded"
                    | "succeed"
                    | "success"
                    | "completed"
                    | "finished"
                    | "done"
                    | "complete"
            ))
    {
        return Ok(Phase::Completed);
    }
    match status.as_str() {
        "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error" => {
            Err(OpenAiError::Provider(status))
        }
        "created" | "queued" | "queueing" | "submitted" | "initializing" | "wait" | "in_queue" => {
            Ok(Phase::Queued)
        }
        _ => Ok(Phase::Processing),
    }
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn OpenAiTransport,
    input_owner: &mut dyn OpenAiInputOwner,
) -> Result<InvocationResult, OpenAiError> {
    transport.check_cancelled(None)?;
    let requirements = media_requirements(input)?;
    let needs_owner = !requirements.is_empty() || input.node == "OpenAIInputFiles";
    let prepared = if needs_owner {
        input_owner.prepare(input, &requirements)?
    } else {
        PreparedInputs::default()
    };
    transport.check_cancelled(None)?;
    if is_local(&input.node) {
        return Ok(InvocationResult {
            node: input.node.clone(),
            value: ResultValue::LocalModel(local_value(input, &prepared)?),
            response_bytes: vec![],
            response_receipt: vec![],
            response_handle: None,
            cost_receipt: vec![],
        });
    }
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let plan = request_plan(input, &prepared)?;
    let boundary = "sim-openai-canonical-boundary";
    let body = encode_request(&plan, &prepared, boundary)?;
    let mut head = request_head(&plan.path, HttpMethod::Post, credentials, false)?;
    if plan.multipart {
        let header = head
            .headers
            .iter_mut()
            .find(|header| header.name == "Content-Type")
            .ok_or(ConformanceError::HeaderLimit)?;
        header.value = format!("multipart/form-data; boundary={boundary}");
    }
    let mut session = Session::default();
    transport.check_cancelled(None)?;
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "openai.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits: maximum_cost_microunits,
        },
    )?;
    if !approval.accepted
        || approval.approved_microunits < maximum_cost_microunits
        || approval.receipt.is_empty()
        || approval.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    let (mut bytes, mut receipt, mut response_handle) = exchange_existing(
        &head,
        &body,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let mut response: Value = serde_json::from_slice(&bytes)?;
    let chat = input.node == "OpenAIChatNode";
    let video = input.node == "OpenAIVideoSora2";
    if chat || video {
        if video && response.get("error").is_some_and(|error| !error.is_null()) {
            return Err(OpenAiError::Response("Source initial video error"));
        }
        let id = route_selector(&response)?;
        let path = format!("{}/{id}", plan.path);
        let poll_head = request_head(&path, HttpMethod::Get, credentials, false)?;
        let mut consumed = 0;
        let mut completed = false;
        for _ in 0..MAXIMUM_TOTAL_POLLS {
            let result = exchange_with_policy(
                &poll_head,
                &[],
                MAXIMUM_RESPONSE_BYTES,
                POLL_RETRY,
                &mut session,
                transport,
            )?;
            bytes = result.0;
            receipt = result.1;
            response_handle = result.2;
            response = serde_json::from_slice(&bytes)?;
            let status = poll_status(&response, &id, video)?;
            let phase = task_phase(status, chat)?;
            let normalized = python_strip(status).to_lowercase();
            transport.report_poll_status(
                response_handle,
                if normalized.is_empty() {
                    "Processing"
                } else {
                    &normalized
                },
                phase == Phase::Queued,
                plan.estimated_duration_seconds,
            )?;
            if phase == Phase::Completed {
                completed = true;
                break;
            }
            transport.check_cancelled(Some(response_handle))?;
            transport.delay(response_handle, if video { 8000 } else { 5000 })?;
            if phase != Phase::Queued {
                consumed += 1;
                if consumed >= MAXIMUM_POLLS {
                    break;
                }
            }
        }
        if !completed {
            return Err(ConformanceError::PollLimit.into());
        }
    }
    let value = if chat {
        ResultValue::Text(chat_text(&response)?)
    } else if video {
        let url = format!("{}/{}/content", plan.path, route_selector(&response)?);
        let head = request_head(&url, HttpMethod::Get, credentials, true)?;
        let (bytes, receipt, handle) = exchange_with_policy(
            &head,
            &[],
            MAXIMUM_MEDIA_BYTES,
            DOWNLOAD_RETRY,
            &mut session,
            transport,
        )?;
        ResultValue::Video {
            bytes,
            url,
            handle,
            receipt,
        }
    } else {
        let data = response
            .get("data")
            .and_then(Value::as_array)
            .filter(|data| !data.is_empty())
            .ok_or(OpenAiError::Response("Missing generated image data"))?;
        if data.len() > 128 {
            return Err(ConformanceError::BodyLimit.into());
        }
        transport.report_image_price(
            response_handle,
            &image_price(plan.image_model.as_deref().unwrap_or(""), &response),
        )?;
        let mut images = vec![];
        let mut total = 0usize;
        for (index, item) in data.iter().enumerate() {
            if let Some(encoded) = item
                .get("b64_json")
                .and_then(Value::as_str)
                .filter(|encoded| !encoded.is_empty())
            {
                total = total
                    .checked_add(encoded.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                images.push(ImageSource::InlineBase64 {
                    index,
                    encoded: encoded.into(),
                });
            } else if let Some(url) = item
                .get("url")
                .and_then(Value::as_str)
                .filter(|url| !url.is_empty())
            {
                if endpoint(url)? {
                    return Err(OpenAiError::Response(
                        "Source image download requires absolute URL",
                    ));
                }
                let head = request_head(url, HttpMethod::Get, credentials, true)?;
                let (bytes, receipt, handle) = exchange_with_policy(
                    &head,
                    &[],
                    MAXIMUM_MEDIA_BYTES,
                    DOWNLOAD_RETRY,
                    &mut session,
                    transport,
                )?;
                total = total
                    .checked_add(bytes.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                images.push(ImageSource::Download {
                    index,
                    url: url.into(),
                    bytes,
                    handle,
                    receipt,
                });
            } else {
                return Err(OpenAiError::Response(
                    "Source image data lacks base64 and URL",
                ));
            }
            if total > MAXIMUM_MEDIA_BYTES {
                return Err(ConformanceError::BodyLimit.into());
            }
        }
        ResultValue::Images(images)
    };
    transport.check_cancelled(Some(response_handle))?;
    Ok(InvocationResult {
        node: input.node.clone(),
        value,
        response_bytes: bytes,
        response_receipt: receipt,
        response_handle: Some(response_handle),
        cost_receipt: approval.receipt,
    })
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, OpenAiError> {
    let name = match node {
        "OpenAIChatConfig" => "OPENAI_CHAT_CONFIG",
        "OpenAIInputFiles" => "OPENAI_INPUT_FILES",
        "OpenAIChatNode" => "STRING",
        "OpenAIVideoSora2" => "VIDEO",
        _ if NODES.iter().any(|claim| claim.node_identifier == node) => "IMAGE",
        _ => return Err(OpenAiError::Input("Unknown source output node")),
    };
    let registry = TypeRegistry::built_in()
        .map_err(|_| OpenAiError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve(name)
        .map_err(|_| OpenAiError::Response("Missing source output identity"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| OpenAiError::Response("Missing source output family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub fn materialize_and_publish(
    result: &InvocationResult,
    transport: &mut dyn OpenAiTransport,
    owner: &mut dyn OpenAiOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), OpenAiError> {
    transport.check_cancelled(result.response_handle)?;
    let expected = expected_outputs(&result.node)?;
    let outputs = owner.materialize(result, &expected)?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(result.response_handle)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}

fn source_oracle() -> Result<Value, ConformanceError> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/openai/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
fn decode_hex(value: &str) -> Result<Vec<u8>, ConformanceError> {
    if !value.len().is_multiple_of(2) || !value.is_ascii() {
        return Err(ConformanceError::InvalidRequest);
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).map_err(|_| ConformanceError::InvalidRequest)?;
            u8::from_str_radix(text, 16).map_err(|_| ConformanceError::InvalidRequest)
        })
        .collect()
}
fn oracle_prepared(input: &Input, case: &Value) -> Result<PreparedInputs, ConformanceError> {
    let requirements = media_requirements(input).map_err(|_| ConformanceError::InvalidRequest)?;
    let mut prepared = PreparedInputs::default();
    if input.node == "OpenAIInputFiles" {
        prepared.file_content = case["output"][0][0]
            .as_object()
            .map(|_| case["output"][0][0].clone());
    }
    let request = &case["requests"][0];
    if input.node == "OpenAIChatNode" {
        prepared.image_contents = request["body"]["input"][0]["content"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?
            .iter()
            .filter(|item| item["type"] == "input_image")
            .cloned()
            .collect();
    } else if !requirements.is_empty() {
        let files = request["files"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if files.len() != requirements.len() {
            return Err(ConformanceError::InvalidRequest);
        }
        for (requirement, file) in requirements.into_iter().zip(files) {
            if file["field"] != requirement.field
                || file["file_name"] != requirement.file_name
                || file["mime_type"] != "image/png"
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let bytes = decode_hex(
                file["bytes_hex"]
                    .as_str()
                    .ok_or(ConformanceError::InvalidRequest)?,
            )?;
            prepared.media.push(PreparedMedia { requirement, bytes });
        }
    }
    Ok(prepared)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture["namespace"] != NAMESPACE
        || fixture["source_sha256"] != SOURCE_SHA256
        || fixture["sora_source_sha256"] != SORA_SOURCE_SHA256
        || fixture["aliases"] != json!(ALIASES)
        || fixture["nodes"] != json!(NODES)
        || fixture["routes"] != json!(ROUTES)
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = source_oracle()?;
    if source["sources"]["comfy_api_nodes/nodes_openai.py"] != SOURCE_SHA256
        || source["sources"]["comfy_api_nodes/nodes_sora.py"] != SORA_SOURCE_SHA256
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let prepared = oracle_prepared(&input, case)?;
        if is_local(&input.node) {
            if local_value(&input, &prepared).map_err(|_| ConformanceError::InvalidRequest)?
                != case["output"][0]
            {
                return Err(ConformanceError::InvalidRequest);
            }
        } else {
            let plan =
                request_plan(&input, &prepared).map_err(|_| ConformanceError::InvalidRequest)?;
            let request = &case["requests"][0];
            if plan.path != request["path"]
                || plan.body != request["body"]
                || json!(plan.form_fields) != request["form_fields"]
                || plan.multipart != (request["content_type"] == "multipart/form-data")
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let bytes = encode_request(&plan, &prepared, "sim-openai-canonical-boundary")
                .map_err(|_| ConformanceError::InvalidRequest)?;
            if plan.multipart
                && bytes
                    != decode_hex(
                        request["wire_hex"]
                            .as_str()
                            .ok_or(ConformanceError::InvalidRequest)?,
                    )?
            {
                return Err(ConformanceError::InvalidRequest);
            }
            if input.node == "OpenAIVideoSora2"
                && json!(plan.estimated_duration_seconds) != case["polls"][0]["estimated_duration"]
            {
                return Err(ConformanceError::InvalidRequest);
            }
            if let Some(price) = request.get("price_usd").and_then(Value::as_f64)
                && image_price(
                    plan.image_model.as_deref().unwrap_or(""),
                    &json!({"usage":{"input_tokens":12,"output_tokens":34}}),
                ) != ImagePrice::AvailableUsd(price)
            {
                return Err(ConformanceError::InvalidCost);
            }
        }
        covered.insert(input.node);
    }
    for case in source["errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let accepted = if is_local(&input.node) {
            local_value(&input, &PreparedInputs::default()).is_ok()
        } else {
            request_plan(&input, &PreparedInputs::default()).is_ok()
        };
        if accepted || case["requests"] != 0 {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    if covered
        != NODES
            .iter()
            .map(|claim| claim.node_identifier.to_owned())
            .collect()
    {
        return Err(ConformanceError::InvalidRequest);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        CostResponse, Progress, SignedPutRequest, StorageCreateRequest, StorageCreateResponse,
    };
    use std::collections::VecDeque;
    struct Response {
        path: String,
        method: HttpMethod,
        body: Vec<u8>,
        status: u16,
        bytes: Vec<u8>,
        headers: Vec<Header>,
        receipt: Vec<u8>,
    }
    struct TestTransport {
        responses: VecDeque<Response>,
        active: Option<(StreamHandle, Response, u64)>,
        next_slot: u32,
        delays: Vec<u64>,
        heads: Vec<RequestHead>,
        costs: usize,
        deadlines: Vec<Option<u64>>,
        statuses: Vec<(String, bool, Option<u64>)>,
        last_head: Option<RequestHead>,
        cancelled: bool,
        reject_cost: bool,
        stale_retry: bool,
        failure: Option<ConformanceError>,
    }
    impl TestTransport {
        fn new(responses: Vec<Response>) -> Self {
            Self {
                responses: responses.into(),
                active: None,
                next_slot: 1,
                delays: vec![],
                heads: vec![],
                costs: 0,
                deadlines: vec![],
                statuses: vec![],
                last_head: None,
                cancelled: false,
                reject_cost: false,
                stale_retry: false,
                failure: None,
            }
        }
        fn finish(&self) -> Result<(), Box<dyn Error>> {
            if !self.responses.is_empty() || self.active.is_some() {
                return Err("unconsumed provider script".into());
            }
            Ok(())
        }
    }
    impl ProviderTransport for TestTransport {
        fn check_cancelled(
            &mut self,
            _handle: Option<StreamHandle>,
        ) -> Result<(), ConformanceError> {
            if self.cancelled {
                return Err(ConformanceError::Cancelled);
            }
            if let Some(error) = &self.failure {
                return Err(error.clone());
            }
            Ok(())
        }
        fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
            if self.active.is_some() {
                return Err(ConformanceError::InvalidHandle);
            }
            let response = self
                .responses
                .pop_front()
                .ok_or(ConformanceError::InvalidRequest)?;
            if head.path != response.path || head.method != response.method {
                return Err(ConformanceError::InvalidRequest);
            }
            if head.path.starts_with("https://")
                && (head.secret_id.is_some()
                    || head.headers.iter().any(|header| {
                        matches!(
                            header.name.as_str(),
                            "Authorization" | "X-API-KEY" | "Comfy-Env" | "Comfy-Usage-Source"
                        )
                    }))
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let handle = StreamHandle {
                invocation: 42,
                slot: self.next_slot,
                generation: 1,
            };
            self.next_slot += 1;
            self.heads.push(head.clone());
            self.last_head = Some(head.clone());
            self.active = Some((handle, response, 0));
            Ok(handle)
        }
        fn write_request_chunk(
            &mut self,
            handle: StreamHandle,
            sequence: u64,
            bytes: &[u8],
            end: bool,
        ) -> Result<(), ConformanceError> {
            let (active, response, _) = self
                .active
                .as_ref()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle
                || sequence != 0
                || !end
                || response.body != bytes
                    && !matches!((serde_json::from_slice::<Value>(&response.body), serde_json::from_slice::<Value>(bytes)), (Ok(expected),Ok(actual)) if expected == actual)
            {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(())
        }
        fn upload(&mut self, _handle: StreamHandle, _bytes: &[u8]) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn create_storage(
            &mut self,
            _handle: StreamHandle,
            _request: &StorageCreateRequest,
        ) -> Result<StorageCreateResponse, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn put_signed(
            &mut self,
            _handle: StreamHandle,
            _request: &SignedPutRequest,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn request_cost(
            &mut self,
            _handle: StreamHandle,
            request: &CostRequest,
        ) -> Result<CostResponse, ConformanceError> {
            self.costs += 1;
            Ok(CostResponse {
                accepted: !self.reject_cost,
                approved_microunits: request.maximum_microunits,
                receipt: b"cost receipt".to_vec(),
            })
        }
        fn report_progress(
            &mut self,
            _handle: StreamHandle,
            _progress: &Progress,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn wait_response(
            &mut self,
            handle: StreamHandle,
            after_sequence: Option<u64>,
            timeout: u64,
        ) -> Result<ResponseEvent, ConformanceError> {
            let (active, response, sequence) = self
                .active
                .as_mut()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle || timeout != 1000 || after_sequence != sequence.checked_sub(1) {
                return Err(ConformanceError::InvalidResponse);
            }
            let event = match *sequence {
                0 => ResponseEvent::Head {
                    status: response.status,
                    headers: response.headers.clone(),
                },
                1 => ResponseEvent::Chunk(response.bytes.clone()),
                2 => ResponseEvent::Completed(response.receipt.clone()),
                _ => return Err(ConformanceError::InvalidResponse),
            };
            *sequence += 1;
            if matches!(event, ResponseEvent::Completed(_)) {
                self.active = None;
            }
            Ok(event)
        }
    }
    impl OpenAiTransport for TestTransport {
        fn report_image_price(
            &mut self,
            _: StreamHandle,
            _: &ImagePrice,
        ) -> Result<(), ConformanceError> {
            Ok(())
        }
        fn begin_response_deadline(
            &mut self,
            handle: StreamHandle,
            timeout_milliseconds: Option<u64>,
        ) -> Result<(), ConformanceError> {
            if self
                .active
                .as_ref()
                .is_none_or(|(active, _, _)| *active != handle)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.deadlines.push(timeout_milliseconds);
            Ok(())
        }

        fn report_poll_status(
            &mut self,
            handle: StreamHandle,
            status: &str,
            queued: bool,
            progress: Option<u64>,
        ) -> Result<(), ConformanceError> {
            if handle.invocation != 42 {
                return Err(ConformanceError::InvalidHandle);
            }
            self.statuses.push((status.into(), queued, progress));
            Ok(())
        }
        fn retry_after_delay(
            &mut self,
            header: &str,
            fallback: u64,
        ) -> Result<u64, ConformanceError> {
            Ok(header
                .parse::<u64>()
                .ok()
                .map_or(fallback, |seconds| seconds.saturating_mul(1000)))
        }
        fn retry_request(
            &mut self,
            handle: StreamHandle,
            delay: u64,
        ) -> Result<StreamHandle, ConformanceError> {
            self.delays.push(delay);
            if self.stale_retry {
                return Ok(handle);
            }
            let head = self
                .last_head
                .clone()
                .ok_or(ConformanceError::InvalidRequest)?;
            self.start_request(&head)
        }
        fn delay(
            &mut self,
            _handle: StreamHandle,
            milliseconds: u64,
        ) -> Result<(), ConformanceError> {
            self.delays.push(milliseconds);
            Ok(())
        }
    }

    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn response(path: &str, method: HttpMethod, body: Vec<u8>, bytes: Vec<u8>) -> Response {
        Response {
            path: path.into(),
            method,
            body,
            bytes,
            status: 200,
            headers: vec![],
            receipt: b"verified test response".to_vec(),
        }
    }
    struct MissingOwner;
    impl OpenAiInputOwner for MissingOwner {}
    impl OpenAiOutputOwner for MissingOwner {}
    struct TestInputOwner {
        prepared: PreparedInputs,
    }
    impl OpenAiInputOwner for TestInputOwner {
        fn prepare(
            &mut self,
            _: &Input,
            requirements: &[MediaRequirement],
        ) -> Result<PreparedInputs, ConformanceError> {
            if !self.prepared.media.is_empty()
                && self
                    .prepared
                    .media
                    .iter()
                    .map(|media| media.requirement.clone())
                    .collect::<Vec<_>>()
                    != requirements
            {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(self.prepared.clone())
        }
    }
    fn source_case(node: &str) -> Result<Value, Box<dyn Error>> {
        source_oracle()?["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .find(|case| case["input"]["node"] == node && case["input"]["images"] == json!([]))
            .cloned()
            .ok_or_else(|| "missing case".into())
    }
    fn generation_script(case: &Value) -> Result<Response, Box<dyn Error>> {
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let prepared = oracle_prepared(&input, case)?;
        let plan = request_plan(&input, &prepared)?;
        Ok(response(
            &plan.path,
            HttpMethod::Post,
            encode_request(&plan, &prepared, "sim-openai-canonical-boundary")?,
            serde_json::to_vec(
                &json!({"data":[{"b64_json":"source-inline-base64","url":"https://must-not-request.invalid/image.png"}],"usage":{"input_tokens":12,"output_tokens":34}}),
            )?,
        ))
    }
    #[test]
    fn exact_pinned_source_requests_helpers_and_multipart() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (index, case) in source["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .enumerate()
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let prepared = oracle_prepared(&input, case)
                .map_err(|error| format!("case {index} {} preparation: {error}", input.node))?;
            if is_local(&input.node) {
                assert_eq!(
                    local_value(&input, &prepared)?,
                    case["output"][0],
                    "case {index} local"
                );
            } else {
                let plan = request_plan(&input, &prepared)
                    .map_err(|error| format!("case {index} {} plan: {error}", input.node))?;
                assert_eq!(
                    plan.body, case["requests"][0]["body"],
                    "case {index} source body"
                );
                assert_eq!(
                    json!(plan.form_fields),
                    case["requests"][0]["form_fields"],
                    "case {index} source field order"
                );
                if plan.multipart {
                    assert_eq!(
                        encode_request(&plan, &prepared, "sim-openai-canonical-boundary")?,
                        decode_hex(
                            case["requests"][0]["wire_hex"]
                                .as_str()
                                .ok_or("missing wire")?
                        )?,
                        "case {index} source wire"
                    );
                }
            }
        }
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/openai/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn all_actual_source_cases_execute_bounded_transport() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (index, case) in source["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .enumerate()
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let prepared = oracle_prepared(&input, case)
                .map_err(|error| format!("case {index} {} preparation: {error}", input.node))?;
            let mut script = vec![];
            if !is_local(&input.node) {
                let plan = request_plan(&input, &prepared)?;
                let body = encode_request(&plan, &prepared, "sim-openai-canonical-boundary")
                    .map_err(|error| format!("case {index} {} encode: {error}", input.node))?;
                let bytes = if input.node == "OpenAIChatNode" {
                    json!({"id":"response-42","status":"queued"})
                } else if input.node == "OpenAIVideoSora2" {
                    json!({"id":"video-42","status":"queued"})
                } else {
                    json!({"data":[{"b64_json":"canonical-source-inline"}],"usage":{"input_tokens":12,"output_tokens":34}})
                };
                script.push(response(
                    &plan.path,
                    HttpMethod::Post,
                    body,
                    serde_json::to_vec(&bytes)?,
                ));
                if input.node == "OpenAIChatNode" {
                    script.push(response("/proxy/openai/v1/responses/response-42",HttpMethod::Get,vec![],serde_json::to_vec(&json!({"id":"response-42","status":"incomplete","output":[{"type":"reasoning","content":[]},{"type":"message","content":[{"type":"other","text":"ignored"},{"type":"output_text","text":"first ☁"},{"type":"output_text","text":"ignored second"}]}]}))?));
                }
                if input.node == "OpenAIVideoSora2" {
                    script.push(response(
                        "/proxy/openai/v1/videos/video-42",
                        HttpMethod::Get,
                        vec![],
                        serde_json::to_vec(&json!({"id":"video-42","status":"completed"}))?,
                    ));
                    script.push(response(
                        "/proxy/openai/v1/videos/video-42/content",
                        HttpMethod::Get,
                        vec![],
                        b"actual video owner remains required".to_vec(),
                    ));
                }
            }
            let mut transport = TestTransport::new(script);
            let result = invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut TestInputOwner { prepared },
            )
            .map_err(|error| format!("case {index} {} invocation: {error}", input.node))?;
            if is_local(&input.node) {
                assert_eq!(
                    result.value,
                    ResultValue::LocalModel(case["output"][0].clone())
                );
                assert_eq!(transport.costs, 0);
                assert!(result.response_handle.is_none());
            } else if input.node == "OpenAIChatNode" {
                assert_eq!(result.value, ResultValue::Text("first ☁".into()));
            } else if input.node == "OpenAIVideoSora2" {
                assert!(matches!(result.value, ResultValue::Video { .. }));
                assert!(
                    transport
                        .heads
                        .last()
                        .is_some_and(|head| head.secret_id.is_some())
                );
            } else {
                assert!(matches!(result.value,ResultValue::Images(ref images) if images.len()==1));
            }
            assert_eq!(transport.costs, usize::from(!is_local(&input.node)));
            transport.finish()?;
        }
        Ok(())
    }
    #[test]
    fn source_chat_first_message_null_and_fallback() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            chat_text(
                &json!({"output":[{"type":"message","content":[{"type":"output_text","text":null},{"type":"output_text","text":"later"}]}]})
            )?,
            "None"
        );
        assert_eq!(
            chat_text(&json!({"output":[{"type":"message","content":[{"type":"other"}]}]}))?,
            "No text output found in response"
        );
        assert!(chat_text(&json!({"output":[{"type":"reasoning","content":[]}]})).is_err());
        assert!(chat_text(&json!({"output":[{"type":"message","content":null}]})).is_err());
        assert_eq!(task_phase(" success ", true)?, Phase::Processing);
        assert_eq!(task_phase(" incomplete ", true)?, Phase::Completed);
        assert_eq!(task_phase("success", false)?, Phase::Completed);
        assert!(task_phase("failed", false).is_err());
        Ok(())
    }
    #[test]
    fn actual_source_chat_response_extraction() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for case in source["chat_response_cases"]
            .as_array()
            .ok_or("missing actual source chat cases")?
        {
            let result = chat_text(&case["response"]);
            if case.get("error_type").is_some() {
                assert!(result.is_err());
            } else {
                assert_eq!(json!(result?), case["output"][0]);
            }
        }
        Ok(())
    }
    #[test]
    fn source_prices_remain_independent_of_authorized_maximum() {
        for (model, value) in [
            ("gpt-image-1", 0.00148),
            ("gpt-image-1.5", 0.001184),
            ("gpt-image-2", 0.001116),
        ] {
            assert_eq!(
                image_price(
                    model,
                    &json!({"usage":{"input_tokens":12,"output_tokens":34}})
                ),
                ImagePrice::AvailableUsd(value)
            );
        }
        assert_eq!(
            image_price("gpt-image-1", &json!({})),
            ImagePrice::Unavailable
        );
        assert_eq!(
            image_price(
                "dall-e-2",
                &json!({"usage":{"input_tokens":12,"output_tokens":34}})
            ),
            ImagePrice::Unavailable
        );
    }
    #[test]
    fn inline_precedence_download_credentials_and_receipt_lineage() -> Result<(), Box<dyn Error>> {
        let case = source_case("OpenAIDalle3")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut generation = generation_script(&case)?;
        generation.bytes = serde_json::to_vec(
            &json!({"data":[{"b64_json":"selected-inline","url":"https://ignored.invalid/x"},{"url":"https://source.fixture.invalid/image.png"}]}),
        )?;
        let png = decode_hex(
            source_oracle()?["response_image_png_hex"]
                .as_str()
                .ok_or("missing PNG")?,
        )?;
        let mut transport = TestTransport::new(vec![
            generation,
            response(
                "https://source.fixture.invalid/image.png",
                HttpMethod::Get,
                vec![],
                png.clone(),
            ),
        ]);
        let result = invoke(
            &input,
            credentials(),
            100,
            &mut transport,
            &mut MissingOwner,
        )?;
        let ResultValue::Images(images) = result.value else {
            return Err("missing images".into());
        };
        assert!(
            matches!(images.first(),Some(ImageSource::InlineBase64{index:0,encoded}) if encoded=="selected-inline")
        );
        assert!(
            matches!(images.get(1),Some(ImageSource::Download{index:1,bytes,handle,receipt,..}) if *bytes==png && Some(*handle)!=result.response_handle && !receipt.is_empty())
        );
        assert!(
            transport
                .heads
                .last()
                .is_some_and(|head| head.secret_id.is_none())
        );
        assert_eq!(transport.deadlines, vec![Some(3_600_000), None]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn rejection_cancellation_stale_retry_and_provider_failure() -> Result<(), Box<dyn Error>> {
        let case = source_case("OpenAIDalle3")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut transport = TestTransport::new(vec![]);
        transport.cancelled = true;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut MissingOwner
            ),
            Err(OpenAiError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        let mut transport = TestTransport::new(vec![generation_script(&case)?]);
        transport.reject_cost = true;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut MissingOwner
            ),
            Err(OpenAiError::Transport(ConformanceError::InvalidCost))
        ));
        let mut initial = generation_script(&case)?;
        initial.status = 500;
        let mut transport = TestTransport::new(vec![initial]);
        transport.stale_retry = true;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut MissingOwner
            ),
            Err(OpenAiError::Transport(ConformanceError::InvalidHandle))
        ));
        let mut initial = generation_script(&case)?;
        initial.status = 429;
        initial.headers = vec![Header {
            name: "Retry-After".into(),
            value: "2".into(),
        }];
        let mut transport = TestTransport::new(vec![initial, generation_script(&case)?]);
        invoke(
            &input,
            credentials(),
            100,
            &mut transport,
            &mut MissingOwner,
        )?;
        assert_eq!(transport.costs, 1);
        assert_eq!(transport.delays, vec![2000]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn nullable_source_poll_metadata_does_not_mint_route_authority() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            poll_status(&json!({"status":null}), "response-42", false)?,
            ""
        );
        assert_eq!(task_phase("", true)?, Phase::Processing);
        assert_eq!(
            poll_status(&json!({"status":"completed"}), "response-42", false)?,
            "completed"
        );
        assert!(
            poll_status(
                &json!({"id":"other","status":"completed"}),
                "response-42",
                false
            )
            .is_err()
        );
        assert!(poll_status(&json!({"status":"completed"}), "video-42", true).is_err());
        assert!(poll_status(&json!({"id":"video-42","status":12}), "video-42", true).is_err());
        Ok(())
    }
    #[test]
    fn changed_poll_identity_and_exact_overridden_completion() -> Result<(), Box<dyn Error>> {
        let case = source_case("OpenAIChatNode")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let plan = request_plan(&input, &PreparedInputs::default())?;
        let script = vec![
            response(
                &plan.path,
                HttpMethod::Post,
                serde_json::to_vec(&plan.body)?,
                serde_json::to_vec(&json!({"id":"response-42","status":"queued"}))?,
            ),
            response(
                "/proxy/openai/v1/responses/response-42",
                HttpMethod::Get,
                vec![],
                serde_json::to_vec(&json!({"id":"other","status":"completed","output":[]}))?,
            ),
        ];
        let mut transport = TestTransport::new(script);
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut MissingOwner
            ),
            Err(OpenAiError::Response("Retained response identity changed"))
        ));
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn canonical_output_ids_and_absent_owners_are_typed() -> Result<(), Box<dyn Error>> {
        for (node, expected) in [
            ("OpenAIChatConfig", "sim:openai-chat-config@1"),
            ("OpenAIInputFiles", "sim:openai-input-files@1"),
        ] {
            assert_eq!(
                expected_outputs(node)?
                    .first()
                    .ok_or("missing output")?
                    .type_id
                    .to_string(),
                expected
            );
        }
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .find(|case| {
                case["input"]["node"] == "OpenAIChatNode"
                    && case["input"]["images"]
                        .as_array()
                        .is_some_and(|images| !images.is_empty())
            })
            .ok_or("missing image case")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut transport = TestTransport::new(vec![]);
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut MissingOwner
            ),
            Err(OpenAiError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        struct Sink(usize);
        impl MaterializationSink for Sink {
            fn publish(
                &mut self,
                _: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.0 += 1;
                Ok(())
            }
        }
        let mut sink = Sink(0);
        let result = InvocationResult {
            node: "OpenAIChatConfig".into(),
            value: ResultValue::LocalModel(json!({})),
            response_bytes: vec![],
            response_receipt: vec![],
            response_handle: None,
            cost_receipt: vec![],
        };
        assert!(
            materialize_and_publish(&result, &mut transport, &mut MissingOwner, &mut sink).is_err()
        );
        assert_eq!(sink.0, 0);
        Ok(())
    }
    #[test]
    fn unsafe_routes_malformed_response_missing_receipt_and_chunk_bounds()
    -> Result<(), Box<dyn Error>> {
        for url in [
            "http://x/y",
            "https://user:secret@x/y",
            "//x/y",
            "https://x/y#fragment",
            "https://x/ bad",
        ] {
            assert!(endpoint(url).is_err());
        }
        let case = source_case("OpenAIDalle3")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        for bytes in [
            b"malformed".to_vec(),
            b"{}".to_vec(),
            b"{\"data\":[]}".to_vec(),
            vec![0; MAXIMUM_CHUNK_BYTES + 1],
        ] {
            let mut initial = generation_script(&case)?;
            initial.bytes = bytes;
            let mut transport = TestTransport::new(vec![initial]);
            assert!(
                invoke(
                    &input,
                    credentials(),
                    100,
                    &mut transport,
                    &mut MissingOwner
                )
                .is_err()
            );
        }
        let mut initial = generation_script(&case)?;
        initial.receipt.clear();
        let mut transport = TestTransport::new(vec![initial]);
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut MissingOwner
            ),
            Err(OpenAiError::Transport(ConformanceError::InvalidResponse))
        ));
        Ok(())
    }
}
