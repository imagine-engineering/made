use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, ProviderTransport, RequestHead, ResponseEvent, StreamHandle, TypeRegistry,
    publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.tripo";
pub const SOURCE_SHA256: &str = "d380b5bb5a9b07427dc2bf40dd06c745341e08f50ed83338f0c45d05c127a10e";
pub const ALIASES: &[&str] = &[];
pub const MAXIMUM_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_IMPORT_BYTES: u64 = 150 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
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
        feature_id: "COMFY-NODE-0686",
        node_identifier: "TripoConversionNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0687",
        node_identifier: "TripoImageToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0688",
        node_identifier: "TripoImportModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0689",
        node_identifier: "TripoMultiviewToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0690",
        node_identifier: "TripoP1ImageToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0691",
        node_identifier: "TripoP1MultiviewToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0692",
        node_identifier: "TripoP1TextToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0693",
        node_identifier: "TripoRefineNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0694",
        node_identifier: "TripoRetargetNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0695",
        node_identifier: "TripoRigNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0699",
        node_identifier: "TripoTextToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0700",
        node_identifier: "TripoTextureNode",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0183",
        method: "POST",
        path: "/proxy/tripo/v2/openapi/import",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0184",
        method: "POST",
        path: "/proxy/tripo/v2/openapi/task",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0185",
        method: "GET",
        path: "/proxy/tripo/v2/openapi/task/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0186",
        method: "GET",
        path: "/proxy/tripo/v2/openapi/task/{task_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum TripoError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    SourceIndex(&'static str),
    SourceAttributeMissing(&'static str),
    Http {
        status: u16,
        retry_after: Option<String>,
        rate_limited: bool,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for TripoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message)
            | Self::Response(message)
            | Self::SourceIndex(message)
            | Self::SourceAttributeMissing(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "Tripo job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Tripo HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for TripoError {}
impl From<ConformanceError> for TripoError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for TripoError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageInput {
    pub batch: u32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FileInput {
    pub format: String,
    pub stream_kind: String,
    #[serde(default)]
    pub oversized: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    pub images: std::collections::BTreeMap<String, Option<ImageInput>>,
    pub file: Option<FileInput>,
}
#[derive(Clone, Debug)]
pub struct PreparedUpload {
    pub urls: Vec<String>,
    pub receipts: Vec<Vec<u8>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestModel {
    Text,
    Image,
    Multiview,
    Texture,
    Refine,
    Rig,
    Retarget,
    Convert,
    Import,
    P1Text,
    P1Image,
    P1Multiview,
}
impl RequestModel {
    pub fn source_name(self) -> &'static str {
        match self {
            Self::Text => "TripoTextToModelRequest",
            Self::Image => "TripoImageToModelRequest",
            Self::Multiview => "TripoMultiviewToModelRequest",
            Self::Texture => "TripoTextureModelRequest",
            Self::Refine => "TripoRefineModelRequest",
            Self::Rig => "TripoAnimateRigRequest",
            Self::Retarget => "TripoAnimateRetargetRequest",
            Self::Convert => "TripoConvertModelRequest",
            Self::Import => "TripoImportModelRequest",
            Self::P1Text => "TripoP1TextToModelRequest",
            Self::P1Image => "TripoP1ImageToModelRequest",
            Self::P1Multiview => "TripoP1MultiviewToModelRequest",
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageUploadPolicy {
    pub maximum_images: u32,
    pub total_pixels: u64,
    pub content_type: Option<String>,
}
pub trait TripoInputOwner {
    fn verify_inputs(&mut self, _: &Input) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn image_upload(
        &mut self,
        _: &Input,
        _: &str,
        _: &ImageUploadPolicy,
    ) -> Result<PreparedUpload, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn file_format(&mut self, _: &Input) -> Result<String, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn file_byte_length(&mut self, _: &Input) -> Result<u64, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn file_upload(
        &mut self,
        _: &Input,
        _: &str,
        _: &str,
    ) -> Result<PreparedUpload, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn normalize_request(&mut self, _: RequestModel, _: &Value) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub node: String,
    pub path: &'static str,
    pub model: RequestModel,
    pub body: Value,
    pub estimated_seconds: Option<u64>,
    pub input_receipts: Vec<Vec<u8>>,
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, TripoError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(TripoError::Input("Missing source string field"))
}
fn value(settings: &Value, key: &str) -> Value {
    settings.get(key).cloned().unwrap_or(Value::Null)
}
fn default_value(settings: &Value, key: &str, default: Value) -> Value {
    settings.get(key).cloned().unwrap_or(default)
}
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
    }
}
fn python_equals(left: &Value, right: &Value) -> bool {
    match (left.as_f64(), right.as_f64()) {
        (Some(left), Some(right)) => left == right,
        _ => left == right,
    }
}
fn non_sentinel(settings: &Value, key: &str, sentinel: Value) -> Value {
    let raw = value(settings, key);
    if python_equals(&raw, &sentinel) {
        Value::Null
    } else {
        raw
    }
}
fn required(settings: &Value, key: &str) -> Result<Value, TripoError> {
    settings
        .get(key)
        .cloned()
        .ok_or(TripoError::Input("Missing source required field"))
}
fn bound_receipt(receipt: &[u8]) -> Result<(), TripoError> {
    if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
        Err(ConformanceError::InvalidResponse.into())
    } else {
        Ok(())
    }
}
fn upload_url(prepared: PreparedUpload, receipts: &mut Vec<Vec<u8>>) -> Result<String, TripoError> {
    if prepared.urls.is_empty() {
        return Err(TripoError::SourceIndex("image upload URLs[0]"));
    }
    if prepared.urls.len() != 1 || prepared.receipts.len() != 1 || receipts.len() >= 4 {
        return Err(ConformanceError::InvalidResponse.into());
    }
    let url = prepared
        .urls
        .into_iter()
        .next()
        .ok_or(TripoError::SourceIndex("upload URLs[0]"))?;
    if endpoint(&url)? {
        return Err(TripoError::Input("Source upload requires absolute URL"));
    }
    for receipt in prepared.receipts {
        bound_receipt(&receipt)?;
        receipts.push(receipt);
    }
    Ok(url)
}
fn image_file(
    input: &Input,
    port: &str,
    owner: &mut dyn TripoInputOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Value, TripoError> {
    let url = upload_url(
        owner.image_upload(
            input,
            port,
            &ImageUploadPolicy {
                maximum_images: 1,
                total_pixels: 2048 * 2048,
                content_type: None,
            },
        )?,
        receipts,
    )?;
    Ok(json!({"url":url,"type":"jpeg"}))
}
fn p1_common(settings: &Value) -> Result<Value, TripoError> {
    let mode = settings
        .get("output_mode")
        .ok_or(TripoError::Input("Missing source dynamic output mode"))?;
    let geometry = string(mode, "output_mode")? == "Geometry only";
    let mut raw = json!({"model_seed":value(settings,"model_seed"),"face_limit":non_sentinel(settings,"face_limit",json!(-1)),"auto_size":default_value(settings,"auto_size",json!(false)),"export_uv":default_value(settings,"export_uv",json!(true)),"compress":if truthy(&default_value(settings,"compress_geometry",json!(false))){json!("geometry")}else{Value::Null},"texture":!geometry,"pbr":if geometry{false}else{truthy(&default_value(mode,"pbr",json!(true)))}});
    if !geometry {
        raw["texture_quality"] = default_value(mode, "texture_quality", json!("standard"));
        raw["texture_seed"] = value(mode, "texture_seed");
        for field in ["texture_alignment", "orientation"] {
            if let Some(value) = mode.get(field) {
                raw[field] = value.clone();
            }
        }
    }
    Ok(raw)
}
pub fn prepare_request(
    input: &Input,
    owner: &mut dyn TripoInputOwner,
) -> Result<RequestPlan, TripoError> {
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(TripoError::Input("Unknown source node"));
    }
    owner.verify_inputs(input)?;
    let settings = &input.settings;
    let mut receipts = vec![];
    let mut path = "/proxy/tripo/v2/openapi/task";
    let (model, raw, estimate) = match input.node.as_str() {
        "TripoTextToModelNode" => {
            let prompt = required(settings, "prompt")?;
            if !truthy(&prompt) {
                return Err(TripoError::Input("Prompt is required"));
            }
            let mut raw = json!({"type":"text_to_model","prompt":prompt,"negative_prompt":if truthy(&value(settings,"negative_prompt")){value(settings,"negative_prompt")}else{Value::Null},"auto_size":true});
            for field in [
                "model_version",
                "style",
                "texture",
                "pbr",
                "image_seed",
                "model_seed",
                "texture_seed",
                "texture_quality",
                "geometry_quality",
                "quad",
            ] {
                raw[field] = value(settings, field);
            }
            raw["face_limit"] = non_sentinel(settings, "face_limit", json!(-1));
            if raw["style"] == "None" {
                raw["style"] = Value::Null;
            }
            (RequestModel::Text, raw, 80)
        }
        "TripoImageToModelNode" => {
            if input.images.get("image").is_none_or(Option::is_none) {
                return Err(TripoError::Input("Image is required"));
            }
            let file = image_file(input, "image", owner, &mut receipts)?;
            let mut raw = json!({"type":"image_to_model","file":file,"auto_size":true});
            for field in [
                "model_version",
                "style",
                "texture",
                "pbr",
                "model_seed",
                "orientation",
                "texture_seed",
                "texture_quality",
                "geometry_quality",
                "texture_alignment",
                "quad",
            ] {
                raw[field] = value(settings, field);
            }
            raw["face_limit"] = non_sentinel(settings, "face_limit", json!(-1));
            if raw["style"] == "None" {
                raw["style"] = Value::Null;
            }
            (RequestModel::Image, raw, 80)
        }
        "TripoMultiviewToModelNode" | "TripoP1MultiviewToModelNode" => {
            let modern = input.node == "TripoP1MultiviewToModelNode";
            let present = |name: &str| input.images.get(name).is_some_and(Option::is_some);
            if modern {
                if ["image", "image_left", "image_back", "image_right"]
                    .iter()
                    .filter(|name| present(name))
                    .count()
                    < 2
                {
                    return Err(TripoError::Input(
                        "Tripo P1 multiview requires at least 2 images (front plus one of left/back/right).",
                    ));
                }
            } else {
                if !present("image") {
                    return Err(TripoError::Input("front image for multiview is required"));
                }
                if !["image_left", "image_back", "image_right"]
                    .iter()
                    .any(|name| present(name))
                {
                    return Err(TripoError::Input(
                        "At least one of left, back, or right image must be provided for multiview",
                    ));
                }
            }
            let mut files = vec![];
            for port in ["image", "image_left", "image_back", "image_right"] {
                files.push(if present(port) {
                    image_file(input, port, owner, &mut receipts)?
                } else {
                    json!({})
                });
            }
            let mut raw = if modern {
                p1_common(settings)?
            } else {
                json!({"type":"multiview_to_model","quad":null})
            };
            raw["files"] = json!(files);
            if !modern {
                for field in [
                    "model_version",
                    "orientation",
                    "texture",
                    "pbr",
                    "model_seed",
                    "texture_seed",
                    "texture_quality",
                    "geometry_quality",
                    "texture_alignment",
                ] {
                    raw[field] = value(settings, field);
                }
                raw["face_limit"] = non_sentinel(settings, "face_limit", json!(-1));
            }
            (
                if modern {
                    RequestModel::P1Multiview
                } else {
                    RequestModel::Multiview
                },
                raw,
                80,
            )
        }
        "TripoTextureNode" => {
            let prompt = default_value(settings, "texture_prompt", json!(""));
            let prompt = python_strip(
                prompt
                    .as_str()
                    .ok_or(TripoError::Input("Invalid source texture prompt"))?,
            );
            let mut raw = json!({"original_model_task_id":required(settings,"model_task_id")?,"texture_prompt":if prompt.is_empty(){Value::Null}else{json!({"text":prompt})}});
            for field in [
                "texture",
                "pbr",
                "texture_seed",
                "texture_quality",
                "texture_alignment",
            ] {
                raw[field] = value(settings, field);
            }
            (RequestModel::Texture, raw, 80)
        }
        "TripoRefineNode" => (
            RequestModel::Refine,
            json!({"draft_model_task_id":required(settings,"model_task_id")?}),
            240,
        ),
        "TripoRigNode" => (
            RequestModel::Rig,
            json!({"original_model_task_id":required(settings,"original_model_task_id")?,"out_format":"glb","spec":"tripo"}),
            180,
        ),
        "TripoRetargetNode" => (
            RequestModel::Retarget,
            json!({"original_model_task_id":required(settings,"original_model_task_id")?,"animation":required(settings,"animation")?,"out_format":"glb","bake_animation":true}),
            30,
        ),
        "TripoConversionNode" => {
            let id = required(settings, "original_model_task_id")?;
            if !truthy(&id) {
                return Err(TripoError::Input("Original model task ID is required"));
            }
            let mut raw =
                json!({"original_model_task_id":id,"format":required(settings,"format")?});
            for key in [
                "quad",
                "force_symmetry",
                "flatten_bottom",
                "pivot_to_center_bottom",
                "with_animation",
                "pack_uv",
                "bake",
                "export_vertex_colors",
                "animate_in_place",
            ] {
                let value = required(settings, key)?;
                raw[key] = if truthy(&value) { value } else { Value::Null };
            }
            for (key, sentinel) in [
                ("face_limit", json!(-1)),
                ("flatten_bottom_threshold", json!(0.0)),
                ("texture_size", json!(4096)),
                ("texture_format", json!("JPEG")),
                ("scale_factor", json!(1.0)),
                ("fbx_preset", json!("blender")),
                ("export_orientation", json!("default")),
            ] {
                let value = required(settings, key)?;
                raw[key] = if python_equals(&value, &sentinel) {
                    Value::Null
                } else {
                    value
                };
            }
            let names = string(settings, "part_names")?;
            raw["part_names"] = if names.is_empty() || python_strip(names).is_empty() {
                Value::Null
            } else {
                json!(
                    names
                        .split(',')
                        .map(python_strip)
                        .filter(|name| !name.is_empty())
                        .collect::<Vec<_>>()
                )
            };
            (RequestModel::Convert, raw, 30)
        }
        "TripoImportModelNode" => {
            if input.file.is_none() {
                return Err(TripoError::Input("Missing native model file"));
            }
            let format = owner
                .file_format(input)?
                .trim_start_matches('.')
                .to_lowercase();
            if format == "gltf" {
                return Err(TripoError::Input("Source rejects external-reference GLTF"));
            }
            if !["glb", "fbx", "obj", "stl"].contains(&format.as_str()) {
                return Err(TripoError::Input("Unsupported source import format"));
            }
            if owner.file_byte_length(input)? > MAXIMUM_IMPORT_BYTES {
                return Err(TripoError::Input("Source import exceeds 150 MiB"));
            }
            let mime = match format.as_str() {
                "glb" => "model/gltf-binary",
                "obj" => "model/obj",
                _ => "application/octet-stream",
            };
            let url = upload_url(owner.file_upload(input, &format, mime)?, &mut receipts)?;
            path = "/proxy/tripo/v2/openapi/import";
            (RequestModel::Import, json!({"url":url,"format":format}), 10)
        }
        "TripoP1TextToModelNode" => {
            let prompt = required(settings, "prompt")?;
            if !truthy(&prompt) {
                return Err(TripoError::Input("Prompt is required"));
            }
            let mut raw = p1_common(settings)?;
            raw["prompt"] = prompt;
            raw["negative_prompt"] = if truthy(&value(settings, "negative_prompt")) {
                value(settings, "negative_prompt")
            } else {
                Value::Null
            };
            raw["image_seed"] = value(settings, "image_seed");
            (RequestModel::P1Text, raw, 60)
        }
        "TripoP1ImageToModelNode" => {
            if input.images.get("image").is_none_or(Option::is_none) {
                return Err(TripoError::Input("Image is required"));
            }
            let file = image_file(input, "image", owner, &mut receipts)?;
            let mut raw = p1_common(settings)?;
            raw["file"] = file;
            raw["enable_image_autofix"] =
                default_value(settings, "enable_image_autofix", json!(false));
            (RequestModel::P1Image, raw, 60)
        }
        _ => return Err(TripoError::Input("Unsupported source node")),
    };
    let body = owner.normalize_request(model, &raw)?;
    if !body.is_object() || serde_json::to_vec(&body)?.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(RequestPlan {
        node: input.node.clone(),
        path,
        model,
        body,
        estimated_seconds: Some(estimate),
        input_receipts: receipts,
    })
}
#[derive(Default)]
struct Session {
    invocation: Option<u64>,
    handles: BTreeSet<(u64, u32, u32)>,
}
impl Session {
    fn retain(&mut self, handle: StreamHandle) -> Result<(), TripoError> {
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
fn endpoint(value: &str) -> Result<bool, TripoError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(TripoError::Response("Invalid Tripo response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(TripoError::Response("Invalid Tripo response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(TripoError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(TripoError::Response("Invalid Tripo endpoint authority"));
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
) -> Result<RequestHead, TripoError> {
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
            return Err(TripoError::Input("Missing Comfy API credentials"));
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
    if matches!(method, HttpMethod::Post | HttpMethod::Patch) {
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
const DOWNLOAD_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 5,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
    timeout_milliseconds: None,
};
const POLL_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 10,
    rate_limit: 16,
    backoff: 1.4,
    download: false,
    timeout_milliseconds: Some(120_000),
};
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    session: &mut Session,
    transport: &mut dyn TripoTransport,
) -> Result<RetainedResponse, TripoError> {
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
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
    transport: &mut dyn TripoTransport,
) -> Result<RetainedResponse, TripoError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(TripoError::Http {
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
                    return Err(TripoError::Http {
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
    transport: &mut dyn TripoTransport,
) -> Result<RetainedResponse, TripoError> {
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
    let mut retained_headers = vec![];
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
                retained_headers = headers;
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
                    return Err(TripoError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
                        rate_limited: false,
                    });
                }
                transport.check_cancelled(Some(handle))?;
                return Ok(RetainedResponse {
                    bytes,
                    receipt,
                    handle,
                    headers: retained_headers,
                });
            }
            _ => return Err(ConformanceError::InvalidResponse.into()),
        }
    }
    Err(ConformanceError::PollLimit.into())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiModel {
    Task,
}
#[derive(Clone, Debug)]
pub struct RetainedResponse {
    pub handle: StreamHandle,
    pub bytes: Vec<u8>,
    pub headers: Vec<Header>,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct TaskProgress {
    pub status: String,
    pub queued: bool,
    pub percentage: Option<f64>,
    pub source_usd: Option<f64>,
    pub estimated_seconds: Option<u64>,
}
pub trait TripoTransport: ProviderTransport {
    fn validate_response(
        &mut self,
        _: &RetainedResponse,
        _: ApiModel,
        _: &Value,
    ) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn authorize_continuation(
        &mut self,
        _: &RetainedResponse,
        _: &RequestHead,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn begin_response_deadline(
        &mut self,
        _: StreamHandle,
        _: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn source_request_labels(
        &mut self,
        _: StreamHandle,
        _: Option<&str>,
        _: Option<&str>,
        _: bool,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn report_task_progress(
        &mut self,
        _: &RetainedResponse,
        _: &TaskProgress,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn retry_after_delay(&mut self, _: &str, _: u64) -> Result<u64, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn retry_request(&mut self, _: StreamHandle, _: u64) -> Result<StreamHandle, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn delay(&mut self, _: StreamHandle, _: u64) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn normalized(
    response: &RetainedResponse,
    model: ApiModel,
    transport: &mut dyn TripoTransport,
) -> Result<Value, TripoError> {
    transport.check_cancelled(Some(response.handle))?;
    let raw: Value = serde_json::from_slice(&response.bytes)?;
    if !raw.is_object() {
        return Err(TripoError::Response("Source response requires API object"));
    }
    let value = transport.validate_response(response, model, &raw)?;
    transport.check_cancelled(Some(response.handle))?;
    if !value.is_object() {
        return Err(TripoError::Response(
            "Invalid canonical normalized API object",
        ));
    }
    if serde_json::to_vec(&value)?.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(value)
}
fn retain_response(
    responses: &mut Vec<RetainedResponse>,
    response: RetainedResponse,
) -> Result<(), TripoError> {
    let retained_bytes = responses
        .iter()
        .chain(std::iter::once(&response))
        .try_fold(0usize, |total, response| {
            let header_bytes = response.headers.iter().try_fold(0usize, |total, header| {
                total
                    .checked_add(header.name.len())?
                    .checked_add(header.value.len())
            })?;
            total
                .checked_add(response.bytes.len())?
                .checked_add(response.receipt.len())?
                .checked_add(header_bytes)
        })
        .ok_or(ConformanceError::BodyLimit)?;
    if retained_bytes > MAXIMUM_MEDIA_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    responses.push(response);
    Ok(())
}
fn retained_id(value: &Value, key: &str) -> Result<String, TripoError> {
    let id = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(TripoError::Response("Missing source retained task ID"))?;
    if id.is_empty()
        || id.len() > 1024
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(TripoError::Response("Unsafe retained task ID"));
    }
    Ok(id.into())
}
fn bounded_receipt(receipt: &[u8]) -> Result<(), TripoError> {
    if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
        return Err(ConformanceError::InvalidResponse.into());
    }
    Ok(())
}
#[derive(Clone, Debug)]
pub struct CompatibilitySave {
    pub filename: String,
    pub format: &'static str,
}
#[derive(Clone, Debug)]
pub struct InvocationResult {
    pub node: String,
    pub task_id: String,
    pub input_receipts: Vec<Vec<u8>>,
    pub responses: Vec<RetainedResponse>,
    pub downloaded: Option<RetainedResponse>,
    pub compatibility_save: Option<CompatibilitySave>,
    pub cost_receipt: Vec<u8>,
    pub url: Option<String>,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn TripoTransport,
    owner: &mut dyn TripoInputOwner,
) -> Result<InvocationResult, TripoError> {
    transport.check_cancelled(None)?;
    let plan = prepare_request(input, owner)?;
    transport.check_cancelled(None)?;
    if maximum_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let head = request_head(plan.path, HttpMethod::Post, credentials, false)?;
    let mut session = Session::default();
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    transport.source_request_labels(handle, None, None, true)?;
    let cost = transport.request_cost(
        handle,
        &CostRequest {
            operation: "tripo.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits,
        },
    )?;
    if !cost.accepted
        || cost.approved_microunits == 0
        || cost.approved_microunits > maximum_microunits
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    bounded_receipt(&cost.receipt)?;
    let creation = exchange_existing(
        &head,
        &serde_json::to_vec(&plan.body)?,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let created = normalized(&creation, ApiModel::Task, transport)?;
    if created.get("code").and_then(Value::as_i64) != Some(0) {
        return Err(TripoError::SourceAttributeMissing(
            "'TripoTaskResponse' object has no attribute 'error'",
        ));
    }
    let task_id = retained_id(
        created
            .get("data")
            .ok_or(TripoError::Response("Missing source task data"))?,
        "task_id",
    )?;
    let import = plan.model == RequestModel::Import;
    let mut responses = vec![];
    retain_response(&mut responses, creation)?;
    let status_path = format!("/proxy/tripo/v2/openapi/task/{task_id}");
    let mut queued_polls = 0u32;
    for _ in 0..4096 {
        let predecessor = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        let head = request_head(&status_path, HttpMethod::Get, credentials, false)?;
        transport.check_cancelled(Some(predecessor.handle))?;
        transport.authorize_continuation(predecessor, &head)?;
        let handle = transport.start_request(&head)?;
        session.retain(handle)?;
        transport.source_request_labels(handle, None, None, true)?;
        let retained = exchange_existing(
            &head,
            &[],
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            handle,
            &mut session,
            transport,
        )?;
        let response = normalized(&retained, ApiModel::Task, transport)?;
        let data = response
            .get("data")
            .ok_or(TripoError::Response("Missing source task data"))?;
        let status = string(data, "status")?.to_owned();
        let progress = data.get("progress").and_then(Value::as_f64);
        if progress.is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value)) {
            return Err(ConformanceError::InvalidResponse.into());
        }
        let credits = data.get("consumed_credit").and_then(Value::as_f64);
        let source_usd = if import {
            None
        } else {
            credits
                .filter(|value| *value != 0.0)
                .map(|value| value * 0.01)
        };
        if source_usd.is_some_and(|value| !value.is_finite()) {
            return Err(ConformanceError::InvalidResponse.into());
        }
        transport.report_task_progress(
            &retained,
            &TaskProgress {
                status: status.clone(),
                queued: status == "queued",
                percentage: progress,
                source_usd,
                estimated_seconds: plan.estimated_seconds,
            },
        )?;
        let completed = status == "success";
        let url = if completed && !import {
            let output = data
                .get("output")
                .ok_or(TripoError::Response("Missing source model output"))?;
            Some(
                ["pbr_model", "model", "base_model"]
                    .iter()
                    .find_map(|key| output.get(key).filter(|value| !value.is_null()))
                    .and_then(Value::as_str)
                    .ok_or(TripoError::Response("Failed to get model URL"))?
                    .to_owned(),
            )
        } else {
            None
        };
        retain_response(&mut responses, retained)?;
        if completed {
            let downloaded = if let Some(url) = &url {
                if endpoint(url)? {
                    return Err(TripoError::Response(
                        "Source model download requires absolute URL",
                    ));
                }
                let head = request_head(url, HttpMethod::Get, credentials, true)?;
                let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
                transport.authorize_continuation(prior, &head)?;
                let downloaded = exchange_with_policy(
                    &head,
                    &[],
                    MAXIMUM_MEDIA_BYTES,
                    DOWNLOAD_RETRY,
                    &mut session,
                    transport,
                )?;
                if downloaded.bytes.is_empty() {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                retain_response(&mut responses, downloaded.clone())?;
                Some(downloaded)
            } else {
                None
            };
            transport.check_cancelled(None)?;
            return Ok(InvocationResult {
                node: input.node.clone(),
                task_id: task_id.clone(),
                input_receipts: plan.input_receipts,
                responses,
                downloaded,
                compatibility_save: if import {
                    None
                } else {
                    Some(CompatibilitySave {
                        filename: format!("{task_id}.glb"),
                        format: "glb",
                    })
                },
                cost_receipt: cost.receipt,
                url,
            });
        }
        if matches!(
            status.as_str(),
            "failed" | "cancelled" | "unknown" | "banned" | "expired"
        ) {
            return Err(TripoError::Provider(status));
        }
        if !matches!(status.as_str(), "queued" | "running") {
            return Err(TripoError::Response("Unknown source task status"));
        }
        if status == "queued" {
            queued_polls += 1;
            if queued_polls >= 480 {
                return Err(ConformanceError::PollLimit.into());
            }
        }
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        transport.check_cancelled(Some(prior.handle))?;
        transport.delay(prior.handle, 5000)?;
    }
    Err(ConformanceError::PollLimit.into())
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, TripoError> {
    if !NODES.iter().any(|claim| claim.node_identifier == node) {
        return Err(TripoError::Input("Unknown source output node"));
    }
    let names: &[&str] = if node == "TripoImportModelNode" {
        &["MODEL_TASK_ID"]
    } else {
        &["STRING", "MODEL_TASK_ID", "FILE_3D_GLB"]
    };
    let registry =
        TypeRegistry::built_in().map_err(|_| TripoError::Response("Missing canonical registry"))?;
    names
        .iter()
        .enumerate()
        .map(|(ordinal, name)| {
            let type_id = registry
                .resolve(name)
                .map_err(|_| TripoError::Response("Missing canonical source type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| TripoError::Response("Missing canonical family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{ordinal}"),
                type_id,
                family,
            })
        })
        .collect()
}
pub trait TripoOutputOwner {
    fn materialize(
        &mut self,
        _: &InvocationResult,
        _: &[ExpectedOutput],
    ) -> Result<Vec<EncodedOutput>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
pub fn materialize_and_publish(
    result: &InvocationResult,
    transport: &mut dyn TripoTransport,
    owner: &mut dyn TripoOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), TripoError> {
    transport.check_cancelled(None)?;
    let expected = expected_outputs(&result.node)?;
    let outputs = owner.materialize(result, &expected)?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}

fn source_oracle() -> Result<Value, ConformanceError> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/tripo/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture["namespace"] != NAMESPACE
        || fixture["source_sha256"] != SOURCE_SHA256
        || fixture["nodes"] != json!(NODES)
        || fixture["routes"] != json!(ROUTES)
        || fixture["aliases"] != json!(ALIASES)
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = source_oracle()?;
    if source["sources"]["comfy_api_nodes/nodes_tripo.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let covered: BTreeSet<&str> = source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
        .iter()
        .map(|case| {
            case["input"]["node"]
                .as_str()
                .ok_or(ConformanceError::InvalidRequest)
        })
        .collect::<Result<_, _>>()?;
    if covered != NODES.iter().map(|claim| claim.node_identifier).collect() {
        return Err(ConformanceError::InvalidRequest);
    }
    for claim in NODES {
        expected_outputs(claim.node_identifier).map_err(|_| ConformanceError::InvalidRequest)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        CostRequest, CostResponse, Progress, SignedPutRequest, StorageCreateRequest,
        StorageCreateResponse,
    };
    use std::{cell::RefCell, collections::VecDeque, rc::Rc};
    type Events = Rc<RefCell<Vec<String>>>;
    type ReportedProgress = (String, bool, Option<f64>, Option<f64>, Option<u64>);
    struct Response {
        path: String,
        method: HttpMethod,
        body: Vec<u8>,
        status: u16,
        bytes: Vec<u8>,
        headers: Vec<Header>,
        receipt: Vec<u8>,
        normalized: Option<(ApiModel, Value)>,
        failure: Option<ConformanceError>,
    }
    struct TestTransport {
        responses: VecDeque<Response>,
        active: Option<(StreamHandle, Response, u64, Vec<u8>)>,
        next_slot: u32,
        invocation: u64,
        events: Events,
        heads: Vec<RequestHead>,
        deadlines: Vec<Option<u64>>,
        progress: Vec<ReportedProgress>,
        completed: Vec<RetainedResponse>,
        validate_models: bool,
        normalized: Vec<(StreamHandle, ApiModel, Value)>,
        continuation_bound: bool,
        labels: Vec<(Option<String>, Option<String>, bool)>,
        delays: Vec<u64>,
        cancelled: bool,
        failure: Option<ConformanceError>,
        stale_retry: bool,
        reject_cost: bool,
    }
    impl TestTransport {
        fn new(responses: Vec<Response>, events: Events) -> Self {
            Self {
                responses: responses.into(),
                active: None,
                next_slot: 1,
                invocation: 42,
                events,
                heads: vec![],
                deadlines: vec![],
                progress: vec![],
                completed: vec![],
                validate_models: true,
                normalized: vec![],
                continuation_bound: true,
                labels: vec![],
                delays: vec![],
                cancelled: false,
                failure: None,
                stale_retry: false,
                reject_cost: false,
            }
        }
        fn finish(&self) -> Result<(), Box<dyn Error>> {
            if self.active.is_some() || !self.responses.is_empty() {
                return Err("unconsumed script".into());
            }
            Ok(())
        }
    }
    impl ProviderTransport for TestTransport {
        fn check_cancelled(&mut self, _: Option<StreamHandle>) -> Result<(), ConformanceError> {
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
            if response.path != head.path || response.method != head.method {
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
                invocation: self.invocation,
                slot: self.next_slot,
                generation: 1,
            };
            self.next_slot += 1;
            self.events
                .borrow_mut()
                .push(format!("request:{}", head.path));
            self.heads.push(head.clone());
            self.active = Some((handle, response, 0, vec![]));
            Ok(handle)
        }
        fn write_request_chunk(
            &mut self,
            handle: StreamHandle,
            sequence: u64,
            bytes: &[u8],
            end: bool,
        ) -> Result<(), ConformanceError> {
            let (active, response, _, received) = self
                .active
                .as_mut()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle
                || sequence != (received.len() / MAXIMUM_CHUNK_BYTES) as u64
                || bytes.len() > MAXIMUM_CHUNK_BYTES
            {
                return Err(ConformanceError::InvalidRequest);
            }
            received.extend_from_slice(bytes);
            if end
                && response.body != *received
                && !matches!((serde_json::from_slice::<Value>(&response.body),serde_json::from_slice::<Value>(received)),(Ok(expected),Ok(actual)) if expected==actual)
            {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(())
        }
        fn upload(&mut self, _: StreamHandle, _: &[u8]) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn create_storage(
            &mut self,
            _: StreamHandle,
            _: &StorageCreateRequest,
        ) -> Result<StorageCreateResponse, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn put_signed(
            &mut self,
            _: StreamHandle,
            _: &SignedPutRequest,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn request_cost(
            &mut self,
            _: StreamHandle,
            _: &CostRequest,
        ) -> Result<CostResponse, ConformanceError> {
            if self.reject_cost {
                return Err(ConformanceError::InvalidCost);
            }
            Ok(CostResponse {
                accepted: true,
                approved_microunits: 1,
                receipt: b"canonical test cost".to_vec(),
            })
        }
        fn report_progress(
            &mut self,
            _: StreamHandle,
            _: &Progress,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn wait_response(
            &mut self,
            handle: StreamHandle,
            after: Option<u64>,
            timeout: u64,
        ) -> Result<ResponseEvent, ConformanceError> {
            let (active, response, sequence, _) = self
                .active
                .as_mut()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle || timeout != 1000 || after != sequence.checked_sub(1) {
                return Err(ConformanceError::InvalidResponse);
            }
            if let Some(error) = &response.failure {
                return Err(error.clone());
            }
            let chunks = response.bytes.len().div_ceil(MAXIMUM_CHUNK_BYTES) as u64;
            let event = if *sequence == 0 {
                ResponseEvent::Head {
                    status: response.status,
                    headers: response.headers.clone(),
                }
            } else if *sequence <= chunks {
                ResponseEvent::Chunk(
                    response
                        .bytes
                        .chunks(MAXIMUM_CHUNK_BYTES)
                        .nth((*sequence - 1) as usize)
                        .ok_or(ConformanceError::InvalidResponse)?
                        .to_vec(),
                )
            } else if *sequence == chunks + 1 {
                ResponseEvent::Completed(response.receipt.clone())
            } else {
                return Err(ConformanceError::InvalidResponse);
            };
            *sequence += 1;
            if matches!(event, ResponseEvent::Completed(_)) {
                self.completed.push(RetainedResponse {
                    bytes: response.bytes.clone(),
                    receipt: response.receipt.clone(),
                    handle,
                    headers: response.headers.clone(),
                });
                if let Some((model, value)) = &response.normalized {
                    self.normalized.push((handle, *model, value.clone()));
                }
                self.active = None;
            }
            Ok(event)
        }
    }
    impl TripoTransport for TestTransport {
        fn begin_response_deadline(
            &mut self,
            handle: StreamHandle,
            timeout: Option<u64>,
        ) -> Result<(), ConformanceError> {
            if self
                .active
                .as_ref()
                .is_none_or(|(active, _, _, _)| *active != handle)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.deadlines.push(timeout);
            Ok(())
        }
        fn validate_response(
            &mut self,
            retained: &RetainedResponse,
            model: ApiModel,
            response: &Value,
        ) -> Result<Value, ConformanceError> {
            if !self.validate_models {
                return Err(ConformanceError::HostFailure);
            }
            if !self.completed.iter().any(|completed| {
                completed.handle == retained.handle
                    && completed.bytes == retained.bytes
                    && completed.receipt == retained.receipt
                    && completed.headers == retained.headers
            }) || !matches!(serde_json::from_slice::<Value>(&retained.bytes),Ok(ref raw) if raw==response)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.normalized
                .iter()
                .find(|(handle, kind, _)| *handle == retained.handle && *kind == model)
                .map(|(_, _, normalized)| normalized.clone())
                .ok_or(ConformanceError::InvalidResponse)
        }
        fn authorize_continuation(
            &mut self,
            retained: &RetainedResponse,
            head: &RequestHead,
        ) -> Result<(), ConformanceError> {
            if !self.continuation_bound {
                return Err(ConformanceError::HostFailure);
            }
            if retained.handle.invocation != self.invocation
                || !self.completed.iter().any(|completed| {
                    completed.handle == retained.handle
                        && completed.bytes == retained.bytes
                        && completed.receipt == retained.receipt
                        && completed.headers == retained.headers
                })
                || self.responses.front().is_none_or(|response| {
                    response.path != head.path || response.method != head.method
                })
            {
                return Err(ConformanceError::InvalidHandle);
            }
            Ok(())
        }
        fn source_request_labels(
            &mut self,
            _: StreamHandle,
            wait: Option<&str>,
            final_label: Option<&str>,
            monitor: bool,
        ) -> Result<(), ConformanceError> {
            self.labels.push((
                wait.map(str::to_owned),
                final_label.map(str::to_owned),
                monitor,
            ));
            Ok(())
        }
        fn report_task_progress(
            &mut self,
            retained: &RetainedResponse,
            progress: &TaskProgress,
        ) -> Result<(), ConformanceError> {
            if !self.completed.iter().any(|response| {
                response.handle == retained.handle && response.receipt == retained.receipt
            }) {
                return Err(ConformanceError::InvalidHandle);
            }
            self.progress.push((
                progress.status.clone(),
                progress.queued,
                progress.percentage,
                progress.source_usd,
                progress.estimated_seconds,
            ));
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
                .heads
                .last()
                .cloned()
                .ok_or(ConformanceError::InvalidHandle)?;
            self.start_request(&head)
        }
        fn delay(&mut self, _: StreamHandle, delay: u64) -> Result<(), ConformanceError> {
            self.delays.push(delay);
            Ok(())
        }
    }
    fn hex(value: &Value) -> Result<Vec<u8>, Box<dyn Error>> {
        let text = value.as_str().ok_or("hex string")?;
        if text.len() % 2 != 0 {
            return Err("odd hex".into());
        }
        text.as_bytes()
            .chunks_exact(2)
            .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
            .collect()
    }
    fn method(value: &str) -> Result<HttpMethod, Box<dyn Error>> {
        match value {
            "GET" => Ok(HttpMethod::Get),
            "POST" => Ok(HttpMethod::Post),
            "PATCH" => Ok(HttpMethod::Patch),
            "PUT" => Ok(HttpMethod::Put),
            _ => Err("method".into()),
        }
    }
    fn response(
        path: &str,
        method: HttpMethod,
        body: Vec<u8>,
        raw: &Value,
        normalized: Option<Value>,
    ) -> Result<Response, Box<dyn Error>> {
        Ok(Response {
            path: path.into(),
            method,
            body,
            status: 200,
            bytes: serde_json::to_vec(raw)?,
            headers: vec![],
            receipt: b"retained canonical test terminal".to_vec(),
            normalized: normalized.map(|value| (ApiModel::Task, value)),
            failure: None,
        })
    }
    fn scripts(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let mut script = vec![];
        for request in case["requests"].as_array().ok_or("source requests")? {
            script.push(response(
                request["path"].as_str().ok_or("path")?,
                method(request["method"].as_str().ok_or("method")?)?,
                serde_json::to_vec(&request["body"])?,
                &request["response"],
                request.get("normalized_response").cloned(),
            )?);
        }
        for poll in case["polls"].as_array().ok_or("source polls")? {
            for retained in poll["responses"]
                .as_array()
                .ok_or("source poll responses")?
            {
                script.push(response(
                    poll["path"].as_str().ok_or("path")?,
                    HttpMethod::Get,
                    vec![],
                    &retained["raw"],
                    retained.get("normalized").cloned(),
                )?);
            }
        }
        for download in case["downloads"].as_array().ok_or("source downloads")? {
            let mut record = response(
                download["url"].as_str().ok_or("url")?,
                HttpMethod::Get,
                vec![],
                &json!({}),
                None,
            )?;
            record.bytes = hex(&download["bytes_hex"])?;
            script.push(record);
        }
        Ok(script)
    }
    struct TestOwner {
        case: Value,
        upload_count: usize,
        raw_checked: bool,
        fail: bool,
    }
    impl TestOwner {
        fn new(case: &Value) -> Self {
            Self {
                case: case.clone(),
                upload_count: 0,
                raw_checked: false,
                fail: false,
            }
        }
        fn uploaded(&mut self) -> Result<PreparedUpload, ConformanceError> {
            let upload = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            if hex(&upload["bytes_hex"])
                .map_err(|_| ConformanceError::InvalidRequest)?
                .is_empty()
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.upload_count += 1;
            Ok(PreparedUpload {
                urls: vec![format!(
                    "https://source.fixture.invalid/input-{}.bin",
                    self.upload_count
                )],
                receipts: vec![format!("verified upload {}", self.upload_count).into_bytes()],
            })
        }
    }
    impl TripoInputOwner for TestOwner {
        fn verify_inputs(&mut self, input: &Input) -> Result<(), ConformanceError> {
            if self.fail
                || serde_json::to_value(input).map_err(|_| ConformanceError::InvalidRequest)?["node"]
                    != self.case["input"]["node"]
            {
                return Err(ConformanceError::HostFailure);
            }
            Ok(())
        }
        fn image_upload(
            &mut self,
            input: &Input,
            port: &str,
            policy: &ImageUploadPolicy,
        ) -> Result<PreparedUpload, ConformanceError> {
            if policy
                != &(ImageUploadPolicy {
                    maximum_images: 1,
                    total_pixels: 2048 * 2048,
                    content_type: None,
                })
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let ports: Vec<_> = ["image", "image_left", "image_back", "image_right"]
                .into_iter()
                .filter(|name| input.images.get(*name).is_some_and(Option::is_some))
                .collect();
            if ports.get(self.upload_count).copied() != Some(port) {
                return Err(ConformanceError::InvalidRequest);
            }
            if input
                .images
                .get(port)
                .and_then(Option::as_ref)
                .is_some_and(|image| image.batch == 0)
            {
                return Ok(PreparedUpload {
                    urls: vec![],
                    receipts: vec![],
                });
            }
            let upload = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            let bytes = hex(&upload["bytes_hex"]).map_err(|_| ConformanceError::InvalidRequest)?;
            if !upload["mime_type"].is_null()
                || !bytes.starts_with(b"\x89PNG\r\n\x1a\n")
                || !upload["file_name"]
                    .as_str()
                    .is_some_and(|name| name.ends_with(".png"))
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.uploaded()
        }
        fn file_format(&mut self, input: &Input) -> Result<String, ConformanceError> {
            let file = input
                .file
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            Ok(if file.format.is_empty() && file.stream_kind == "disk" {
                "glb".into()
            } else {
                file.format.clone()
            })
        }
        fn file_byte_length(&mut self, input: &Input) -> Result<u64, ConformanceError> {
            let file = input
                .file
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            Ok(if file.oversized {
                MAXIMUM_IMPORT_BYTES + 1
            } else if file.format.trim_start_matches('.').to_lowercase() == "obj" {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/tripo/input.obj"
                ))
                .len() as u64
            } else {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/tripo/encoded-model.glb"
                ))
                .len() as u64
            })
        }
        fn file_upload(
            &mut self,
            _: &Input,
            format: &str,
            mime: &str,
        ) -> Result<PreparedUpload, ConformanceError> {
            let record = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            let expected = if format == "glb" {
                "model/gltf-binary"
            } else if format == "obj" {
                "model/obj"
            } else {
                "application/octet-stream"
            };
            if mime != expected
                || record["mime_type"] != expected
                || !record["file_name"]
                    .as_str()
                    .is_some_and(|name| name.ends_with(&format!(".{format}")))
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.uploaded()
        }
        fn normalize_request(
            &mut self,
            model: RequestModel,
            raw: &Value,
        ) -> Result<Value, ConformanceError> {
            let record = self.case["constructors"]
                .get(0)
                .ok_or(ConformanceError::InvalidRequest)?;
            if record["model"] != model.source_name() || record["raw"] != *raw {
                return Err(ConformanceError::InvalidRequest);
            }
            self.raw_checked = true;
            record
                .get("normalized")
                .cloned()
                .ok_or(ConformanceError::InvalidRequest)
        }
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn input(case: &Value) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(case["input"].clone())?)
    }
    fn first_case(node: &str) -> Result<Value, Box<dyn Error>> {
        source_oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == node)
            .cloned()
            .ok_or_else(|| "node case".into())
    }
    #[test]
    fn all_actual_source_successes_match_requests_upload_order_models_and_glb_save()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (ordinal, case) in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .enumerate()
        {
            let mut owner = TestOwner::new(case);
            let mut transport = TestTransport::new(scripts(case)?, Rc::default());
            let result = invoke(&input(case)?, credentials(), 1, &mut transport, &mut owner)
                .map_err(|error| {
                    format!(
                        "case {ordinal} {} {:?}: {error}; raw checked={}",
                        case["input"]["node"], case["input"]["settings"], owner.raw_checked
                    )
                })?;
            transport.finish()?;
            assert!(owner.raw_checked, "case {ordinal}");
            assert_eq!(
                owner.upload_count,
                case["uploads"].as_array().ok_or("uploads")?.len()
            );
            let output = case["output"].as_array().ok_or("output")?;
            if result.node == "TripoImportModelNode" {
                assert_eq!(
                    result.task_id,
                    output.first().and_then(Value::as_str).ok_or("ID")?
                );
                assert!(result.downloaded.is_none());
                assert!(result.compatibility_save.is_none());
                assert_eq!(expected_outputs(&result.node)?.len(), 1);
            } else {
                let save = result.compatibility_save.as_ref().ok_or("save")?;
                assert_eq!(
                    save.filename,
                    output.first().and_then(Value::as_str).ok_or("filename")?
                );
                assert_eq!(save.format, "glb");
                assert_eq!(
                    result.task_id,
                    output.get(1).and_then(Value::as_str).ok_or("ID")?
                );
                assert_eq!(
                    result.downloaded.as_ref().ok_or("download")?.bytes,
                    hex(&output.get(2).ok_or("model")?["bytes_hex"])?
                );
                assert_eq!(save.filename, case["saves"][0]["file_name"]);
                assert_eq!(expected_outputs(&result.node)?.len(), 3);
            }
            let source_statuses = &case["polls"][0]["statuses"];
            for (actual, expected) in transport
                .progress
                .iter()
                .zip(source_statuses.as_array().ok_or("statuses")?)
            {
                assert_eq!(actual.0, expected["status"].as_str().ok_or("status")?);
                assert_eq!(actual.2, expected["progress"].as_f64());
                assert_eq!(actual.3, expected["price"].as_f64());
                assert_eq!(
                    actual.4,
                    case["polls"][0]["options"]["estimated_duration"].as_u64()
                );
            }
            assert_eq!(
                transport.progress.len(),
                source_statuses.as_array().ok_or("statuses")?.len()
            );
        }
        Ok(())
    }
    #[test]
    fn all_actual_source_failures_preserve_upload_and_request_prefixes()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (ordinal, case) in source["errors"]
            .as_array()
            .ok_or("errors")?
            .iter()
            .enumerate()
        {
            let mut owner = TestOwner::new(case);
            let mut transport = TestTransport::new(scripts(case)?, Rc::default());
            let result = invoke(&input(case)?, credentials(), 1, &mut transport, &mut owner);
            assert!(result.is_err(), "error {ordinal}: {}", case["input"]);
            if case["error_type"] == "AttributeError" {
                assert!(
                    matches!(result, Err(TripoError::SourceAttributeMissing(_))),
                    "error {ordinal}: {result:?}"
                );
            }
            assert_eq!(
                owner.upload_count,
                case["uploads"].as_array().ok_or("uploads")?.len(),
                "error {ordinal}: {result:?}"
            );
            let expected = case["requests"].as_array().ok_or("requests")?.len();
            let actual = transport
                .heads
                .iter()
                .filter(|head| head.method == HttpMethod::Post)
                .count();
            assert_eq!(
                actual, expected,
                "error {ordinal}: {result:?}, raw checked={}",
                owner.raw_checked
            );
            if !case["constructors"]
                .as_array()
                .ok_or("constructors")?
                .is_empty()
            {
                assert!(owner.raw_checked, "error {ordinal}: {result:?}");
            }
            assert!(
                !transport
                    .heads
                    .iter()
                    .any(|head| head.path.starts_with("https://")),
                "error {ordinal}"
            );
        }
        Ok(())
    }
    #[test]
    fn authority_cancel_cost_worker_loss_and_default_owners_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let case = first_case("TripoTextToModelNode")?;
        let input = input(&case)?;
        struct UnboundInput;
        impl TripoInputOwner for UnboundInput {}
        let mut transport = TestTransport::new(scripts(&case)?, Rc::default());
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut transport, &mut UnboundInput),
            Err(TripoError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        for error in [ConformanceError::Cancelled, ConformanceError::WorkerLost] {
            let mut owner = TestOwner::new(&case);
            let mut transport = TestTransport::new(scripts(&case)?, Rc::default());
            transport.failure = Some(error.clone());
            assert!(
                matches!(invoke(&input,credentials(),1,&mut transport,&mut owner),Err(TripoError::Transport(ref actual)) if *actual==error)
            );
            assert!(transport.heads.is_empty());
        }
        for (validate, continuation) in [(false, true), (true, false)] {
            let mut owner = TestOwner::new(&case);
            let mut transport = TestTransport::new(scripts(&case)?, Rc::default());
            transport.validate_models = validate;
            transport.continuation_bound = continuation;
            assert!(matches!(
                invoke(&input, credentials(), 1, &mut transport, &mut owner),
                Err(TripoError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(transport.heads.len(), 1);
        }
        let mut owner = TestOwner::new(&case);
        let mut transport = TestTransport::new(scripts(&case)?, Rc::default());
        transport.reject_cost = true;
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut transport, &mut owner),
            Err(TripoError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(transport.completed.is_empty());
        Ok(())
    }
    #[test]
    fn retained_response_tampering_cross_scope_and_replay_are_rejected()
    -> Result<(), Box<dyn Error>> {
        let case = first_case("TripoTextToModelNode")?;
        let mut owner = TestOwner::new(&case);
        let mut transport = TestTransport::new(scripts(&case)?, Rc::default());
        let result = invoke(&input(&case)?, credentials(), 1, &mut transport, &mut owner)?;
        let original = result.responses.first().ok_or("response")?;
        for variant in 0..4 {
            let mut changed = original.clone();
            match variant {
                0 => changed.bytes.push(0),
                1 => changed.receipt.push(0),
                2 => changed.headers.push(Header {
                    name: "X-Test".into(),
                    value: "tamper".into(),
                }),
                _ => changed.handle.invocation += 1,
            }
            assert!(matches!(
                transport.validate_response(
                    &changed,
                    ApiModel::Task,
                    &serde_json::from_slice(&original.bytes)?
                ),
                Err(ConformanceError::InvalidHandle)
            ));
        }
        let mut session = Session::default();
        session.retain(original.handle)?;
        assert!(matches!(
            session.retain(original.handle),
            Err(TripoError::Transport(ConformanceError::InvalidHandle))
        ));
        let mut cross = original.handle;
        cross.invocation += 1;
        assert!(session.retain(cross).is_err());
        Ok(())
    }
    #[test]
    fn http_retries_bounds_and_fresh_invocation_recovery() -> Result<(), Box<dyn Error>> {
        let case = first_case("TripoTextToModelNode")?;
        for stale in [false, true] {
            let mut script = scripts(&case)?;
            let mut transient = response(
                "/proxy/tripo/v2/openapi/task",
                HttpMethod::Post,
                script.first().ok_or("first")?.body.clone(),
                &json!({}),
                None,
            )?;
            transient.status = 503;
            script.insert(0, transient);
            let mut owner = TestOwner::new(&case);
            let mut transport = TestTransport::new(script, Rc::default());
            transport.stale_retry = stale;
            let result = invoke(&input(&case)?, credentials(), 1, &mut transport, &mut owner);
            if stale {
                assert!(matches!(
                    result,
                    Err(TripoError::Transport(ConformanceError::InvalidHandle))
                ));
            } else {
                result?;
                transport.finish()?;
                assert_eq!(transport.delays.first(), Some(&1000));
            }
        }
        let mut script = scripts(&case)?;
        script.first_mut().ok_or("first")?.bytes = vec![b'x'; MAXIMUM_RESPONSE_BYTES + 1];
        let mut owner = TestOwner::new(&case);
        let mut transport = TestTransport::new(script, Rc::default());
        assert!(matches!(
            invoke(&input(&case)?, credentials(), 1, &mut transport, &mut owner),
            Err(TripoError::Transport(ConformanceError::BodyLimit))
        ));
        let mut script = scripts(&case)?;
        script.first_mut().ok_or("first")?.failure = Some(ConformanceError::WorkerLost);
        let mut owner = TestOwner::new(&case);
        let mut transport = TestTransport::new(script, Rc::default());
        assert!(matches!(
            invoke(&input(&case)?, credentials(), 1, &mut transport, &mut owner),
            Err(TripoError::Transport(ConformanceError::WorkerLost))
        ));
        let mut fresh = TestTransport::new(scripts(&case)?, Rc::default());
        fresh.invocation = 43;
        invoke(
            &input(&case)?,
            credentials(),
            1,
            &mut fresh,
            &mut TestOwner::new(&case),
        )?;
        fresh.finish()?;
        Ok(())
    }
    #[test]
    fn canonical_model_ids_and_tombstone_are_not_transport_grants() -> Result<(), Box<dyn Error>> {
        assert_eq!(NODES.len(), 12);
        assert_eq!(ROUTES.len(), 4);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            3
        );
        let outputs = expected_outputs("TripoImportModelNode")?;
        assert_eq!(outputs[0].type_id.to_string(), "sim:model-task-id@1");
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/tripo/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn unbound_native_compatibility_materialization_publishes_nothing() -> Result<(), Box<dyn Error>>
    {
        struct Output;
        impl TripoOutputOwner for Output {}
        #[derive(Default)]
        struct Sink {
            published: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(
                &mut self,
                _: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.published += 1;
                Ok(())
            }
        }
        let case = first_case("TripoTextToModelNode")?;
        let mut owner = TestOwner::new(&case);
        let mut transport = TestTransport::new(scripts(&case)?, Rc::default());
        let result = invoke(&input(&case)?, credentials(), 1, &mut transport, &mut owner)?;
        let mut sink = Sink::default();
        assert!(matches!(
            materialize_and_publish(&result, &mut transport, &mut Output, &mut sink),
            Err(TripoError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(sink.published, 0);
        Ok(())
    }

    #[test]
    fn multiview_source_errors_keep_exact_text_and_preparation_priority()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let errors = source["errors"].as_array().ok_or("errors")?;
        let mut checked = 0;
        for case in errors.iter().filter(|case| {
            matches!(case["error_message"].as_str(),
                Some("front image for multiview is required")
                | Some("At least one of left, back, or right image must be provided for multiview")
                | Some("Tripo P1 multiview requires at least 2 images (front plus one of left/back/right)."))
        }) {
            assert_eq!(case["error_type"], "RuntimeError");
            for compound_fault in [false, true] {
                let mut input = input(case)?;
                if compound_fault {
                    input.settings = json!({"output_mode": 42, "face_limit": 47});
                }
                let mut owner = TestOwner::new(case);
                let events = Events::default();
                let mut transport = TestTransport::new(vec![], events.clone());
                match invoke(&input, credentials(), 1, &mut transport, &mut owner) {
                    Err(TripoError::Input(message)) => {
                        assert_eq!(message, case["error_message"].as_str().ok_or("message")?);
                    }
                    other => return Err(format!("unexpected multiview result: {other:?}").into()),
                }
                assert_eq!(owner.upload_count, 0);
                assert!(!owner.raw_checked);
                assert!(transport.heads.is_empty());
                assert!(transport.completed.is_empty());
                assert!(events.borrow().is_empty());
                transport.finish()?;
            }
            checked += 1;
        }
        assert_eq!(checked, 3);
        let case = first_case("TripoMultiviewToModelNode")?;
        let mut input = input(&case)?;
        input.images.clear();
        let mut owner = TestOwner::new(&case);
        let mut transport = TestTransport::new(vec![], Events::default());
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut transport, &mut owner),
            Err(TripoError::Input("front image for multiview is required"))
        ));
        assert_eq!(owner.upload_count, 0);
        assert!(transport.heads.is_empty());
        transport.finish()?;
        Ok(())
    }

    #[test]
    fn multiview_cancellation_and_input_authority_precede_source_admission()
    -> Result<(), Box<dyn Error>> {
        let case = first_case("TripoP1MultiviewToModelNode")?;
        let mut input = input(&case)?;
        input.images.clear();
        for cancelled in [true, false] {
            let mut owner = TestOwner::new(&case);
            owner.fail = true;
            let mut transport = TestTransport::new(vec![], Events::default());
            transport.cancelled = cancelled;
            let expected = if cancelled {
                ConformanceError::Cancelled
            } else {
                ConformanceError::HostFailure
            };
            assert!(matches!(
                invoke(&input, credentials(), 1, &mut transport, &mut owner),
                Err(TripoError::Transport(actual)) if actual == expected
            ));
            assert_eq!(owner.upload_count, 0);
            assert!(!owner.raw_checked);
            assert!(transport.heads.is_empty());
            assert!(transport.completed.is_empty());
            transport.finish()?;
        }
        Ok(())
    }

    #[test]
    fn p1_two_side_views_keep_source_upload_order_and_fresh_recovery() -> Result<(), Box<dyn Error>>
    {
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| {
                case["input"]["node"] == "TripoP1MultiviewToModelNode"
                    && case["input"]["images"]["image"].is_null()
                    && case["input"]["images"]["image_right"].is_null()
                    && case["input"]["images"]["image_left"].is_object()
                    && case["input"]["images"]["image_back"].is_object()
            })
            .ok_or("source two-side case")?;
        let mut invalid = input(case)?;
        invalid.images.clear();
        let mut owner = TestOwner::new(case);
        let mut rejected = TestTransport::new(vec![], Events::default());
        assert!(matches!(
            invoke(&invalid, credentials(), 1, &mut rejected, &mut owner),
            Err(TripoError::Input(
                "Tripo P1 multiview requires at least 2 images (front plus one of left/back/right)."
            ))
        ));
        assert_eq!(owner.upload_count, 0);
        rejected.finish()?;
        let mut transport = TestTransport::new(scripts(case)?, Events::default());
        let result = invoke(&input(case)?, credentials(), 1, &mut transport, &mut owner)?;
        transport.finish()?;
        assert_eq!(owner.upload_count, 2);
        assert!(owner.raw_checked);
        assert!(!result.responses.is_empty());
        assert_eq!(
            transport.heads.first().ok_or("creation")?.method,
            HttpMethod::Post
        );
        Ok(())
    }
}
