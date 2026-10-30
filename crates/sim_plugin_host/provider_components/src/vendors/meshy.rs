use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, ProviderTransport, RequestHead, ResponseEvent, StreamHandle, TypeRegistry,
    publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.meshy";
pub const SOURCE_SHA256: &str = "712c12a06dbc1aa3ab1562fe841d7259c3e65de30cdffc9fcc2d3d6f9e7fb827";
pub const ALIASES: &[&str] = &[];
pub const MAXIMUM_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
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
        feature_id: "COMFY-NODE-0408",
        node_identifier: "MeshyAnimateModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0409",
        node_identifier: "MeshyImageToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0410",
        node_identifier: "MeshyMultiImageToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0411",
        node_identifier: "MeshyRefineNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0412",
        node_identifier: "MeshyRigModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0413",
        node_identifier: "MeshyTextToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0414",
        node_identifier: "MeshyTextureNode",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0099",
        method: "POST",
        path: "/proxy/meshy/openapi/v1/animations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0100",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/animations/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0101",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/animations/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0102",
        method: "POST",
        path: "/proxy/meshy/openapi/v1/image-to-3d",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0103",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/image-to-3d/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0104",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/image-to-3d/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0105",
        method: "POST",
        path: "/proxy/meshy/openapi/v1/multi-image-to-3d",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0106",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/multi-image-to-3d/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0107",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/multi-image-to-3d/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0108",
        method: "POST",
        path: "/proxy/meshy/openapi/v1/retexture",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0109",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/retexture/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0110",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/retexture/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0111",
        method: "POST",
        path: "/proxy/meshy/openapi/v1/rigging",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0112",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/rigging/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0113",
        method: "GET",
        path: "/proxy/meshy/openapi/v1/rigging/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0114",
        method: "POST",
        path: "/proxy/meshy/openapi/v2/text-to-3d",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0115",
        method: "GET",
        path: "/proxy/meshy/openapi/v2/text-to-3d/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0116",
        method: "GET",
        path: "/proxy/meshy/openapi/v2/text-to-3d/{task_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum MeshyError {
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
impl fmt::Display for MeshyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "Meshy job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Meshy HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for MeshyError {}
impl From<ConformanceError> for MeshyError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for MeshyError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, MeshyError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(MeshyError::Input("Missing source string field"))
}
fn integer(value: &Value, key: &str) -> Result<i64, MeshyError> {
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
    .ok_or(MeshyError::Input("Missing source integer field"))
}
fn boolean(value: &Value, key: &str) -> Result<bool, MeshyError> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or(MeshyError::Input("Missing source boolean field"))
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct ImageInput {
    pub role: String,
    pub width: u32,
    pub height: u32,
    pub batch: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    pub images: Vec<ImageInput>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameRequirement {
    pub role: String,
    pub batch_index: u32,
    pub width: u32,
    pub height: u32,
    pub total_pixels: u64,
    pub encoded_mime_type: &'static str,
    pub declared_content_type: Option<&'static str>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UploadGroup {
    pub role: String,
    pub frames: Vec<FrameRequirement>,
    pub first_url: bool,
    pub wait_label: &'static str,
}
fn validate_prompt(value: &str, minimum: usize) -> Result<(), MeshyError> {
    let count = python_strip(value).chars().count();
    if count < minimum || count > 600 {
        return Err(MeshyError::Input("Invalid source prompt length"));
    }
    Ok(())
}
fn image_group(
    input: &Input,
    role: &str,
    first_url: bool,
    label: &'static str,
) -> Result<UploadGroup, MeshyError> {
    let mut frames = vec![];
    for image in &input.images {
        let matches = if role == "images" {
            image.role.starts_with("image") && image.role != "image_style"
        } else {
            image.role == role
        };
        if !matches {
            continue;
        }
        if image.width == 0
            || image.height == 0
            || image.role.len() > 1024
            || image.role.chars().any(char::is_control)
        {
            return Err(MeshyError::Input("Invalid canonical image descriptor"));
        }
        for batch_index in 0..image.batch.min(8 - frames.len() as u32) {
            frames.push(FrameRequirement {
                role: image.role.clone(),
                batch_index,
                width: image.width,
                height: image.height,
                total_pixels: 2048 * 2048,
                encoded_mime_type: "image/png",
                declared_content_type: None,
            });
        }
        if frames.len() == 8 {
            break;
        }
    }
    Ok(UploadGroup {
        role: role.into(),
        frames,
        first_url,
        wait_label: label,
    })
}
pub fn upload_groups(input: &Input) -> Result<Vec<UploadGroup>, MeshyError> {
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(MeshyError::Input("Unknown Meshy node"));
    }
    let has = |role: &str| input.images.iter().any(|image| image.role == role);
    let mut groups = vec![];
    match input.node.as_str() {
        "MeshyTextToModelNode" => validate_prompt(string(&input.settings, "prompt")?, 1)?,
        "MeshyRefineNode" => {
            let prompt = string(&input.settings, "texture_prompt")?;
            if !prompt.is_empty() && has("texture_image") {
                return Err(MeshyError::Input(
                    "Mutually exclusive source texture inputs",
                ));
            }
            if !prompt.is_empty() {
                validate_prompt(prompt, 0)?;
            }
            if has("texture_image") {
                groups.push(image_group(
                    input,
                    "texture_image",
                    true,
                    "Uploading texture",
                )?);
            }
        }
        "MeshyImageToModelNode" | "MeshyMultiImageToModelNode" => {
            let texture = input
                .settings
                .get("should_texture")
                .ok_or(MeshyError::Input("Missing texture controls"))?;
            if string(texture, "should_texture")? == "true" {
                let prompt = string(texture, "texture_prompt")?;
                if !prompt.is_empty() && has("texture_image") {
                    return Err(MeshyError::Input(
                        "Mutually exclusive source texture inputs",
                    ));
                }
                if !prompt.is_empty() {
                    validate_prompt(prompt, 0)?;
                }
                if has("texture_image") {
                    groups.push(image_group(
                        input,
                        "texture_image",
                        true,
                        "Uploading texture",
                    )?);
                }
            }
            if input.node == "MeshyImageToModelNode" {
                groups.push(image_group(input, "image", true, "Uploading base image")?);
            } else {
                groups.push(image_group(
                    input,
                    "images",
                    false,
                    "Uploading base images",
                )?);
            }
        }
        "MeshyRigModelNode" => {
            if has("texture_image") {
                groups.push(image_group(
                    input,
                    "texture_image",
                    true,
                    "Uploading texture",
                )?);
            }
        }
        "MeshyTextureNode" => {
            let prompt = string(&input.settings, "text_style_prompt")?;
            if (!prompt.is_empty() && has("image_style"))
                || (prompt.is_empty() && !has("image_style"))
            {
                return Err(MeshyError::Input("Source requires exactly one style input"));
            }
            if has("image_style") {
                groups.push(image_group(input, "image_style", true, "Uploading style")?);
            }
        }
        "MeshyAnimateModelNode" => {}
        _ => return Err(MeshyError::Input("Unknown Meshy node")),
    }
    Ok(groups)
}
#[derive(Clone, Debug)]
pub struct CompletedUploadRequest {
    pub handle: StreamHandle,
    pub response_receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct UploadedGroup {
    pub role: String,
    pub urls: Vec<String>,
    pub completed_requests: Vec<CompletedUploadRequest>,
}
#[derive(Clone, Debug)]
pub struct PreparedPng {
    pub file_name: String,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct PreparedGroup {
    pub requirement: UploadGroup,
    pub frames: Vec<PreparedPng>,
}
pub trait MeshyUploadOwner {
    // None is the source omission on both storage create and signed PUT. The
    // canonical owner must preserve it across the host/guest boundary, retain
    // scope and verify every completion instead of minting new grants.
    fn upload(
        &mut self,
        _: &PreparedGroup,
        _: &mut dyn MeshyTransport,
    ) -> Result<UploadedGroup, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn optional_boolean(body: &mut Value, settings: &Value, key: &str) -> Result<(), MeshyError> {
    match settings.get(key) {
        None | Some(Value::Null) => {}
        Some(Value::Bool(value)) => body[key] = json!(value),
        _ => return Err(MeshyError::Input("Invalid optional source boolean")),
    };
    Ok(())
}
fn group_url<'a>(uploaded: &'a [UploadedGroup], role: &str) -> Result<Option<&'a str>, MeshyError> {
    match uploaded.iter().find(|group| group.role == role) {
        None => Ok(None),
        Some(group) => {
            Ok(Some(group.urls.first().ok_or(MeshyError::Input(
                "Source upload returned no first URL",
            ))?))
        }
    }
}
pub fn request(input: &Input, uploaded: &[UploadedGroup]) -> Result<(String, Vec<u8>), MeshyError> {
    let settings = &input.settings;
    let mut body = json!({});
    let path = match input.node.as_str() {
        "MeshyTextToModelNode" | "MeshyImageToModelNode" | "MeshyMultiImageToModelNode" => {
            let remesh = settings
                .get("should_remesh")
                .ok_or(MeshyError::Input("Missing remesh controls"))?;
            body["ai_model"] = json!(string(settings, "model")?);
            body["symmetry_mode"] = json!(string(settings, "symmetry_mode")?);
            body["pose_mode"] = json!(string(settings, "pose_mode")?.to_lowercase());
            body["seed"] = json!(integer(settings, "seed")?);
            body["should_remesh"] = json!(string(remesh, "should_remesh")? == "true");
            body["moderation"] = json!(false);
            match remesh.get("topology") {
                None | Some(Value::Null) => {}
                Some(Value::String(value)) => body["topology"] = json!(value),
                _ => return Err(MeshyError::Input("Invalid source topology")),
            };
            if remesh
                .get("target_polycount")
                .is_some_and(|value| !value.is_null())
            {
                let value = integer(remesh, "target_polycount")?;
                if !(100..=300000).contains(&value) {
                    return Err(MeshyError::Input("Source polygon range"));
                }
                body["target_polycount"] = json!(value);
            }
            if input.node == "MeshyTextToModelNode" {
                let prompt = string(settings, "prompt")?;
                if prompt.chars().count() > 600 {
                    return Err(MeshyError::Input("Raw source prompt length"));
                }
                body["mode"] = json!("preview");
                body["prompt"] = json!(prompt);
                body["art_style"] = json!(string(settings, "style")?);
                "/proxy/meshy/openapi/v2/text-to-3d"
            } else {
                let texture = settings
                    .get("should_texture")
                    .ok_or(MeshyError::Input("Missing texture controls"))?;
                let enabled = string(texture, "should_texture")? == "true";
                body["should_texture"] = json!(enabled);
                optional_boolean(&mut body, texture, "enable_pbr")?;
                if enabled {
                    let prompt = string(texture, "texture_prompt")?;
                    if !prompt.is_empty() {
                        if prompt.chars().count() > 600 {
                            return Err(MeshyError::Input("Raw source texture prompt length"));
                        }
                        body["texture_prompt"] = json!(prompt);
                    }
                    if let Some(url) = group_url(uploaded, "texture_image")? {
                        body["texture_image_url"] = json!(url);
                    }
                }
                if input.node == "MeshyImageToModelNode" {
                    body["image_url"] = json!(
                        group_url(uploaded, "image")?
                            .ok_or(MeshyError::Input("Missing image URL"))?
                    );
                    "/proxy/meshy/openapi/v1/image-to-3d"
                } else {
                    body["image_urls"] = json!(
                        uploaded
                            .iter()
                            .find(|group| group.role == "images")
                            .ok_or(MeshyError::Input("Missing images upload"))?
                            .urls
                    );
                    "/proxy/meshy/openapi/v1/multi-image-to-3d"
                }
            }
        }
        "MeshyRefineNode" => {
            body = json!({"mode":"refine","preview_task_id":string(settings,"meshy_task_id")?,"enable_pbr":boolean(settings,"enable_pbr")?,"ai_model":string(settings,"model")?,"moderation":false});
            let prompt = string(settings, "texture_prompt")?;
            if !prompt.is_empty() {
                body["texture_prompt"] = json!(prompt);
            }
            if let Some(url) = group_url(uploaded, "texture_image")? {
                body["texture_image_url"] = json!(url);
            }
            "/proxy/meshy/openapi/v2/text-to-3d"
        }
        "MeshyRigModelNode" => {
            let height = settings
                .get("height_meters")
                .and_then(Value::as_f64)
                .filter(|height| height.is_finite())
                .ok_or(MeshyError::Input("Invalid source height"))?;
            body =
                json!({"input_task_id":string(settings,"meshy_task_id")?,"height_meters":height});
            if let Some(url) = group_url(uploaded, "texture_image")? {
                body["texture_image_url"] = json!(url);
            }
            "/proxy/meshy/openapi/v1/rigging"
        }
        "MeshyAnimateModelNode" => {
            body = json!({"rig_task_id":string(settings,"rig_task_id")?,"action_id":integer(settings,"action_id")?});
            "/proxy/meshy/openapi/v1/animations"
        }
        "MeshyTextureNode" => {
            body = json!({"input_task_id":string(settings,"meshy_task_id")?,"ai_model":string(settings,"model")?,"enable_original_uv":boolean(settings,"enable_original_uv")?,"enable_pbr":boolean(settings,"pbr")?});
            let prompt = string(settings, "text_style_prompt")?;
            if !prompt.is_empty() {
                body["text_style_prompt"] = json!(prompt);
            }
            if let Some(url) = group_url(uploaded, "image_style")? {
                body["image_style_url"] = json!(url);
            }
            "/proxy/meshy/openapi/v1/retexture"
        }
        _ => return Err(MeshyError::Input("Unknown Meshy node")),
    };
    let bytes = serde_json::to_vec(&body)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok((path.into(), bytes))
}
pub trait MeshyTransport: ProviderTransport {
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
        _progress: i64,
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
    fn retain(&mut self, handle: StreamHandle) -> Result<(), MeshyError> {
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
fn endpoint(value: &str) -> Result<bool, MeshyError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(MeshyError::Response("Invalid Meshy response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(MeshyError::Response("Invalid Meshy response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(MeshyError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(MeshyError::Response("Invalid Meshy endpoint authority"));
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
) -> Result<RequestHead, MeshyError> {
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
            return Err(MeshyError::Input("Missing Comfy API credentials"));
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
    transport: &mut dyn MeshyTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MeshyError> {
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
    transport: &mut dyn MeshyTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MeshyError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(MeshyError::Http {
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
                    return Err(MeshyError::Http {
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
    transport: &mut dyn MeshyTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MeshyError> {
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
                    return Err(MeshyError::Http {
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Queued,
    Processing,
    Completed,
}
pub fn task_phase(status: &str) -> Result<Phase, MeshyError> {
    match python_strip(status).to_lowercase().as_str() {
        "succeeded" | "succeed" | "success" | "completed" | "finished" | "done" | "complete" => {
            Ok(Phase::Completed)
        }
        "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error" => {
            Err(MeshyError::Provider(status.into()))
        }
        "created" | "queued" | "queueing" | "submitted" | "initializing" | "wait" | "in_queue" => {
            Ok(Phase::Queued)
        }
        _ => Ok(Phase::Processing),
    }
}
#[derive(Clone, Debug)]
pub struct PollResponse {
    pub id: String,
    pub status: String,
    pub progress: i64,
    pub glb: String,
    pub fbx: String,
}
fn optional_string(value: &Value, key: &str) -> Result<(), MeshyError> {
    match value.get(key) {
        None | Some(Value::Null) | Some(Value::String(_)) => Ok(()),
        _ => Err(MeshyError::Response("Invalid source optional string")),
    }
}
pub fn poll_response(node: &str, bytes: &[u8]) -> Result<PollResponse, MeshyError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let value: Value = serde_json::from_slice(bytes)?;
    let id = string(&value, "id")?.to_owned();
    string(&value, "type")?;
    let status = string(&value, "status")?.to_owned();
    let progress = if value.get("progress").is_some() {
        integer(&value, "progress")?
    } else {
        0
    };
    if let Some(error) = value.get("task_error").filter(|value| !value.is_null()) {
        if !error.is_object() {
            return Err(MeshyError::Response("Invalid source task error"));
        }
        optional_string(error, "message")?;
    }
    let (key, glb_key, fbx_key) = match node {
        "MeshyRigModelNode" => (
            "result",
            "rigged_character_glb_url",
            "rigged_character_fbx_url",
        ),
        "MeshyAnimateModelNode" => ("result", "animation_glb_url", "animation_fbx_url"),
        _ => ("model_urls", "glb", "fbx"),
    };
    if key == "model_urls" {
        string(&value, "thumbnail_url")?;
        optional_string(&value, "video_url")?;
        if let Some(textures) = value.get("texture_urls").filter(|value| !value.is_null()) {
            for texture in textures
                .as_array()
                .ok_or(MeshyError::Response("Invalid source texture URLs"))?
            {
                string(texture, "base_color")?;
                for key in ["metallic", "normal", "roughness"] {
                    optional_string(texture, key)?;
                }
            }
        }
    }
    let empty = json!({});
    let urls = value.get(key).unwrap_or(&empty);
    if !urls.is_object() {
        return Err(MeshyError::Response("Invalid source model URL object"));
    }
    for key in if key == "model_urls" {
        vec!["glb", "fbx", "usdz", "obj"]
    } else {
        vec![glb_key, fbx_key]
    } {
        if urls.get(key).is_some_and(|value| !value.is_string()) {
            return Err(MeshyError::Response("Invalid source model URL"));
        }
    }
    Ok(PollResponse {
        id,
        status,
        progress,
        glb: urls
            .get(glb_key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .into(),
        fbx: urls
            .get(fbx_key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .into(),
    })
}
#[derive(Clone, Debug)]
pub struct DownloadedFile {
    pub task_id: String,
    pub format: &'static str,
    pub save_file_name: String,
    pub source_url: String,
    pub bytes: Vec<u8>,
    pub handle: StreamHandle,
    pub download_receipt: Vec<u8>,
}
pub trait MeshyFileOwner {
    // Source saves GLB before requesting FBX. The canonical owner must retain
    // real File3D resources and perform the authorized compatibility save at
    // this boundary; any save/allocation failure stops the next download.
    fn complete_download(&mut self, _: &DownloadedFile) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct DownloadedResult {
    pub node: String,
    pub task_id: String,
    pub uploaded: Vec<UploadedGroup>,
    pub files: Vec<DownloadedFile>,
    pub cost_receipt: Vec<u8>,
    pub submission_receipt: Vec<u8>,
    pub status_receipt: Vec<u8>,
    pub handle: StreamHandle,
}
pub fn invoke(
    input: &Input,
    prepared: &[PreparedGroup],
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn MeshyTransport,
    upload_owner: &mut dyn MeshyUploadOwner,
    file_owner: &mut dyn MeshyFileOwner,
) -> Result<DownloadedResult, MeshyError> {
    let groups = upload_groups(input)?;
    if prepared.len() != groups.len() {
        return Err(MeshyError::Input(
            "Missing canonical prepared upload groups",
        ));
    }
    let mut total = 0usize;
    for (group, prepared) in groups.iter().zip(prepared) {
        if *group != prepared.requirement || group.frames.len() != prepared.frames.len() {
            return Err(MeshyError::Input(
                "Altered canonical preparation requirement",
            ));
        }
        for frame in &prepared.frames {
            if frame.bytes.is_empty()
                || frame.file_name.is_empty()
                || frame.file_name.len() > 1024
                || !frame.file_name.ends_with(".png")
                || frame
                    .file_name
                    .chars()
                    .any(|character| character.is_control() || matches!(character, '/' | '\\'))
            {
                return Err(MeshyError::Input("Invalid canonical prepared PNG"));
            }
            total = total
                .checked_add(frame.bytes.len())
                .ok_or(ConformanceError::BodyLimit)?;
        }
    }
    if total > MAXIMUM_MEDIA_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    transport.check_cancelled(None)?;
    let mut session = Session::default();
    let mut uploaded = vec![];
    for prepared in prepared {
        transport.check_cancelled(None)?;
        let result = if prepared.frames.is_empty() {
            UploadedGroup {
                role: prepared.requirement.role.clone(),
                urls: vec![],
                completed_requests: vec![],
            }
        } else {
            upload_owner.upload(prepared, transport)?
        };
        if result.role != prepared.requirement.role
            || result.urls.len() != prepared.frames.len()
            || result.completed_requests.len() < prepared.frames.len() * 2
            || result.completed_requests.len() > 256
        {
            return Err(MeshyError::Response(
                "Invalid retained upload group lineage",
            ));
        }
        for url in &result.urls {
            if endpoint(url)? {
                return Err(MeshyError::Response("Source requires absolute storage URL"));
            }
        }
        for completed in &result.completed_requests {
            if completed.response_receipt.is_empty()
                || completed.response_receipt.len() > MAXIMUM_RECEIPT_BYTES
            {
                return Err(ConformanceError::InvalidResponse.into());
            }
            session.retain(completed.handle)?;
        }
        if prepared.requirement.first_url && result.urls.is_empty() {
            return Err(MeshyError::Input("Source upload returned no first URL"));
        }
        uploaded.push(result);
    }
    let (path, body) = request(input, &uploaded)?;
    transport.check_cancelled(None)?;
    let head = request_head(&path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "meshy.generation.maximum".into(),
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
    let (bytes, submission_receipt, _) = exchange_existing(
        &head,
        &body,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let created: Value = serde_json::from_slice(&bytes)?;
    let task_id = string(&created, "result")?.to_owned();
    if task_id.is_empty()
        || task_id.len() > 1024
        || !task_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(MeshyError::Response("Unsafe task route selector"));
    }
    let poll_head = request_head(
        &format!("{path}/{task_id}"),
        HttpMethod::Get,
        credentials,
        false,
    )?;
    let mut consumed = 0;
    for _ in 0..MAXIMUM_TOTAL_POLLS {
        let (bytes, status_receipt, poll_handle) = exchange_with_policy(
            &poll_head,
            &[],
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            &mut session,
            transport,
        )?;
        let result = poll_response(&input.node, &bytes)?;
        if result.id != task_id {
            return Err(MeshyError::Response("Response task identity changed"));
        }
        let phase = task_phase(&result.status)?;
        let normalized = python_strip(&result.status).to_lowercase();
        transport.report_poll_status(
            poll_handle,
            if normalized.is_empty() {
                "Processing"
            } else {
                &normalized
            },
            phase == Phase::Queued,
            result.progress,
        )?;
        if phase == Phase::Completed {
            let mut files = vec![];
            let mut downloaded_bytes = 0usize;
            let mut last_handle = poll_handle;
            for (format, url) in [("glb", result.glb), ("fbx", result.fbx)] {
                if endpoint(&url)? {
                    return Err(MeshyError::Response(
                        "Source file download requires absolute URL",
                    ));
                }
                let download_head = request_head(&url, HttpMethod::Get, credentials, true)?;
                let (bytes, receipt, handle) = exchange_with_policy(
                    &download_head,
                    &[],
                    MAXIMUM_MEDIA_BYTES,
                    DOWNLOAD_RETRY,
                    &mut session,
                    transport,
                )?;
                downloaded_bytes = downloaded_bytes
                    .checked_add(bytes.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                if downloaded_bytes > MAXIMUM_MEDIA_BYTES {
                    return Err(ConformanceError::BodyLimit.into());
                }
                let file = DownloadedFile {
                    task_id: task_id.clone(),
                    format,
                    save_file_name: format!("{task_id}.{format}"),
                    source_url: url,
                    bytes,
                    handle,
                    download_receipt: receipt,
                };
                transport.check_cancelled(Some(handle))?;
                file_owner.complete_download(&file)?;
                transport.check_cancelled(Some(handle))?;
                files.push(file);
                last_handle = handle;
            }
            return Ok(DownloadedResult {
                node: input.node.clone(),
                task_id,
                uploaded,
                files,
                cost_receipt: approval.receipt,
                submission_receipt,
                status_receipt,
                handle: last_handle,
            });
        }
        transport.check_cancelled(Some(poll_handle))?;
        transport.delay(poll_handle, 5000)?;
        if phase != Phase::Queued {
            consumed += 1;
            if consumed >= MAXIMUM_POLLS {
                break;
            }
        }
    }
    Err(ConformanceError::PollLimit.into())
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, MeshyError> {
    let names = match node {
        "MeshyAnimateModelNode" => vec!["STRING", "FILE_3D_GLB", "FILE_3D_FBX"],
        "MeshyRigModelNode" => vec![
            "STRING",
            "MESHY_RIGGED_TASK_ID",
            "FILE_3D_GLB",
            "FILE_3D_FBX",
        ],
        "MeshyTextureNode" => vec!["STRING", "MODEL_TASK_ID", "FILE_3D_GLB", "FILE_3D_FBX"],
        _ if NODES.iter().any(|claim| claim.node_identifier == node) => {
            vec!["STRING", "MESHY_TASK_ID", "FILE_3D_GLB", "FILE_3D_FBX"]
        }
        _ => return Err(MeshyError::Input("Unknown Meshy output node")),
    };
    let registry = TypeRegistry::built_in()
        .map_err(|_| MeshyError::Response("Missing canonical type registry"))?;
    names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let type_id = registry
                .resolve(name)
                .map_err(|_| MeshyError::Response("Missing canonical source type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| MeshyError::Response("Missing canonical source family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect()
}
pub trait MeshyOutputOwner {
    fn materialize(
        &mut self,
        _: &DownloadedResult,
        _: &[ExpectedOutput],
    ) -> Result<Vec<EncodedOutput>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
pub fn materialize_and_publish(
    result: &DownloadedResult,
    transport: &mut dyn MeshyTransport,
    owner: &mut dyn MeshyOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), MeshyError> {
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
        "/../../sim_test_support/fixtures/providers/meshy/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
fn oracle_uploaded(groups: &[UploadGroup]) -> Vec<UploadedGroup> {
    groups
        .iter()
        .map(|group| UploadedGroup {
            role: group.role.clone(),
            urls: group
                .frames
                .iter()
                .map(|frame| {
                    format!(
                        "https://storage.fixture.invalid/{}-{}.png",
                        frame.role, frame.batch_index
                    )
                })
                .collect(),
            completed_requests: vec![],
        })
        .collect()
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture["namespace"] != NAMESPACE
        || fixture["source_sha256"] != SOURCE_SHA256
        || fixture["aliases"] != json!(ALIASES)
        || fixture["nodes"] != json!(NODES)
        || fixture["routes"] != json!(ROUTES)
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = source_oracle()?;
    if source["sources"]["comfy_api_nodes/nodes_meshy.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let groups = upload_groups(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let mut observed = vec![];
        for group in &groups {
            for (index, frame) in group.frames.iter().enumerate() {
                let label = if group.frames.len() > 1 {
                    format!(
                        "{} ({}/{})",
                        group.wait_label,
                        index + 1,
                        group.frames.len()
                    )
                } else {
                    group.wait_label.into()
                };
                observed.push(json!({"role":format!("{}-{}",frame.role,frame.batch_index),"width":frame.width,"height":frame.height,"total_pixels":frame.total_pixels,"declared_content_type":frame.declared_content_type,"encoded_mime_type":frame.encoded_mime_type,"file_name":format!("{}-{}.png",frame.role,frame.batch_index),"wait_label":label}));
            }
        }
        let (path, bytes) = request(&input, &oracle_uploaded(&groups))
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let body: Value =
            serde_json::from_slice(&bytes).map_err(|_| ConformanceError::InvalidRequest)?;
        if json!(observed) != case["uploads"]
            || path != case["request"]["path"]
            || body != case["request"]["body"]
            || case["poll"]["poll_interval"] != 5.0
        {
            return Err(ConformanceError::InvalidRequest);
        }
        let poll = poll_response(
            &input.node,
            &serde_json::to_vec(&case["poll"]["body"])
                .map_err(|_| ConformanceError::InvalidRequest)?,
        )
        .map_err(|_| ConformanceError::InvalidRequest)?;
        if json!(poll.progress) != case["poll"]["progress"]
            || poll.status != case["poll"]["status"]
            || case["downloads"]
                != json!([{"url":poll.glb,"format":"glb","task_id":"task-42","save_file_name":"task-42.glb"},{"url":poll.fbx,"format":"fbx","task_id":"task-42","save_file_name":"task-42.fbx"}])
        {
            return Err(ConformanceError::InvalidRequest);
        }
        covered.insert(input.node);
    }
    for case in source["errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let accepted = upload_groups(&input).is_ok_and(|groups| {
            !groups
                .iter()
                .any(|group| group.first_url && group.frames.is_empty())
                && request(&input, &oracle_uploaded(&groups)).is_ok()
        });
        if accepted || case["requests"] != 0 {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    if covered
        != NODES
            .iter()
            .map(|node| node.node_identifier.to_owned())
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
        statuses: Vec<(String, bool, i64)>,
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
    impl MeshyTransport for TestTransport {
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
            progress: i64,
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
            status: 200,
            bytes,
            headers: vec![],
            receipt: b"retained HTTP completion".to_vec(),
        }
    }
    fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        if !value.len().is_multiple_of(2) {
            return Err("odd source hex".into());
        }
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
            .collect()
    }
    fn baseline(index: usize) -> Result<(Input, Value), Box<dyn Error>> {
        let source = source_oracle()?;
        let case = source["cases"]
            .get(index)
            .ok_or("missing source case")?
            .clone();
        Ok((serde_json::from_value(case["input"].clone())?, case))
    }
    fn prepared(input: &Input) -> Result<Vec<PreparedGroup>, Box<dyn Error>> {
        let source = source_oracle()?;
        let bytes = decode_hex(source["actual_png_hex"].as_str().ok_or("actual PNG")?)?;
        Ok(upload_groups(input)?
            .into_iter()
            .map(|requirement| PreparedGroup {
                frames: requirement
                    .frames
                    .iter()
                    .map(|frame| PreparedPng {
                        file_name: format!("{}-{}.png", frame.role, frame.batch_index),
                        bytes: bytes.clone(),
                    })
                    .collect(),
                requirement,
            })
            .collect())
    }
    struct UploadOwner {
        next_slot: u32,
        calls: Vec<String>,
        replay: bool,
        cross_invocation: bool,
        empty_receipt: bool,
    }
    impl Default for UploadOwner {
        fn default() -> Self {
            Self {
                next_slot: 1,
                calls: vec![],
                replay: false,
                cross_invocation: false,
                empty_receipt: false,
            }
        }
    }
    impl MeshyUploadOwner for UploadOwner {
        fn upload(
            &mut self,
            group: &PreparedGroup,
            transport: &mut dyn MeshyTransport,
        ) -> Result<UploadedGroup, ConformanceError> {
            transport.check_cancelled(None)?;
            let mut completed_requests = vec![];
            let mut urls = vec![];
            for (frame, prepared) in group.requirement.frames.iter().zip(&group.frames) {
                if frame.declared_content_type.is_some()
                    || frame.encoded_mime_type != "image/png"
                    || !prepared.bytes.starts_with(b"\x89PNG\r\n\x1a\n")
                {
                    return Err(ConformanceError::InvalidRequest);
                }
                let role = format!("{}-{}", frame.role, frame.batch_index);
                self.calls.push(role.clone());
                urls.push(format!("https://storage.fixture.invalid/{role}.png"));
                for _ in 0..2 {
                    completed_requests.push(CompletedUploadRequest {
                        handle: StreamHandle {
                            invocation: if self.cross_invocation && self.next_slot == 2 {
                                43
                            } else {
                                42
                            },
                            slot: if self.replay { 1 } else { self.next_slot },
                            generation: 1,
                        },
                        response_receipt: if self.empty_receipt {
                            vec![]
                        } else {
                            b"trusted upload completion".to_vec()
                        },
                    });
                    self.next_slot += 1;
                }
            }
            Ok(UploadedGroup {
                role: group.requirement.role.clone(),
                urls,
                completed_requests,
            })
        }
    }
    #[derive(Default)]
    struct FileOwner {
        observed: Vec<DownloadedFile>,
        fail_format: Option<&'static str>,
    }
    impl MeshyFileOwner for FileOwner {
        fn complete_download(&mut self, file: &DownloadedFile) -> Result<(), ConformanceError> {
            if file.handle.invocation != 42
                || file.download_receipt.is_empty()
                || file.save_file_name != format!("{}.{}", file.task_id, file.format)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            if self.fail_format == Some(file.format) {
                return Err(ConformanceError::HostFailure);
            }
            self.observed.push(file.clone());
            Ok(())
        }
    }
    fn script(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let source = source_oracle()?;
        let path = case["request"]["path"].as_str().ok_or("request path")?;
        let poll = case["poll"]["path"].as_str().ok_or("poll path")?;
        Ok(vec![
            response(
                path,
                HttpMethod::Post,
                serde_json::to_vec(&case["request"]["body"])?,
                serde_json::to_vec(&json!({"result":"task-42"}))?,
            ),
            response(
                poll,
                HttpMethod::Get,
                vec![],
                serde_json::to_vec(&case["poll"]["body"])?,
            ),
            response(
                "https://output.fixture.invalid/model.glb",
                HttpMethod::Get,
                vec![],
                decode_hex(source["file_probe"][0]["bytes_hex"].as_str().ok_or("GLB")?)?,
            ),
            response(
                "https://output.fixture.invalid/model.fbx",
                HttpMethod::Get,
                vec![],
                decode_hex(source["file_probe"][1]["bytes_hex"].as_str().ok_or("FBX")?)?,
            ),
        ])
    }
    fn transport_for(script: Vec<Response>, groups: &[PreparedGroup]) -> TestTransport {
        let mut transport = TestTransport::new(script);
        transport.next_slot = 1 + groups
            .iter()
            .map(|group| group.frames.len() as u32 * 2)
            .sum::<u32>();
        transport
    }
    #[test]
    fn pinned_execute_and_real_upload_helper_oracle_covers_all_nodes_and_order()
    -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/meshy/conformance.fixture"
        )))?;
        let source = source_oracle()?;
        assert_eq!(source["cases"].as_array().ok_or("cases")?.len(), 105);
        assert_eq!(source["errors"].as_array().ok_or("errors")?.len(), 19);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "synthetic_prefix_tombstone")
                .count(),
            6
        );
        assert_eq!(NODES.len(), 7);
        assert!(ALIASES.is_empty());
        Ok(())
    }
    #[test]
    fn all_seven_requests_and_exact_glb_fbx_bytes_reach_file_owner_in_save_order()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for index in 0..7 {
            let (input, case) = baseline(index)?;
            let groups = prepared(&input)?;
            let mut transport = transport_for(script(&case)?, &groups);
            let mut uploads = UploadOwner::default();
            let mut files = FileOwner::default();
            let result = invoke(
                &input,
                &groups,
                credentials(),
                900_000,
                &mut transport,
                &mut uploads,
                &mut files,
            )?;
            assert_eq!(result.task_id, "task-42");
            assert_eq!(result.files.len(), 2);
            assert_eq!(files.observed.len(), 2);
            assert_eq!(transport.costs, 1);
            assert_eq!(
                transport.deadlines,
                vec![Some(3_600_000), Some(120_000), None, None]
            );
            assert_eq!(transport.statuses, vec![("succeeded".into(), false, 100)]);
            let observed: Vec<_> = uploads.calls.iter().map(|value| json!(value)).collect();
            let expected: Vec<_> = case["uploads"]
                .as_array()
                .ok_or("uploads")?
                .iter()
                .map(|upload| upload["role"].clone())
                .collect();
            assert_eq!(observed, expected);
            for (index, file) in files.observed.iter().enumerate() {
                let probe = &source["file_probe"][index];
                assert_eq!(
                    file.bytes,
                    decode_hex(
                        probe["saved_bytes_hex"]
                            .as_str()
                            .ok_or("saved source bytes")?
                    )?
                );
                assert_eq!(file.format, probe["format"]);
                assert_eq!(file.save_file_name, probe["file_name"]);
                assert!(!file.download_receipt.is_empty());
            }
            for head in transport
                .heads
                .iter()
                .filter(|head| head.path.starts_with("https://"))
            {
                assert!(head.secret_id.is_none());
                assert!(head.headers.iter().all(|header| header.name == "Accept"));
            }
            transport.finish()?;
        }
        Ok(())
    }
    #[test]
    fn actual_source_file_helper_rewinds_and_exposes_partial_save_residue()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for file in source["file_probe"].as_array().ok_or("file probes")? {
            assert_eq!(file["position_after_creation"], 0);
            assert_eq!(file["disk_backed"], false);
            assert_eq!(file["bytes_hex"], file["saved_bytes_hex"]);
        }
        assert_eq!(
            source["file_failure_residue"],
            json!({"glb_saved":true,"fbx_saved":false})
        );
        Ok(())
    }
    #[test]
    fn source_failure_timing_retains_uploaded_frames_before_api_model_errors()
    -> Result<(), Box<dyn Error>> {
        for case in source_oracle()?["errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let groups = match prepared(&input) {
                Ok(groups) => groups,
                Err(_) => {
                    assert!(case["uploads"].as_array().ok_or("uploads")?.is_empty());
                    continue;
                }
            };
            let mut transport = transport_for(vec![], &groups);
            let mut uploads = UploadOwner::default();
            let mut files = FileOwner::default();
            assert!(
                invoke(
                    &input,
                    &groups,
                    credentials(),
                    1,
                    &mut transport,
                    &mut uploads,
                    &mut files
                )
                .is_err()
            );
            let expected: Vec<_> = case["uploads"]
                .as_array()
                .ok_or("uploads")?
                .iter()
                .map(|upload| upload["role"].as_str().ok_or("role"))
                .collect::<Result<_, _>>()?;
            assert_eq!(
                uploads.calls.iter().map(String::as_str).collect::<Vec<_>>(),
                expected
            );
            assert!(transport.heads.is_empty());
            assert!(files.observed.is_empty());
        }
        Ok(())
    }
    #[test]
    fn first_and_second_file_owner_failures_stop_later_transport() -> Result<(), Box<dyn Error>> {
        let (input, case) = baseline(0)?;
        let groups = prepared(&input)?;
        for format in ["glb", "fbx"] {
            let mut transport = transport_for(script(&case)?, &groups);
            let mut uploads = UploadOwner::default();
            let mut files = FileOwner {
                fail_format: Some(format),
                ..FileOwner::default()
            };
            assert!(matches!(
                invoke(
                    &input,
                    &groups,
                    credentials(),
                    1,
                    &mut transport,
                    &mut uploads,
                    &mut files
                ),
                Err(MeshyError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(files.observed.len(), usize::from(format == "fbx"));
            assert_eq!(transport.heads.len(), if format == "glb" { 3 } else { 4 });
        }
        Ok(())
    }
    #[test]
    fn exact_registered_model_id_and_artifact_output_layouts() -> Result<(), Box<dyn Error>> {
        use crate::common::ValueFamily;
        for (node, name) in [
            ("MeshyTextToModelNode", "sim:meshy-task-id@1"),
            ("MeshyRigModelNode", "sim:meshy-rigged-task-id@1"),
            ("MeshyTextureNode", "sim:model-task-id@1"),
        ] {
            let expected = expected_outputs(node)?;
            assert_eq!(expected.len(), 4);
            assert_eq!(expected[1].type_id.to_string(), name);
            assert_eq!(expected[1].family, ValueFamily::Model);
            assert_eq!(expected[2].family, ValueFamily::Artifact);
            assert_eq!(expected[3].family, ValueFamily::Artifact);
        }
        assert_eq!(expected_outputs("MeshyAnimateModelNode")?.len(), 3);
        assert!(expected_outputs("unknown").is_err());
        Ok(())
    }
    #[test]
    fn unbound_native_owners_cancel_and_replay_leave_zero_publication() -> Result<(), Box<dyn Error>>
    {
        struct MissingUpload;
        impl MeshyUploadOwner for MissingUpload {}
        struct MissingOutput;
        impl MeshyOutputOwner for MissingOutput {}
        #[derive(Default)]
        struct Sink {
            count: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(
                &mut self,
                _: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.count += 1;
                Ok(())
            }
        }
        let (input, case) = baseline(2)?;
        let groups = prepared(&input)?;
        let mut transport = transport_for(vec![], &groups);
        let mut files = FileOwner::default();
        assert!(matches!(
            invoke(
                &input,
                &groups,
                credentials(),
                1,
                &mut transport,
                &mut MissingUpload,
                &mut files
            ),
            Err(MeshyError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        for (replay, cross_invocation, empty_receipt) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let mut uploads = UploadOwner {
                replay,
                cross_invocation,
                empty_receipt,
                ..UploadOwner::default()
            };
            let mut transport = transport_for(vec![], &groups);
            assert!(
                invoke(
                    &input,
                    &groups,
                    credentials(),
                    1,
                    &mut transport,
                    &mut uploads,
                    &mut files
                )
                .is_err()
            );
            assert!(transport.heads.is_empty());
        }
        let mut transport = transport_for(script(&case)?, &groups);
        let mut uploads = UploadOwner::default();
        let result = invoke(
            &input,
            &groups,
            credentials(),
            1,
            &mut transport,
            &mut uploads,
            &mut files,
        )?;
        let mut sink = Sink::default();
        assert!(matches!(
            materialize_and_publish(&result, &mut transport, &mut MissingOutput, &mut sink),
            Err(MeshyError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(sink.count, 0);
        transport.cancelled = true;
        assert!(matches!(
            materialize_and_publish(&result, &mut transport, &mut MissingOutput, &mut sink),
            Err(MeshyError::Transport(ConformanceError::Cancelled))
        ));
        assert_eq!(sink.count, 0);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn malformed_changed_identity_cost_and_worker_failures_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let (input, case) = baseline(0)?;
        let groups = prepared(&input)?;
        for error in [ConformanceError::Cancelled, ConformanceError::WorkerLost] {
            let mut transport = transport_for(vec![], &groups);
            transport.failure = Some(error.clone());
            let mut uploads = UploadOwner::default();
            let mut files = FileOwner::default();
            assert!(
                matches!(invoke(&input,&groups,credentials(),1,&mut transport,&mut uploads,&mut files),Err(MeshyError::Transport(value)) if value==error)
            );
            assert!(transport.heads.is_empty());
        }
        for mode in 0..7 {
            let mut replies = script(&case)?;
            match mode {
                0 => replies[0].bytes = b"malformed".to_vec(),
                1 => replies[0].bytes = serde_json::to_vec(&json!({"result":"../escape"}))?,
                2 => {
                    let mut value = case["poll"]["body"].clone();
                    value["id"] = json!("other");
                    replies[1].bytes = serde_json::to_vec(&value)?;
                }
                3 => replies[1].receipt = vec![],
                4 => {
                    replies[1].headers = vec![Header {
                        name: "bad\nname".into(),
                        value: "x".into(),
                    }]
                }
                5 => replies[1].bytes = vec![0; MAXIMUM_CHUNK_BYTES + 1],
                _ => {}
            }
            let mut transport = transport_for(replies, &groups);
            transport.reject_cost = mode == 6;
            let mut uploads = UploadOwner::default();
            let mut files = FileOwner::default();
            assert!(
                invoke(
                    &input,
                    &groups,
                    credentials(),
                    1,
                    &mut transport,
                    &mut uploads,
                    &mut files
                )
                .is_err()
            );
            assert!(files.observed.is_empty());
        }
        Ok(())
    }
    #[test]
    fn source_queued_states_skip_processing_budget_and_poll_every_five_seconds()
    -> Result<(), Box<dyn Error>> {
        let (input, case) = baseline(0)?;
        let groups = prepared(&input)?;
        let mut replies = script(&case)?;
        replies.truncate(1);
        for index in 0..483 {
            let mut body = case["poll"]["body"].clone();
            body["status"] = json!(if index < 3 { "IN_QUEUE" } else { "IN_PROGRESS" });
            body["progress"] = json!(25);
            replies.push(response(
                case["poll"]["path"].as_str().ok_or("poll path")?,
                HttpMethod::Get,
                vec![],
                serde_json::to_vec(&body)?,
            ));
        }
        let mut transport = transport_for(replies, &groups);
        let mut uploads = UploadOwner::default();
        let mut files = FileOwner::default();
        assert!(matches!(
            invoke(
                &input,
                &groups,
                credentials(),
                1,
                &mut transport,
                &mut uploads,
                &mut files
            ),
            Err(MeshyError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.statuses.len(), 483);
        assert_eq!(
            transport
                .statuses
                .iter()
                .filter(|(_, queued, _)| *queued)
                .count(),
            3
        );
        assert_eq!(transport.delays, vec![5000; 483]);
        assert!(files.observed.is_empty());
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn retry_budgets_deadlines_and_fresh_handle_failure_are_exact() -> Result<(), Box<dyn Error>> {
        for (path, policy, limit) in [
            ("/proxy/meshy/test", GENERATION_RETRY, 3),
            ("/proxy/meshy/test/x", POLL_RETRY, 10),
            ("https://file.fixture.invalid/model", DOWNLOAD_RETRY, 5),
        ] {
            let head = request_head(path, HttpMethod::Get, credentials(), policy.download)?;
            for status in [503, 429] {
                let limit = if status == 429 && !policy.download {
                    16
                } else {
                    limit
                };
                let mut replies = vec![];
                for _ in 0..=limit {
                    let mut reply = response(path, HttpMethod::Get, vec![], b"busy".to_vec());
                    reply.status = status;
                    replies.push(reply);
                }
                let mut transport = TestTransport::new(replies);
                assert!(
                    matches!(exchange_with_policy(&head,&[],1024,policy,&mut Session::default(),&mut transport),Err(MeshyError::Http{status:value,..}) if value==status)
                );
                assert_eq!(transport.delays.len(), limit);
                assert!(
                    transport
                        .deadlines
                        .iter()
                        .all(|value| *value == policy.timeout_milliseconds)
                );
                transport.finish()?;
            }
        }
        let head = request_head("/proxy/meshy/test", HttpMethod::Get, credentials(), false)?;
        let mut reply = response(&head.path, HttpMethod::Get, vec![], b"busy".to_vec());
        reply.status = 503;
        let mut transport = TestTransport::new(vec![reply]);
        transport.stale_retry = true;
        assert!(matches!(
            exchange_with_policy(
                &head,
                &[],
                1024,
                GENERATION_RETRY,
                &mut Session::default(),
                &mut transport
            ),
            Err(MeshyError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
}
