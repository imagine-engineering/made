use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, ProviderTransport, RequestHead, ResponseEvent, StreamHandle, TypeRegistry,
    publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
pub const NAMESPACE: &str = "zed.sim.provider.bria";
pub const SOURCE_SHA256: &str = "267c7bf3cde3d9e57242de1782ee4fcb3394b9c845d79d0f1170379545d67c4b";
pub const IMAGE_EDIT_PATH: &str = "/proxy/bria/v2/image/edit";
pub const MAXIMUM_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
const MAXIMUM_POLLS: u32 = 480;
const MAXIMUM_TOTAL_POLLS: u32 = 4096;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0024",
        node_identifier: "BriaImageEditNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0025",
        node_identifier: "BriaRemoveImageBackground",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0026",
        node_identifier: "BriaRemoveVideoBackground",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0027",
        node_identifier: "BriaTransparentVideoBackground",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0028",
        node_identifier: "BriaVideoGreenScreen",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0029",
        node_identifier: "BriaVideoReplaceBackground",
    },
];
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0218",
        method: "POST",
        path: IMAGE_EDIT_PATH,
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0014",
        method: "POST",
        path: "/proxy/bria/v2/image/edit/remove_background",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0015",
        method: "GET",
        path: "/proxy/bria/v2/status/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0016",
        method: "GET",
        path: "/proxy/bria/v2/status/{response.request_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0017",
        method: "POST",
        path: "/proxy/bria/v2/video/edit/green_screen",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0018",
        method: "POST",
        path: "/proxy/bria/v2/video/edit/remove_background",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0019",
        method: "POST",
        path: "/proxy/bria/v2/video/edit/replace_background",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum BriaError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    Http {
        status: u16,
        retry_after: Option<String>,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for BriaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => write!(formatter, "Bria task failed: {message}"),
            Self::Http { status, .. } => write!(formatter, "Bria HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for BriaError {}
impl From<serde_json::Error> for BriaError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
impl From<ConformanceError> for BriaError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub model: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub structured_prompt: String,
    pub seed: u32,
    pub guidance_scale: f64,
    pub steps: u32,
    pub moderation: BTreeMap<String, bool>,
    pub mask: bool,
    pub background_color: String,
    pub green_shade: String,
    pub background_image: bool,
    pub background_video: bool,
    pub duration_seconds: f64,
    pub background_duration_seconds: f64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadRole {
    Image,
    Mask,
    Video,
    BackgroundImage,
    BackgroundVideo,
}
impl UploadRole {
    pub fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Mask => "mask",
            Self::Video => "video",
            Self::BackgroundImage => "background_image",
            Self::BackgroundVideo => "background_video",
        }
    }
    pub fn wait_label(self) -> &'static str {
        match self {
            Self::Image => "Uploading image",
            Self::Mask => "Uploading mask",
            Self::Video => "Uploading",
            Self::BackgroundImage | Self::BackgroundVideo => "Uploading background",
        }
    }
    pub fn is_video(self) -> bool {
        matches!(self, Self::Video | Self::BackgroundVideo)
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedMedia {
    pub role: UploadRole,
    pub file_name: String,
    pub content_type: Option<String>,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct UploadedMedia {
    pub role: UploadRole,
    pub url: String,
}
pub fn upload_roles(input: &Input) -> Result<Vec<UploadRole>, BriaError> {
    if input.seed > 2147483647 {
        return Err(BriaError::Input("Bria seed exceeds its schema"));
    }
    for text in [
        &input.prompt,
        &input.negative_prompt,
        &input.structured_prompt,
    ] {
        if text.len() > 65536 {
            return Err(BriaError::Input("Bria prompt exceeds its bound"));
        }
    }
    match input.node.as_str() {
        "BriaImageEditNode" => {
            if input.prompt.is_empty() && input.structured_prompt.is_empty() {
                return Err(BriaError::Input(
                    "One of prompt or structured_prompt is required to be non-empty.",
                ));
            }
            if input.model != "FIBO"
                || input.seed == 0
                || !input.guidance_scale.is_finite()
                || !(3.0..=5.0).contains(&input.guidance_scale)
                || !(20..=50).contains(&input.steps)
            {
                return Err(BriaError::Input("Invalid Bria image edit controls"));
            }
            let mut roles = vec![];
            if input.mask {
                roles.push(UploadRole::Mask);
            }
            roles.push(UploadRole::Image);
            Ok(roles)
        }
        "BriaRemoveImageBackground" => Ok(vec![UploadRole::Image]),
        "BriaVideoReplaceBackground" => {
            if input.background_image == input.background_video {
                return Err(BriaError::Input(
                    "Provide either a background image or a background video, not both.",
                ));
            }
            duration(input.duration_seconds)?;
            if input.background_video {
                duration(input.background_duration_seconds)?;
                Ok(vec![UploadRole::BackgroundVideo, UploadRole::Video])
            } else {
                Ok(vec![UploadRole::BackgroundImage, UploadRole::Video])
            }
        }
        "BriaRemoveVideoBackground" | "BriaTransparentVideoBackground" | "BriaVideoGreenScreen" => {
            duration(input.duration_seconds)?;
            if input.node == "BriaRemoveVideoBackground"
                && !COLORS.contains(&input.background_color.as_str())
            {
                return Err(BriaError::Input("Invalid Bria background color"));
            }
            if input.node == "BriaVideoGreenScreen" && !SHADES.contains(&input.green_shade.as_str())
            {
                return Err(BriaError::Input("Invalid Bria green shade"));
            }
            Ok(vec![UploadRole::Video])
        }
        _ => Err(BriaError::Input("Unknown Bria node")),
    }
}
const COLORS: &[&str] = &[
    "Black", "White", "Gray", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "Orange",
];
const SHADES: &[&str] = &["broadcast_green", "chroma_green", "blue_screen"];
fn duration(seconds: f64) -> Result<(), BriaError> {
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 60.0001 {
        return Err(BriaError::Input("Bria video exceeds its duration bound"));
    }
    Ok(())
}
pub fn price_badge_microunits(input: &Input) -> Result<u64, BriaError> {
    upload_roles(input)?;
    Ok(match input.node.as_str() {
        "BriaImageEditNode" => 40000,
        "BriaRemoveImageBackground" => 18000,
        _ => (input.duration_seconds * 4200.0).ceil() as u64,
    })
}
pub fn request(
    input: &Input,
    uploads: &[UploadedMedia],
) -> Result<(&'static str, Vec<u8>), BriaError> {
    let roles = upload_roles(input)?;
    if uploads.len() != roles.len()
        || uploads
            .iter()
            .zip(roles)
            .any(|(upload, role)| upload.role != role)
    {
        return Err(BriaError::Input("Bria upload order differs from source"));
    }
    for upload in uploads {
        if endpoint(&upload.url)? {
            return Err(BriaError::Input("Bria upload URL must be absolute"));
        }
    }
    let url = |role| {
        uploads
            .iter()
            .find(|upload| upload.role == role)
            .map(|upload| upload.url.clone())
            .ok_or(BriaError::Input("Missing Bria upload"))
    };
    let mut body = json!({"seed":input.seed});
    let path = match input.node.as_str() {
        "BriaImageEditNode" => {
            body["images"] = json!([url(UploadRole::Image)?]);
            if !input.prompt.is_empty() {
                body["instruction"] = input.prompt.clone().into();
            }
            if !input.structured_prompt.is_empty() {
                body["structured_instruction"] = input.structured_prompt.clone().into();
            }
            if !input.negative_prompt.is_empty() {
                body["negative_prompt"] = input.negative_prompt.clone().into();
            }
            if input.mask {
                body["mask"] = url(UploadRole::Mask)?.into();
            }
            body["guidance_scale"] = input.guidance_scale.into();
            body["model_version"] = input.model.clone().into();
            body["steps_num"] = input.steps.into();
            body["ip_signal"] = false.into();
            body["prompt_content_moderation"] = input
                .moderation
                .get("prompt_content_moderation")
                .copied()
                .unwrap_or(false)
                .into();
            body["visual_input_content_moderation"] = input
                .moderation
                .get("visual_input_moderation")
                .copied()
                .unwrap_or(false)
                .into();
            body["visual_output_content_moderation"] = input
                .moderation
                .get("visual_output_moderation")
                .copied()
                .unwrap_or(false)
                .into();
            IMAGE_EDIT_PATH
        }
        "BriaRemoveImageBackground" => {
            body["image"] = url(UploadRole::Image)?.into();
            body["sync"] = false.into();
            body["visual_input_content_moderation"] = input
                .moderation
                .get("visual_input_moderation")
                .copied()
                .unwrap_or(false)
                .into();
            body["visual_output_content_moderation"] = input
                .moderation
                .get("visual_output_moderation")
                .copied()
                .unwrap_or(false)
                .into();
            "/proxy/bria/v2/image/edit/remove_background"
        }
        _ => {
            body["video"] = url(UploadRole::Video)?.into();
            body["preserve_audio"] = true.into();
            body["output_container_and_codec"] =
                if input.node == "BriaTransparentVideoBackground" {
                    "webm_vp9"
                } else {
                    "mp4_h264"
                }
                .into();
            match input.node.as_str() {
                "BriaRemoveVideoBackground" => {
                    body["background_color"] = input.background_color.clone().into();
                    "/proxy/bria/v2/video/edit/remove_background"
                }
                "BriaTransparentVideoBackground" => {
                    body["background_color"] = "Transparent".into();
                    "/proxy/bria/v2/video/edit/remove_background"
                }
                "BriaVideoGreenScreen" => {
                    body["green_shade"] = input.green_shade.clone().into();
                    "/proxy/bria/v2/video/edit/green_screen"
                }
                "BriaVideoReplaceBackground" => {
                    body["background_url"] = url(if input.background_image {
                        UploadRole::BackgroundImage
                    } else {
                        UploadRole::BackgroundVideo
                    })?
                    .into();
                    "/proxy/bria/v2/video/edit/replace_background"
                }
                _ => return Err(BriaError::Input("Unknown Bria node")),
            }
        }
    };
    let bytes = serde_json::to_vec(&body)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(BriaError::Input("Bria request exceeds its bound"));
    }
    Ok((path, bytes))
}
#[derive(Debug, Deserialize)]
pub struct SubmittedTask {
    pub request_id: String,
    pub status_url: String,
    pub warning: Option<String>,
}
pub fn submitted_task(bytes: &[u8]) -> Result<SubmittedTask, BriaError> {
    bounded_response(bytes)?;
    let task: SubmittedTask = serde_json::from_slice(bytes)?;
    if task.request_id.is_empty()
        || task.request_id.len() > 1024
        || !task
            .request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        || task.status_url.len() > 8192
        || task
            .warning
            .as_ref()
            .is_some_and(|warning| warning.len() > 65536)
    {
        return Err(BriaError::Response("Invalid Bria submission"));
    }
    Ok(task)
}
#[derive(Debug, Deserialize)]
pub struct TaskStatus {
    pub status: String,
    pub result: Option<serde_json::Map<String, Value>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Completed,
    Queued,
    Processing,
}
fn source_trim(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
pub fn task_status(input: &Input, bytes: &[u8]) -> Result<(TaskStatus, Phase), BriaError> {
    bounded_response(bytes)?;
    let task: TaskStatus = serde_json::from_slice(bytes)?;
    if task.status.len() > 1024 {
        return Err(BriaError::Response("Bria status exceeds its bound"));
    }
    if let Some(result) = &task.result {
        let field =
            if input.node == "BriaImageEditNode" || input.node == "BriaRemoveImageBackground" {
                "image_url"
            } else {
                "video_url"
            };
        let url = result
            .get(field)
            .and_then(Value::as_str)
            .ok_or(BriaError::Response("Missing Bria media URL"))?;
        if endpoint(url)? {
            return Err(BriaError::Response(
                "Bria downloads require an absolute URL",
            ));
        }
        if input.node == "BriaImageEditNode"
            && result
                .get("structured_prompt")
                .and_then(Value::as_str)
                .is_none()
        {
            return Err(BriaError::Response("Missing Bria structured prompt"));
        }
    }
    let status = source_trim(&task.status).to_lowercase();
    let phase =
        match status.as_str() {
            "succeeded" | "succeed" | "success" | "completed" | "finished" | "done"
            | "complete" => Phase::Completed,
            "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error" => {
                return Err(BriaError::Provider(task.status));
            }
            "created" | "queued" | "queueing" | "submitted" | "initializing" | "wait"
            | "in_queue" => Phase::Queued,
            _ => Phase::Processing,
        };
    if phase == Phase::Completed && task.result.is_none() {
        return Err(BriaError::Response("Completed Bria task has no result"));
    }
    Ok((task, phase))
}
fn bounded_response(bytes: &[u8]) -> Result<(), BriaError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(BriaError::Response("Bria response exceeds its bound"));
    }
    Ok(())
}
fn endpoint(value: &str) -> Result<bool, BriaError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(BriaError::Response("Invalid Bria response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(BriaError::Response("Invalid Bria response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(BriaError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(BriaError::Response("Invalid Bria endpoint authority"));
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
) -> Result<RequestHead, BriaError> {
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
            return Err(BriaError::Input("Missing Comfy API credentials"));
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

pub trait BriaTransport: ProviderTransport {
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
    fn upload_prepared(
        &mut self,
        handle: StreamHandle,
        media: &PreparedMedia,
        wait_label: &str,
    ) -> Result<UploadedMedia, ConformanceError>;
}
#[derive(Debug)]
pub struct DownloadedMedia {
    pub node: String,
    pub bytes: Vec<u8>,
    pub response_receipt: Vec<u8>,
    pub source_url: String,
    pub structured_prompt: Option<String>,
    pub warning: Option<String>,
    pub source_status_url: String,
}
pub fn invoke(
    input: &Input,
    media: &[PreparedMedia],
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn BriaTransport,
) -> Result<DownloadedMedia, BriaError> {
    let roles = upload_roles(input)?;
    let mut size = 0usize;
    if roles.len() != media.len() {
        return Err(BriaError::Input("Missing Bria prepared media"));
    }
    for (role, media) in roles.iter().zip(media) {
        if *role != media.role
            || media.bytes.is_empty()
            || media.file_name.is_empty()
            || media.file_name.len() > 1024
            || media
                .file_name
                .chars()
                .any(|character| character.is_control() || matches!(character, '/' | '\\'))
            || media.content_type.as_deref()
                != if role.is_video() {
                    Some("video/mp4")
                } else {
                    None
                }
        {
            return Err(BriaError::Input("Invalid canonical Bria media preparation"));
        }
        size = size
            .checked_add(media.bytes.len())
            .ok_or(BriaError::Input("Bria media byte overflow"))?;
    }
    if size > MAXIMUM_MEDIA_BYTES {
        return Err(BriaError::Input(
            "Bria prepared media exceeds its aggregate bound",
        ));
    }
    let minimum = price_badge_microunits(input)?
        .checked_mul(20)
        .ok_or(ConformanceError::InvalidCost)?;
    if maximum_cost_microunits < minimum {
        return Err(ConformanceError::InvalidCost.into());
    }
    let preliminary = media
        .iter()
        .map(|media| UploadedMedia {
            role: media.role,
            url: format!("https://preflight.invalid/{}", media.role.name()),
        })
        .collect::<Vec<_>>();
    let (path, _) = request(input, &preliminary)?;
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    transport.check_cancelled(None)?;
    let handle = transport.start_request(&head)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "bria.generation.maximum".into(),
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
    let mut uploads = vec![];
    for media in media {
        transport.check_cancelled(Some(handle))?;
        let upload = transport.upload_prepared(handle, media, media.role.wait_label())?;
        if upload.role != media.role || endpoint(&upload.url)? {
            return Err(BriaError::Response(
                "Invalid canonical Bria upload response",
            ));
        }
        uploads.push(upload);
    }
    let (_, body) = request(input, &uploads)?;
    let (bytes, _, _) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    let task = submitted_task(&bytes)?;
    let poll_path = format!("/proxy/bria/v2/status/{}", task.request_id);
    let mut consumed = 0;
    for _ in 0..MAXIMUM_TOTAL_POLLS {
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
        let (bytes, _, poll_handle) = exchange_with_policy(
            &head,
            &[],
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            transport,
        )?;
        let (status, phase) = task_status(input, &bytes)?;
        if phase == Phase::Completed {
            let result = status
                .result
                .ok_or(BriaError::Response("Missing Bria result"))?;
            let field =
                if input.node == "BriaImageEditNode" || input.node == "BriaRemoveImageBackground" {
                    "image_url"
                } else {
                    "video_url"
                };
            let url = result
                .get(field)
                .and_then(Value::as_str)
                .ok_or(BriaError::Response("Missing Bria download URL"))?;
            let head = request_head(url, HttpMethod::Get, credentials, true)?;
            let (bytes, receipt, _) = exchange_with_policy(
                &head,
                &[],
                None,
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                transport,
            )?;
            if bytes.is_empty() {
                return Err(BriaError::Response("Empty Bria media download"));
            }
            return Ok(DownloadedMedia {
                node: input.node.clone(),
                bytes,
                response_receipt: receipt,
                source_url: url.into(),
                structured_prompt: result
                    .get("structured_prompt")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                warning: task.warning,
                source_status_url: task.status_url,
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
#[derive(Clone, Copy)]
struct RetryPolicy {
    ordinary_limit: u32,
    rate_limit: u32,
    backoff: f64,
    download: bool,
}
const GENERATION_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 3,
    rate_limit: 16,
    backoff: 2.0,
    download: false,
};
const POLL_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 10,
    rate_limit: 16,
    backoff: 1.4,
    download: false,
};
const DOWNLOAD_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 5,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
};
#[cfg(test)]
fn exchange(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    transport: &mut dyn BriaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BriaError> {
    exchange_with_policy(head, body, cost, maximum_bytes, GENERATION_RETRY, transport)
}
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn BriaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BriaError> {
    transport.check_cancelled(None)?;
    let handle = transport.start_request(head)?;
    exchange_existing(head, body, cost, maximum_bytes, policy, handle, transport)
}
fn exchange_existing(
    _head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    mut handle: StreamHandle,
    transport: &mut dyn BriaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BriaError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(BriaError::Http {
                status,
                retry_after,
            }) => {
                let delay = if status == 429
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
                    return Err(BriaError::Http {
                        status,
                        retry_after,
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
                if fresh == handle || fresh.invocation != handle.invocation {
                    return Err(ConformanceError::InvalidHandle.into());
                }
                handle = fresh;
            }
            result => return result,
        }
    }
}

fn exchange_on_handle(
    handle: StreamHandle,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    transport: &mut dyn BriaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BriaError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "bria.generation.maximum".into(),
                currency: "USD".into(),
                maximum_microunits,
            },
        )?;
        if !approval.accepted
            || approval.approved_microunits < maximum_microunits
            || approval.receipt.is_empty()
            || approval.receipt.len() > MAXIMUM_RECEIPT_BYTES
        {
            return Err(ConformanceError::InvalidCost.into());
        }
    }
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
                if headers.len() > 16
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
                    return Err(BriaError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
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

pub trait BriaMediaMaterializer {
    fn decode_media(
        &mut self,
        media: &DownloadedMedia,
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub fn invoke_and_materialize(
    input: &Input,
    media: &[PreparedMedia],
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn BriaTransport,
    materializer: &mut dyn BriaMediaMaterializer,
    sink: &mut dyn MaterializationSink,
) -> Result<(), BriaError> {
    let result = invoke(
        input,
        media,
        credentials,
        maximum_cost_microunits,
        transport,
    )?;
    transport.check_cancelled(None)?;
    let outputs = materializer.decode_media(&result)?;
    let registry =
        TypeRegistry::built_in().map_err(|_| BriaError::Response("Missing type registry"))?;
    let aliases = match input.node.as_str() {
        "BriaImageEditNode" => vec!["IMAGE", "STRING"],
        "BriaTransparentVideoBackground" => vec!["IMAGE", "MASK"],
        "BriaRemoveImageBackground" => vec!["IMAGE"],
        _ => vec!["VIDEO"],
    };
    let expected = aliases
        .into_iter()
        .enumerate()
        .map(|(index, alias)| {
            let type_id = registry
                .resolve(alias)
                .map_err(|_| BriaError::Response("Missing output type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| BriaError::Response("Missing output family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect::<Result<Vec<_>, BriaError>>()?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":[],"nodes":NODES,"routes":ROUTES,"scope":"source request and bounded transport; canonical media integration pending"})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let oracle: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/bria/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)?;
    if oracle["source_sha256"]["comfy_api_nodes/nodes_bria.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in oracle["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let uploads = upload_roles(&input)
            .map_err(|_| ConformanceError::InvalidRequest)?
            .into_iter()
            .map(|role| UploadedMedia {
                role,
                url: format!("https://storage.fixture.invalid/{}", role.name()),
            })
            .collect::<Vec<_>>();
        let (path, bytes) =
            request(&input, &uploads).map_err(|_| ConformanceError::InvalidRequest)?;
        let body: Value =
            serde_json::from_slice(&bytes).map_err(|_| ConformanceError::InvalidRequest)?;
        let expected_path = case["request"]["path"]
            .as_str()
            .ok_or(ConformanceError::InvalidRequest)?;
        if path != format!("/{}", expected_path.trim_start_matches('/'))
            || case["request"]["body"] != body
            || case["request"]["method"] != "POST"
        {
            return Err(ConformanceError::InvalidRequest);
        }
        let roles = upload_roles(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let observed = case["uploads"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if roles.len() != observed.len()
            || roles.iter().zip(observed).any(|(role, upload)| {
                upload["role"] != role.name() || upload["wait_label"] != role.wait_label()
            })
        {
            return Err(ConformanceError::InvalidRequest);
        }
        covered.insert(input.node);
    }
    for case in oracle["preflight_errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        if upload_roles(&input).is_ok() || case["requests"] != 0 || case["uploads"] != 0 {
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
    use crate::common::{CostResponse, Progress, ScriptStep, ScriptedActuator};
    use std::collections::VecDeque;
    struct TestTransport {
        actuator: ScriptedActuator,
        head: Option<RequestHead>,
        delays: Vec<u64>,
        uploads: VecDeque<PreparedMedia>,
        observed_uploads: Vec<UploadRole>,
    }
    impl TestTransport {
        fn new(steps: Vec<ScriptStep>, uploads: Vec<PreparedMedia>) -> Self {
            Self {
                actuator: ScriptedActuator::new(steps),
                head: None,
                delays: vec![],
                uploads: uploads.into(),
                observed_uploads: vec![],
            }
        }
        fn finish(self) -> Result<(), ConformanceError> {
            if !self.uploads.is_empty() {
                return Err(ConformanceError::HostFailure);
            }
            self.actuator.finish()?;
            Ok(())
        }
    }
    impl ProviderTransport for TestTransport {
        fn check_cancelled(
            &mut self,
            handle: Option<StreamHandle>,
        ) -> Result<(), ConformanceError> {
            self.actuator.check_cancelled(handle)
        }
        fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
            self.head = Some(head.clone());
            self.actuator.start_request(head)
        }
        fn write_request_chunk(
            &mut self,
            handle: StreamHandle,
            sequence: u64,
            bytes: &[u8],
            end: bool,
        ) -> Result<(), ConformanceError> {
            self.actuator
                .write_request_chunk(handle, sequence, bytes, end)
        }
        fn upload(&mut self, handle: StreamHandle, bytes: &[u8]) -> Result<(), ConformanceError> {
            self.actuator.upload(handle, bytes)
        }
        fn create_storage(
            &mut self,
            handle: StreamHandle,
            request: &crate::common::StorageCreateRequest,
        ) -> Result<crate::common::StorageCreateResponse, ConformanceError> {
            self.actuator.create_storage(handle, request)
        }
        fn put_signed(
            &mut self,
            handle: StreamHandle,
            request: &crate::common::SignedPutRequest,
        ) -> Result<(), ConformanceError> {
            self.actuator.put_signed(handle, request)
        }
        fn request_cost(
            &mut self,
            handle: StreamHandle,
            request: &CostRequest,
        ) -> Result<CostResponse, ConformanceError> {
            self.actuator.request_cost(handle, request)
        }
        fn report_progress(
            &mut self,
            handle: StreamHandle,
            progress: &Progress,
        ) -> Result<(), ConformanceError> {
            self.actuator.report_progress(handle, progress)
        }
        fn wait_response(
            &mut self,
            handle: StreamHandle,
            after_sequence: Option<u64>,
            timeout: u64,
        ) -> Result<ResponseEvent, ConformanceError> {
            self.actuator.wait_response(handle, after_sequence, timeout)
        }
    }

    impl BriaTransport for TestTransport {
        fn retry_after_delay(
            &mut self,
            header: &str,
            fallback: u64,
        ) -> Result<u64, ConformanceError> {
            Ok(header
                .parse::<u64>()
                .ok()
                .map(|seconds| seconds.min(150) * 1000)
                .unwrap_or(fallback))
        }
        fn retry_request(
            &mut self,
            _handle: StreamHandle,
            delay: u64,
        ) -> Result<StreamHandle, ConformanceError> {
            self.delays.push(delay);
            let head = self.head.clone().ok_or(ConformanceError::InvalidRequest)?;
            self.actuator.start_request(&head)
        }
        fn delay(
            &mut self,
            _handle: StreamHandle,
            milliseconds: u64,
        ) -> Result<(), ConformanceError> {
            self.delays.push(milliseconds);
            Ok(())
        }
        fn upload_prepared(
            &mut self,
            _handle: StreamHandle,
            media: &PreparedMedia,
            label: &str,
        ) -> Result<UploadedMedia, ConformanceError> {
            let expected = self
                .uploads
                .pop_front()
                .ok_or(ConformanceError::InvalidRequest)?;
            if expected != *media || label != media.role.wait_label() {
                return Err(ConformanceError::InvalidRequest);
            }
            self.observed_uploads.push(media.role);
            Ok(UploadedMedia {
                role: media.role,
                url: format!("https://storage.fixture.invalid/{}", media.role.name()),
            })
        }
    }
    fn input(node: &str) -> Result<Input, Box<dyn Error>> {
        let oracle: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bria/source-contract.json"
        )))?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing oracle cases")?
            .iter()
            .find(|case| case["input"]["node"] == node)
            .ok_or("missing source node")?;
        Ok(serde_json::from_value(case["input"].clone())?)
    }
    fn handle(slot: u32, generation: u32) -> StreamHandle {
        StreamHandle {
            invocation: 42,
            slot,
            generation,
        }
    }
    fn media(input: &Input) -> Result<Vec<PreparedMedia>, BriaError> {
        if upload_roles(input)?.iter().any(|role| role.is_video()) {
            return Err(BriaError::Input(
                "Tests do not fabricate encoded video media",
            ));
        }
        Ok(upload_roles(input)?
            .into_iter()
            .map(|role| PreparedMedia {
                role,
                file_name: format!("{}.png", role.name()),
                content_type: if role.is_video() {
                    Some("video/mp4".into())
                } else {
                    None
                },
                bytes: include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
                ))
                .to_vec(),
            })
            .collect())
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn exchange_steps(
        head: RequestHead,
        body: Vec<u8>,
        response: Vec<u8>,
        handle: StreamHandle,
    ) -> Vec<ScriptStep> {
        let mut steps = vec![
            ScriptStep::CheckCancelled(None),
            ScriptStep::Start(head, Ok(handle)),
            ScriptStep::CheckCancelled(Some(handle)),
            ScriptStep::RequestChunk {
                handle,
                sequence: 0,
                bytes: body,
                end: true,
            },
        ];
        for (after_sequence, outcome) in [
            (
                None,
                ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                },
            ),
            (Some(0), ResponseEvent::Chunk(response)),
            (
                Some(1),
                ResponseEvent::Completed(b"download-receipt".to_vec()),
            ),
        ] {
            steps.push(ScriptStep::CheckCancelled(Some(handle)));
            steps.push(ScriptStep::Wait {
                handle,
                after_sequence,
                timeout_milliseconds: 1000,
                outcome,
            });
        }
        steps.push(ScriptStep::CheckCancelled(Some(handle)));
        steps
    }
    fn success_steps(input: &Input, media: &[PreparedMedia]) -> Result<Vec<ScriptStep>, BriaError> {
        let uploads = upload_roles(input)?
            .into_iter()
            .map(|role| UploadedMedia {
                role,
                url: format!("https://storage.fixture.invalid/{}", role.name()),
            })
            .collect::<Vec<_>>();
        let (path, body) = request(input, &uploads)?;
        let mut steps = exchange_steps(
            request_head(path, HttpMethod::Post, credentials(), false)?,
            body,
            serde_json::to_vec(&json!({
                "request_id": "task-42",
                "status_url": "https://unused-status.fixture.invalid/task",
                "warning": "admitted source warning"
            }))?,
            handle(0, 1),
        );
        steps.insert(
            2,
            ScriptStep::Cost(
                handle(0, 1),
                CostRequest {
                    operation: "bria.generation.maximum".into(),
                    currency: "USD".into(),
                    maximum_microunits: 1_000_000,
                },
                CostResponse {
                    accepted: true,
                    approved_microunits: 1_000_000,
                    receipt: b"cost-receipt".to_vec(),
                },
            ),
        );
        for _ in media {
            steps.insert(3, ScriptStep::CheckCancelled(Some(handle(0, 1))));
        }
        let response = if input.node == "BriaImageEditNode" {
            br#"{"status":" Completed ","result":{"image_url":"https://download.fixture.invalid/image.png","structured_prompt":"source structure"}}"#.to_vec()
        } else {
            br#"{"status":"completed","result":{"image_url":"https://download.fixture.invalid/image.png"}}"#.to_vec()
        };
        steps.extend(exchange_steps(
            request_head(
                "/proxy/bria/v2/status/task-42",
                HttpMethod::Get,
                credentials(),
                false,
            )?,
            vec![],
            response,
            handle(1, 1),
        ));
        steps.extend(exchange_steps(
            request_head(
                "https://download.fixture.invalid/image.png",
                HttpMethod::Get,
                credentials(),
                true,
            )?,
            vec![],
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
            ))
            .to_vec(),
            handle(2, 1),
        ));
        Ok(steps)
    }
    #[test]
    fn bria_claims_requests_defaults_and_upload_order_match_pinned_source()
    -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bria/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 6);
        assert_eq!(ROUTES.len(), 7);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            6
        );
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "synthetic_prefix_tombstone")
                .count(),
            1
        );
        assert!(ROUTES.iter().any(|route| route.path == IMAGE_EDIT_PATH
            && route.feature_id == "COMFY-API-EXT-0218"
            && route.disposition == "executable"));
        Ok(())
    }
    #[test]
    fn bria_preflight_prompt_background_duration_and_cost_fail_before_actuation()
    -> Result<(), Box<dyn Error>> {
        let mut candidate = input("BriaImageEditNode")?;
        candidate.prompt.clear();
        candidate.structured_prompt.clear();
        assert!(upload_roles(&candidate).is_err());
        let mut candidate = input("BriaVideoReplaceBackground")?;
        candidate.background_image = false;
        candidate.background_video = false;
        assert!(upload_roles(&candidate).is_err());
        candidate.background_image = true;
        candidate.background_video = true;
        assert!(upload_roles(&candidate).is_err());
        for seconds in [f64::NAN, f64::INFINITY, 0.0, 60.001] {
            assert!(duration(seconds).is_err());
        }
        assert!(duration(60.0001).is_ok());
        let candidate = input("BriaRemoveImageBackground")?;
        let prepared = media(&candidate)?;
        let mut transport = TestTransport::new(vec![], vec![]);
        assert!(matches!(
            invoke(&candidate, &prepared, credentials(), 0, &mut transport),
            Err(BriaError::Transport(ConformanceError::InvalidCost))
        ));
        transport.finish()?;
        assert_eq!(price_badge_microunits(&candidate)?, 18000);
        assert_eq!(
            price_badge_microunits(&input("BriaVideoGreenScreen")?)?,
            10500
        );
        Ok(())
    }
    #[test]
    fn bria_poll_statuses_are_source_normalized_and_result_schemas_are_strict()
    -> Result<(), Box<dyn Error>> {
        let input = input("BriaRemoveImageBackground")?;
        for status in [
            "succeeded",
            "succeed",
            "success",
            "completed",
            "finished",
            "done",
            "complete",
        ] {
            let bytes = serde_json::to_vec(
                &json!({"status":format!(" {status} "),"result":{"image_url":"https://fixture.invalid/image.png"}}),
            )?;
            assert_eq!(task_status(&input, &bytes)?.1, Phase::Completed);
        }
        for status in [
            "created",
            "queued",
            "queueing",
            "submitted",
            "initializing",
            "wait",
            "in_queue",
        ] {
            let bytes = serde_json::to_vec(&json!({"status":status}))?;
            assert_eq!(task_status(&input, &bytes)?.1, Phase::Queued);
        }
        for status in [
            "cancelled",
            "canceled",
            "canceling",
            "fail",
            "failed",
            "error",
        ] {
            let bytes = serde_json::to_vec(&json!({"status":status}))?;
            assert!(matches!(
                task_status(&input, &bytes),
                Err(BriaError::Provider(_))
            ));
        }
        for bytes in [
            r#"{"status":"completed"}"#,
            r#"{"status":"completed","result":{"image_url":42}}"#,
            r#"{"status":"completed","result":{"image_url":"/proxy/result"}}"#,
            r#"{"status":"queued"}trailing"#,
        ] {
            assert!(task_status(&input, bytes.as_bytes()).is_err());
        }
        assert!(task_status(&input, &vec![b' '; MAXIMUM_RESPONSE_BYTES + 1]).is_err());
        assert!(submitted_task(br#"{"request_id":"../escape","status_url":"ignored"}"#).is_err());
        Ok(())
    }
    #[test]
    fn bria_real_image_download_preserves_receipt_warning_and_actual_poll_path()
    -> Result<(), Box<dyn Error>> {
        let input = input("BriaRemoveImageBackground")?;
        let prepared = media(&input)?;
        let mut transport = TestTransport::new(success_steps(&input, &prepared)?, prepared.clone());
        let result = invoke(&input, &prepared, credentials(), 1_000_000, &mut transport)?;
        assert!(result.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(result.response_receipt, b"download-receipt");
        assert_eq!(result.warning.as_deref(), Some("admitted source warning"));
        assert_eq!(
            result.source_status_url,
            "https://unused-status.fixture.invalid/task"
        );
        assert_eq!(transport.observed_uploads, vec![UploadRole::Image]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bria_edit_uploads_mask_before_image_and_retains_structured_prompt()
    -> Result<(), Box<dyn Error>> {
        let mut input = input("BriaImageEditNode")?;
        input.mask = true;
        let prepared = media(&input)?;
        let mut transport = TestTransport::new(success_steps(&input, &prepared)?, prepared.clone());
        let result = invoke(&input, &prepared, credentials(), 1_000_000, &mut transport)?;
        assert_eq!(
            result.structured_prompt.as_deref(),
            Some("source structure")
        );
        assert_eq!(
            transport.observed_uploads,
            vec![UploadRole::Mask, UploadRole::Image]
        );
        assert!(prepared.iter().all(|media| media.content_type.is_none()));
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bria_decoder_failure_and_cancelled_worker_publish_no_native_outputs()
    -> Result<(), Box<dyn Error>> {
        struct Decoder;
        impl BriaMediaMaterializer for Decoder {
            fn decode_media(
                &mut self,
                media: &DownloadedMedia,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                assert!(media.bytes.starts_with(b"\x89PNG"));
                Err(ConformanceError::InvalidResponse)
            }
        }
        #[derive(Default)]
        struct Sink {
            batches: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(
                &mut self,
                _batch: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.batches += 1;
                Ok(())
            }
        }
        let input = input("BriaRemoveImageBackground")?;
        let prepared = media(&input)?;
        let mut steps = success_steps(&input, &prepared)?;
        steps.push(ScriptStep::CheckCancelled(None));
        let mut transport = TestTransport::new(steps, prepared.clone());
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input,
                &prepared,
                credentials(),
                1_000_000,
                &mut transport,
                &mut Decoder,
                &mut sink
            ),
            Err(BriaError::Transport(ConformanceError::InvalidResponse))
        ));
        assert_eq!(sink.batches, 0);
        transport.finish()?;
        let mut transport = TestTransport::new(vec![ScriptStep::Cancel(None)], vec![]);
        assert!(matches!(
            invoke(&input, &prepared, credentials(), 1_000_000, &mut transport),
            Err(BriaError::Transport(ConformanceError::Cancelled))
        ));
        transport.finish()?;
        let (path, _) = request(
            &input,
            &[UploadedMedia {
                role: UploadRole::Image,
                url: "https://storage.fixture.invalid/image".into(),
            }],
        )?;
        let mut transport = TestTransport::new(
            vec![
                ScriptStep::CheckCancelled(None),
                ScriptStep::Start(
                    request_head(path, HttpMethod::Post, credentials(), false)?,
                    Err(ConformanceError::WorkerLost),
                ),
            ],
            vec![],
        );
        assert!(matches!(
            invoke(&input, &prepared, credentials(), 1_000_000, &mut transport),
            Err(BriaError::Transport(ConformanceError::WorkerLost))
        ));
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bria_retry_requires_fresh_handle_and_replays_exact_body() -> Result<(), Box<dyn Error>> {
        let head = request_head(IMAGE_EDIT_PATH, HttpMethod::Post, credentials(), false)?;
        let body = r#"{"instruction":"preserve exact ☁ bytes"}"#.as_bytes().to_vec();
        let mut steps = exchange_steps(head.clone(), body.clone(), b"busy".to_vec(), handle(0, 1));
        if let Some(ScriptStep::Wait {
            outcome: ResponseEvent::Head { status, .. },
            ..
        }) = steps.get_mut(5)
        {
            *status = 503;
        } else {
            return Err("missing scripted response head".into());
        }
        let mut retry = exchange_steps(
            head.clone(),
            body.clone(),
            b"success".to_vec(),
            handle(0, 2),
        );
        retry.remove(0);
        steps.extend(retry);
        let mut transport = TestTransport::new(steps, vec![]);
        assert_eq!(
            exchange(&head, &body, None, MAXIMUM_RESPONSE_BYTES, &mut transport)?.0,
            b"success"
        );
        assert_eq!(transport.delays, vec![1000]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bria_video_source_contract_is_mp4_h264_but_transparent_outputs_are_image_and_mask()
    -> Result<(), Box<dyn Error>> {
        let input = input("BriaTransparentVideoBackground")?;
        let uploads = vec![UploadedMedia {
            role: UploadRole::Video,
            url: "https://storage.fixture.invalid/video".into(),
        }];
        let (path, bytes) = request(&input, &uploads)?;
        let body: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(path, "/proxy/bria/v2/video/edit/remove_background");
        assert_eq!(body["output_container_and_codec"], "webm_vp9");
        assert_eq!(body["background_color"], "Transparent");
        assert_eq!(body["preserve_audio"], true);
        assert_eq!(upload_roles(&input)?, vec![UploadRole::Video]);
        Ok(())
    }
    #[test]
    fn bria_credentials_and_denied_cost_prevent_uploads() -> Result<(), Box<dyn Error>> {
        let input = input("BriaRemoveImageBackground")?;
        let prepared = media(&input)?;
        let absent = CredentialAvailability {
            auth_token: false,
            api_key: false,
        };
        let mut transport = TestTransport::new(vec![], vec![]);
        assert!(matches!(
            invoke(&input, &prepared, absent, 1_000_000, &mut transport),
            Err(BriaError::Input(_))
        ));
        transport.finish()?;
        let head = request_head(
            "/proxy/bria/v2/image/edit/remove_background",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.auth_token"));
        let key = request_head(
            IMAGE_EDIT_PATH,
            HttpMethod::Post,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(key.secret_id.as_deref(), Some("sim_org.api_key"));
        let download = request_head(
            "https://download.fixture.invalid/image.png",
            HttpMethod::Get,
            credentials(),
            true,
        )?;
        assert!(download.secret_id.is_none());
        assert!(
            download
                .headers
                .iter()
                .all(|header| header.name != "Authorization" && header.name != "X-API-KEY")
        );
        let mut transport = TestTransport::new(
            vec![
                ScriptStep::CheckCancelled(None),
                ScriptStep::Start(head, Ok(handle(0, 1))),
                ScriptStep::Cost(
                    handle(0, 1),
                    CostRequest {
                        operation: "bria.generation.maximum".into(),
                        currency: "USD".into(),
                        maximum_microunits: 1_000_000,
                    },
                    CostResponse {
                        accepted: false,
                        approved_microunits: 0,
                        receipt: b"denied".to_vec(),
                    },
                ),
            ],
            vec![],
        );
        assert!(matches!(
            invoke(&input, &prepared, credentials(), 1_000_000, &mut transport),
            Err(BriaError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(transport.observed_uploads.is_empty());
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bria_queued_polls_preserve_source_budget_and_delay() -> Result<(), Box<dyn Error>> {
        let input = input("BriaRemoveImageBackground")?;
        let prepared = media(&input)?;
        let mut steps = success_steps(&input, &prepared)?;
        let poll_start = steps.iter().position(|step| matches!(step, ScriptStep::Start(head, _) if head.path == "/proxy/bria/v2/status/task-42")).ok_or("missing poll start")?;
        let terminal = steps.split_off(
            poll_start
                .checked_sub(1)
                .ok_or("missing poll cancellation check")?,
        );
        let queued_count = MAXIMUM_POLLS + 1;
        for number in 0..queued_count {
            let poll_handle = handle(3 + number, 1);
            steps.extend(exchange_steps(
                request_head(
                    "/proxy/bria/v2/status/task-42",
                    HttpMethod::Get,
                    credentials(),
                    false,
                )?,
                vec![],
                br#"{"status":" queued "}"#.to_vec(),
                poll_handle,
            ));
            steps.push(ScriptStep::CheckCancelled(Some(poll_handle)));
        }
        steps.extend(terminal);
        let mut transport = TestTransport::new(steps, prepared.clone());
        let result = invoke(&input, &prepared, credentials(), 1_000_000, &mut transport)?;
        assert!(result.bytes.starts_with(b"\x89PNG"));
        assert_eq!(transport.delays, vec![5000; queued_count as usize]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bria_oversized_transport_frame_cannot_reach_materialization() -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "https://download.fixture.invalid/image.png",
            HttpMethod::Get,
            credentials(),
            true,
        )?;
        let mut steps = exchange_steps(
            head.clone(),
            vec![],
            vec![0; MAXIMUM_CHUNK_BYTES + 1],
            handle(0, 1),
        );
        steps.truncate(8);
        let mut transport = TestTransport::new(steps, vec![]);
        assert!(matches!(
            exchange(&head, &[], None, MAXIMUM_MEDIA_BYTES, &mut transport),
            Err(BriaError::Transport(ConformanceError::ChunkLimit))
        ));
        transport.finish()?;
        Ok(())
    }

    #[test]
    fn bria_python_status_strip_covers_all_nodes_and_preserves_raw_status()
    -> Result<(), Box<dyn Error>> {
        for claim in NODES {
            let input = input(claim.node_identifier)?;
            let image = matches!(
                claim.node_identifier,
                "BriaImageEditNode" | "BriaRemoveImageBackground"
            );
            let result = if image {
                json!({"image_url":"https://fixture.invalid/image.png","structured_prompt":"source structure"})
            } else {
                json!({"video_url":"https://fixture.invalid/video.mp4"})
            };
            for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                for (status, phase) in [
                    ("Completed", Phase::Completed),
                    ("SUCCESS", Phase::Completed),
                    (" Queued ", Phase::Queued),
                    (" IN_QUEUE ", Phase::Queued),
                ] {
                    let raw = format!("{separator}{status}{separator}");
                    let bytes = serde_json::to_vec(&json!({"status":raw,"result":result}))?;
                    let (task, observed) = task_status(&input, &bytes)?;
                    assert_eq!(observed, phase);
                    assert_eq!(task.status, raw);
                }
                for status in ["FAILED", "cancelled", " Error "] {
                    let raw = format!("{separator}{status}{separator}");
                    let bytes = serde_json::to_vec(&json!({"status":raw}))?;
                    match task_status(&input, &bytes) {
                        Err(BriaError::Provider(observed)) => assert_eq!(observed, raw),
                        _ => return Err("Source failure status not preserved".into()),
                    }
                }
            }
            for raw in [
                "\u{200b}completed\u{200b}",
                "\u{feff}completed\u{feff}",
                "com\u{1c}pleted",
                "\u{1c}\u{1f}",
            ] {
                let bytes = serde_json::to_vec(&json!({"status":raw,"result":result}))?;
                let (task, phase) = task_status(&input, &bytes)?;
                assert_eq!(phase, Phase::Processing);
                assert_eq!(task.status, raw);
            }
            let invalid = serde_json::to_vec(&json!({"status":"\u{1c}failed\u{1f}","result":{}}))?;
            assert!(matches!(
                task_status(&input, &invalid),
                Err(BriaError::Response("Missing Bria media URL"))
            ));
        }
        Ok(())
    }
    #[test]
    fn bria_python_status_strip_drives_real_invoke_completion_and_failure()
    -> Result<(), Box<dyn Error>> {
        for node in ["BriaImageEditNode", "BriaRemoveImageBackground"] {
            let input = input(node)?;
            let prepared = media(&input)?;
            for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                for failed in [false, true] {
                    let raw = format!(
                        "{separator}{}{separator}",
                        if failed { " FAILED " } else { " Completed " }
                    );
                    let mut steps = success_steps(&input, &prepared)?;
                    let mut changed = false;
                    for step in &mut steps {
                        if let ScriptStep::Wait {
                            handle: poll_handle,
                            outcome: ResponseEvent::Chunk(bytes),
                            ..
                        } = step
                            && *poll_handle == handle(1, 1)
                        {
                            let mut response: Value = serde_json::from_slice(bytes)?;
                            response["status"] = json!(raw);
                            if failed {
                                response["result"] = Value::Null;
                            }
                            *bytes = serde_json::to_vec(&response)?;
                            changed = true;
                        }
                    }
                    assert!(changed);
                    if failed {
                        let download_start = steps.iter().position(|step| matches!(step, ScriptStep::Start(head, _) if head.path == "https://download.fixture.invalid/image.png")).ok_or("download step missing")?;
                        steps.truncate(
                            download_start
                                .checked_sub(1)
                                .ok_or("download cancellation check missing")?,
                        );
                    }
                    let mut transport = TestTransport::new(steps, prepared.clone());
                    if failed {
                        match invoke(&input, &prepared, credentials(), 1_000_000, &mut transport) {
                            Err(BriaError::Provider(observed)) => assert_eq!(observed, raw),
                            _ => {
                                return Err(
                                    "Failed task proceeded past source terminal status".into()
                                );
                            }
                        }
                    } else {
                        let result =
                            invoke(&input, &prepared, credentials(), 1_000_000, &mut transport)?;
                        assert_eq!(
                            result.bytes,
                            prepared.first().ok_or("prepared image missing")?.bytes
                        );
                        assert_eq!(result.response_receipt, b"download-receipt");
                        assert_eq!(
                            result.source_url,
                            "https://download.fixture.invalid/image.png"
                        );
                        assert_eq!(result.warning.as_deref(), Some("admitted source warning"));
                    }
                    assert!(transport.delays.is_empty());
                    assert_eq!(transport.observed_uploads, vec![UploadRole::Image]);
                    transport.finish()?;
                }
            }
        }
        Ok(())
    }
    #[test]
    fn bria_python_queued_status_preserves_budget_delay_and_prestart_cancel()
    -> Result<(), Box<dyn Error>> {
        let input = input("BriaRemoveImageBackground")?;
        let prepared = media(&input)?;
        let mut steps = success_steps(&input, &prepared)?;
        let poll_start = steps.iter().position(|step| matches!(step, ScriptStep::Start(head, _) if head.path == "/proxy/bria/v2/status/task-42")).ok_or("poll step missing")?;
        let terminal = steps.split_off(
            poll_start
                .checked_sub(1)
                .ok_or("poll cancellation missing")?,
        );
        let queued_count = MAXIMUM_POLLS + 1;
        for number in 0..queued_count {
            let poll_handle = handle(3 + number, 1);
            steps.extend(exchange_steps(
                request_head(
                    "/proxy/bria/v2/status/task-42",
                    HttpMethod::Get,
                    credentials(),
                    false,
                )?,
                vec![],
                serde_json::to_vec(&json!({"status":"\u{1c} QuEuEd \u{1f}"}))?,
                poll_handle,
            ));
            steps.push(ScriptStep::CheckCancelled(Some(poll_handle)));
        }
        steps.extend(terminal);
        let mut transport = TestTransport::new(steps, prepared.clone());
        let result = invoke(&input, &prepared, credentials(), 1_000_000, &mut transport)?;
        assert_eq!(
            result.bytes,
            prepared.first().ok_or("prepared image missing")?.bytes
        );
        assert_eq!(transport.delays, vec![5000; queued_count as usize]);
        assert_eq!(transport.observed_uploads, vec![UploadRole::Image]);
        transport.finish()?;
        let mut cancelled = TestTransport::new(vec![ScriptStep::Cancel(None)], vec![]);
        assert!(matches!(
            invoke(&input, &prepared, credentials(), 1_000_000, &mut cancelled),
            Err(BriaError::Transport(ConformanceError::Cancelled))
        ));
        assert!(cancelled.observed_uploads.is_empty());
        cancelled.finish()?;
        Ok(())
    }
}
