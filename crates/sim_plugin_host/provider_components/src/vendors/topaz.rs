use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TransportLimits, TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.topaz";
pub const SOURCE_SHA256: &str = "fb080452ffabf0fd519dfc4ca0da8393f0e0a548c1dfdbb62bd5f5fc542fc4b6";
pub const ALIASES: &[&str] = &[];
pub const MAXIMUM_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 64 * 1024 * 1024;
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
        feature_id: "COMFY-NODE-0677",
        node_identifier: "TopazImageEnhance",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0678",
        node_identifier: "TopazVideoEnhance",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0679",
        node_identifier: "TopazVideoEnhanceV2",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0174",
        method: "GET",
        path: "/proxy/topaz/image/v1/download/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0175",
        method: "GET",
        path: "/proxy/topaz/image/v1/download/{initial_response.process_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0176",
        method: "POST",
        path: "/proxy/topaz/image/v1/enhance-gen/async",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0177",
        method: "GET",
        path: "/proxy/topaz/image/v1/status/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0178",
        method: "GET",
        path: "/proxy/topaz/image/v1/status/{initial_response.process_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0179",
        method: "POST",
        path: "/proxy/topaz/video/",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0180",
        method: "PATCH",
        path: "/proxy/topaz/video/{initial_res.requestId}/accept",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0181",
        method: "PATCH",
        path: "/proxy/topaz/video/{initial_res.requestId}/complete-upload",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0182",
        method: "GET",
        path: "/proxy/topaz/video/{initial_res.requestId}/status",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum TopazError {
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
impl fmt::Display for TopazError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "Topaz job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Topaz HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for TopazError {}
impl From<ConformanceError> for TopazError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for TopazError {
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
pub struct VideoInput {
    pub container: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
    pub frames: u64,
    #[serde(default)]
    pub failure: Option<String>,
    #[serde(default)]
    pub stream_kind: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    pub image: Option<ImageInput>,
    pub video: Option<VideoInput>,
}
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub download_url: String,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct PreparedVideo {
    pub container: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub duration: f64,
    pub frame_count: u64,
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestModel {
    ImageEnhance,
    VideoCreate,
}
pub trait TopazInputOwner {
    fn validate_request_model(
        &mut self,
        _: RequestModel,
        _: &Value,
    ) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn verify_inputs(&mut self, _: &Input) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn image_upload(
        &mut self,
        _: &Input,
        _: u64,
        _: &str,
    ) -> Result<PreparedImage, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn original_video_source(&mut self, _: &Input) -> Result<PreparedVideo, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn image_model_fields(&mut self, _: &Value) -> Result<Vec<MultipartField>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn string<'a>(settings: &'a Value, key: &str) -> Result<&'a str, TopazError> {
    settings
        .get(key)
        .and_then(Value::as_str)
        .ok_or(TopazError::Input("Missing source string"))
}
fn boolean(settings: &Value, key: &str, default: bool) -> Result<bool, TopazError> {
    match settings.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or(TopazError::Input("Invalid source boolean")),
    }
}
fn source_integer(number: f64) -> Result<i64, TopazError> {
    if !number.is_finite() || number < i64::MIN as f64 || number >= i64::MAX as f64 {
        return Err(TopazError::Input("Unrepresentable source integer"));
    }
    Ok(number.trunc() as i64)
}
fn model_id(model: &str) -> Result<&'static str, TopazError> {
    match model {
        "Astra 2" => Ok("ast-2"),
        "Starlight (Astra) Fast" => Ok("slf-1"),
        "Starlight (Astra) Creative" => Ok("slc-1"),
        "Starlight Precise 2.5" => Ok("slp-2.5"),
        _ => Err(TopazError::Input("Unknown source model lookup")),
    }
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn source_target_resolution(
    source: &PreparedVideo,
    resolution: &str,
) -> Result<(i64, i64), TopazError> {
    if source.width == 0 || source.height == 0 {
        return Err(TopazError::Input("Invalid native video dimensions"));
    }
    let (target, maximum) = if resolution.contains("1080p") {
        (1080i64, 1920i64)
    } else {
        (2160i64, 3840i64)
    };
    let ratio = source.width as f64 / source.height as f64;
    let (mut width, mut height) = if source.width >= source.height {
        let width = source_integer(target as f64 * ratio)?;
        if width > maximum {
            (maximum, source_integer(maximum as f64 / ratio)?)
        } else {
            (width, target)
        }
    } else {
        let height = source_integer(target as f64 / ratio)?;
        if height > maximum {
            (source_integer(maximum as f64 * ratio)?, maximum)
        } else {
            (target, height)
        }
    };
    if width % 2 != 0 {
        width = width
            .checked_add(1)
            .ok_or(TopazError::Input("Source dimension overflow"))?;
    }
    if height % 2 != 0 {
        height = height
            .checked_add(1)
            .ok_or(TopazError::Input("Source dimension overflow"))?;
    }
    Ok((width, height))
}
pub fn video_request(input: &Input, source: &PreparedVideo) -> Result<Value, TopazError> {
    let modern = input.node == "TopazVideoEnhanceV2";
    let upscaler = if modern {
        input
            .settings
            .get("upscaler_model")
            .ok_or(TopazError::Input("Missing source dynamic upscaler"))?
    } else {
        &input.settings
    };
    let interpolation = if modern {
        input
            .settings
            .get("interpolation_model")
            .ok_or(TopazError::Input("Missing source dynamic interpolation"))?
    } else {
        &input.settings
    };
    let upscaler_choice = string(upscaler, "upscaler_model")?;
    let upscaler_enabled = if modern {
        upscaler_choice != "Disabled"
    } else {
        upscaler
            .get("upscaler_enabled")
            .and_then(Value::as_bool)
            .ok_or(TopazError::Input("Missing required source upscaler flag"))?
    };
    let interpolation_enabled = if modern {
        string(interpolation, "interpolation_model")? != "Disabled"
    } else {
        boolean(interpolation, "interpolation_enabled", false)?
    };
    if !upscaler_enabled && !interpolation_enabled {
        return Err(TopazError::Input("Both source filters disabled"));
    }
    if !["mp4", "mov,mp4,m4a,3gp,3g2,mj2"].contains(&source.container.as_str()) {
        return Err(TopazError::Input("Source requires MP4 container"));
    }
    let mut resolution = (source.width as i64, source.height as i64);
    let mut frame_rate = source_integer(source.frame_rate)?;
    let mut filters = Vec::new();
    if upscaler_enabled {
        resolution = source_target_resolution(source, string(upscaler, "upscaler_resolution")?)?;
        let model = model_id(upscaler_choice)?;
        let mut filter = json!({"model":model});
        if model == "slc-1" {
            filter["creativity"] = if modern {
                upscaler
                    .get("creativity")
                    .cloned()
                    .ok_or(TopazError::Input("Missing source creativity"))?
            } else {
                upscaler
                    .get("upscaler_creativity")
                    .cloned()
                    .unwrap_or(json!("low"))
            };
            filter["isOptimizedMode"] = json!(true);
        } else if modern && model == "ast-2" {
            let prompt = upscaler
                .get("prompt")
                .filter(|value| !value.is_null())
                .map(|value| {
                    value
                        .as_str()
                        .ok_or(TopazError::Input("Invalid source prompt"))
                })
                .transpose()?
                .unwrap_or("");
            let prompt = python_strip(prompt);
            if (!prompt.is_empty() && source.frame_count > 450) || source.frame_count > 9000 {
                return Err(TopazError::Input("Source Astra frame count limit"));
            }
            filter["creativity"] = upscaler
                .get("creativity")
                .cloned()
                .ok_or(TopazError::Input("Missing source creativity"))?;
            filter["sharp"] = upscaler
                .get("sharp")
                .cloned()
                .ok_or(TopazError::Input("Missing source sharpness"))?;
            if !prompt.is_empty() {
                filter["prompt"] = json!(prompt);
            }
            let realism = upscaler
                .get("realism")
                .and_then(Value::as_f64)
                .ok_or(TopazError::Input("Invalid source realism"))?;
            if realism > 0.0 {
                filter["realism"] = json!(realism);
            }
        }
        filters.push(filter);
    }
    if interpolation_enabled {
        frame_rate = if modern {
            interpolation
                .get("interpolation_frame_rate")
                .and_then(Value::as_i64)
                .ok_or(TopazError::Input("Missing source dynamic FPS"))?
        } else {
            optional_integer(interpolation, "interpolation_frame_rate", 60)?
        };
        let field = |name: &str, default: Value| -> Result<Value, TopazError> {
            match interpolation.get(name) {
                Some(value) => Ok(value.clone()),
                None if !modern => Ok(default),
                _ => Err(TopazError::Input(
                    "Missing source dynamic interpolation field",
                )),
            }
        };
        filters.push(json!({"model":field("interpolation_model",json!("apo-8"))?,"slowmo":field("interpolation_slowmo",json!(1))?,"fps":frame_rate,"duplicate":field("interpolation_duplicate",json!(false))?,"duplicate_threshold":field("interpolation_duplicate_threshold",json!(0.01))?}));
    }
    Ok(
        json!({"source":{"container":"mp4","size":source.bytes.len(),"duration":source_integer(source.duration)?,"frameCount":source.frame_count,"frameRate":source_integer(source.frame_rate)?,"resolution":{"width":source.width,"height":source.height}},"filters":filters,"output":{"resolution":{"width":resolution.0,"height":resolution.1},"frameRate":frame_rate,"audioCodec":"AAC","audioTransfer":"Copy","dynamicCompressionLevel":input.settings.get("dynamic_compression_level").cloned().unwrap_or(json!("Low"))},"overrides":{"isPaidDiffusion":true}}),
    )
}
fn optional_integer(settings: &Value, key: &str, default: i64) -> Result<i64, TopazError> {
    match settings.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_i64()
            .ok_or(TopazError::Input("Invalid source integer field")),
    }
}
fn optional_float(settings: &Value, key: &str, default: f64) -> Result<f64, TopazError> {
    match settings.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or(TopazError::Input("Invalid source float field")),
    }
}
pub fn image_request(input: &Input, image: &PreparedImage) -> Result<Value, TopazError> {
    if input.image.as_ref().map(|image| image.batch) != Some(1) {
        return Err(TopazError::Input("Source requires exactly one image"));
    }
    let settings = &input.settings;
    let mut body = json!({
        "model": settings.get("model").cloned().unwrap_or(json!("Reimagine")),
        "output_format": "png",
        "subject_detection": settings.get("subject_detection").cloned().unwrap_or(json!("All")),
        "face_enhancement": boolean(settings,"face_enhancement",true)?,
        "face_enhancement_creativity": optional_float(settings,"face_enhancement_creativity",1.0)?,
        "face_enhancement_strength": optional_float(settings,"face_enhancement_strength",0.8)?,
        "source_url": image.download_url,
        "crop_to_fill": boolean(settings,"crop_to_fill",false)?,
        "creativity": optional_integer(settings,"creativity",3)?,
        "face_preservation": if boolean(settings,"face_preservation",true)? {"true"}else{"false"},
        "color_preservation": if boolean(settings,"color_preservation",true)? {"true"}else{"false"}
    });
    let prompt = settings.get("prompt").cloned().unwrap_or(json!(""));
    if !prompt.is_null() {
        if !prompt.is_string() {
            return Err(TopazError::Input("Invalid optional source prompt"));
        }
        body["prompt"] = prompt;
    }
    for key in ["output_width", "output_height"] {
        let value = optional_integer(settings, key, 0)?;
        if value != 0 {
            body[key] = json!(value);
        }
    }
    Ok(body)
}
pub const IMAGE_MODEL_FIELD_ORDER: &[&str] = &[
    "model",
    "output_format",
    "subject_detection",
    "face_enhancement",
    "face_enhancement_creativity",
    "face_enhancement_strength",
    "source_url",
    "output_width",
    "output_height",
    "crop_to_fill",
    "prompt",
    "creativity",
    "face_preservation",
    "color_preservation",
];
#[derive(Default)]
struct Session {
    invocation: Option<u64>,
    handles: BTreeSet<(u64, u32, u32)>,
}
impl Session {
    fn retain(&mut self, handle: StreamHandle) -> Result<(), TopazError> {
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
fn endpoint(value: &str) -> Result<bool, TopazError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(TopazError::Response("Invalid Topaz response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(TopazError::Response("Invalid Topaz response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(TopazError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(TopazError::Response("Invalid Topaz endpoint authority"));
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
) -> Result<RequestHead, TopazError> {
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
            return Err(TopazError::Input("Missing Comfy API credentials"));
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
    transport: &mut dyn TopazTransport,
) -> Result<RetainedResponse, TopazError> {
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
    transport: &mut dyn TopazTransport,
) -> Result<RetainedResponse, TopazError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(TopazError::Http {
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
                    return Err(TopazError::Http {
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
    transport: &mut dyn TopazTransport,
) -> Result<RetainedResponse, TopazError> {
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
                    return Err(TopazError::Http {
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

const SIGNED_PUT_POLICY: RetryPolicy = RetryPolicy {
    ordinary_limit: 0,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
    timeout_milliseconds: None,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiModel {
    ImageCreation,
    ImageStatus,
    ImageDownload,
    VideoCreation,
    VideoAccept,
    VideoComplete,
    VideoStatus,
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
pub trait TopazTransport: ProviderTransport {
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
#[derive(Clone, Debug)]
pub struct InvocationResult {
    pub node: String,
    pub input_receipt: Vec<u8>,
    pub responses: Vec<RetainedResponse>,
    pub downloaded: RetainedResponse,
    pub cost_receipt: Vec<u8>,
    pub url: String,
}
fn normalized(
    response: &RetainedResponse,
    model: ApiModel,
    transport: &mut dyn TopazTransport,
) -> Result<Value, TopazError> {
    transport.check_cancelled(Some(response.handle))?;
    let raw: Value = serde_json::from_slice(&response.bytes)?;
    if !raw.is_object() {
        return Err(TopazError::Response("Source response requires API object"));
    }
    let value = transport.validate_response(response, model, &raw)?;
    transport.check_cancelled(Some(response.handle))?;
    if !value.is_object() {
        return Err(TopazError::Response(
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
) -> Result<(), TopazError> {
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
fn retained_id(value: &Value, key: &str) -> Result<String, TopazError> {
    let id = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(TopazError::Response("Missing source retained task ID"))?;
    if id.is_empty()
        || id.len() > 1024
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(TopazError::Response("Unsafe retained task ID"));
    }
    Ok(id.into())
}
fn bounded_receipt(receipt: &[u8]) -> Result<(), TopazError> {
    if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
        return Err(ConformanceError::InvalidResponse.into());
    }
    Ok(())
}
fn checked_multipart(
    body: &Value,
    fields: Vec<MultipartField>,
    boundary: &str,
) -> Result<Vec<u8>, TopazError> {
    let names: Vec<&str> = IMAGE_MODEL_FIELD_ORDER
        .iter()
        .copied()
        .filter(|name| body.get(*name).is_some_and(|value| !value.is_null()))
        .collect();
    if fields.len() != names.len()
        || fields.iter().zip(names).any(|(field, name)| {
            field.name != name
                || field.file_name.is_some()
                || field.media_type.as_deref() != Some("text/plain; charset=utf-8")
        })
    {
        return Err(TopazError::Input(
            "Canonical scalar fields must retain source declaration order",
        ));
    }
    let mut limits = TransportLimits::conformance();
    limits.maximum_body_bytes = MAXIMUM_REQUEST_BYTES;
    Ok(MultipartField::encode_checked(&fields, boundary, limits)?)
}
fn filters_enabled(input: &Input) -> Result<bool, TopazError> {
    match input.node.as_str() {
        "TopazVideoEnhance" => Ok(input
            .settings
            .get("upscaler_enabled")
            .and_then(Value::as_bool)
            .ok_or(TopazError::Input("Missing required source flag"))?
            || boolean(&input.settings, "interpolation_enabled", false)?),
        "TopazVideoEnhanceV2" => Ok(string(
            input
                .settings
                .get("upscaler_model")
                .ok_or(TopazError::Input("Missing source dynamic upscaler"))?,
            "upscaler_model",
        )? != "Disabled"
            || string(
                input
                    .settings
                    .get("interpolation_model")
                    .ok_or(TopazError::Input("Missing source dynamic interpolation"))?,
                "interpolation_model",
            )? != "Disabled"),
        _ => Err(TopazError::Input("Unknown source video node")),
    }
}
fn successor(
    head: &RequestHead,
    body: &[u8],
    predecessor: &RetainedResponse,
    policy: RetryPolicy,
    maximum: usize,
    session: &mut Session,
    transport: &mut dyn TopazTransport,
) -> Result<RetainedResponse, TopazError> {
    transport.check_cancelled(Some(predecessor.handle))?;
    transport.authorize_continuation(predecessor, head)?;
    exchange_with_policy(head, body, maximum, policy, session, transport)
}
struct ApiRequest<'a> {
    path: &'a str,
    method: HttpMethod,
    body: &'a Value,
    model: ApiModel,
    labels: (Option<&'a str>, Option<&'a str>, bool),
}
fn api_successor(
    request: ApiRequest<'_>,
    predecessor: &RetainedResponse,
    credentials: CredentialAvailability,
    session: &mut Session,
    transport: &mut dyn TopazTransport,
) -> Result<(RetainedResponse, Value), TopazError> {
    let ApiRequest {
        path,
        method,
        body,
        model,
        labels,
    } = request;
    let head = request_head(path, method, credentials, false)?;
    let encoded = if method == HttpMethod::Get {
        vec![]
    } else {
        serde_json::to_vec(body)?
    };
    if encoded.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    transport.check_cancelled(Some(predecessor.handle))?;
    transport.authorize_continuation(predecessor, &head)?;
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    transport.source_request_labels(handle, labels.0, labels.1, labels.2)?;
    let retained = exchange_existing(
        &head,
        &encoded,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        session,
        transport,
    )?;
    let value = normalized(&retained, model, transport)?;
    Ok((retained, value))
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    boundary: &str,
    transport: &mut dyn TopazTransport,
    owner: &mut dyn TopazInputOwner,
) -> Result<InvocationResult, TopazError> {
    transport.check_cancelled(None)?;
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(TopazError::Input("Unknown source node"));
    }
    owner.verify_inputs(input)?;
    let image = input.node == "TopazImageEnhance";
    let (encoded, input_receipt, video) = if image {
        if input.image.as_ref().map(|image| image.batch) != Some(1) {
            return Err(TopazError::Input("Source requires exactly one image"));
        }
        let image = owner.image_upload(input, 4096 * 4096, "image/png")?;
        bounded_receipt(&image.receipt)?;
        if endpoint(&image.download_url)? {
            return Err(TopazError::Input(
                "Source image upload requires absolute URL",
            ));
        }
        let body = owner
            .validate_request_model(RequestModel::ImageEnhance, &image_request(input, &image)?)?;
        if !body.is_object() {
            return Err(TopazError::Input("Invalid canonical image request model"));
        }
        let fields = owner.image_model_fields(&body)?;
        let encoded = checked_multipart(&body, fields, boundary)?;
        (encoded, image.receipt, None)
    } else {
        if !filters_enabled(input)? {
            return Err(TopazError::Input("Both source filters disabled"));
        }
        let video = owner.original_video_source(input)?;
        bounded_receipt(&video.receipt)?;
        if video.bytes.is_empty() || video.bytes.len() > MAXIMUM_MEDIA_BYTES {
            return Err(ConformanceError::BodyLimit.into());
        }
        let body = owner
            .validate_request_model(RequestModel::VideoCreate, &video_request(input, &video)?)?;
        if !body.is_object() {
            return Err(TopazError::Input("Invalid canonical video request model"));
        }
        let encoded = serde_json::to_vec(&body)?;
        (encoded, video.receipt.clone(), Some(video))
    };
    if maximum_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    if encoded.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let path = if image {
        "/proxy/topaz/image/v1/enhance-gen/async"
    } else {
        "/proxy/topaz/video/"
    };
    let mut head = request_head(path, HttpMethod::Post, credentials, false)?;
    if image {
        let header = head
            .headers
            .iter_mut()
            .find(|header| header.name == "Content-Type")
            .ok_or(ConformanceError::InvalidRequest)?;
        header.value = format!("multipart/form-data; boundary={boundary}");
    }
    let mut session = Session::default();
    transport.check_cancelled(None)?;
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    transport.source_request_labels(
        handle,
        if image { None } else { Some("Creating task") },
        if image { None } else { Some("Task created") },
        true,
    )?;
    let cost = transport.request_cost(
        handle,
        &CostRequest {
            operation: "topaz.generation.maximum".into(),
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
        &encoded,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let created = normalized(
        &creation,
        if image {
            ApiModel::ImageCreation
        } else {
            ApiModel::VideoCreation
        },
        transport,
    )?;
    let id = retained_id(&created, if image { "process_id" } else { "requestId" })?;
    let mut responses = vec![];
    retain_response(&mut responses, creation)?;
    if let Some(video) = video {
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        let (accepted, response) = api_successor(
            ApiRequest {
                path: &format!("/proxy/topaz/video/{id}/accept"),
                method: HttpMethod::Patch,
                body: &json!({}),
                model: ApiModel::VideoAccept,
                labels: (Some("Preparing upload"), Some("Upload started"), true),
            },
            prior,
            credentials,
            &mut session,
            transport,
        )?;
        let urls = response
            .get("urls")
            .and_then(Value::as_array)
            .ok_or(TopazError::Response("Missing signed PUT URLs"))?;
        retain_response(&mut responses, accepted)?;
        if urls.len() > 1 {
            return Err(TopazError::Response(
                "Source does not support multipart video uploads",
            ));
        }
        let url = urls
            .first()
            .and_then(Value::as_str)
            .ok_or(TopazError::Response("Source signed PUT URL unavailable"))?;
        if endpoint(url)? {
            return Err(TopazError::Response(
                "Source signed PUT requires absolute URL",
            ));
        }
        let head = RequestHead {
            method: HttpMethod::Put,
            path: url.into(),
            secret_id: None,
            headers: vec![Header {
                name: "Content-Type".into(),
                value: "video/mp4".into(),
            }],
        };
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        let uploaded = successor(
            &head,
            &video.bytes,
            prior,
            SIGNED_PUT_POLICY,
            MAXIMUM_RESPONSE_BYTES,
            &mut session,
            transport,
        )?;
        let etags: Vec<&str> = uploaded
            .headers
            .iter()
            .filter(|header| header.name.eq_ignore_ascii_case("etag"))
            .map(|header| header.value.as_str())
            .collect();
        if etags.len() != 1 || etags.first().is_none_or(|etag| etag.is_empty()) {
            return Err(TopazError::Response("Missing or ambiguous source ETag"));
        }
        let completion = json!({"uploadResults":[{"partNum":1,"eTag":etags.first().ok_or(ConformanceError::InvalidResponse)?}]});
        retain_response(&mut responses, uploaded)?;
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        let (completed, _) = api_successor(
            ApiRequest {
                path: &format!("/proxy/topaz/video/{id}/complete-upload"),
                method: HttpMethod::Patch,
                body: &completion,
                model: ApiModel::VideoComplete,
                labels: (Some("Finalizing upload"), Some("Upload completed"), true),
            },
            prior,
            credentials,
            &mut session,
            transport,
        )?;
        retain_response(&mut responses, completed)?;
    }
    let poll_path = if image {
        format!("/proxy/topaz/image/v1/status/{id}")
    } else {
        format!("/proxy/topaz/video/{id}/status")
    };
    let mut active_attempts = 0u32;
    for _ in 0..4096 {
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        let retained = successor(
            &head,
            &[],
            prior,
            POLL_RETRY,
            MAXIMUM_RESPONSE_BYTES,
            &mut session,
            transport,
        )?;
        let status_response = normalized(
            &retained,
            if image {
                ApiModel::ImageStatus
            } else {
                ApiModel::VideoStatus
            },
            transport,
        )?;
        let status = python_strip(string(&status_response, "status")?).to_lowercase();
        let percentage = match status_response.get("progress") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or(TopazError::Response("Invalid normalized source progress"))?,
            ),
        };
        let source_usd = if image {
            Some(
                status_response
                    .get("credits")
                    .and_then(Value::as_i64)
                    .ok_or(TopazError::Response("Missing source credit count"))?
                    as f64
                    * 0.08,
            )
        } else {
            match status_response.get("estimates") {
                None | Some(Value::Null) => None,
                Some(estimates) => {
                    let cost = estimates
                        .get("cost")
                        .and_then(Value::as_array)
                        .and_then(|cost| cost.first())
                        .and_then(Value::as_i64)
                        .ok_or(TopazError::Response("Missing first source estimate"))?;
                    if cost == 0 {
                        None
                    } else {
                        Some(cost as f64 * 0.08)
                    }
                }
            }
        };
        let queued = [
            "created",
            "queued",
            "queueing",
            "submitted",
            "initializing",
            "wait",
            "waiting",
            "in_queue",
            "in-queue",
        ]
        .contains(&status.as_str());
        transport.report_task_progress(
            &retained,
            &TaskProgress {
                status: status.clone(),
                queued,
                percentage,
                source_usd,
                estimated_seconds: if image { Some(60) } else { None },
            },
        )?;
        let handle = retained.handle;
        retain_response(&mut responses, retained)?;
        if [
            "complete",
            "completed",
            "succeeded",
            "success",
            "done",
            "finished",
        ]
        .contains(&status.as_str())
        {
            let url = if image {
                let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
                let (metadata, value) = api_successor(
                    ApiRequest {
                        path: &format!("/proxy/topaz/image/v1/download/{id}"),
                        method: HttpMethod::Get,
                        body: &Value::Null,
                        model: ApiModel::ImageDownload,
                        labels: (None, None, false),
                    },
                    prior,
                    credentials,
                    &mut session,
                    transport,
                )?;
                let url = string(&value, "download_url")?.to_string();
                retain_response(&mut responses, metadata)?;
                url
            } else {
                string(
                    status_response
                        .get("download")
                        .ok_or(TopazError::Response("Missing source video download"))?,
                    "url",
                )?
                .to_string()
            };
            if endpoint(&url)? {
                return Err(TopazError::Response(
                    "Source media GET requires absolute URL",
                ));
            }
            let head = request_head(&url, HttpMethod::Get, credentials, true)?;
            let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
            let downloaded = successor(
                &head,
                &[],
                prior,
                DOWNLOAD_RETRY,
                MAXIMUM_MEDIA_BYTES,
                &mut session,
                transport,
            )?;
            return Ok(InvocationResult {
                node: input.node.clone(),
                input_receipt,
                responses,
                downloaded,
                cost_receipt: cost.receipt,
                url,
            });
        }
        if [
            "cancelled",
            "canceled",
            "canceling",
            "fail",
            "failed",
            "error",
        ]
        .contains(&status.as_str())
        {
            return Err(TopazError::Response("Source task failed"));
        }
        if !queued {
            active_attempts += 1;
            if active_attempts >= 480 {
                return Err(ConformanceError::PollLimit.into());
            }
        }
        transport.check_cancelled(Some(handle))?;
        transport.delay(handle, if image { 8000 } else { 10000 })?;
    }
    Err(ConformanceError::PollLimit.into())
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, TopazError> {
    let name = match node {
        "TopazImageEnhance" => "IMAGE",
        "TopazVideoEnhance" | "TopazVideoEnhanceV2" => "VIDEO",
        _ => return Err(TopazError::Input("Unknown source output node")),
    };
    let registry =
        TypeRegistry::built_in().map_err(|_| TopazError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve(name)
        .map_err(|_| TopazError::Response("Missing canonical source type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| TopazError::Response("Missing canonical family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub trait TopazOutputOwner {
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
    transport: &mut dyn TopazTransport,
    owner: &mut dyn TopazOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), TopazError> {
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
        "/../../sim_test_support/fixtures/providers/topaz/source-contract.json"
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
    if source["sources"]["comfy_api_nodes/nodes_topaz.py"] != SOURCE_SHA256 {
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
    impl TopazTransport for TestTransport {
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
    fn api_model(path: &str) -> ApiModel {
        if path.contains("/image/") {
            if path.ends_with("/async") {
                ApiModel::ImageCreation
            } else if path.contains("/status/") {
                ApiModel::ImageStatus
            } else {
                ApiModel::ImageDownload
            }
        } else if path.ends_with("/accept") {
            ApiModel::VideoAccept
        } else if path.ends_with("/complete-upload") {
            ApiModel::VideoComplete
        } else if path.ends_with("/status") {
            ApiModel::VideoStatus
        } else {
            ApiModel::VideoCreation
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
            normalized: normalized.map(|value| (api_model(path), value)),
            failure: None,
        })
    }
    fn scripts(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let requests = case["requests"].as_array().ok_or("requests")?;
        let polls = case["polls"].as_array().ok_or("polls")?;
        let puts = case["puts"].as_array().ok_or("puts")?;
        let downloads = case["downloads"].as_array().ok_or("downloads")?;
        let mut script = vec![];
        for event in case["events"].as_array().ok_or("events")? {
            if let Some(path) = event["request"].as_str() {
                let record = requests
                    .iter()
                    .find(|record| record["path"] == path)
                    .ok_or("source request")?;
                let method = method(record["method"].as_str().ok_or("method")?)?;
                let body = if record.get("wire_hex").is_some() {
                    hex(&record["wire_hex"])?
                } else if method == HttpMethod::Get {
                    vec![]
                } else {
                    serde_json::to_vec(&record["source_payload"]["json"])?
                };
                script.push(response(
                    path,
                    method,
                    body,
                    &record["response"],
                    record.get("normalized_response").cloned(),
                )?);
            } else if let Some(path) = event["poll"].as_str() {
                let record = polls
                    .iter()
                    .find(|record| record["path"] == path)
                    .ok_or("source poll")?;
                for poll in record["responses"]
                    .as_array()
                    .ok_or("source poll responses")?
                {
                    script.push(response(
                        path,
                        HttpMethod::Get,
                        vec![],
                        &poll["raw"],
                        Some(poll["normalized"].clone()),
                    )?);
                }
            } else if let Some(url) = event["put"].as_str() {
                let record = puts
                    .iter()
                    .find(|record| record["url"] == url)
                    .ok_or("source put")?;
                let mut retained = response(
                    url,
                    HttpMethod::Put,
                    hex(&record["bytes_hex"])?,
                    &json!({}),
                    None,
                )?;
                retained.bytes = vec![];
                retained.headers = record["response_headers"]
                    .as_object()
                    .ok_or("source headers")?
                    .iter()
                    .map(|(name, value)| {
                        Ok(Header {
                            name: name.clone(),
                            value: value.as_str().ok_or("header value")?.into(),
                        })
                    })
                    .collect::<Result<_, Box<dyn Error>>>()?;
                if record.get("error").is_some() {
                    retained.failure = Some(ConformanceError::HostFailure);
                }
                script.push(retained);
            } else if let Some(url) = event["download"].as_str() {
                let record = downloads
                    .iter()
                    .find(|record| record["url"] == url)
                    .ok_or("source download")?;
                let mut retained = response(url, HttpMethod::Get, vec![], &json!({}), None)?;
                retained.bytes = hex(&record["bytes_hex"])?;
                script.push(retained);
            }
        }
        Ok(script)
    }
    struct InputOwner {
        case: Value,
        events: Events,
        uploads: usize,
        scalar_bound: bool,
    }
    impl InputOwner {
        fn new(case: &Value, events: Events) -> Self {
            Self {
                case: case.clone(),
                events,
                uploads: 0,
                scalar_bound: true,
            }
        }
    }
    impl TopazInputOwner for InputOwner {
        fn validate_request_model(
            &mut self,
            _: RequestModel,
            body: &Value,
        ) -> Result<Value, ConformanceError> {
            let request = self.case["requests"]
                .as_array()
                .and_then(|requests| requests.first())
                .ok_or(ConformanceError::InvalidRequest)?;
            if request["body"] != *body {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(request["body"].clone())
        }
        fn verify_inputs(&mut self, input: &Input) -> Result<(), ConformanceError> {
            if json!(input.node) != self.case["input"]["node"] {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(())
        }
        fn image_upload(
            &mut self,
            input: &Input,
            pixels: u64,
            mime: &str,
        ) -> Result<PreparedImage, ConformanceError> {
            if pixels != 4096 * 4096
                || mime != "image/png"
                || input.image.as_ref().is_none_or(|image| image.batch != 1)
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let record = self.case["uploads"]
                .as_array()
                .and_then(|uploads| uploads.get(self.uploads))
                .ok_or(ConformanceError::InvalidRequest)?;
            if record["mime_type"] != "image/png"
                || record["wait_label"] != "Uploading"
                || hex(&record["bytes_hex"])
                    .map_err(|_| ConformanceError::InvalidRequest)?
                    .get(..8)
                    != Some(b"\x89PNG\r\n\x1a\n")
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.uploads += 1;
            self.events.borrow_mut().push("upload:image".into());
            Ok(PreparedImage {
                download_url: "https://source.fixture.invalid/input-1.bin".into(),
                receipt: b"canonical input upload receipt".to_vec(),
            })
        }
        fn original_video_source(
            &mut self,
            input: &Input,
        ) -> Result<PreparedVideo, ConformanceError> {
            self.events
                .borrow_mut()
                .push("prepare:original-video".into());
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            if video.failure.is_some() {
                return Err(ConformanceError::HostFailure);
            }
            Ok(PreparedVideo {
                container: video.container.clone(),
                width: video.width,
                height: video.height,
                frame_rate: video.fps,
                duration: video.duration,
                frame_count: video.frames,
                bytes: include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/topaz/encoded-video.mp4"
                ))
                .to_vec(),
                receipt: b"canonical original video receipt".to_vec(),
            })
        }
        fn image_model_fields(
            &mut self,
            body: &Value,
        ) -> Result<Vec<MultipartField>, ConformanceError> {
            if !self.scalar_bound {
                return Err(ConformanceError::HostFailure);
            }
            let request = self.case["requests"]
                .as_array()
                .and_then(|requests| requests.first())
                .ok_or(ConformanceError::InvalidRequest)?;
            if request["body"] != *body {
                return Err(ConformanceError::InvalidRequest);
            }
            request["form_fields"]
                .as_array()
                .ok_or(ConformanceError::InvalidRequest)?
                .iter()
                .map(|field| {
                    Ok(MultipartField {
                        name: field["name"]
                            .as_str()
                            .ok_or(ConformanceError::InvalidRequest)?
                            .into(),
                        file_name: None,
                        media_type: Some("text/plain; charset=utf-8".into()),
                        bytes: field["value"]
                            .as_str()
                            .ok_or(ConformanceError::InvalidRequest)?
                            .as_bytes()
                            .to_vec(),
                    })
                })
                .collect()
        }
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    const BOUNDARY: &str = "sim-topaz-canonical-boundary";
    fn first_case(node: &str) -> Result<Value, Box<dyn Error>> {
        source_oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == node)
            .cloned()
            .ok_or("node case".into())
    }
    #[test]
    fn exact_claims_original_video_and_source_schema_defaults() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/topaz/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 3);
        assert_eq!(ROUTES.len(), 9);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "synthetic_prefix_tombstone")
                .count(),
            2
        );
        let source = source_oracle()?;
        assert_eq!(source["cases"].as_array().ok_or("cases")?.len(), 102);
        assert_eq!(source["errors"].as_array().ok_or("errors")?.len(), 36);
        let case = first_case("TopazImageEnhance")?;
        assert_eq!(
            case["requests"][0]["body"]["face_enhancement_creativity"],
            1.0
        );
        assert_eq!(
            case["requests"][0]["body"]["face_enhancement_strength"],
            0.8
        );
        for case in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| case["input"]["node"] != "TopazImageEnhance")
        {
            assert_eq!(
                hex(&case["puts"][0]["bytes_hex"])?[..],
                *include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/topaz/encoded-video.mp4"
                ))
            );
            assert_eq!(case["requests"][1]["source_payload"]["json"], json!({}));
            assert!(case["requests"][1]["body"].is_null());
        }
        Ok(())
    }
    #[test]
    fn all_actual_source_requests_multipart_uploads_and_ordered_errors()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (errors, rows) in [(false, &source["cases"]), (true, &source["errors"])] {
            for (index, case) in rows.as_array().ok_or("rows")?.iter().enumerate() {
                let input: Input = serde_json::from_value(case["input"].clone())?;
                let events = Rc::new(RefCell::new(vec![]));
                let script = scripts(case)?;
                let expected_heads: Vec<_> = script
                    .iter()
                    .map(|response| (response.path.clone(), response.method))
                    .collect();
                let mut transport = TestTransport::new(script, events.clone());
                let mut owner = InputOwner::new(case, events);
                let result = invoke(
                    &input,
                    credentials(),
                    1_000_000,
                    BOUNDARY,
                    &mut transport,
                    &mut owner,
                );
                if errors {
                    if result.is_ok() {
                        return Err(format!(
                            "source error {index}: {} unexpectedly succeeded",
                            input.node
                        )
                        .into());
                    }
                } else {
                    let result = result
                        .map_err(|error| format!("source case {index}: {}: {error}", input.node))?;
                    transport.finish()?;
                    assert_eq!(
                        result.downloaded.bytes,
                        hex(&case["downloads"][0]["bytes_hex"])?
                    );
                    assert_eq!(result.url, case["downloads"][0]["url"]);
                    if input.node != "TopazImageEnhance" {
                        assert!(
                            transport
                                .heads
                                .iter()
                                .any(|head| head.method == HttpMethod::Put
                                    && head.headers
                                        == vec![Header {
                                            name: "Content-Type".into(),
                                            value: "video/mp4".into()
                                        }])
                        );
                    }
                }
                assert_eq!(
                    transport
                        .heads
                        .iter()
                        .map(|head| (head.path.clone(), head.method))
                        .collect::<Vec<_>>(),
                    expected_heads,
                    "source case {index} error={errors} {}",
                    input.node
                );
                assert_eq!(
                    owner.uploads,
                    case["uploads"].as_array().ok_or("uploads")?.len()
                );
                let statuses = case["polls"]
                    .as_array()
                    .ok_or("polls")?
                    .iter()
                    .flat_map(|poll| {
                        poll["statuses"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .map(move |status| (status, poll))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(transport.progress.len(), statuses.len());
                for (actual, (expected, poll)) in transport.progress.iter().zip(statuses) {
                    assert_eq!(
                        actual.0,
                        expected["status"].as_str().ok_or("status")?.to_lowercase()
                    );
                    assert_eq!(actual.2, expected["progress"].as_f64());
                    assert_eq!(actual.3, expected["price"].as_f64());
                    assert_eq!(actual.4, poll["estimated_duration"].as_u64());
                }
            }
        }
        Ok(())
    }
    #[test]
    fn canonical_scalar_spellings_are_injected_and_wire_is_not_leaf_encoded()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let mut spellings = BTreeSet::new();
        for case in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| case["input"]["node"] == "TopazImageEnhance")
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let events = Rc::new(RefCell::new(vec![]));
            let mut owner = InputOwner::new(case, events);
            let image = owner.image_upload(&input, 4096 * 4096, "image/png")?;
            let body = image_request(&input, &image)?;
            if !body.is_object() {
                return Err(TopazError::Input("Invalid canonical image request model").into());
            }
            let fields = owner.image_model_fields(&body)?;
            for field in &fields {
                if field.name == "face_enhancement_creativity" {
                    spellings.insert(std::str::from_utf8(&field.bytes)?.to_string());
                }
            }
            assert_eq!(
                checked_multipart(&body, fields.clone(), BOUNDARY)?,
                hex(&case["requests"][0]["wire_hex"])?
            );
            let mut reversed = fields.clone();
            reversed.reverse();
            assert!(checked_multipart(&body, reversed, BOUNDARY).is_err());
            assert!(checked_multipart(&body, fields, "bad\r\nBoundary").is_err());
        }
        for required in ["1.0", "0.0", "-0.0", "1e-06", "1e+16"] {
            assert!(spellings.contains(required));
        }
        Ok(())
    }
    #[test]
    fn source_geometry_astra_limits_and_disabled_filters_precede_video_owner()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for case in source["errors"]
            .as_array()
            .ok_or("errors")?
            .iter()
            .filter(|case| {
                case["error_message"]
                    .as_str()
                    .is_some_and(|message| message.contains("both upscaling"))
            })
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let events = Rc::new(RefCell::new(vec![]));
            let mut owner = InputOwner::new(case, events.clone());
            let mut transport = TestTransport::new(vec![], events.clone());
            assert!(
                invoke(
                    &input,
                    credentials(),
                    1,
                    BOUNDARY,
                    &mut transport,
                    &mut owner
                )
                .is_err()
            );
            assert!(events.borrow().is_empty());
        }
        let case = first_case("TopazVideoEnhanceV2")?;
        let mut input: Input = serde_json::from_value(case["input"].clone())?;
        let mut owner = InputOwner::new(&case, Rc::new(RefCell::new(vec![])));
        let mut video = owner.original_video_source(&input)?;
        input.settings["upscaler_model"]["prompt"] = json!(" \u{1c}scene \u{1f}");
        video.frame_count = 450;
        assert_eq!(
            video_request(&input, &video)?["filters"][0]["prompt"],
            "scene"
        );
        video.frame_count = 451;
        assert!(video_request(&input, &video).is_err());
        input.settings["upscaler_model"]["prompt"] = json!("");
        video.frame_count = 9000;
        assert!(video_request(&input, &video).is_ok());
        video.frame_count = 9001;
        assert!(video_request(&input, &video).is_err());
        video.frame_count = 1;
        video.width = 123;
        video.height = 321;
        let body = video_request(&input, &video)?;
        assert_eq!(body["output"]["resolution"]["width"], 736);
        assert_eq!(body["output"]["resolution"]["height"], 1920);
        Ok(())
    }
    #[test]
    fn native_scalar_and_continuation_owners_fail_closed_without_publication()
    -> Result<(), Box<dyn Error>> {
        struct Unbound;
        impl TopazInputOwner for Unbound {}
        impl TopazOutputOwner for Unbound {}
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
        let case = first_case("TopazImageEnhance")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let events = Rc::new(RefCell::new(vec![]));
        let mut transport = TestTransport::new(scripts(&case)?, events.clone());
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                1,
                BOUNDARY,
                &mut transport,
                &mut Unbound
            ),
            Err(TopazError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        let mut owner = InputOwner::new(&case, events.clone());
        owner.scalar_bound = false;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                1,
                BOUNDARY,
                &mut transport,
                &mut owner
            ),
            Err(TopazError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        assert_eq!(owner.uploads, 1);
        let mut owner = InputOwner::new(&case, events);
        transport.continuation_bound = false;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                1,
                BOUNDARY,
                &mut transport,
                &mut owner
            ),
            Err(TopazError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(transport.heads.len(), 1);
        transport = TestTransport::new(scripts(&case)?, Rc::new(RefCell::new(vec![])));
        let mut owner = InputOwner::new(&case, Rc::new(RefCell::new(vec![])));
        let result = invoke(
            &input,
            credentials(),
            1,
            BOUNDARY,
            &mut transport,
            &mut owner,
        )?;
        let mut sink = Sink(0);
        assert!(materialize_and_publish(&result, &mut transport, &mut Unbound, &mut sink).is_err());
        assert_eq!(sink.0, 0);
        Ok(())
    }
    #[test]
    fn retained_response_headers_and_scope_prevent_replay_and_worker_loss()
    -> Result<(), Box<dyn Error>> {
        let case = first_case("TopazVideoEnhance")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let events = Rc::new(RefCell::new(vec![]));
        let mut transport = TestTransport::new(scripts(&case)?, events.clone());
        let mut owner = InputOwner::new(&case, events);
        let result = invoke(
            &input,
            credentials(),
            1,
            BOUNDARY,
            &mut transport,
            &mut owner,
        )?;
        let creation = result.responses.first().ok_or("creation")?;
        let raw: Value = serde_json::from_slice(&creation.bytes)?;
        let mut altered = creation.clone();
        altered.headers.push(Header {
            name: "Etag".into(),
            value: "substitute".into(),
        });
        assert!(
            transport
                .validate_response(&altered, ApiModel::VideoCreation, &raw)
                .is_err()
        );
        altered = creation.clone();
        altered.receipt.push(1);
        assert!(
            transport
                .validate_response(&altered, ApiModel::VideoCreation, &raw)
                .is_err()
        );
        let mut fresh = TestTransport::new(scripts(&case)?, Rc::new(RefCell::new(vec![])));
        fresh.invocation = 43;
        assert!(
            fresh
                .validate_response(creation, ApiModel::VideoCreation, &raw)
                .is_err()
        );
        fresh.failure = Some(ConformanceError::WorkerLost);
        let mut owner = InputOwner::new(&case, Rc::new(RefCell::new(vec![])));
        assert!(matches!(
            invoke(&input, credentials(), 1, BOUNDARY, &mut fresh, &mut owner),
            Err(TopazError::Transport(ConformanceError::WorkerLost))
        ));
        assert!(fresh.heads.is_empty());
        Ok(())
    }
    #[test]
    fn retry_is_fresh_and_signed_put_does_not_retry() -> Result<(), Box<dyn Error>> {
        let case = first_case("TopazVideoEnhance")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut script = scripts(&case)?;
        let mut unavailable = response(
            "/proxy/topaz/video/",
            HttpMethod::Post,
            script.first().ok_or("first")?.body.clone(),
            &json!({}),
            None,
        )?;
        unavailable.status = 503;
        script.insert(0, unavailable);
        let events = Rc::new(RefCell::new(vec![]));
        let mut transport = TestTransport::new(script, events.clone());
        let mut owner = InputOwner::new(&case, events);
        let result = invoke(
            &input,
            credentials(),
            1,
            BOUNDARY,
            &mut transport,
            &mut owner,
        )?;
        assert!(!result.responses.is_empty());
        assert_eq!(transport.delays.first(), Some(&1000));
        assert_ne!(
            transport
                .completed
                .first()
                .ok_or("first retry terminal")?
                .handle,
            transport
                .completed
                .get(1)
                .ok_or("second retry terminal")?
                .handle
        );
        let mut script = scripts(&case)?;
        let put = script
            .iter_mut()
            .find(|response| response.method == HttpMethod::Put)
            .ok_or("put")?;
        put.status = 503;
        let events = Rc::new(RefCell::new(vec![]));
        let mut transport = TestTransport::new(script, events.clone());
        let mut owner = InputOwner::new(&case, events);
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                1,
                BOUNDARY,
                &mut transport,
                &mut owner
            ),
            Err(TopazError::Http { status: 503, .. })
        ));
        assert_eq!(
            transport
                .heads
                .iter()
                .filter(|head| head.method == HttpMethod::Put)
                .count(),
            1
        );
        assert!(transport.delays.is_empty());
        Ok(())
    }
    #[test]
    fn cumulative_receipts_and_headers_are_bounded_before_retention() -> Result<(), Box<dyn Error>>
    {
        let mut retained = vec![];
        let response = RetainedResponse {
            handle: StreamHandle {
                invocation: 1,
                slot: 1,
                generation: 1,
            },
            bytes: vec![],
            headers: vec![],
            receipt: vec![1; MAXIMUM_RECEIPT_BYTES],
        };
        for _ in 0..MAXIMUM_MEDIA_BYTES / MAXIMUM_RECEIPT_BYTES {
            retain_response(&mut retained, response.clone())?;
        }
        assert!(matches!(
            retain_response(&mut retained, response),
            Err(TopazError::Transport(ConformanceError::BodyLimit))
        ));
        assert_eq!(retained.len(), MAXIMUM_MEDIA_BYTES / MAXIMUM_RECEIPT_BYTES);
        Ok(())
    }
    #[test]
    fn cancellation_rejected_budget_and_bounded_responses_stop_before_later_requests()
    -> Result<(), Box<dyn Error>> {
        let case = first_case("TopazImageEnhance")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        for (cancelled, reject_cost) in [(true, false), (false, true)] {
            let events = Rc::new(RefCell::new(vec![]));
            let mut transport = TestTransport::new(scripts(&case)?, events.clone());
            transport.cancelled = cancelled;
            transport.reject_cost = reject_cost;
            let mut owner = InputOwner::new(&case, events);
            assert!(
                invoke(
                    &input,
                    credentials(),
                    1,
                    BOUNDARY,
                    &mut transport,
                    &mut owner
                )
                .is_err()
            );
            assert!(transport.heads.len() <= 1);
        }
        let mut script = scripts(&case)?;
        script.first_mut().ok_or("first")?.bytes = vec![0; MAXIMUM_RESPONSE_BYTES + 1];
        let events = Rc::new(RefCell::new(vec![]));
        let mut transport = TestTransport::new(script, events.clone());
        let mut owner = InputOwner::new(&case, events);
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                1,
                BOUNDARY,
                &mut transport,
                &mut owner
            ),
            Err(TopazError::Transport(ConformanceError::BodyLimit))
        ));
        assert_eq!(transport.heads.len(), 1);
        Ok(())
    }
}
