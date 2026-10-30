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
pub const NAMESPACE: &str = "zed.sim.provider.bytedance";
pub const ALIASES: &[&str] = &["byteplus", "byteplus-seedance2", "seedance"];
pub const SOURCE_SHA256: &str = "16dbad871ceef57796ee77794da976fa6e95ea03fb436b577b1d31dfede52247";
pub const LLM_SOURCE_SHA256: &str =
    "dc6b3e23d101ac5b384cb9c5b445eead54faacff95834082f1a8ab95b23d9220";
pub const IMAGE_PATH: &str = "/proxy/byteplus/api/v3/images/generations";
pub const VIDEO_PATH: &str = "/proxy/byteplus/api/v3/contents/generations/tasks";
pub const VIDEO2_PATH: &str = "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks";
pub const TEXT_PATH: &str = "/proxy/byteplus/api/v3/responses";
pub const AUDIO_PATH: &str = "/proxy/byteplus/api/v3/tts/create";
pub const MAXIMUM_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
    pub source_class: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0031",
        node_identifier: "ByteDance2FirstLastFrameNode",
        source_class: "ByteDance2FirstLastFrameNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0032",
        node_identifier: "ByteDance2ReferenceNode",
        source_class: "ByteDance2ReferenceNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0033",
        node_identifier: "ByteDance2TextToVideoNode",
        source_class: "ByteDance2TextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0034",
        node_identifier: "ByteDanceCreateImageAsset",
        source_class: "ByteDanceCreateImageAsset",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0035",
        node_identifier: "ByteDanceCreateVideoAsset",
        source_class: "ByteDanceCreateVideoAsset",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0036",
        node_identifier: "ByteDanceFirstLastFrameNode",
        source_class: "ByteDanceFirstLastFrameNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0037",
        node_identifier: "ByteDanceImageNode",
        source_class: "ByteDanceImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0038",
        node_identifier: "ByteDanceImageReferenceNode",
        source_class: "ByteDanceImageReferenceNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0039",
        node_identifier: "ByteDanceImageToVideoNode",
        source_class: "ByteDanceImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0040",
        node_identifier: "ByteDanceSeedAudio",
        source_class: "ByteDanceSeedAudioNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0041",
        node_identifier: "ByteDanceSeedNode",
        source_class: "ByteDanceSeedNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0042",
        node_identifier: "ByteDanceSeedreamNode",
        source_class: "ByteDanceSeedreamNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0043",
        node_identifier: "ByteDanceSeedreamNodeV2",
        source_class: "ByteDanceSeedreamNodeV2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0044",
        node_identifier: "ByteDanceTextToVideoNode",
        source_class: "ByteDanceTextToVideoNode",
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
        feature_id: "COMFY-API-EXT-0020",
        method: "POST",
        path: "/proxy/byteplus/api/v3/contents/generations/tasks",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0021",
        method: "POST",
        path: "/proxy/byteplus/api/v3/images/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0022",
        method: "POST",
        path: "/proxy/byteplus/api/v3/responses",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0023",
        method: "POST",
        path: "/proxy/byteplus/api/v3/tts/create",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0024",
        method: "GET",
        path: "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0155",
        method: "POST",
        path: "/proxy/seedance/assets",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0156",
        method: "GET",
        path: "/proxy/seedance/assets/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0157",
        method: "GET",
        path: "/proxy/seedance/assets/{asset_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0158",
        method: "POST",
        path: "/proxy/seedance/virtual-library/assets",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0159",
        method: "POST",
        path: "/proxy/seedance/visual-validate/sessions",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0160",
        method: "GET",
        path: "/proxy/seedance/visual-validate/sessions/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0161",
        method: "GET",
        path: "/proxy/seedance/visual-validate/sessions/{session.session_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0219",
        method: "GET",
        path: "/proxy/byteplus/api/v3/contents/generations/tasks/{initial_response.id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0220",
        method: "GET",
        path: "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks/{initial_response.id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum BytedanceError {
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
impl fmt::Display for BytedanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => write!(formatter, "Bytedance task failed: {message}"),
            Self::Http { status, .. } => write!(formatter, "Bytedance HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for BytedanceError {}
impl From<serde_json::Error> for BytedanceError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
impl From<ConformanceError> for BytedanceError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
fn endpoint(value: &str) -> Result<bool, BytedanceError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(BytedanceError::Response(
            "Invalid Bytedance response endpoint",
        ));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(BytedanceError::Response(
            "Invalid Bytedance response endpoint",
        ))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(BytedanceError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(BytedanceError::Response(
            "Invalid Bytedance endpoint authority",
        ));
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
) -> Result<RequestHead, BytedanceError> {
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
            return Err(BytedanceError::Input("Missing Comfy API credentials"));
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

pub trait BytedanceTransport: ProviderTransport {
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
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn BytedanceTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BytedanceError> {
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
    transport: &mut dyn BytedanceTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BytedanceError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(BytedanceError::Http {
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
                    return Err(BytedanceError::Http {
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
    transport: &mut dyn BytedanceTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BytedanceError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "bytedance.generation.maximum".into(),
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
                    return Err(BytedanceError::Http {
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

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: serde_json::Map<String, Value>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct MediaBinding {
    pub slot: String,
    pub kind: String,
    pub url: String,
    pub mime_type: Option<String>,
    pub wait_label: String,
}
#[derive(Clone, Debug, Default)]
pub struct Resolution {
    pub media: Vec<MediaBinding>,
    pub assets: BTreeMap<String, String>,
    pub group_id: Option<String>,
    pub inline_media: BTreeMap<String, String>,
}
#[derive(Clone, Debug)]
pub struct Invocation {
    pub node: String,
    pub head: RequestHead,
    pub body: Vec<u8>,
    pub poll: Option<PollPlan>,
    pub output: OutputKind,
    pub fail_on_partial: bool,
    pub model_id: String,
    pub has_video_input: bool,
    pub resolution: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputKind {
    Image,
    Video,
    Audio,
    Text,
    Asset,
}
#[derive(Clone, Debug)]
pub struct PollPlan {
    pub prefix: String,
    pub interval_milliseconds: u64,
    pub maximum_nonqueued_polls: u32,
    pub completed_status: String,
    pub failed_status: String,
    pub estimated_seconds: Option<u64>,
}
fn oracle() -> Result<Value, BytedanceError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/bytedance/source-contract.json"
    )))?)
}
fn text(value: &Value, key: &str) -> Result<String, BytedanceError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or(BytedanceError::Input("Missing Bytedance string argument"))
}
fn optional_text(value: &Value, key: &str, default: &str) -> Result<String, BytedanceError> {
    match value.get(key) {
        Some(value) => value
            .as_str()
            .map(str::to_owned)
            .ok_or(BytedanceError::Input("Invalid Bytedance string argument")),
        None => Ok(default.into()),
    }
}
fn number(value: &Value, key: &str, default: u64) -> Result<u64, BytedanceError> {
    match value.get(key) {
        Some(value) => value
            .as_u64()
            .ok_or(BytedanceError::Input("Invalid Bytedance integer argument")),
        None => Ok(default),
    }
}
fn boolean(value: &Value, key: &str, default: bool) -> Result<bool, BytedanceError> {
    match value.get(key) {
        Some(value) => value
            .as_bool()
            .ok_or(BytedanceError::Input("Invalid Bytedance boolean argument")),
        None => Ok(default),
    }
}
fn model_id(table: &Value, label: &str) -> Result<String, BytedanceError> {
    table
        .get(label)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or(BytedanceError::Input("Unknown Bytedance model"))
}
fn prompt(value: &str, maximum: usize) -> Result<(), BytedanceError> {
    if value.trim().is_empty() || value.len() > maximum {
        return Err(BytedanceError::Input("Invalid Bytedance prompt"));
    }
    Ok(())
}
fn active_id(value: &str) -> Result<&str, BytedanceError> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > 1024
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(BytedanceError::Response(
            "Invalid Bytedance task or asset identity",
        ));
    }
    Ok(value)
}
fn media_count(value: &Value) -> u64 {
    if let Some(count) = value.get("count").and_then(Value::as_u64) {
        count
    } else if let Some(map) = value.as_object() {
        map.values().map(media_count).sum()
    } else {
        0
    }
}
fn urls(resolution: &Resolution, kind: &str) -> Vec<String> {
    resolution
        .media
        .iter()
        .filter(|media| media.kind == kind)
        .map(|media| media.url.clone())
        .collect()
}
fn image_content(url: &str, role: Option<&str>) -> Value {
    let mut value = json!({"type":"image_url","image_url":{"url":url}});
    if let Some(role) = role {
        value["role"] = role.into();
    }
    value
}
fn video_content(url: &str) -> Value {
    json!({"type":"video_url","video_url":{"url":url},"role":"reference_video"})
}
fn audio_content(url: &str) -> Value {
    json!({"type":"audio_url","audio_url":{"url":url},"role":"reference_audio"})
}
fn size(arguments: &Value, presets: &Value, default: &str) -> Result<(u64, u64), BytedanceError> {
    let preset = optional_text(arguments, "size_preset", default)?;
    if let Some(entries) = presets.as_array() {
        for entry in entries {
            if entry.get(0).and_then(Value::as_str) == Some(preset.as_str())
                && let (Some(width), Some(height)) = (
                    entry.get(1).and_then(Value::as_u64),
                    entry.get(2).and_then(Value::as_u64),
                )
            {
                return Ok((width, height));
            }
        }
    }
    Ok((
        number(arguments, "width", 2048)?,
        number(arguments, "height", 2048)?,
    ))
}
pub fn build_invocation(
    input: &Input,
    resolved: &Resolution,
    credentials: CredentialAvailability,
) -> Result<Invocation, BytedanceError> {
    let node = if input.node == "ByteDanceSeedAudio" {
        "ByteDanceSeedAudioNode"
    } else {
        input.node.as_str()
    };
    if !NODES.iter().any(|claim| claim.source_class == node) {
        return Err(BytedanceError::Input("Unknown Bytedance node"));
    }
    let arguments = Value::Object(input.arguments.clone());
    let source = oracle()?;
    let constants = &source["constants"];
    let fields = constants["argument_names"][node]
        .as_array()
        .ok_or(BytedanceError::Input("Missing source argument contract"))?;
    if input.arguments.keys().any(|key| {
        !fields
            .iter()
            .any(|field| field.as_str() == Some(key.as_str()))
    }) {
        return Err(BytedanceError::Input("Unknown Bytedance node argument"));
    }
    let seed = number(&arguments, "seed", 0)?;
    if seed > 2147483647 {
        return Err(BytedanceError::Input("Bytedance seed exceeds schema"));
    }
    let watermark = boolean(&arguments, "watermark", false)?;
    for media in &resolved.media {
        if media.url.starts_with("asset://") {
            active_id(media.url.trim_start_matches("asset://"))?;
        } else if endpoint(&media.url)? {
            return Err(BytedanceError::Input(
                "Bytedance media URL must be absolute",
            ));
        }
    }
    let dynamic = arguments
        .get("model")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or(Value::Null);
    let mut body;
    let mut poll = None;
    let mut fail_on_partial = false;
    let mut output = OutputKind::Video;
    let mut model = String::new();
    let mut resolution = String::new();
    let mut has_video_input = false;
    let path = match node {
        "ByteDanceImageNode" => {
            let instruction = text(&arguments, "prompt")?;
            prompt(&instruction, 65536)?;
            model = text(&arguments, "model")?;
            let (width, height) = size(&arguments, &constants["RECOMMENDED_PRESETS"], "Custom")?;
            if !(512..=2048).contains(&width) || !(512..=2048).contains(&height) {
                return Err(BytedanceError::Input(
                    "Bytedance custom image size out of range",
                ));
            }
            let guidance = arguments
                .get("guidance_scale")
                .and_then(Value::as_f64)
                .ok_or(BytedanceError::Input("Missing guidance scale"))?;
            if !guidance.is_finite() || !(1.0..=10.0).contains(&guidance) {
                return Err(BytedanceError::Input("Invalid guidance scale"));
            }
            body = json!({"model":model,"prompt":instruction,"response_format":"url","size":format!("{width}x{height}"),"seed":seed,"guidance_scale":guidance,"watermark":watermark});
            output = OutputKind::Image;
            IMAGE_PATH
        }
        "ByteDanceSeedreamNode" | "ByteDanceSeedreamNodeV2" => {
            let instruction = text(&arguments, "prompt")?;
            prompt(&instruction, 65536)?;
            let modern = node.ends_with("V2");
            let controls = if modern { &dynamic } else { &arguments };
            model = model_id(&constants["SEEDREAM_MODELS"], &text(controls, "model")?)?;
            let presets = if modern {
                &constants["SEEDREAM_PRESETS"][&model]
            } else {
                &constants["RECOMMENDED_PRESETS_SEEDREAM_4"]
            };
            let default = presets
                .get(0)
                .and_then(|value| value.get(0))
                .and_then(Value::as_str)
                .unwrap_or("Custom");
            let (width, height) = size(controls, presets, default)?;
            let pixels = width
                .checked_mul(height)
                .ok_or(BytedanceError::Input("Bytedance image size overflow"))?;
            let pro = model.contains("seedream-5-0-pro");
            let minimum = if pro || model.contains("seedream-4-0") {
                921600
            } else {
                3686400
            };
            let maximum = if pro {
                4194304
            } else if !modern && model.contains("seedream-5-0") {
                10404496
            } else {
                16777216
            };
            if pixels < minimum || pixels > maximum {
                return Err(BytedanceError::Input("Seedream pixel budget exceeded"));
            }
            let reference_count = if modern {
                media_count(&controls["images"])
            } else {
                media_count(&arguments["image"])
            };
            let maximum_inputs = if model == "seedream-5-0-260128" {
                14
            } else {
                10
            };
            if reference_count > maximum_inputs {
                return Err(BytedanceError::Input("Too many Seedream images"));
            }
            let maximum_images = number(controls, "max_images", 1)?;
            if !(1..=15).contains(&maximum_images) {
                return Err(BytedanceError::Input("Invalid Seedream image count"));
            }
            let sequential = if modern {
                if maximum_images == 1 {
                    "disabled".into()
                } else {
                    "auto".into()
                }
            } else {
                optional_text(controls, "sequential_image_generation", "disabled")?
            };
            if sequential == "auto" && reference_count + maximum_images > 15 {
                return Err(BytedanceError::Input(
                    "Seedream input and output images exceed15",
                ));
            }
            let references = urls(resolved, "image");
            if references.len() as u64 != reference_count {
                return Err(BytedanceError::Input(
                    "Missing canonical Seedream image uploads",
                ));
            }
            body = json!({"model":model,"prompt":instruction,"response_format":"url","image":references,"size":format!("{width}x{height}"),"seed":seed,"watermark":watermark});
            if !pro {
                body["sequential_image_generation"] = sequential.into();
                body["sequential_image_generation_options"] = json!({"max_images":maximum_images});
            }
            if !modern && model == "seedream-5-0-260128" {
                body["output_format"] = "png".into();
            }
            fail_on_partial = boolean(controls, "fail_on_partial", !modern)?;
            output = OutputKind::Image;
            IMAGE_PATH
        }
        "ByteDanceTextToVideoNode"
        | "ByteDanceImageToVideoNode"
        | "ByteDanceFirstLastFrameNode"
        | "ByteDanceImageReferenceNode" => {
            model = text(&arguments, "model")?;
            let mut instruction = text(&arguments, "prompt")?;
            prompt(&instruction, 65536)?;
            for flag in ["resolution", "ratio", "duration", "seed", "watermark"] {
                if instruction.contains(&format!("--{flag} ")) {
                    return Err(BytedanceError::Input("Reserved Bytedance prompt flag"));
                }
            }
            let camera = node != "ByteDanceImageReferenceNode";
            if camera && instruction.contains("--camerafixed ") {
                return Err(BytedanceError::Input("Reserved Bytedance prompt flag"));
            }
            resolution = text(&arguments, "resolution")?;
            let ratio = text(&arguments, "aspect_ratio")?;
            let duration = number(&arguments, "duration", 5)?;
            if model == "seedance-1-5-pro-251215" && duration < 4 {
                return Err(BytedanceError::Input("Minimum Seedance1.5 duration is4"));
            }
            instruction.push_str(&format!(
                " --resolution {resolution} --ratio {ratio} --duration {duration} --seed {seed}"
            ));
            if camera {
                instruction.push_str(&format!(
                    " --camerafixed {}",
                    boolean(&arguments, "camera_fixed", false)?
                ));
            }
            instruction.push_str(&format!(" --watermark {watermark}"));
            let mut content = vec![json!({"type":"text","text":instruction})];
            let references = urls(resolved, "image");
            let expected = match node {
                "ByteDanceTextToVideoNode" => 0,
                "ByteDanceImageToVideoNode" => 1,
                "ByteDanceFirstLastFrameNode" => 2,
                _ => media_count(&arguments["images"]).min(4),
            };
            if references.len() as u64 != expected {
                return Err(BytedanceError::Input("Missing legacy video image uploads"));
            }
            for (index, url) in references.iter().enumerate() {
                let role = match node {
                    "ByteDanceFirstLastFrameNode" => Some(if index == 0 {
                        "first_frame"
                    } else {
                        "last_frame"
                    }),
                    "ByteDanceImageReferenceNode" => Some("reference_image"),
                    _ => None,
                };
                content.push(image_content(url, role));
            }
            body = json!({"model":model,"content":content});
            if model == "seedance-1-5-pro-251215" && node != "ByteDanceImageReferenceNode" {
                body["generate_audio"] = boolean(&arguments, "generate_audio", false)?.into();
            }
            let estimate = constants["VIDEO_TASKS_EXECUTION_TIME"][&model][&resolution]
                .as_u64()
                .ok_or(BytedanceError::Input(
                    "Unsupported legacy video model/resolution",
                ))?
                .checked_mul(duration)
                .ok_or(BytedanceError::Input("Video estimate overflow"))?
                .div_ceil(10)
                .max(1);
            poll = Some(PollPlan {
                prefix: VIDEO_PATH.into(),
                interval_milliseconds: 5000,
                maximum_nonqueued_polls: 480,
                completed_status: "succeeded".into(),
                failed_status: "failed".into(),
                estimated_seconds: Some(estimate),
            });
            VIDEO_PATH
        }
        "ByteDance2TextToVideoNode"
        | "ByteDance2FirstLastFrameNode"
        | "ByteDance2ReferenceNode" => {
            let mut instruction = text(&dynamic, "prompt")?;
            prompt(&instruction, 65536)?;
            model = model_id(&constants["SEEDANCE_MODELS"], &text(&dynamic, "model")?)?;
            resolution = text(&dynamic, "resolution")?;
            let duration = number(&dynamic, "duration", 7)?;
            if !(4..=15).contains(&duration) {
                return Err(BytedanceError::Input("Seedance2 duration exceeds schema"));
            }
            let mut ratio = text(&dynamic, "ratio")?;
            let mut content = vec![];
            if node == "ByteDance2FirstLastFrameNode" {
                let first_id = optional_text(&arguments, "first_frame_asset_id", "")?;
                let last_id = optional_text(&arguments, "last_frame_asset_id", "")?;
                let first_id = first_id.trim();
                let last_id = last_id.trim();
                let first = arguments
                    .get("first_frame")
                    .is_some_and(|value| !value.is_null());
                let last = arguments
                    .get("last_frame")
                    .is_some_and(|value| !value.is_null());
                if first == !first_id.is_empty() || last && !last_id.is_empty() {
                    return Err(BytedanceError::Input(
                        "Invalid first/last frame and asset selection",
                    ));
                }
                if first_id.is_empty() && last_id.is_empty() {
                    ratio = "adaptive".into();
                }
                let frame_url = |slot: &str, id: &str| -> Result<String, BytedanceError> {
                    if !id.is_empty() {
                        active_id(id)?;
                        resolved
                            .assets
                            .get(id)
                            .cloned()
                            .ok_or(BytedanceError::Input("Missing resolved Image asset"))
                    } else {
                        resolved
                            .media
                            .iter()
                            .find(|media| media.slot == slot)
                            .map(|media| media.url.clone())
                            .ok_or(BytedanceError::Input(
                                "Missing canonical first/last frame upload",
                            ))
                    }
                };
                content.push(image_content(
                    &frame_url("first_frame", first_id)?,
                    Some("first_frame"),
                ));
                if last || !last_id.is_empty() {
                    content.push(image_content(
                        &frame_url("last_frame", last_id)?,
                        Some("last_frame"),
                    ));
                }
            } else if node == "ByteDance2ReferenceNode" {
                let reference_images = dynamic.get("reference_images").and_then(Value::as_object);
                let reference_videos = dynamic.get("reference_videos").and_then(Value::as_object);
                let reference_audios = dynamic.get("reference_audios").and_then(Value::as_object);
                let image_count = reference_images.map_or(0, |map| map.len());
                let video_count = reference_videos.map_or(0, |map| map.len());
                let audio_count = reference_audios.map_or(0, |map| map.len());
                let image_assets = resolved
                    .assets
                    .values()
                    .filter(|value| value.starts_with("Image:"))
                    .count();
                let video_assets = resolved
                    .assets
                    .values()
                    .filter(|value| value.starts_with("Video:"))
                    .count();
                let audio_assets = resolved
                    .assets
                    .values()
                    .filter(|value| value.starts_with("Audio:"))
                    .count();
                if image_count + video_count + image_assets + video_assets == 0 {
                    return Err(BytedanceError::Input(
                        "At least one reference image/video/asset is required",
                    ));
                }
                if image_count + image_assets > 9
                    || video_count + video_assets > 3
                    || audio_count + audio_assets > 3
                {
                    return Err(BytedanceError::Input(
                        "Reference count exceeds source limits",
                    ));
                }
                has_video_input = video_count + video_assets > 0;
                for map in [reference_videos, reference_audios].into_iter().flatten() {
                    let mut duration = 0.0;
                    for media in map.values() {
                        let seconds = media.get("duration").and_then(Value::as_f64).ok_or(
                            BytedanceError::Input("Missing canonical reference duration"),
                        )?;
                        if !seconds.is_finite() || seconds < 1.8 {
                            return Err(BytedanceError::Input(
                                "Reference media below minimum duration",
                            ));
                        }
                        duration += seconds;
                    }
                    if duration > 15.1 {
                        return Err(BytedanceError::Input(
                            "Reference media exceeds total duration",
                        ));
                    }
                }
                let mut labels = BTreeMap::new();
                let mut counts = [image_count, video_count, audio_count];
                if let Some(assets) = dynamic.get("reference_assets").and_then(Value::as_object) {
                    for (slot, id) in assets {
                        if let Some(id) = id.as_str() {
                            let id = id.trim();
                            if let Some(uri) = resolved.assets.get(id) {
                                let (kind, position) = if uri.starts_with("Image:") {
                                    ("Image", 0)
                                } else if uri.starts_with("Video:") {
                                    ("Video", 1)
                                } else if uri.starts_with("Audio:") {
                                    ("Audio", 2)
                                } else {
                                    return Err(BytedanceError::Input("Unknown asset type"));
                                };
                                if let Some(number) = slot
                                    .rsplit('_')
                                    .next()
                                    .and_then(|value| value.parse::<u32>().ok())
                                {
                                    counts[position] += 1;
                                    labels.insert(number, format!("{kind} {}", counts[position]));
                                }
                            }
                        }
                    }
                }
                instruction = rewrite_asset_refs(&instruction, &labels);
                for kind in ["image", "video", "audio"] {
                    for media in resolved.media.iter().filter(|media| media.kind == kind) {
                        content.push(match kind {
                            "image" => image_content(&media.url, Some("reference_image")),
                            "video" => video_content(&media.url),
                            _ => audio_content(&media.url),
                        });
                    }
                }
                for kind in ["Image", "Video", "Audio"] {
                    for uri in resolved
                        .assets
                        .values()
                        .filter(|uri| uri.starts_with(&format!("{kind}:")))
                    {
                        let url = uri
                            .split_once(':')
                            .map(|(_, url)| url)
                            .ok_or(BytedanceError::Input("Malformed asset URI"))?;
                        content.push(match kind {
                            "Image" => image_content(url, Some("reference_image")),
                            "Video" => video_content(url),
                            _ => audio_content(url),
                        });
                    }
                }
            }
            content.insert(0, json!({"type":"text","text":instruction}));
            body = json!({"model":model,"content":content,"generate_audio":boolean(&dynamic,"generate_audio",true)?,"resolution":resolution,"ratio":ratio,"duration":duration,"seed":seed,"watermark":watermark});
            poll = Some(PollPlan {
                prefix: VIDEO2_PATH.into(),
                interval_milliseconds: 9000,
                maximum_nonqueued_polls: 480,
                completed_status: "succeeded".into(),
                failed_status: "failed".into(),
                estimated_seconds: None,
            });
            VIDEO_PATH
        }
        "ByteDanceSeedNode" => {
            let instruction = text(&arguments, "prompt")?;
            prompt(&instruction, 65536)?;
            model = model_id(&constants["SEED_MODELS"], &text(&dynamic, "model")?)?;
            if media_count(&dynamic["images"]) > 20
                || dynamic
                    .get("videos")
                    .and_then(Value::as_object)
                    .map_or(0, |map| map.len())
                    > 4
            {
                return Err(BytedanceError::Input("Too many Seed context media"));
            }
            let temperature = dynamic
                .get("temperature")
                .and_then(Value::as_f64)
                .ok_or(BytedanceError::Input("Missing Seed temperature"))?;
            if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
                return Err(BytedanceError::Input("Invalid Seed temperature"));
            }
            let mut content = vec![];
            for media in resolved.media.iter().filter(|media| media.kind == "image") {
                content.push(json!({"type":"input_image","image_url":media.url,"detail":"auto"}));
            }
            for media in resolved.media.iter().filter(|media| media.kind == "video") {
                content.push(json!({"type":"input_video","video_url":media.url}));
            }
            content.push(json!({"type":"input_text","text":instruction}));
            body = json!({"model":model,"input":[{"type":"message","role":"user","content":content}],"temperature":temperature,"store":false,"stream":false});
            let system = optional_text(&arguments, "system_prompt", "")?;
            if !system.is_empty() {
                body["instructions"] = system.into();
            }
            output = OutputKind::Text;
            TEXT_PATH
        }
        "ByteDanceSeedAudioNode" => {
            let instruction = text(&arguments, "text_prompt")?;
            let instruction_length = instruction
                .trim_matches(|character: char| {
                    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
                })
                .chars()
                .count();
            if instruction_length == 0 || instruction_length > 3000 {
                return Err(BytedanceError::Input("Seed Audio text length invalid"));
            }
            let controls = arguments
                .get("reference_mode")
                .ok_or(BytedanceError::Input("Missing Seed Audio reference mode"))?;
            let mode = text(controls, "reference_mode")?;
            let maximum_tag = max_audio_tag(&instruction);
            let references = match mode.as_str() {
                "text only" => {
                    if maximum_tag > 0 {
                        return Err(BytedanceError::Input("Audio tag requires references"));
                    }
                    None
                }
                "preset voice" => {
                    if maximum_tag > 1 {
                        return Err(BytedanceError::Input("Audio tag exceeds preset voice"));
                    }
                    let voice = text(controls, "preset_voice")?;
                    let speaker = constants["SEED_AUDIO_VOICE_MAP"][&voice]
                        .as_str()
                        .ok_or(BytedanceError::Input("Unknown Seed Audio voice"))?;
                    Some(json!([{"speaker":speaker}]))
                }
                "image reference" => {
                    if maximum_tag > 0 || controls.get("reference_image").is_none_or(Value::is_null)
                    {
                        return Err(BytedanceError::Input("Missing Seed Audio reference image"));
                    }
                    Some(
                        json!([{"image_data":resolved.inline_media.get("reference_image").ok_or(BytedanceError::Input("Missing canonical encoded image"))?}]),
                    )
                }
                "audio reference" => {
                    let mut values = vec![];
                    let mut found_gap = false;
                    for number in 1..=3 {
                        let slot = format!("reference_audio_{number}");
                        if controls.get(&slot).is_some_and(|value| !value.is_null()) {
                            if found_gap {
                                return Err(BytedanceError::Input(
                                    "Seed Audio reference sockets must be consecutive",
                                ));
                            }
                            values.push(json!({"audio_data":resolved.inline_media.get(&slot).ok_or(BytedanceError::Input("Missing canonical encoded audio"))?}));
                        } else {
                            found_gap = true;
                        }
                    }
                    if values.is_empty() || maximum_tag > values.len() as u64 {
                        return Err(BytedanceError::Input("Seed Audio reference tags invalid"));
                    }
                    Some(values.into())
                }
                _ => return Err(BytedanceError::Input("Unknown Seed Audio mode")),
            };
            let sample_rate = text(&arguments, "sample_rate")?
                .parse::<u64>()
                .map_err(|_| BytedanceError::Input("Invalid Seed Audio sample rate"))?;
            if ![8000, 16000, 24000, 32000, 44100, 48000].contains(&sample_rate) {
                return Err(BytedanceError::Input("Unsupported Seed Audio sample rate"));
            }
            for (field, minimum, maximum) in [
                ("speech_rate", -50, 100),
                ("loudness_rate", -50, 100),
                ("pitch_rate", -12, 12),
            ] {
                let value = arguments
                    .get(field)
                    .and_then(Value::as_i64)
                    .ok_or(BytedanceError::Input("Missing signed Seed Audio control"))?;
                if !(minimum..=maximum).contains(&value) {
                    return Err(BytedanceError::Input(
                        "Seed Audio control exceeds source schema",
                    ));
                }
            }
            body = json!({"model":"seed-audio-1.0","text_prompt":instruction,"audio_config":{"format":"mp3","sample_rate":sample_rate,"speech_rate":arguments["speech_rate"],"loudness_rate":arguments["loudness_rate"],"pitch_rate":arguments["pitch_rate"]},"watermark":{}});
            if let Some(references) = references {
                body["references"] = references;
            }
            output = OutputKind::Audio;
            AUDIO_PATH
        }
        "ByteDanceCreateImageAsset" | "ByteDanceCreateVideoAsset" => {
            let group = optional_text(&arguments, "group_id", "")?;
            let group = if group.trim().is_empty() {
                resolved.group_id.as_deref().ok_or(BytedanceError::Input(
                    "Canonical visual authentication group is required",
                ))?
            } else {
                group.trim()
            };
            active_id(group)?;
            let kind = if node == "ByteDanceCreateImageAsset" {
                "Image"
            } else {
                "Video"
            };
            let media = resolved
                .media
                .first()
                .ok_or(BytedanceError::Input("Missing canonical asset upload"))?;
            body = create_asset_body(group, &media.url, kind)?;
            output = OutputKind::Asset;
            poll = Some(asset_poll_plan());
            "/proxy/seedance/assets"
        }
        _ => return Err(BytedanceError::Input("Unsupported Bytedance node")),
    };
    let bytes = serde_json::to_vec(&body)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(BytedanceError::Input("Bytedance request exceeds its bound"));
    }
    Ok(Invocation {
        node: node.into(),
        head: request_head(path, HttpMethod::Post, credentials, false)?,
        body: bytes,
        poll,
        output,
        fail_on_partial,
        model_id: model,
        has_video_input,
        resolution,
    })
}
pub fn create_asset_body(group_id: &str, url: &str, kind: &str) -> Result<Value, BytedanceError> {
    active_id(group_id)?;
    if endpoint(url)? || !matches!(kind, "Image" | "Video") {
        return Err(BytedanceError::Input("Invalid Seedance asset creation"));
    }
    Ok(json!({"group_id":group_id,"url":url,"asset_type":kind}))
}
pub fn virtual_asset_body(
    url: &str,
    canonical_hash: &str,
    kind: &str,
) -> Result<Value, BytedanceError> {
    if endpoint(url)?
        || canonical_hash.len() != 64
        || !canonical_hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(BytedanceError::Input(
            "Canonical Seedance media hash is required",
        ));
    }
    let mut body = json!({"url":url,"hash":canonical_hash});
    match kind {
        "Image" => {}
        "Video" => body["asset_type"] = "Video".into(),
        _ => return Err(BytedanceError::Input("Unsupported virtual asset type")),
    };
    Ok(body)
}
pub fn asset_poll_plan() -> PollPlan {
    PollPlan {
        prefix: "/proxy/seedance/assets".into(),
        interval_milliseconds: 5000,
        maximum_nonqueued_polls: 1200,
        completed_status: "active".into(),
        failed_status: "failed".into(),
        estimated_seconds: None,
    }
}
pub fn visual_session_poll_plan() -> PollPlan {
    PollPlan {
        prefix: "/proxy/seedance/visual-validate/sessions".into(),
        interval_milliseconds: 3000,
        maximum_nonqueued_polls: 39,
        completed_status: "completed".into(),
        failed_status: "failed".into(),
        estimated_seconds: Some(119),
    }
}
pub fn active_asset_uri(
    bytes: &[u8],
    expected_kind: Option<&str>,
) -> Result<String, BytedanceError> {
    let response = bounded_json(bytes)?;
    let id = text(&response, "id")?;
    active_id(&id)?;
    let status = text(&response, "status")?;
    let kind = text(&response, "asset_type")?;
    text(&response, "group_id")?;
    if status != "Active" || expected_kind.is_some_and(|expected| kind != expected) {
        return Err(BytedanceError::Provider(format!(
            "Asset {id} is not an Active {} asset",
            expected_kind.unwrap_or("supported")
        )));
    }
    if !matches!(kind.as_str(), "Image" | "Video" | "Audio") {
        return Err(BytedanceError::Response("Unknown asset type"));
    }
    Ok(format!("{kind}:asset://{id}"))
}
fn max_audio_tag(value: &str) -> u64 {
    let bytes = value.as_bytes();
    let mut maximum = 0;
    for (offset, _) in value.char_indices() {
        if bytes
            .get(offset..offset.saturating_add(6))
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"@Audio"))
        {
            let mut digits = 0u64;
            for byte in bytes.get(offset + 6..).unwrap_or_default() {
                if !byte.is_ascii_digit() {
                    break;
                }
                digits = digits
                    .saturating_mul(10)
                    .saturating_add(u64::from(byte - b'0'));
            }
            maximum = maximum.max(digits);
        }
    }
    maximum
}
fn rewrite_asset_refs(value: &str, labels: &BTreeMap<u32, String>) -> String {
    let characters = value.chars().collect::<Vec<_>>();
    let mut output = String::new();
    let mut offset = 0;
    while let Some(character) = characters.get(offset) {
        let previous = offset
            .checked_sub(1)
            .and_then(|index| characters.get(index));
        let boundary =
            previous.is_none_or(|character| !character.is_alphanumeric() && *character != '_');
        let prefix = characters
            .get(offset..offset + 5)
            .map(|part| part.iter().collect::<String>());
        if boundary && prefix.is_some_and(|part| part.eq_ignore_ascii_case("asset")) {
            let mut end = offset + 5;
            if characters.get(end) == Some(&' ') {
                end += 1;
            }
            let mut number = 0u32;
            let mut digits = 0;
            while digits < 2 {
                if let Some(digit) = characters
                    .get(end)
                    .and_then(|character| character.to_digit(10))
                {
                    number = number * 10 + digit;
                    digits += 1;
                    end += 1;
                } else {
                    break;
                }
            }
            if digits > 0
                && characters
                    .get(end)
                    .is_none_or(|character| !character.is_alphanumeric() && *character != '_')
                && let Some(label) = labels.get(&number)
            {
                output.push_str(label);
                offset = end;
                continue;
            }
        }
        output.push(*character);
        offset += 1;
    }
    output
}
fn bounded_json(bytes: &[u8]) -> Result<Value, BytedanceError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(BytedanceError::Response("Bytedance JSON exceeds bound"));
    }
    Ok(serde_json::from_slice(bytes)?)
}
#[derive(Clone, Debug)]
pub struct DownloadedMedia {
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub url: String,
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub node: String,
    pub kind: OutputKind,
    pub media: Vec<DownloadedMedia>,
    pub strings: Vec<String>,
    pub audio_base64: Option<String>,
    pub raw_response: Vec<u8>,
    pub receipt: Vec<u8>,
    pub source_cost_usd: Option<f64>,
}
fn source_token_count(usage: &Value, key: &str) -> Result<u64, BytedanceError> {
    match usage.get(key) {
        None | Some(Value::Null) => Ok(0),
        Some(value) => value
            .as_u64()
            .ok_or(BytedanceError::Response("Invalid provider token count")),
    }
}
fn source_price(invocation: &Invocation, response: &Value) -> Result<Option<f64>, BytedanceError> {
    if invocation.output == OutputKind::Text {
        let Some(usage) = response.get("usage").filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        let (input_rate, cache_rate, output_rate) = match invocation.model_id.as_str() {
            "seed-2-0-pro-260328" => (0.50, 0.10, 3.00),
            "seed-2-0-lite-260228" => (0.25, 0.05, 2.00),
            "seed-2-0-mini-260215" => (0.10, 0.02, 0.40),
            _ => return Ok(None),
        };
        let input = source_token_count(usage, "input_tokens")?;
        let output = source_token_count(usage, "output_tokens")?;
        let cache = source_token_count(&usage["input_tokens_details"], "cached_tokens")?;
        return Ok(Some(
            (input.saturating_sub(cache) as f64 * input_rate
                + cache as f64 * cache_rate
                + output as f64 * output_rate)
                / 1_000_000.0,
        ));
    }
    if invocation
        .poll
        .as_ref()
        .is_some_and(|poll| poll.prefix == VIDEO2_PATH)
    {
        let Some(usage) = response.get("usage").filter(|value| !value.is_null()) else {
            return Ok(None);
        };
        let source = oracle()?;
        let rates = source["constants"]["seedance2_prices"]
            .as_array()
            .ok_or(BytedanceError::Response("Missing source price rates"))?;
        let rate = rates
            .iter()
            .find(|rate| {
                rate["model"] == invocation.model_id
                    && rate["has_video"] == invocation.has_video_input
                    && rate["resolution"] == invocation.resolution
            })
            .and_then(|rate| rate["usd_per_1k"].as_f64());
        if let Some(rate) = rate {
            return Ok(Some(
                number(usage, "total_tokens", 0)? as f64 * 1.43 * rate / 1000.0,
            ));
        }
    }
    Ok(None)
}
pub fn text_from_response(bytes: &[u8]) -> Result<String, BytedanceError> {
    let response = bounded_json(bytes)?;
    if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
        return Err(BytedanceError::Provider(format!(
            "Seed API error ({}): {}",
            text(error, "code")?,
            text(error, "message")?
        )));
    }
    let mut chunks = vec![];
    for item in response
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        for block in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match block.get("type").and_then(Value::as_str) {
                Some("output_text") => {
                    let value = text(block, "text")?;
                    if !value.is_empty() {
                        chunks.push(value);
                    }
                }
                Some("refusal") => {
                    let value = text(block, "refusal")?;
                    if !value.is_empty() {
                        return Err(BytedanceError::Provider(format!(
                            "Model refused to respond: {value}"
                        )));
                    }
                }
                _ => {}
            }
        }
    }
    if chunks.is_empty() {
        return Err(BytedanceError::Response("Empty response from Seed model"));
    }
    Ok(chunks.join("\n"))
}
fn media_urls(invocation: &Invocation, response: &Value) -> Result<Vec<String>, BytedanceError> {
    if invocation.output == OutputKind::Image {
        if (invocation.node == "ByteDanceImageNode"
            || response
                .get("data")
                .and_then(Value::as_array)
                .is_some_and(|data| data.len() == 1))
            && let Some(error) = response
                .get("error")
                .filter(|value| value.as_object().is_some_and(|error| !error.is_empty()))
        {
            return Err(BytedanceError::Provider(format!(
                "ByteDance request failed. Code: {}, message: {}",
                error["code"], error["message"]
            )));
        }
        let data = response
            .get("data")
            .and_then(Value::as_array)
            .ok_or(BytedanceError::Response("Missing Bytedance image data"))?;
        let urls = data
            .iter()
            .filter_map(|item| item.get("url").and_then(Value::as_str).map(str::to_owned))
            .collect::<Vec<_>>();
        if urls.is_empty() || invocation.fail_on_partial && urls.len() != data.len() {
            return Err(BytedanceError::Response(
                "Incomplete Bytedance image response",
            ));
        }
        if invocation.node == "ByteDanceImageNode" {
            return Ok(urls.into_iter().take(1).collect());
        }
        return Ok(urls);
    }
    Ok(vec![text(&response["content"], "video_url")?])
}
pub fn invoke_resolved(
    input: &Input,
    resolved: &Resolution,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn BytedanceTransport,
) -> Result<ResultValue, BytedanceError> {
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let invocation = build_invocation(input, resolved, credentials)?;
    let (mut bytes, mut receipt, _) = exchange_with_policy(
        &invocation.head,
        &invocation.body,
        Some(maximum_cost_microunits),
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        transport,
    )?;
    let mut response = bounded_json(&bytes)?;
    if let Some(plan) = &invocation.poll {
        let identifier = if invocation.output == OutputKind::Asset {
            text(&response, "asset_id")?
        } else {
            text(&response, "id")?
        };
        active_id(&identifier)?;
        let path = format!("{}/{identifier}", plan.prefix);
        let mut consumed = 0u32;
        let mut completed = false;
        for _ in 0..4096 {
            let head = request_head(&path, HttpMethod::Get, credentials, false)?;
            let (next, proof, handle) = exchange_with_policy(
                &head,
                &[],
                None,
                MAXIMUM_RESPONSE_BYTES,
                POLL_RETRY,
                transport,
            )?;
            response = bounded_json(&next)?;
            let status = text(&response, "status")?.trim().to_lowercase();
            if status == plan.failed_status
                || matches!(
                    status.as_str(),
                    "cancelled" | "canceled" | "canceling" | "fail" | "error"
                )
            {
                return Err(BytedanceError::Provider(format!(
                    "Bytedance task failed: {status}"
                )));
            }
            if status == plan.completed_status {
                bytes = next;
                receipt = proof;
                completed = true;
                break;
            }
            transport.check_cancelled(Some(handle))?;
            transport.delay(handle, plan.interval_milliseconds)?;
            if !matches!(
                status.as_str(),
                "created"
                    | "queued"
                    | "queueing"
                    | "submitted"
                    | "initializing"
                    | "wait"
                    | "in_queue"
            ) {
                consumed += 1;
                if consumed >= plan.maximum_nonqueued_polls {
                    break;
                }
            }
        }
        if !completed {
            return Err(ConformanceError::PollLimit.into());
        }
        if invocation.output == OutputKind::Asset {
            let group = optional_text(&Value::Object(input.arguments.clone()), "group_id", "")?;
            let group = if group.trim().is_empty() {
                resolved
                    .group_id
                    .clone()
                    .ok_or(BytedanceError::Input("Missing resolved asset group"))?
            } else {
                group.trim().into()
            };
            return Ok(ResultValue {
                node: invocation.node,
                kind: OutputKind::Asset,
                media: vec![],
                strings: vec![identifier, group],
                audio_base64: None,
                raw_response: bytes,
                receipt,
                source_cost_usd: None,
            });
        }
    }
    let source_cost_usd = source_price(&invocation, &response)?;
    let mut result = ResultValue {
        node: invocation.node.clone(),
        kind: invocation.output,
        media: vec![],
        strings: vec![],
        audio_base64: None,
        raw_response: bytes.clone(),
        receipt,
        source_cost_usd,
    };
    match invocation.output {
        OutputKind::Text => result.strings.push(text_from_response(&bytes)?),
        OutputKind::Audio => {
            let audio = response
                .get("audio")
                .and_then(Value::as_str)
                .filter(|audio| !audio.is_empty())
                .ok_or(BytedanceError::Response("Seed Audio returned no audio"))?;
            result.audio_base64 = Some(audio.into());
        }
        OutputKind::Image | OutputKind::Video => {
            for url in media_urls(&invocation, &response)? {
                if endpoint(&url)? {
                    return Err(BytedanceError::Response(
                        "Bytedance download URL must be absolute",
                    ));
                }
                let head = request_head(&url, HttpMethod::Get, credentials, true)?;
                let (bytes, receipt, _) = exchange_with_policy(
                    &head,
                    &[],
                    None,
                    MAXIMUM_MEDIA_BYTES,
                    DOWNLOAD_RETRY,
                    transport,
                )?;
                if bytes.is_empty() {
                    return Err(BytedanceError::Response("Empty Bytedance media download"));
                }
                let aggregate = result
                    .media
                    .iter()
                    .map(|media| media.bytes.len())
                    .sum::<usize>()
                    .checked_add(bytes.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                if aggregate > MAXIMUM_MEDIA_BYTES {
                    return Err(ConformanceError::BodyLimit.into());
                }
                result.media.push(DownloadedMedia {
                    bytes,
                    receipt,
                    url,
                });
            }
        }
        OutputKind::Asset => return Err(BytedanceError::Response("Missing asset poll")),
    };
    Ok(result)
}
pub trait BytedanceMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub fn invoke_and_materialize(
    input: &Input,
    resolved: &Resolution,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn BytedanceTransport,
    materializer: &mut dyn BytedanceMaterializer,
    sink: &mut dyn MaterializationSink,
) -> Result<(), BytedanceError> {
    let result = invoke_resolved(
        input,
        resolved,
        credentials,
        maximum_cost_microunits,
        transport,
    )?;
    transport.check_cancelled(None)?;
    let outputs = materializer.materialize(&result)?;
    let registry = TypeRegistry::built_in()
        .map_err(|_| BytedanceError::Response("Missing canonical type registry"))?;
    let aliases = match result.kind {
        OutputKind::Image => vec!["IMAGE"],
        OutputKind::Video => vec!["VIDEO"],
        OutputKind::Audio => vec!["AUDIO"],
        OutputKind::Text => vec!["STRING"],
        OutputKind::Asset => vec!["STRING", "STRING"],
    };
    let expected = aliases
        .into_iter()
        .enumerate()
        .map(|(index, alias)| {
            let type_id = registry
                .resolve(alias)
                .map_err(|_| BytedanceError::Response("Missing canonical output type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| BytedanceError::Response("Missing canonical output family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect::<Result<Vec<_>, BytedanceError>>()?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn fixture_resolution(case: &Value) -> Result<Resolution, BytedanceError> {
    let mut resolution = Resolution {
        media: serde_json::from_value(case["uploads"].clone())?,
        ..Resolution::default()
    };
    for request in case["requests"].as_array().into_iter().flatten() {
        if request["method"] == "GET"
            && let Some(path) = request["path"].as_str()
            && let Some(id) = path.strip_prefix("/proxy/seedance/assets/")
        {
            let uri = if case["input"]["node"] == "ByteDance2ReferenceNode" {
                format!("Image:asset://{id}")
            } else {
                format!("asset://{id}")
            };
            resolution.assets.insert(id.into(), uri);
        }
    }
    resolution.inline_media.insert(
        "reference_image".into(),
        "canonical-image-base64-boundary".into(),
    );
    Ok(resolution)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    let source = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"source_sha256":source["source_sha256"],"scope":"source requests and bounded transport; canonical media preparation, asset/auth UI orchestration and compiled guest integration pending"})
        || source["source_sha256"]["comfy_api_nodes/nodes_bytedance.py"] != SOURCE_SHA256
        || source["source_sha256"]["comfy_api_nodes/nodes_bytedance_llm.py"] != LLM_SOURCE_SHA256
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let credentials = CredentialAvailability {
        auth_token: true,
        api_key: true,
    };
    let mut covered = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let resolved = fixture_resolution(case).map_err(|_| ConformanceError::InvalidRequest)?;
        let invocation = build_invocation(&input, &resolved, credentials)
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let expected = case["requests"]
            .as_array()
            .and_then(|requests| {
                requests
                    .iter()
                    .rev()
                    .find(|request| request["method"] == "POST")
            })
            .ok_or(ConformanceError::InvalidRequest)?;
        let body: Value = serde_json::from_slice(&invocation.body)
            .map_err(|_| ConformanceError::InvalidRequest)?;
        if expected["path"] != invocation.head.path || expected["body"] != body {
            return Err(ConformanceError::InvalidRequest);
        }
        if let Some(poll) = &invocation.poll {
            let expected = case["polls"]
                .as_array()
                .and_then(|polls| polls.last())
                .ok_or(ConformanceError::InvalidRequest)?;
            let id = if invocation.output == OutputKind::Asset {
                "asset-42"
            } else {
                "task-42"
            };
            if expected["path"] != format!("{}/{id}", poll.prefix)
                || expected["interval"].as_u64().map(|seconds| seconds * 1000)
                    != Some(poll.interval_milliseconds)
                || expected["maximum_attempts"] != poll.maximum_nonqueued_polls
                || expected["estimated_duration"].as_u64() != poll.estimated_seconds
            {
                return Err(ConformanceError::InvalidRequest);
            }
        }
        covered.insert(input.node);
    }
    if covered
        != NODES
            .iter()
            .map(|claim| claim.source_class.to_owned())
            .collect()
    {
        return Err(ConformanceError::InvalidRequest);
    }
    for case in source["preflight_errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        if build_invocation(&input, &Resolution::default(), credentials).is_ok()
            || case["requests"] != json!([])
            || case["uploads"] != json!([])
        {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{CostResponse, Progress, ScriptStep, ScriptedActuator};
    struct TestTransport {
        actuator: ScriptedActuator,
        head: Option<RequestHead>,
        delays: Vec<u64>,
    }
    impl TestTransport {
        fn new(steps: Vec<ScriptStep>) -> Self {
            Self {
                actuator: ScriptedActuator::new(steps),
                head: None,
                delays: vec![],
            }
        }
        fn finish(self) -> Result<(), ConformanceError> {
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

    impl BytedanceTransport for TestTransport {
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
    }
    fn handle(slot: u32, generation: u32) -> StreamHandle {
        StreamHandle {
            invocation: 42,
            slot,
            generation,
        }
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
    fn fixture_case(node: &str) -> Result<Value, BytedanceError> {
        oracle()?["cases"]
            .as_array()
            .and_then(|cases| cases.iter().find(|case| case["input"]["node"] == node))
            .cloned()
            .ok_or(BytedanceError::Input("Missing source oracle case"))
    }
    fn fixture_input(node: &str) -> Result<(Input, Resolution), BytedanceError> {
        let case = fixture_case(node)?;
        Ok((
            serde_json::from_value(case["input"].clone())?,
            fixture_resolution(&case)?,
        ))
    }
    fn cost_steps(invocation: &Invocation, response: Vec<u8>) -> Vec<ScriptStep> {
        let mut steps = exchange_steps(
            invocation.head.clone(),
            invocation.body.clone(),
            response,
            handle(0, 1),
        );
        steps.insert(
            2,
            ScriptStep::Cost(
                handle(0, 1),
                CostRequest {
                    operation: "bytedance.generation.maximum".into(),
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
        steps
    }
    #[test]
    fn bytedance_all_fourteen_nodes_and_eleven_executable_routes_match_source()
    -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bytedance/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 14);
        assert_eq!(ROUTES.len(), 14);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            11
        );
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "synthetic_prefix_tombstone")
                .count(),
            3
        );
        assert_eq!(ALIASES, &["byteplus", "byteplus-seedance2", "seedance"]);
        Ok(())
    }
    #[test]
    fn bytedance_flags_seed_schema_and_nonforwarded_seed_match_source() -> Result<(), Box<dyn Error>>
    {
        let (mut input, resolved) = fixture_input("ByteDanceTextToVideoNode")?;
        input
            .arguments
            .insert("prompt".into(), "avoid --duration 4".into());
        assert!(build_invocation(&input, &resolved, credentials()).is_err());
        let (mut input, resolved) = fixture_input("ByteDanceSeedNode")?;
        let first = build_invocation(&input, &resolved, credentials())?.body;
        input.arguments.insert("seed".into(), 43.into());
        assert_eq!(
            build_invocation(&input, &resolved, credentials())?.body,
            first
        );
        input.arguments.insert("seed".into(), 2147483648u64.into());
        assert!(build_invocation(&input, &resolved, credentials()).is_err());
        Ok(())
    }
    #[test]
    fn bytedance_real_png_download_retains_receipts_and_uses_no_download_secret()
    -> Result<(), Box<dyn Error>> {
        let (input, resolved) = fixture_input("ByteDanceImageNode")?;
        let invocation = build_invocation(&input, &resolved, credentials())?;
        let response = serde_json::to_vec(
            &json!({"model":"seedream-3-0-t2i-250415","created":0,"data":[{"url":"https://download.fixture.invalid/image.png"}],"error":{}}),
        )?;
        let mut steps = cost_steps(&invocation, response);
        let png = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
        ));
        let head = request_head(
            "https://download.fixture.invalid/image.png",
            HttpMethod::Get,
            credentials(),
            true,
        )?;
        assert!(head.secret_id.is_none());
        steps.extend(exchange_steps(head, vec![], png.to_vec(), handle(1, 1)));
        let mut transport = TestTransport::new(steps);
        let result = invoke_resolved(&input, &resolved, credentials(), 1_000_000, &mut transport)?;
        assert_eq!(result.media.len(), 1);
        let media = result.media.first().ok_or("missing media")?;
        assert_eq!(media.bytes, png);
        assert_eq!(media.receipt, b"download-receipt");
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bytedance_cost_rejection_and_missing_auth_block_request_body() -> Result<(), Box<dyn Error>>
    {
        let (input, resolved) = fixture_input("ByteDanceImageNode")?;
        let missing = CredentialAvailability {
            auth_token: false,
            api_key: false,
        };
        let mut transport = TestTransport::new(vec![]);
        assert!(matches!(
            invoke_resolved(&input, &resolved, missing, 1_000_000, &mut transport),
            Err(BytedanceError::Input(_))
        ));
        transport.finish()?;
        let invocation = build_invocation(&input, &resolved, credentials())?;
        let mut steps = cost_steps(&invocation, vec![]);
        if let Some(ScriptStep::Cost(_, _, approval)) = steps.get_mut(2) {
            approval.accepted = false;
        } else {
            return Err("missing cost step".into());
        }
        steps.truncate(3);
        let mut transport = TestTransport::new(steps);
        assert!(matches!(
            invoke_resolved(&input, &resolved, credentials(), 1_000_000, &mut transport),
            Err(BytedanceError::Transport(ConformanceError::InvalidCost))
        ));
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bytedance_text_preserves_all_message_blocks_and_refusal_failure()
    -> Result<(), Box<dyn Error>> {
        let bytes = serde_json::to_vec(
            &json!({"output":[{"type":"reasoning","content":[{"type":"output_text","text":"ignore"}]},{"type":"message","content":[{"type":"output_text","text":"one"},{"type":"output_text","text":"two"}]}]}),
        )?;
        assert_eq!(text_from_response(&bytes)?, "one\ntwo");
        assert!(text_from_response(br#"{"output":[{"type":"message","content":[{"type":"refusal","refusal":"source refusal"}]}]}"#).is_err());
        assert!(text_from_response(br#"{"output":[]}"#).is_err());
        assert!(text_from_response(br#"{"error":{"code":"denied","message":"No"}}"#).is_err());
        Ok(())
    }
    #[test]
    fn bytedance_seed_price_uses_cached_and_video_input_rates() -> Result<(), Box<dyn Error>> {
        let (input, resolved) = fixture_input("ByteDanceSeedNode")?;
        let invocation = build_invocation(&input, &resolved, credentials())?;
        assert_eq!(
            source_price(
                &invocation,
                &json!({"usage":{"input_tokens":1000,"output_tokens":100,"input_tokens_details":{"cached_tokens":200}}})
            )?,
            Some(0.00072)
        );
        assert_eq!(
            source_price(
                &invocation,
                &json!({"usage":{"input_tokens":null,"output_tokens":null,"input_tokens_details":{"cached_tokens":null}}})
            )?,
            Some(0.0)
        );
        let (input, resolved) = fixture_input("ByteDance2TextToVideoNode")?;
        let mut invocation = build_invocation(&input, &resolved, credentials())?;
        let response = json!({"usage":{"total_tokens":1000}});
        let without = source_price(&invocation, &response)?.ok_or("missing price")?;
        invocation.has_video_input = true;
        let with = source_price(&invocation, &response)?.ok_or("missing price")?;
        assert!((without - 0.01001).abs() < 1e-12);
        assert!((with - 0.006149).abs() < 1e-12);
        assert_eq!(source_price(&invocation, &json!({}))?, None);
        Ok(())
    }
    #[test]
    fn bytedance_asset_and_visual_auth_helpers_keep_exact_source_fields()
    -> Result<(), Box<dyn Error>> {
        assert_eq!(
            create_asset_body("group-42", "https://fixture.invalid/image.png", "Image")?,
            json!({"group_id":"group-42","url":"https://fixture.invalid/image.png","asset_type":"Image"})
        );
        let hash = "a".repeat(64);
        assert_eq!(
            virtual_asset_body("https://fixture.invalid/image.png", &hash, "Image")?,
            json!({"url":"https://fixture.invalid/image.png","hash":hash})
        );
        assert!(
            virtual_asset_body(
                "https://fixture.invalid/image.png",
                "invented-hash",
                "Image"
            )
            .is_err()
        );
        let visual = visual_session_poll_plan();
        assert_eq!(visual.interval_milliseconds, 3000);
        assert_eq!(visual.maximum_nonqueued_polls, 39);
        assert_eq!(visual.estimated_seconds, Some(119));
        assert_eq!(asset_poll_plan().maximum_nonqueued_polls, 1200);
        assert_eq!(active_asset_uri(br#"{"id":"asset-42","status":"Active","asset_type":"Image","group_id":"group-42"}"#,Some("Image"))?,"Image:asset://asset-42");
        assert!(active_asset_uri(br#"{"id":"asset-42","status":"Failed","asset_type":"Image","group_id":"group-42"}"#,None).is_err());
        assert!(active_asset_uri(br#"{"id":"asset-42","status":"Active","asset_type":"Video","group_id":"group-42"}"#,Some("Image")).is_err());
        Ok(())
    }
    #[test]
    fn bytedance_asset_reference_rewriting_and_audio_tags_follow_source()
    -> Result<(), Box<dyn Error>> {
        let labels = BTreeMap::from([(1, "Image 2".into()), (2, "Video 1".into())]);
        assert_eq!(
            rewrite_asset_refs("ASSET1 asset 2 asset3 myasset1 asset123", &labels),
            "Image 2 Video 1 asset3 myasset1 asset123"
        );
        assert_eq!(max_audio_tag("@Audio1 plus @aUdIo30"), 30);
        let (mut input, resolved) = fixture_input("ByteDanceSeedAudioNode")?;
        input
            .arguments
            .insert("text_prompt".into(), "@Audio1 hello".into());
        assert!(build_invocation(&input, &resolved, credentials()).is_err());
        Ok(())
    }
    #[test]
    fn bytedance_retries_preserve_body_and_require_fresh_generation() -> Result<(), Box<dyn Error>>
    {
        let head = request_head(TEXT_PATH, HttpMethod::Post, credentials(), false)?;
        let body = br#"{"store":false,"stream":false}"#.to_vec();
        let mut steps = exchange_steps(head.clone(), body.clone(), b"busy".to_vec(), handle(0, 1));
        if let Some(ScriptStep::Wait {
            outcome: ResponseEvent::Head { status, headers },
            ..
        }) = steps.get_mut(5)
        {
            *status = 429;
            headers.push(Header {
                name: "Retry-After".into(),
                value: "200".into(),
            });
        } else {
            return Err("missing response head".into());
        }
        let mut retry = exchange_steps(
            head.clone(),
            body.clone(),
            b"success".to_vec(),
            handle(0, 2),
        );
        retry.remove(0);
        steps.extend(retry);
        let mut transport = TestTransport::new(steps);
        assert_eq!(
            exchange_with_policy(
                &head,
                &body,
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                &mut transport
            )?
            .0,
            b"success"
        );
        assert_eq!(transport.delays, vec![150_000]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bytedance_cancelled_worker_and_malformed_frames_never_succeed() -> Result<(), Box<dyn Error>>
    {
        let (input, resolved) = fixture_input("ByteDanceImageNode")?;
        let mut transport = TestTransport::new(vec![ScriptStep::Cancel(None)]);
        assert!(matches!(
            invoke_resolved(&input, &resolved, credentials(), 1_000_000, &mut transport),
            Err(BytedanceError::Transport(ConformanceError::Cancelled))
        ));
        transport.finish()?;
        let invocation = build_invocation(&input, &resolved, credentials())?;
        let mut transport = TestTransport::new(vec![
            ScriptStep::CheckCancelled(None),
            ScriptStep::Start(invocation.head.clone(), Err(ConformanceError::WorkerLost)),
        ]);
        assert!(matches!(
            invoke_resolved(&input, &resolved, credentials(), 1_000_000, &mut transport),
            Err(BytedanceError::Transport(ConformanceError::WorkerLost))
        ));
        transport.finish()?;
        assert!(bounded_json(br#"{}trailing"#).is_err());
        assert!(bounded_json(&vec![b' '; MAXIMUM_RESPONSE_BYTES + 1]).is_err());
        let mut steps = exchange_steps(
            invocation.head.clone(),
            invocation.body.clone(),
            vec![0; MAXIMUM_CHUNK_BYTES + 1],
            handle(0, 1),
        );
        steps.truncate(8);
        let mut transport = TestTransport::new(steps);
        assert!(matches!(
            exchange_with_policy(
                &invocation.head,
                &invocation.body,
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                &mut transport
            ),
            Err(BytedanceError::Transport(ConformanceError::ChunkLimit))
        ));
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bytedance_partial_image_policy_and_unknown_arguments_are_source_bounded()
    -> Result<(), Box<dyn Error>> {
        let (mut input, resolved) = fixture_input("ByteDanceSeedreamNode")?;
        let mut invocation = build_invocation(&input, &resolved, credentials())?;
        let response = json!({"data":[{"url":"https://fixture.invalid/one.png"},{"error":{"code":"partial"}}],"error":{"code":"partial","message":"some images failed"}});
        assert!(media_urls(&invocation, &response).is_err());
        invocation.fail_on_partial = false;
        assert_eq!(
            media_urls(&invocation, &response)?,
            vec!["https://fixture.invalid/one.png"]
        );
        assert!(media_urls(&invocation, &json!({"data":[{"url":"https://fixture.invalid/one.png"}],"error":{"code":"denied","message":"no"}})).is_err());
        input
            .arguments
            .insert("unreviewed_field".into(), true.into());
        assert!(build_invocation(&input, &resolved, credentials()).is_err());
        Ok(())
    }
    #[test]
    fn bytedance_decoder_failure_retains_real_media_and_publishes_nothing()
    -> Result<(), Box<dyn Error>> {
        struct Decoder;
        impl BytedanceMaterializer for Decoder {
            fn materialize(
                &mut self,
                result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                assert!(
                    result
                        .media
                        .first()
                        .is_some_and(|media| media.bytes.starts_with(b"\x89PNG"))
                );
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
        let (input, resolved) = fixture_input("ByteDanceImageNode")?;
        let invocation = build_invocation(&input, &resolved, credentials())?;
        let mut steps = cost_steps(
            &invocation,
            serde_json::to_vec(
                &json!({"model":"seedream-3-0-t2i-250415","created":0,"data":[{"url":"https://download.fixture.invalid/image.png"}],"error":{}}),
            )?,
        );
        let png = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
        ));
        steps.extend(exchange_steps(
            request_head(
                "https://download.fixture.invalid/image.png",
                HttpMethod::Get,
                credentials(),
                true,
            )?,
            vec![],
            png.to_vec(),
            handle(1, 1),
        ));
        steps.push(ScriptStep::CheckCancelled(None));
        let mut transport = TestTransport::new(steps);
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input,
                &resolved,
                credentials(),
                1_000_000,
                &mut transport,
                &mut Decoder,
                &mut sink
            ),
            Err(BytedanceError::Transport(ConformanceError::InvalidResponse))
        ));
        assert_eq!(sink.batches, 0);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bytedance_audio_preflight_trims_length_and_validates_signed_controls()
    -> Result<(), Box<dyn Error>> {
        let (mut input, resolved) = fixture_input("ByteDanceSeedAudioNode")?;
        input.arguments.insert("text_prompt".into(), "   ".into());
        assert!(build_invocation(&input, &resolved, credentials()).is_err());
        input.arguments.insert(
            "text_prompt".into(),
            format!("  {}  ", "x".repeat(3000)).into(),
        );
        input.arguments.insert("speech_rate".into(), (-50).into());
        input.arguments.insert("loudness_rate".into(), 100.into());
        input.arguments.insert("pitch_rate".into(), (-12).into());
        let body: Value =
            serde_json::from_slice(&build_invocation(&input, &resolved, credentials())?.body)?;
        assert_eq!(body["audio_config"]["speech_rate"], -50);
        assert_eq!(body["audio_config"]["pitch_rate"], -12);
        input.arguments.insert("pitch_rate".into(), (-13).into());
        assert!(build_invocation(&input, &resolved, credentials()).is_err());
        input.arguments.insert("pitch_rate".into(), 0.into());
        input
            .arguments
            .insert("sample_rate".into(), "invalid".into());
        assert!(build_invocation(&input, &resolved, credentials()).is_err());
        Ok(())
    }
    #[test]
    fn bytedance_seed_audio_python_strip_rejects_controls_before_transport()
    -> Result<(), Box<dyn Error>> {
        let (mut input, resolved) = fixture_input("ByteDanceSeedAudioNode")?;
        input.arguments.insert(
            "reference_mode".into(),
            json!({"reference_mode":"text only"}),
        );
        for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            input.arguments.insert(
                "text_prompt".into(),
                format!("{separator} {separator}").into(),
            );
            assert!(matches!(
                build_invocation(&input, &resolved, credentials()),
                Err(BytedanceError::Input("Seed Audio text length invalid"))
            ));
            let mut transport = TestTransport::new(vec![]);
            assert!(matches!(
                invoke_resolved(&input, &resolved, credentials(), 1_000_000, &mut transport),
                Err(BytedanceError::Input("Seed Audio text length invalid"))
            ));
            assert!(transport.head.is_none());
            transport.finish()?;
        }
        Ok(())
    }

    #[test]
    fn bytedance_seed_audio_python_strip_retains_raw_text_and_negative_controls()
    -> Result<(), Box<dyn Error>> {
        let (mut input, resolved) = fixture_input("ByteDanceSeedAudioNode")?;
        input.arguments.insert(
            "reference_mode".into(),
            json!({"reference_mode":"text only"}),
        );
        for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            for character in ['x', '☃'] {
                let raw = format!(
                    "{separator} {} {separator}",
                    character.to_string().repeat(3000)
                );
                input
                    .arguments
                    .insert("text_prompt".into(), raw.clone().into());
                let invocation = build_invocation(&input, &resolved, credentials())?;
                let body: Value = serde_json::from_slice(&invocation.body)?;
                assert_eq!(body["text_prompt"], raw);
                input.arguments.insert(
                    "text_prompt".into(),
                    format!(
                        "{separator} {} {separator}",
                        character.to_string().repeat(3001)
                    )
                    .into(),
                );
                assert!(matches!(
                    build_invocation(&input, &resolved, credentials()),
                    Err(BytedanceError::Input("Seed Audio text length invalid"))
                ));
            }
        }
        for raw in ["\u{200b}", "\u{feff}"] {
            input.arguments.insert("text_prompt".into(), raw.into());
            let body: Value =
                serde_json::from_slice(&build_invocation(&input, &resolved, credentials())?.body)?;
            assert_eq!(body["text_prompt"], raw);
        }
        Ok(())
    }

    #[test]
    fn bytedance_seed_audio_control_padded_text_reaches_checked_request_unchanged()
    -> Result<(), Box<dyn Error>> {
        let (mut input, resolved) = fixture_input("ByteDanceSeedAudioNode")?;
        input.arguments.insert(
            "reference_mode".into(),
            json!({"reference_mode":"text only"}),
        );
        for raw in [
            format!("\u{1c} {} \u{1f}", "x".repeat(3000)),
            "\u{200b}".into(),
            "\u{feff}".into(),
        ] {
            input
                .arguments
                .insert("text_prompt".into(), raw.clone().into());
            let invocation = build_invocation(&input, &resolved, credentials())?;
            let body: Value = serde_json::from_slice(&invocation.body)?;
            assert_eq!(body["text_prompt"], raw);
            let response = serde_json::to_vec(
                &json!({"audio":null,"code":99,"message":"captured no media response"}),
            )?;
            let mut transport = TestTransport::new(cost_steps(&invocation, response));
            assert!(matches!(
                invoke_resolved(&input, &resolved, credentials(), 1_000_000, &mut transport),
                Err(BytedanceError::Response("Seed Audio returned no audio"))
            ));
            assert_eq!(transport.head.as_ref(), Some(&invocation.head));
            transport.finish()?;
        }
        Ok(())
    }
}
