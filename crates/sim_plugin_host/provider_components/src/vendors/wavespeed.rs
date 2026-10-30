use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, ProviderTransport, RequestHead, ResponseEvent, StreamHandle, TypeRegistry,
    publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.wavespeed";
pub const SOURCE_SHA256: &str = "f7c7fec19d8ddfd1bc885498911de9a3555c2c19aa08c6c982a0a45e2a9cf699";
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
        feature_id: "COMFY-NODE-0786",
        node_identifier: "WavespeedFlashVSRNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0787",
        node_identifier: "WavespeedImageUpscaleNode",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0206",
        method: "GET",
        path: "/proxy/wavespeed/api/v3/predictions/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0207",
        method: "GET",
        path: "/proxy/wavespeed/api/v3/predictions/{initial_res.data.id}/result",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0208",
        method: "POST",
        path: "/proxy/wavespeed/api/v3/wavespeed-ai/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0209",
        method: "POST",
        path: "/proxy/wavespeed/api/v3/wavespeed-ai/flashvsr",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0210",
        method: "POST",
        path: "/proxy/wavespeed/api/v3/wavespeed-ai/{model_path}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum WavespeedError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    SourceIndex(&'static str),
    Http {
        status: u16,
        retry_after: Option<String>,
        rate_limited: bool,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for WavespeedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) | Self::SourceIndex(message) => {
                formatter.write_str(message)
            }
            Self::Provider(status) => write!(formatter, "Wavespeed job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Wavespeed HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for WavespeedError {}
impl From<ConformanceError> for WavespeedError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for WavespeedError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageInput {
    pub batch: u32,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VideoInput {
    pub duration: f64,
    pub container: String,
    #[serde(default)]
    pub duration_error: bool,
    #[serde(default)]
    pub container_error: bool,
    #[serde(default)]
    pub conversion_error: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    pub image: Option<ImageInput>,
    pub video: Option<VideoInput>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageUploadPolicy {
    pub maximum_images: u32,
    pub total_pixels: u64,
    pub content_type: Option<String>,
    pub wait_label: String,
    pub show_batch_index: bool,
}
#[derive(Clone, Debug)]
pub struct PreparedUpload {
    pub urls: Vec<String>,
    pub receipts: Vec<Vec<u8>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestModel {
    FlashVsr,
    Image,
}
impl RequestModel {
    pub fn source_name(self) -> &'static str {
        match self {
            Self::FlashVsr => "FlashVSRRequest",
            Self::Image => "SeedVR2ImageRequest",
        }
    }
}
pub trait WavespeedInputOwner {
    fn verify_inputs(&mut self, _: &Input) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn source_lower(&mut self, _: &Value) -> Result<String, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn normalize_request(&mut self, _: RequestModel, _: &Value) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn validate_video_container(&mut self, _: &Input) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn validate_video_duration(
        &mut self,
        _: &Input,
        _: f64,
        _: f64,
        _: f64,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn video_duration(&mut self, _: &Input) -> Result<f64, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn video_upload_mp4_h264(
        &mut self,
        _: &Input,
        _: &str,
    ) -> Result<PreparedUpload, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn image_upload(
        &mut self,
        _: &Input,
        _: &ImageUploadPolicy,
    ) -> Result<PreparedUpload, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub node: String,
    pub path: &'static str,
    pub body: Value,
    pub input_receipts: Vec<Vec<u8>>,
}
fn first_upload(
    prepared: PreparedUpload,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<String, WavespeedError> {
    if prepared.urls.len() != 1 || prepared.receipts.len() != 1 {
        return Err(ConformanceError::InvalidResponse.into());
    }
    let url = prepared
        .urls
        .into_iter()
        .next()
        .ok_or(ConformanceError::InvalidResponse)?;
    if endpoint(&url)? {
        return Err(WavespeedError::Input("Source upload requires absolute URL"));
    }
    for receipt in prepared.receipts {
        bounded_receipt(&receipt)?;
        receipts.push(receipt);
    }
    Ok(url)
}
pub fn prepare_request(
    input: &Input,
    owner: &mut dyn WavespeedInputOwner,
) -> Result<RequestPlan, WavespeedError> {
    owner.verify_inputs(input)?;
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(WavespeedError::Input("Unsupported source node"));
    }
    let resolution = input
        .settings
        .get("target_resolution")
        .ok_or(WavespeedError::Input("Missing source target resolution"))?;
    let mut receipts = vec![];
    let (path, model, raw) = match input.node.as_str() {
        "WavespeedFlashVSRNode" => {
            owner.validate_video_container(input)?;
            owner.validate_video_duration(input, 5.0, 600.0, 0.0001)?;
            let resolution = owner.source_lower(resolution)?;
            let uploaded = owner.video_upload_mp4_h264(input, "Uploading")?;
            let video = first_upload(uploaded, &mut receipts)?;
            let duration = owner.video_duration(input)?;
            if !duration.is_finite() {
                return Err(WavespeedError::Input(
                    "Non-finite source duration unsupported",
                ));
            }
            (
                "/proxy/wavespeed/api/v3/wavespeed-ai/flashvsr",
                RequestModel::FlashVsr,
                json!({"target_resolution":resolution,"video":video,"duration":duration}),
            )
        }
        "WavespeedImageUpscaleNode" => {
            let image = input
                .image
                .as_ref()
                .ok_or(WavespeedError::Input("Missing source image"))?;
            if image.batch != 1 {
                return Err(WavespeedError::Input(
                    "Exactly one input image is required.",
                ));
            }
            let model = input
                .settings
                .get("model")
                .ok_or(WavespeedError::Input("Missing source model"))?;
            let path = if model == "SeedVR2" {
                "/proxy/wavespeed/api/v3/wavespeed-ai/seedvr2/image"
            } else {
                "/proxy/wavespeed/api/v3/wavespeed-ai/ultimate-image-upscaler"
            };
            let resolution = owner.source_lower(resolution)?;
            let uploaded = owner.image_upload(
                input,
                &ImageUploadPolicy {
                    maximum_images: 1,
                    total_pixels: 2048 * 2048,
                    content_type: None,
                    wait_label: "Uploading".into(),
                    show_batch_index: true,
                },
            )?;
            let image = first_upload(uploaded, &mut receipts)?;
            (
                path,
                RequestModel::Image,
                json!({"target_resolution":resolution,"image":image}),
            )
        }
        _ => return Err(WavespeedError::Input("Unsupported source node")),
    };
    let body = owner.normalize_request(model, &raw)?;
    if !body.is_object() {
        return Err(ConformanceError::InvalidRequest.into());
    }
    if serde_json::to_vec(&body)?.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(RequestPlan {
        node: input.node.clone(),
        path,
        body,
        input_receipts: receipts,
    })
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, WavespeedError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(WavespeedError::Response("Missing source string field"))
}
#[derive(Default)]
struct Session {
    invocation: Option<u64>,
    handles: BTreeSet<(u64, u32, u32)>,
}
impl Session {
    fn retain(&mut self, handle: StreamHandle) -> Result<(), WavespeedError> {
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
fn endpoint(value: &str) -> Result<bool, WavespeedError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(WavespeedError::Response(
            "Invalid Wavespeed response endpoint",
        ));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(WavespeedError::Response(
            "Invalid Wavespeed response endpoint",
        ))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(WavespeedError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(WavespeedError::Response(
            "Invalid Wavespeed endpoint authority",
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
) -> Result<RequestHead, WavespeedError> {
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
            return Err(WavespeedError::Input("Missing Comfy API credentials"));
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
    transport: &mut dyn WavespeedTransport,
) -> Result<RetainedResponse, WavespeedError> {
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
    transport: &mut dyn WavespeedTransport,
) -> Result<RetainedResponse, WavespeedError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(WavespeedError::Http {
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
                    return Err(WavespeedError::Http {
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
    transport: &mut dyn WavespeedTransport,
) -> Result<RetainedResponse, WavespeedError> {
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
                    return Err(WavespeedError::Http {
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
    Creation,
    Status,
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
pub trait WavespeedTransport: ProviderTransport {
    fn validate_response(
        &mut self,
        _: &RetainedResponse,
        _: ApiModel,
        _: &Value,
    ) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn validate_status_response(
        &mut self,
        _: &RetainedResponse,
        _: &Value,
    ) -> Result<Option<Value>, ConformanceError> {
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
    transport: &mut dyn WavespeedTransport,
) -> Result<Value, WavespeedError> {
    transport.check_cancelled(Some(response.handle))?;
    let raw: Value = serde_json::from_slice(&response.bytes)?;
    if !raw.is_object() {
        return Err(WavespeedError::Response(
            "Source response requires API object",
        ));
    }
    let value = transport.validate_response(response, model, &raw)?;
    transport.check_cancelled(Some(response.handle))?;
    if !value.is_object() {
        return Err(WavespeedError::Response(
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
) -> Result<(), WavespeedError> {
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
fn retained_id(value: &Value, key: &str) -> Result<String, WavespeedError> {
    let id = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(WavespeedError::Response("Missing source retained task ID"))?;
    if id.is_empty()
        || id.len() > 1024
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(WavespeedError::Response("Unsafe retained task ID"));
    }
    Ok(id.into())
}
fn bounded_receipt(receipt: &[u8]) -> Result<(), WavespeedError> {
    if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
        return Err(ConformanceError::InvalidResponse.into());
    }
    Ok(())
}
#[derive(Clone, Debug)]
pub struct InvocationResult {
    pub node: String,
    pub task_id: String,
    pub input_receipts: Vec<Vec<u8>>,
    pub responses: Vec<RetainedResponse>,
    pub downloaded: RetainedResponse,
    pub cost_receipt: Vec<u8>,
    pub url: String,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn WavespeedTransport,
    owner: &mut dyn WavespeedInputOwner,
) -> Result<InvocationResult, WavespeedError> {
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
    transport.source_request_labels(handle, Some("Waiting for server"), Some("Completed"), true)?;
    let cost = transport.request_cost(
        handle,
        &CostRequest {
            operation: "wavespeed.generation.maximum".into(),
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
    let created = normalized(&creation, ApiModel::Creation, transport)?;
    if created.get("code").and_then(Value::as_i64) != Some(200) {
        return Err(WavespeedError::Provider(format!(
            "Task creation fails with code={} and message={}",
            created
                .get("code")
                .ok_or(WavespeedError::Response("Missing source creation code"))?,
            string(&created, "message")?
        )));
    }
    let data = created
        .get("data")
        .filter(|value| value.is_object())
        .ok_or(WavespeedError::Response(
            "Source missing creation data AttributeError",
        ))?;
    let task_id = retained_id(data, "id")?;
    let mut responses = vec![];
    retain_response(&mut responses, creation)?;
    let poll_path = format!("/proxy/wavespeed/api/v3/predictions/{task_id}/result");
    let mut active_polls = 0u32;
    for _ in 0..8191 {
        if active_polls >= 480 {
            return Err(ConformanceError::PollLimit.into());
        }
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
        transport.check_cancelled(Some(prior.handle))?;
        transport.authorize_continuation(prior, &head)?;
        let handle = transport.start_request(&head)?;
        session.retain(handle)?;
        transport.source_request_labels(handle, Some("Checking"), None, false)?;
        let retained = exchange_existing(
            &head,
            &[],
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            handle,
            &mut session,
            transport,
        )?;
        let raw: Value = serde_json::from_slice(&retained.bytes)?;
        if !raw.is_object() {
            return Err(ConformanceError::InvalidResponse.into());
        }
        let response = transport.validate_status_response(&retained, &raw)?;
        transport.check_cancelled(Some(retained.handle))?;
        if let Some(value) = response.as_ref() {
            if !value.is_object() {
                return Err(ConformanceError::InvalidResponse.into());
            }
            if serde_json::to_vec(value)?.len() > MAXIMUM_RESPONSE_BYTES {
                return Err(ConformanceError::BodyLimit.into());
            }
        }
        let data = response
            .as_ref()
            .and_then(|value| value.get("data"))
            .filter(|value| value.is_object());
        let status = match (response.as_ref(), data) {
            (Some(_), Some(data)) => python_strip(string(data, "status")?).to_lowercase(),
            (Some(_), None) => "failed".into(),
            (None, _) => String::new(),
        };
        let queued = matches!(
            status.as_str(),
            "created" | "queued" | "queueing" | "submitted" | "initializing" | "wait" | "in_queue"
        );
        let completed = matches!(
            status.as_str(),
            "succeeded" | "succeed" | "success" | "completed" | "finished" | "done" | "complete"
        );
        transport.report_task_progress(
            &retained,
            &TaskProgress {
                status: if status.is_empty() {
                    "Processing".into()
                } else {
                    status.clone()
                },
                queued,
                percentage: None,
                source_usd: None,
                estimated_seconds: None,
            },
        )?;
        let url = if completed {
            if response
                .as_ref()
                .and_then(|value| value.get("code"))
                .and_then(Value::as_i64)
                != Some(200)
            {
                let response = response.as_ref().ok_or(WavespeedError::Response(
                    "Missing source completed response",
                ))?;
                return Err(WavespeedError::Provider(format!(
                    "Task processing failed with code={} and message={}",
                    response
                        .get("code")
                        .ok_or(WavespeedError::Response("Missing source final code"))?,
                    string(response, "message")?
                )));
            }
            Some(
                data.and_then(|value| value.get("outputs"))
                    .and_then(Value::as_array)
                    .and_then(|values| values.first())
                    .and_then(Value::as_str)
                    .ok_or(WavespeedError::Response(
                        "Source outputs[0] unavailable IndexError",
                    ))?
                    .to_owned(),
            )
        } else {
            None
        };
        retain_response(&mut responses, retained)?;
        if let Some(url) = url {
            if endpoint(&url)? {
                return Err(WavespeedError::Response(
                    "Source download requires absolute URL",
                ));
            }
            let head = request_head(&url, HttpMethod::Get, credentials, true)?;
            let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
            transport.check_cancelled(Some(prior.handle))?;
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
            transport.check_cancelled(None)?;
            return Ok(InvocationResult {
                node: input.node.clone(),
                task_id,
                input_receipts: plan.input_receipts,
                responses,
                downloaded,
                cost_receipt: cost.receipt,
                url,
            });
        }
        if matches!(
            status.as_str(),
            "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error"
        ) {
            return Err(WavespeedError::Provider(format!("Task failed: {raw}")));
        }
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        transport.check_cancelled(Some(prior.handle))?;
        transport.delay(prior.handle, 10_000)?;
        if !queued {
            active_polls += 1;
        }
    }
    Err(ConformanceError::PollLimit.into())
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, WavespeedError> {
    let name = match node {
        "WavespeedFlashVSRNode" => "VIDEO",
        "WavespeedImageUpscaleNode" => "IMAGE",
        _ => return Err(WavespeedError::Input("Unknown source output node")),
    };
    let registry = TypeRegistry::built_in()
        .map_err(|_| WavespeedError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve(name)
        .map_err(|_| WavespeedError::Response("Missing canonical output type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| WavespeedError::Response("Missing canonical output family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub trait WavespeedOutputOwner {
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
    transport: &mut dyn WavespeedTransport,
    owner: &mut dyn WavespeedOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), WavespeedError> {
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
        "/../../sim_test_support/fixtures/providers/wavespeed/source-contract.json"
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
    if source["sources"]["comfy_api_nodes/nodes_wavespeed.py"] != SOURCE_SHA256 {
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
        rejected_status: Vec<StreamHandle>,
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
                rejected_status: vec![],
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
                } else if response.method == HttpMethod::Get
                    && response
                        .path
                        .starts_with("/proxy/wavespeed/api/v3/predictions/")
                {
                    self.rejected_status.push(handle);
                }
                self.active = None;
            }
            Ok(event)
        }
    }
    impl WavespeedTransport for TestTransport {
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
        fn validate_status_response(
            &mut self,
            retained: &RetainedResponse,
            raw: &Value,
        ) -> Result<Option<Value>, ConformanceError> {
            if !self.validate_models {
                return Err(ConformanceError::HostFailure);
            }
            if !self.completed.iter().any(|value| {
                value.handle == retained.handle
                    && value.bytes == retained.bytes
                    && value.receipt == retained.receipt
                    && value.headers == retained.headers
            }) || !matches!(serde_json::from_slice::<Value>(&retained.bytes),Ok(ref value) if value==raw)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            if self.rejected_status.contains(&retained.handle) {
                return Ok(None);
            }
            self.normalized
                .iter()
                .find(|(handle, kind, _)| *handle == retained.handle && *kind == ApiModel::Status)
                .map(|(_, _, value)| Some(value.clone()))
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
            normalized: normalized.map(|value| {
                (
                    if method == HttpMethod::Post {
                        ApiModel::Creation
                    } else {
                        ApiModel::Status
                    },
                    value,
                )
            }),
            failure: None,
        })
    }
    fn scripts(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let mut script = vec![];
        for request in case["requests"].as_array().ok_or("requests")? {
            script.push(response(
                request["path"].as_str().ok_or("path")?,
                HttpMethod::Post,
                serde_json::to_vec(&request["body"])?,
                &request["response"],
                request.get("normalized_response").cloned(),
            )?);
        }
        for poll in case["polls"].as_array().ok_or("polls")? {
            for retained in poll["responses"].as_array().ok_or("poll responses")? {
                script.push(response(
                    poll["path"].as_str().ok_or("path")?,
                    HttpMethod::Get,
                    vec![],
                    &retained["raw"],
                    retained.get("normalized").cloned(),
                )?);
            }
        }
        for download in case["downloads"].as_array().ok_or("downloads")? {
            let mut retained = response(
                download["url"].as_str().ok_or("URL")?,
                HttpMethod::Get,
                vec![],
                &json!({}),
                None,
            )?;
            retained.bytes = hex(&download["bytes_hex"])?;
            script.push(retained);
        }
        Ok(script)
    }
    struct TestOwner {
        case: Value,
        constructor_count: usize,
        upload_count: usize,
        events: Events,
        fail: bool,
    }
    impl TestOwner {
        fn new(case: &Value, events: Events) -> Self {
            Self {
                case: case.clone(),
                constructor_count: 0,
                upload_count: 0,
                events,
                fail: false,
            }
        }
        fn uploaded(&mut self) -> Result<PreparedUpload, ConformanceError> {
            let record = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            if hex(&record["bytes_hex"])
                .map_err(|_| ConformanceError::InvalidRequest)?
                .is_empty()
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.events
                .borrow_mut()
                .push(format!("upload:{}", self.upload_count));
            self.upload_count += 1;
            Ok(PreparedUpload {
                urls: vec![format!(
                    "https://source.fixture.invalid/input-{}.bin",
                    self.upload_count
                )],
                receipts: vec![
                    format!("canonical upload receipt:{}", self.upload_count).into_bytes(),
                ],
            })
        }
    }
    impl WavespeedInputOwner for TestOwner {
        fn verify_inputs(&mut self, input: &Input) -> Result<(), ConformanceError> {
            let expected: Input = serde_json::from_value(self.case["input"].clone())
                .map_err(|_| ConformanceError::InvalidRequest)?;
            if self.fail
                || serde_json::to_value(input).map_err(|_| ConformanceError::InvalidRequest)?
                    != serde_json::to_value(expected)
                        .map_err(|_| ConformanceError::InvalidRequest)?
            {
                return Err(ConformanceError::HostFailure);
            }
            Ok(())
        }
        fn source_lower(&mut self, raw: &Value) -> Result<String, ConformanceError> {
            let record = self.case["lowers"]
                .get(0)
                .ok_or(ConformanceError::InvalidRequest)?;
            self.events.borrow_mut().push(format!("lower:{raw}"));
            if record["raw"] != *raw {
                return Err(ConformanceError::InvalidRequest);
            }
            record["resolved"]
                .as_str()
                .map(str::to_owned)
                .ok_or(ConformanceError::InvalidRequest)
        }
        fn normalize_request(
            &mut self,
            model: RequestModel,
            raw: &Value,
        ) -> Result<Value, ConformanceError> {
            let record = self.case["constructors"]
                .get(self.constructor_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            if record["model"] != model.source_name() || record["raw"] != *raw {
                return Err(ConformanceError::InvalidRequest);
            }
            self.events
                .borrow_mut()
                .push(format!("constructor:{}", self.constructor_count));
            self.constructor_count += 1;
            record
                .get("normalized")
                .cloned()
                .ok_or(ConformanceError::InvalidRequest)
        }
        fn validate_video_container(&mut self, input: &Input) -> Result<(), ConformanceError> {
            self.events.borrow_mut().push("container".into());
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            if video.container_error
                || !matches!(video.container.as_str(), "mp4" | "mov,mp4,m4a,3gp,3g2,mj2")
            {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(())
        }
        fn validate_video_duration(
            &mut self,
            input: &Input,
            minimum: f64,
            maximum: f64,
            epsilon: f64,
        ) -> Result<(), ConformanceError> {
            if (minimum, maximum, epsilon) != (5.0, 600.0, 0.0001) {
                return Err(ConformanceError::InvalidRequest);
            }
            self.events.borrow_mut().push("duration".into());
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            if video.duration_error {
                return Ok(());
            }
            if minimum - epsilon > video.duration || video.duration > maximum + epsilon {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(())
        }
        fn video_duration(&mut self, input: &Input) -> Result<f64, ConformanceError> {
            self.events.borrow_mut().push("duration".into());
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            if video.duration_error {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(video.duration)
        }
        fn video_upload_mp4_h264(
            &mut self,
            _: &Input,
            label: &str,
        ) -> Result<PreparedUpload, ConformanceError> {
            self.events.borrow_mut().push("convert:video".into());
            if self.case["conversions"].get(0) != Some(&json!({"format":"mp4","codec":"h264"})) {
                return Err(ConformanceError::InvalidRequest);
            }
            let record = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            if label != "Uploading"
                || record["wait_label"] != label
                || record["mime_type"] != "video/mp4"
                || hex(&record["bytes_hex"]).map_err(|_| ConformanceError::InvalidRequest)?
                    != include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../sim_test_support/fixtures/providers/wavespeed/encoded-video.mp4"
                    ))
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.uploaded()
        }
        fn image_upload(
            &mut self,
            _: &Input,
            policy: &ImageUploadPolicy,
        ) -> Result<PreparedUpload, ConformanceError> {
            let record = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            let bytes = hex(&record["bytes_hex"]).map_err(|_| ConformanceError::InvalidRequest)?;
            if policy.maximum_images != 1
                || policy.total_pixels != 2048 * 2048
                || policy.content_type.is_some()
                || !policy.show_batch_index
                || record["wait_label"] != policy.wait_label
                || !record["mime_type"].is_null()
                || !bytes.starts_with(b"\x89PNG\r\n\x1a\n")
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.uploaded()
        }
    }
    fn input(case: &Value) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(case["input"].clone())?)
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn source_events(case: &Value) -> Result<Vec<String>, Box<dyn Error>> {
        case["events"]
            .as_array()
            .ok_or("events")?
            .iter()
            .map(|event| {
                if let Some(value) = event.get("lower") {
                    return Ok(format!("lower:{value}"));
                }
                for key in ["container", "duration"] {
                    if event.get(key).is_some() {
                        return Ok(key.into());
                    }
                }
                for key in [
                    "convert",
                    "upload",
                    "constructor",
                    "request",
                    "poll_http",
                    "download",
                ] {
                    if let Some(value) = event.get(key) {
                        return Ok(format!(
                            "{}:{}",
                            if matches!(key, "poll_http" | "download") {
                                "request"
                            } else {
                                key
                            },
                            if let Some(text) = value.as_str() {
                                text.to_owned()
                            } else {
                                value.to_string()
                            }
                        ));
                    }
                }
                Err("unknown source event".into())
            })
            .collect()
    }
    fn successful_case(node: &str) -> Result<Value, Box<dyn Error>> {
        source_oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| {
                case["input"]["node"] == node && case["input"].get("oracle_options").is_none()
            })
            .cloned()
            .ok_or_else(|| "source case".into())
    }
    struct NoInputOwner;
    impl WavespeedInputOwner for NoInputOwner {}
    struct NoOutputOwner;
    impl WavespeedOutputOwner for NoOutputOwner {}
    struct Sink {
        calls: usize,
    }
    impl MaterializationSink for Sink {
        fn publish(
            &mut self,
            _: crate::common::MaterializedBatch,
        ) -> Result<(), crate::common::MaterializationError> {
            self.calls += 1;
            Ok(())
        }
    }
    #[test]
    fn exact_actual_source_successes_and_ordered_failure_prefixes() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for kind in ["cases", "errors"] {
            for (index, case) in source[kind]
                .as_array()
                .ok_or("source cases")?
                .iter()
                .enumerate()
            {
                let input = input(case)?;
                let events = Rc::new(RefCell::new(vec![]));
                let mut owner = TestOwner::new(case, events.clone());
                let mut transport = TestTransport::new(scripts(case)?, events.clone());
                let actual = invoke(&input, credentials(), 1_000, &mut transport, &mut owner);
                let empty_id = case["input"]["oracle_options"]["creation"]["data"]["id"] == "";
                let decode_error = case["error_type"] == "UnidentifiedImageError";
                if kind == "cases" && !empty_id || decode_error {
                    let result = actual.map_err(|error| {
                        format!("{kind} case{index} {:?}: {error}", case["input"])
                    })?;
                    transport.finish()?;
                    if result.downloaded.bytes != hex(&case["downloads"][0]["bytes_hex"])?
                        || result.url
                            != case["downloads"][0]["url"].as_str().ok_or("download URL")?
                    {
                        return Err(format!("download {kind}/{index}").into());
                    }
                    let expected_updates: Vec<f64> = case["polls"][0]["progress_updates"]
                        .as_array()
                        .ok_or("progress")?
                        .iter()
                        .filter_map(|value| value["current"].as_f64())
                        .collect();
                    let actual_updates: Vec<f64> = transport
                        .progress
                        .iter()
                        .filter_map(|value| value.2)
                        .collect();
                    if actual_updates != expected_updates {
                        return Err(format!(
                            "progress {kind}/{index}: {actual_updates:?} != {expected_updates:?}"
                        )
                        .into());
                    }
                    let expected_delays: Vec<u64> = case["polls"][0]["delays"]
                        .as_array()
                        .ok_or("delays")?
                        .iter()
                        .filter_map(|value| value.as_f64().map(|value| (value * 1000.0) as u64))
                        .collect();
                    if transport.delays != expected_delays {
                        return Err(format!("delay {kind}/{index}").into());
                    }
                    let mut sink = Sink { calls: 0 };
                    assert!(matches!(
                        materialize_and_publish(
                            &result,
                            &mut transport,
                            &mut NoOutputOwner,
                            &mut sink
                        ),
                        Err(WavespeedError::Transport(ConformanceError::HostFailure))
                    ));
                    assert_eq!(sink.calls, 0);
                } else if empty_id {
                    assert!(matches!(
                        actual,
                        Err(WavespeedError::Response("Unsafe retained task ID"))
                    ));
                    assert_eq!(transport.heads.len(), 1);
                } else if actual.is_ok() {
                    return Err(format!("unexpected success {kind}/{index}").into());
                }
                if !empty_id && *events.borrow() != source_events(case)? {
                    return Err(format!(
                        "phase mismatch {kind}/{index}: {:?} != {:?}",
                        events.borrow(),
                        source_events(case)?
                    )
                    .into());
                }
                if !empty_id
                    && (owner.upload_count != case["uploads"].as_array().ok_or("uploads")?.len()
                        || owner.constructor_count
                            != case["constructors"].as_array().ok_or("constructors")?.len())
                {
                    return Err(format!("owner calls {kind}/{index}").into());
                }
            }
        }
        Ok(())
    }
    #[test]
    fn canonical_image_pixels_and_claims_remain_source_owned() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/wavespeed/conformance.fixture"
        )))?;
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "synthetic_prefix_tombstone")
                .count(),
            2
        );
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            3
        );
        let case = successful_case("WavespeedImageUpscaleNode")?;
        assert_eq!(case["output"][0]["dtype"], "torch.float32");
        assert_eq!(case["output"][0]["shape"][0], 1);
        assert_eq!(case["output"][0]["shape"][3], 4);
        let pixels = case["output"][0]["pixels"]
            .as_array()
            .ok_or("source pixels")?;
        let actual: Vec<f64> = pixels
            .iter()
            .map(|value| value.as_f64().ok_or("float pixel"))
            .collect::<Result<_, _>>()?;
        let expected: Vec<f64> = [128.0f32 / 255.0, 64.0f32 / 255.0, 32.0f32 / 255.0, 1.0f32]
            .into_iter()
            .map(f64::from)
            .collect();
        assert_eq!(actual, expected);
        let image = expected_outputs("WavespeedImageUpscaleNode")?;
        let video = expected_outputs("WavespeedFlashVSRNode")?;
        assert_ne!(image[0].type_id, video[0].type_id);
        Ok(())
    }
    #[test]
    fn input_cost_cancel_lost_worker_api_and_continuation_gates_stop_before_later_work()
    -> Result<(), Box<dyn Error>> {
        let case = successful_case("WavespeedImageUpscaleNode")?;
        let input = input(&case)?;
        let events = Rc::new(RefCell::new(vec![]));
        let mut transport = TestTransport::new(scripts(&case)?, events.clone());
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut NoInputOwner
            ),
            Err(WavespeedError::Transport(ConformanceError::HostFailure))
        ));
        assert!(events.borrow().is_empty());
        for failure in [ConformanceError::Cancelled, ConformanceError::WorkerLost] {
            let events = Rc::new(RefCell::new(vec![]));
            let mut owner = TestOwner::new(&case, events.clone());
            let mut transport = TestTransport::new(scripts(&case)?, events.clone());
            transport.failure = Some(failure.clone());
            assert!(
                matches!(invoke(&input,credentials(),100,&mut transport,&mut owner),Err(WavespeedError::Transport(error)) if error==failure)
            );
            assert!(events.borrow().is_empty());
        }
        for mode in 0..4 {
            let events = Rc::new(RefCell::new(vec![]));
            let mut owner = TestOwner::new(&case, events.clone());
            let mut transport = TestTransport::new(scripts(&case)?, events);
            match mode {
                0 => transport.reject_cost = true,
                1 => transport.validate_models = false,
                2 => transport.continuation_bound = false,
                _ => transport.cancelled = true,
            }
            assert!(invoke(&input, credentials(), 100, &mut transport, &mut owner).is_err());
            assert!(transport.completed.len() <= 1);
        }
        let events = Rc::new(RefCell::new(vec![]));
        let mut owner = TestOwner::new(&case, events.clone());
        let mut fresh = TestTransport::new(scripts(&case)?, events);
        assert!(invoke(&input, credentials(), 100, &mut fresh, &mut owner).is_ok());
        fresh.finish()?;
        Ok(())
    }
    #[test]
    fn full_retained_bytes_headers_receipts_and_invocation_identity_prevent_replay()
    -> Result<(), Box<dyn Error>> {
        let case = successful_case("WavespeedFlashVSRNode")?;
        let events = Rc::new(RefCell::new(vec![]));
        let mut owner = TestOwner::new(&case, events.clone());
        let mut transport = TestTransport::new(scripts(&case)?, events);
        let result = invoke(
            &input(&case)?,
            credentials(),
            100,
            &mut transport,
            &mut owner,
        )?;
        let retained = result.responses.first().ok_or("creation")?.clone();
        let head = request_head(
            "/proxy/wavespeed/api/v3/predictions/source-task-42/result",
            HttpMethod::Get,
            credentials(),
            false,
        )?;
        for mode in 0..4 {
            let mut changed = retained.clone();
            match mode {
                0 => changed.bytes.push(0),
                1 => changed.receipt.push(0),
                2 => changed.headers.push(Header {
                    name: "changed".into(),
                    value: "value".into(),
                }),
                _ => changed.handle.invocation += 1,
            }
            assert!(
                transport
                    .validate_response(
                        &changed,
                        ApiModel::Creation,
                        &serde_json::from_slice(&retained.bytes)?
                    )
                    .is_err()
            );
            assert!(transport.authorize_continuation(&changed, &head).is_err());
        }
        let mut session = Session::default();
        session.retain(retained.handle)?;
        assert!(session.retain(retained.handle).is_err());
        assert!(
            session
                .retain(StreamHandle {
                    invocation: 99,
                    ..retained.handle
                })
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn retry_handles_bounds_http_errors_and_credential_isolation_are_checked()
    -> Result<(), Box<dyn Error>> {
        let case = successful_case("WavespeedImageUpscaleNode")?;
        for stale in [false, true] {
            let events = Rc::new(RefCell::new(vec![]));
            let mut owner = TestOwner::new(&case, events.clone());
            let mut script = scripts(&case)?;
            let mut rejected = response(
                &script[0].path,
                HttpMethod::Post,
                script[0].body.clone(),
                &json!({}),
                None,
            )?;
            rejected.status = 429;
            rejected.headers.push(Header {
                name: "Retry-After".into(),
                value: "2".into(),
            });
            script.insert(0, rejected);
            let mut transport = TestTransport::new(script, events);
            transport.stale_retry = stale;
            let actual = invoke(
                &input(&case)?,
                credentials(),
                100,
                &mut transport,
                &mut owner,
            );
            if stale {
                assert!(matches!(
                    actual,
                    Err(WavespeedError::Transport(ConformanceError::InvalidHandle))
                ));
            } else {
                assert!(actual.is_ok());
                assert_eq!(transport.delays.first(), Some(&2000));
                transport.finish()?;
            }
        }
        for mode in 0..5 {
            let events = Rc::new(RefCell::new(vec![]));
            let mut owner = TestOwner::new(&case, events.clone());
            let mut script = scripts(&case)?;
            match mode {
                0 => script[0].bytes = vec![0; MAXIMUM_RESPONSE_BYTES + 1],
                1 => script[0].receipt.clear(),
                2 => script[0].headers.push(Header {
                    name: "x".into(),
                    value: "\n".into(),
                }),
                3 => script[0].bytes = b"malformed".to_vec(),
                _ => script[0].failure = Some(ConformanceError::WorkerLost),
            }
            let mut transport = TestTransport::new(script, events);
            assert!(
                invoke(
                    &input(&case)?,
                    credentials(),
                    100,
                    &mut transport,
                    &mut owner
                )
                .is_err()
            );
            assert_eq!(transport.heads.len(), 1);
        }
        let relative = request_head(
            "/proxy/wavespeed/api/v3/wavespeed-ai/flashvsr",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        assert_eq!(relative.secret_id.as_deref(), Some("sim_org.auth_token"));
        let api = request_head(
            "/proxy/wavespeed/api/v3/wavespeed-ai/flashvsr",
            HttpMethod::Post,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(api.secret_id.as_deref(), Some("sim_org.api_key"));
        assert!(
            request_head(
                "/proxy/wavespeed/api/v3/wavespeed-ai/flashvsr",
                HttpMethod::Post,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                false
            )
            .is_err()
        );
        let absolute = request_head(
            "https://source.fixture.invalid/result.png",
            HttpMethod::Get,
            credentials(),
            true,
        )?;
        assert!(absolute.secret_id.is_none());
        assert_eq!(absolute.headers.len(), 1);
        Ok(())
    }
}
