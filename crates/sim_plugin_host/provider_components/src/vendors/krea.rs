use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TransportLimits, TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};

pub const NAMESPACE: &str = "zed.sim.provider.krea";
pub const SOURCE_SHA256: &str = "f67aa4c46d33b30c05525ab0f27ebf6e2cbf59e698c62ac5b226786be4153ea7";
pub const ALIASES: &[&str] = &[];
pub const MAXIMUM_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
const MAXIMUM_POLLS: u32 = 480;
const MAXIMUM_TOTAL_POLLS: u32 = 4096;
pub const MAXIMUM_STYLE_PIXELS: u64 = 2048 * 2048;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0304",
        node_identifier: "Krea2ImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0305",
        node_identifier: "Krea2StyleReferenceNode",
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
        feature_id: "COMFY-API-EXT-0084",
        method: "POST",
        path: "/proxy/krea/assets",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0085",
        method: "POST",
        path: "/proxy/krea/generate/image/krea/krea-2/large",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0086",
        method: "POST",
        path: "/proxy/krea/generate/image/krea/krea-2/medium",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0087",
        method: "POST",
        path: "/proxy/krea/generate/image/krea/krea-2/medium-turbo",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0088",
        method: "GET",
        path: "/proxy/krea/jobs/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0089",
        method: "GET",
        path: "/proxy/krea/jobs/{initial.job_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum KreaError {
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
impl fmt::Display for KreaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "Krea job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Krea HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for KreaError {}
impl From<ConformanceError> for KreaError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for KreaError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageInput {
    pub prompt: String,
    pub model: Value,
    pub seed: i64,
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn numeric(value: &Value) -> Result<f64, KreaError> {
    let number = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(number) => python_strip(number).parse().ok(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
    .ok_or(KreaError::Input("Invalid source float"))?;
    if !number.is_finite() {
        return Err(KreaError::Input("Non-finite source float is unsupported"));
    }
    Ok(number)
}
fn required_string<'a>(object: &'a Value, key: &str) -> Result<&'a str, KreaError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or(KreaError::Input("Missing source string field"))
}
fn uuid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}
pub fn request(input: &ImageInput) -> Result<(&'static str, Vec<u8>), KreaError> {
    if input.prompt.is_empty() {
        return Err(KreaError::Input("Prompt must have at least one character"));
    }
    let path = match required_string(&input.model, "model")? {
        "Krea 2 Medium" => ROUTES[2].path,
        "Krea 2 Medium Turbo" => ROUTES[3].path,
        "Krea 2 Large" => ROUTES[1].path,
        _ => return Err(KreaError::Input("Unknown Krea model")),
    };
    let mut optional_fields = serde_json::Map::new();
    let identifier = match input.model.get("moodboard_id") {
        None | Some(Value::Null) => "",
        Some(Value::String(identifier)) => python_strip(identifier),
        _ => return Err(KreaError::Input("Invalid moodboard ID")),
    };
    if !identifier.is_empty() {
        if !uuid(identifier) {
            return Err(KreaError::Input("Moodboard ID must be a UUID"));
        }
        let strength = match input.model.get("moodboard_strength") {
            None | Some(Value::Null) => 0.35,
            Some(strength) => numeric(strength)?,
        };
        if !(-0.5..=1.5).contains(&strength) {
            return Err(KreaError::Input("Invalid moodboard strength"));
        }
        optional_fields.insert(
            "moodboards".into(),
            json!([{"id":identifier,"strength":strength}]),
        );
    }
    match input.model.get("style_reference") {
        None | Some(Value::Null) => {}
        Some(Value::Array(references)) => {
            if references.len() > 10 {
                return Err(KreaError::Input(
                    "Krea permits at most ten style references",
                ));
            }
            if !references.is_empty() {
                let references = references
                    .iter()
                    .map(|reference| {
                        let strength = numeric(
                            reference
                                .get("strength")
                                .ok_or(KreaError::Input("Missing style strength"))?,
                        )?;
                        if !(-2.0..=2.0).contains(&strength) {
                            return Err(KreaError::Input("Invalid style strength"));
                        }
                        let mut reference_body = json!({"strength":strength});
                        match reference.get("url") {
                            Some(Value::String(url)) => reference_body["url"] = json!(url),
                            Some(Value::Null) => {}
                            _ => return Err(KreaError::Input("Missing style URL")),
                        };
                        Ok(reference_body)
                    })
                    .collect::<Result<Vec<_>, KreaError>>()?;
                optional_fields.insert("image_style_references".into(), json!(references));
            }
        }
        _ => return Err(KreaError::Input("Invalid style reference chain")),
    }
    let mut body = json!({"prompt":input.prompt,"aspect_ratio":required_string(&input.model,"aspect_ratio")?,"resolution":required_string(&input.model,"resolution")?,"seed":input.seed,"creativity":required_string(&input.model,"creativity")?});
    for (field, value) in optional_fields {
        body[&field] = value;
    }
    let body = serde_json::to_vec(&body)?;
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok((path, body))
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StylePreparationRequirement {
    pub content_type: &'static str,
    pub total_pixels: u64,
    pub resize_method: &'static str,
    pub crop: &'static str,
    pub first_batch_only: bool,
}
pub fn style_preparation_requirement(
    incoming: &[Value],
) -> Result<StylePreparationRequirement, KreaError> {
    if incoming.len() >= 10 {
        return Err(KreaError::Input(
            "Krea permits at most ten style references",
        ));
    }
    if serde_json::to_vec(incoming)?.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(StylePreparationRequirement {
        content_type: "image/png",
        total_pixels: MAXIMUM_STYLE_PIXELS,
        resize_method: "lanczos",
        crop: "disabled",
        first_batch_only: true,
    })
}
#[derive(Clone, Debug)]
pub struct PreparedStyleImage {
    pub file_name: String,
    pub bytes: Vec<u8>,
    pub source_width: u32,
    pub source_height: u32,
    pub prepared_width: u32,
    pub prepared_height: u32,
}
pub fn source_downscale(width: u32, height: u32) -> Result<(u32, u32), KreaError> {
    if width == 0 || height == 0 {
        return Err(KreaError::Input("Empty style image"));
    }
    if u64::from(width) * u64::from(height) <= MAXIMUM_STYLE_PIXELS {
        return Ok((width, height));
    }
    let scale = (MAXIMUM_STYLE_PIXELS as f64 / (f64::from(width) * f64::from(height))).sqrt();
    let width = (f64::from(width) * scale) as u32;
    let height = (f64::from(height) * scale) as u32;
    Ok((width.max(2) & !1, height.max(2) & !1))
}
pub fn multipart(image: &PreparedStyleImage, boundary: &str) -> Result<Vec<u8>, KreaError> {
    if source_downscale(image.source_width, image.source_height)?
        != (image.prepared_width, image.prepared_height)
        || image.bytes.is_empty()
        || image.bytes.len() > MAXIMUM_MEDIA_BYTES
        || !image.file_name.ends_with(".png")
    {
        return Err(KreaError::Input("Invalid canonical PNG preparation"));
    }
    let mut limits = TransportLimits::conformance();
    limits.maximum_body_bytes = MAXIMUM_MEDIA_BYTES;
    MultipartField::encode_checked(
        &[MultipartField {
            name: "file".into(),
            file_name: Some(image.file_name.clone()),
            media_type: Some("image/png".into()),
            bytes: image.bytes.clone(),
        }],
        boundary,
        limits,
    )
    .map_err(KreaError::from)
}
#[derive(Clone, Debug, Deserialize)]
pub struct JobResult {
    pub urls: Option<Vec<String>>,
    pub style_id: Option<String>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Job {
    pub job_id: String,
    pub status: String,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub result: Option<JobResult>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Queued,
    Processing,
    Completed,
}
pub fn task_status(bytes: &[u8]) -> Result<(Job, Phase), KreaError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let job: Job = serde_json::from_slice(bytes)?;
    let status = python_strip(&job.status).to_lowercase();
    let phase = match status.as_str() {
        "succeeded" | "succeed" | "success" | "completed" | "finished" | "done" | "complete" => {
            Phase::Completed
        }
        "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error" => {
            return Err(KreaError::Provider(job.status));
        }
        "backlogged" | "queued" | "scheduled" => Phase::Queued,
        _ => Phase::Processing,
    };
    Ok((job, phase))
}
#[derive(Clone, Debug, Deserialize)]
pub struct AssetResponse {
    pub id: String,
    pub image_url: String,
    pub uploaded_at: String,
}
#[derive(Clone, Debug)]
pub enum SourceOutput {
    Image {
        job_id: String,
        url: String,
        bytes: Vec<u8>,
        cost_receipt: Vec<u8>,
        submission_receipt: Vec<u8>,
        status_receipt: Vec<u8>,
        download_receipt: Vec<u8>,
        handle: StreamHandle,
    },
    StyleChain {
        chain: Vec<Value>,
        asset_url: String,
        asset_receipt: Vec<u8>,
        handle: StreamHandle,
    },
}
impl SourceOutput {
    pub fn alias(&self) -> &'static str {
        match self {
            Self::Image { .. } => "IMAGE",
            Self::StyleChain { .. } => "KreaIO.STYLE_REF",
        }
    }
}
pub trait KreaTransport: ProviderTransport {
    fn begin_response_deadline(
        &mut self,
        _handle: StreamHandle,
        _timeout_milliseconds: Option<u64>,
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
    fn retain(&mut self, handle: StreamHandle) -> Result<(), KreaError> {
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
fn endpoint(value: &str) -> Result<bool, KreaError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(KreaError::Response("Invalid Krea3D response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(KreaError::Response("Invalid Krea3D response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(KreaError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(KreaError::Response("Invalid Krea3D endpoint authority"));
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
) -> Result<RequestHead, KreaError> {
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
            return Err(KreaError::Input("Missing Comfy API credentials"));
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
const ASSET_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 1,
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
    transport: &mut dyn KreaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), KreaError> {
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
    transport: &mut dyn KreaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), KreaError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(KreaError::Http {
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
                    return Err(KreaError::Http {
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
    transport: &mut dyn KreaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), KreaError> {
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
                    return Err(KreaError::Http {
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

pub fn invoke_image(
    input: &ImageInput,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn KreaTransport,
) -> Result<SourceOutput, KreaError> {
    let (path, body) = request(input)?;
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    transport.check_cancelled(None)?;
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    let mut session = Session::default();
    session.retain(handle)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "krea.generation.maximum".into(),
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
    let (bytes, submission_receipt, _handle) = exchange_existing(
        &head,
        &body,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let initial: Job = serde_json::from_slice(&bytes)?;
    if initial.job_id.is_empty()
        || initial.job_id.len() > 1024
        || !initial
            .job_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(KreaError::Response("Unsafe job route selector"));
    }
    let poll_head = request_head(
        &format!("/proxy/krea/jobs/{}", initial.job_id),
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
        let (job, phase) = task_status(&bytes)?;
        if job.job_id != initial.job_id {
            return Err(KreaError::Response("Response job identity changed"));
        }
        if phase == Phase::Completed {
            let url = job
                .result
                .and_then(|result| result.urls)
                .and_then(|urls| urls.into_iter().next())
                .ok_or(KreaError::Response("Completed Krea job has no image URLs"))?;
            if endpoint(&url)? {
                return Err(KreaError::Response("Expected absolute image URL"));
            }
            let download_head = request_head(&url, HttpMethod::Get, credentials, true)?;
            let (bytes, download_receipt, handle) = exchange_with_policy(
                &download_head,
                &[],
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                &mut session,
                transport,
            )?;
            if bytes.is_empty() {
                return Err(KreaError::Response("Downloaded image is empty"));
            }
            return Ok(SourceOutput::Image {
                job_id: initial.job_id,
                url,
                bytes,
                cost_receipt: approval.receipt,
                submission_receipt,
                status_receipt,
                download_receipt,
                handle,
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
pub fn invoke_style_reference(
    image: &PreparedStyleImage,
    strength: f64,
    incoming: &[Value],
    boundary: &str,
    credentials: CredentialAvailability,
    transport: &mut dyn KreaTransport,
) -> Result<SourceOutput, KreaError> {
    style_preparation_requirement(incoming)?;
    if !strength.is_finite() {
        return Err(KreaError::Input("Non-finite style strength is unsupported"));
    }
    let body = multipart(image, boundary)?;
    let mut head = request_head("/proxy/krea/assets", HttpMethod::Post, credentials, false)?;
    for header in &mut head.headers {
        if header.name == "Content-Type" {
            header.value = format!("multipart/form-data; boundary={boundary}");
        }
    }
    let mut session = Session::default();
    let (bytes, asset_receipt, handle) = exchange_with_policy(
        &head,
        &body,
        MAXIMUM_RESPONSE_BYTES,
        ASSET_RETRY,
        &mut session,
        transport,
    )?;
    let asset: AssetResponse = serde_json::from_slice(&bytes)?;
    if endpoint(&asset.image_url)? {
        return Err(KreaError::Response("Expected absolute Krea asset URL"));
    }
    let mut chain = incoming.to_vec();
    chain.push(json!({"url":asset.image_url,"strength":strength}));
    if serde_json::to_vec(&chain)?.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(SourceOutput::StyleChain {
        chain,
        asset_url: asset.image_url,
        asset_receipt,
        handle,
    })
}
pub trait KreaMaterializer {
    // The retained host verifies receipts and allocates source RGBA pixels or the registered
    // custom chain under the caller's NativeNodeContext before returning SDK ABI values.
    fn materialize(
        &mut self,
        result: &SourceOutput,
        expected: &ExpectedOutput,
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub fn materialize_and_publish(
    result: &SourceOutput,
    registry: &TypeRegistry,
    transport: &mut dyn KreaTransport,
    owner: &mut dyn KreaMaterializer,
    sink: &mut dyn MaterializationSink,
) -> Result<(), KreaError> {
    transport.check_cancelled(None)?;
    let type_id = registry
        .resolve(result.alias())
        .map_err(|_| KreaError::Response("Canonical Krea output type is unbound"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| KreaError::Response("Canonical Krea output family is unbound"))?;
    let expected = ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    };
    let outputs = owner.materialize(result, &expected)?;
    let batch = validate_outputs(&[expected], outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn source_oracle() -> Result<Value, ConformanceError> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/krea/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture.get("namespace") != Some(&json!(NAMESPACE))
        || fixture.get("source_sha256") != Some(&json!(SOURCE_SHA256))
        || fixture.get("aliases") != Some(&json!(ALIASES))
        || fixture.get("nodes") != Some(&json!(NODES))
        || fixture.get("routes") != Some(&json!(ROUTES))
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = source_oracle()?;
    if source["sources"]["comfy_api_nodes/nodes_krea.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: ImageInput = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let (path, bytes) = request(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let body: Value =
            serde_json::from_slice(&bytes).map_err(|_| ConformanceError::InvalidRequest)?;
        if path
            != case["request"]["path"]
                .as_str()
                .ok_or(ConformanceError::InvalidRequest)?
            || body != case["request"]["body"]
        {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    for case in source["errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: ImageInput = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        if request(&input).is_ok() {
            return Err(ConformanceError::InvalidRequest);
        }
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

    #[test]
    fn test_transport_rejects_unequal_non_json_bodies() -> Result<(), Box<dyn Error>> {
        for (expected, actual, accepted) in [
            (b"{bad".as_slice(), b"{worse".as_slice(), false),
            (&[0, 255, 1], &[0, 255, 2], false),
            (&[0, 255, 1], &[0, 255, 1], true),
            (
                b"{\"one\":1,\"two\":2}".as_slice(),
                b"{\"two\":2,\"one\":1}".as_slice(),
                true,
            ),
        ] {
            let mut transport = TestTransport::new(vec![response(
                "/fixture",
                HttpMethod::Post,
                expected.to_vec(),
                vec![],
            )]);
            let handle = transport.start_request(&RequestHead {
                path: "/fixture".into(),
                method: HttpMethod::Post,
                secret_id: None,
                headers: vec![],
            })?;
            let result = transport.write_request_chunk(handle, 0, actual, true);
            if accepted {
                result?;
            } else {
                assert_eq!(result, Err(ConformanceError::InvalidRequest));
            }
        }
        Ok(())
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
                    && (!matches!(
                        (serde_json::from_slice::<Value>(&response.body), serde_json::from_slice::<Value>(bytes)),
                        (Ok(expected), Ok(actual)) if expected == actual
                    ))
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
    impl KreaTransport for TestTransport {
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
            receipt: format!("receipt:{path}").into_bytes(),
        }
    }
    fn cases() -> Result<Vec<Value>, Box<dyn Error>> {
        source_oracle()?["cases"]
            .as_array()
            .cloned()
            .ok_or_else(|| "missing cases".into())
    }
    fn script(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let path = case["request"]["path"].as_str().ok_or("missing path")?;
        Ok(vec![
            response(
                path,
                HttpMethod::Post,
                serde_json::to_vec(&case["request"]["body"])?,
                serde_json::to_vec(
                    &json!({"job_id":"job-42","status":"queued","created_at":"source-time"}),
                )?,
            ),
            response(
                "/proxy/krea/jobs/job-42",
                HttpMethod::Get,
                vec![],
                serde_json::to_vec(
                    &json!({"job_id":"job-42","status":"completed","created_at":"source-time","result":{"urls":["https://output.fixture.invalid/first.png","https://output.fixture.invalid/ignored.png"]}}),
                )?,
            ),
            response(
                "https://output.fixture.invalid/first.png",
                HttpMethod::Get,
                vec![],
                b"actual downloaded encoded image bytes; native decode owner separate".to_vec(),
            ),
        ])
    }
    fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        if !value.len().is_multiple_of(2) {
            return Err("invalid hex".into());
        }
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
            .collect()
    }
    fn prepared() -> Result<PreparedStyleImage, Box<dyn Error>> {
        let source = source_oracle()?;
        let upload = &source["chain_cases"][0]["upload"];
        Ok(PreparedStyleImage {
            file_name: upload["file_name"].as_str().ok_or("missing name")?.into(),
            bytes: decode_hex(upload["bytes_hex"].as_str().ok_or("missing pixels")?)?,
            source_width: 2,
            source_height: 2,
            prepared_width: 2,
            prepared_height: 2,
        })
    }
    fn style_script(image: &PreparedStyleImage) -> Result<Vec<Response>, KreaError> {
        Ok(vec![response(
            "/proxy/krea/assets",
            HttpMethod::Post,
            multipart(image, "krea-source-boundary")?,
            serde_json::to_vec(
                &json!({"id":"asset-42","image_url":"https://asset.fixture.invalid/style.png","uploaded_at":"source-time"}),
            )?,
        )])
    }
    #[test]
    fn pinned_source_generation_matrix_and_preflight() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/krea/conformance.fixture"
        )))?;
        let cases = cases()?;
        assert_eq!(cases.len(), 103);
        for case in cases {
            let input: ImageInput = serde_json::from_value(case["input"].clone())?;
            let mut transport = TestTransport::new(script(&case)?);
            let result = invoke_image(&input, credentials(), 70_000, &mut transport)?;
            match result {
                SourceOutput::Image {
                    job_id,
                    url,
                    submission_receipt,
                    status_receipt,
                    download_receipt,
                    handle,
                    ..
                } => {
                    assert_eq!(job_id, "job-42");
                    assert_eq!(url, "https://output.fixture.invalid/first.png");
                    assert!(!submission_receipt.is_empty());
                    assert!(!status_receipt.is_empty());
                    assert!(!download_receipt.is_empty());
                    assert_eq!(handle.invocation, 42);
                }
                _ => return Err("wrong result".into()),
            }
            assert_eq!(transport.costs, 1);
            assert_eq!(
                transport.deadlines,
                vec![Some(3_600_000), Some(120_000), None]
            );
            transport.finish()?;
        }
        Ok(())
    }
    #[test]
    fn real_png_upload_and_chain_copy_match_source() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let image = prepared()?;
        assert!(image.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        for case in source["chain_cases"].as_array().ok_or("missing chains")? {
            let incoming = case["incoming"]
                .as_array()
                .ok_or("missing incoming")?
                .clone();
            let original = incoming.clone();
            let mut transport = TestTransport::new(style_script(&image)?);
            let result = invoke_style_reference(
                &image,
                case["strength"].as_f64().ok_or("strength")?,
                &incoming,
                "krea-source-boundary",
                credentials(),
                &mut transport,
            )?;
            match result {
                SourceOutput::StyleChain {
                    chain,
                    asset_url,
                    asset_receipt,
                    ..
                } => {
                    assert_eq!(json!(chain), case["output"]);
                    assert_eq!(asset_url, "https://asset.fixture.invalid/style.png");
                    assert!(!asset_receipt.is_empty());
                }
                _ => return Err("wrong result".into()),
            }
            assert_eq!(incoming, original);
            assert_eq!(transport.costs, 0);
            assert_eq!(transport.deadlines, vec![Some(3_600_000)]);
            transport.finish()?;
        }
        assert!(style_preparation_requirement(&vec![json!({}); 10]).is_err());
        let requirement = style_preparation_requirement(&[])?;
        assert_eq!(requirement.content_type, "image/png");
        assert_eq!(requirement.total_pixels, MAXIMUM_STYLE_PIXELS);
        assert!(requirement.first_batch_only);
        assert_eq!(requirement.resize_method, "lanczos");
        assert_eq!(requirement.crop, "disabled");
        let mut transport = TestTransport::new(vec![]);
        assert!(
            invoke_style_reference(
                &image,
                1.0,
                &vec![json!({}); 10],
                "krea-source-boundary",
                credentials(),
                &mut transport
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        assert_eq!(source_downscale(4097, 2049)?, (2894, 1448));
        Ok(())
    }
    #[test]
    fn source_exact_multipart_wire_diagnostic() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        assert_eq!(
            multipart(&prepared()?, "krea-source-boundary")?,
            decode_hex(
                source["multipart"]["bytes_hex"]
                    .as_str()
                    .ok_or("missing multipart")?
            )?
        );
        Ok(())
    }
    #[test]
    fn pinned_client_defaults_match_transport_policies() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let defaults = &source["client_defaults"];
        assert_eq!(
            defaults["sync_op"]["max_retries"],
            GENERATION_RETRY.ordinary_limit
        );
        assert_eq!(
            defaults["sync_op"]["timeout"]
                .as_f64()
                .ok_or("create deadline")?
                * 1000.0,
            GENERATION_RETRY.timeout_milliseconds.ok_or("deadline")? as f64
        );
        assert_eq!(
            defaults["poll_op"]["max_retries_per_poll"],
            POLL_RETRY.ordinary_limit
        );
        assert_eq!(defaults["poll_op"]["max_poll_attempts"], MAXIMUM_POLLS);
        assert_eq!(defaults["poll_op"]["poll_interval"], 5.0);
        assert_eq!(
            defaults["poll_op"]["retry_backoff_per_poll"],
            POLL_RETRY.backoff
        );
        assert_eq!(
            defaults["download_url_to_bytesio"]["max_retries"],
            DOWNLOAD_RETRY.ordinary_limit
        );
        assert_eq!(defaults["download_url_to_bytesio"]["timeout"], Value::Null);
        Ok(())
    }
    #[test]
    fn source_status_aliases_and_queue_override() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for status in source["status_constants"]["COMPLETED_STATUSES"]
            .as_array()
            .ok_or("completed")?
        {
            assert_eq!(task_status(&serde_json::to_vec(&json!({"job_id":"x","status":format!(" {} ",status.as_str().ok_or("status")?.to_uppercase()),"created_at":"x"}))?)?.1,Phase::Completed);
        }
        for status in source["status_constants"]["FAILED_STATUSES"]
            .as_array()
            .ok_or("failed")?
        {
            assert!(matches!(
                task_status(&serde_json::to_vec(
                    &json!({"job_id":"x","status":status,"created_at":"x"})
                )?),
                Err(KreaError::Provider(_))
            ));
        }
        for (status, phase) in [
            ("backlogged", Phase::Queued),
            ("queued", Phase::Queued),
            ("scheduled", Phase::Queued),
            ("in_queue", Phase::Processing),
            ("unknown", Phase::Processing),
        ] {
            assert_eq!(
                task_status(&serde_json::to_vec(
                    &json!({"job_id":"x","status":status,"created_at":"x"})
                )?)?
                .1,
                phase
            );
        }
        assert!(task_status(b"{\"job_id\":\"x\",\"status\":\"done\"}").is_err());
        Ok(())
    }
    fn exchange_script(statuses: &[u16]) -> TestTransport {
        TestTransport::new(
            statuses
                .iter()
                .map(|status| {
                    let mut response = response(
                        "/proxy/krea/assets",
                        HttpMethod::Post,
                        b"{}".to_vec(),
                        b"{}".to_vec(),
                    );
                    response.status = *status;
                    response
                })
                .collect(),
        )
    }
    #[test]
    fn bounded_retry_budgets_fresh_receipts_and_identity() -> Result<(), Box<dyn Error>> {
        let head = request_head("/proxy/krea/assets", HttpMethod::Post, credentials(), false)?;
        for (policy, limit, base) in [
            (ASSET_RETRY, 1, 2.0f64),
            (GENERATION_RETRY, 3, 2.0),
            (POLL_RETRY, 10, 1.4),
            (DOWNLOAD_RETRY, 5, 2.0),
        ] {
            let mut statuses = vec![500; limit];
            statuses.push(200);
            let mut transport = exchange_script(&statuses);
            exchange_with_policy(
                &head,
                b"{}",
                1024,
                policy,
                &mut Session::default(),
                &mut transport,
            )?;
            assert_eq!(
                transport.delays,
                (0..limit)
                    .map(|index| (1000.0 * base.powi(index as i32)).round() as u64)
                    .collect::<Vec<_>>()
            );
            transport.finish()?;
        }
        let mut transport = exchange_script(&[429; 17]);
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                1024,
                GENERATION_RETRY,
                &mut Session::default(),
                &mut transport
            ),
            Err(KreaError::Http { status: 429, .. })
        ));
        assert_eq!(transport.delays.len(), 16);
        transport.finish()?;
        let mut transport = exchange_script(&[429, 200]);
        transport
            .responses
            .front_mut()
            .ok_or("missing response")?
            .headers
            .push(Header {
                name: "Retry-After".into(),
                value: "900".into(),
            });
        exchange_with_policy(
            &head,
            b"{}",
            1024,
            GENERATION_RETRY,
            &mut Session::default(),
            &mut transport,
        )?;
        assert_eq!(transport.delays, vec![150_000]);
        let mut transport = exchange_script(&[500]);
        transport.stale_retry = true;
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                1024,
                GENERATION_RETRY,
                &mut Session::default(),
                &mut transport
            ),
            Err(KreaError::Transport(ConformanceError::InvalidHandle))
        ));
        let mut session = Session::default();
        let handle = StreamHandle {
            invocation: 42,
            slot: 1,
            generation: 1,
        };
        session.retain(handle)?;
        session.retain(StreamHandle { slot: 2, ..handle })?;
        assert!(session.retain(handle).is_err());
        assert!(
            session
                .retain(StreamHandle {
                    invocation: 43,
                    slot: 3,
                    ..handle
                })
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn missing_results_changed_identity_and_path_injection_fail() -> Result<(), Box<dyn Error>> {
        let case = cases()?.first().cloned().ok_or("missing case")?;
        let input: ImageInput = serde_json::from_value(case["input"].clone())?;
        for result in [Value::Null, json!({}), json!({"urls":[]})] {
            let mut responses = script(&case)?;
            responses.get_mut(1).ok_or("poll")?.bytes = serde_json::to_vec(
                &json!({"job_id":"job-42","status":"done","created_at":"x","result":result}),
            )?;
            let mut transport = TestTransport::new(responses);
            assert!(invoke_image(&input, credentials(), 70_000, &mut transport).is_err());
            assert_eq!(transport.heads.len(), 2);
        }
        for identifier in ["different", "../../untrusted"] {
            let mut responses = script(&case)?;
            let index = if identifier == "different" { 1 } else { 0 };
            let response = responses.get_mut(index).ok_or("response")?;
            let mut body: Value = serde_json::from_slice(&response.bytes)?;
            body["job_id"] = json!(identifier);
            response.bytes = serde_json::to_vec(&body)?;
            let mut transport = TestTransport::new(responses);
            assert!(invoke_image(&input, credentials(), 70_000, &mut transport).is_err());
            assert!(transport.heads.len() < 3);
        }
        Ok(())
    }
    #[test]
    fn processing_budget_counts_in_queue_but_excludes_krea_queued_states()
    -> Result<(), Box<dyn Error>> {
        let case = cases()?.first().cloned().ok_or("missing case")?;
        let input: ImageInput = serde_json::from_value(case["input"].clone())?;
        let mut responses = script(&case)?;
        let submitted = responses.remove(0);
        let mut script = vec![submitted];
        for index in 0..MAXIMUM_POLLS + 3 {
            script.push(response("/proxy/krea/jobs/job-42",HttpMethod::Get,vec![],serde_json::to_vec(&json!({"job_id":"job-42","status":if index<3{"scheduled"}else{"in_queue"},"created_at":"x"}))?));
        }
        let mut transport = TestTransport::new(script);
        assert!(matches!(
            invoke_image(&input, credentials(), 70_000, &mut transport),
            Err(KreaError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.delays.len(), 483);
        transport.finish()?;
        Ok(())
    }
    struct FailingMaterializer {
        calls: usize,
    }
    impl KreaMaterializer for FailingMaterializer {
        fn materialize(
            &mut self,
            _result: &SourceOutput,
            _expected: &ExpectedOutput,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            self.calls += 1;
            Err(ConformanceError::HostFailure)
        }
    }
    struct Sink {
        calls: usize,
    }
    impl MaterializationSink for Sink {
        fn publish(
            &mut self,
            _batch: crate::common::MaterializedBatch,
        ) -> Result<(), crate::common::MaterializationError> {
            self.calls += 1;
            Ok(())
        }
    }
    #[test]
    fn canonical_owner_failures_cancel_cost_and_bounds_publish_nothing()
    -> Result<(), Box<dyn Error>> {
        let case = cases()?.first().cloned().ok_or("missing case")?;
        let input: ImageInput = serde_json::from_value(case["input"].clone())?;
        let mut transport = TestTransport::new(script(&case)?);
        let result = invoke_image(&input, credentials(), 70_000, &mut transport)?;
        let registry = TypeRegistry::built_in()?;
        let mut owner = FailingMaterializer { calls: 0 };
        let mut sink = Sink { calls: 0 };
        assert!(
            materialize_and_publish(&result, &registry, &mut transport, &mut owner, &mut sink)
                .is_err()
        );
        assert_eq!(owner.calls, 1);
        assert_eq!(sink.calls, 0);
        transport.cancelled = true;
        assert!(matches!(
            materialize_and_publish(&result, &registry, &mut transport, &mut owner, &mut sink),
            Err(KreaError::Transport(ConformanceError::Cancelled))
        ));
        assert_eq!(owner.calls, 1);
        let mut transport = TestTransport::new(vec![]);
        let custom = SourceOutput::StyleChain {
            chain: vec![],
            asset_url: "https://asset.invalid/x".into(),
            asset_receipt: b"receipt".to_vec(),
            handle: StreamHandle {
                invocation: 42,
                slot: 1,
                generation: 1,
            },
        };
        assert!(
            materialize_and_publish(
                &custom,
                &TypeRegistry::from_registrations(&[])?,
                &mut transport,
                &mut owner,
                &mut sink
            )
            .is_err()
        );
        assert_eq!(owner.calls, 1);
        assert_eq!(sink.calls, 0);
        assert_eq!(
            registry.resolve(custom.alias())?.to_string(),
            "sim:custom-krea-style-ref@1"
        );
        assert_eq!(
            registry.family(registry.resolve(custom.alias())?)?,
            crate::common::ValueFamily::Model
        );
        assert!(
            materialize_and_publish(&custom, &registry, &mut transport, &mut owner, &mut sink)
                .is_err()
        );
        assert_eq!(owner.calls, 2);
        assert_eq!(sink.calls, 0);
        for error in [ConformanceError::Cancelled, ConformanceError::WorkerLost] {
            let mut transport = TestTransport::new(script(&case)?);
            transport.failure = Some(error.clone());
            assert!(
                matches!(invoke_image(&input,credentials(),70_000,&mut transport),Err(KreaError::Transport(actual)) if actual==error)
            );
            assert!(transport.heads.is_empty());
        }
        let mut transport = TestTransport::new(script(&case)?);
        transport.reject_cost = true;
        assert!(matches!(
            invoke_image(&input, credentials(), 70_000, &mut transport),
            Err(KreaError::Transport(ConformanceError::InvalidCost))
        ));
        assert_eq!(transport.heads.len(), 1);
        let mut image = prepared()?;
        image.prepared_width = 3;
        assert!(multipart(&image, "boundary").is_err());
        image.prepared_width = 2;
        image.file_name = "bad\r\n.png".into();
        assert!(multipart(&image, "boundary").is_err());
        let head = request_head("/proxy/krea/assets", HttpMethod::Post, credentials(), false)?;
        for (bytes, receipt, expected) in [
            (
                vec![0; MAXIMUM_CHUNK_BYTES + 1],
                b"receipt".to_vec(),
                ConformanceError::ChunkLimit,
            ),
            (b"{}".to_vec(), vec![], ConformanceError::InvalidResponse),
        ] {
            let mut transport = exchange_script(&[200]);
            let response = transport.responses.front_mut().ok_or("response")?;
            response.bytes = bytes;
            response.receipt = receipt;
            assert!(
                matches!(exchange_with_policy(&head,b"{}",MAXIMUM_MEDIA_BYTES,GENERATION_RETRY,&mut Session::default(),&mut transport),Err(KreaError::Transport(actual)) if actual==expected)
            );
        }
        Ok(())
    }

    fn phase_input() -> ImageInput {
        ImageInput {
            prompt: " keep prompt ".into(),
            model: json!({"model":"Krea 2 Medium","aspect_ratio":"1:1","resolution":"1K","creativity":"medium"}),
            seed: 42,
        }
    }
    #[test]
    fn krea_optional_admission_precedes_base_request_fields() -> Result<(), Box<dyn Error>> {
        for (field, updates, expected) in [
            (
                "aspect_ratio",
                json!({"moodboard_id":"bad"}),
                "Moodboard ID must be a UUID",
            ),
            (
                "resolution",
                json!({"moodboard_id":"01234567-89ab-cdef-0123-456789abcdef","moodboard_strength":2}),
                "Invalid moodboard strength",
            ),
            (
                "aspect_ratio",
                json!({"style_reference":[{"url":"https://example.invalid/a","strength":"bad"}]}),
                "Invalid source float",
            ),
        ] {
            let mut input = phase_input();
            let model = input.model.as_object_mut().ok_or("model")?;
            model.remove(field);
            model.extend(updates.as_object().ok_or("updates")?.clone());
            assert!(matches!(request(&input), Err(KreaError::Input(actual)) if actual == expected));
        }
        let mut input = phase_input();
        let model = input.model.as_object_mut().ok_or("model")?;
        model.remove("creativity");
        model.insert(
            "style_reference".into(),
            json!(vec![json!({"url":null,"strength":0.2}); 11]),
        );
        assert!(matches!(
            request(&input),
            Err(KreaError::Input(
                "Krea permits at most ten style references"
            ))
        ));
        Ok(())
    }
    #[test]
    fn krea_ordered_admission_errors_stop_before_transport_and_cost() -> Result<(), Box<dyn Error>>
    {
        let mut input = phase_input();
        let model = input.model.as_object_mut().ok_or("model")?;
        model.remove("aspect_ratio");
        model.insert("moodboard_id".into(), json!("bad"));
        let mut transport = TestTransport::new(vec![]);
        assert!(matches!(
            invoke_image(&input, credentials(), 70_000, &mut transport),
            Err(KreaError::Input("Moodboard ID must be a UUID"))
        ));
        assert!(transport.heads.is_empty());
        assert_eq!(transport.costs, 0);
        assert_eq!(transport.next_slot, 1);
        assert!(transport.active.is_none());
        input.prompt.clear();
        assert!(matches!(
            request(&input),
            Err(KreaError::Input("Prompt must have at least one character"))
        ));
        input.prompt = "prompt".into();
        input.model["model"] = json!("unknown");
        assert!(matches!(
            request(&input),
            Err(KreaError::Input("Unknown Krea model"))
        ));
        input.model.as_object_mut().ok_or("model")?.remove("model");
        assert!(matches!(
            request(&input),
            Err(KreaError::Input("Missing source string field"))
        ));
        Ok(())
    }
    #[test]
    fn krea_valid_optional_requests_preserve_raw_bytes_and_base_lookup_failure()
    -> Result<(), Box<dyn Error>> {
        let mut input = phase_input();
        input.model["moodboard_id"] = json!("01234567-89ab-cdef-0123-456789abcdef");
        input.model["style_reference"] = json!([{"url":null,"strength":0.2}]);
        let (path, bytes) = request(&input)?;
        assert_eq!(path, "/proxy/krea/generate/image/krea/krea-2/medium");
        let body: Value = serde_json::from_slice(&bytes)?;
        let expected = json!({"prompt":" keep prompt ","aspect_ratio":"1:1","resolution":"1K","seed":42,"creativity":"medium","moodboards":[{"id":"01234567-89ab-cdef-0123-456789abcdef","strength":0.35}],"image_style_references":[{"strength":0.2}]});
        assert_eq!(bytes, serde_json::to_vec(&expected)?);
        assert_eq!(
            body,
            json!({"prompt":" keep prompt ","seed":42,"aspect_ratio":"1:1","resolution":"1K","creativity":"medium","moodboards":[{"id":"01234567-89ab-cdef-0123-456789abcdef","strength":0.35}],"image_style_references":[{"strength":0.2}]})
        );
        input
            .model
            .as_object_mut()
            .ok_or("model")?
            .remove("aspect_ratio");
        assert!(matches!(
            request(&input),
            Err(KreaError::Input("Missing source string field"))
        ));
        let case = cases()?.into_iter().next().ok_or("source case")?;
        let valid: ImageInput = serde_json::from_value(case["input"].clone())?;
        let mut transport = TestTransport::new(script(&case)?);
        let result = invoke_image(&valid, credentials(), 70_000, &mut transport)?;
        match result {
            SourceOutput::Image {
                job_id,
                url,
                submission_receipt,
                status_receipt,
                download_receipt,
                ..
            } => {
                assert_eq!(job_id, "job-42");
                assert_eq!(url, "https://output.fixture.invalid/first.png");
                assert!(!submission_receipt.is_empty());
                assert!(!status_receipt.is_empty());
                assert!(!download_receipt.is_empty());
            }
            _ => return Err("wrong source output".into()),
        }
        assert_eq!(transport.costs, 1);
        assert_eq!(
            transport.deadlines,
            vec![Some(3_600_000), Some(120_000), None]
        );
        transport.finish()?;
        Ok(())
    }
}
