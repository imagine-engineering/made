use std::{collections::BTreeSet, error::Error, fmt};

use crate::common::{
    ConformanceError, CostRequest, Header, HttpMethod, ProviderTransport, RequestHead,
    ResponseEvent, StreamHandle,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const NAMESPACE: &str = "zed.sim.provider.bfl";
pub const SOURCE_SHA256: &str = "78f943d92611ef196cb9e01da6ac686be815e270650760abc168f3c7c525d643";
pub const MAXIMUM_REQUEST_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_IMAGE_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
pub const MAXIMUM_POLLS: u32 = 480;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0174",
        node_identifier: "Flux2ImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0175",
        node_identifier: "Flux2MaxImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0176",
        node_identifier: "Flux2ProImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0179",
        node_identifier: "FluxEraseNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0182",
        node_identifier: "FluxKontextMaxImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0184",
        node_identifier: "FluxKontextProImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0186",
        node_identifier: "FluxProExpandNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0187",
        node_identifier: "FluxProFillNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0188",
        node_identifier: "FluxProUltraImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0189",
        node_identifier: "FluxVTONode",
    },
];
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
}
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0005",
        method: "POST",
        path: "/proxy/bfl/flux-2-max/generate",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0006",
        method: "POST",
        path: "/proxy/bfl/flux-2-pro/generate",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0007",
        method: "POST",
        path: "/proxy/bfl/flux-kontext-max/generate",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0008",
        method: "POST",
        path: "/proxy/bfl/flux-kontext-pro/generate",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0009",
        method: "POST",
        path: "/proxy/bfl/flux-pro-1.0-expand/generate",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0010",
        method: "POST",
        path: "/proxy/bfl/flux-pro-1.0-fill/generate",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0011",
        method: "POST",
        path: "/proxy/bfl/flux-pro-1.1-ultra/generate",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0012",
        method: "POST",
        path: "/proxy/bfl/v1/flux-tools/erase-v1",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0013",
        method: "POST",
        path: "/proxy/bfl/v1/flux-tools/vto-v1",
    },
];

#[derive(Debug)]
pub enum BflError {
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
impl fmt::Display for BflError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "BFL task failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "BFL HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for BflError {}
impl From<serde_json::Error> for BflError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
impl From<ConformanceError> for BflError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: String,
    pub guidance: f64,
    pub steps: u32,
    pub prompt_upsampling: bool,
    pub raw: bool,
    pub image_prompt_strength: f64,
    pub top: u32,
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
    pub dilate_pixels: u32,
    pub model: String,
    pub images: Vec<String>,
    pub mask: Option<String>,
}

pub fn request(input: &Input) -> Result<(&'static str, Vec<u8>), BflError> {
    if input.prompt.len() > 65_536 || input.images.len() > 9 {
        return Err(BflError::Input("BFL input exceeds its bound"));
    }
    let encoded_bytes = input
        .images
        .iter()
        .chain(input.mask.iter())
        .try_fold(input.prompt.len(), |total, image| {
            total.checked_add(image.len())
        })
        .ok_or(BflError::Input("BFL media bytes overflow"))?;
    if encoded_bytes > MAXIMUM_REQUEST_BYTES {
        return Err(BflError::Input(
            "BFL media bytes exceed their aggregate bound",
        ));
    }
    for image in input.images.iter().chain(input.mask.iter()) {
        let padding = image.bytes().rev().take_while(|byte| *byte == b'=').count();
        if padding > 2
            || image.trim_end_matches('=').contains('=')
            || image.trim_end_matches('=').is_empty()
            || image.is_empty()
            || image.len() > MAXIMUM_REQUEST_BYTES
            || !image.len().is_multiple_of(4)
            || !image
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='))
        {
            return Err(BflError::Input("Invalid canonical base64 image input"));
        }
    }
    let mut body = json!({"prompt":input.prompt,"seed":input.seed,"output_format":"png"});
    let image = |index| {
        input
            .images
            .get(index)
            .cloned()
            .ok_or(BflError::Input("Missing BFL image"))
    };
    let path = match input.node.as_str() {
        "FluxProUltraImageNode" => {
            aspect_ratio(&input.aspect_ratio)?;
            if input.images.len() > 1
                || !input.image_prompt_strength.is_finite()
                || !(0.0..=1.0).contains(&input.image_prompt_strength)
            {
                return Err(BflError::Input("Invalid BFL Ultra image input"));
            }
            body["prompt_upsampling"] = input.prompt_upsampling.into();
            body["raw"] = input.raw.into();
            body["aspect_ratio"] = input.aspect_ratio.clone().into();
            body["safety_tolerance"] = 6.into();
            if !input.images.is_empty() {
                body["image_prompt"] = image(0)?.into();
                body["image_prompt_strength"] =
                    decimal_round(input.image_prompt_strength, 2)?.into();
            }
            ROUTES[6].path
        }
        "FluxKontextProImageNode" | "FluxKontextMaxImageNode" => {
            aspect_ratio(&input.aspect_ratio)?;
            if input.images.len() > 1
                || !input.guidance.is_finite()
                || !(0.1..=99.0).contains(&input.guidance)
                || !(1..=150).contains(&input.steps)
            {
                return Err(BflError::Input("Invalid BFL Kontext controls"));
            }
            body["prompt_upsampling"] = input.prompt_upsampling.into();
            body["guidance"] = decimal_round(input.guidance, 1)?.into();
            body["steps"] = input.steps.into();
            body["aspect_ratio"] = input.aspect_ratio.clone().into();
            body["safety_tolerance"] = 2.into();
            if !input.images.is_empty() {
                body["input_image"] = image(0)?.into();
            }
            if input.node == "FluxKontextProImageNode" {
                ROUTES[3].path
            } else {
                ROUTES[2].path
            }
        }
        "FluxProExpandNode" | "FluxProFillNode" => {
            if input.images.len() != 1
                || !input.guidance.is_finite()
                || !(1.5..=100.0).contains(&input.guidance)
                || !(15..=50).contains(&input.steps)
            {
                return Err(BflError::Input("Invalid BFL Fill/Expand controls"));
            }
            body["image"] = image(0)?.into();
            body["prompt_upsampling"] = input.prompt_upsampling.into();
            body["guidance"] = input.guidance.into();
            body["steps"] = input.steps.into();
            body["safety_tolerance"] = 6.into();
            if input.node == "FluxProExpandNode" {
                if [input.top, input.bottom, input.left, input.right]
                    .into_iter()
                    .any(|value| value > 2048)
                {
                    return Err(BflError::Input("Invalid BFL expansion extent"));
                }
                body["top"] = input.top.into();
                body["bottom"] = input.bottom.into();
                body["left"] = input.left.into();
                body["right"] = input.right.into();
                ROUTES[4].path
            } else {
                body["mask"] = input
                    .mask
                    .clone()
                    .ok_or(BflError::Input("Missing BFL fill mask"))?
                    .into();
                ROUTES[5].path
            }
        }
        "FluxEraseNode" => {
            if input.images.len() != 1 || input.dilate_pixels > 25 || input.seed > 2_147_483_647 {
                return Err(BflError::Input("Invalid BFL erase controls"));
            }
            body.as_object_mut()
                .ok_or(BflError::Input("Invalid request object"))?
                .remove("prompt");
            body["image"] = image(0)?.into();
            body["mask"] = input
                .mask
                .clone()
                .ok_or(BflError::Input("Missing BFL erase mask"))?
                .into();
            body["dilate_pixels"] = input.dilate_pixels.into();
            ROUTES[7].path
        }
        "FluxVTONode" => {
            if input.images.len() != 2 {
                return Err(BflError::Input("BFL try-on requires person and garment"));
            }
            body["person"] = image(0)?.into();
            body["garment"] = image(1)?.into();
            body["safety_tolerance"] = 5.into();
            ROUTES[8].path
        }
        "Flux2ImageNode" | "Flux2ProImageNode" | "Flux2MaxImageNode" => {
            if !(256..=2048).contains(&input.width)
                || !(256..=2048).contains(&input.height)
                || !input.width.is_multiple_of(32)
                || !input.height.is_multiple_of(32)
            {
                return Err(BflError::Input("Invalid BFL Flux2 dimensions"));
            }
            if input.node == "Flux2ImageNode" && input.images.len() > 8 {
                return Err(BflError::Input(
                    "The current maximum number of supported images is 8.",
                ));
            }
            body["width"] = input.width.into();
            body["height"] = input.height.into();
            body["safety_tolerance"] = 5.into();
            if input.node != "Flux2ImageNode" {
                body["prompt_upsampling"] = input.prompt_upsampling.into();
            }
            for (index, image) in input.images.iter().enumerate() {
                let key = if index == 0 {
                    "input_image".into()
                } else {
                    format!("input_image_{}", index + 1)
                };
                body[key] = image.clone().into();
            }
            match (input.node.as_str(), input.model.as_str()) {
                ("Flux2MaxImageNode", _) | ("Flux2ImageNode", "Flux.2 [max]") => ROUTES[0].path,
                ("Flux2ProImageNode", _) | ("Flux2ImageNode", "Flux.2 [pro]") => ROUTES[1].path,
                _ => return Err(BflError::Input("Invalid BFL Flux2 model")),
            }
        }
        _ => return Err(BflError::Input("Unknown BFL node")),
    };
    let bytes = serde_json::to_vec(&body)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(BflError::Input("BFL request exceeds its bound"));
    }
    Ok((path, bytes))
}
fn decimal_round(value: f64, digits: usize) -> Result<f64, BflError> {
    format!("{value:.digits$}")
        .parse()
        .map_err(|_| BflError::Input("Invalid decimal input"))
}
fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{001c}'..='\u{001f}')
}
fn aspect_ratio_integer(value: &str) -> Result<u64, BflError> {
    let value = value.trim_matches(python_whitespace);
    let value = value.strip_prefix('+').unwrap_or(value);
    let mut integer = 0_u64;
    let mut previous_was_digit = false;
    for character in value.bytes() {
        if character.is_ascii_digit() {
            integer = integer
                .checked_mul(10)
                .and_then(|integer| integer.checked_add(u64::from(character - b'0')))
                .ok_or(BflError::Input("Invalid aspect ratio"))?;
            previous_was_digit = true;
        } else if character == b'_' && previous_was_digit {
            previous_was_digit = false;
        } else {
            return Err(BflError::Input("Invalid aspect ratio"));
        }
    }
    if !previous_was_digit {
        return Err(BflError::Input("Invalid aspect ratio"));
    }
    Ok(integer)
}
fn aspect_ratio(value: &str) -> Result<(), BflError> {
    let (width, height) = value
        .split_once(':')
        .ok_or(BflError::Input("Invalid aspect ratio"))?;
    let width = aspect_ratio_integer(width)?;
    let height = aspect_ratio_integer(height)?;
    let ratio = crate::common::positive_integer_ratio_to_f64(width, height)
        .map_err(|_| BflError::Input("Invalid aspect ratio"))?;
    if !(0.25..=4.0).contains(&ratio) {
        return Err(BflError::Input("Invalid aspect ratio"));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct SubmittedTask {
    pub id: String,
    pub polling_url: String,
    pub cost: Option<f64>,
}
#[derive(Debug, Deserialize)]
pub struct TaskStatus {
    pub id: String,
    pub status: String,
    pub result: Option<serde_json::Map<String, Value>>,
    pub progress: Option<f64>,
}
pub fn submitted_task(bytes: &[u8]) -> Result<SubmittedTask, BflError> {
    bounded_response(bytes)?;
    let task: SubmittedTask = serde_json::from_slice(bytes)?;
    if task.id.is_empty()
        || task.id.chars().any(char::is_control)
        || task.id.len() > 1024
        || task
            .cost
            .is_some_and(|cost| !cost.is_finite() || cost < 0.0)
    {
        return Err(BflError::Response("Invalid BFL submission"));
    }
    endpoint(&task.polling_url)?;
    Ok(task)
}
pub fn task_status(bytes: &[u8]) -> Result<TaskStatus, BflError> {
    bounded_response(bytes)?;
    let task: TaskStatus = serde_json::from_slice(bytes)?;
    if task.id.is_empty()
        || task.id.chars().any(char::is_control)
        || task.id.len() > 1024
        || task
            .progress
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(BflError::Response("Invalid BFL status"));
    }
    match task.status.as_str() {
        "Pending" => {}
        "Ready" => {
            let relative = endpoint(
                task.result
                    .as_ref()
                    .and_then(|result| result.get("sample"))
                    .and_then(Value::as_str)
                    .ok_or(BflError::Response("Missing BFL image URL"))?,
            )?;
            if relative {
                return Err(BflError::Response(
                    "BFL image downloads require an absolute URL",
                ));
            }
        }
        "Request Moderated" | "Content Moderated" | "Error" | "Task not found" => {
            return Err(BflError::Provider(task.status));
        }
        _ => return Err(BflError::Response("Unknown BFL task state")),
    }
    Ok(task)
}
fn bounded_response(bytes: &[u8]) -> Result<(), BflError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(BflError::Response("BFL response exceeds its bound"));
    }
    Ok(())
}
fn endpoint(value: &str) -> Result<bool, BflError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(BflError::Response("Invalid BFL response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(BflError::Response("Invalid BFL response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(BflError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(BflError::Response("Invalid BFL endpoint authority"));
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
) -> Result<RequestHead, BflError> {
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
            return Err(BflError::Input("Missing Comfy API credentials"));
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
#[derive(Debug)]
pub struct DownloadedImage {
    pub bytes: Vec<u8>,
    pub response_receipt: Vec<u8>,
    pub source_url: String,
    pub authoritative_cost_usd: Option<f64>,
}

pub trait BflTransport: ProviderTransport {
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
    fn report_source_progress(
        &mut self,
        handle: StreamHandle,
        sequence: u64,
        fraction: f64,
        status: &str,
    ) -> Result<(), ConformanceError>;
}

pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn BflTransport,
) -> Result<DownloadedImage, BflError> {
    let (path, body) = request(input)?;
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let (submission, _, _submission_handle) = exchange(
        &head,
        &body,
        Some(maximum_cost_microunits),
        MAXIMUM_RESPONSE_BYTES,
        transport,
    )?;
    let task = submitted_task(&submission)?;
    if task
        .cost
        .is_some_and(|cost| cost / 100.0 * 1_000_000.0 > maximum_cost_microunits as f64)
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    let mut last_progress = None;
    for poll in 0..MAXIMUM_POLLS {
        let head = request_head(&task.polling_url, HttpMethod::Get, credentials, false)?;
        let (bytes, _, poll_handle) = exchange_with_policy(
            &head,
            &[],
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            transport,
        )?;
        let status = task_status(&bytes)?;
        if status.id != task.id {
            return Err(BflError::Response("BFL poll returned a different task"));
        }
        if let Some(progress) = status
            .progress
            .filter(|progress| Some(*progress) != last_progress)
        {
            transport.check_cancelled(Some(poll_handle))?;
            transport.report_source_progress(
                poll_handle,
                u64::from(poll) * 2,
                progress,
                &status.status,
            )?;
            last_progress = status.progress;
        }
        if status.status == "Ready" {
            transport.check_cancelled(Some(poll_handle))?;
            transport.report_source_progress(
                poll_handle,
                u64::from(poll) * 2 + 1,
                100.0,
                &status.status,
            )?;
            let url = status
                .result
                .as_ref()
                .and_then(|result| result.get("sample"))
                .and_then(Value::as_str)
                .ok_or(BflError::Response("Missing BFL sample"))?;
            let head = request_head(url, HttpMethod::Get, credentials, true)?;
            let (bytes, receipt, _) = exchange_with_policy(
                &head,
                &[],
                None,
                MAXIMUM_IMAGE_BYTES,
                DOWNLOAD_RETRY,
                transport,
            )?;
            if bytes.is_empty() {
                return Err(BflError::Response("Empty BFL image download"));
            }
            return Ok(DownloadedImage {
                bytes,
                response_receipt: receipt,
                source_url: url.into(),
                authoritative_cost_usd: task.cost.map(|cost| cost / 100.0),
            });
        }
        transport.check_cancelled(Some(poll_handle))?;
        transport.delay(poll_handle, 5000)?;
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
fn exchange(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    transport: &mut dyn BflTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BflError> {
    exchange_with_policy(head, body, cost, maximum_bytes, GENERATION_RETRY, transport)
}
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn BflTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BflError> {
    transport.check_cancelled(None)?;
    let mut handle = transport.start_request(head)?;
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(BflError::Http {
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
                    return Err(BflError::Http {
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
    transport: &mut dyn BflTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), BflError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "bfl.image.generation.maximum".into(),
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
                    return Err(BflError::Http {
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

pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":[],"nodes":NODES,"routes":ROUTES,"scope":"source request and bounded response transport; canonical media integration pending", "source_ratio_validation": {
            "python_strip_control_separators":[28,29,30,31],
            "preserve_aspect_ratio_and_prompt_bytes":true,
            "nodes":["FluxProUltraImageNode","FluxKontextProImageNode","FluxKontextMaxImageNode"],
            "validation_utils_sha256":"77554cc809f651f54943f73e6391b8899b3a52779d949edfc08166d8fdf86c66",
            "underscore_integer_within_u64":true,
            "broader_python_integer_admission":"pending"
        }})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let oracle: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/bfl/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)?;
    if oracle["source_sha256"]["comfy_api_nodes/nodes_bfl.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let cases = oracle["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?;
    let mut covered = BTreeSet::new();
    for case in cases {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let (path, bytes) = request(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let body: Value =
            serde_json::from_slice(&bytes).map_err(|_| ConformanceError::InvalidRequest)?;
        if case["request"]["path"] != path
            || case["request"]["method"] != "POST"
            || case["request"]["body"] != body
        {
            return Err(ConformanceError::InvalidRequest);
        }
        covered.insert(input.node);
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

pub trait BflImageMaterializer {
    fn decode_image(
        &mut self,
        image: &DownloadedImage,
    ) -> Result<crate::common::EncodedOutput, ConformanceError>;
}

pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn BflTransport,
    materializer: &mut dyn BflImageMaterializer,
    sink: &mut dyn crate::common::MaterializationSink,
) -> Result<(), BflError> {
    let image = invoke(input, credentials, maximum_cost_microunits, transport)?;
    transport.check_cancelled(None)?;
    let output = materializer.decode_image(&image)?;
    let registry = crate::common::TypeRegistry::built_in()
        .map_err(|_| BflError::Response("Invalid built-in types"))?;
    let image_type = registry
        .resolve("IMAGE")
        .map_err(|_| BflError::Response("Missing image type"))?
        .clone();
    let batch = crate::common::validate_outputs(
        &[crate::common::ExpectedOutput {
            port_id: "output_0".into(),
            type_id: image_type,
            family: crate::common::ValueFamily::Tensor,
        }],
        vec![output],
        MAXIMUM_IMAGE_BYTES,
    )
    .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    crate::common::publish_validated(batch, sink).map_err(ConformanceError::from)?;
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
        progress: Vec<f64>,
    }
    impl TestTransport {
        fn new(steps: Vec<ScriptStep>) -> Self {
            Self {
                actuator: ScriptedActuator::new(steps),
                head: None,
                delays: vec![],
                progress: vec![],
            }
        }
        fn finish(self) -> Result<Vec<ScriptStep>, ConformanceError> {
            self.actuator.finish()
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
    impl BflTransport for TestTransport {
        fn retry_after_delay(
            &mut self,
            header: &str,
            fallback: u64,
        ) -> Result<u64, ConformanceError> {
            Ok(header
                .parse::<f64>()
                .ok()
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                .map(|seconds| (seconds.min(150.0) * 1000.0).ceil() as u64)
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
        fn delay(&mut self, _handle: StreamHandle, delay: u64) -> Result<(), ConformanceError> {
            self.delays.push(delay);
            Ok(())
        }
        fn report_source_progress(
            &mut self,
            _handle: StreamHandle,
            _sequence: u64,
            fraction: f64,
            _status: &str,
        ) -> Result<(), ConformanceError> {
            self.progress.push(fraction);
            Ok(())
        }
    }
    fn fixture_input() -> Result<Input, Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/source-contract.json"
        )))?;
        Ok(serde_json::from_value(
            fixture["cases"][0]["input"].clone(),
        )?)
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn handle(slot: u32, generation: u32) -> StreamHandle {
        StreamHandle {
            invocation: 42,
            slot,
            generation,
        }
    }
    fn script_exchange(
        head: RequestHead,
        body: Vec<u8>,
        cost: Option<u64>,
        response: Vec<u8>,
        handle: StreamHandle,
    ) -> Vec<ScriptStep> {
        let ready = serde_json::from_slice::<Value>(&response)
            .ok()
            .is_some_and(|value| value["status"] == "Ready");
        let mut steps = vec![
            ScriptStep::CheckCancelled(None),
            ScriptStep::Start(head, Ok(handle)),
        ];
        if let Some(maximum_microunits) = cost {
            steps.push(ScriptStep::Cost(
                handle,
                CostRequest {
                    operation: "bfl.image.generation.maximum".into(),
                    currency: "USD".into(),
                    maximum_microunits,
                },
                CostResponse {
                    accepted: true,
                    approved_microunits: maximum_microunits,
                    receipt: b"approved-cost".to_vec(),
                },
            ));
        }
        steps.extend([
            ScriptStep::CheckCancelled(Some(handle)),
            ScriptStep::RequestChunk {
                handle,
                sequence: 0,
                bytes: body,
                end: true,
            },
        ]);
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
                ResponseEvent::Completed(b"terminal-receipt".to_vec()),
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
        if ready {
            steps.push(ScriptStep::CheckCancelled(Some(handle)));
        }
        steps
    }
    #[test]
    fn all_bfl_node_and_route_claims_match_executed_pinned_source() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 10);
        assert_eq!(ROUTES.len(), 9);
        assert_eq!(
            NODES
                .iter()
                .map(|claim| claim.feature_id)
                .collect::<BTreeSet<_>>()
                .len(),
            10
        );
        assert_eq!(
            ROUTES
                .iter()
                .map(|claim| claim.path)
                .collect::<BTreeSet<_>>()
                .len(),
            9
        );
        Ok(())
    }
    #[test]
    fn bfl_reference_limits_defaults_and_decimal_rounding_are_source_exact()
    -> Result<(), Box<dyn Error>> {
        let mut input = fixture_input()?;
        assert_eq!(decimal_round(2.675, 2)?, 2.67);
        assert_eq!(decimal_round(3.15, 1)?, 3.1);
        assert_eq!(decimal_round(0.125, 2)?, 0.12);
        input.node = "Flux2ImageNode".into();
        input.images = vec!["aW1hZ2U=".into(); 9];
        assert!(request(&input).is_err());
        input.images.pop();
        let (_, bytes) = request(&input)?;
        let body: Value = serde_json::from_slice(&bytes)?;
        assert!(body.get("prompt_upsampling").is_none());
        assert!(body.get("input_image_8").is_some());
        assert!(body.get("input_image_9").is_none());
        input.node = "Flux2ProImageNode".into();
        input.images.push("aW1hZ2U=".into());
        assert!(request(&input).is_ok());
        input.images.push("aW1hZ2U=".into());
        assert!(request(&input).is_err());
        Ok(())
    }
    #[test]
    fn bfl_python_strip_ratio_keeps_original_request_and_prompt_bytes() -> Result<(), Box<dyn Error>>
    {
        let oracle: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/source-contract.json"
        )))?;
        let cases = oracle["cases"].as_array().ok_or("Missing cases")?;
        for node in [
            "FluxProUltraImageNode",
            "FluxKontextProImageNode",
            "FluxKontextMaxImageNode",
        ] {
            let case = cases
                .iter()
                .find(|case| case["input"]["node"] == node)
                .ok_or("Missing node")?;
            for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
                let ratio = format!("{separator}1{separator}:{separator}1{separator}");
                for original_prompt in [
                    "",
                    " \u{001c}\u{001f} ",
                    "\u{001c} original prompt \u{001f}",
                    "\u{200b}",
                    "\u{feff}",
                ] {
                    let mut input: Input = serde_json::from_value(case["input"].clone())?;
                    input.aspect_ratio = ratio.clone();
                    input.prompt = original_prompt.into();
                    let mut expected = case["request"]["body"].clone();
                    expected["aspect_ratio"] = ratio.clone().into();
                    expected["prompt"] = original_prompt.into();
                    let (path, body) = request(&input)?;
                    assert_eq!(
                        path,
                        case["request"]["path"].as_str().ok_or("Missing path")?
                    );
                    assert_eq!(serde_json::from_slice::<Value>(&body)?, expected);
                }
                let mut input: Input = serde_json::from_value(case["input"].clone())?;
                input.aspect_ratio = ratio.clone();
                let mut expected = case["request"]["body"].clone();
                expected["aspect_ratio"] = ratio.into();
                let (path, canonical_body) = request(&input)?;
                assert_eq!(serde_json::from_slice::<Value>(&canonical_body)?, expected);
                assert_eq!(
                    path,
                    case["request"]["path"].as_str().ok_or("Missing path")?
                );
                let mut steps = script_exchange(
                    request_head(
                        case["request"]["path"].as_str().ok_or("Missing path")?,
                        HttpMethod::Post,
                        credentials(),
                        false,
                    )?,
                    canonical_body,
                    Some(100_000),
                    br#"{"id":"task","polling_url":"https://poll.fixture.invalid/task"}"#.to_vec(),
                    handle(0, 1),
                );
                steps.extend(script_exchange(
                    request_head("https://poll.fixture.invalid/task", HttpMethod::Get, credentials(), false)?,
                    vec![], None,
                    br#"{"id":"task","status":"Ready","result":{"sample":"https://download.fixture.invalid/image.png"}}"#.to_vec(), handle(1, 1),
                ));
                let image = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
                ))
                .to_vec();
                steps.extend(script_exchange(
                    request_head(
                        "https://download.fixture.invalid/image.png",
                        HttpMethod::Get,
                        credentials(),
                        true,
                    )?,
                    vec![],
                    None,
                    image.clone(),
                    handle(2, 1),
                ));
                let mut transport = TestTransport::new(steps);
                let result = invoke(&input, credentials(), 100_000, &mut transport)?;
                transport.finish()?;
                assert_eq!(result.bytes, image);
                assert_eq!(result.response_receipt, b"terminal-receipt");
            }
        }
        Ok(())
    }

    #[test]
    fn bfl_source_underscore_integer_requests_preserve_bytes_and_execute()
    -> Result<(), Box<dyn Error>> {
        let oracle: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/source-contract.json"
        )))?;
        let cases = oracle["cases"].as_array().ok_or("Missing cases")?;
        for node in [
            "FluxProUltraImageNode",
            "FluxKontextProImageNode",
            "FluxKontextMaxImageNode",
        ] {
            let case = cases
                .iter()
                .find(|case| case["input"]["node"] == node)
                .ok_or("Missing node")?;
            for ratio in [
                "1_0:10",
                "1_0:1_0",
                "+1_0:+1_0",
                "0_1:1",
                "\u{001c}+0_1\u{001f}:1",
                "1:0_4",
                "0_4:1",
                "18_446_744_073_709_551_615:18_446_744_073_709_551_615",
                "1_000:250",
                "250:1_000",
            ] {
                let mut input: Input = serde_json::from_value(case["input"].clone())?;
                input.aspect_ratio = ratio.into();
                input.prompt = " \u{001c} original prompt \u{001f} ".into();
                let mut expected = case["request"]["body"].clone();
                expected["aspect_ratio"] = ratio.into();
                expected["prompt"] = input.prompt.clone().into();
                let (path, body) = request(&input)?;
                assert_eq!(
                    path,
                    case["request"]["path"].as_str().ok_or("Missing path")?
                );
                assert_eq!(serde_json::from_slice::<Value>(&body)?, expected);
            }
            let mut input: Input = serde_json::from_value(case["input"].clone())?;
            input.aspect_ratio = "+1_0:1_0".into();
            let (path, body) = request(&input)?;
            let mut expected = case["request"]["body"].clone();
            expected["aspect_ratio"] = input.aspect_ratio.clone().into();
            assert_eq!(serde_json::from_slice::<Value>(&body)?, expected);
            assert_eq!(
                path,
                case["request"]["path"].as_str().ok_or("Missing path")?
            );
            let mut steps = script_exchange(
                request_head(path, HttpMethod::Post, credentials(), false)?,
                body,
                Some(100_000),
                br#"{"id":"task","polling_url":"https://poll.fixture.invalid/task"}"#.to_vec(),
                handle(0, 1),
            );
            steps.extend(script_exchange(request_head("https://poll.fixture.invalid/task", HttpMethod::Get, credentials(), false)?, vec![], None, br#"{"id":"task","status":"Ready","result":{"sample":"https://download.fixture.invalid/image.png"}}"#.to_vec(), handle(1, 1)));
            let image = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
            ))
            .to_vec();
            steps.extend(script_exchange(
                request_head(
                    "https://download.fixture.invalid/image.png",
                    HttpMethod::Get,
                    credentials(),
                    true,
                )?,
                vec![],
                None,
                image.clone(),
                handle(2, 1),
            ));
            let mut transport = TestTransport::new(steps);
            let result = invoke(&input, credentials(), 100_000, &mut transport)?;
            transport.finish()?;
            assert_eq!(result.bytes, image);
            assert_eq!(result.response_receipt, b"terminal-receipt");
        }
        Ok(())
    }

    #[test]
    fn bfl_source_invalid_underscore_grammar_never_approves_cost_or_transport()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "FluxProUltraImageNode",
            "FluxKontextProImageNode",
            "FluxKontextMaxImageNode",
        ] {
            for ratio in [
                "_1:1", "1_:1", "1__0:1", "+_1:1", "1_+0:1", "1 _0:1", "+ 1:1", "++1:1", "1:1_",
                "1:_1", "0_0:1", "1:0_0", "1.0:1", "0x10:1", "1_0:1", "1:1_0",
            ] {
                let mut input = fixture_input()?;
                input.node = node.into();
                input.aspect_ratio = ratio.into();
                let mut transport = TestTransport::new(vec![]);
                assert!(matches!(
                    invoke(&input, credentials(), 100_000, &mut transport),
                    Err(BflError::Input("Invalid aspect ratio"))
                ));
                assert!(transport.finish()?.is_empty());
            }
        }
        assert_eq!(
            aspect_ratio_integer("18_446_744_073_709_551_615")?,
            u64::MAX
        );
        assert!(aspect_ratio_integer("18_446_744_073_709_551_616").is_err());
        Ok(())
    }

    #[test]
    fn bfl_invalid_ratio_is_rejected_before_cost_or_transport() -> Result<(), Box<dyn Error>> {
        for node in [
            "FluxProUltraImageNode",
            "FluxKontextProImageNode",
            "FluxKontextMaxImageNode",
        ] {
            for ratio in [
                "1:5",
                "5:1",
                "0:1",
                "1:0",
                "-1:1",
                "1::1",
                "",
                "1.0:1",
                "\u{200b}1:1\u{200b}",
                "\u{feff}1:1\u{feff}",
                "\u{180e}1:1\u{180e}",
            ] {
                let mut input = fixture_input()?;
                input.node = node.into();
                input.aspect_ratio = ratio.into();
                let mut transport = TestTransport::new(vec![]);
                assert!(matches!(
                    invoke(&input, credentials(), 100_000, &mut transport),
                    Err(BflError::Input("Invalid aspect ratio"))
                ));
                assert!(transport.finish()?.is_empty());
            }
        }
        for ratio in ["1:4", "4:1", "\u{001c}1\u{001c}:4", "4:\u{001f}1\u{001f}"] {
            assert!(aspect_ratio(ratio).is_ok());
        }
        Ok(())
    }

    #[test]
    fn bfl_auth_prefers_token_and_never_leaks_credentials_to_absolute_urls()
    -> Result<(), Box<dyn Error>> {
        let token = request_head(ROUTES[0].path, HttpMethod::Post, credentials(), false)?;
        assert_eq!(token.headers[1].name, "Authorization");
        assert_eq!(token.secret_id.as_deref(), Some("sim_org.auth_token"));
        let key = request_head(
            ROUTES[0].path,
            HttpMethod::Post,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(key.headers[1].name, "X-API-KEY");
        for url in [
            "https://poll.fixture.invalid/task",
            "https://download.fixture.invalid/image.png?signature=fixture",
        ] {
            let head = request_head(url, HttpMethod::Get, credentials(), false)?;
            assert!(head.secret_id.is_none());
            assert_eq!(head.headers.len(), 1);
        }
        assert!(
            request_head(
                ROUTES[0].path,
                HttpMethod::Post,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                false
            )
            .is_err()
        );
        for url in [
            "//host/path",
            "https://user:password@host/path",
            "http://host/path",
            "https://host/path\n",
            "https://host/path#fragment",
        ] {
            assert!(endpoint(url).is_err());
        }
        Ok(())
    }
    #[test]
    fn bfl_response_states_and_malformed_tails_are_fail_closed() {
        for status in [
            "Request Moderated",
            "Content Moderated",
            "Error",
            "Task not found",
        ] {
            assert!(matches!(
                task_status(format!("{{\"id\":\"task\",\"status\":\"{status}\"}}").as_bytes()),
                Err(BflError::Provider(_))
            ));
        }
        for body in [
            r#"{"id":"task","status":"Ready"}"#,
            r#"{"id":"task","status":"Ready","result":{"sample":42}}"#,
            r#"{"id":"task","status":"Pending","progress":1.1}"#,
            r#"{"id":"task","status":"Unknown"}"#,
            r#"{"id":"task","status":"Pending"}x"#,
            r#"{"id":"task","status":"Pending"}{}"#,
        ] {
            assert!(task_status(body.as_bytes()).is_err());
        }
        assert!(task_status(&vec![b' '; MAXIMUM_RESPONSE_BYTES + 1]).is_err());
        assert!(
            submitted_task(br#"{"id":"task","polling_url":"https://host/task","cost":-1}"#)
                .is_err()
        );
    }
    #[test]
    fn bfl_poll_download_preserves_response_bytes_receipt_and_authoritative_cents()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input()?;
        let (path, body) = request(&input)?;
        let mut steps = script_exchange(
            request_head(path, HttpMethod::Post, credentials(), false)?,
            body,
            Some(100_000),
            br#"{"id":"task","polling_url":"https://poll.fixture.invalid/task","cost":3.75}"#
                .to_vec(),
            handle(0, 1),
        );
        steps.extend(script_exchange(request_head("https://poll.fixture.invalid/task",HttpMethod::Get,credentials(),false)?,vec![],None,br#"{"id":"task","status":"Ready","result":{"sample":"https://download.fixture.invalid/image.png"}}"#.to_vec(),handle(1,1)));
        let image = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
        ))
        .to_vec();
        steps.extend(script_exchange(
            request_head(
                "https://download.fixture.invalid/image.png",
                HttpMethod::Get,
                credentials(),
                true,
            )?,
            vec![],
            None,
            image.clone(),
            handle(2, 1),
        ));
        let mut transport = TestTransport::new(steps);
        let result = invoke(&input, credentials(), 100_000, &mut transport)?;
        transport.finish()?;
        assert_eq!(result.bytes, image);
        assert_eq!(result.response_receipt, b"terminal-receipt");
        assert_eq!(result.authoritative_cost_usd, Some(0.0375));
        Ok(())
    }
    #[test]
    fn bfl_cancel_worker_loss_and_rejected_cost_preserve_terminal_reasons()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input()?;
        let (path, body) = request(&input)?;
        let head = request_head(path, HttpMethod::Post, credentials(), false)?;
        let mut transport = TestTransport::new(vec![ScriptStep::Cancel(None)]);
        assert!(matches!(
            invoke(&input, credentials(), 100_000, &mut transport),
            Err(BflError::Transport(ConformanceError::Cancelled))
        ));
        transport.finish()?;
        let mut transport = TestTransport::new(vec![
            ScriptStep::CheckCancelled(None),
            ScriptStep::Start(head.clone(), Err(ConformanceError::WorkerLost)),
        ]);
        assert!(matches!(
            invoke(&input, credentials(), 100_000, &mut transport),
            Err(BflError::Transport(ConformanceError::WorkerLost))
        ));
        transport.finish()?;
        let mut steps = script_exchange(head, body, Some(100_000), br#"{}"#.to_vec(), handle(0, 2));
        steps.truncate(3);
        if let Some(ScriptStep::Cost(_, _, response)) = steps.last_mut() {
            response.accepted = false;
        }
        let mut transport = TestTransport::new(steps);
        assert!(matches!(
            invoke(&input, credentials(), 100_000, &mut transport),
            Err(BflError::Transport(ConformanceError::InvalidCost))
        ));
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bfl_retry_backoff_uses_new_handles_and_preserves_request_bytes() -> Result<(), Box<dyn Error>>
    {
        let input = fixture_input()?;
        let (path, body) = request(&input)?;
        let head = request_head(path, HttpMethod::Post, credentials(), false)?;
        let mut steps = script_exchange(
            head.clone(),
            body.clone(),
            None,
            b"busy".to_vec(),
            handle(0, 1),
        );
        if let Some(ScriptStep::Wait {
            outcome: ResponseEvent::Head { status, .. },
            ..
        }) = steps.get_mut(5)
        {
            *status = 503;
        } else {
            return Err("missing scripted response header".into());
        }
        let mut fresh = script_exchange(
            head.clone(),
            body.clone(),
            None,
            b"success".to_vec(),
            handle(0, 2),
        );
        fresh.remove(0);
        steps.extend(fresh);
        let mut transport = TestTransport::new(steps);
        let (bytes, _, fresh) =
            exchange(&head, &body, None, MAXIMUM_RESPONSE_BYTES, &mut transport)?;
        assert_eq!(bytes, b"success");
        assert_eq!(fresh, handle(0, 2));
        assert_eq!(transport.delays, vec![1000]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bfl_pending_polls_delegate_delay_and_keep_exact_fractional_progress()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input()?;
        let (path, body) = request(&input)?;
        let mut steps = script_exchange(
            request_head(path, HttpMethod::Post, credentials(), false)?,
            body,
            Some(100_000),
            br#"{"id":"task","polling_url":"https://poll.fixture.invalid/task"}"#.to_vec(),
            handle(0, 1),
        );
        steps.extend(script_exchange(
            request_head(
                "https://poll.fixture.invalid/task",
                HttpMethod::Get,
                credentials(),
                false,
            )?,
            vec![],
            None,
            br#"{"id":"task","status":"Pending","progress":0.125}"#.to_vec(),
            handle(1, 1),
        ));
        steps.extend([
            ScriptStep::CheckCancelled(Some(handle(1, 1))),
            ScriptStep::CheckCancelled(Some(handle(1, 1))),
        ]);
        steps.extend(script_exchange(request_head("https://poll.fixture.invalid/task",HttpMethod::Get,credentials(),false)?,vec![],None,br#"{"id":"task","status":"Ready","result":{"sample":"https://download.fixture.invalid/image.png"}}"#.to_vec(),handle(2,1)));
        steps.extend(script_exchange(
            request_head(
                "https://download.fixture.invalid/image.png",
                HttpMethod::Get,
                credentials(),
                true,
            )?,
            vec![],
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
            ))
            .to_vec(),
            handle(3, 1),
        ));
        let mut transport = TestTransport::new(steps);
        let result = invoke(&input, credentials(), 100_000, &mut transport)?;
        assert!(result.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(transport.delays, vec![5000]);
        assert_eq!(transport.progress, vec![0.125, 100.0]);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bfl_stream_cancellation_oversize_and_timeouts_preserve_fresh_sequence()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(ROUTES[0].path, HttpMethod::Get, credentials(), false)?;
        let make = |outcome| {
            vec![
                ScriptStep::CheckCancelled(None),
                ScriptStep::Start(head.clone(), Ok(handle(0, 1))),
                ScriptStep::CheckCancelled(Some(handle(0, 1))),
                ScriptStep::RequestChunk {
                    handle: handle(0, 1),
                    sequence: 0,
                    bytes: vec![],
                    end: true,
                },
                ScriptStep::CheckCancelled(Some(handle(0, 1))),
                ScriptStep::Wait {
                    handle: handle(0, 1),
                    after_sequence: None,
                    timeout_milliseconds: 1000,
                    outcome,
                },
            ]
        };
        let mut transport = TestTransport::new(make(ResponseEvent::Cancelled));
        assert!(matches!(
            exchange(&head, &[], None, 64, &mut transport),
            Err(BflError::Transport(ConformanceError::Cancelled))
        ));
        transport.finish()?;
        let mut steps = make(ResponseEvent::Head {
            status: 200,
            headers: vec![],
        });
        steps.extend([
            ScriptStep::CheckCancelled(Some(handle(0, 1))),
            ScriptStep::Wait {
                handle: handle(0, 1),
                after_sequence: Some(0),
                timeout_milliseconds: 1000,
                outcome: ResponseEvent::Chunk(vec![0; 65]),
            },
        ]);
        let mut transport = TestTransport::new(steps);
        assert!(matches!(
            exchange(&head, &[], None, 64, &mut transport),
            Err(BflError::Transport(ConformanceError::BodyLimit))
        ));
        transport.finish()?;
        let mut steps = make(ResponseEvent::TimedOut);
        let mut success =
            script_exchange(head.clone(), vec![], None, b"valid".to_vec(), handle(0, 1));
        success.drain(0..4);
        steps.extend(success);
        let mut transport = TestTransport::new(steps);
        assert_eq!(exchange(&head, &[], None, 64, &mut transport)?.0, b"valid");
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bfl_rejected_canonical_image_decode_publishes_nothing() -> Result<(), Box<dyn Error>> {
        struct RejectDecoder {
            seen: bool,
        }
        impl BflImageMaterializer for RejectDecoder {
            fn decode_image(
                &mut self,
                image: &DownloadedImage,
            ) -> Result<crate::common::EncodedOutput, ConformanceError> {
                assert_eq!(
                    image.bytes,
                    include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
                    ))
                );
                assert_eq!(image.response_receipt, b"terminal-receipt");
                self.seen = true;
                Err(ConformanceError::InvalidResponse)
            }
        }
        #[derive(Default)]
        struct Sink {
            batches: usize,
        }
        impl crate::common::MaterializationSink for Sink {
            fn publish(
                &mut self,
                _batch: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.batches += 1;
                Ok(())
            }
        }
        let input = fixture_input()?;
        let (path, body) = request(&input)?;
        let mut steps = script_exchange(
            request_head(path, HttpMethod::Post, credentials(), false)?,
            body,
            Some(100_000),
            br#"{"id":"task","polling_url":"https://poll.fixture.invalid/task"}"#.to_vec(),
            handle(0, 1),
        );
        steps.extend(script_exchange(request_head("https://poll.fixture.invalid/task",HttpMethod::Get,credentials(),false)?,vec![],None,br#"{"id":"task","status":"Ready","result":{"sample":"https://download.fixture.invalid/image.png"}}"#.to_vec(),handle(1,1)));
        steps.extend(script_exchange(
            request_head(
                "https://download.fixture.invalid/image.png",
                HttpMethod::Get,
                credentials(),
                true,
            )?,
            vec![],
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
            ))
            .to_vec(),
            handle(2, 1),
        ));
        steps.push(ScriptStep::CheckCancelled(None));
        let mut transport = TestTransport::new(steps);
        let mut decoder = RejectDecoder { seen: false };
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input,
                credentials(),
                100_000,
                &mut transport,
                &mut decoder,
                &mut sink
            ),
            Err(BflError::Transport(ConformanceError::InvalidResponse))
        ));
        assert!(decoder.seen);
        assert_eq!(sink.batches, 0);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn bfl_generation_poll_and_download_retry_limits_are_distinct() -> Result<(), Box<dyn Error>> {
        let head = request_head(ROUTES[0].path, HttpMethod::Get, credentials(), false)?;
        for (policy, status, limit, expected_delays) in [
            (GENERATION_RETRY, 503, 3, vec![1000, 2000, 4000]),
            (
                POLL_RETRY,
                503,
                10,
                vec![
                    1000, 1400, 1960, 2744, 3842, 5378, 7530, 10541, 14758, 20661,
                ],
            ),
            (DOWNLOAD_RETRY, 429, 5, vec![1000, 2000, 4000, 8000, 16000]),
        ] {
            let mut steps = vec![];
            for attempt in 0..=limit {
                let mut next = script_exchange(
                    head.clone(),
                    vec![],
                    None,
                    b"busy".to_vec(),
                    handle(0, attempt + 1),
                );
                if let Some(ScriptStep::Wait {
                    outcome:
                        ResponseEvent::Head {
                            status: response_status,
                            ..
                        },
                    ..
                }) = next.get_mut(5)
                {
                    *response_status = status;
                } else {
                    return Err("missing retry header".into());
                }
                if attempt != 0 {
                    next.remove(0);
                }
                if attempt == limit {
                    next.pop();
                }
                steps.extend(next);
            }
            let mut transport = TestTransport::new(steps);
            assert!(
                matches!(exchange_with_policy(&head,&[],None,MAXIMUM_RESPONSE_BYTES,policy,&mut transport),Err(BflError::Http{status:observed,..}) if observed==status)
            );
            assert_eq!(transport.delays, expected_delays);
            transport.finish()?;
        }
        Ok(())
    }
    #[test]
    fn bfl_source_single_rounded_ratios_preserve_requests_and_execute() -> Result<(), Box<dyn Error>>
    {
        let oracle: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/source-contract.json"
        )))?;
        let cases = oracle["cases"].as_array().ok_or("Missing cases")?;
        for node in [
            "FluxProUltraImageNode",
            "FluxKontextProImageNode",
            "FluxKontextMaxImageNode",
        ] {
            let case = cases
                .iter()
                .find(|case| case["input"]["node"] == node)
                .ok_or("Missing node")?;
            for ratio in [
                "4611686018427388417:1152921504606847104",
                "1152921504606847104:4611686018427388417",
                "9007199254740993:2251799813685248",
                "18014398509481983:72057594037927936",
            ] {
                let mut input: Input = serde_json::from_value(case["input"].clone())?;
                input.aspect_ratio = ratio.into();
                input.prompt = " \u{001c} original prompt \u{001f} ".into();
                let (path, body) = request(&input)?;
                let mut expected = case["request"]["body"].clone();
                expected["aspect_ratio"] = ratio.into();
                expected["prompt"] = input.prompt.clone().into();
                assert_eq!(serde_json::from_slice::<Value>(&body)?, expected);
                assert_eq!(
                    path,
                    case["request"]["path"].as_str().ok_or("Missing path")?
                );
                let mut steps = script_exchange(
                    request_head(path, HttpMethod::Post, credentials(), false)?,
                    body,
                    Some(100_000),
                    br#"{"id":"task","polling_url":"https://poll.fixture.invalid/task"}"#.to_vec(),
                    handle(0, 1),
                );
                steps.extend(script_exchange(request_head("https://poll.fixture.invalid/task", HttpMethod::Get, credentials(), false)?, vec![], None, br#"{"id":"task","status":"Ready","result":{"sample":"https://download.fixture.invalid/image.png"}}"#.to_vec(), handle(1, 1)));
                let image = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/bfl/downloaded-image.png"
                ))
                .to_vec();
                steps.extend(script_exchange(
                    request_head(
                        "https://download.fixture.invalid/image.png",
                        HttpMethod::Get,
                        credentials(),
                        true,
                    )?,
                    vec![],
                    None,
                    image.clone(),
                    handle(2, 1),
                ));
                let mut transport = TestTransport::new(steps);
                let result = invoke(&input, credentials(), 100_000, &mut transport)?;
                transport.finish()?;
                assert_eq!(result.bytes, image);
                assert_eq!(result.response_receipt, b"terminal-receipt");
            }
            for ratio in [
                "9007199254740994:2251799813685248",
                "18014398509481982:72057594037927936",
            ] {
                let mut input: Input = serde_json::from_value(case["input"].clone())?;
                input.aspect_ratio = ratio.into();
                assert!(matches!(
                    request(&input),
                    Err(BflError::Input("Invalid aspect ratio"))
                ));
                let mut transport = TestTransport::new(Vec::new());
                assert!(matches!(
                    invoke(&input, credentials(), 100_000, &mut transport),
                    Err(BflError::Input("Invalid aspect ratio"))
                ));
                assert!(transport.head.is_none());
                transport.finish()?;
            }
        }
        Ok(())
    }

    #[test]
    fn bfl_single_rounded_source_boundary_retains_pre_request_cancellation()
    -> Result<(), Box<dyn Error>> {
        let oracle: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/bfl/source-contract.json"
        )))?;
        let cases = oracle["cases"].as_array().ok_or("Missing cases")?;
        for node in [
            "FluxProUltraImageNode",
            "FluxKontextProImageNode",
            "FluxKontextMaxImageNode",
        ] {
            let case = cases
                .iter()
                .find(|case| case["input"]["node"] == node)
                .ok_or("Missing node")?;
            for ratio in [
                "4611686018427388417:1152921504606847104",
                "1152921504606847104:4611686018427388417",
            ] {
                let mut input: Input = serde_json::from_value(case["input"].clone())?;
                input.aspect_ratio = ratio.into();
                let mut transport = TestTransport::new(vec![ScriptStep::Cancel(None)]);
                assert!(matches!(
                    invoke(&input, credentials(), 100_000, &mut transport),
                    Err(BflError::Transport(ConformanceError::Cancelled))
                ));
                assert!(transport.head.is_none());
                transport.finish()?;
            }
        }
        Ok(())
    }

    #[test]
    fn bfl_source_remaining_invalid_ratio_parts_reject_before_cost_or_transport()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "FluxProUltraImageNode",
            "FluxKontextProImageNode",
            "FluxKontextMaxImageNode",
        ] {
            for ratio in ["0:0", "1:-1", "1:1:1"] {
                let mut input = fixture_input()?;
                input.node = node.into();
                input.aspect_ratio = ratio.into();
                let mut transport = TestTransport::new(vec![]);
                assert!(matches!(
                    invoke(&input, credentials(), 100_000, &mut transport),
                    Err(BflError::Input("Invalid aspect ratio"))
                ));
                assert!(transport.finish()?.is_empty());
            }
        }
        Ok(())
    }
}
