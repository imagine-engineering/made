use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, Progress, ProviderTransport, RequestHead, ResponseEvent, StreamHandle,
    TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
pub const NAMESPACE: &str = "zed.sim.provider.hitpaw";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "8f479c2060c399b3e5a57c0770a623dd4be6a00c711f2ce9c10ccdd55651fc6e";
pub const IMAGE_COLOR_MODE: &str = "RGBA";
pub const MAXIMUM_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0226",
        node_identifier: "HitPawGeneralImageEnhance",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0227",
        node_identifier: "HitPawVideoEnhance",
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
        feature_id: "COMFY-API-EXT-0049",
        method: "POST",
        path: "/proxy/hitpaw/api/photo-enhancer",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0050",
        method: "POST",
        path: "/proxy/hitpaw/api/task-status",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0051",
        method: "POST",
        path: "/proxy/hitpaw/api/video-enhancer",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum HitPawError {
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
impl fmt::Display for HitPawError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "HitPaw HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for HitPawError {}
impl From<ConformanceError> for HitPawError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for HitPawError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, HitPawError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/hitpaw/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, HitPawError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(HitPawError::Input("Unknown HitPaw node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(HitPawError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(HitPawError::Input("Unknown HitPaw source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(HitPawError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(HitPawError::Input("Missing HitPaw source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, HitPawError> {
    input
        .arguments
        .get(name)
        .ok_or(HitPawError::Input("Missing HitPaw argument"))
}
fn text(value: &Value) -> Result<&str, HitPawError> {
    value
        .as_str()
        .ok_or(HitPawError::Input("Expected HitPaw string"))
}
fn integer(value: &Value, maximum: u64) -> Result<u64, HitPawError> {
    let number = value
        .as_u64()
        .ok_or(HitPawError::Input("Expected HitPaw unsigned integer"))?;
    if number > maximum {
        return Err(HitPawError::Input("HitPaw integer bound exceeded"));
    }
    Ok(number)
}
fn choice<'a>(value: &'a Value, choices: &[&str]) -> Result<&'a str, HitPawError> {
    let text = text(value)?;
    if !choices.contains(&text) {
        return Err(HitPawError::Input("Unknown HitPaw source choice"));
    }
    Ok(text)
}
fn endpoint(value: &str) -> Result<bool, HitPawError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(HitPawError::Response("Invalid HitPaw response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(HitPawError::Response("Invalid HitPaw response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(HitPawError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(HitPawError::Response("Invalid HitPaw endpoint authority"));
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
) -> Result<RequestHead, HitPawError> {
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
            return Err(HitPawError::Input("Missing Comfy API credentials"));
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

pub trait HitPawTransport: ProviderTransport {
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
    fn poll_delay(
        &mut self,
        handle: StreamHandle,
        delay_milliseconds: u64,
    ) -> Result<(), ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct MediaInventory {
    pub source: Value,
    pub width: u64,
    pub height: u64,
    pub duration_seconds: Option<f64>,
}
pub trait HitPawInputOwner {
    fn inventory(&mut self, input: &Input) -> Result<MediaInventory, ConformanceError>;
    fn downscale_image(
        &mut self,
        image: &Value,
        total_pixels: u64,
        filter: &str,
    ) -> Result<Value, ConformanceError>;
    fn upload_image(
        &mut self,
        image: &Value,
        mime_type: Option<&str>,
        total_pixels: Option<u64>,
    ) -> Result<String, ConformanceError>;
    fn upload_video(
        &mut self,
        video: &Value,
        container: &str,
        codec: &str,
    ) -> Result<String, ConformanceError>;
}
pub const MAXIMUM_IMAGE_OUTPUT_PIXELS: u64 = 32_000_000;
#[derive(Clone, Debug)]
pub struct PlannedRequest {
    pub path: &'static str,
    pub body: Value,
    pub is_video: bool,
    pub effective_upscale: u64,
    pub downscale_total_pixels: Option<u64>,
    pub target_dimensions: Option<[u64; 2]>,
    pub wait_label: &'static str,
    pub final_label: &'static str,
}
fn pixels(inventory: &MediaInventory) -> Result<u64, HitPawError> {
    if inventory.width == 0 || inventory.height == 0 {
        return Err(HitPawError::Input("Invalid canonical media dimensions"));
    }
    inventory
        .width
        .checked_mul(inventory.height)
        .ok_or(HitPawError::Input("Canonical media dimension overflow"))
}
fn commas(value: u64) -> String {
    let source = value.to_string();
    source
        .chars()
        .enumerate()
        .flat_map(|(index, character)| {
            if index > 0 && (source.len() - index).is_multiple_of(3) {
                vec![',', character]
            } else {
                vec![character]
            }
        })
        .collect()
}
fn image_scale(
    input: &Input,
    inventory: &MediaInventory,
) -> Result<(u64, Option<u64>), HitPawError> {
    let scale = integer(argument(input, "upscale_factor")?, 4)?;
    if ![1, 2, 4].contains(&scale) {
        return Err(HitPawError::Input("Unknown HitPaw upscale factor"));
    }
    let input_pixels = pixels(inventory)?;
    let output = input_pixels
        .checked_mul(scale * scale)
        .ok_or(HitPawError::Input("HitPaw output pixel overflow"))?;
    if output <= MAXIMUM_IMAGE_OUTPUT_PIXELS {
        return Ok((scale, None));
    }
    let automatic = argument(input, "auto_downscale")?
        .as_bool()
        .ok_or(HitPawError::Input("Expected HitPaw auto_downscale boolean"))?;
    if !automatic {
        let width = inventory
            .width
            .checked_mul(scale)
            .ok_or(HitPawError::Input("HitPaw output width overflow"))?;
        let height = inventory
            .height
            .checked_mul(scale)
            .ok_or(HitPawError::Input("HitPaw output height overflow"))?;
        return Err(HitPawError::Provider(format!(
            "Output size ({width}x{height} = {} pixels) exceeds maximum allowed size of 32,000,000 pixels (32MP). Enable auto_downscale or use a smaller input image or a lower upscale factor.",
            commas(output)
        )));
    }
    for candidate in [4, 2, 1] {
        if candidate > scale {
            continue;
        }
        let output = input_pixels
            .checked_mul(candidate * candidate)
            .ok_or(HitPawError::Input("HitPaw candidate pixel overflow"))?;
        if output <= MAXIMUM_IMAGE_OUTPUT_PIXELS {
            return Ok((candidate, None));
        }
        if (output as f64 / MAXIMUM_IMAGE_OUTPUT_PIXELS as f64).sqrt() <= 2.0 {
            return Ok((
                candidate,
                Some(MAXIMUM_IMAGE_OUTPUT_PIXELS / (candidate * candidate)),
            ));
        }
    }
    Ok((1, Some(MAXIMUM_IMAGE_OUTPUT_PIXELS)))
}
fn video_dimensions(
    input: &Input,
    inventory: &MediaInventory,
) -> Result<([u64; 2], String), HitPawError> {
    pixels(inventory)?;
    if let Some(duration) = inventory.duration_seconds
        && (!duration.is_finite() || !(0.5 - 0.0001..=3600.0 + 0.0001).contains(&duration))
    {
        return Err(HitPawError::Input(
            "HitPaw video duration exceeds source limits",
        ));
    }
    let model = argument(input, "model")?;
    let resolution = text(&model["resolution"])?;
    let model_name = text(&model["model"])?;
    let oracle = oracle()?;
    let api_model = text(&oracle["model_map"][model_name])?.to_owned();
    if resolution == "original" {
        return Ok(([inventory.width, inventory.height], api_model));
    }
    let square = inventory.width == inventory.height;
    let target = oracle[if square { "square_map" } else { "target_map" }][resolution]
        .as_u64()
        .ok_or(HitPawError::Input("Unknown HitPaw resolution"))?;
    let dimensions = if square {
        if target < inventory.width {
            return Err(HitPawError::Provider(format!(
                "Selected resolution {resolution} ({target}x{target}) is smaller than the input video ({}x{}). Please select a higher resolution or 'original'.",
                inventory.width, inventory.height
            )));
        }
        [target, target]
    } else {
        let shorter = inventory.width.min(inventory.height);
        if target < shorter {
            return Err(HitPawError::Provider(format!(
                "Selected resolution {resolution} ({target}p) is smaller than the input video's shorter dimension ({shorter}p). Please select a higher resolution or 'original'."
            )));
        }
        if inventory.width > inventory.height {
            [
                (target as f64 * (inventory.width as f64 / inventory.height as f64)) as u64,
                target,
            ]
        } else {
            [
                target,
                (target as f64 * (inventory.height as f64 / inventory.width as f64)) as u64,
            ]
        }
    };
    Ok((dimensions, api_model))
}
pub fn request(
    input: &Input,
    inventory: &MediaInventory,
    uploaded_url: Option<&str>,
) -> Result<PlannedRequest, HitPawError> {
    let input = normalized(input)?;
    let (path, body, is_video, effective_upscale, downscale_total_pixels, target_dimensions) =
        match input.node.as_str() {
            "HitPawGeneralImageEnhance" => {
                choice(
                    argument(&input, "model")?,
                    &["generative_portrait", "generative"],
                )?;
                argument(&input, "auto_downscale")?
                    .as_bool()
                    .ok_or(HitPawError::Input("Expected HitPaw auto_downscale boolean"))?;
                let (scale, total) = image_scale(&input, inventory)?;
                (
                    "/proxy/hitpaw/api/photo-enhancer",
                    json!({"model_name":format!("{}_{}x",text(argument(&input,"model")?)?,scale),"img_url":uploaded_url,"extension":".png","exif":false}),
                    false,
                    scale,
                    total,
                    None,
                )
            }
            "HitPawVideoEnhance" => {
                let (dimensions, model_name) = video_dimensions(&input, inventory)?;
                (
                    "/proxy/hitpaw/api/video-enhancer",
                    json!({"model_name":model_name,"video_url":uploaded_url,"extension":".mp4","resolution":dimensions,"original_resolution":[inventory.width,inventory.height]}),
                    true,
                    1,
                    None,
                    Some(dimensions),
                )
            }
            _ => return Err(HitPawError::Input("Unknown HitPaw node")),
        };
    Ok(PlannedRequest {
        path,
        body,
        is_video,
        effective_upscale,
        downscale_total_pixels,
        target_dimensions,
        wait_label: "Creating task",
        final_label: "Task created",
    })
}
fn check_url(url: &str) -> Result<(), HitPawError> {
    if endpoint(url)? {
        return Err(HitPawError::Response(
            "HitPaw requires an absolute result or upload URL",
        ));
    }
    Ok(())
}
fn parse_json(bytes: &[u8]) -> Result<Value, HitPawError> {
    let response: Value = serde_json::from_slice(bytes)?;
    if !response.is_object() {
        return Err(HitPawError::Response("Expected HitPaw response object"));
    }
    Ok(response)
}
fn task_creation(response: &Value, is_video: bool) -> Result<(String, f64), HitPawError> {
    let code = response["code"]
        .as_i64()
        .ok_or(HitPawError::Response("Missing HitPaw task creation code"))?;
    let message = text(&response["message"])
        .map_err(|_| HitPawError::Response("Missing HitPaw task creation message"))?;
    if code != 200 && !is_video {
        return Err(HitPawError::Provider(format!(
            "Task creation failed with code {code}: {message}"
        )));
    }
    let coins = response["data"]["consume_coins"]
        .as_u64()
        .ok_or(HitPawError::Response(
            "Missing HitPaw task creation consume_coins",
        ))?;
    let usd = coins as f64 / 1000.0;
    if code != 200 {
        return Err(HitPawError::Provider(format!(
            "Task creation failed with code {code}: {message}"
        )));
    }
    let job_id = text(&response["data"]["job_id"])
        .map_err(|_| HitPawError::Response("Missing HitPaw job identifier"))?;
    if job_id.is_empty() || job_id.len() > 1024 || job_id.chars().any(char::is_control) {
        return Err(HitPawError::Response("Invalid HitPaw job identifier"));
    }
    Ok((job_id.into(), usd))
}
fn task_status(response: &Value) -> Result<String, HitPawError> {
    response["code"]
        .as_i64()
        .ok_or(HitPawError::Response("Invalid HitPaw poll code"))?;
    text(&response["message"]).map_err(|_| HitPawError::Response("Invalid HitPaw poll message"))?;
    text(&response["data"]["job_id"])
        .map_err(|_| HitPawError::Response("Invalid HitPaw poll job identifier"))?;
    if response["data"]
        .get("res_url")
        .is_some_and(|value| !value.is_string())
    {
        return Err(HitPawError::Response("Invalid HitPaw result URL"));
    }
    Ok(text(&response["data"]["status"])
        .map_err(|_| HitPawError::Response("Missing HitPaw status"))?
        .trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, '\u{001c}'..='\u{001f}')
        })
        .to_lowercase())
}
fn source_status(status: &str, category: &str) -> Result<bool, HitPawError> {
    Ok(oracle()?["polling"][category]
        .as_array()
        .ok_or(HitPawError::Response("Missing source poll statuses"))?
        .iter()
        .any(|value| value.as_str() == Some(status)))
}
#[derive(Clone, Debug)]
pub struct DownloadedMedia {
    pub url: String,
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub node: String,
    pub is_video: bool,
    pub media: DownloadedMedia,
    pub usd: f64,
    pub creation_response_bytes: Vec<u8>,
    pub creation_receipt: Vec<u8>,
    pub response_bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub plan: PlannedRequest,
}
fn approve(
    handle: StreamHandle,
    maximum: u64,
    transport: &mut dyn HitPawTransport,
) -> Result<(), HitPawError> {
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "hitpaw.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits: maximum,
        },
    )?;
    if !approval.accepted
        || approval.approved_microunits < maximum
        || approval.receipt.is_empty()
        || approval.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    Ok(())
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn HitPawTransport,
    owner: &mut dyn HitPawInputOwner,
) -> Result<ResultValue, HitPawError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    let mut head = request_head(
        "/proxy/hitpaw/api/photo-enhancer",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let inventory = owner.inventory(&input)?;
    let preflight = request(&input, &inventory, None)?;
    head.path = preflight.path.into();
    let initial_handle = transport.start_request(&head)?;
    approve(initial_handle, maximum_cost_microunits, transport)?;
    transport.check_cancelled(Some(initial_handle))?;
    let url = if preflight.is_video {
        owner.upload_video(&inventory.source, "mp4", "h264")?
    } else {
        let image = if let Some(total) = preflight.downscale_total_pixels {
            owner.downscale_image(&inventory.source, total, "lanczos")?
        } else {
            inventory.source.clone()
        };
        transport.check_cancelled(Some(initial_handle))?;
        owner.upload_image(&image, None, None)?
    };
    check_url(&url)?;
    let plan = request(&input, &inventory, Some(&url))?;
    let body = serde_json::to_vec(&plan.body)?;
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let (creation_response_bytes, creation_receipt, _creation_handle) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        initial_handle,
        transport,
    )?;
    let (job_id, usd) = task_creation(&parse_json(&creation_response_bytes)?, plan.is_video)?;
    let head = request_head(
        "/proxy/hitpaw/api/task-status",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    let body = serde_json::to_vec(&json!({"job_id":job_id}))?;
    let mut active = 0;
    // Queued source statuses do not consume active attempts; the supervisor
    // supplies a finite total bound to prevent indefinite worker retention.
    for sequence in 0..4096 {
        let (response_bytes, receipt, handle) = exchange_with_policy(
            &head,
            &body,
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            transport,
        )?;
        let response = parse_json(&response_bytes)?;
        let status = task_status(&response)?;
        if source_status(&status, "COMPLETED_STATUSES")? {
            let url = text(&response["data"]["res_url"])
                .map_err(|_| HitPawError::Response("Missing HitPaw completed media URL"))?
                .to_owned();
            check_url(&url)?;
            let download_head = request_head(&url, HttpMethod::Get, credentials, true)?;
            let (bytes, download_receipt, _download_handle) = exchange_with_policy(
                &download_head,
                &[],
                None,
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                transport,
            )?;
            if bytes.is_empty() {
                return Err(HitPawError::Response("Empty HitPaw media download"));
            }
            transport.check_cancelled(Some(handle))?;
            return Ok(ResultValue {
                node: input.node,
                is_video: plan.is_video,
                media: DownloadedMedia {
                    url,
                    bytes,
                    receipt: download_receipt,
                },
                usd,
                creation_response_bytes,
                creation_receipt,
                response_bytes,
                receipt,
                plan,
            });
        }
        if source_status(&status, "FAILED_STATUSES")? {
            return Err(HitPawError::Provider(format!(
                "Task failed: {}",
                serde_json::to_string(&response)?
            )));
        }
        if !source_status(&status, "QUEUED_STATUSES")? {
            active += 1;
        }
        if active >= 480 || sequence == 4095 {
            return Err(ConformanceError::PollLimit.into());
        }
        transport.report_progress(
            handle,
            &Progress {
                sequence: sequence as u64,
                completed: 0,
                total: 100,
                message: Some(status),
            },
        )?;
        transport.check_cancelled(Some(handle))?;
        transport.poll_delay(handle, 10_000)?;
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
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn HitPawTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), HitPawError> {
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
    transport: &mut dyn HitPawTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), HitPawError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(HitPawError::Http {
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
                    return Err(HitPawError::Http {
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
    transport: &mut dyn HitPawTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), HitPawError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "hitpaw.generation.maximum".into(),
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
                    return Err(HitPawError::Http {
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

pub trait HitPawMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn HitPawTransport,
    owner: &mut dyn HitPawInputOwner,
    materializer: &mut dyn HitPawMaterializer,
    sink: &mut dyn MaterializationSink,
) -> Result<(), HitPawError> {
    let result = invoke(
        input,
        credentials,
        maximum_cost_microunits,
        transport,
        owner,
    )?;
    transport.check_cancelled(None)?;
    let outputs = materializer.materialize(&result)?;
    let registry = TypeRegistry::built_in()
        .map_err(|_| HitPawError::Response("Missing canonical type registry"))?;
    let type_id = registry
        .resolve(if result.is_video { "VIDEO" } else { "IMAGE" })
        .map_err(|_| HitPawError::Response("Missing canonical output type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| HitPawError::Response("Missing canonical output family"))?;
    let batch = validate_outputs(
        &[ExpectedOutput {
            port_id: "output_0".into(),
            type_id,
            family,
        }],
        outputs,
        MAXIMUM_MEDIA_BYTES,
    )
    .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn fixture_inventory(input: &Input) -> Result<MediaInventory, HitPawError> {
    let source = argument(
        input,
        if input.node == "HitPawGeneralImageEnhance" {
            "image"
        } else {
            "video"
        },
    )?;
    Ok(MediaInventory {
        source: source.clone(),
        width: source["width"]
            .as_u64()
            .ok_or(HitPawError::Input("Missing fixture width"))?,
        height: source["height"]
            .as_u64()
            .ok_or(HitPawError::Input("Missing fixture height"))?,
        duration_seconds: source["duration"].as_f64(),
    })
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"scope":"source request and bounded transport; canonical native media and guest integration pending"})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
    if source["source_sha256"]["comfy_api_nodes/nodes_hitpaw.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let inventory = fixture_inventory(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let body = &case["requests"][0]["body"];
        let url = body["img_url"]
            .as_str()
            .or_else(|| body["video_url"].as_str())
            .ok_or(ConformanceError::InvalidRequest)?;
        let plan =
            request(&input, &inventory, Some(url)).map_err(|_| ConformanceError::InvalidRequest)?;
        let expected = case["media"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?
            .iter()
            .find(|media| media["operation"] == "downscale_image")
            .and_then(|media| media["total_pixels"].as_u64());
        if plan.body != *body
            || plan.path
                != case["requests"][0]["path"]
                    .as_str()
                    .ok_or(ConformanceError::InvalidRequest)?
            || plan.downscale_total_pixels != expected
            || plan.wait_label != case["requests"][0]["wait_label"]
            || plan.final_label != case["requests"][0]["final_label_on_success"]
            || case["requests"][1]["body"] != json!({"job_id":"source-job-42"})
            || case["requests"][1]["poll_interval"] != 10.0
        {
            return Err(ConformanceError::InvalidRequest);
        }
        covered.insert(input.node);
    }
    for case in source["preflight_errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        if request(
            &input,
            &fixture_inventory(&input).map_err(|_| ConformanceError::InvalidRequest)?,
            None,
        )
        .is_ok()
            || case["requests"] != json!([])
        {
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
        CostResponse, MaterializationError, MaterializedBatch, SignedPutRequest,
        StorageCreateRequest, StorageCreateResponse,
    };
    use std::collections::VecDeque;
    #[derive(Default)]
    struct Transport {
        heads: Vec<RequestHead>,
        bodies: Vec<Vec<u8>>,
        costs: Vec<CostRequest>,
        events: VecDeque<Result<ResponseEvent, ConformanceError>>,
        delays: Vec<u64>,
        cancelled: bool,
        deny_cost: bool,
        same_retry_handle: bool,
        progresses: Vec<Progress>,
        poll_delays: Vec<u64>,
    }
    fn handle(slot: u32) -> StreamHandle {
        StreamHandle {
            invocation: 42,
            slot,
            generation: 1,
        }
    }
    impl ProviderTransport for Transport {
        fn check_cancelled(
            &mut self,
            _handle: Option<StreamHandle>,
        ) -> Result<(), ConformanceError> {
            if self.cancelled {
                Err(ConformanceError::Cancelled)
            } else {
                Ok(())
            }
        }
        fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
            self.heads.push(head.clone());
            self.bodies.push(vec![]);
            Ok(handle(self.heads.len() as u32))
        }
        fn write_request_chunk(
            &mut self,
            handle: StreamHandle,
            sequence: u64,
            bytes: &[u8],
            _end: bool,
        ) -> Result<(), ConformanceError> {
            let body = self
                .bodies
                .get_mut(handle.slot as usize - 1)
                .ok_or(ConformanceError::InvalidHandle)?;
            if body.len().div_ceil(MAXIMUM_CHUNK_BYTES) != sequence as usize {
                return Err(ConformanceError::InvalidRequest);
            }
            body.extend_from_slice(bytes);
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
            self.costs.push(request.clone());
            Ok(CostResponse {
                accepted: !self.deny_cost,
                approved_microunits: request.maximum_microunits,
                receipt: vec![42],
            })
        }
        fn report_progress(
            &mut self,
            _handle: StreamHandle,
            progress: &Progress,
        ) -> Result<(), ConformanceError> {
            self.progresses.push(progress.clone());
            Ok(())
        }
        fn wait_response(
            &mut self,
            _handle: StreamHandle,
            _after: Option<u64>,
            _timeout: u64,
        ) -> Result<ResponseEvent, ConformanceError> {
            self.events
                .pop_front()
                .ok_or(ConformanceError::HostFailure)?
        }
    }
    impl HitPawTransport for Transport {
        fn poll_delay(
            &mut self,
            _handle: StreamHandle,
            delay: u64,
        ) -> Result<(), ConformanceError> {
            self.poll_delays.push(delay);
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
                .map(|seconds| seconds.saturating_mul(1000))
                .unwrap_or(fallback))
        }
        fn retry_request(
            &mut self,
            old: StreamHandle,
            delay: u64,
        ) -> Result<StreamHandle, ConformanceError> {
            self.delays.push(delay);
            if self.same_retry_handle {
                return Ok(old);
            }
            let head = self
                .heads
                .last()
                .cloned()
                .ok_or(ConformanceError::InvalidHandle)?;
            self.start_request(&head)
        }
    }

    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn png() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/hitpaw/downloaded-image.png"
        ))
        .to_vec()
    }
    fn events(
        status: u16,
        bytes: Vec<u8>,
        headers: Vec<Header>,
    ) -> VecDeque<Result<ResponseEvent, ConformanceError>> {
        vec![
            Ok(ResponseEvent::Head { status, headers }),
            Ok(ResponseEvent::Chunk(bytes)),
            Ok(ResponseEvent::Completed(vec![7, 8, 9])),
        ]
        .into()
    }

    fn fixture_case(node: &str) -> Result<Value, Box<dyn Error>> {
        Ok(oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == node)
            .ok_or("node")?
            .clone())
    }
    fn fixture_input(node: &str) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(
            fixture_case(node)?["input"].clone(),
        )?)
    }
    fn creation(code: i64, coins: Option<u64>) -> Value {
        json!({"code":code,"message":if code==200{"success"}else{"source creation failure"},"data":{"job_id":"source-job-42","consume_coins":coins}})
    }
    fn poll(status: &str) -> Value {
        json!({"code":200,"message":"success","data":{"job_id":"source-job-42","status":status,"res_url":"https://fixture.invalid/result.png"}})
    }
    fn success_events()
    -> Result<VecDeque<Result<ResponseEvent, ConformanceError>>, serde_json::Error> {
        let mut values = events(200, serde_json::to_vec(&creation(200, Some(123)))?, vec![]);
        values.extend(events(200, serde_json::to_vec(&poll("completed"))?, vec![]));
        values.extend(events(200, png(), vec![]));
        Ok(values)
    }
    struct InputOwner {
        inventory: MediaInventory,
        image_calls: Vec<Value>,
        video_calls: Vec<Value>,
        downscale_calls: Vec<u64>,
        fail_upload: bool,
    }
    impl Default for InputOwner {
        fn default() -> Self {
            Self {
                inventory: MediaInventory {
                    source: json!("actual-png-input"),
                    width: 1,
                    height: 1,
                    duration_seconds: None,
                },
                image_calls: vec![],
                video_calls: vec![],
                downscale_calls: vec![],
                fail_upload: false,
            }
        }
    }
    impl HitPawInputOwner for InputOwner {
        fn inventory(&mut self, _input: &Input) -> Result<MediaInventory, ConformanceError> {
            Ok(self.inventory.clone())
        }
        fn downscale_image(
            &mut self,
            _image: &Value,
            total: u64,
            filter: &str,
        ) -> Result<Value, ConformanceError> {
            assert_eq!(filter, "lanczos");
            self.downscale_calls.push(total);
            Err(ConformanceError::HostFailure)
        }
        fn upload_image(
            &mut self,
            image: &Value,
            mime: Option<&str>,
            total: Option<u64>,
        ) -> Result<String, ConformanceError> {
            assert_eq!(mime, None);
            assert_eq!(total, None);
            self.image_calls.push(image.clone());
            if self.fail_upload {
                Err(ConformanceError::HostFailure)
            } else {
                Ok("https://storage.fixture.invalid/image".into())
            }
        }
        fn upload_video(
            &mut self,
            video: &Value,
            container: &str,
            codec: &str,
        ) -> Result<String, ConformanceError> {
            assert_eq!((container, codec), ("mp4", "h264"));
            self.video_calls.push(video.clone());
            Err(ConformanceError::HostFailure)
        }
    }
    #[derive(Default)]
    struct Sink {
        batches: usize,
    }
    impl MaterializationSink for Sink {
        fn publish(&mut self, _batch: MaterializedBatch) -> Result<(), MaterializationError> {
            self.batches += 1;
            Ok(())
        }
    }
    struct RejectDecoder {
        seen: bool,
    }
    impl HitPawMaterializer for RejectDecoder {
        fn materialize(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            assert_eq!(result.media.bytes, png());
            assert_eq!(result.media.receipt, [7, 8, 9]);
            assert!(!result.is_video);
            self.seen = true;
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn hitpaw_claims_and_all_source_requests_are_exact() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/hitpaw/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 2);
        assert_eq!(ROUTES.len(), 3);
        assert!(
            ROUTES
                .iter()
                .all(|route| route.method == "POST" && route.disposition == "executable")
        );
        assert!(ALIASES.is_empty());
        let source = oracle()?;
        assert_eq!(source["cases"].as_array().ok_or("cases")?.len(), 56);
        assert_eq!(
            source["preflight_errors"].as_array().ok_or("errors")?.len(),
            5
        );
        assert_eq!(
            source["response_errors"].as_array().ok_or("errors")?.len(),
            4
        );
        for contract in source["contracts"].as_object().ok_or("contracts")?.values() {
            let hidden = text(&contract["hidden"])?;
            assert!(
                hidden.contains("auth_token_sim_org")
                    && hidden.contains("api_key_sim_org")
                    && hidden.contains("unique_id")
            );
        }
        Ok(())
    }
    #[test]
    fn hitpaw_image_scale_selection_delegates_only_source_required_downscales()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("HitPawGeneralImageEnhance")?;
        let source = oracle()?;
        for case in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| case["input"]["node"] == "HitPawGeneralImageEnhance")
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let inventory = fixture_inventory(&input)?;
            let plan = request(&input, &inventory, None)?;
            let scale = case["requests"][0]["body"]["model_name"]
                .as_str()
                .ok_or("name")?;
            assert!(scale.ends_with(&format!("_{}x", plan.effective_upscale)));
            let recorded = case["media"]
                .as_array()
                .ok_or("media")?
                .iter()
                .find(|media| media["operation"] == "downscale_image")
                .and_then(|media| media["total_pixels"].as_u64());
            assert_eq!(plan.downscale_total_pixels, recorded);
        }
        let mut owner = InputOwner {
            inventory: MediaInventory {
                source: json!("native-large-image"),
                width: 4000,
                height: 4000,
                duration_seconds: None,
            },
            ..InputOwner::default()
        };
        let mut input = input;
        input.arguments.insert("upscale_factor".into(), json!(4));
        input.arguments.insert("auto_downscale".into(), json!(true));
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(HitPawError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(owner.downscale_calls, [8_000_000]);
        assert!(owner.image_calls.is_empty() && transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn hitpaw_square_and_aspect_targets_match_actual_source_dimensions()
    -> Result<(), Box<dyn Error>> {
        let mut input = fixture_input("HitPawVideoEnhance")?;
        let mut inventory = MediaInventory {
            source: json!("video"),
            width: 720,
            height: 720,
            duration_seconds: Some(3.0),
        };
        input.arguments.insert(
            "model".into(),
            json!({"model":"General Restore Model (2x)","resolution":"4K/UHD"}),
        );
        assert_eq!(
            request(&input, &inventory, None)?.target_dimensions,
            Some([2048, 2048])
        );
        input.arguments.insert(
            "model".into(),
            json!({"model":"General Restore Model (2x)","resolution":"8K"}),
        );
        assert_eq!(
            request(&input, &inventory, None)?.target_dimensions,
            Some([4096, 4096])
        );
        inventory.width = 1919;
        inventory.height = 1081;
        input.arguments.insert(
            "model".into(),
            json!({"model":"General Restore Model (2x)","resolution":"2K/QHD"}),
        );
        let plan = request(&input, &inventory, None)?;
        assert_eq!(plan.target_dimensions, Some([2556, 1440]));
        assert_eq!(plan.body["original_resolution"], json!([1919, 1081]));
        Ok(())
    }
    #[test]
    fn hitpaw_source_preflight_failures_are_exact_and_precede_upload() -> Result<(), Box<dyn Error>>
    {
        for case in oracle()?["preflight_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let inventory = fixture_inventory(&input)?;
            let error = request(&input, &inventory, None)
                .err()
                .ok_or("expected failure")?;
            if case["input"]["node"] == "HitPawGeneralImageEnhance"
                || case["message"]
                    .as_str()
                    .is_some_and(|value| value.starts_with("Selected resolution"))
            {
                assert_eq!(
                    error.to_string(),
                    case["message"].as_str().ok_or("message")?
                );
            }
            let mut owner = InputOwner {
                inventory,
                ..InputOwner::default()
            };
            let mut transport = Transport::default();
            assert!(invoke(&input, credentials(), 100, &mut transport, &mut owner).is_err());
            assert!(
                owner.image_calls.is_empty()
                    && owner.video_calls.is_empty()
                    && transport.heads.is_empty()
            );
        }
        Ok(())
    }
    #[test]
    fn hitpaw_video_duration_epsilon_and_native_upload_failure_are_preserved()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("HitPawVideoEnhance")?;
        let mut inventory = MediaInventory {
            source: json!("native-video"),
            width: 640,
            height: 360,
            duration_seconds: Some(0.49995),
        };
        assert!(request(&input, &inventory, None).is_ok());
        inventory.duration_seconds = Some(3600.00005);
        assert!(request(&input, &inventory, None).is_ok());
        inventory.duration_seconds = Some(3600.0002);
        assert!(request(&input, &inventory, None).is_err());
        inventory.duration_seconds = None;
        let mut owner = InputOwner {
            inventory,
            ..InputOwner::default()
        };
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(HitPawError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(owner.video_calls, [json!("native-video")]);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn hitpaw_response_error_order_and_authoritative_price_match_source()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["response_errors"].as_array().ok_or("errors")? {
            let is_video = case["input"]["node"] == "HitPawVideoEnhance";
            let error = task_creation(&case["requests"][0]["response"], is_video)
                .err()
                .ok_or("expected error")?;
            if case["response_mode"] == "creation-error" {
                assert_eq!(
                    error.to_string(),
                    case["message"].as_str().ok_or("message")?
                );
            } else {
                assert!(matches!(error, HitPawError::Response(_)));
            }
        }
        assert_eq!(
            task_creation(&creation(200, Some(123)), false)?,
            ("source-job-42".into(), 0.123)
        );
        assert!(matches!(
            task_creation(&creation(400, None), false),
            Err(HitPawError::Provider(_))
        ));
        assert!(matches!(
            task_creation(&creation(400, None), true),
            Err(HitPawError::Response(_))
        ));
        Ok(())
    }
    #[test]
    fn hitpaw_real_png_receipts_price_post_poll_and_upload_defaults_are_exact()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("HitPawGeneralImageEnhance")?;
        let mut transport = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        let mut owner = InputOwner::default();
        let result = invoke(&input, credentials(), 100, &mut transport, &mut owner)?;
        assert_eq!(result.media.bytes, png());
        assert_eq!(result.media.receipt, [7, 8, 9]);
        assert_eq!(result.creation_receipt, [7, 8, 9]);
        assert_eq!(result.receipt, [7, 8, 9]);
        assert_eq!(result.usd, 0.123);
        assert_eq!(
            result.creation_response_bytes,
            serde_json::to_vec(&creation(200, Some(123)))?
        );
        assert_eq!(
            result.response_bytes,
            serde_json::to_vec(&poll("completed"))?
        );
        assert_eq!(transport.heads[1].method, HttpMethod::Post);
        assert_eq!(transport.heads[1].path, "/proxy/hitpaw/api/task-status");
        assert_eq!(
            serde_json::from_slice::<Value>(&transport.bodies[1])?,
            json!({"job_id":"source-job-42"})
        );
        assert_eq!(transport.heads[2].secret_id, None);
        assert_eq!(transport.heads[2].headers.len(), 1);
        assert_eq!(owner.image_calls, [json!("actual-png-input")]);
        Ok(())
    }
    #[test]
    fn hitpaw_auth_cost_and_upload_errors_prevent_body_submission() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("HitPawGeneralImageEnhance")?;
        let mut owner = InputOwner::default();
        let mut transport = Transport {
            deny_cost: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(HitPawError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(owner.image_calls.is_empty());
        let mut transport = Transport::default();
        assert!(
            invoke(
                &input,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                100,
                &mut transport,
                &mut owner
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        let mut owner = InputOwner {
            fail_upload: true,
            ..InputOwner::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(HitPawError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn hitpaw_retries_preserve_creation_and_post_poll_bodies_on_fresh_handles()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("HitPawGeneralImageEnhance")?;
        let mut values = events(
            429,
            b"retry".to_vec(),
            vec![Header {
                name: "Retry-After".into(),
                value: "2".into(),
            }],
        );
        values.extend(events(
            200,
            serde_json::to_vec(&creation(200, Some(123)))?,
            vec![],
        ));
        values.extend(events(503, b"retry".to_vec(), vec![]));
        values.extend(events(200, serde_json::to_vec(&poll("completed"))?, vec![]));
        values.extend(events(200, png(), vec![]));
        let mut transport = Transport {
            events: values,
            ..Transport::default()
        };
        invoke(
            &input,
            credentials(),
            100,
            &mut transport,
            &mut InputOwner::default(),
        )?;
        assert_eq!(transport.delays, [2000, 1000]);
        assert_eq!(transport.bodies[0], transport.bodies[1]);
        assert_eq!(transport.bodies[2], transport.bodies[3]);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(
            transport.costs.first().map(|cost| cost.operation.as_str()),
            Some("hitpaw.generation.maximum")
        );
        let mut transport = Transport {
            events: events(503, b"retry".to_vec(), vec![]),
            same_retry_handle: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut InputOwner::default()
            ),
            Err(HitPawError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn hitpaw_source_poll_statuses_delegate_ten_second_delay_and_fail() -> Result<(), Box<dyn Error>>
    {
        let input = fixture_input("HitPawGeneralImageEnhance")?;
        let mut values = events(200, serde_json::to_vec(&creation(200, Some(123)))?, vec![]);
        for status in [" queued ", "processing", "failed"] {
            values.extend(events(200, serde_json::to_vec(&poll(status))?, vec![]));
        }
        let mut transport = Transport {
            events: values,
            ..Transport::default()
        };
        assert!(
            matches!(invoke(&input,credentials(),100,&mut transport,&mut InputOwner::default()),Err(HitPawError::Provider(message))if message.starts_with("Task failed:"))
        );
        assert_eq!(transport.poll_delays, [10000, 10000]);
        assert_eq!(transport.progresses[0].message.as_deref(), Some("queued"));
        assert!(
            task_status(&json!({"code":200,"message":"success","data":{"job_id":"job"}})).is_err()
        );
        Ok(())
    }
    #[test]
    fn hitpaw_cancel_worker_loss_malformed_and_oversized_frames_are_typed()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("HitPawGeneralImageEnhance")?;
        for (mut transport, expected) in [
            (
                Transport {
                    cancelled: true,
                    ..Transport::default()
                },
                ConformanceError::Cancelled,
            ),
            (
                Transport {
                    events: vec![Err(ConformanceError::HostFailure)].into(),
                    ..Transport::default()
                },
                ConformanceError::HostFailure,
            ),
            (
                Transport {
                    events: vec![Ok(ResponseEvent::Cancelled)].into(),
                    ..Transport::default()
                },
                ConformanceError::Cancelled,
            ),
            (
                Transport {
                    events: events(200, vec![b'x'; MAXIMUM_CHUNK_BYTES + 1], vec![]),
                    ..Transport::default()
                },
                ConformanceError::ChunkLimit,
            ),
        ] {
            assert!(
                matches!(invoke(&input,credentials(),100,&mut transport,&mut InputOwner::default()),Err(HitPawError::Transport(error))if error==expected)
            );
        }
        let mut transport = Transport {
            events: events(200, b"{} trailing".to_vec(), vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut InputOwner::default()
            ),
            Err(HitPawError::Json(_))
        ));
        let mut values = vec![Ok(ResponseEvent::Head {
            status: 200,
            headers: vec![],
        })];
        for _ in 0..=MAXIMUM_RESPONSE_BYTES / MAXIMUM_CHUNK_BYTES {
            values.push(Ok(ResponseEvent::Chunk(vec![b'x'; MAXIMUM_CHUNK_BYTES])));
        }
        let mut transport = Transport {
            events: values.into(),
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut InputOwner::default()
            ),
            Err(HitPawError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn hitpaw_real_png_materialization_failure_is_atomic_and_clean_retry_is_possible()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("HitPawGeneralImageEnhance")?;
        let mut transport = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        let mut decoder = RejectDecoder { seen: false };
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut InputOwner::default(),
                &mut decoder,
                &mut sink
            ),
            Err(HitPawError::Transport(ConformanceError::HostFailure))
        ));
        assert!(decoder.seen);
        assert_eq!(sink.batches, 0);
        let mut transport = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        let result = invoke(
            &input,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            100,
            &mut transport,
            &mut InputOwner::default(),
        )?;
        assert_eq!(result.media.bytes, png());
        assert_eq!(
            transport.heads[0].secret_id.as_deref(),
            Some("sim_org.api_key")
        );
        Ok(())
    }

    struct PollInputOwner(InputOwner);
    impl HitPawInputOwner for PollInputOwner {
        fn inventory(&mut self, input: &Input) -> Result<MediaInventory, ConformanceError> {
            self.0.inventory(input)
        }
        fn downscale_image(
            &mut self,
            image: &Value,
            total: u64,
            filter: &str,
        ) -> Result<Value, ConformanceError> {
            self.0.downscale_image(image, total, filter)
        }
        fn upload_image(
            &mut self,
            image: &Value,
            mime: Option<&str>,
            total: Option<u64>,
        ) -> Result<String, ConformanceError> {
            self.0.upload_image(image, mime, total)
        }
        fn upload_video(
            &mut self,
            video: &Value,
            container: &str,
            codec: &str,
        ) -> Result<String, ConformanceError> {
            assert_eq!((container, codec), ("mp4", "h264"));
            self.0.video_calls.push(video.clone());
            Ok("https://storage.fixture.invalid/source-video".into())
        }
    }
    fn status_poll(node: &str, status: &str) -> Value {
        let mut response = poll(status);
        response["data"]["res_url"] = json!(if node == "HitPawVideoEnhance" {
            "https://fixture.invalid/result.mp4"
        } else {
            "https://fixture.invalid/result.png"
        });
        response
    }
    fn status_download(node: &str) -> Vec<u8> {
        if node == "HitPawVideoEnhance" {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/ltxv/response-video.mp4"
            ))
            .to_vec()
        } else {
            png()
        }
    }
    #[test]
    fn hitpaw_control_padded_completed_statuses_download_exact_bytes_for_both_nodes()
    -> Result<(), Box<dyn Error>> {
        for node in ["HitPawGeneralImageEnhance", "HitPawVideoEnhance"] {
            let input = fixture_input(node)?;
            for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
                let raw = status_poll(node, &format!("{separator} CoMpLeTeD {separator}"));
                let response_bytes = serde_json::to_vec(&raw)?;
                let mut values =
                    events(200, serde_json::to_vec(&creation(200, Some(123)))?, vec![]);
                values.extend(events(200, response_bytes.clone(), vec![]));
                values.extend(events(200, status_download(node), vec![]));
                let mut transport = Transport {
                    events: values,
                    ..Transport::default()
                };
                let mut owner = PollInputOwner(InputOwner::default());
                let result = invoke(&input, credentials(), 100, &mut transport, &mut owner)?;
                assert_eq!(result.is_video, node == "HitPawVideoEnhance");
                assert_eq!(result.media.bytes, status_download(node));
                assert_eq!(result.response_bytes, response_bytes);
                assert_eq!(result.media.receipt, [7, 8, 9]);
                assert_eq!(result.usd, 0.123);
                assert_eq!(transport.heads.len(), 3);
                assert_eq!(transport.costs.len(), 1);
                assert_eq!(transport.costs[0].operation, "hitpaw.generation.maximum");
                assert_eq!(transport.heads[1].method, HttpMethod::Post);
                assert_eq!(transport.heads[1].path, "/proxy/hitpaw/api/task-status");
                assert_eq!(
                    serde_json::from_slice::<Value>(&transport.bodies[1])?,
                    json!({"job_id":"source-job-42"})
                );
                assert_eq!(transport.heads[2].method, HttpMethod::Get);
                assert_eq!(transport.heads[2].secret_id, None);
                assert!(transport.poll_delays.is_empty());
                assert!(transport.progresses.is_empty());
                assert_eq!(
                    owner.0.video_calls.len(),
                    usize::from(node == "HitPawVideoEnhance")
                );
                assert_eq!(
                    owner.0.image_calls.len(),
                    usize::from(node != "HitPawVideoEnhance")
                );
            }
        }
        Ok(())
    }
    #[test]
    fn hitpaw_control_queued_statuses_preserve_budget_and_nonstrip_statuses_stay_active()
    -> Result<(), Box<dyn Error>> {
        for node in ["HitPawGeneralImageEnhance", "HitPawVideoEnhance"] {
            let input = fixture_input(node)?;
            let mut values = events(200, serde_json::to_vec(&creation(200, Some(123)))?, vec![]);
            for _ in 0..481 {
                values.extend(events(
                    200,
                    serde_json::to_vec(&status_poll(node, "\u{001c} QuEuEd \u{001f}"))?,
                    vec![],
                ));
            }
            values.extend(events(
                200,
                serde_json::to_vec(&status_poll(node, "\u{001e}Completed\u{001d}"))?,
                vec![],
            ));
            values.extend(events(200, status_download(node), vec![]));
            let mut transport = Transport {
                events: values,
                ..Transport::default()
            };
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut PollInputOwner(InputOwner::default()),
            )?;
            assert_eq!(transport.heads.len(), 484);
            assert_eq!(transport.poll_delays, vec![10_000; 481]);
            assert_eq!(transport.progresses.len(), 481);
            assert!(
                transport
                    .progresses
                    .iter()
                    .all(|progress| progress.message.as_deref() == Some("queued"))
            );
            for (raw, expected) in [
                ("\u{200b}Completed\u{200b}", "\u{200b}completed\u{200b}"),
                ("\u{feff}Completed\u{feff}", "\u{feff}completed\u{feff}"),
                ("\u{001c}\u{001d}\u{001e}\u{001f}", ""),
            ] {
                let mut values =
                    events(200, serde_json::to_vec(&creation(200, Some(123)))?, vec![]);
                values.extend(events(
                    200,
                    serde_json::to_vec(&status_poll(node, raw))?,
                    vec![],
                ));
                values.extend(events(
                    200,
                    serde_json::to_vec(&status_poll(node, "completed"))?,
                    vec![],
                ));
                values.extend(events(200, status_download(node), vec![]));
                let mut transport = Transport {
                    events: values,
                    ..Transport::default()
                };
                invoke(
                    &input,
                    credentials(),
                    100,
                    &mut transport,
                    &mut PollInputOwner(InputOwner::default()),
                )?;
                assert_eq!(transport.heads.len(), 4);
                assert_eq!(transport.poll_delays, [10_000]);
                assert_eq!(transport.progresses[0].message.as_deref(), Some(expected));
            }
        }
        Ok(())
    }
    #[test]
    fn hitpaw_control_failed_statuses_keep_raw_error_and_active_limit_and_cancel()
    -> Result<(), Box<dyn Error>> {
        for node in ["HitPawGeneralImageEnhance", "HitPawVideoEnhance"] {
            let input = fixture_input(node)?;
            for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
                let raw = status_poll(node, &format!("{separator} FAILED {separator}"));
                let expected = format!("Task failed: {}", serde_json::to_string(&raw)?);
                let mut values =
                    events(200, serde_json::to_vec(&creation(200, Some(123)))?, vec![]);
                values.extend(events(200, serde_json::to_vec(&raw)?, vec![]));
                let mut transport = Transport {
                    events: values,
                    ..Transport::default()
                };
                assert!(
                    matches!(invoke(&input, credentials(), 100, &mut transport, &mut PollInputOwner(InputOwner::default())), Err(HitPawError::Provider(message)) if message == expected)
                );
                assert_eq!(transport.heads.len(), 2);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.poll_delays.is_empty());
                assert!(transport.progresses.is_empty());
            }
            let mut values = events(200, serde_json::to_vec(&creation(200, Some(123)))?, vec![]);
            for _ in 0..480 {
                values.extend(events(
                    200,
                    serde_json::to_vec(&status_poll(node, "\u{001c}Processing\u{001f}"))?,
                    vec![],
                ));
            }
            let mut transport = Transport {
                events: values,
                ..Transport::default()
            };
            assert!(matches!(
                invoke(
                    &input,
                    credentials(),
                    100,
                    &mut transport,
                    &mut PollInputOwner(InputOwner::default())
                ),
                Err(HitPawError::Transport(ConformanceError::PollLimit))
            ));
            assert_eq!(transport.heads.len(), 481);
            assert_eq!(transport.poll_delays, vec![10_000; 479]);
            let mut transport = Transport {
                cancelled: true,
                ..Transport::default()
            };
            let mut owner = PollInputOwner(InputOwner::default());
            assert!(matches!(
                invoke(&input, credentials(), 100, &mut transport, &mut owner),
                Err(HitPawError::Transport(ConformanceError::Cancelled))
            ));
            assert!(transport.heads.is_empty());
            assert!(transport.bodies.is_empty());
            assert!(transport.costs.is_empty());
            assert!(owner.0.video_calls.is_empty());
            assert!(owner.0.image_calls.is_empty());
        }
        Ok(())
    }
}
