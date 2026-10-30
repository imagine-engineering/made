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
pub const NAMESPACE: &str = "zed.sim.provider.ltxv";
pub const ALIASES: &[&str] = &["ltx"];
pub const SOURCE_SHA256: &str = "f31e70cd810520baf844badfac39ac5b923ddc8de35dcad9a2159189c5ea0c58";
pub const MAXIMUM_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
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
        feature_id: "COMFY-NODE-0359",
        node_identifier: "LtxvApiImageToVideo",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0360",
        node_identifier: "LtxvApiTextToVideo",
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
        feature_id: "COMFY-API-EXT-0090",
        method: "POST",
        path: "/proxy/ltx/v1/image-to-video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0091",
        method: "POST",
        path: "/proxy/ltx/v1/text-to-video",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum LtxvError {
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
impl fmt::Display for LtxvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Ltxv HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for LtxvError {}
impl From<ConformanceError> for LtxvError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for LtxvError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, LtxvError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/ltxv/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, LtxvError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(LtxvError::Input("Unknown Ltxv node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(LtxvError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(LtxvError::Input("Unknown Ltxv source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(LtxvError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(LtxvError::Input("Missing Ltxv source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, LtxvError> {
    input
        .arguments
        .get(name)
        .ok_or(LtxvError::Input("Missing Ltxv argument"))
}
fn text(value: &Value) -> Result<&str, LtxvError> {
    value
        .as_str()
        .ok_or(LtxvError::Input("Expected Ltxv string"))
}
fn endpoint(value: &str) -> Result<bool, LtxvError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(LtxvError::Response("Invalid Ltxv response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(LtxvError::Response("Invalid Ltxv response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(LtxvError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(LtxvError::Response("Invalid Ltxv endpoint authority"));
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
) -> Result<RequestHead, LtxvError> {
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
            return Err(LtxvError::Input("Missing Comfy API credentials"));
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponsePolicy {
    pub timeout_milliseconds: u64,
    pub monitor_progress: bool,
    pub wait_label: &'static str,
    pub final_label: &'static str,
}
impl ResponsePolicy {
    fn source() -> Self {
        Self {
            timeout_milliseconds: 3_600_000,
            monitor_progress: true,
            wait_label: "Waiting for server",
            final_label: "Completed",
        }
    }
}
pub trait LtxvTransport: ProviderTransport {
    fn configure_response(
        &mut self,
        handle: StreamHandle,
        policy: &ResponsePolicy,
    ) -> Result<(), ConformanceError>;
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
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageUploadPolicy {
    pub maximum_images: u64,
    pub mime_type: &'static str,
    pub total_pixels: u64,
    pub wait_label: &'static str,
    pub show_batch_index: bool,
}
impl ImageUploadPolicy {
    fn source() -> Self {
        Self {
            maximum_images: 1,
            mime_type: "image/png",
            total_pixels: 4_194_304,
            wait_label: "Uploading",
            show_batch_index: true,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ImageUpload {
    pub url: String,
    pub receipt: Vec<u8>,
}
pub trait LtxvImageOwner {
    fn number_of_images(&mut self, image: &Value) -> Result<u64, ConformanceError>;
    fn upload_png(
        &mut self,
        image: &Value,
        policy: &ImageUploadPolicy,
    ) -> Result<ImageUpload, ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: &'static str,
    pub body: Value,
    pub upload_receipt: Option<Vec<u8>>,
}
fn duration(input: &Input) -> Result<i64, LtxvError> {
    argument(input, "duration")?
        .as_i64()
        .ok_or(LtxvError::Input("Expected source integer duration"))
}
fn api_integer(value: &Value) -> Result<i64, LtxvError> {
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_str() {
        return value
            .parse()
            .map_err(|_| LtxvError::Input("Invalid source integer"));
    }
    Err(LtxvError::Input("Invalid source integer"))
}
fn api_boolean(value: &Value) -> Result<bool, LtxvError> {
    if let Some(value) = value.as_bool() {
        return Ok(value);
    }
    if let Some(value) = value.as_str() {
        match value.to_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "on" | "1" => return Ok(true),
            "false" | "f" | "no" | "n" | "off" | "0" => return Ok(false),
            _ => {}
        }
    }
    if let Some(value) = value.as_i64().filter(|value| [0, 1].contains(value)) {
        return Ok(value == 1);
    }
    Err(LtxvError::Input("Invalid source boolean"))
}
fn source_trim(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn preflight(input: &Input) -> Result<(), LtxvError> {
    let prompt = text(argument(input, "prompt")?)?;
    let length = source_trim(prompt).chars().count();
    if length == 0 || length > 10000 {
        return Err(LtxvError::Input(
            "Source prompt outside 1–10000 character bounds",
        ));
    }
    let model = text(argument(input, "model")?)?;
    let resolution = text(argument(input, "resolution")?)?;
    if duration(input)? > 10
        && (model != "LTX-2 (Fast)"
            || resolution != "1920x1080"
            || argument(input, "fps")? != &json!(25))
    {
        return Err(LtxvError::Input(
            "Durations over 10s are only available for the Fast model at 1920x1080 resolution and 25 FPS.",
        ));
    }
    Ok(())
}
pub fn request(input: &Input, owner: &mut dyn LtxvImageOwner) -> Result<Request, LtxvError> {
    let input = normalized(input)?;
    preflight(&input)?;
    let mut upload_receipt = None;
    let image_uri = if input.node == "LtxvApiImageToVideo" {
        let image = argument(&input, "image")?;
        if owner.number_of_images(image)? != 1 {
            return Err(LtxvError::Input(
                "Currently only one input image is supported.",
            ));
        }
        let uploaded = owner.upload_png(image, &ImageUploadPolicy::source())?;
        if endpoint(&uploaded.url)?
            || uploaded.receipt.is_empty()
            || uploaded.receipt.len() > MAXIMUM_RECEIPT_BYTES
        {
            return Err(LtxvError::Response(
                "Invalid canonical PNG upload receipt or URL",
            ));
        }
        upload_receipt = Some(uploaded.receipt);
        Some(uploaded.url)
    } else {
        None
    };
    let source = oracle()?;
    let model = source["models"]
        .get(text(argument(&input, "model")?)?)
        .and_then(Value::as_str)
        .ok_or(LtxvError::Input("Unknown source model"))?;
    let mut body = json!({"prompt":argument(&input,"prompt")?,"model":model,"duration":duration(&input)?,"resolution":argument(&input,"resolution")?});
    for (name, boolean) in [("fps", false), ("generate_audio", true)] {
        let value = argument(&input, name)?;
        if !value.is_null() {
            body[name] = if boolean {
                json!(api_boolean(value)?)
            } else {
                json!(api_integer(value)?)
            };
        }
    }
    if let Some(uri) = image_uri {
        body["image_uri"] = json!(uri);
    }
    Ok(Request {
        path: if input.node == "LtxvApiImageToVideo" {
            "/proxy/ltx/v1/image-to-video"
        } else {
            "/proxy/ltx/v1/text-to-video"
        },
        body,
        upload_receipt,
    })
}
pub fn maximum_cost_microunits(input: &Input) -> Result<u64, LtxvError> {
    let input = normalized(input)?;
    let model = text(argument(&input, "model")?)?.to_lowercase();
    let resolution = text(argument(&input, "resolution")?)?;
    let source = oracle()?;
    let rate = source["price_rates"][model][resolution]
        .as_f64()
        .ok_or(LtxvError::Input("Source price unavailable for widgets"))?;
    let scaled = (rate * 1_000_000.0).round();
    if !scaled.is_finite() || !(0.0..=1_000_000_000.0).contains(&scaled) {
        return Err(LtxvError::Input("Invalid source price rate"));
    }
    let duration = u64::try_from(duration(&input)?)
        .map_err(|_| LtxvError::Input("Invalid priced duration"))?;
    (scaled as u64)
        .checked_mul(duration)
        .ok_or(LtxvError::Input("Source price overflow"))
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub approved_maximum_microunits: u64,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn LtxvTransport,
    owner: &mut dyn LtxvImageOwner,
) -> Result<ResultValue, LtxvError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    preflight(&input)?;
    let cost = maximum_cost_microunits(&input)?;
    if cost > maximum_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    let path = if input.node == "LtxvApiImageToVideo" {
        "/proxy/ltx/v1/image-to-video"
    } else {
        "/proxy/ltx/v1/text-to-video"
    };
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "ltxv.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits: cost,
        },
    )?;
    if !approval.accepted
        || approval.approved_microunits < cost
        || approval.receipt.is_empty()
        || approval.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    transport.check_cancelled(Some(handle))?;
    let plan = request(&input, owner)?;
    let body = serde_json::to_vec(&plan.body)?;
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let (bytes, receipt, handle) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    if bytes.is_empty() {
        return Err(LtxvError::Response("Empty raw video response"));
    }
    transport.report_progress(
        handle,
        &Progress {
            sequence: 0,
            completed: 100,
            total: 100,
            message: Some("Completed".into()),
        },
    )?;
    Ok(ResultValue {
        plan,
        bytes,
        receipt,
        approval_receipt: approval.receipt,
        approved_maximum_microunits: cost,
    })
}
pub trait LtxvVideoMaterializer {
    fn materialize_video(
        &mut self,
        result: &ResultValue,
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub struct MaterializationOwners<'a> {
    pub image: &'a mut dyn LtxvImageOwner,
    pub video: &'a mut dyn LtxvVideoMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn LtxvTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), LtxvError> {
    let result = invoke(
        input,
        credentials,
        maximum_microunits,
        transport,
        owners.image,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.video.materialize_video(&result)?;
    let registry = TypeRegistry::built_in()
        .map_err(|_| LtxvError::Response("Missing canonical type registry"))?;
    let type_id = registry
        .resolve("VIDEO")
        .map_err(|_| LtxvError::Response("Missing canonical VIDEO type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| LtxvError::Response("Missing canonical VIDEO family"))?;
    let expected = vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }];
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, owners.sink).map_err(ConformanceError::from)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct RetryPolicy {
    ordinary_limit: u32,
    rate_limit: u32,
    backoff: f64,
    download: bool,
}
const GENERATION_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 1,
    rate_limit: 16,
    backoff: 2.0,
    download: false,
};
#[cfg(test)]
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn LtxvTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), LtxvError> {
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
    transport: &mut dyn LtxvTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), LtxvError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(LtxvError::Http {
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
                    return Err(LtxvError::Http {
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
    transport: &mut dyn LtxvTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), LtxvError> {
    transport.configure_response(handle, &ResponsePolicy::source())?;
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "ltxv.generation.maximum".into(),
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
                let total_header_bytes = headers
                    .iter()
                    .try_fold(0usize, |total, header| {
                        total
                            .checked_add(header.name.len())?
                            .checked_add(header.value.len())
                    })
                    .ok_or(ConformanceError::HeaderLimit)?;
                if headers.len() > 16
                    || total_header_bytes > 16 * 1024
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
                    return Err(LtxvError::Http {
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

#[derive(Default)]
struct SourceProbeOwner {
    operations: Vec<Value>,
    fail: bool,
}
impl LtxvImageOwner for SourceProbeOwner {
    fn number_of_images(&mut self, image: &Value) -> Result<u64, ConformanceError> {
        let shape = image["shape"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if shape.len() >= 4 {
            shape
                .first()
                .and_then(Value::as_u64)
                .ok_or(ConformanceError::InvalidRequest)
        } else {
            Ok(1)
        }
    }
    fn upload_png(
        &mut self,
        image: &Value,
        policy: &ImageUploadPolicy,
    ) -> Result<ImageUpload, ConformanceError> {
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        self.operations.push(json!({"operation":"upload_images","image":image,"max_images":policy.maximum_images,"mime_type":policy.mime_type,"wait_label":policy.wait_label,"show_batch_index":policy.show_batch_index,"total_pixels":policy.total_pixels}));
        Ok(ImageUpload {
            url: "https://storage.fixture.invalid/source-image.png".into(),
            receipt: vec![4, 5, 6],
        })
    }
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
    if source["source_sha256"]["comfy_api_nodes/nodes_ltxv.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let mut owner = SourceProbeOwner::default();
        let plan = request(&input, &mut owner).map_err(|_| ConformanceError::InvalidRequest)?;
        let expected = case["requests"]
            .as_array()
            .and_then(|requests| requests.first())
            .ok_or(ConformanceError::InvalidRequest)?;
        if plan.path != expected["path"]
            || plan.body != expected["body"]
            || expected["method"] != "POST"
            || expected["options"] != json!({"as_binary":true,"max_retries":1})
            || json!(owner.operations) != case["media"]
        {
            return Err(ConformanceError::InvalidRequest);
        }
        if case["outputs"][0]["canonical_video_from_file"]["bytes"] != source["response_bytes"]
            || case["outputs"][0]["canonical_video_from_file"]["sha256"]
                != source["response_sha256"]
        {
            return Err(ConformanceError::InvalidRequest);
        }
        covered.insert(input.node);
        paths.insert(plan.path);
    }
    for case in source["preflight_errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let mut owner = SourceProbeOwner::default();
        if request(&input, &mut owner).is_ok()
            || json!(owner.operations) != case["media"]
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
        || paths != ROUTES.iter().map(|route| route.path).collect()
    {
        return Err(ConformanceError::InvalidRequest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        CostResponse, MaterializationError, MaterializedBatch, Progress, SignedPutRequest,
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
        policies: Vec<(StreamHandle, ResponsePolicy)>,
        fail_policy: bool,
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
                .get_mut(
                    handle
                        .slot
                        .checked_sub(1)
                        .ok_or(ConformanceError::InvalidHandle)? as usize,
                )
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
    impl LtxvTransport for Transport {
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
        fn configure_response(
            &mut self,
            handle: StreamHandle,
            policy: &ResponsePolicy,
        ) -> Result<(), ConformanceError> {
            self.policies.push((handle, policy.clone()));
            if self.fail_policy {
                Err(ConformanceError::HostFailure)
            } else {
                Ok(())
            }
        }
    }

    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn video() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/ltxv/response-video.mp4"
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
    fn input(node: &str) -> Result<Input, Box<dyn Error>> {
        let source = oracle()?;
        let case = source["cases"]
            .as_array()
            .and_then(|cases| cases.iter().find(|case| case["input"]["node"] == node))
            .ok_or("source case")?;
        Ok(serde_json::from_value(case["input"].clone())?)
    }
    #[test]
    fn ltxv_source_claims_and_request_wrapper_order_are_exact() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/ltxv/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 2);
        assert_eq!(ROUTES.len(), 2);
        assert_eq!(ALIASES, ["ltx"]);
        Ok(())
    }
    #[test]
    fn ltxv_all_ui_success_and_error_requests_match_actual_source() -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        for case in source["cases"].as_array().ok_or("source cases")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            let plan = request(&input, &mut owner)?;
            assert_eq!(plan.body, case["requests"][0]["body"], "{}", input.node);
            assert_eq!(json!(owner.operations), case["media"], "{}", input.node);
        }
        for case in source["preflight_errors"]
            .as_array()
            .ok_or("source errors")?
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            assert!(request(&input, &mut owner).is_err());
            assert_eq!(json!(owner.operations), case["media"], "{}", input.node);
        }
        Ok(())
    }
    #[test]
    fn ltxv_defaults_override_model_audio_true_and_preserve_optional_omission()
    -> Result<(), Box<dyn Error>> {
        let source = input("LtxvApiTextToVideo")?;
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert_eq!(plan.body["fps"], 25);
        assert_eq!(plan.body["generate_audio"], false);
        assert_eq!(plan.body["prompt"], "  source  ");
        assert!(plan.body.get("image_uri").is_none());
        let mut omitted = source;
        omitted.arguments.insert("fps".into(), Value::Null);
        omitted
            .arguments
            .insert("generate_audio".into(), Value::Null);
        let plan = request(&omitted, &mut SourceProbeOwner::default())?;
        assert!(plan.body.get("fps").is_none());
        assert!(plan.body.get("generate_audio").is_none());
        Ok(())
    }
    #[test]
    fn ltxv_all_source_price_badge_literal_rates_match_duration_multiplication()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["price_cases"].as_array().ok_or("prices")? {
            let mut source = input("LtxvApiTextToVideo")?;
            for field in ["model", "resolution", "duration"] {
                source.arguments.insert(field.into(), case[field].clone());
            }
            assert_eq!(
                maximum_cost_microunits(&source)?,
                case["maximum_microunits"].as_u64().ok_or("price")?
            );
        }
        Ok(())
    }
    #[test]
    fn ltxv_binary_response_has_no_poll_or_download_and_keeps_exact_bytes()
    -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            events: events(200, video(), vec![]),
            ..Transport::default()
        };
        let result = invoke(
            &input("LtxvApiTextToVideo")?,
            credentials(),
            2_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(result.bytes, video());
        assert_eq!(result.receipt, [7, 8, 9]);
        assert_eq!(result.approval_receipt, [42]);
        assert_eq!(result.approved_maximum_microunits, 320_000);
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs[0].operation, "ltxv.generation.maximum");
        assert_eq!(transport.costs[0].maximum_microunits, 320_000);
        assert_eq!(transport.policies[0].1, ResponsePolicy::source());
        assert_eq!(
            transport.progresses[0].message.as_deref(),
            Some("Completed")
        );
        Ok(())
    }
    #[test]
    fn ltxv_auth_policy_and_cost_failures_send_no_unapproved_body() -> Result<(), Box<dyn Error>> {
        let source = input("LtxvApiImageToVideo")?;
        let mut owner = SourceProbeOwner::default();
        let mut transport = Transport::default();
        assert!(
            invoke(
                &source,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                2_000_000,
                &mut transport,
                &mut owner
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        assert!(owner.operations.is_empty());
        assert!(matches!(
            invoke(&source, credentials(), 319_999, &mut transport, &mut owner),
            Err(LtxvError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(transport.heads.is_empty());
        transport.deny_cost = true;
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut owner
            ),
            Err(LtxvError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(owner.operations.is_empty());
        transport.deny_cost = false;
        transport.fail_policy = true;
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut owner
            ),
            Err(LtxvError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn ltxv_png_upload_policy_and_native_upload_failure_are_exact() -> Result<(), Box<dyn Error>> {
        let source = input("LtxvApiImageToVideo")?;
        let mut owner = SourceProbeOwner::default();
        let plan = request(&source, &mut owner)?;
        assert_eq!(plan.upload_receipt, Some(vec![4, 5, 6]));
        assert_eq!(
            plan.body["image_uri"],
            "https://storage.fixture.invalid/source-image.png"
        );
        assert_eq!(owner.operations[0]["mime_type"], "image/png");
        assert_eq!(owner.operations[0]["max_images"], 1);
        assert_eq!(owner.operations[0]["total_pixels"], 4_194_304);
        let mut owner = SourceProbeOwner {
            fail: true,
            ..SourceProbeOwner::default()
        };
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut owner
            ),
            Err(LtxvError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn ltxv_source_timeout_and_retry_budgets_are_delegated_to_supervisor()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/ltx/v1/text-to-video",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut transport = Transport {
            events: events(503, b"retry".to_vec(), vec![]),
            ..Transport::default()
        };
        transport.events.extend(events(200, video(), vec![]));
        assert_eq!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                &mut transport
            )?
            .0,
            video()
        );
        assert_eq!(transport.delays, [1000]);
        assert_eq!(transport.policies.len(), 2);
        assert_eq!(transport.policies[0].1.timeout_milliseconds, 3_600_000);
        assert_eq!(transport.bodies[0], transport.bodies[1]);
        let mut exhausted = Transport {
            events: events(503, b"retry".to_vec(), vec![]),
            ..Transport::default()
        };
        exhausted
            .events
            .extend(events(503, b"retry".to_vec(), vec![]));
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                &mut exhausted
            ),
            Err(LtxvError::Http { status: 503, .. })
        ));
        assert_eq!(exhausted.heads.len(), 2);
        Ok(())
    }
    #[test]
    fn ltxv_rate_limit_retry_after_and_fresh_handle_identity_are_checked()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/ltx/v1/text-to-video",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut transport = Transport {
            events: events(
                429,
                b"rate".to_vec(),
                vec![Header {
                    name: "Retry-After".into(),
                    value: "2".into(),
                }],
            ),
            ..Transport::default()
        };
        transport.events.extend(events(200, video(), vec![]));
        exchange_with_policy(
            &head,
            b"{}",
            None,
            MAXIMUM_RESPONSE_BYTES,
            GENERATION_RETRY,
            &mut transport,
        )?;
        assert_eq!(transport.delays, [2000]);
        let mut stale = Transport {
            same_retry_handle: true,
            events: events(503, b"retry".to_vec(), vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                &mut stale
            ),
            Err(LtxvError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn ltxv_malformed_oversized_and_empty_binary_responses_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/ltx/v1/text-to-video",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        for sequence in [
            vec![Ok(ResponseEvent::Chunk(vec![1]))],
            vec![Ok(ResponseEvent::Head {
                status: 200,
                headers: vec![Header {
                    name: "bad\nheader".into(),
                    value: "x".into(),
                }],
            })],
            vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES + 1])),
            ],
            vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Completed(vec![])),
            ],
        ] {
            let mut transport = Transport {
                events: sequence.into(),
                ..Transport::default()
            };
            assert!(
                exchange_with_policy(
                    &head,
                    b"{}",
                    None,
                    MAXIMUM_RESPONSE_BYTES,
                    GENERATION_RETRY,
                    &mut transport
                )
                .is_err()
            );
        }
        let mut empty = Transport {
            events: events(200, vec![], vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &input("LtxvApiTextToVideo")?,
                credentials(),
                2_000_000,
                &mut empty,
                &mut SourceProbeOwner::default()
            ),
            Err(LtxvError::Response(_))
        ));
        Ok(())
    }
    #[test]
    fn ltxv_cancel_worker_loss_and_fresh_reinvocation_are_atomic() -> Result<(), Box<dyn Error>> {
        let source = input("LtxvApiTextToVideo")?;
        let mut cancelled = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut cancelled,
                &mut SourceProbeOwner::default()
            ),
            Err(LtxvError::Transport(ConformanceError::Cancelled))
        ));
        assert!(cancelled.heads.is_empty());
        let mut lost = Transport::default();
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut lost,
                &mut SourceProbeOwner::default()
            ),
            Err(LtxvError::Transport(ConformanceError::HostFailure))
        ));
        let mut fresh = Transport {
            events: events(200, video(), vec![]),
            ..Transport::default()
        };
        assert_eq!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut fresh,
                &mut SourceProbeOwner::default()
            )?
            .bytes,
            video()
        );
        Ok(())
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
    #[derive(Default)]
    struct RejectDecoder {
        seen: bool,
    }
    impl LtxvVideoMaterializer for RejectDecoder {
        fn materialize_video(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            assert_eq!(result.bytes, video());
            assert_eq!(result.receipt, [7, 8, 9]);
            self.seen = true;
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn ltxv_encoded_source_video_reaches_canonical_rejection_without_placeholder()
    -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            events: events(200, video(), vec![]),
            ..Transport::default()
        };
        let mut image = SourceProbeOwner::default();
        let mut decoder = RejectDecoder::default();
        let mut sink = Sink::default();
        let mut owners = MaterializationOwners {
            image: &mut image,
            video: &mut decoder,
            sink: &mut sink,
        };
        assert!(matches!(
            invoke_and_materialize(
                &input("LtxvApiTextToVideo")?,
                credentials(),
                2_000_000,
                &mut transport,
                &mut owners
            ),
            Err(LtxvError::Transport(ConformanceError::HostFailure))
        ));
        assert!(decoder.seen);
        assert_eq!(sink.batches, 0);
        Ok(())
    }
    #[test]
    fn ltxv_sixteen_source_rate_retries_and_response_aggregate_limits_are_exact()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/ltx/v1/text-to-video",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut rates = Transport::default();
        for _ in 0..17 {
            rates.events.extend(events(429, b"rate".to_vec(), vec![]));
        }
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                &mut rates
            ),
            Err(LtxvError::Http { status: 429, .. })
        ));
        assert_eq!(rates.heads.len(), 17);
        assert_eq!(rates.delays.len(), 16);
        assert_eq!(rates.delays[0], 1000);
        assert_eq!(rates.delays[15], 30_000);
        let mut aggregate = Transport::default();
        aggregate.events.push_back(Ok(ResponseEvent::Head {
            status: 200,
            headers: vec![],
        }));
        for _ in 0..MAXIMUM_RESPONSE_BYTES / MAXIMUM_CHUNK_BYTES {
            aggregate
                .events
                .push_back(Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES])));
        }
        aggregate
            .events
            .push_back(Ok(ResponseEvent::Chunk(vec![1])));
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                &mut aggregate
            ),
            Err(LtxvError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn ltxv_python_prompt_whitespace_rejects_before_transport() -> Result<(), Box<dyn Error>> {
        for node in ["LtxvApiTextToVideo", "LtxvApiImageToVideo"] {
            for prompt in [
                "\u{1c}", "\u{1d}", "\u{1e}", "\u{1f}", " \t\n", "\u{85}", "\u{a0}", "\u{2028}", "",
            ] {
                let mut input = input(node)?;
                input.arguments.insert("prompt".into(), json!(prompt));
                let mut owner = SourceProbeOwner {
                    fail: true,
                    ..SourceProbeOwner::default()
                };
                assert!(matches!(
                    request(&input, &mut owner),
                    Err(LtxvError::Input(
                        "Source prompt outside 1–10000 character bounds"
                    ))
                ));
                let mut transport = Transport::default();
                assert!(matches!(
                    invoke(&input, credentials(), 2_000_000, &mut transport, &mut owner),
                    Err(LtxvError::Input(
                        "Source prompt outside 1–10000 character bounds"
                    ))
                ));
                assert!(transport.heads.is_empty());
                assert!(transport.costs.is_empty());
                assert!(transport.bodies.is_empty());
                assert!(owner.operations.is_empty());
            }
            let mut too_long = input(node)?;
            too_long
                .arguments
                .insert("prompt".into(), json!("x".repeat(10001)));
            assert!(matches!(
                request(&too_long, &mut SourceProbeOwner::default()),
                Err(LtxvError::Input(
                    "Source prompt outside 1–10000 character bounds"
                ))
            ));
        }
        Ok(())
    }

    #[test]
    fn ltxv_python_prompt_whitespace_preserves_original_request_bytes() -> Result<(), Box<dyn Error>>
    {
        for node in ["LtxvApiTextToVideo", "LtxvApiImageToVideo"] {
            for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                for prompt in [
                    format!("{separator}{}{separator}", "x".repeat(10000)),
                    "\u{200b}".into(),
                    "\u{feff}".into(),
                    "\u{1c}x\u{1f}x\u{1c}".into(),
                ] {
                    let mut input = input(node)?;
                    input.arguments.insert("prompt".into(), json!(prompt));
                    let mut request_owner = SourceProbeOwner::default();
                    let plan = request(&input, &mut request_owner)?;
                    assert_eq!(plan.body["prompt"], input.arguments["prompt"]);
                    let mut invoked_owner = SourceProbeOwner::default();
                    let mut transport = Transport::default();
                    assert!(matches!(
                        invoke(
                            &input,
                            credentials(),
                            2_000_000,
                            &mut transport,
                            &mut invoked_owner
                        ),
                        Err(LtxvError::Transport(ConformanceError::HostFailure))
                    ));
                    assert_eq!(transport.heads.len(), 1);
                    assert_eq!(transport.costs.len(), 1);
                    assert_eq!(
                        transport.costs.first().ok_or("cost absent")?.operation,
                        "ltxv.generation.maximum"
                    );
                    assert_eq!(transport.bodies, [serde_json::to_vec(&plan.body)?]);
                    assert_eq!(request_owner.operations, invoked_owner.operations);
                    assert_eq!(
                        request_owner.operations.len(),
                        usize::from(node == "LtxvApiImageToVideo")
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn ltxv_python_prompt_phase_preserves_duration_upload_and_authority()
    -> Result<(), Box<dyn Error>> {
        let mut input = input("LtxvApiImageToVideo")?;
        input.arguments.insert("prompt".into(), json!("\u{1c}"));
        input.arguments.insert("duration".into(), json!(16));
        input.arguments.insert("model".into(), json!("LTX-2 (Pro)"));
        input
            .arguments
            .insert("image".into(), json!({"shape":[2,1,1,3]}));
        let mut owner = SourceProbeOwner {
            fail: true,
            ..SourceProbeOwner::default()
        };
        assert!(matches!(
            request(&input, &mut owner),
            Err(LtxvError::Input(
                "Source prompt outside 1–10000 character bounds"
            ))
        ));
        assert!(owner.operations.is_empty());
        input.arguments.insert("prompt".into(), json!("source"));
        assert!(matches!(
            request(&input, &mut owner),
            Err(LtxvError::Input(
                "Durations over 10s are only available for the Fast model at 1920x1080 resolution and 25 FPS."
            ))
        ));
        input.arguments.insert("duration".into(), json!(4));
        assert!(matches!(
            request(&input, &mut owner),
            Err(LtxvError::Input(
                "Currently only one input image is supported."
            ))
        ));
        input
            .arguments
            .insert("image".into(), json!({"shape":[1,1,1,3]}));
        let mut denied = Transport {
            deny_cost: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 2_000_000, &mut denied, &mut owner),
            Err(LtxvError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(denied.bodies.iter().all(Vec::is_empty));
        assert!(owner.operations.is_empty());
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 2_000_000, &mut transport, &mut owner),
            Err(LtxvError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(transport.costs.len(), 1);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        assert!(owner.operations.is_empty());
        input.arguments.insert("prompt".into(), json!("\u{1c}"));
        let mut cancelled = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 2_000_000, &mut cancelled, &mut owner),
            Err(LtxvError::Transport(ConformanceError::Cancelled))
        ));
        assert!(cancelled.heads.is_empty());
        assert!(cancelled.costs.is_empty());
        assert!(cancelled.bodies.is_empty());
        Ok(())
    }
}
