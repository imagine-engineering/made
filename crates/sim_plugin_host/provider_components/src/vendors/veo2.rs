use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, Progress, ProviderTransport, RequestHead, ResponseEvent, StreamHandle,
    TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    sync::OnceLock,
};
pub const NAMESPACE: &str = "zed.sim.provider.veo2";
pub const ALIASES: &[&str] = &["veo"];
pub const SOURCE_SHA256: &str = "90c1e81fcc00499d43786cf947c07ae7123e8112b88586e324c2952d39468e85";
pub const MAXIMUM_BODY_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
#[derive(Clone, Copy, Debug, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0722",
        node_identifier: "Veo3FirstLastFrameNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0723",
        node_identifier: "Veo3VideoGenerationNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0724",
        node_identifier: "VeoVideoGenerationNode",
    },
];
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0187",
        method: "POST",
        path: "/proxy/veo/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0188",
        method: "POST",
        path: "/proxy/veo/{model}/generate",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0189",
        method: "POST",
        path: "/proxy/veo/{model}/poll",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum VeoError {
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
impl fmt::Display for VeoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Veo HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for VeoError {}
impl From<ConformanceError> for VeoError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for VeoError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<&'static Value, VeoError> {
    static SOURCE: OnceLock<Result<Value, String>> = OnceLock::new();
    SOURCE
        .get_or_init(|| {
            serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/veo2/source-contract.json"
            )))
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| VeoError::Provider(format!("Invalid pinned Veo source contract: {error}")))
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, VeoError> {
    if !NODES.iter().any(|node| node.node_identifier == input.node) {
        return Err(VeoError::Input("Unknown Veo node"));
    }
    let source = oracle()?;
    let contract = &source["contracts"][&input.node];
    let names = contract["arguments"]
        .as_array()
        .ok_or(VeoError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(VeoError::Input("Unknown Veo source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(VeoError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            input.arguments.insert(
                name.into(),
                contract["defaults"]
                    .get(name)
                    .ok_or(VeoError::Input("Missing Veo source argument"))?
                    .clone(),
            );
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, VeoError> {
    input
        .arguments
        .get(name)
        .ok_or(VeoError::Input("Missing Veo argument"))
}
fn text(value: &Value) -> Result<&str, VeoError> {
    value
        .as_str()
        .ok_or(VeoError::Input("Expected source string"))
}
fn preflight(input: &Input) -> Result<String, VeoError> {
    let label = text(argument(input, "model")?)?;
    if input.node != "VeoVideoGenerationNode"
        && argument(input, "resolution")? == "4k"
        && (label.contains("lite")
            || input.node == "Veo3VideoGenerationNode" && label.contains("3.0"))
    {
        return Err(VeoError::Input(
            "4K resolution is not supported by the selected source model.",
        ));
    }
    oracle()?["models"]
        .get(label)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or(VeoError::Input("Unknown source model"))
}
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_none_or(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}
fn receipt(value: &[u8]) -> Result<(), ConformanceError> {
    if value.is_empty() || value.len() > MAXIMUM_RECEIPT_BYTES {
        Err(ConformanceError::InvalidResponse)
    } else {
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiKind {
    Request,
    GenerateResponse,
    PollResponse,
}
impl ApiKind {
    fn name(self) -> &'static str {
        match self {
            Self::Request => "Request",
            Self::GenerateResponse => "GenerateResponse",
            Self::PollResponse => "PollResponse",
        }
    }
}
#[derive(Clone, Debug)]
pub struct PreparedBytes {
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub base64: String,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImagePolicy {
    pub total_pixels: u64,
    pub mime_type: &'static str,
    pub use_first_batch_image: bool,
}
pub trait VeoPreparation {
    fn png_base64(
        &mut self,
        _image: &Value,
        _policy: &ImagePolicy,
    ) -> Result<PreparedImage, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn validate_api(&mut self, _kind: ApiKind, _value: &Value) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn encode_json(&mut self, _value: &Value) -> Result<PreparedBytes, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn decode_base64(&mut self, _value: &Value) -> Result<PreparedBytes, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn python_string(&mut self, _value: &Value) -> Result<String, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct Request {
    pub model: String,
    pub generate_path: String,
    pub poll_path: String,
    pub body: Value,
    pub image_receipts: Vec<Vec<u8>>,
    pub poll_interval_milliseconds: u64,
}
fn image(
    image: &Value,
    owner: &mut dyn VeoPreparation,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Value, VeoError> {
    let result = owner.png_base64(
        image,
        &ImagePolicy {
            total_pixels: 4_194_304,
            mime_type: "image/png",
            use_first_batch_image: true,
        },
    )?;
    receipt(&result.receipt)?;
    if result.base64.len() > MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    receipts.push(result.receipt);
    Ok(json!({"bytesBase64Encoded":result.base64,"mimeType":"image/png"}))
}
pub fn request(input: &Input, owner: &mut dyn VeoPreparation) -> Result<Request, VeoError> {
    let input = normalized(input)?;
    let model = preflight(&input)?;
    let legacy = input.node == "VeoVideoGenerationNode";
    let first_last = input.node == "Veo3FirstLastFrameNode";
    let mut image_receipts = vec![];
    let mut instance = json!({"prompt":argument(&input,"prompt")?});
    if first_last {
        instance["image"] = image(argument(&input, "first_frame")?, owner, &mut image_receipts)?;
        instance["lastFrame"] = image(argument(&input, "last_frame")?, owner, &mut image_receipts)?;
    } else if let Some(value) = input
        .arguments
        .get("image")
        .filter(|value| !value.is_null())
    {
        let value = image(value, owner, &mut image_receipts)?;
        if truthy(&value["bytesBase64Encoded"]) {
            instance["image"] = value;
        }
    }
    let mut parameters = json!({"aspectRatio":argument(&input,"aspect_ratio")?,"personGeneration":if first_last{json!("ALLOW")}else{argument(&input,"person_generation")?.clone()},"durationSeconds":argument(&input,if first_last{"duration"}else{"duration_seconds"})?,"enhancePrompt":if legacy{argument(&input,"enhance_prompt")?.clone()}else{json!(true)}});
    if first_last {
        parameters["seed"] = argument(&input, "seed")?.clone();
        parameters["negativePrompt"] = argument(&input, "negative_prompt")?.clone();
        parameters["generateAudio"] = argument(&input, "generate_audio")?.clone();
        parameters["resolution"] = argument(&input, "resolution")?.clone();
    } else {
        let negative = argument(&input, "negative_prompt")?;
        if truthy(negative) {
            parameters["negativePrompt"] = negative.clone();
        }
        let seed = argument(&input, "seed")?;
        let seed_number = seed
            .as_f64()
            .or_else(|| seed.as_bool().map(|value| if value { 1.0 } else { 0.0 }))
            .ok_or(VeoError::Input("Source seed comparison failed"))?;
        if seed_number > 0.0 {
            parameters["seed"] = seed.clone();
        }
        if !legacy || !model.contains("veo-2.0") {
            parameters["generateAudio"] = argument(&input, "generate_audio")?.clone();
            parameters["enhancePrompt"] = json!(true);
        }
        if !legacy && model.contains("veo-3.1") {
            parameters["resolution"] = argument(&input, "resolution")?.clone();
        }
    }
    let body = owner.validate_api(
        ApiKind::Request,
        &json!({"instances":[instance],"parameters":parameters}),
    )?;
    Ok(Request {
        generate_path: format!("/proxy/veo/{model}/generate"),
        poll_path: format!("/proxy/veo/{model}/poll"),
        model,
        body,
        image_receipts,
        poll_interval_milliseconds: if legacy { 5000 } else { 9000 },
    })
}
#[derive(Clone, Copy, Debug)]
pub struct CredentialAvailability {
    pub auth_token: bool,
    pub api_key: bool,
}
fn head(
    path: &str,
    credentials: CredentialAvailability,
    download: bool,
) -> Result<RequestHead, VeoError> {
    if download {
        if !path.starts_with("https://")
            || path.len() > 8192
            || path.contains(['\r', '\n', '#', '\\'])
            || path.chars().any(char::is_whitespace)
        {
            return Err(VeoError::Response(
                "Canonical HTTPS download authority required",
            ));
        }
        return Ok(RequestHead {
            method: HttpMethod::Get,
            path: path.into(),
            secret_id: None,
            headers: vec![],
        });
    }
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
        return Err(VeoError::Input("Missing Comfy API credentials"));
    };
    Ok(RequestHead {
        method: HttpMethod::Post,
        path: path.into(),
        secret_id: Some(secret.into()),
        headers: vec![
            Header {
                name: name.into(),
                value: value.into(),
            },
            Header {
                name: "Comfy-Env".into(),
                value: "host".into(),
            },
            Header {
                name: "Comfy-Usage-Source".into(),
                value: "comfyui-api".into(),
            },
            Header {
                name: "Content-Type".into(),
                value: "application/json".into(),
            },
        ],
    })
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Generate,
    Poll,
    Download,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ResponsePolicy {
    pub stage: Stage,
    pub timeout_milliseconds: Option<u64>,
    pub ordinary_retries: u32,
    pub rate_limit_retries: u32,
    pub retry_backoff: f64,
    pub monitor_progress: bool,
    pub wait_label: Option<&'static str>,
    pub final_label: Option<&'static str>,
}
impl ResponsePolicy {
    fn source(stage: Stage) -> Self {
        match stage {
            Stage::Generate => Self {
                stage,
                timeout_milliseconds: Some(3_600_000),
                ordinary_retries: 3,
                rate_limit_retries: 16,
                retry_backoff: 2.0,
                monitor_progress: true,
                wait_label: Some("Waiting for server"),
                final_label: Some("Completed"),
            },
            Stage::Poll => Self {
                stage,
                timeout_milliseconds: Some(120_000),
                ordinary_retries: 10,
                rate_limit_retries: 16,
                retry_backoff: 1.4,
                monitor_progress: false,
                wait_label: Some("Checking"),
                final_label: None,
            },
            Stage::Download => Self {
                stage,
                timeout_milliseconds: None,
                ordinary_retries: 5,
                rate_limit_retries: 0,
                retry_backoff: 2.0,
                monitor_progress: false,
                wait_label: None,
                final_label: None,
            },
        }
    }
}
pub trait VeoTransport: ProviderTransport {
    fn begin_poll(
        &mut self,
        _interval_milliseconds: u64,
        _estimated_duration_seconds: u64,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
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
    fn poll_delay(&mut self, delay_milliseconds: u64) -> Result<(), ConformanceError>;
    fn poll_status(
        &mut self,
        _status: &str,
        _estimated_duration_seconds: u64,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn status_extraction_failed(
        &mut self,
        _error: &ConformanceError,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn exchange_on_handle(
    body: &[u8],
    handle: StreamHandle,
    policy: &ResponsePolicy,
    transport: &mut dyn VeoTransport,
) -> Result<(Vec<u8>, Vec<u8>), VeoError> {
    transport.configure_response(handle, policy)?;
    if body.len() > MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    if body.is_empty() {
        transport.write_request_chunk(handle, 0, &[], true)?;
    } else {
        for (sequence, chunk) in body.chunks(MAXIMUM_CHUNK_BYTES).enumerate() {
            transport.check_cancelled(Some(handle))?;
            transport.write_request_chunk(
                handle,
                sequence as u64,
                chunk,
                (sequence + 1) * MAXIMUM_CHUNK_BYTES >= body.len(),
            )?;
        }
    }
    let mut status = None;
    let mut retry_after = None;
    let mut bytes = vec![];
    let mut sequence = None;
    loop {
        transport.check_cancelled(Some(handle))?;
        match transport.wait_response(
            handle,
            sequence,
            policy.timeout_milliseconds.unwrap_or(300_000),
        )? {
            ResponseEvent::Head {
                status: code,
                headers,
            } => {
                if status.is_some()
                    || !(100..=599).contains(&code)
                    || headers.len() > 16
                    || headers
                        .iter()
                        .map(|header| header.name.len() + header.value.len())
                        .sum::<usize>()
                        > 16384
                    || headers.iter().any(|header| {
                        header.name.contains(['\r', '\n']) || header.value.contains(['\r', '\n'])
                    })
                {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                status = Some(code);
                retry_after = headers
                    .iter()
                    .find(|header| header.name.eq_ignore_ascii_case("retry-after"))
                    .map(|header| header.value.clone());
            }
            ResponseEvent::Chunk(chunk) => {
                if status.is_none() {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                if chunk.len() > MAXIMUM_CHUNK_BYTES {
                    return Err(ConformanceError::ChunkLimit.into());
                }
                if bytes
                    .len()
                    .checked_add(chunk.len())
                    .is_none_or(|size| size > MAXIMUM_BODY_BYTES)
                {
                    return Err(ConformanceError::BodyLimit.into());
                }
                bytes.extend_from_slice(&chunk);
            }
            ResponseEvent::Completed(value) => {
                let status = status.ok_or(ConformanceError::InvalidResponse)?;
                receipt(&value)?;
                if status >= 400 {
                    return Err(VeoError::Http {
                        status,
                        retry_after,
                    });
                }
                return Ok((bytes, value));
            }
            ResponseEvent::Cancelled => return Err(ConformanceError::Cancelled.into()),
            ResponseEvent::TimedOut => return Err(ConformanceError::HostFailure.into()),
        }
        sequence = Some(sequence.map_or(0u64, |sequence| sequence + 1));
    }
}
fn exchange_existing(
    body: &[u8],
    mut handle: StreamHandle,
    policy: &ResponsePolicy,
    transport: &mut dyn VeoTransport,
) -> Result<(Vec<u8>, Vec<u8>), VeoError> {
    let mut ordinary = 0u32;
    let mut rate = 0u32;
    loop {
        match exchange_on_handle(body, handle, policy, transport) {
            Err(VeoError::Http {
                status,
                retry_after,
            }) => {
                let fallback = if status == 429
                    && policy.stage != Stage::Download
                    && rate < policy.rate_limit_retries
                {
                    let delay = (1000.0 * policy.retry_backoff.powi(rate as i32))
                        .round()
                        .min(30_000.0) as u64;
                    rate += 1;
                    delay
                } else if (matches!(status, 408 | 500 | 502 | 503 | 504)
                    || status == 429 && policy.stage == Stage::Download)
                    && ordinary < policy.ordinary_retries
                {
                    let delay =
                        (1000.0 * policy.retry_backoff.powi(ordinary as i32)).round() as u64;
                    ordinary += 1;
                    delay
                } else {
                    return Err(VeoError::Http {
                        status,
                        retry_after,
                    });
                };
                let delay = if policy.stage != Stage::Download {
                    if let Some(header) = retry_after {
                        transport.retry_after_delay(&header, fallback)?.min(150_000)
                    } else {
                        fallback
                    }
                } else {
                    fallback
                };
                transport.check_cancelled(Some(handle))?;
                let fresh = transport.retry_request(handle, delay)?;
                if fresh == handle
                    || fresh.invocation != handle.invocation
                    || fresh.slot == 0
                    || fresh.generation == 0
                {
                    return Err(ConformanceError::InvalidHandle.into());
                }
                handle = fresh;
            }
            result => return result,
        }
    }
}
fn exchange(
    head: &RequestHead,
    body: &[u8],
    stage: Stage,
    transport: &mut dyn VeoTransport,
) -> Result<(Vec<u8>, Vec<u8>), VeoError> {
    transport.check_cancelled(None)?;
    let handle = transport.start_request(head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    exchange_existing(body, handle, &ResponsePolicy::source(stage), transport)
}
fn prepared_json(value: &Value, owner: &mut dyn VeoPreparation) -> Result<PreparedBytes, VeoError> {
    let result = owner.encode_json(value)?;
    receipt(&result.receipt)?;
    if result.bytes.len() > MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(result)
}
fn response_video(
    node: &str,
    response: &Value,
    owner: &mut dyn VeoPreparation,
) -> Result<VideoLocation, VeoError> {
    if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
        let message = owner.python_string(error.get("message").unwrap_or(&Value::Null))?;
        let code = owner.python_string(error.get("code").unwrap_or(&Value::Null))?;
        return Err(VeoError::Provider(format!(
            "Veo API error: {message} (code: {code})"
        )));
    }
    let legacy = node == "VeoVideoGenerationNode";
    let body = response.get("response").filter(|value| !value.is_null());
    if let Some(body) = body {
        let count = body.get("raiMediaFilteredCount").unwrap_or(&Value::Null);
        if legacy && count.is_null() {
            return Err(VeoError::Response(
                "Source nullable filtered-count comparison failed",
            ));
        }
        let filtered = if legacy {
            count.as_i64().is_some_and(|count| count > 0)
        } else {
            truthy(count)
        };
        if filtered {
            let count_text = owner.python_string(count)?;
            let reasons = body
                .get("raiMediaFilteredReasons")
                .and_then(Value::as_array)
                .and_then(|reasons| reasons.first());
            let message = if legacy {
                if let Some(reason) = reasons {
                    format!(
                        "Content filtered by Google's Responsible AI practices: {} ({count_text} videos filtered.)",
                        owner.python_string(reason)?
                    )
                } else {
                    format!(
                        "Content filtered by Google's Responsible AI practices ({count_text} videos filtered.)"
                    )
                }
            } else {
                let reason = if let Some(reason) = reasons {
                    format!(": {}", owner.python_string(reason)?)
                } else {
                    String::new()
                };
                format!(
                    "Content blocked by Google's Responsible AI filters{reason} ({count_text} video{} filtered).",
                    if count == &json!(1) { "" } else { "s" }
                )
            };
            return Err(VeoError::Provider(message));
        }
        if let Some(video) = body
            .get("videos")
            .and_then(Value::as_array)
            .and_then(|videos| videos.first())
        {
            if let Some(value) = video
                .get("bytesBase64Encoded")
                .filter(|value| truthy(value))
            {
                return Ok(VideoLocation::Inline(value.clone()));
            }
            if let Some(value) = video.get("gcsUri").filter(|value| truthy(value)) {
                return Ok(VideoLocation::Download(text(value)?.into()));
            }
            return Err(VeoError::Provider(
                "Video returned but no data or URL was provided".into(),
            ));
        }
    } else if !legacy {
        return Err(VeoError::Response("Missing source Veo3 response object"));
    }
    Err(VeoError::Provider(
        "Video generation completed but no video was returned".into(),
    ))
}
#[derive(Clone, Debug)]
pub enum VideoLocation {
    Inline(Value),
    Download(String),
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub video_bytes: Vec<u8>,
    pub transport_receipts: Vec<Vec<u8>>,
    pub preparation_receipts: Vec<Vec<u8>>,
    pub approval_receipt: Vec<u8>,
    pub approved_maximum_microunits: u64,
    pub poll_count: u32,
    pub downloaded: bool,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn VeoTransport,
    owner: &mut dyn VeoPreparation,
) -> Result<ResultValue, VeoError> {
    let input = normalized(input)?;
    let model = preflight(&input)?;
    transport.check_cancelled(None)?;
    let generate = head(&format!("/proxy/veo/{model}/generate"), credentials, false)?;
    let handle = transport.start_request(&generate)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "veo2.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits,
        },
    )?;
    if !approval.accepted || approval.approved_microunits < maximum_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    receipt(&approval.receipt)?;
    transport.check_cancelled(Some(handle))?;
    let plan = request(&input, owner)?;
    let generation_body = prepared_json(&plan.body, owner)?;
    let (bytes, generation_receipt) = exchange_existing(
        &generation_body.bytes,
        handle,
        &ResponsePolicy::source(Stage::Generate),
        transport,
    )?;
    let generation =
        owner.validate_api(ApiKind::GenerateResponse, &serde_json::from_slice(&bytes)?)?;
    let name = generation
        .get("name")
        .ok_or(VeoError::Response("Missing source operation name"))?;
    let poll_body = prepared_json(&json!({"operationName":name}), owner)?;
    let poll_head = head(&plan.poll_path, credentials, false)?;
    let mut transport_receipts = vec![generation_receipt];
    let mut preparation_receipts = vec![generation_body.receipt, poll_body.receipt];
    let mut poll_count = 0;
    transport.begin_poll(plan.poll_interval_milliseconds, 32)?;
    let final_response = loop {
        if poll_count >= 480 {
            return Err(ConformanceError::PollLimit.into());
        }
        let (bytes, poll_receipt) = exchange(&poll_head, &poll_body.bytes, Stage::Poll, transport)?;
        transport_receipts.push(poll_receipt);
        poll_count += 1;
        let raw: Value = serde_json::from_slice(&bytes)?;
        let validated = match owner.validate_api(ApiKind::PollResponse, &raw) {
            Ok(value) => Some(value),
            Err(error @ ConformanceError::InvalidRequest) => {
                transport.status_extraction_failed(&error)?;
                None
            }
            Err(error) => return Err(error.into()),
        };
        let completed = validated
            .as_ref()
            .is_some_and(|value| value["done"] == true);
        transport.poll_status(
            if completed {
                "completed"
            } else if validated.is_some() {
                "pending"
            } else {
                "Processing"
            },
            32,
        )?;
        if completed {
            break validated.ok_or(VeoError::Response("Missing validated completed response"))?;
        }
        transport.check_cancelled(None)?;
        transport.poll_delay(plan.poll_interval_milliseconds)?;
    };
    let (video_bytes, downloaded) = match response_video(&input.node, &final_response, owner)? {
        VideoLocation::Inline(value) => {
            let prepared = owner.decode_base64(&value)?;
            receipt(&prepared.receipt)?;
            preparation_receipts.push(prepared.receipt);
            (prepared.bytes, false)
        }
        VideoLocation::Download(url) => {
            let (bytes, value) = exchange(
                &head(&url, credentials, true)?,
                &[],
                Stage::Download,
                transport,
            )?;
            transport_receipts.push(value);
            (bytes, true)
        }
    };
    if video_bytes.len() > MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(ResultValue {
        plan,
        video_bytes,
        transport_receipts,
        preparation_receipts,
        approval_receipt: approval.receipt,
        approved_maximum_microunits: maximum_microunits,
        poll_count,
        downloaded,
    })
}
pub trait VeoVideoMaterializer {
    fn materialize_video(
        &mut self,
        result: &ResultValue,
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub struct MaterializationOwners<'a> {
    pub preparation: &'a mut dyn VeoPreparation,
    pub video: &'a mut dyn VeoVideoMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn VeoTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), VeoError> {
    let result = invoke(
        input,
        credentials,
        maximum_microunits,
        transport,
        owners.preparation,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.video.materialize_video(&result)?;
    let registry = TypeRegistry::built_in().map_err(|_| ConformanceError::HostFailure)?;
    let type_id = registry
        .resolve("VIDEO")
        .map_err(|_| ConformanceError::HostFailure)?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| ConformanceError::HostFailure)?;
    let batch = validate_outputs(
        &[ExpectedOutput {
            port_id: "output_0".into(),
            type_id,
            family,
        }],
        outputs,
        MAXIMUM_BODY_BYTES,
    )
    .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, owners.sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn decode_hex(value: &Value) -> Result<Vec<u8>, ConformanceError> {
    let text = value.as_str().ok_or(ConformanceError::InvalidRequest)?;
    if !text.len().is_multiple_of(2) || text.len() > 2 * MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::InvalidRequest);
    }
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            std::str::from_utf8(pair)
                .map_err(|_| ConformanceError::InvalidRequest)
                .and_then(|pair| {
                    u8::from_str_radix(pair, 16).map_err(|_| ConformanceError::InvalidRequest)
                })
        })
        .collect()
}
struct SourceProbePreparation {
    source: &'static Value,
    case: Value,
    media: Vec<Value>,
    unbound: bool,
    bad_receipt: bool,
}
impl SourceProbePreparation {
    fn new(case: &Value) -> Result<Self, VeoError> {
        Ok(Self {
            source: oracle()?,
            case: case.clone(),
            media: vec![],
            unbound: false,
            bad_receipt: false,
        })
    }
    fn proof_receipt(&self) -> Vec<u8> {
        if self.bad_receipt {
            vec![]
        } else {
            b"injected-actual-source-preparation-receipt".to_vec()
        }
    }
}
impl VeoPreparation for SourceProbePreparation {
    fn png_base64(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<PreparedImage, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        if *policy
            != (ImagePolicy {
                total_pixels: 4_194_304,
                mime_type: "image/png",
                use_first_batch_image: true,
            })
        {
            return Err(ConformanceError::InvalidRequest);
        }
        let expected = self.case["media"]
            .as_array()
            .and_then(|values| values.get(self.media.len()))
            .filter(|record| record["image"] == *image)
            .ok_or(ConformanceError::InvalidRequest)?;
        self.media.push(expected.clone());
        let base64 = expected["base64"]
            .as_str()
            .ok_or(ConformanceError::InvalidRequest)?
            .into();
        Ok(PreparedImage {
            base64,
            receipt: self.proof_receipt(),
        })
    }
    fn validate_api(&mut self, kind: ApiKind, value: &Value) -> Result<Value, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        self.source["api_probes"]
            .as_array()
            .and_then(|probes| {
                probes
                    .iter()
                    .find(|probe| probe["kind"] == kind.name() && probe["input"] == *value)
            })
            .and_then(|probe| probe.get("output"))
            .cloned()
            .ok_or(ConformanceError::InvalidRequest)
    }
    fn encode_json(&mut self, value: &Value) -> Result<PreparedBytes, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        let request = self.case["requests"]
            .as_array()
            .and_then(|requests| requests.iter().find(|request| request["body"] == *value))
            .ok_or(ConformanceError::InvalidRequest)?;
        Ok(PreparedBytes {
            bytes: decode_hex(&request["json_body_hex"])?,
            receipt: self.proof_receipt(),
        })
    }
    fn decode_base64(&mut self, value: &Value) -> Result<PreparedBytes, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        let probe = self.source["base64_probes"]
            .as_array()
            .and_then(|probes| probes.iter().find(|probe| probe["value"] == *value))
            .ok_or(ConformanceError::InvalidRequest)?;
        Ok(PreparedBytes {
            bytes: decode_hex(&probe["bytes_hex"])?,
            receipt: self.proof_receipt(),
        })
    }
    fn python_string(&mut self, value: &Value) -> Result<String, ConformanceError> {
        if self.unbound {
            return Err(ConformanceError::HostFailure);
        }
        self.source["scalar_probes"]
            .as_array()
            .and_then(|probes| probes.iter().find(|probe| probe["value"] == *value))
            .and_then(|probe| probe["output"].as_str())
            .map(str::to_owned)
            .ok_or(ConformanceError::InvalidRequest)
    }
}
struct SourceProbeTransport {
    case: Value,
    heads: Vec<RequestHead>,
    bodies: Vec<Vec<u8>>,
    events: VecDeque<Result<ResponseEvent, ConformanceError>>,
    costs: Vec<CostRequest>,
    policies: Vec<ResponsePolicy>,
    delays: Vec<u64>,
    retry_delays: Vec<u64>,
    statuses: Vec<String>,
    diagnostics: usize,
    cancelled: bool,
    deny_cost: bool,
    bad_receipt: bool,
    same_handle: bool,
}
impl SourceProbeTransport {
    fn new(case: &Value) -> Self {
        Self {
            case: case.clone(),
            heads: vec![],
            bodies: vec![],
            events: VecDeque::new(),
            costs: vec![],
            policies: vec![],
            delays: vec![],
            retry_delays: vec![],
            statuses: vec![],
            diagnostics: 0,
            cancelled: false,
            deny_cost: false,
            bad_receipt: false,
            same_handle: false,
        }
    }
}
fn handle(slot: u32) -> StreamHandle {
    StreamHandle {
        invocation: 42,
        slot,
        generation: 1,
    }
}
impl ProviderTransport for SourceProbeTransport {
    fn check_cancelled(&mut self, _handle: Option<StreamHandle>) -> Result<(), ConformanceError> {
        if self.cancelled {
            Err(ConformanceError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
        let index = self.heads.len();
        self.heads.push(head.clone());
        self.bodies.push(vec![]);
        if self.events.is_empty() {
            let response = if head.method == HttpMethod::Get {
                None
            } else if head.path.ends_with("/generate") {
                Some(self.case["override"].get("generate").cloned().unwrap_or(oracle().map_err(|_|ConformanceError::InvalidRequest)?["generation_response"].clone()))
            } else {
                self.case["polls"]
                    .as_array()
                    .and_then(|polls| polls.get(index.checked_sub(1)?))
                    .cloned()
            };
            let bytes = if let Some(response) = response {
                serde_json::to_vec(&response).map_err(|_| ConformanceError::InvalidRequest)?
            } else if head.method == HttpMethod::Get {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/veo2/source-response.mp4"
                ))
                .to_vec()
            } else {
                return Err(ConformanceError::InvalidRequest);
            };
            self.events.push_back(Ok(ResponseEvent::Head {
                status: 200,
                headers: vec![],
            }));
            for chunk in bytes.chunks(MAXIMUM_CHUNK_BYTES) {
                self.events
                    .push_back(Ok(ResponseEvent::Chunk(chunk.to_vec())));
            }
            self.events
                .push_back(Ok(ResponseEvent::Completed(if self.bad_receipt {
                    vec![]
                } else {
                    b"injected-source-response-receipt".to_vec()
                })));
        }
        Ok(handle((index + 1) as u32))
    }
    fn write_request_chunk(
        &mut self,
        handle: StreamHandle,
        sequence: u64,
        bytes: &[u8],
        _end: bool,
    ) -> Result<(), ConformanceError> {
        let index = handle
            .slot
            .checked_sub(1)
            .ok_or(ConformanceError::InvalidHandle)? as usize;
        let body = self
            .bodies
            .get_mut(index)
            .ok_or(ConformanceError::InvalidHandle)?;
        if sequence != body.len().div_ceil(MAXIMUM_CHUNK_BYTES) as u64 {
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
        _request: &crate::common::StorageCreateRequest,
    ) -> Result<crate::common::StorageCreateResponse, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn put_signed(
        &mut self,
        _handle: StreamHandle,
        _request: &crate::common::SignedPutRequest,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn request_cost(
        &mut self,
        _handle: StreamHandle,
        cost: &CostRequest,
    ) -> Result<crate::common::CostResponse, ConformanceError> {
        self.costs.push(cost.clone());
        Ok(crate::common::CostResponse {
            accepted: !self.deny_cost,
            approved_microunits: cost.maximum_microunits,
            receipt: b"source-cost-receipt".to_vec(),
        })
    }
    fn report_progress(
        &mut self,
        _handle: StreamHandle,
        _progress: &Progress,
    ) -> Result<(), ConformanceError> {
        Ok(())
    }
    fn wait_response(
        &mut self,
        _handle: StreamHandle,
        _after_sequence: Option<u64>,
        _timeout_milliseconds: u64,
    ) -> Result<ResponseEvent, ConformanceError> {
        self.events
            .pop_front()
            .ok_or(ConformanceError::InvalidResponse)?
    }
}
impl VeoTransport for SourceProbeTransport {
    fn begin_poll(&mut self, interval: u64, estimated: u64) -> Result<(), ConformanceError> {
        if ![5000, 9000].contains(&interval) || estimated != 32 {
            return Err(ConformanceError::InvalidProgress);
        }
        Ok(())
    }
    fn configure_response(
        &mut self,
        _handle: StreamHandle,
        policy: &ResponsePolicy,
    ) -> Result<(), ConformanceError> {
        self.policies.push(policy.clone());
        Ok(())
    }
    fn retry_after_delay(&mut self, header: &str, _fallback: u64) -> Result<u64, ConformanceError> {
        header
            .parse::<u64>()
            .map(|seconds| seconds.saturating_mul(1000))
            .map_err(|_| ConformanceError::HostFailure)
    }
    fn retry_request(
        &mut self,
        old: StreamHandle,
        delay: u64,
    ) -> Result<StreamHandle, ConformanceError> {
        self.retry_delays.push(delay);
        if self.same_handle {
            return Ok(old);
        }
        let previous = self
            .heads
            .last()
            .cloned()
            .ok_or(ConformanceError::InvalidRequest)?;
        self.heads.push(previous);
        self.bodies.push(vec![]);
        Ok(handle(self.heads.len() as u32))
    }
    fn poll_delay(&mut self, delay: u64) -> Result<(), ConformanceError> {
        self.delays.push(delay);
        Ok(())
    }
    fn poll_status(&mut self, status: &str, estimated: u64) -> Result<(), ConformanceError> {
        if estimated != 32 {
            return Err(ConformanceError::InvalidProgress);
        }
        self.statuses.push(status.into());
        Ok(())
    }
    fn status_extraction_failed(
        &mut self,
        _error: &ConformanceError,
    ) -> Result<(), ConformanceError> {
        self.diagnostics += 1;
        Ok(())
    }
}
fn source_input(case: &Value) -> Result<Input, VeoError> {
    Ok(serde_json::from_value(case["input"].clone())?)
}
fn credentials() -> CredentialAvailability {
    CredentialAvailability {
        auth_token: true,
        api_key: true,
    }
}
fn compare_case(case: &Value, failure: bool) -> Result<(), Box<dyn Error>> {
    let mut transport = SourceProbeTransport::new(case);
    let mut preparation = SourceProbePreparation::new(case)?;
    let result = invoke(
        &source_input(case)?,
        credentials(),
        1,
        &mut transport,
        &mut preparation,
    );
    if failure {
        match result {
            Ok(_) => return Err("Expected source Veo failure".into()),
            Err(VeoError::Provider(message)) => {
                if case["message"].as_str() != Some(&message) {
                    return Err(format!(
                        "Source Veo provider message differs: {message} vs {}",
                        case["message"]
                    )
                    .into());
                }
            }
            Err(_) => {}
        }
    } else {
        let result = result?;
        if result.video_bytes
            != include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/veo2/source-response.mp4"
            ))
        {
            return Err("Source encoded video bytes differ".into());
        }
        if result.poll_count as usize != case["polls"].as_array().ok_or("source polls")?.len() {
            return Err("Source poll count differs".into());
        }
    }
    if json!(preparation.media) != case["media"] {
        return Err("Source PNG preparation ordering differs".into());
    }
    let requests = case["requests"].as_array().ok_or("requests")?;
    let actual: Vec<_> = transport
        .heads
        .iter()
        .zip(&transport.bodies)
        .filter(|(_, body)| !body.is_empty())
        .collect();
    if actual.len() != requests.len() {
        return Err(format!(
            "Source request count differs: {} vs {} for {}",
            actual.len(),
            requests.len(),
            case["input"]
        )
        .into());
    }
    for ((head, body), expected) in actual.iter().zip(requests) {
        if head.path != expected["path"]
            || head.method != HttpMethod::Post
            || json!(
                head.headers
                    .iter()
                    .map(|header| &header.name)
                    .collect::<Vec<_>>()
            ) != expected["header_order"]
            || **body != decode_hex(&expected["json_body_hex"])?
        {
            return Err("Source path/ordered JSON wire differs".into());
        }
    }
    if json!(
        transport
            .delays
            .iter()
            .map(|value| *value as f64 / 1000.0)
            .collect::<Vec<_>>()
    ) != case["delays"]
    {
        return Err("Source active poll delays differ".into());
    }
    let source_downloads = case["downloads"].as_array().ok_or("downloads")?;
    let downloads: Vec<_> = transport
        .heads
        .iter()
        .filter(|head| head.method == HttpMethod::Get)
        .collect();
    let admitted = source_downloads
        .iter()
        .filter(|download| {
            download["url"]
                .as_str()
                .is_some_and(|url| url.starts_with("https://"))
        })
        .collect::<Vec<_>>();
    if downloads.len() != admitted.len()
        || downloads.iter().zip(admitted).any(|(head, expected)| {
            head.path != expected["url"] || head.secret_id.is_some() || !head.headers.is_empty()
        })
    {
        return Err("Source download selection or authority differs".into());
    }
    if transport
        .costs
        .iter()
        .any(|cost| cost.operation != "veo2.generation.maximum" || cost.maximum_microunits != 1)
    {
        return Err("Source cost approval identity differs".into());
    }
    Ok(())
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let checked = || -> Result<(), Box<dyn Error>> {
        let claims: Value = serde_json::from_str(fixture)?;
        if claims["namespace"] != NAMESPACE
            || claims["source_sha256"] != SOURCE_SHA256
            || claims["nodes"] != serde_json::to_value(NODES)?
            || claims["routes"] != serde_json::to_value(ROUTES)?
            || claims["aliases"] != json!(ALIASES)
        {
            return Err("Veo source claim drift".into());
        }
        let source = oracle()?;
        if source["source_sha256"]["comfy_api_nodes/nodes_veo2.py"] != SOURCE_SHA256 {
            return Err("Veo source fingerprint drift".into());
        }
        let mut covered = BTreeSet::new();
        for case in source["cases"].as_array().ok_or("cases")? {
            compare_case(case, false)?;
            covered.insert(source_input(case)?.node);
        }
        for key in ["preflight_errors", "response_errors"] {
            for case in source[key].as_array().ok_or("failures")? {
                compare_case(case, true)?;
            }
        }
        if covered
            != NODES
                .iter()
                .map(|node| node.node_identifier.to_owned())
                .collect()
        {
            return Err("Veo source coverage differs".into());
        }
        Ok(())
    };
    checked().map_err(|_| ConformanceError::InvalidRequest)
}
#[cfg(test)]
mod tests {
    use super::*;
    type TestResult = Result<(), Box<dyn Error>>;
    fn first_case() -> Result<Value, VeoError> {
        oracle()?["cases"]
            .as_array()
            .and_then(|cases| cases.first())
            .cloned()
            .ok_or(VeoError::Response("Missing source case"))
    }
    #[test]
    fn veo2_exact_claims_and_all_actual_source_branches() -> TestResult {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/veo2/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn veo2_defaults_alias_tombstone_and_seed_domains() -> TestResult {
        let input = normalized(&Input {
            node: "Veo3VideoGenerationNode".into(),
            arguments: BTreeMap::from([("prompt".into(), json!("music"))]),
        })?;
        assert_eq!(input.arguments["model"], json!("veo-3.0-generate-001"));
        assert_eq!(input.arguments["duration_seconds"], json!(8));
        assert_eq!(ALIASES, &["veo"]);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            2
        );
        assert_eq!(
            ROUTES.first().map(|route| route.disposition),
            Some("synthetic_prefix_tombstone")
        );
        let mut invalid = input;
        invalid.arguments.insert("unknown".into(), Value::Null);
        assert!(normalized(&invalid).is_err());
        Ok(())
    }
    #[test]
    fn veo2_real_png_two_frame_order_and_source_validation_order() -> TestResult {
        for key in ["cases", "preflight_errors"] {
            for case in oracle()?[key].as_array().ok_or("cases")? {
                let mut owner = SourceProbePreparation::new(case)?;
                let result = request(&source_input(case)?, &mut owner);
                if key == "cases" {
                    let result = result?;
                    assert_eq!(result.body, case["requests"][0]["body"]);
                } else {
                    assert!(result.is_err());
                }
                assert_eq!(json!(owner.media), case["media"]);
            }
        }
        Ok(())
    }
    #[test]
    fn veo2_legacy_nullable_count_and_modern_negative_count_follow_source() -> TestResult {
        for case in oracle()?["response_errors"].as_array().ok_or("errors")? {
            compare_case(case, true)?;
        }
        for case in oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| {
                case["override"]["polls"][0]["response"]["raiMediaFilteredCount"] == json!(-1)
            })
        {
            compare_case(case, false)?;
        }
        Ok(())
    }
    #[test]
    fn veo2_poll_interval_terminal_errors_and_480_attempts_are_source_exact() -> TestResult {
        for case in oracle()?["response_errors"]
            .as_array()
            .ok_or("errors")?
            .iter()
            .filter(|case| {
                case["polls"]
                    .as_array()
                    .is_some_and(|polls| polls.len() == 480)
            })
        {
            compare_case(case, true)?;
            let mut transport = SourceProbeTransport::new(case);
            let result = invoke(
                &source_input(case)?,
                credentials(),
                1,
                &mut transport,
                &mut SourceProbePreparation::new(case)?,
            );
            assert!(matches!(
                result,
                Err(VeoError::Transport(ConformanceError::PollLimit))
            ));
            assert_eq!(transport.delays.len(), 480);
            assert_eq!(transport.statuses.len(), 480);
            if case["override"]["polls"][0]["done"] == "invalid" {
                assert_eq!(transport.diagnostics, 480);
            }
        }
        Ok(())
    }
    #[test]
    fn veo2_auth_cost_and_canonical_owner_failure_send_no_generation_body() -> TestResult {
        let case = first_case()?;
        for missing_auth in [true, false] {
            let mut transport = SourceProbeTransport::new(&case);
            transport.deny_cost = true;
            let auth = if missing_auth {
                CredentialAvailability {
                    auth_token: false,
                    api_key: false,
                }
            } else {
                credentials()
            };
            let mut owner = SourceProbePreparation::new(&case)?;
            assert!(invoke(&source_input(&case)?, auth, 1, &mut transport, &mut owner).is_err());
            assert!(transport.bodies.iter().all(Vec::is_empty));
            assert!(owner.media.is_empty());
            if !missing_auth {
                assert_eq!(
                    transport.costs.first().map(|cost| cost.operation.as_str()),
                    Some("veo2.generation.maximum")
                );
            }
        }
        let mut owner = SourceProbePreparation::new(&case)?;
        owner.unbound = true;
        let mut transport = SourceProbeTransport::new(&case);
        assert!(
            invoke(
                &source_input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut owner
            )
            .is_err()
        );
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn veo2_source_inline_precedence_permissive_base64_and_external_no_secret() -> TestResult {
        for case in oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .filter(|case| {
                !case["override"]
                    .as_object()
                    .is_none_or(|value| value.is_empty())
            })
        {
            compare_case(case, false)?;
        }
        Ok(())
    }
    #[test]
    fn veo2_generation_poll_and_download_retry_budgets_are_independent() -> TestResult {
        let case = first_case()?;
        for stage in [Stage::Generate, Stage::Poll, Stage::Download] {
            let policy = ResponsePolicy::source(stage);
            let mut transport = SourceProbeTransport::new(&case);
            for _ in 0..=policy.ordinary_retries {
                transport.events.push_back(Ok(ResponseEvent::Head {
                    status: 503,
                    headers: vec![],
                }));
                transport.events.push_back(Ok(ResponseEvent::Completed(
                    b"failed-response-receipt".to_vec(),
                )));
            }
            let request = head(
                if stage == Stage::Download {
                    "https://fixture.invalid/video.mp4"
                } else {
                    "/proxy/veo/veo-2.0-generate-001/generate"
                },
                credentials(),
                stage == Stage::Download,
            )?;
            assert!(matches!(
                exchange(&request, b"{}", stage, &mut transport),
                Err(VeoError::Http { status: 503, .. })
            ));
            assert_eq!(
                transport.retry_delays.len(),
                policy.ordinary_retries as usize
            );
            assert!(transport.bodies.iter().all(|body| body == b"{}"));
            for (index, delay) in transport.retry_delays.iter().enumerate() {
                assert_eq!(
                    *delay,
                    (1000.0 * policy.retry_backoff.powi(index as i32)).round() as u64
                );
            }
        }
        Ok(())
    }
    #[test]
    fn veo2_rate_limits_retry_after_and_stale_handles_stay_supervised() -> TestResult {
        let case = first_case()?;
        let mut transport = SourceProbeTransport::new(&case);
        for _ in 0..17 {
            transport.events.push_back(Ok(ResponseEvent::Head {
                status: 429,
                headers: vec![Header {
                    name: "Retry-After".into(),
                    value: "200".into(),
                }],
            }));
            transport
                .events
                .push_back(Ok(ResponseEvent::Completed(b"receipt".to_vec())));
        }
        let request = head(
            "/proxy/veo/veo-2.0-generate-001/generate",
            credentials(),
            false,
        )?;
        assert!(exchange(&request, b"{}", Stage::Generate, &mut transport).is_err());
        assert_eq!(transport.retry_delays, vec![150_000; 16]);
        let mut transport = SourceProbeTransport::new(&case);
        transport.same_handle = true;
        transport.events = VecDeque::from([
            Ok(ResponseEvent::Head {
                status: 503,
                headers: vec![],
            }),
            Ok(ResponseEvent::Completed(b"receipt".to_vec())),
        ]);
        assert!(matches!(
            exchange(&request, b"{}", Stage::Generate, &mut transport),
            Err(VeoError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn veo2_api_key_header_priority_matches_actual_source() -> TestResult {
        let request = head(
            "/proxy/veo/veo-2.0-generate-001/generate",
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(request.secret_id.as_deref(), Some("sim_org.api_key"));
        let names: Vec<_> = request
            .headers
            .iter()
            .map(|header| header.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "X-API-KEY",
                "Comfy-Env",
                "Comfy-Usage-Source",
                "Content-Type"
            ]
        );
        assert_eq!(
            oracle()?["auth_header_probes"][1]["headers"]["X-API-KEY"],
            json!("capability-api-key")
        );
        Ok(())
    }
    #[test]
    fn veo2_no_approximate_badge_policy_overrides_caller_ceiling() -> TestResult {
        let case = first_case()?;
        let mut transport = SourceProbeTransport::new(&case);
        let result = invoke(
            &source_input(&case)?,
            credentials(),
            1,
            &mut transport,
            &mut SourceProbePreparation::new(&case)?,
        )?;
        assert_eq!(result.approved_maximum_microunits, 1);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(
            transport.costs.first().map(|cost| cost.maximum_microunits),
            Some(1)
        );
        Ok(())
    }
    #[test]
    fn veo2_worker_loss_cancel_bad_frames_and_current_fresh_injection() -> TestResult {
        let case = first_case()?;
        for event in [
            Err(ConformanceError::WorkerLost),
            Ok(ResponseEvent::Cancelled),
            Ok(ResponseEvent::Chunk(vec![1])),
            Ok(ResponseEvent::TimedOut),
        ] {
            let mut transport = SourceProbeTransport::new(&case);
            transport.events.push_back(event);
            assert!(
                invoke(
                    &source_input(&case)?,
                    credentials(),
                    1,
                    &mut transport,
                    &mut SourceProbePreparation::new(&case)?
                )
                .is_err()
            );
            assert!(transport.retry_delays.is_empty());
        }
        compare_case(&case, false)?;
        Ok(())
    }
    #[test]
    fn veo2_oversized_chunk_receipt_and_aggregate_fail_before_media() -> TestResult {
        let case = first_case()?;
        let request = head(
            "/proxy/veo/veo-2.0-generate-001/generate",
            credentials(),
            false,
        )?;
        let mut transport = SourceProbeTransport::new(&case);
        transport.events = VecDeque::from([
            Ok(ResponseEvent::Head {
                status: 200,
                headers: vec![],
            }),
            Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES + 1])),
        ]);
        assert!(matches!(
            exchange(&request, b"{}", Stage::Generate, &mut transport),
            Err(VeoError::Transport(ConformanceError::ChunkLimit))
        ));
        let mut transport = SourceProbeTransport::new(&case);
        transport.events = VecDeque::from([
            Ok(ResponseEvent::Head {
                status: 200,
                headers: vec![],
            }),
            Ok(ResponseEvent::Completed(vec![0; MAXIMUM_RECEIPT_BYTES + 1])),
        ]);
        assert!(exchange(&request, b"{}", Stage::Generate, &mut transport).is_err());
        let mut transport = SourceProbeTransport::new(&case);
        transport.events.push_back(Ok(ResponseEvent::Head {
            status: 200,
            headers: vec![],
        }));
        for _ in 0..MAXIMUM_BODY_BYTES / MAXIMUM_CHUNK_BYTES {
            transport
                .events
                .push_back(Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES])));
        }
        transport
            .events
            .push_back(Ok(ResponseEvent::Chunk(vec![0])));
        assert!(matches!(
            exchange(&request, b"{}", Stage::Generate, &mut transport),
            Err(VeoError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn veo2_default_unbound_adapter_and_empty_preparation_receipt_fail_closed() -> TestResult {
        struct Unbound;
        impl VeoPreparation for Unbound {}
        let case = first_case()?;
        assert!(request(&source_input(&case)?, &mut Unbound).is_err());
        let mut owner = SourceProbePreparation::new(&case)?;
        owner.bad_receipt = true;
        assert!(
            invoke(
                &source_input(&case)?,
                credentials(),
                1,
                &mut SourceProbeTransport::new(&case),
                &mut owner
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn veo2_real_encoded_video_rejection_publishes_no_native_placeholder() -> TestResult {
        use crate::common::{MaterializationError, MaterializedBatch};
        struct Decoder {
            calls: usize,
        }
        impl VeoVideoMaterializer for Decoder {
            fn materialize_video(
                &mut self,
                result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                if result.video_bytes
                    != include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../sim_test_support/fixtures/providers/veo2/source-response.mp4"
                    ))
                {
                    return Err(ConformanceError::InvalidResponse);
                }
                self.calls += 1;
                Err(ConformanceError::HostFailure)
            }
        }
        #[derive(Default)]
        struct Sink {
            calls: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(&mut self, _batch: MaterializedBatch) -> Result<(), MaterializationError> {
                self.calls += 1;
                Ok(())
            }
        }
        let case = first_case()?;
        let mut owner = SourceProbePreparation::new(&case)?;
        let mut transport = SourceProbeTransport::new(&case);
        let mut decoder = Decoder { calls: 0 };
        let mut sink = Sink::default();
        assert!(
            invoke_and_materialize(
                &source_input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut MaterializationOwners {
                    preparation: &mut owner,
                    video: &mut decoder,
                    sink: &mut sink
                }
            )
            .is_err()
        );
        assert_eq!(decoder.calls, 1);
        assert_eq!(sink.calls, 0);
        Ok(())
    }
}
