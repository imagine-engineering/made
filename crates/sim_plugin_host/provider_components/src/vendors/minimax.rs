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
pub const NAMESPACE: &str = "zed.sim.provider.minimax";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "33cbc7389f0c5c0f79431ed677b7aa750d4204ab47b9f73549d7d168cddb8a3c";
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
        feature_id: "COMFY-NODE-0415",
        node_identifier: "MinimaxHailuoVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0416",
        node_identifier: "MinimaxImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0417",
        node_identifier: "MinimaxTextToVideoNode",
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
        feature_id: "COMFY-API-EXT-0117",
        method: "GET",
        path: "/proxy/minimax/files/retrieve",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0118",
        method: "GET",
        path: "/proxy/minimax/query/video_generation",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0119",
        method: "POST",
        path: "/proxy/minimax/video_generation",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum MinimaxError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    Construction(String),
    Http {
        status: u16,
        retry_after: Option<String>,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for MinimaxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) | Self::Construction(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Minimax HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for MinimaxError {}
impl From<ConformanceError> for MinimaxError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for MinimaxError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, MinimaxError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/minimax/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, MinimaxError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(MinimaxError::Input("Unknown Minimax node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(MinimaxError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(MinimaxError::Input("Unknown Minimax source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(MinimaxError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(MinimaxError::Input("Missing Minimax source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, MinimaxError> {
    input
        .arguments
        .get(name)
        .ok_or(MinimaxError::Input("Missing Minimax argument"))
}
fn text(value: &Value) -> Result<&str, MinimaxError> {
    value
        .as_str()
        .ok_or(MinimaxError::Input("Expected Minimax string"))
}
fn endpoint(value: &str) -> Result<bool, MinimaxError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(MinimaxError::Response("Invalid Minimax response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(MinimaxError::Response("Invalid Minimax response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(MinimaxError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(MinimaxError::Response("Invalid Minimax endpoint authority"));
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
) -> Result<RequestHead, MinimaxError> {
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
            return Err(MinimaxError::Input("Missing Comfy API credentials"));
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
pub struct ImageUploadPolicy {
    pub maximum_images: u64,
    pub mime_type: Option<&'static str>,
    pub total_pixels: u64,
    pub wait_label: &'static str,
    pub show_batch_index: bool,
}
impl ImageUploadPolicy {
    fn source() -> Self {
        Self {
            maximum_images: 1,
            mime_type: None,
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
pub trait MinimaxImageOwner {
    fn upload(
        &mut self,
        image: &Value,
        policy: &ImageUploadPolicy,
    ) -> Result<ImageUpload, ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct Request {
    pub body: Value,
    pub estimated_seconds: u64,
    pub upload_receipt: Option<Vec<u8>>,
}
fn preflight(input: &Input) -> Result<(), MinimaxError> {
    let frame = match input.node.as_str() {
        "MinimaxImageToVideoNode" => argument(input, "image")?,
        "MinimaxHailuoVideoNode" => argument(input, "first_frame_image")?,
        "MinimaxTextToVideoNode" => &Value::Null,
        _ => return Err(MinimaxError::Input("Unknown MiniMax node")),
    };
    if frame.is_null() && argument(input, "prompt_text")?.is_null() {
        return Err(MinimaxError::Input("Field 'prompt_text' cannot be empty."));
    }
    if input.node == "MinimaxHailuoVideoNode"
        && argument(input, "model")? == "MiniMax-Hailuo-02"
        && text(argument(input, "resolution")?)?.to_uppercase() == "1080P"
        && argument(input, "duration")?.as_f64() != Some(6.0)
    {
        return Err(MinimaxError::Input(
            "When model is MiniMax-Hailuo-02 and resolution is 1080P, duration is limited to 6 seconds.",
        ));
    }
    Ok(())
}
fn api_integer(value: &Value) -> Result<i64, MinimaxError> {
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_str() {
        return value
            .parse()
            .map_err(|_| MinimaxError::Input("Invalid source integer"));
    }
    Err(MinimaxError::Input("Invalid source integer"))
}
fn api_boolean(value: &Value) -> Result<bool, MinimaxError> {
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
    Err(MinimaxError::Input("Invalid source boolean"))
}
pub fn request(input: &Input, owner: &mut dyn MinimaxImageOwner) -> Result<Request, MinimaxError> {
    let input = normalized(input)?;
    preflight(&input)?;
    let frame = match input.node.as_str() {
        "MinimaxImageToVideoNode" => argument(&input, "image")?,
        "MinimaxHailuoVideoNode" => argument(&input, "first_frame_image")?,
        _ => &Value::Null,
    };
    let mut upload_receipt = None;
    let mut body = json!({});
    if !frame.is_null() {
        let uploaded = owner.upload(frame, &ImageUploadPolicy::source())?;
        if endpoint(&uploaded.url)?
            || uploaded.receipt.is_empty()
            || uploaded.receipt.len() > MAXIMUM_RECEIPT_BYTES
        {
            return Err(MinimaxError::Response(
                "Invalid canonical image upload result",
            ));
        }
        body["first_frame_image"] = json!(uploaded.url);
        upload_receipt = Some(uploaded.receipt);
    }
    let model = text(argument(&input, "model")?)?;
    let source = oracle()?;
    if !source["api_models"]
        .as_array()
        .is_some_and(|models| models.iter().any(|value| value.as_str() == Some(model)))
    {
        return Err(MinimaxError::Input("Invalid MiniMax model enum"));
    }
    body["model"] = json!(model);
    let prompt = argument(&input, "prompt_text")?;
    if !prompt.is_null() {
        if text(prompt)?.chars().count() > 2000 {
            return Err(MinimaxError::Input("Source prompt exceeds 2000 characters"));
        }
        body["prompt"] = prompt.clone();
    }
    let estimated_seconds = if input.node == "MinimaxTextToVideoNode" {
        234
    } else if input.node == "MinimaxImageToVideoNode" {
        114
    } else {
        for (name, boolean) in [("prompt_optimizer", true), ("duration", false)] {
            let value = argument(&input, name)?;
            if !value.is_null() {
                body[name] = if boolean {
                    json!(api_boolean(value)?)
                } else {
                    json!(api_integer(value)?)
                };
            }
        }
        let resolution = argument(&input, "resolution")?;
        if !resolution.is_null() {
            body["resolution"] = json!(text(resolution)?);
        }
        if resolution == "768P" { 120 } else { 240 }
    };
    Ok(Request {
        body,
        estimated_seconds,
        upload_receipt,
    })
}
pub fn maximum_cost_microunits(input: &Input) -> Result<u64, MinimaxError> {
    let input = normalized(input)?;
    if input.node != "MinimaxHailuoVideoNode" {
        return Ok(430_000);
    }
    let source = oracle()?;
    let resolution = text(argument(&input, "resolution")?)?.to_lowercase();
    let duration = argument(&input, "duration")?;
    let duration = if let Some(value) = duration.as_str() {
        value.to_owned()
    } else {
        duration.to_string()
    };
    let rate = source["price_rates"][resolution][duration]
        .as_f64()
        .unwrap_or(0.43);
    let scaled = (rate * 1_000_000.0).round();
    if !scaled.is_finite() || !(0.0..=1_000_000_000.0).contains(&scaled) {
        return Err(MinimaxError::Input("Invalid source price rate"));
    }
    Ok(scaled as u64)
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponsePolicy {
    pub timeout_milliseconds: Option<u64>,
    pub monitor_progress: bool,
    pub wait_label: &'static str,
    pub final_label: Option<&'static str>,
}
impl ResponsePolicy {
    fn source() -> Self {
        Self {
            timeout_milliseconds: Some(3_600_000),
            monitor_progress: true,
            wait_label: "Waiting for server",
            final_label: Some("Completed"),
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PollPolicy {
    pub estimated_seconds: u64,
    pub timeout_milliseconds: u64,
    pub maximum_active_attempts: u32,
    pub interval_milliseconds: u64,
}
pub trait MinimaxTransport: ProviderTransport {
    fn configure_response(
        &mut self,
        handle: StreamHandle,
        policy: &ResponsePolicy,
    ) -> Result<(), ConformanceError>;
    fn configure_poll(
        &mut self,
        handle: StreamHandle,
        policy: &PollPolicy,
    ) -> Result<(), ConformanceError>;
    fn query_path(
        &mut self,
        path: &str,
        parameters: &[(String, String)],
    ) -> Result<String, ConformanceError>;
    fn source_integer_query(&mut self, value: &str) -> Result<String, ConformanceError>;
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
pub struct GenerationResult {
    pub plan: Request,
    pub creation_bytes: Vec<u8>,
    pub creation_receipt: Vec<u8>,
    pub response_bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub file_bytes: Vec<u8>,
    pub file_receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub primary_url: String,
    pub backup_url: Option<String>,
}
fn base_response(value: &Value) -> Result<(), MinimaxError> {
    if !value["base_resp"]["status_code"].is_i64() || !value["base_resp"]["status_msg"].is_string()
    {
        return Err(MinimaxError::Response("Missing MiniMax base response"));
    }
    Ok(())
}
fn json_response(bytes: &[u8]) -> Result<Value, MinimaxError> {
    let value: Value = serde_json::from_slice(bytes)?;
    if !value.is_object() {
        return Err(MinimaxError::Response("Invalid MiniMax response object"));
    }
    base_response(&value)?;
    Ok(value)
}
fn query_head(
    path: &str,
    name: &str,
    value: &str,
    credentials: CredentialAvailability,
    transport: &mut dyn MinimaxTransport,
) -> Result<RequestHead, MinimaxError> {
    let query = transport.query_path(path, &[(name.into(), value.into())])?;
    if !query.starts_with(&format!("{path}?{name}=")) || query.contains('#') {
        return Err(MinimaxError::Response("Invalid canonical query path"));
    }
    request_head(&query, HttpMethod::Get, credentials, false)
}
fn generation_identifier(response: &Value) -> Result<&str, MinimaxError> {
    let task_id = response["task_id"]
        .as_str()
        .ok_or(MinimaxError::Response("Missing MiniMax task ID"))?;
    if task_id.is_empty() {
        return Err(MinimaxError::Provider(format!(
            "MiniMax generation failed: {}",
            serde_json::to_string(&response["base_resp"])?
        )));
    }
    if task_id.len() > 8192 {
        return Err(MinimaxError::Response("Oversized MiniMax task ID"));
    }
    Ok(task_id)
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn MinimaxTransport,
    owner: &mut dyn MinimaxImageOwner,
) -> Result<GenerationResult, MinimaxError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    preflight(&input)?;
    let cost = maximum_cost_microunits(&input)?;
    if cost > ceiling_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    let head = request_head(
        "/proxy/minimax/video_generation",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    let handle = transport.start_request(&head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "minimax.generation.maximum".into(),
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
    let (creation_bytes, creation_receipt, _) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    let creation = json_response(&creation_bytes)?;
    let task_id = generation_identifier(&creation)?;
    let source = oracle()?;
    let mut active = 0u32;
    for sequence in 0..4096 {
        let head = query_head(
            "/proxy/minimax/query/video_generation",
            "task_id",
            task_id,
            credentials,
            transport,
        )?;
        transport.check_cancelled(None)?;
        let handle = transport.start_request(&head)?;
        transport.configure_poll(
            handle,
            &PollPolicy {
                estimated_seconds: plan.estimated_seconds,
                timeout_milliseconds: 120000,
                maximum_active_attempts: 480,
                interval_milliseconds: 5000,
            },
        )?;
        let (response_bytes, receipt, handle) = exchange_existing(
            &head,
            &[],
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            handle,
            transport,
        )?;
        let response = json_response(&response_bytes)?;
        if !response["task_id"].is_string() {
            return Err(MinimaxError::Response("Missing poll task ID"));
        }
        let status = response["status"]
            .as_str()
            .ok_or(MinimaxError::Response("Missing source MiniMax status"))?;
        if !["Queueing", "Preparing", "Processing", "Success", "Fail"].contains(&status) {
            return Err(MinimaxError::Response("Invalid source MiniMax status enum"));
        }
        let normalized_status = status.to_lowercase();
        let member = |name: &str| {
            source["polling"][name].as_array().is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some(&normalized_status))
            })
        };
        if member("FAILED_STATUSES") {
            return Err(MinimaxError::Provider(format!(
                "Task failed: {}",
                serde_json::to_string(&response)?
            )));
        }
        if member("COMPLETED_STATUSES") {
            let file_id = response["file_id"].as_str().ok_or(MinimaxError::Response(
                "Request was not successful. Missing file ID.",
            ))?;
            let integer = transport.source_integer_query(file_id)?;
            let digits = integer.strip_prefix('-').unwrap_or(&integer);
            if digits.is_empty()
                || digits.len() > 8192
                || !digits.bytes().all(|byte| byte.is_ascii_digit())
                || digits.len() > 1 && digits.starts_with('0')
            {
                return Err(MinimaxError::Response(
                    "Invalid canonical Python integer query",
                ));
            }
            let head = query_head(
                "/proxy/minimax/files/retrieve",
                "file_id",
                &integer,
                credentials,
                transport,
            )?;
            let (file_bytes, file_receipt, _) = exchange_with_policy(
                &head,
                &[],
                None,
                MAXIMUM_RESPONSE_BYTES,
                GENERATION_RETRY,
                transport,
            )?;
            let file_response = json_response(&file_bytes)?;
            if !file_response["file"].is_object() {
                return Err(MinimaxError::Response("Missing MiniMax file object"));
            }
            let primary_url = file_response["file"]["download_url"]
                .as_str()
                .ok_or_else(|| {
                    MinimaxError::Provider(format!(
                        "No video was found in the response. Full response: {}",
                        file_response
                    ))
                })?
                .to_owned();
            let backup_url = match file_response["file"].get("backup_download_url") {
                Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
                Some(Value::Null) | None | Some(Value::String(_)) => None,
                _ => return Err(MinimaxError::Response("Invalid backup URL field")),
            };
            transport.report_progress(
                handle,
                &Progress {
                    sequence: sequence as u64,
                    completed: 100,
                    total: 100,
                    message: Some("Completed".into()),
                },
            )?;
            return Ok(GenerationResult {
                plan,
                creation_bytes,
                creation_receipt,
                response_bytes,
                receipt,
                file_bytes,
                file_receipt,
                approval_receipt: approval.receipt,
                primary_url,
                backup_url,
            });
        }
        if !member("QUEUED_STATUSES") {
            active += 1
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
                message: Some(normalized_status),
            },
        )?;
        transport.check_cancelled(Some(handle))?;
        transport.poll_delay(handle, 5000)?;
    }
    Err(ConformanceError::PollLimit.into())
}
#[derive(Clone, Debug)]
pub struct DownloadedMedia {
    pub url: String,
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
#[derive(Debug)]
pub enum ConstructionError {
    InvalidMedia(String),
    Host(ConformanceError),
}
pub trait MinimaxVideoMaterializer {
    fn materialize(
        &mut self,
        result: &GenerationResult,
        media: &DownloadedMedia,
    ) -> Result<Vec<EncodedOutput>, ConstructionError>;
}
pub struct MaterializationOwners<'a> {
    pub image: &'a mut dyn MinimaxImageOwner,
    pub video: &'a mut dyn MinimaxVideoMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
fn video_schema() -> Result<Vec<ExpectedOutput>, MinimaxError> {
    let registry = TypeRegistry::built_in()
        .map_err(|_| MinimaxError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve("VIDEO")
        .map_err(|_| MinimaxError::Response("Missing canonical VIDEO type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| MinimaxError::Response("Missing canonical VIDEO family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
fn candidate(
    result: &GenerationResult,
    url: &str,
    primary_with_backup: bool,
    backup: bool,
    credentials: CredentialAvailability,
    transport: &mut dyn MinimaxTransport,
    materializer: &mut dyn MinimaxVideoMaterializer,
) -> Result<crate::common::MaterializedBatch, MinimaxError> {
    if endpoint(url)? {
        return Err(MinimaxError::Response(
            "Source download requires absolute URL without node context",
        ));
    }
    let head = request_head(url, HttpMethod::Get, credentials, true)?;
    let policy = RetryPolicy {
        ordinary_limit: if primary_with_backup {
            2
        } else if backup {
            3
        } else {
            5
        },
        rate_limit: 0,
        backoff: 2.0,
        download: true,
        timeout_milliseconds: primary_with_backup.then_some(10000),
    };
    let (bytes, receipt, _) =
        exchange_with_policy(&head, &[], None, MAXIMUM_MEDIA_BYTES, policy, transport)?;
    if bytes.is_empty() {
        return Err(MinimaxError::Response("Empty video download"));
    }
    transport.check_cancelled(None)?;
    let media = DownloadedMedia {
        url: url.into(),
        bytes,
        receipt,
    };
    let outputs = materializer
        .materialize(result, &media)
        .map_err(|error| match error {
            ConstructionError::InvalidMedia(message) => MinimaxError::Construction(message),
            ConstructionError::Host(error) => MinimaxError::Transport(error),
        })?;
    Ok(
        validate_outputs(&video_schema()?, outputs, MAXIMUM_MEDIA_BYTES)
            .map_err(ConformanceError::from)?,
    )
}
fn fallback_allowed(error: &MinimaxError) -> bool {
    !matches!(error, MinimaxError::Transport(_))
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn MinimaxTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), MinimaxError> {
    let result = invoke(
        input,
        credentials,
        ceiling_microunits,
        transport,
        owners.image,
    )?;
    let first = candidate(
        &result,
        &result.primary_url,
        result.backup_url.is_some(),
        false,
        credentials,
        transport,
        owners.video,
    );
    let batch = match first {
        Ok(batch) => batch,
        Err(error) if result.backup_url.is_some() && fallback_allowed(&error) => {
            transport.check_cancelled(None)?;
            candidate(
                &result,
                result
                    .backup_url
                    .as_deref()
                    .ok_or(MinimaxError::Response("Missing backup URL"))?,
                false,
                true,
                credentials,
                transport,
                owners.video,
            )?
        }
        Err(error) => return Err(error),
    };
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
    timeout_milliseconds: Some(120000),
};
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn MinimaxTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MinimaxError> {
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
    transport: &mut dyn MinimaxTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MinimaxError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        let mut response_policy = ResponsePolicy::source();
        response_policy.timeout_milliseconds = policy.timeout_milliseconds;
        if policy.download {
            response_policy.monitor_progress = false;
            response_policy.final_label = None;
        }
        transport.configure_response(handle, &response_policy)?;
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(MinimaxError::Http {
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
                    return Err(MinimaxError::Http {
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
    transport: &mut dyn MinimaxTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MinimaxError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "minimax.generation.maximum".into(),
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
                    return Err(MinimaxError::Http {
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
    cancelled: bool,
}
impl MinimaxImageOwner for SourceProbeOwner {
    fn upload(
        &mut self,
        image: &Value,
        policy: &ImageUploadPolicy,
    ) -> Result<ImageUpload, ConformanceError> {
        if self.cancelled {
            return Err(ConformanceError::Cancelled);
        }
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        self.operations.push(json!({"operation":"upload_images","image":image,"max_images":policy.maximum_images,"mime_type":policy.mime_type,"total_pixels":policy.total_pixels,"wait_label":policy.wait_label,"show_batch_index":policy.show_batch_index}));
        if image["shape"]
            .as_array()
            .and_then(|shape| shape.first())
            .and_then(Value::as_u64)
            == Some(0)
        {
            return Err(ConformanceError::InvalidResponse);
        }
        Ok(ImageUpload {
            url: "https://storage.fixture.invalid/image-0.png".into(),
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
    if source["source_sha256"]["comfy_api_nodes/nodes_minimax.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
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
        if plan.body != expected["body"]
            || expected["path"] != "/proxy/minimax/video_generation"
            || json!(owner.operations) != case["media"]
            || json!(plan.estimated_seconds) != case["polls"][0]["estimated_duration"]
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
        poll_policies: Vec<(StreamHandle, PollPolicy)>,
        integer_queries: Vec<String>,
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
    impl MinimaxTransport for Transport {
        fn configure_poll(
            &mut self,
            handle: StreamHandle,
            policy: &PollPolicy,
        ) -> Result<(), ConformanceError> {
            self.poll_policies.push((handle, policy.clone()));
            Ok(())
        }
        fn poll_delay(
            &mut self,
            _handle: StreamHandle,
            delay: u64,
        ) -> Result<(), ConformanceError> {
            self.delays.push(delay);
            Ok(())
        }
        fn query_path(
            &mut self,
            path: &str,
            parameters: &[(String, String)],
        ) -> Result<String, ConformanceError> {
            let (name, value) = parameters.first().ok_or(ConformanceError::InvalidRequest)?;
            Ok(format!("{path}?{name}={value}"))
        }
        fn source_integer_query(&mut self, value: &str) -> Result<String, ConformanceError> {
            self.integer_queries.push(value.into());
            Ok(value
                .parse::<i64>()
                .map_err(|_| ConformanceError::InvalidRequest)?
                .to_string())
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
    fn append(transport: &mut Transport, status: u16, value: Value) {
        transport
            .events
            .extend(events(status, value.to_string().into_bytes(), vec![]));
    }
    fn prepared(backup: bool) -> Transport {
        let mut transport = Transport::default();
        append(
            &mut transport,
            200,
            json!({"task_id":"source-id","base_resp":{"status_code":0,"status_msg":"source"}}),
        );
        append(
            &mut transport,
            200,
            json!({"task_id":"source-id","status":"Success","file_id":"00042","base_resp":{"status_code":0,"status_msg":"source"}}),
        );
        append(
            &mut transport,
            200,
            json!({"base_resp":{"status_code":0,"status_msg":"source"},"file":{"download_url":"https://media.fixture.invalid/primary.mp4","backup_download_url":if backup{json!("https://media.fixture.invalid/backup.mp4")}else{Value::Null}}}),
        );
        transport
    }
    fn video() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/minimax/downloaded-video.mp4"
        ))
        .to_vec()
    }
    #[derive(Default)]
    struct Sink {
        published: usize,
    }
    impl MaterializationSink for Sink {
        fn publish(&mut self, _batch: MaterializedBatch) -> Result<(), MaterializationError> {
            self.published += 1;
            Ok(())
        }
    }
    #[derive(Default)]
    struct RejectingMaterializer {
        seen: Vec<DownloadedMedia>,
        host_failure: bool,
        empty: bool,
    }
    impl MinimaxVideoMaterializer for RejectingMaterializer {
        fn materialize(
            &mut self,
            _result: &GenerationResult,
            media: &DownloadedMedia,
        ) -> Result<Vec<EncodedOutput>, ConstructionError> {
            self.seen.push(media.clone());
            if self.host_failure {
                return Err(ConstructionError::Host(ConformanceError::HostFailure));
            }
            if self.empty {
                return Ok(vec![]);
            }
            Err(ConstructionError::InvalidMedia(
                "test canonical construction rejection; no native result".into(),
            ))
        }
    }
    #[test]
    fn minimax_all_registered_source_requests_defaults_and_preflight_order_match()
    -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        for case in source["cases"].as_array().ok_or("cases")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            let plan = request(&input, &mut owner)?;
            assert_eq!(plan.body, case["requests"][0]["body"], "{}", input.node);
            assert_eq!(
                json!(plan.estimated_seconds),
                case["polls"][0]["estimated_duration"]
            );
            assert_eq!(json!(owner.operations), case["media"]);
        }
        for case in source["preflight_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            assert!(request(&input, &mut owner).is_err(), "{}", case["message"]);
            assert_eq!(json!(owner.operations), case["media"]);
        }
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/minimax/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn minimax_subject_class_is_not_registered_and_seed_never_reaches_api()
    -> Result<(), Box<dyn Error>> {
        assert_eq!(NODES.len(), 3);
        assert_eq!(ROUTES.len(), 3);
        assert!(ALIASES.is_empty());
        let mut source = input("MinimaxTextToVideoNode")?;
        source.node = "MinimaxSubjectToVideoNode".into();
        assert!(request(&source, &mut SourceProbeOwner::default()).is_err());
        for node in [
            "MinimaxTextToVideoNode",
            "MinimaxImageToVideoNode",
            "MinimaxHailuoVideoNode",
        ] {
            let plan = request(&input(node)?, &mut SourceProbeOwner::default())?;
            assert!(plan.body.get("seed").is_none());
            assert!(plan.body.get("callback_url").is_none());
            assert!(plan.body.get("subject_reference").is_none());
            if node != "MinimaxHailuoVideoNode" {
                assert!(plan.body.get("prompt_optimizer").is_none());
                assert!(plan.body.get("duration").is_none());
                assert!(plan.body.get("resolution").is_none());
            }
        }
        Ok(())
    }
    #[test]
    fn minimax_all_source_price_literals_and_hailuo_fallback_match() -> Result<(), Box<dyn Error>> {
        for quote in oracle()?["price_cases"].as_array().ok_or("quotes")? {
            let mut source = input(quote["node"].as_str().ok_or("node")?)?;
            if source.node == "MinimaxHailuoVideoNode" {
                source
                    .arguments
                    .insert("resolution".into(), quote["resolution"].clone());
                source
                    .arguments
                    .insert("duration".into(), quote["duration"].clone());
            }
            assert_eq!(
                maximum_cost_microunits(&source)?,
                quote["maximum_microunits"].as_u64().ok_or("price")?
            );
        }
        Ok(())
    }
    #[test]
    fn minimax_submit_poll_and_file_queries_preserve_order_receipts_and_auth()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "MinimaxTextToVideoNode",
            "MinimaxImageToVideoNode",
            "MinimaxHailuoVideoNode",
        ] {
            let mut transport = prepared(true);
            let result = invoke(
                &input(node)?,
                credentials(),
                2_000_000,
                &mut transport,
                &mut SourceProbeOwner::default(),
            )?;
            assert_eq!(transport.heads.len(), 3);
            assert_eq!(
                transport.heads.get(1).ok_or("poll")?.path,
                "/proxy/minimax/query/video_generation?task_id=source-id"
            );
            assert_eq!(
                transport.heads.get(2).ok_or("file")?.path,
                "/proxy/minimax/files/retrieve?file_id=42"
            );
            assert_eq!(transport.integer_queries, vec!["00042"]);
            assert_eq!(transport.costs.len(), 1);
            let cost = transport.costs.first().ok_or("cost")?;
            assert_eq!(cost.operation, "minimax.generation.maximum");
            assert_eq!(cost.currency, "USD");
            assert_eq!(
                cost.maximum_microunits,
                if node == "MinimaxHailuoVideoNode" {
                    280000
                } else {
                    430000
                }
            );
            assert_eq!(result.creation_receipt, vec![7, 8, 9]);
            assert_eq!(result.receipt, vec![7, 8, 9]);
            assert_eq!(result.file_receipt, vec![7, 8, 9]);
            assert_eq!(result.approval_receipt, vec![42]);
            assert_eq!(
                result.backup_url.as_deref(),
                Some("https://media.fixture.invalid/backup.mp4")
            );
            assert!(
                transport
                    .heads
                    .iter()
                    .all(|head| head.secret_id.as_deref() == Some("sim_org.auth_token"))
            );
            assert_eq!(
                transport
                    .poll_policies
                    .first()
                    .ok_or("policy")?
                    .1
                    .estimated_seconds,
                result.plan.estimated_seconds
            );
        }
        Ok(())
    }
    #[test]
    fn minimax_source_response_failures_stop_before_file_or_download() -> Result<(), Box<dyn Error>>
    {
        for case in oracle()?["response_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut transport = Transport::default();
            let mut initial =
                json!({"task_id":"source-id","base_resp":{"status_code":0,"status_msg":"source"}});
            if let Some(fields) = case["response_override"]["initial"].as_object() {
                for (key, value) in fields {
                    initial[key] = value.clone()
                }
            }
            append(&mut transport, 200, initial);
            let mut poll = json!({"task_id":"source-id","status":"Success","file_id":"00042","base_resp":{"status_code":0,"status_msg":"source"}});
            if let Some(fields) = case["response_override"]["poll"].as_object() {
                for (key, value) in fields {
                    poll[key] = value.clone()
                }
            }
            append(&mut transport, 200, poll);
            let mut file = json!({"base_resp":{"status_code":0,"status_msg":"source"},"file":{"download_url":"https://media.fixture.invalid/primary.mp4"}});
            if let Some(fields) = case["response_override"]["file"].as_object() {
                for (key, value) in fields {
                    file[key] = value.clone()
                }
            }
            append(&mut transport, 200, file);
            let error = invoke(
                &input,
                credentials(),
                2_000_000,
                &mut transport,
                &mut SourceProbeOwner::default(),
            )
            .err()
            .ok_or("failure")?;
            assert!(
                transport
                    .heads
                    .iter()
                    .all(|head| head.path.starts_with('/'))
            );
            if case["response_override"]["initial"]["task_id"] == "" {
                assert!(error.to_string().starts_with("MiniMax generation failed:"))
            }
            if case["response_override"]["poll"].get("file_id") == Some(&Value::Null) {
                assert_eq!(
                    error.to_string(),
                    "Request was not successful. Missing file ID."
                )
            }
        }
        Ok(())
    }
    #[test]
    fn minimax_native_construction_failure_tries_backup_without_paid_replay()
    -> Result<(), Box<dyn Error>> {
        let mut transport = prepared(true);
        transport.events.extend(events(200, video(), vec![]));
        transport.events.extend(events(200, video(), vec![]));
        let mut owner = SourceProbeOwner::default();
        let mut materializer = RejectingMaterializer::default();
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input("MinimaxHailuoVideoNode")?,
                credentials(),
                2_000_000,
                &mut transport,
                &mut MaterializationOwners {
                    image: &mut owner,
                    video: &mut materializer,
                    sink: &mut sink
                }
            ),
            Err(MinimaxError::Construction(_))
        ));
        assert_eq!(sink.published, 0);
        assert_eq!(materializer.seen.len(), 2);
        assert!(
            materializer
                .seen
                .iter()
                .all(|media| media.bytes == video() && media.receipt == vec![7, 8, 9])
        );
        assert_eq!(
            materializer.seen.first().ok_or("primary")?.url,
            "https://media.fixture.invalid/primary.mp4"
        );
        assert_eq!(
            materializer.seen.get(1).ok_or("backup")?.url,
            "https://media.fixture.invalid/backup.mp4"
        );
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(transport.heads.len(), 5);
        assert!(
            transport
                .heads
                .iter()
                .skip(3)
                .all(|head| head.secret_id.is_none())
        );
        assert!(
            transport
                .policies
                .iter()
                .any(|(_, policy)| policy.timeout_milliseconds == Some(10000))
        );
        assert!(
            transport
                .policies
                .last()
                .ok_or("policy")?
                .1
                .timeout_milliseconds
                .is_none()
        );
        Ok(())
    }
    #[test]
    fn minimax_primary_two_retries_and_backup_three_retries_are_source_exact()
    -> Result<(), Box<dyn Error>> {
        let mut transport = prepared(true);
        for _ in 0..3 {
            transport.events.extend(events(500, vec![], vec![]));
        }
        for _ in 0..4 {
            transport.events.extend(events(429, vec![], vec![]));
        }
        let mut owner = SourceProbeOwner::default();
        let mut materializer = RejectingMaterializer::default();
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input("MinimaxTextToVideoNode")?,
                credentials(),
                2_000_000,
                &mut transport,
                &mut MaterializationOwners {
                    image: &mut owner,
                    video: &mut materializer,
                    sink: &mut sink
                }
            ),
            Err(MinimaxError::Http { status: 429, .. })
        ));
        assert_eq!(transport.delays, vec![1000, 2000, 1000, 2000, 4000]);
        assert!(materializer.seen.is_empty());
        assert_eq!(sink.published, 0);
        assert_eq!(transport.costs.len(), 1);
        let primary = transport
            .policies
            .iter()
            .filter(|(_, policy)| policy.timeout_milliseconds == Some(10000))
            .count();
        assert_eq!(primary, 3);
        let downloads = transport
            .heads
            .iter()
            .filter(|head| head.path.starts_with("https://"))
            .count();
        assert_eq!(downloads, 7);
        Ok(())
    }
    #[test]
    fn minimax_no_backup_has_five_download_retries_and_no_ten_second_override()
    -> Result<(), Box<dyn Error>> {
        let mut transport = prepared(false);
        for _ in 0..6 {
            transport.events.extend(events(503, vec![], vec![]));
        }
        let mut owner = SourceProbeOwner::default();
        let mut materializer = RejectingMaterializer::default();
        let mut sink = Sink::default();
        assert!(
            invoke_and_materialize(
                &input("MinimaxTextToVideoNode")?,
                credentials(),
                2_000_000,
                &mut transport,
                &mut MaterializationOwners {
                    image: &mut owner,
                    video: &mut materializer,
                    sink: &mut sink
                }
            )
            .is_err()
        );
        assert_eq!(transport.delays, vec![1000, 2000, 4000, 8000, 16000]);
        assert!(
            transport
                .policies
                .iter()
                .all(|(_, policy)| policy.timeout_milliseconds != Some(10000))
        );
        assert_eq!(sink.published, 0);
        Ok(())
    }
    #[test]
    fn minimax_host_loss_cancellation_and_invalid_abi_batch_never_try_backup()
    -> Result<(), Box<dyn Error>> {
        for mode in ["host", "empty", "cancelled"] {
            let mut transport = prepared(true);
            transport.events.extend(events(200, video(), vec![]));
            if mode == "cancelled" {
                transport.events = vec![Ok(ResponseEvent::Cancelled)].into();
            }
            let mut owner = SourceProbeOwner::default();
            let mut materializer = RejectingMaterializer {
                host_failure: mode == "host",
                empty: mode == "empty",
                ..RejectingMaterializer::default()
            };
            let mut sink = Sink::default();
            assert!(
                invoke_and_materialize(
                    &input("MinimaxImageToVideoNode")?,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut MaterializationOwners {
                        image: &mut owner,
                        video: &mut materializer,
                        sink: &mut sink
                    }
                )
                .is_err()
            );
            assert_eq!(sink.published, 0);
            assert!(
                transport
                    .heads
                    .iter()
                    .all(|head| !head.path.contains("backup.mp4"))
            );
        }
        Ok(())
    }
    #[test]
    fn minimax_cost_auth_and_canonical_upload_fail_before_generation_body()
    -> Result<(), Box<dyn Error>> {
        let source = input("MinimaxImageToVideoNode")?;
        for mode in ["ceiling", "auth", "approval", "upload", "cancelled"] {
            let mut transport = Transport {
                deny_cost: mode == "approval",
                ..Transport::default()
            };
            let mut owner = SourceProbeOwner {
                fail: mode == "upload",
                cancelled: mode == "cancelled",
                ..SourceProbeOwner::default()
            };
            assert!(
                invoke(
                    &source,
                    CredentialAvailability {
                        auth_token: mode != "auth",
                        api_key: false
                    },
                    if mode == "ceiling" { 0 } else { 2_000_000 },
                    &mut transport,
                    &mut owner
                )
                .is_err()
            );
            assert!(transport.bodies.iter().all(Vec::is_empty));
            if ["ceiling", "auth", "approval"].contains(&mode) {
                assert!(owner.operations.is_empty());
            }
        }
        let mut transport = prepared(false);
        invoke(
            &source,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            2_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert!(
            transport
                .heads
                .first()
                .ok_or("head")?
                .headers
                .iter()
                .any(|header| header.name == "X-API-KEY")
        );
        Ok(())
    }
    #[test]
    fn minimax_queued_statuses_do_not_consume_active_budget_and_use_clock_owner()
    -> Result<(), Box<dyn Error>> {
        let mut transport = Transport::default();
        append(
            &mut transport,
            200,
            json!({"task_id":"source-id","base_resp":{"status_code":0,"status_msg":"source"}}),
        );
        for status in ["Queueing", "Preparing", "Processing"] {
            append(
                &mut transport,
                200,
                json!({"task_id":"source-id","status":status,"base_resp":{"status_code":0,"status_msg":"source"}}),
            );
        }
        append(
            &mut transport,
            200,
            json!({"task_id":"source-id","status":"Success","file_id":"00042","base_resp":{"status_code":0,"status_msg":"source"}}),
        );
        append(
            &mut transport,
            200,
            json!({"file":{"download_url":"https://media.fixture.invalid/primary.mp4"},"base_resp":{"status_code":0,"status_msg":"source"}}),
        );
        let result = invoke(
            &input("MinimaxHailuoVideoNode")?,
            credentials(),
            2_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(transport.delays, vec![5000; 3]);
        assert_eq!(result.plan.estimated_seconds, 120);
        assert!(
            transport
                .poll_policies
                .iter()
                .all(|(_, policy)| policy.timeout_milliseconds == 120000
                    && policy.maximum_active_attempts == 480)
        );
        Ok(())
    }
    #[test]
    fn minimax_active_and_queued_poll_limits_and_fail_status_are_bounded()
    -> Result<(), Box<dyn Error>> {
        for (status, count) in [("Processing", 480), ("Queueing", 4096), ("Fail", 1)] {
            let mut transport = Transport::default();
            append(
                &mut transport,
                200,
                json!({"task_id":"source-id","base_resp":{"status_code":0,"status_msg":"source"}}),
            );
            for _ in 0..count {
                append(
                    &mut transport,
                    200,
                    json!({"task_id":"source-id","status":status,"base_resp":{"status_code":7,"status_msg":"source-failure"}}),
                );
            }
            let error = invoke(
                &input("MinimaxTextToVideoNode")?,
                credentials(),
                2_000_000,
                &mut transport,
                &mut SourceProbeOwner::default(),
            )
            .err()
            .ok_or("failure")?;
            if status == "Fail" {
                assert!(error.to_string().contains("source-failure"))
            } else {
                assert!(matches!(
                    error,
                    MinimaxError::Transport(ConformanceError::PollLimit)
                ));
                assert_eq!(transport.delays.len(), count - 1);
            }
        }
        Ok(())
    }
    #[test]
    fn minimax_generation_and_poll_retry_limits_preserve_fresh_handles()
    -> Result<(), Box<dyn Error>> {
        for (policy, status, retries) in [
            (GENERATION_RETRY, 500, 3),
            (GENERATION_RETRY, 429, 16),
            (POLL_RETRY, 503, 10),
        ] {
            let mut transport = Transport::default();
            for _ in 0..=retries {
                transport.events.extend(events(
                    status,
                    vec![],
                    vec![Header {
                        name: "Retry-After".into(),
                        value: "200".into(),
                    }],
                ));
            }
            assert!(matches!(
                exchange_with_policy(
                    &request_head(
                        "/proxy/minimax/video_generation",
                        HttpMethod::Post,
                        credentials(),
                        false
                    )?,
                    b"{}",
                    None,
                    1024,
                    policy,
                    &mut transport
                ),
                Err(MinimaxError::Http { .. })
            ));
            assert_eq!(transport.delays.len(), retries);
            assert!(transport.delays.iter().all(|delay| *delay == 150000));
        }
        let mut transport = Transport {
            same_retry_handle: true,
            events: events(500, vec![], vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(
                &request_head(
                    "/proxy/minimax/video_generation",
                    HttpMethod::Post,
                    credentials(),
                    false
                )?,
                b"{}",
                None,
                1024,
                GENERATION_RETRY,
                &mut transport
            ),
            Err(MinimaxError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn minimax_malformed_response_headers_chunks_receipts_and_aggregate_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/minimax/video_generation",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        for mut transport in [
            Transport {
                events: vec![Ok(ResponseEvent::Chunk(vec![1]))].into(),
                ..Transport::default()
            },
            Transport {
                events: events(200, vec![1; MAXIMUM_CHUNK_BYTES + 1], vec![]),
                ..Transport::default()
            },
            Transport {
                events: events(
                    200,
                    vec![1],
                    vec![Header {
                        name: "bad\nheader".into(),
                        value: "x".into(),
                    }],
                ),
                ..Transport::default()
            },
            Transport {
                events: vec![
                    Ok(ResponseEvent::Head {
                        status: 200,
                        headers: vec![],
                    }),
                    Ok(ResponseEvent::Completed(vec![])),
                ]
                .into(),
                ..Transport::default()
            },
        ] {
            assert!(
                exchange_with_policy(
                    &head,
                    b"{}",
                    None,
                    MAXIMUM_MEDIA_BYTES,
                    GENERATION_RETRY,
                    &mut transport
                )
                .is_err()
            );
        }
        let mut transport = Transport::default();
        transport.events.push_back(Ok(ResponseEvent::Head {
            status: 200,
            headers: vec![],
        }));
        for _ in 0..MAXIMUM_MEDIA_BYTES / MAXIMUM_CHUNK_BYTES {
            transport
                .events
                .push_back(Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES])));
        }
        transport
            .events
            .push_back(Ok(ResponseEvent::Chunk(vec![1])));
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                MAXIMUM_MEDIA_BYTES,
                GENERATION_RETRY,
                &mut transport
            ),
            Err(MinimaxError::Transport(ConformanceError::BodyLimit))
        ));
        for bytes in [b"bad".as_slice(), b"[]".as_slice(), b"{}".as_slice()] {
            assert!(json_response(bytes).is_err())
        }
        Ok(())
    }
    #[test]
    fn minimax_fresh_transport_without_evidence_does_not_replay_generation()
    -> Result<(), Box<dyn Error>> {
        for _ in 0..2 {
            let mut transport = Transport::default();
            assert!(matches!(
                invoke(
                    &input("MinimaxTextToVideoNode")?,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                ),
                Err(MinimaxError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(transport.heads.len(), 1);
            assert!(transport.delays.is_empty());
        }
        Ok(())
    }
}
