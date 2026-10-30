use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, Progress, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TransportLimits, TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
pub const NAMESPACE: &str = "zed.sim.provider.pixverse";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "3b96c96e9b5f6458e8ec6e93392f0cf9dae76b733b721ac55ead15a5351e7f84";
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
        feature_id: "COMFY-NODE-0481",
        node_identifier: "PixverseImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0482",
        node_identifier: "PixverseTemplateNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0483",
        node_identifier: "PixverseTextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0484",
        node_identifier: "PixverseTransitionVideoNode",
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
        feature_id: "COMFY-API-EXT-0128",
        method: "POST",
        path: "/proxy/pixverse/image/upload",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0129",
        method: "POST",
        path: "/proxy/pixverse/video/img/generate",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0130",
        method: "GET",
        path: "/proxy/pixverse/video/result/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0131",
        method: "GET",
        path: "/proxy/pixverse/video/result/{response_api.Resp.video_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0132",
        method: "POST",
        path: "/proxy/pixverse/video/text/generate",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0133",
        method: "POST",
        path: "/proxy/pixverse/video/transition/generate",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum PixverseError {
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
impl fmt::Display for PixverseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Pixverse HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for PixverseError {}
impl From<ConformanceError> for PixverseError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for PixverseError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, PixverseError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/pixverse/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, PixverseError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(PixverseError::Input("Unknown Pixverse node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(PixverseError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(PixverseError::Input("Unknown Pixverse source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(PixverseError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(PixverseError::Input("Missing Pixverse source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, PixverseError> {
    input
        .arguments
        .get(name)
        .ok_or(PixverseError::Input("Missing Pixverse argument"))
}
fn text(value: &Value) -> Result<&str, PixverseError> {
    value
        .as_str()
        .ok_or(PixverseError::Input("Expected Pixverse string"))
}
fn endpoint(value: &str) -> Result<bool, PixverseError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(PixverseError::Response(
            "Invalid Pixverse response endpoint",
        ));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(PixverseError::Response(
            "Invalid Pixverse response endpoint",
        ))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(PixverseError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(PixverseError::Response(
            "Invalid Pixverse endpoint authority",
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
) -> Result<RequestHead, PixverseError> {
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
            return Err(PixverseError::Input("Missing Comfy API credentials"));
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
pub struct ImagePolicy {
    pub mime_type: &'static str,
    pub total_pixels: u64,
    pub first_batch: bool,
}
impl ImagePolicy {
    fn source() -> Self {
        Self {
            mime_type: "image/png",
            total_pixels: 4_194_304,
            first_batch: true,
        }
    }
}
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub bytes: Vec<u8>,
    pub filename: String,
}
pub trait PixverseImageOwner {
    fn prepare(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<PreparedImage, ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct UploadResult {
    pub image_id: Option<i64>,
    pub response_bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
pub trait PixverseUploadOwner {
    fn upload(&mut self, image: &Value) -> Result<UploadResult, PixverseError>;
}
#[derive(Clone, Debug)]
pub struct Request {
    pub body: Value,
    pub path: Option<&'static str>,
    pub estimated_seconds: u64,
    pub uploads: Vec<UploadResult>,
}
fn preflight(input: &Input) -> Result<(), PixverseError> {
    if input.node == "PixverseTemplateNode" {
        return Ok(());
    }
    let prompt = argument(input, "prompt")?;
    if prompt.is_null() {
        return Err(PixverseError::Input("Field 'prompt' cannot be empty."));
    }
    if text(prompt)?.is_empty() && input.node == "PixverseTextToVideoNode" {
        return Err(PixverseError::Input(
            "Source text prompt must contain at least one character",
        ));
    }
    Ok(())
}
fn omit_negative_prompt(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::Number(value) => value.as_f64() == Some(0.0),
        Value::String(value) => value.is_empty(),
        Value::Array(value) => value.is_empty(),
        Value::Object(value) => value.is_empty(),
        Value::Bool(true) => false,
    }
}
fn source_integer(value: &Value) -> Result<i64, PixverseError> {
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_str() {
        return value
            .parse()
            .map_err(|_| PixverseError::Input("Invalid source integer"));
    }
    Err(PixverseError::Input("Invalid source integer"))
}
fn source_enum(value: &Value, name: &str) -> Result<(), PixverseError> {
    if oracle()?["api_enums"][name]
        .as_array()
        .is_some_and(|values| values.contains(value))
    {
        Ok(())
    } else {
        Err(PixverseError::Input("Invalid source enum"))
    }
}
pub fn request(
    input: &Input,
    uploads: &mut dyn PixverseUploadOwner,
) -> Result<Request, PixverseError> {
    let input = normalized(input)?;
    preflight(&input)?;
    if input.node == "PixverseTemplateNode" {
        let name = text(argument(&input, "template")?)?;
        let value = oracle()?["templates"]
            .get(name)
            .and_then(Value::as_i64)
            .ok_or_else(|| {
                PixverseError::Provider(format!("Template '{name}' is not recognized."))
            })?;
        return Ok(Request {
            body: json!(value),
            path: None,
            estimated_seconds: 0,
            uploads: vec![],
        });
    }
    let mut images = vec![];
    let mut body = json!({});
    match input.node.as_str() {
        "PixverseImageToVideoNode" => {
            let result = uploads.upload(argument(&input, "image")?)?;
            body["img_id"] = json!(result.image_id);
            images.push(result);
        }
        "PixverseTransitionVideoNode" => {
            for (input_name, field) in [
                ("first_frame", "first_frame_img"),
                ("last_frame", "last_frame_img"),
            ] {
                let result = uploads.upload(argument(&input, input_name)?)?;
                body[field] = json!(result.image_id);
                images.push(result);
            }
        }
        _ => {}
    }
    for field in ["img_id", "first_frame_img", "last_frame_img"] {
        if body.get(field) == Some(&Value::Null) {
            return Err(PixverseError::Input("Missing source required image ID"));
        }
    }
    let quality = argument(&input, "quality")?;
    let mut duration = argument(&input, "duration_seconds")?.clone();
    let mut motion = argument(&input, "motion_mode")?.clone();
    if quality == "1080p" {
        duration = json!(5);
        motion = json!("normal");
    } else if duration != json!(5) {
        motion = json!("normal");
    }
    source_enum(quality, "PixverseQuality")?;
    source_enum(&json!(source_integer(&duration)?), "PixverseDuration")?;
    source_enum(&motion, "PixverseMotionMode")?;
    body["quality"] = quality.clone();
    body["duration"] = json!(source_integer(&duration)?);
    body["motion_mode"] = motion;
    body["model"] = json!("v3.5");
    body["prompt"] = json!(text(argument(&input, "prompt")?)?);
    let seed = argument(&input, "seed")?;
    if !seed.is_null() {
        body["seed"] = json!(source_integer(seed)?);
    }
    if input.node != "PixverseTransitionVideoNode" {
        let negative = argument(&input, "negative_prompt")?;
        if !omit_negative_prompt(negative) {
            text(negative)?;
            body["negative_prompt"] = negative.clone();
        }
        let template = argument(&input, "pixverse_template")?;
        if !template.is_null() {
            body["template_id"] = json!(source_integer(template)?);
        }
    }
    let (path, estimated_seconds) = match input.node.as_str() {
        "PixverseTextToVideoNode" => {
            let ratio = argument(&input, "aspect_ratio")?;
            source_enum(ratio, "PixverseAspectRatio")?;
            body["aspect_ratio"] = ratio.clone();
            ("/proxy/pixverse/video/text/generate", 32)
        }
        "PixverseImageToVideoNode" => ("/proxy/pixverse/video/img/generate", 30),
        "PixverseTransitionVideoNode" => ("/proxy/pixverse/video/transition/generate", 32),
        _ => return Err(PixverseError::Input("Unknown Pixverse node")),
    };
    Ok(Request {
        body,
        path: Some(path),
        estimated_seconds,
        uploads: images,
    })
}
pub fn maximum_cost_microunits(input: &Input) -> Result<u64, PixverseError> {
    let input = normalized(input)?;
    if input.node == "PixverseTemplateNode" {
        return Ok(0);
    }
    let source = oracle()?;
    let duration = argument(&input, "duration_seconds")?.to_string();
    let quality = text(argument(&input, "quality")?)?;
    let motion = text(argument(&input, "motion_mode")?)?;
    let rate = source["price_rates"][duration][quality][motion]
        .as_f64()
        .unwrap_or(0.9);
    let scaled = (rate * 1_000_000.0).round();
    if !scaled.is_finite() || !(0.0..=1_000_000_000.0).contains(&scaled) {
        return Err(PixverseError::Input("Invalid source price"));
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
pub trait PixverseTransport: ProviderTransport {
    fn multipart_boundary(&mut self) -> Result<String, ConformanceError>;
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
fn json_response(bytes: &[u8]) -> Result<Value, PixverseError> {
    let value: Value = serde_json::from_slice(bytes)?;
    if !value.is_object() {
        return Err(PixverseError::Response("Invalid Pixverse response object"));
    }
    Ok(value)
}
fn provider_response(response: &Value, upload: bool) -> Result<&Value, PixverseError> {
    let value = &response["Resp"];
    if value.is_null() {
        let message = response["ErrMsg"].as_str().unwrap_or("None");
        return Err(PixverseError::Provider(if upload {
            format!("PixVerse image upload request failed: '{message}'")
        } else {
            format!("PixVerse request failed: '{message}'")
        }));
    }
    if !value.is_object() {
        return Err(PixverseError::Response("Invalid Pixverse Resp object"));
    }
    Ok(value)
}
struct LiveUploads<'a> {
    credentials: CredentialAvailability,
    transport: &'a mut dyn PixverseTransport,
    image: &'a mut dyn PixverseImageOwner,
}
impl PixverseUploadOwner for LiveUploads<'_> {
    fn upload(&mut self, image: &Value) -> Result<UploadResult, PixverseError> {
        self.transport.check_cancelled(None)?;
        let prepared = self.image.prepare(image, &ImagePolicy::source())?;
        if !prepared.bytes.starts_with(b"\x89PNG\r\n\x1a\n")
            || prepared.filename.is_empty()
            || !prepared.filename.ends_with(".png")
        {
            return Err(PixverseError::Response("Invalid canonical named PNG"));
        }
        let boundary = self.transport.multipart_boundary()?;
        let mut limits = TransportLimits::conformance();
        limits.maximum_body_bytes = MAXIMUM_REQUEST_BYTES;
        let body = MultipartField::encode_checked(
            &[MultipartField {
                name: "image".into(),
                file_name: Some(prepared.filename),
                media_type: Some("application/octet-stream".into()),
                bytes: prepared.bytes,
            }],
            &boundary,
            limits,
        )?;
        let mut head = request_head(
            "/proxy/pixverse/image/upload",
            HttpMethod::Post,
            self.credentials,
            false,
        )?;
        let header = head
            .headers
            .iter_mut()
            .find(|header| header.name == "Content-Type")
            .ok_or(PixverseError::Response("Missing multipart content type"))?;
        header.value = format!("multipart/form-data; boundary={boundary}");
        let (response_bytes, receipt, _) = exchange_with_policy(
            &head,
            &body,
            None,
            MAXIMUM_RESPONSE_BYTES,
            GENERATION_RETRY,
            self.transport,
        )?;
        let response = json_response(&response_bytes)?;
        let value = &provider_response(&response, true)?["img_id"];
        let image_id = match value {
            Value::Null => None,
            Value::Number(number) => Some(
                number
                    .as_i64()
                    .ok_or(PixverseError::Response("Invalid Pixverse image ID"))?,
            ),
            _ => return Err(PixverseError::Response("Invalid Pixverse image ID")),
        };
        Ok(UploadResult {
            image_id,
            response_bytes,
            receipt,
        })
    }
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub creation_bytes: Vec<u8>,
    pub creation_receipt: Vec<u8>,
    pub poll_bytes: Vec<u8>,
    pub poll_receipt: Vec<u8>,
    pub downloaded_bytes: Vec<u8>,
    pub download_receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub url: Option<String>,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn PixverseTransport,
    image: &mut dyn PixverseImageOwner,
) -> Result<ResultValue, PixverseError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    preflight(&input)?;
    if input.node == "PixverseTemplateNode" {
        let plan = request(&input, &mut SourceProbeOwner::default())?;
        return Ok(ResultValue {
            plan,
            creation_bytes: vec![],
            creation_receipt: vec![],
            poll_bytes: vec![],
            poll_receipt: vec![],
            downloaded_bytes: vec![],
            download_receipt: vec![],
            approval_receipt: vec![],
            url: None,
        });
    }
    let cost = maximum_cost_microunits(&input)?;
    if cost > ceiling_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    let path = match input.node.as_str() {
        "PixverseTextToVideoNode" => "/proxy/pixverse/video/text/generate",
        "PixverseImageToVideoNode" => "/proxy/pixverse/video/img/generate",
        "PixverseTransitionVideoNode" => "/proxy/pixverse/video/transition/generate",
        _ => return Err(PixverseError::Input("Unknown Pixverse node")),
    };
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    let generation_handle = transport.start_request(&head)?;
    let approval = transport.request_cost(
        generation_handle,
        &CostRequest {
            operation: "pixverse.generation.maximum".into(),
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
    let plan = request(
        &input,
        &mut LiveUploads {
            credentials,
            transport,
            image,
        },
    )?;
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
        generation_handle,
        transport,
    )?;
    let creation = json_response(&creation_bytes)?;
    let video_id = provider_response(&creation, false)?["video_id"]
        .as_i64()
        .ok_or(PixverseError::Response("Missing Pixverse video ID"))?;
    let poll_path = format!("/proxy/pixverse/video/result/{video_id}");
    for sequence in 0..480 {
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
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
        let (poll_bytes, poll_receipt, handle) = exchange_existing(
            &head,
            &[],
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            handle,
            transport,
        )?;
        let poll = json_response(&poll_bytes)?;
        let status = match poll["Resp"].get("status") {
            Some(Value::Number(number)) => Some(
                number
                    .as_i64()
                    .ok_or(PixverseError::Response("Invalid Pixverse status integer"))?,
            ),
            Some(Value::Null) | None => None,
            _ => return Err(PixverseError::Response("Invalid Pixverse status type")),
        };
        if status.is_some_and(|status| [6, 7, 8].contains(&status)) {
            return Err(PixverseError::Provider(format!("Task failed: {poll}")));
        }
        if status == Some(1) {
            let url = poll["Resp"]["url"]
                .as_str()
                .ok_or(PixverseError::Response("Missing Pixverse download URL"))?
                .to_owned();
            if endpoint(&url)? {
                return Err(PixverseError::Response(
                    "Source download requires absolute URL without node context",
                ));
            }
            let head = request_head(&url, HttpMethod::Get, credentials, true)?;
            let (downloaded_bytes, download_receipt, _) = exchange_with_policy(
                &head,
                &[],
                None,
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                transport,
            )?;
            if downloaded_bytes.is_empty() {
                return Err(PixverseError::Response("Empty Pixverse video download"));
            }
            transport.report_progress(
                handle,
                &Progress {
                    sequence: sequence as u64,
                    completed: 100,
                    total: 100,
                    message: Some("Completed".into()),
                },
            )?;
            return Ok(ResultValue {
                plan,
                creation_bytes,
                creation_receipt,
                poll_bytes,
                poll_receipt,
                downloaded_bytes,
                download_receipt,
                approval_receipt: approval.receipt,
                url: Some(url),
            });
        }
        if sequence == 479 {
            return Err(ConformanceError::PollLimit.into());
        }
        transport.report_progress(
            handle,
            &Progress {
                sequence: sequence as u64,
                completed: 0,
                total: 100,
                message: status.map(|value| value.to_string()),
            },
        )?;
        transport.check_cancelled(Some(handle))?;
        transport.poll_delay(handle, 5000)?;
    }
    Err(ConformanceError::PollLimit.into())
}
pub trait PixverseMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
fn expected_outputs(template: bool) -> Result<Vec<ExpectedOutput>, PixverseError> {
    let registry = TypeRegistry::built_in()
        .map_err(|_| PixverseError::Response("Missing canonical registry"))?;
    let name = if template {
        "PixverseIO.TEMPLATE"
    } else {
        "VIDEO"
    };
    let type_id = registry
        .resolve(name)
        .map_err(|_| PixverseError::Response("Missing canonical output type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| PixverseError::Response("Missing canonical output family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub struct MaterializationOwners<'a> {
    pub image: &'a mut dyn PixverseImageOwner,
    pub materializer: &'a mut dyn PixverseMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn PixverseTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), PixverseError> {
    let result = invoke(
        input,
        credentials,
        ceiling_microunits,
        transport,
        owners.image,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.materializer.materialize(&result)?;
    let batch = validate_outputs(
        &expected_outputs(input.node == "PixverseTemplateNode")?,
        outputs,
        MAXIMUM_MEDIA_BYTES,
    )
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
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn PixverseTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), PixverseError> {
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
    transport: &mut dyn PixverseTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), PixverseError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        let mut response_policy = ResponsePolicy::source();
        response_policy.timeout_milliseconds = policy.timeout_milliseconds;
        if policy.download {
            response_policy.monitor_progress = false;
            response_policy.final_label = None;
        } else if policy.timeout_milliseconds == Some(120000) {
            response_policy.wait_label = "Checking";
            response_policy.monitor_progress = false;
            response_policy.final_label = None;
        }
        transport.configure_response(handle, &response_policy)?;
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(PixverseError::Http {
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
                    return Err(PixverseError::Http {
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
    transport: &mut dyn PixverseTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), PixverseError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "pixverse.generation.maximum".into(),
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
                    return Err(PixverseError::Http {
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
}
impl PixverseUploadOwner for SourceProbeOwner {
    fn upload(&mut self, image: &Value) -> Result<UploadResult, PixverseError> {
        self.operations.push(json!({"operation":"png_prepare","image":image,"mime_type":"image/png","total_pixels":4194304,"first_batch":true}));
        if image["shape"]
            .as_array()
            .and_then(|shape| shape.first())
            .and_then(Value::as_u64)
            == Some(0)
        {
            return Err(ConformanceError::InvalidResponse.into());
        }
        Ok(UploadResult {
            image_id: Some(100 + self.operations.len() as i64),
            response_bytes: vec![],
            receipt: vec![],
        })
    }
}
fn check_case(case: &Value) -> Result<String, ConformanceError> {
    let input: Input = serde_json::from_value(case["input"].clone())
        .map_err(|_| ConformanceError::InvalidRequest)?;
    let mut owner = SourceProbeOwner::default();
    let plan = request(&input, &mut owner).map_err(|_| ConformanceError::InvalidRequest)?;
    if json!(owner.operations) != case["media"] {
        return Err(ConformanceError::InvalidRequest);
    }
    if plan.path.is_none() {
        if case["requests"] != json!([]) || case["outputs"] != json!([plan.body]) {
            return Err(ConformanceError::InvalidRequest);
        }
    } else {
        let expected = case["requests"]
            .as_array()
            .and_then(|requests| requests.last())
            .ok_or(ConformanceError::InvalidRequest)?;
        if plan.body != expected["body"]
            || json!(plan.path) != expected["path"]
            || json!(plan.estimated_seconds) != case["polls"][0]["estimated_duration"]
        {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    Ok(input.node)
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
    if source["source_sha256"]["comfy_api_nodes/nodes_pixverse.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        covered.insert(check_case(case)?);
    }
    for case in source["preflight_errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let mut owner = SourceProbeOwner::default();
        if request(&input, &mut owner).is_ok() || json!(owner.operations) != case["media"] {
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
        bad_boundary: bool,
        body_starts: Vec<u32>,
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
            if sequence == 0 {
                self.body_starts.push(handle.slot);
            }
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
    impl PixverseTransport for Transport {
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
        fn multipart_boundary(&mut self) -> Result<String, ConformanceError> {
            Ok(if self.bad_boundary {
                "bad\r\nheader"
            } else {
                "pixverse-source-test"
            }
            .into())
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
    fn prepared(input: &Input, override_response: &Value) -> Result<Transport, Box<dyn Error>> {
        let mut transport = Transport::default();
        let count = match input.node.as_str() {
            "PixverseImageToVideoNode" => 1,
            "PixverseTransitionVideoNode" => 2,
            _ => 0,
        };
        for index in 0..count {
            let mut upload = json!({"ErrCode":0,"ErrMsg":"source","Resp":{"img_id":101+index}});
            if let Some(fields) = override_response["upload"].as_object() {
                for (key, value) in fields {
                    upload[key] = value.clone();
                }
            }
            append(&mut transport, 200, upload);
        }
        let mut generation = json!({"ErrCode":0,"ErrMsg":"source","Resp":{"video_id":42}});
        if let Some(fields) = override_response["generation"].as_object() {
            for (key, value) in fields {
                generation[key] = value.clone();
            }
        }
        append(&mut transport, 200, generation);
        let mut poll = json!({"ErrCode":0,"ErrMsg":"source","Resp":{"status":1,"url":"https://media.fixture.invalid/video.mp4"}});
        if let Some(fields) = override_response["poll"].as_object() {
            for (key, value) in fields {
                poll[key] = value.clone();
            }
        }
        append(&mut transport, 200, poll);
        transport.events.extend(events(200, video(), vec![]));
        Ok(transport)
    }
    fn video() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/pixverse/downloaded-video.mp4"
        ))
        .to_vec()
    }
    fn png() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/pixverse/downloaded-image.png"
        ))
        .to_vec()
    }
    #[derive(Default)]
    struct ImageOwner {
        operations: Vec<Value>,
        fail: bool,
        cancelled: bool,
        bad_png: bool,
    }
    impl PixverseImageOwner for ImageOwner {
        fn prepare(
            &mut self,
            image: &Value,
            policy: &ImagePolicy,
        ) -> Result<PreparedImage, ConformanceError> {
            self.operations.push(json!({"operation":"png_prepare","image":image,"mime_type":policy.mime_type,"total_pixels":policy.total_pixels,"first_batch":policy.first_batch}));
            if self.cancelled {
                return Err(ConformanceError::Cancelled);
            }
            if self.fail {
                return Err(ConformanceError::HostFailure);
            }
            if image["shape"][0] == 0 {
                return Err(ConformanceError::InvalidResponse);
            }
            Ok(PreparedImage {
                bytes: if self.bad_png { vec![1] } else { png() },
                filename: "00000000-0000-4000-8000-000000000000.png".into(),
            })
        }
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
    struct Materializer {
        calls: Vec<(Value, Vec<u8>)>,
        empty: bool,
        cancelled: bool,
    }
    impl PixverseMaterializer for Materializer {
        fn materialize(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            self.calls
                .push((result.plan.body.clone(), result.downloaded_bytes.clone()));
            if self.cancelled {
                return Err(ConformanceError::Cancelled);
            }
            if self.empty {
                return Ok(vec![]);
            }
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn pixverse_all_pinned_source_requests_defaults_upload_order_and_failures_match()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["cases"].as_array().ok_or("cases")? {
            check_case(case)?;
        }
        for case in oracle()?["preflight_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            assert!(request(&input, &mut owner).is_err(), "{}", case["message"]);
            assert_eq!(json!(owner.operations), case["media"]);
        }
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/pixverse/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 4);
        assert_eq!(ROUTES.len(), 6);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            5
        );
        assert!(ALIASES.is_empty());
        Ok(())
    }
    #[test]
    fn pixverse_template_is_pure_and_uses_existing_model_identity() -> Result<(), Box<dyn Error>> {
        for (name, id) in oracle()?["templates"].as_object().ok_or("templates")? {
            let source = Input {
                node: "PixverseTemplateNode".into(),
                arguments: [("template".into(), json!(name))].into(),
            };
            let mut transport = Transport::default();
            let mut image = ImageOwner::default();
            let result = invoke(
                &source,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false,
                },
                0,
                &mut transport,
                &mut image,
            )?;
            assert_eq!(result.plan.body, *id);
            assert!(transport.heads.is_empty());
            assert!(transport.costs.is_empty());
            assert!(image.operations.is_empty());
            assert!(result.downloaded_bytes.is_empty());
        }
        let expected = expected_outputs(true)?;
        assert_eq!(
            expected.first().ok_or("type")?.type_id.to_string(),
            "sim:custom-pixverse-template@1"
        );
        assert_eq!(
            format!("{:?}", expected.first().ok_or("type")?.family),
            "Model"
        );
        assert_eq!(
            expected_outputs(false)?
                .first()
                .ok_or("video")?
                .type_id
                .to_string(),
            "sim:video@1"
        );
        Ok(())
    }
    #[test]
    fn pixverse_source_price_table_and_fallback_use_widgets_before_request_coercion()
    -> Result<(), Box<dyn Error>> {
        for quote in oracle()?["price_cases"].as_array().ok_or("quotes")? {
            let mut source = input("PixverseTextToVideoNode")?;
            for name in ["duration_seconds", "quality", "motion_mode"] {
                source.arguments.insert(name.into(), quote[name].clone());
            }
            assert_eq!(
                maximum_cost_microunits(&source)?,
                quote["maximum_microunits"].as_u64().ok_or("price")?
            );
        }
        let mut source = input("PixverseTextToVideoNode")?;
        source.arguments.insert("duration_seconds".into(), json!(8));
        source.arguments.insert("motion_mode".into(), json!("fast"));
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert_eq!(plan.body["motion_mode"], "normal");
        assert_eq!(maximum_cost_microunits(&source)?, 1_200_000);
        Ok(())
    }
    #[test]
    fn pixverse_real_png_multipart_then_submit_poll_download_keep_auth_receipts_and_order()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "PixverseTextToVideoNode",
            "PixverseImageToVideoNode",
            "PixverseTransitionVideoNode",
        ] {
            let source = input(node)?;
            let mut transport = prepared(&source, &json!({}))?;
            let mut image = ImageOwner::default();
            let result = invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut image,
            )?;
            assert_eq!(result.downloaded_bytes, video());
            assert_eq!(result.download_receipt, vec![7, 8, 9]);
            assert_eq!(result.creation_receipt, vec![7, 8, 9]);
            assert_eq!(result.poll_receipt, vec![7, 8, 9]);
            assert_eq!(result.approval_receipt, vec![42]);
            assert_eq!(transport.costs.len(), 1);
            assert_eq!(
                transport.costs.first().ok_or("cost")?.operation,
                "pixverse.generation.maximum"
            );
            assert_eq!(
                transport.costs.first().ok_or("cost")?.maximum_microunits,
                450_000
            );
            let generation = transport.bodies.first().ok_or("body")?;
            assert_eq!(
                serde_json::from_slice::<Value>(generation)?,
                result.plan.body
            );
            let upload_heads: Vec<_> = transport
                .heads
                .iter()
                .enumerate()
                .filter(|(_, head)| head.path == "/proxy/pixverse/image/upload")
                .collect();
            assert_eq!(upload_heads.len(), result.plan.uploads.len());
            for (index, head) in upload_heads {
                let body = transport.bodies.get(index).ok_or("upload body")?;
                assert!(body.windows(png().len()).any(|window| window == png()));
                assert!(
                    String::from_utf8_lossy(body)
                        .contains("Content-Type: application/octet-stream\r\n")
                );
                assert!(
                    String::from_utf8_lossy(body)
                        .contains("filename=\"00000000-0000-4000-8000-000000000000.png\"")
                );
                assert!(
                    head.headers
                        .iter()
                        .any(|header| header.name == "Content-Type"
                            && header.value
                                == "multipart/form-data; boundary=pixverse-source-test")
                );
            }
            assert_eq!(
                transport.body_starts.last().copied(),
                Some(transport.heads.len() as u32)
            );
            if !result.plan.uploads.is_empty() {
                assert_ne!(transport.body_starts.first().copied(), Some(1));
                assert!(
                    transport
                        .body_starts
                        .iter()
                        .position(|slot| *slot == 1)
                        .ok_or("generation body")?
                        > 0
                )
            }
            let download = transport.heads.last().ok_or("download")?;
            assert!(download.secret_id.is_none());
            assert_eq!(
                download.headers,
                vec![Header {
                    name: "Accept".into(),
                    value: "*/*".into()
                }]
            );
            assert!(
                transport
                    .heads
                    .iter()
                    .any(|head| head.path == "/proxy/pixverse/video/result/42")
            );
            assert!(
                transport
                    .poll_policies
                    .iter()
                    .all(|(_, policy)| policy.estimated_seconds
                        == if node == "PixverseImageToVideoNode" {
                            30
                        } else {
                            32
                        })
            );
            assert!(
                transport
                    .policies
                    .iter()
                    .any(|(_, policy)| policy.wait_label == "Checking"
                        && !policy.monitor_progress
                        && policy.final_label.is_none())
            );
        }
        Ok(())
    }
    #[test]
    fn pixverse_source_response_errors_preserve_upload_then_api_validation_boundary()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["response_errors"].as_array().ok_or("errors")? {
            let source: Input = serde_json::from_value(case["input"].clone())?;
            let mut transport = prepared(&source, &case["response_override"])?;
            let mut image = ImageOwner::default();
            assert!(
                invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut image
                )
                .is_err(),
                "{}",
                case["message"]
            );
            assert!(
                transport
                    .heads
                    .iter()
                    .all(|head| head.path.starts_with('/'))
            );
            if case["response_override"]["upload"]["Resp"]["img_id"].is_null()
                && case["response_override"]["upload"]["Resp"].is_object()
            {
                assert_eq!(
                    transport
                        .heads
                        .iter()
                        .filter(|head| head.path.ends_with("/upload"))
                        .count(),
                    if source.node == "PixverseTransitionVideoNode" {
                        2
                    } else {
                        1
                    }
                );
                assert!(transport.bodies.first().ok_or("generation")?.is_empty());
            }
            if case["response_override"]["generation"]["Resp"] == Value::Null
                && case["response_override"].get("generation").is_some()
            {
                assert!(
                    case["message"]
                        .as_str()
                        .ok_or("message")?
                        .contains("request failed")
                );
            }
        }
        Ok(())
    }
    #[test]
    fn pixverse_source_transition_ignores_negative_and_seed_is_forwarded()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("PixverseTransitionVideoNode")?;
        source
            .arguments
            .insert("negative_prompt".into(), json!("ignored source extra"));
        source.arguments.insert("seed".into(), json!(2147483647));
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert!(plan.body.get("negative_prompt").is_none());
        assert!(plan.body.get("template_id").is_none());
        assert!(plan.body.get("style").is_none());
        assert!(plan.body.get("water_mark").is_none());
        assert_eq!(plan.body["seed"], 2147483647);
        assert_eq!(plan.body["model"], "v3.5");
        assert_eq!(plan.body["first_frame_img"], 101);
        assert_eq!(plan.body["last_frame_img"], 102);
        Ok(())
    }
    #[test]
    fn pixverse_native_materialization_and_pure_model_encoding_never_publish_placeholders()
    -> Result<(), Box<dyn Error>> {
        for node in ["PixverseTextToVideoNode", "PixverseTemplateNode"] {
            for empty in [false, true] {
                let source = input(node)?;
                let mut transport = if node == "PixverseTemplateNode" {
                    Transport::default()
                } else {
                    prepared(&source, &json!({}))?
                };
                let mut image = ImageOwner::default();
                let mut materializer = Materializer {
                    empty,
                    ..Materializer::default()
                };
                let mut sink = Sink::default();
                assert!(
                    invoke_and_materialize(
                        &source,
                        credentials(),
                        2_000_000,
                        &mut transport,
                        &mut MaterializationOwners {
                            image: &mut image,
                            materializer: &mut materializer,
                            sink: &mut sink
                        }
                    )
                    .is_err()
                );
                assert_eq!(sink.published, 0);
                assert_eq!(materializer.calls.len(), 1);
                let bytes = &materializer.calls.first().ok_or("call")?.1;
                if node == "PixverseTemplateNode" {
                    assert!(bytes.is_empty())
                } else {
                    assert_eq!(*bytes, video());
                    assert_eq!(bytes.len(), 6163);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn pixverse_cost_credentials_and_native_failure_stop_generation_body()
    -> Result<(), Box<dyn Error>> {
        let source = input("PixverseImageToVideoNode")?;
        for mode in [
            "ceiling",
            "auth",
            "approval",
            "native",
            "cancelled",
            "png",
            "boundary",
        ] {
            let mut transport = Transport {
                deny_cost: mode == "approval",
                bad_boundary: mode == "boundary",
                ..Transport::default()
            };
            let mut image = ImageOwner {
                fail: mode == "native",
                cancelled: mode == "cancelled",
                bad_png: mode == "png",
                ..ImageOwner::default()
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
                    &mut image
                )
                .is_err()
            );
            assert!(transport.bodies.first().is_none_or(Vec::is_empty));
        }
        let source = input("PixverseTextToVideoNode")?;
        let mut transport = prepared(&source, &json!({}))?;
        invoke(
            &source,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            2_000_000,
            &mut transport,
            &mut ImageOwner::default(),
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
    fn pixverse_numeric_status_none_and_unknown_consume_active_budget_and_delegate_clock()
    -> Result<(), Box<dyn Error>> {
        let source = input("PixverseTextToVideoNode")?;
        for status in [json!(5), Value::Null, json!(99)] {
            let mut transport = Transport::default();
            append(&mut transport, 200, json!({"Resp":{"video_id":42}}));
            for _ in 0..480 {
                append(&mut transport, 200, json!({"Resp":{"status":status}}));
            }
            assert!(matches!(
                invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut ImageOwner::default()
                ),
                Err(PixverseError::Transport(ConformanceError::PollLimit))
            ));
            assert_eq!(transport.delays, vec![5000; 479]);
        }
        let mut transport = Transport::default();
        append(&mut transport, 200, json!({"Resp":{"video_id":42}}));
        append(&mut transport, 200, json!({"Resp":null}));
        append(&mut transport, 200, json!({"Resp":{"status":5}}));
        append(
            &mut transport,
            200,
            json!({"Resp":{"status":1,"url":"https://media.fixture.invalid/video.mp4"}}),
        );
        transport.events.extend(events(200, video(), vec![]));
        invoke(
            &source,
            credentials(),
            2_000_000,
            &mut transport,
            &mut ImageOwner::default(),
        )?;
        assert_eq!(transport.delays, vec![5000; 2]);
        Ok(())
    }
    #[test]
    fn pixverse_numeric_failed_states_terminate_without_download() -> Result<(), Box<dyn Error>> {
        for status in [6, 7, 8] {
            let source = input("PixverseTextToVideoNode")?;
            let mut transport = prepared(
                &source,
                &json!({"poll":{"Resp":{"status":status,"url":"https://media.fixture.invalid/video.mp4"}}}),
            )?;
            assert!(
                invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut ImageOwner::default()
                )
                .err()
                .ok_or("error")?
                .to_string()
                .starts_with("Task failed:")
            );
            assert!(
                transport
                    .heads
                    .iter()
                    .all(|head| head.path.starts_with('/'))
            );
        }
        Ok(())
    }
    #[test]
    fn pixverse_source_retry_budgets_backoff_and_stale_handles_are_exact()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/pixverse/video/text/generate",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        for (policy, status, count) in [
            (GENERATION_RETRY, 503, 4),
            (POLL_RETRY, 503, 11),
            (GENERATION_RETRY, 429, 17),
            (DOWNLOAD_RETRY, 429, 6),
        ] {
            let mut transport = Transport::default();
            for _ in 0..count {
                transport
                    .events
                    .extend(events(status, b"bad".to_vec(), vec![]));
            }
            assert!(matches!(
                exchange_with_policy(&head, b"{}", None, 1024, policy, &mut transport),
                Err(PixverseError::Http { .. })
            ));
            assert_eq!(transport.heads.len(), count);
            assert!(transport.bodies.iter().all(|body| body == b"{}"));
            assert_eq!(transport.delays.len(), count - 1);
        }
        let mut transport = Transport {
            same_retry_handle: true,
            events: events(503, b"bad".to_vec(), vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, b"{}", None, 1024, GENERATION_RETRY, &mut transport),
            Err(PixverseError::Transport(ConformanceError::InvalidHandle))
        ));
        let mut transport = Transport::default();
        transport.events.extend(events(
            503,
            b"bad".to_vec(),
            vec![Header {
                name: "Retry-After".into(),
                value: "200".into(),
            }],
        ));
        transport.events.extend(events(200, b"{}".to_vec(), vec![]));
        exchange_with_policy(&head, b"{}", None, 1024, GENERATION_RETRY, &mut transport)?;
        assert_eq!(transport.delays, vec![150000]);
        Ok(())
    }
    #[test]
    fn pixverse_worker_loss_cancellation_and_fresh_invocation_do_not_replay_generation()
    -> Result<(), Box<dyn Error>> {
        for _ in 0..2 {
            let source = input("PixverseTextToVideoNode")?;
            let mut transport = Transport::default();
            assert!(matches!(
                invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut ImageOwner::default()
                ),
                Err(PixverseError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(transport.heads.len(), 1);
            assert!(transport.delays.is_empty());
        }
        let source = input("PixverseTextToVideoNode")?;
        let mut transport = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut ImageOwner::default()
            ),
            Err(PixverseError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        let mut transport = Transport {
            events: vec![Ok(ResponseEvent::Cancelled)].into(),
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut ImageOwner::default()
            ),
            Err(PixverseError::Transport(ConformanceError::Cancelled))
        ));
        Ok(())
    }
    #[test]
    fn pixverse_malformed_response_headers_chunks_receipts_and_aggregate_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/pixverse/video/text/generate",
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
            Err(PixverseError::Transport(ConformanceError::BodyLimit))
        ));
        for bytes in [b"bad".as_slice(), b"[]".as_slice()] {
            assert!(json_response(bytes).is_err())
        }
        Ok(())
    }

    #[test]
    fn pixverse_falsy_negative_prompts_match_none_without_changing_request_bytes()
    -> Result<(), Box<dyn Error>> {
        for node in ["PixverseTextToVideoNode", "PixverseImageToVideoNode"] {
            let mut source = input(node)?;
            source
                .arguments
                .insert("negative_prompt".into(), Value::Null);
            let mut baseline_owner = SourceProbeOwner::default();
            let baseline = request(&source, &mut baseline_owner)?;
            let expected = serde_json::to_vec(&baseline.body)?;
            for negative in [
                Value::Null,
                json!(""),
                json!(false),
                json!(0),
                json!(0.0),
                json!(-0.0),
                json!([]),
                json!({}),
            ] {
                source.arguments.insert("negative_prompt".into(), negative);
                let mut owner = SourceProbeOwner::default();
                let actual = request(&source, &mut owner)?;
                assert_eq!(serde_json::to_vec(&actual.body)?, expected);
                assert_eq!(owner.operations, baseline_owner.operations);
                assert_eq!(actual.path, baseline.path);
                assert_eq!(actual.estimated_seconds, baseline.estimated_seconds);
                assert!(actual.body.get("negative_prompt").is_none());
            }
        }
        let mut source = input("PixverseTransitionVideoNode")?;
        source
            .arguments
            .insert("negative_prompt".into(), Value::Null);
        let baseline = request(&source, &mut SourceProbeOwner::default())?;
        for negative in [
            json!(false),
            json!(0),
            json!(0.0),
            json!(-0.0),
            json!([]),
            json!({}),
            json!(["ignored"]),
            json!({"ignored":1}),
        ] {
            source.arguments.insert("negative_prompt".into(), negative);
            let actual = request(&source, &mut SourceProbeOwner::default())?;
            assert_eq!(
                serde_json::to_vec(&actual.body)?,
                serde_json::to_vec(&baseline.body)?
            );
        }
        Ok(())
    }

    #[test]
    fn pixverse_truthy_negative_fields_preserve_raw_strings_and_source_upload_order()
    -> Result<(), Box<dyn Error>> {
        for node in ["PixverseTextToVideoNode", "PixverseImageToVideoNode"] {
            for negative in [" raw negative prompt ", " \u{1c} ", "0", "false"] {
                let mut source = input(node)?;
                source
                    .arguments
                    .insert("negative_prompt".into(), json!(negative));
                let actual = request(&source, &mut SourceProbeOwner::default())?;
                assert_eq!(actual.body["negative_prompt"], negative);
            }
            for negative in [
                json!(true),
                json!(1),
                json!(1.5),
                json!(["bad"]),
                json!({"bad":1}),
            ] {
                let mut source = input(node)?;
                source.arguments.insert("negative_prompt".into(), negative);
                let mut owner = SourceProbeOwner::default();
                assert!(matches!(
                    request(&source, &mut owner),
                    Err(PixverseError::Input("Expected Pixverse string"))
                ));
                assert_eq!(
                    owner.operations.len(),
                    usize::from(node == "PixverseImageToVideoNode")
                );
                source.arguments.insert("prompt".into(), Value::Null);
                let mut owner = SourceProbeOwner::default();
                assert!(matches!(
                    request(&source, &mut owner),
                    Err(PixverseError::Input("Field 'prompt' cannot be empty."))
                ));
                assert!(owner.operations.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn pixverse_falsy_negative_invoke_keeps_authority_cancellation_and_upload_failure_boundaries()
    -> Result<(), Box<dyn Error>> {
        for node in ["PixverseTextToVideoNode", "PixverseImageToVideoNode"] {
            for negative in [
                json!(false),
                json!(0),
                json!(0.0),
                json!(-0.0),
                json!([]),
                json!({}),
            ] {
                let mut source = input(node)?;
                source.arguments.insert("negative_prompt".into(), negative);
                let mut transport = prepared(&source, &json!({}))?;
                let mut image = ImageOwner::default();
                let actual = invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut image,
                )?;
                assert!(actual.plan.body.get("negative_prompt").is_none());
                assert_eq!(actual.downloaded_bytes, video());
                assert_eq!(actual.creation_receipt, vec![7, 8, 9]);
                assert_eq!(actual.download_receipt, vec![7, 8, 9]);
                assert_eq!(transport.costs.len(), 1);
                assert_eq!(
                    transport.costs.first().ok_or("approval")?.operation,
                    "pixverse.generation.maximum"
                );
                assert_eq!(
                    serde_json::from_slice::<Value>(
                        transport.bodies.first().ok_or("generation body")?
                    )?,
                    actual.plan.body
                );
                assert_eq!(
                    image.operations.len(),
                    usize::from(node == "PixverseImageToVideoNode")
                );
                let mut cancelled = Transport {
                    cancelled: true,
                    ..Default::default()
                };
                assert!(matches!(
                    invoke(
                        &source,
                        credentials(),
                        2_000_000,
                        &mut cancelled,
                        &mut ImageOwner::default()
                    ),
                    Err(PixverseError::Transport(ConformanceError::Cancelled))
                ));
                assert!(cancelled.heads.is_empty());
                assert!(cancelled.bodies.is_empty());
                assert!(cancelled.costs.is_empty());
            }
        }
        let mut source = input("PixverseImageToVideoNode")?;
        source
            .arguments
            .insert("negative_prompt".into(), json!(["bad"]));
        let mut transport = prepared(&source, &json!({}))?;
        let mut image = ImageOwner {
            fail: true,
            ..Default::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut image
            ),
            Err(PixverseError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(image.operations.len(), 1);
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert!(transport.bodies.first().ok_or("armed body")?.is_empty());
        assert!(transport.body_starts.is_empty());
        Ok(())
    }
}
