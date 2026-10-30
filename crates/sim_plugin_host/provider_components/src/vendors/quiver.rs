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
pub const NAMESPACE: &str = "zed.sim.provider.quiver";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "5ab37aba355046eb66dc2f36c5edec4ff5e42fec17cbb0195fd90f4ee3115e0d";
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
        feature_id: "COMFY-NODE-0501",
        node_identifier: "QuiverImageToSVGNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0502",
        node_identifier: "QuiverTextToSVGNode",
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
        feature_id: "COMFY-API-EXT-0134",
        method: "POST",
        path: "/proxy/quiver/v1/svgs/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0135",
        method: "POST",
        path: "/proxy/quiver/v1/svgs/vectorizations",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum QuiverError {
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
impl fmt::Display for QuiverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Quiver HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for QuiverError {}
impl From<ConformanceError> for QuiverError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for QuiverError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, QuiverError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/quiver/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, QuiverError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(QuiverError::Input("Unknown Quiver node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(QuiverError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(QuiverError::Input("Unknown Quiver source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(QuiverError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(QuiverError::Input("Missing Quiver source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, QuiverError> {
    input
        .arguments
        .get(name)
        .ok_or(QuiverError::Input("Missing Quiver argument"))
}
fn text(value: &Value) -> Result<&str, QuiverError> {
    value
        .as_str()
        .ok_or(QuiverError::Input("Expected Quiver string"))
}
fn endpoint(value: &str) -> Result<bool, QuiverError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(QuiverError::Response("Invalid Quiver response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(QuiverError::Response("Invalid Quiver response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(QuiverError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(QuiverError::Response("Invalid Quiver endpoint authority"));
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
) -> Result<RequestHead, QuiverError> {
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
            return Err(QuiverError::Input("Missing Comfy API credentials"));
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
    pub maximum_images: u64,
    pub total_pixels: u64,
    pub wait_label: &'static str,
    pub show_batch_index: bool,
}
impl ImagePolicy {
    fn source() -> Self {
        Self {
            mime_type: "image/png",
            maximum_images: 1,
            total_pixels: 4194304,
            wait_label: "Uploading",
            show_batch_index: false,
        }
    }
}
#[derive(Clone, Debug)]
pub struct UploadResult {
    pub url: String,
    pub receipt: Vec<u8>,
}
pub trait QuiverImageOwner {
    fn upload(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<UploadResult, ConformanceError>;
    fn ordered_references(&mut self, value: &Value) -> Result<Vec<Value>, ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: &'static str,
    pub body: Value,
    pub upload_receipts: Vec<Vec<u8>>,
}
fn preflight(input: &Input) -> Result<(), QuiverError> {
    if input.node == "QuiverTextToSVGNode" {
        let prompt = argument(input, "prompt")?;
        if prompt.is_null() {
            return Err(QuiverError::Input("Field 'prompt' cannot be empty."));
        }
        if text(prompt)?.is_empty() {
            return Err(QuiverError::Input(
                "Source prompt must contain at least one character",
            ));
        }
    }
    Ok(())
}
fn upload(
    image: &Value,
    owner: &mut dyn QuiverImageOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<String, QuiverError> {
    let uploaded = owner.upload(image, &ImagePolicy::source())?;
    if endpoint(&uploaded.url)?
        || uploaded.receipt.is_empty()
        || uploaded.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(QuiverError::Response(
            "Invalid canonical image upload result",
        ));
    }
    receipts.push(uploaded.receipt);
    Ok(uploaded.url)
}
fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}
fn source_float(value: &Value, minimum: f64, maximum: f64) -> Result<Value, QuiverError> {
    let number = if let Some(number) = value.as_f64() {
        number
    } else if let Some(value) = value.as_str() {
        value
            .parse()
            .map_err(|_| QuiverError::Input("Invalid source float"))?
    } else {
        return Err(QuiverError::Input("Invalid source float"));
    };
    if !number.is_finite() || number < minimum || number > maximum {
        return Err(QuiverError::Input("Source numeric option out of range"));
    }
    Ok(json!(number))
}
pub fn request(input: &Input, owner: &mut dyn QuiverImageOwner) -> Result<Request, QuiverError> {
    let input = normalized(input)?;
    preflight(&input)?;
    let mut body = json!({});
    let mut upload_receipts = vec![];
    if input.node == "QuiverTextToSVGNode" {
        let references_value = argument(&input, "reference_images")?;
        if !references_value.is_null()
            && !references_value
                .as_object()
                .is_some_and(|references| references.is_empty())
        {
            let images = owner.ordered_references(references_value)?;
            if images.len() > 128 {
                return Err(ConformanceError::BodyLimit.into());
            }
            let mut references = vec![];
            for image in images {
                references.push(json!({"url":upload(&image,owner,&mut upload_receipts)?}));
            }
            if references.len() > 4 {
                return Err(QuiverError::Input(
                    "Maximum 4 reference images are allowed.",
                ));
            }
            if !references.is_empty() {
                body["references"] = json!(references);
            }
        }
        let instructions = argument(&input, "instructions")?;
        if !instructions.is_null() {
            let value = text(instructions)?.trim_matches(python_whitespace);
            if !value.is_empty() {
                body["instructions"] = json!(value);
            }
        }
        body["prompt"] = json!(text(argument(&input, "prompt")?)?);
    } else {
        body["image"] =
            json!({"url":upload(argument(&input,"image")?,owner,&mut upload_receipts)?});
        let crop = argument(&input, "auto_crop")?
            .as_bool()
            .ok_or(QuiverError::Input("Invalid source auto_crop"))?;
        if crop {
            body["auto_crop"] = json!(true);
        }
    }
    let model = argument(&input, "model")?
        .as_object()
        .ok_or(QuiverError::Input("Invalid source model dictionary"))?;
    body["model"] = json!(text(
        model
            .get("model")
            .ok_or(QuiverError::Input("Missing source model key"))?
    )?);
    for (field, minimum, maximum) in [
        ("temperature", 0.0, 2.0),
        ("top_p", 0.0, 1.0),
        ("presence_penalty", -2.0, 2.0),
    ] {
        if let Some(value) = model.get(field).filter(|value| !value.is_null()) {
            body[field] = source_float(value, minimum, maximum)?;
        }
    }
    if input.node == "QuiverImageToSVGNode"
        && let Some(value) = model.get("target_size").filter(|value| !value.is_null())
    {
        let number = value
            .as_i64()
            .ok_or(QuiverError::Input("Invalid source target size"))?;
        if !(128..=4096).contains(&number) {
            return Err(QuiverError::Input("Source target size out of range"));
        }
        body["target_size"] = json!(number);
    }
    Ok(Request {
        path: if input.node == "QuiverTextToSVGNode" {
            "/proxy/quiver/v1/svgs/generations"
        } else {
            "/proxy/quiver/v1/svgs/vectorizations"
        },
        body,
        upload_receipts,
    })
}
pub fn maximum_cost_microunits(input: &Input) -> Result<u64, QuiverError> {
    let input = normalized(input)?;
    let name = text(&argument(&input, "model")?["model"])?;
    let rates = oracle()?["price_rates"].clone();
    let price = rates[if name.contains("max") {
        "max"
    } else if name.contains("preview") {
        "preview"
    } else {
        "base"
    }]
    .as_f64()
    .ok_or(QuiverError::Input("Missing source price"))?;
    let scaled = (price * 1_000_000.0).round();
    if !scaled.is_finite() || !(0.0..=1_000_000_000.0).contains(&scaled) {
        return Err(QuiverError::Input("Invalid source price"));
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
pub trait QuiverTransport: ProviderTransport {
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
pub fn svg_bytes(response_bytes: &[u8]) -> Result<Vec<Vec<u8>>, QuiverError> {
    let response: Value = serde_json::from_slice(response_bytes)?;
    if !response.is_object() {
        return Err(QuiverError::Response("Invalid Quiver response object"));
    }
    let data = response["data"]
        .as_array()
        .ok_or(QuiverError::Response("Missing Quiver SVG data array"))?;
    if data.len() > 256 {
        return Err(ConformanceError::BodyLimit.into());
    }
    if response
        .get("id")
        .is_some_and(|value| !value.is_null() && !value.is_string())
    {
        return Err(QuiverError::Response("Invalid Quiver metadata"));
    }
    if response
        .get("created")
        .is_some_and(|value| !value.is_null() && !value.is_i64())
    {
        return Err(QuiverError::Response("Invalid Quiver created metadata"));
    }
    if let Some(usage) = response.get("usage").filter(|value| !value.is_null()) {
        if !usage.is_object() {
            return Err(QuiverError::Response("Invalid Quiver usage"));
        }
        for field in ["total_tokens", "input_tokens", "output_tokens"] {
            if usage
                .get(field)
                .is_some_and(|value| !value.is_null() && !value.is_i64())
            {
                return Err(QuiverError::Response("Invalid Quiver usage integer"));
            }
        }
    }
    let mut values = vec![];
    let mut total = 0usize;
    for item in data {
        let value = item["svg"]
            .as_str()
            .ok_or(QuiverError::Response("Missing source SVG string"))?;
        if item
            .get("mime_type")
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err(QuiverError::Response("Invalid source MIME metadata"));
        }
        total = total
            .checked_add(value.len())
            .ok_or(ConformanceError::BodyLimit)?;
        if total > MAXIMUM_MEDIA_BYTES {
            return Err(ConformanceError::BodyLimit.into());
        }
        values.push(value.as_bytes().to_vec());
    }
    Ok(values)
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub response_bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub svg_buffers: Vec<Vec<u8>>,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn QuiverTransport,
    image: &mut dyn QuiverImageOwner,
) -> Result<ResultValue, QuiverError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    preflight(&input)?;
    let cost = maximum_cost_microunits(&input)?;
    if cost > ceiling_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    let path = if input.node == "QuiverTextToSVGNode" {
        "/proxy/quiver/v1/svgs/generations"
    } else {
        "/proxy/quiver/v1/svgs/vectorizations"
    };
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "quiver.generation.maximum".into(),
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
    let plan = request(&input, image)?;
    let body = serde_json::to_vec(&plan.body)?;
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let (response_bytes, receipt, _) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    let svg_buffers = svg_bytes(&response_bytes)?;
    Ok(ResultValue {
        plan,
        response_bytes,
        receipt,
        approval_receipt: approval.receipt,
        svg_buffers,
    })
}
pub trait QuiverSvgMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
fn expected_outputs() -> Result<Vec<ExpectedOutput>, QuiverError> {
    let registry = TypeRegistry::built_in()
        .map_err(|_| QuiverError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve("SVG")
        .map_err(|_| QuiverError::Response("Missing canonical SVG type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| QuiverError::Response("Missing canonical SVG family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub struct MaterializationOwners<'a> {
    pub image: &'a mut dyn QuiverImageOwner,
    pub svg: &'a mut dyn QuiverSvgMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn QuiverTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), QuiverError> {
    let result = invoke(
        input,
        credentials,
        ceiling_microunits,
        transport,
        owners.image,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.svg.materialize(&result)?;
    let batch = validate_outputs(&expected_outputs()?, outputs, MAXIMUM_MEDIA_BYTES)
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
#[cfg(test)]
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn QuiverTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), QuiverError> {
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
    transport: &mut dyn QuiverTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), QuiverError> {
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
            Err(QuiverError::Http {
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
                    return Err(QuiverError::Http {
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
    transport: &mut dyn QuiverTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), QuiverError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "quiver.generation.maximum".into(),
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
                    return Err(QuiverError::Http {
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
impl QuiverImageOwner for SourceProbeOwner {
    fn upload(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<UploadResult, ConformanceError> {
        if self.cancelled {
            return Err(ConformanceError::Cancelled);
        }
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        self.operations.push(json!({"operation":"upload_image","image":image,"mime_type":policy.mime_type,"max_images":policy.maximum_images,"wait_label":policy.wait_label,"show_batch_index":policy.show_batch_index,"total_pixels":policy.total_pixels}));
        if image["shape"]
            .as_array()
            .and_then(|shape| shape.first())
            .and_then(Value::as_u64)
            == Some(0)
        {
            return Err(ConformanceError::InvalidResponse);
        }
        let slot = image["native_slot"]
            .as_str()
            .ok_or(ConformanceError::InvalidRequest)?;
        Ok(UploadResult {
            url: format!("https://storage.fixture.invalid/{slot}.png"),
            receipt: vec![4, 5, 6],
        })
    }
    fn ordered_references(&mut self, value: &Value) -> Result<Vec<Value>, ConformanceError> {
        value["ordered_entries"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?
            .iter()
            .map(|entry| {
                entry
                    .get("image")
                    .cloned()
                    .ok_or(ConformanceError::InvalidRequest)
            })
            .collect()
    }
}
fn check_case(case: &Value) -> Result<String, ConformanceError> {
    let input: Input = serde_json::from_value(case["input"].clone())
        .map_err(|_| ConformanceError::InvalidRequest)?;
    let mut owner = SourceProbeOwner::default();
    let plan = request(&input, &mut owner).map_err(|_| ConformanceError::InvalidRequest)?;
    let expected = case["requests"]
        .as_array()
        .and_then(|values| values.first())
        .ok_or(ConformanceError::InvalidRequest)?;
    if plan.body != expected["body"]
        || json!(plan.path) != expected["path"]
        || json!(owner.operations) != case["media"]
    {
        return Err(ConformanceError::InvalidRequest);
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
    if source["source_sha256"]["comfy_api_nodes/nodes_quiver.py"] != SOURCE_SHA256 {
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
    impl QuiverTransport for Transport {
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
    fn default_response() -> Result<Value, Box<dyn Error>> {
        Ok(serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/quiver/svg-response.json"
        )))?)
    }
    fn prepared(response: Value) -> Transport {
        let mut transport = Transport::default();
        append(&mut transport, 200, response);
        transport
    }
    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|value| format!("{value:02x}")).collect()
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
    struct SvgOwner {
        buffers: Vec<Vec<Vec<u8>>>,
        empty: bool,
        cancelled: bool,
    }
    impl QuiverSvgMaterializer for SvgOwner {
        fn materialize(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            self.buffers.push(result.svg_buffers.clone());
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
    fn quiver_actual_source_requests_and_upload_before_validation_match()
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
            "/../../sim_test_support/fixtures/providers/quiver/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 2);
        assert_eq!(ROUTES.len(), 2);
        assert!(
            ROUTES
                .iter()
                .all(|route| route.method == "POST" && route.disposition == "executable")
        );
        assert!(ALIASES.is_empty());
        Ok(())
    }
    #[test]
    fn quiver_literal_model_price_substrings_preserve_max_precedence() -> Result<(), Box<dyn Error>>
    {
        for quote in oracle()?["price_cases"].as_array().ok_or("prices")? {
            for node in ["QuiverTextToSVGNode", "QuiverImageToSVGNode"] {
                let mut source = input(node)?;
                source
                    .arguments
                    .insert("model".into(), json!({"model":quote["model"]}));
                assert_eq!(
                    maximum_cost_microunits(&source)?,
                    quote["maximum_microunits"].as_u64().ok_or("price")?
                );
            }
        }
        Ok(())
    }
    #[test]
    fn quiver_source_seed_omission_crop_omission_and_whitespace_are_exact()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("QuiverTextToSVGNode")?;
        source.arguments.insert(
            "instructions".into(),
            json!("\u{1c}\u{1f}\u{85}\u{a0}source\u{1c}\u{1f}"),
        );
        source.arguments.insert("seed".into(), json!(2147483647));
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert_eq!(plan.body["instructions"], "source");
        assert_eq!(plan.body["prompt"], "  source prompt  ");
        assert!(plan.body.get("seed").is_none());
        assert!(plan.body.get("references").is_none());
        let source = input("QuiverImageToSVGNode")?;
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert!(plan.body.get("auto_crop").is_none());
        assert!(plan.body.get("target_size").is_none());
        assert!(plan.body.get("seed").is_none());
        Ok(())
    }
    #[test]
    fn quiver_reference_order_is_source_insertion_order_and_limit_follows_all_uploads()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("QuiverTextToSVGNode")?;
        let entries:Vec<Value>=(0..5).map(|index|json!({"key":format!("ref_{}",9-index),"image":{"native_slot":format!("ref-{index}"),"shape":[1,720,720,3]}})).collect();
        source.arguments.insert(
            "reference_images".into(),
            json!({"ordered_entries":entries}),
        );
        let mut owner = SourceProbeOwner::default();
        assert_eq!(
            request(&source, &mut owner)
                .err()
                .ok_or("limit")?
                .to_string(),
            "Maximum 4 reference images are allowed."
        );
        assert_eq!(owner.operations.len(), 5);
        let entries: Vec<Value> = entries.into_iter().take(4).collect();
        source.arguments.insert(
            "reference_images".into(),
            json!({"ordered_entries":entries}),
        );
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert_eq!(plan.body["references"],json!((0..4).map(|index|json!({"url":format!("https://storage.fixture.invalid/ref-{index}.png")})).collect::<Vec<_>>()));
        assert_eq!(plan.upload_receipts, vec![vec![4, 5, 6]; 4]);
        Ok(())
    }
    #[test]
    fn quiver_ordered_utf8_svg_buffers_match_actual_source_wrapper_including_empty_and_nonxml()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["cases"].as_array().ok_or("cases")? {
            let response = case.get("response").cloned().unwrap_or(default_response()?);
            let parsed = svg_bytes(&serde_json::to_vec(&response)?)?;
            assert_eq!(
                json!(parsed.iter().map(|bytes| hex(bytes)).collect::<Vec<_>>()),
                case["svg_utf8_hex"]
            );
        }
        assert!(svg_bytes(&serde_json::to_vec(&json!({"data":[]}))?)?.is_empty());
        assert_eq!(
            svg_bytes(&serde_json::to_vec(
                &json!({"data":[{"svg":"not XML","mime_type":"ignored"},{"svg":""}]})
            )?)?,
            vec![b"not XML".to_vec(), vec![]]
        );
        Ok(())
    }
    #[test]
    fn quiver_source_response_failures_are_rejected_before_materialization()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["response_errors"]
            .as_array()
            .ok_or("response errors")?
        {
            assert!(
                svg_bytes(&serde_json::to_vec(&case["response"])?).is_err(),
                "{}",
                case["message"]
            );
            let source: Input = serde_json::from_value(case["input"].clone())?;
            let mut transport = prepared(case["response"].clone());
            assert!(
                invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                )
                .is_err()
            );
            assert_eq!(transport.heads.len(), 1);
            assert!(transport.delays.is_empty());
        }
        Ok(())
    }
    #[test]
    fn quiver_actual_json_transport_retains_receipts_auth_cost_and_has_no_poll_or_download()
    -> Result<(), Box<dyn Error>> {
        for node in ["QuiverTextToSVGNode", "QuiverImageToSVGNode"] {
            let source = input(node)?;
            let response = default_response()?;
            let mut transport = prepared(response.clone());
            let mut owner = SourceProbeOwner::default();
            let result = invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut owner,
            )?;
            assert_eq!(result.response_bytes, serde_json::to_vec(&response)?);
            assert_eq!(result.receipt, vec![7, 8, 9]);
            assert_eq!(result.approval_receipt, vec![42]);
            assert_eq!(transport.heads.len(), 1);
            assert_eq!(
                serde_json::from_slice::<Value>(transport.bodies.first().ok_or("body")?)?,
                result.plan.body
            );
            assert_eq!(
                transport.costs.first().ok_or("cost")?.operation,
                "quiver.generation.maximum"
            );
            assert_eq!(
                transport.costs.first().ok_or("cost")?.maximum_microunits,
                286000
            );
            let head = transport.heads.first().ok_or("head")?;
            assert_eq!(head.path, result.plan.path);
            assert!(
                head.headers
                    .iter()
                    .any(|header| header.name == "Authorization")
            );
            assert!(
                transport
                    .policies
                    .iter()
                    .all(|(_, policy)| policy.timeout_milliseconds == Some(3600000)
                        && policy.monitor_progress
                        && policy.wait_label == "Waiting for server")
            );
            assert!(transport.delays.is_empty());
        }
        Ok(())
    }
    #[test]
    fn quiver_canonical_svg_artifact_owner_rejects_without_fake_publication()
    -> Result<(), Box<dyn Error>> {
        let expected = expected_outputs()?;
        assert_eq!(
            expected.first().ok_or("type")?.type_id.to_string(),
            "sim:svg@1"
        );
        assert_eq!(
            format!("{:?}", expected.first().ok_or("family")?.family),
            "Artifact"
        );
        for response in [
            default_response()?,
            json!({"data":[]}),
            json!({"data":[{"svg":""}]}),
        ] {
            for mode in ["failure", "empty", "cancelled"] {
                let mut transport = prepared(response.clone());
                let mut image = SourceProbeOwner::default();
                let mut svg = SvgOwner {
                    empty: mode == "empty",
                    cancelled: mode == "cancelled",
                    ..SvgOwner::default()
                };
                let mut sink = Sink::default();
                assert!(
                    invoke_and_materialize(
                        &input("QuiverTextToSVGNode")?,
                        credentials(),
                        2_000_000,
                        &mut transport,
                        &mut MaterializationOwners {
                            image: &mut image,
                            svg: &mut svg,
                            sink: &mut sink
                        }
                    )
                    .is_err()
                );
                assert_eq!(sink.published, 0);
                assert_eq!(
                    svg.buffers,
                    vec![svg_bytes(&serde_json::to_vec(&response)?)?]
                );
            }
        }
        Ok(())
    }
    #[test]
    fn quiver_auth_cost_and_canonical_upload_failure_prevent_body() -> Result<(), Box<dyn Error>> {
        let source = input("QuiverImageToSVGNode")?;
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
        let mut transport = prepared(default_response()?);
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
    fn quiver_retries_preserve_body_and_supervisor_clock_ownership() -> Result<(), Box<dyn Error>> {
        let source = input("QuiverTextToSVGNode")?;
        for (status, count) in [(503, 4), (429, 17)] {
            let mut transport = Transport::default();
            for _ in 0..count {
                transport
                    .events
                    .extend(events(status, b"failure".to_vec(), vec![]));
            }
            assert!(matches!(
                invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                ),
                Err(QuiverError::Http { .. })
            ));
            assert_eq!(transport.heads.len(), count);
            assert_eq!(transport.delays.len(), count - 1);
            let body = transport.bodies.first().ok_or("body")?;
            assert!(transport.bodies.iter().all(|candidate| candidate == body));
        }
        let mut transport = Transport {
            same_retry_handle: true,
            events: events(503, b"bad".to_vec(), vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut SourceProbeOwner::default()
            ),
            Err(QuiverError::Transport(ConformanceError::InvalidHandle))
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
        append(&mut transport, 200, default_response()?);
        invoke(
            &source,
            credentials(),
            2_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(transport.delays, vec![150000]);
        Ok(())
    }
    #[test]
    fn quiver_cancel_worker_loss_and_clean_invocation_never_publish_or_replay()
    -> Result<(), Box<dyn Error>> {
        let source = input("QuiverTextToSVGNode")?;
        for _ in 0..2 {
            let mut transport = Transport::default();
            assert!(matches!(
                invoke(
                    &source,
                    credentials(),
                    2_000_000,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                ),
                Err(QuiverError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(transport.heads.len(), 1);
            assert!(transport.delays.is_empty());
        }
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
                &mut SourceProbeOwner::default()
            ),
            Err(QuiverError::Transport(ConformanceError::Cancelled))
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
                &mut SourceProbeOwner::default()
            ),
            Err(QuiverError::Transport(ConformanceError::Cancelled))
        ));
        Ok(())
    }
    #[test]
    fn quiver_malformed_response_headers_chunks_receipts_and_aggregate_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/quiver/v1/svgs/generations",
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
            Err(QuiverError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[derive(Default)]
    struct UnavailableReferenceOwner {
        ordered_calls: usize,
        upload_calls: usize,
    }
    impl QuiverImageOwner for UnavailableReferenceOwner {
        fn ordered_references(&mut self, _value: &Value) -> Result<Vec<Value>, ConformanceError> {
            self.ordered_calls += 1;
            Err(ConformanceError::HostFailure)
        }
        fn upload(
            &mut self,
            _image: &Value,
            _policy: &ImagePolicy,
        ) -> Result<UploadResult, ConformanceError> {
            self.upload_calls += 1;
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn quiver_empty_reference_mapping_matches_none_without_image_owner()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("QuiverTextToSVGNode")?;
        source
            .arguments
            .insert("reference_images".into(), Value::Null);
        let mut owner = UnavailableReferenceOwner::default();
        let none = request(&source, &mut owner)?;
        source
            .arguments
            .insert("reference_images".into(), json!({}));
        let empty = request(&source, &mut owner)?;
        assert_eq!(empty.body, none.body);
        assert!(empty.body.get("references").is_none());
        assert!(empty.upload_receipts.is_empty());
        assert_eq!(owner.ordered_calls, 0);
        assert_eq!(owner.upload_calls, 0);
        Ok(())
    }
    #[test]
    fn quiver_empty_reference_mapping_reaches_checked_request_without_image_preparation()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("QuiverTextToSVGNode")?;
        source
            .arguments
            .insert("reference_images".into(), json!({}));
        let response = default_response()?;
        let mut transport = prepared(response.clone());
        let mut owner = UnavailableReferenceOwner::default();
        let result = invoke(
            &source,
            credentials(),
            2_000_000,
            &mut transport,
            &mut owner,
        )?;
        assert_eq!(owner.ordered_calls, 0);
        assert_eq!(owner.upload_calls, 0);
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(
            transport.costs.first().ok_or("cost")?.operation,
            "quiver.generation.maximum"
        );
        let body: Value = serde_json::from_slice(transport.bodies.first().ok_or("body")?)?;
        assert_eq!(body, result.plan.body);
        assert!(body.get("references").is_none());
        assert_eq!(result.response_bytes, serde_json::to_vec(&response)?);
        assert!(transport.events.is_empty());
        Ok(())
    }
    #[test]
    fn quiver_nonempty_reference_owner_and_empty_prompt_order_remain_required()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("QuiverTextToSVGNode")?;
        source.arguments.insert(
            "reference_images".into(),
            json!({"ordered_entries":[{"image":{"native_slot":"one"}}]}),
        );
        let mut owner = UnavailableReferenceOwner::default();
        assert!(matches!(
            request(&source, &mut owner),
            Err(QuiverError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(owner.ordered_calls, 1);
        assert_eq!(owner.upload_calls, 0);
        source.arguments.insert("prompt".into(), json!(""));
        assert!(matches!(
            request(&source, &mut owner),
            Err(QuiverError::Input(
                "Source prompt must contain at least one character"
            ))
        ));
        assert_eq!(owner.ordered_calls, 1);
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
                &mut owner
            ),
            Err(QuiverError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        assert!(transport.costs.is_empty());
        assert!(transport.bodies.is_empty());
        assert_eq!(owner.ordered_calls, 1);
        Ok(())
    }
}
