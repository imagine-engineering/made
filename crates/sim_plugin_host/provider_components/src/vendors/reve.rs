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
pub const NAMESPACE: &str = "zed.sim.provider.reve";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "d80ca98264a8a665f979682fd40324210af88483d2d74920ba8952c77ff63a73";
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
        feature_id: "COMFY-NODE-0546",
        node_identifier: "ReveImageCreateNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0547",
        node_identifier: "ReveImageEditNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0548",
        node_identifier: "ReveImageRemixNode",
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
        feature_id: "COMFY-API-EXT-0145",
        method: "POST",
        path: "/proxy/reve/v1/image/create",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0146",
        method: "POST",
        path: "/proxy/reve/v1/image/edit",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0147",
        method: "POST",
        path: "/proxy/reve/v1/image/remix",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum ReveError {
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
impl fmt::Display for ReveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Reve HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for ReveError {}
impl From<ConformanceError> for ReveError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for ReveError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, ReveError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/reve/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, ReveError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(ReveError::Input("Unknown Reve node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(ReveError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(ReveError::Input("Unknown Reve source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(ReveError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(ReveError::Input("Missing Reve source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, ReveError> {
    input
        .arguments
        .get(name)
        .ok_or(ReveError::Input("Missing Reve argument"))
}
fn text(value: &Value) -> Result<&str, ReveError> {
    value
        .as_str()
        .ok_or(ReveError::Input("Expected Reve string"))
}
fn endpoint(value: &str) -> Result<bool, ReveError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(ReveError::Response("Invalid Reve response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(ReveError::Response("Invalid Reve response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(ReveError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(ReveError::Response("Invalid Reve endpoint authority"));
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
) -> Result<RequestHead, ReveError> {
    let relative = endpoint(path)?;
    let mut headers = vec![Header {
        name: "Accept".into(),
        value: if binary {
            "image/webp"
        } else {
            "application/json"
        }
        .into(),
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
            return Err(ReveError::Input("Missing Comfy API credentials"));
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImagePolicy {
    pub media_type: &'static str,
    pub total_pixels: usize,
    pub first_batch: bool,
}
#[derive(Clone, Debug)]
pub struct Base64Image {
    pub encoded: String,
    pub receipt: Vec<u8>,
}
pub trait ReveImageOwner {
    fn encode_base64(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<Base64Image, ConformanceError>;
    fn ordered_references(&mut self, value: &Value) -> Result<Vec<Value>, ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: &'static str,
    pub body: Value,
    pub encoding_receipts: Vec<Vec<u8>>,
}
fn image_policy() -> ImagePolicy {
    ImagePolicy {
        media_type: "image/png",
        total_pixels: 4194304,
        first_batch: true,
    }
}
fn python_whitespace(value: char) -> bool {
    value.is_whitespace() || matches!(value, '\u{001c}'..='\u{001f}')
}
fn preflight(input: &Input) -> Result<(), ReveError> {
    let value = text(argument(
        input,
        if input.node == "ReveImageEditNode" {
            "edit_instruction"
        } else {
            "prompt"
        },
    )?)?;
    let length = value.trim_matches(python_whitespace).chars().count();
    if !(1..=2560).contains(&length) {
        return Err(ReveError::Input("Invalid Reve prompt length"));
    }
    Ok(())
}
fn encode(
    image: &Value,
    owner: &mut dyn ReveImageOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<String, ReveError> {
    let encoded = owner.encode_base64(image, &image_policy())?;
    if encoded.encoded.is_empty()
        || encoded.encoded.len() > MAXIMUM_REQUEST_BYTES
        || encoded.receipt.is_empty()
        || encoded.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ConformanceError::InvalidResponse.into());
    }
    receipts.push(encoded.receipt);
    Ok(encoded.encoded)
}
fn model_value<'a>(model: &'a Value, name: &str) -> Result<&'a Value, ReveError> {
    model
        .get(name)
        .ok_or(ReveError::Input("Missing Reve model option"))
}
fn postprocessing(input: &Input) -> Result<Option<Value>, ReveError> {
    let upscale = argument(input, "upscale")?;
    let enabled = text(
        upscale
            .get("upscale")
            .ok_or(ReveError::Input("Missing Reve upscale mode"))?,
    )? == "enabled";
    let remove = argument(input, "remove_background")?
        .as_bool()
        .ok_or(ReveError::Input("Invalid Reve remove_background"))?;
    let mut values = vec![];
    if enabled {
        let factor = upscale
            .get("upscale_factor")
            .ok_or(ReveError::Input("Missing Reve upscale factor"))?;
        let mut value = json!({"process":"upscale"});
        if !factor.is_null() {
            let factor = factor
                .as_i64()
                .filter(|value| (2..=4).contains(value))
                .ok_or(ReveError::Input("Invalid Reve upscale factor"))?;
            value["upscale_factor"] = json!(factor);
        }
        values.push(value);
    }
    if remove {
        values.push(json!({"process":"remove_background"}));
    }
    Ok((!values.is_empty()).then(|| json!(values)))
}
pub fn request(input: &Input, owner: &mut dyn ReveImageOwner) -> Result<Request, ReveError> {
    let input = normalized(input)?;
    preflight(&input)?;
    let model = argument(&input, "model")?;
    let mut receipts = vec![];
    let mut body = serde_json::Map::new();
    let path;
    if input.node == "ReveImageCreateNode" {
        path = "/proxy/reve/v1/image/create";
        body.insert("prompt".into(), argument(&input, "prompt")?.clone());
        let ratio = model_value(model, "aspect_ratio")?;
        if !ratio.is_null() {
            body.insert("aspect_ratio".into(), json!(text(ratio)?));
        }
        body.insert("version".into(), json!(text(model_value(model, "model")?)?));
        let scaling = model_value(model, "test_time_scaling")?
            .as_i64()
            .filter(|value| (1..=15).contains(value))
            .ok_or(ReveError::Input("Invalid Reve test_time_scaling"))?;
        body.insert("test_time_scaling".into(), json!(scaling));
    } else {
        let references = if input.node == "ReveImageRemixNode" {
            let references = argument(&input, "reference_images")?;
            if references.is_null()
                || references
                    .as_object()
                    .is_some_and(|references| references.is_empty())
            {
                return Err(ReveError::Input(
                    "At least one reference image is required.",
                ));
            }
            let values = owner.ordered_references(references)?;
            if values.is_empty() {
                return Err(ReveError::Input(
                    "At least one reference image is required.",
                ));
            }
            if values.len() > 128 {
                return Err(ConformanceError::BodyLimit.into());
            }
            let mut encoded = vec![];
            for value in values {
                encoded.push(encode(&value, owner, &mut receipts)?);
            }
            if encoded.len() > 6 {
                return Err(ReveError::Input("Maximum 6 reference images are allowed."));
            }
            Some(encoded)
        } else {
            None
        };
        let scaling = model_value(model, "test_time_scaling")?;
        let ratio = model_value(model, "aspect_ratio")?;
        if input.node == "ReveImageEditNode" {
            path = "/proxy/reve/v1/image/edit";
            body.insert(
                "edit_instruction".into(),
                argument(&input, "edit_instruction")?.clone(),
            );
            body.insert(
                "reference_image".into(),
                json!(encode(argument(&input, "image")?, owner, &mut receipts)?),
            );
        } else {
            path = "/proxy/reve/v1/image/remix";
            body.insert("prompt".into(), argument(&input, "prompt")?.clone());
            body.insert(
                "reference_images".into(),
                json!(references.ok_or(ReveError::Input("Missing Reve references"))?),
            );
        }
        if !ratio.is_null() && text(ratio)? != "auto" {
            body.insert("aspect_ratio".into(), json!(text(ratio)?));
        }
        body.insert("version".into(), json!(text(model_value(model, "model")?)?));
        if !scaling.is_null() {
            let scaling = scaling
                .as_i64()
                .ok_or(ReveError::Input("Invalid Reve test_time_scaling"))?;
            if scaling > 1 {
                if scaling > 15 {
                    return Err(ReveError::Input("Invalid Reve test_time_scaling"));
                }
                body.insert("test_time_scaling".into(), json!(scaling));
            }
        }
    }
    if let Some(value) = postprocessing(&input)? {
        body.insert("postprocessing".into(), value);
    }
    Ok(Request {
        path,
        body: Value::Object(body),
        encoding_receipts: receipts,
    })
}
pub fn estimated_cost_microunits(input: &Input) -> Result<u64, ReveError> {
    let input = normalized(input)?;
    let upscale = argument(&input, "upscale")?;
    let enabled = text(
        upscale
            .get("upscale")
            .ok_or(ReveError::Input("Missing Reve upscale mode"))?,
    )? == "enabled";
    let factor = upscale.get("upscale_factor").and_then(Value::as_i64);
    let rates = oracle()?["price_rates"][&input.node]
        .as_array()
        .ok_or(ReveError::Input("Missing Reve price literals"))?
        .clone();
    let index = if input.node == "ReveImageCreateNode" {
        if !enabled {
            3
        } else if factor == Some(4) {
            0
        } else if factor == Some(3) {
            1
        } else {
            2
        }
    } else if text(model_value(argument(&input, "model")?, "model")?)?.contains("fast") {
        0
    } else if !enabled {
        4
    } else if factor == Some(4) {
        1
    } else if factor == Some(3) {
        2
    } else {
        3
    };
    let price = rates
        .get(index)
        .and_then(Value::as_f64)
        .ok_or(ReveError::Input("Missing Reve price literal"))?;
    Ok((price * 1_000_000.0).round() as u64)
}
pub fn source_header_price(headers: &[Header]) -> Option<f64> {
    headers
        .iter()
        .rev()
        .find(|header| header.name.eq_ignore_ascii_case("x-reve-credits-used"))
        .and_then(|header| {
            header
                .value
                .trim_matches(python_whitespace)
                .parse::<f64>()
                .ok()
        })
        .map(|value| value / 524.48)
}
fn validate_binary_headers(headers: &[Header]) -> Result<(), ReveError> {
    let value = |name: &str| {
        headers
            .iter()
            .rev()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .map(|header| header.value.as_str())
    };
    if let Some(code) = value("x-reve-error-code").filter(|value| !value.is_empty()) {
        return Err(ReveError::Provider(format!("Reve API error: {code}")));
    }
    if value("x-reve-content-violation").is_some_and(|value| value.eq_ignore_ascii_case("true")) {
        return Err(ReveError::Provider(
            "The generated image was flagged for content policy violation.".into(),
        ));
    }
    Ok(())
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
pub trait ReveTransport: ProviderTransport {
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
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub response_bytes: Vec<u8>,
    pub response_headers: Vec<Header>,
    pub receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub authoritative_cost_usd: Option<f64>,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn ReveTransport,
    image: &mut dyn ReveImageOwner,
) -> Result<ResultValue, ReveError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    preflight(&input)?;
    if estimated_cost_microunits(&input)? > ceiling_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    let path = match input.node.as_str() {
        "ReveImageCreateNode" => "/proxy/reve/v1/image/create",
        "ReveImageEditNode" => "/proxy/reve/v1/image/edit",
        _ => "/proxy/reve/v1/image/remix",
    };
    let head = request_head(path, HttpMethod::Post, credentials, true)?;
    let handle = transport.start_request(&head)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "reve.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits: ceiling_microunits,
        },
    )?;
    if !approval.accepted
        || approval.approved_microunits < ceiling_microunits
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
    let (response_bytes, receipt, _, response_headers) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    let authoritative_cost_usd = source_header_price(&response_headers);
    validate_binary_headers(&response_headers)?;
    Ok(ResultValue {
        plan,
        response_bytes,
        response_headers,
        receipt,
        approval_receipt: approval.receipt,
        authoritative_cost_usd,
    })
}
pub trait ReveImageMaterializer {
    fn materialize_rgba(
        &mut self,
        result: &ResultValue,
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
fn expected_outputs() -> Result<Vec<ExpectedOutput>, ReveError> {
    let registry =
        TypeRegistry::built_in().map_err(|_| ReveError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve("IMAGE")
        .map_err(|_| ReveError::Response("Missing canonical IMAGE type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| ReveError::Response("Missing canonical IMAGE family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub struct MaterializationOwners<'a> {
    pub image: &'a mut dyn ReveImageOwner,
    pub decoder: &'a mut dyn ReveImageMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn ReveTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), ReveError> {
    let result = invoke(
        input,
        credentials,
        ceiling_microunits,
        transport,
        owners.image,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.decoder.materialize_rgba(&result)?;
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
type BinaryResponse = (Vec<u8>, Vec<u8>, StreamHandle, Vec<Header>);
fn exchange_existing(
    _head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    mut handle: StreamHandle,
    transport: &mut dyn ReveTransport,
) -> Result<BinaryResponse, ReveError> {
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
            Err(ReveError::Http {
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
                    return Err(ReveError::Http {
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
    transport: &mut dyn ReveTransport,
) -> Result<BinaryResponse, ReveError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "reve.generation.maximum".into(),
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
    let mut response_headers = vec![];
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
                response_headers = headers;
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
                    return Err(ReveError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
                    });
                }
                transport.check_cancelled(Some(handle))?;
                return Ok((bytes, receipt, handle, response_headers));
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
impl ReveImageOwner for SourceProbeOwner {
    fn encode_base64(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<Base64Image, ConformanceError> {
        if self.cancelled {
            return Err(ConformanceError::Cancelled);
        }
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        self.operations.push(json!({"operation":"encode_base64","image":image,"mime_type":policy.media_type,"total_pixels":policy.total_pixels,"first_batch":policy.first_batch}));
        if image["shape"]
            .as_array()
            .and_then(|shape| shape.first())
            .and_then(Value::as_u64)
            == Some(0)
        {
            return Err(ConformanceError::InvalidResponse);
        }
        let encoded =
            oracle().map_err(|_| ConformanceError::InvalidRequest)?["encoded_image"]["base64"]
                .as_str()
                .ok_or(ConformanceError::InvalidResponse)?
                .to_owned();
        Ok(Base64Image {
            encoded,
            receipt: b"source-probe-only-not-native".to_vec(),
        })
    }
    fn ordered_references(&mut self, value: &Value) -> Result<Vec<Value>, ConformanceError> {
        value["ordered_entries"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?
            .iter()
            .map(|item| {
                item.get("image")
                    .cloned()
                    .ok_or(ConformanceError::InvalidRequest)
            })
            .collect()
    }
}
fn input_from(case: &Value) -> Result<Input, Box<dyn Error>> {
    Ok(serde_json::from_value(case["input"].clone())?)
}
fn compare_case(case: &Value) -> Result<(), Box<dyn Error>> {
    let input = input_from(case)?;
    let mut owner = SourceProbeOwner::default();
    let plan = request(&input, &mut owner)?;
    let requests = case["requests"].as_array().ok_or("requests")?;
    if requests.len() != 1 {
        return Err("Reve source request count".into());
    }
    let expected = requests.first().ok_or("request")?;
    if plan.body != expected["body"]
        || json!(plan.path) != expected["path"]
        || expected["headers"] != json!({"Accept":"image/webp"})
        || owner.operations != case["media"].as_array().ok_or("media")?.clone()
    {
        return Err("Reve source request differs".into());
    }
    Ok(())
}
fn run_source_conformance(fixture: &str) -> Result<(), Box<dyn Error>> {
    let claims: Value = serde_json::from_str(fixture)?;
    if claims["nodes"] != serde_json::to_value(NODES)?
        || claims["routes"] != serde_json::to_value(ROUTES)?
        || claims["aliases"] != json!(ALIASES)
        || claims["source_sha256"] != SOURCE_SHA256
        || claims["namespace"] != NAMESPACE
    {
        return Err("Reve claims differ".into());
    }
    let source = oracle()?;
    if source["source_sha256"]["comfy_api_nodes/nodes_reve.py"] != SOURCE_SHA256 {
        return Err("Reve source differs".into());
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"].as_array().ok_or("source cases")? {
        compare_case(case)?;
        covered.insert(input_from(case)?.node);
    }
    for case in source["preflight_errors"]
        .as_array()
        .ok_or("source failures")?
    {
        let mut owner = SourceProbeOwner::default();
        if request(&input_from(case)?, &mut owner).is_ok()
            || owner.operations != case["media"].as_array().ok_or("source media")?.clone()
        {
            return Err("Reve source failure/order differs".into());
        }
    }
    if covered
        != NODES
            .iter()
            .map(|claim| claim.node_identifier.to_owned())
            .collect()
    {
        return Err("Reve claim coverage differs".into());
    }
    Ok(())
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let value: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if value
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"scope":"Source request and direct binary transport; canonical native RGBA and guest integration pending"})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    run_source_conformance(fixture).map_err(|_| ConformanceError::InvalidRequest)
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
    impl ReveTransport for Transport {
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
    fn prepared(headers: Vec<Header>) -> Transport {
        Transport {
            events: events(
                200,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/reve/source-response.webp"
                ))
                .to_vec(),
                headers,
            ),
            ..Default::default()
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
    struct ImageOwner {
        bytes: Vec<Vec<u8>>,
        empty: bool,
        cancelled: bool,
    }
    impl ReveImageMaterializer for ImageOwner {
        fn materialize_rgba(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            self.bytes.push(result.response_bytes.clone());
            if self.cancelled {
                return Err(ConformanceError::Cancelled);
            }
            if self.empty {
                return Ok(vec![]);
            }
            Err(ConformanceError::HostFailure)
        }
    }
    fn header(name: &str, value: &str) -> Header {
        Header {
            name: name.into(),
            value: value.into(),
        }
    }
    #[test]
    fn reve_exact_claims_and_all_source_request_preflight_traces_match()
    -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/reve/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 3);
        assert_eq!(ROUTES.len(), 3);
        assert!(ALIASES.is_empty());
        assert!(
            ROUTES
                .iter()
                .all(|route| route.method == "POST" && route.disposition == "executable")
        );
        Ok(())
    }
    #[test]
    fn reve_postprocessing_order_null_factor_scaling_and_seed_follow_actual_source()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["cases"].as_array().ok_or("cases")? {
            compare_case(case)?;
            let mut source = input_from(case)?;
            source.arguments.insert("seed".into(), json!(999));
            let plan = request(&source, &mut SourceProbeOwner::default())?;
            assert_eq!(plan.body, case["requests"][0]["body"]);
            assert!(plan.body.get("seed").is_none());
        }
        let mut source = input("ReveImageEditNode")?;
        source.arguments.insert(
            "model".into(),
            json!({"model":"custom","aspect_ratio":"auto","test_time_scaling":-3}),
        );
        source.arguments.insert(
            "upscale".into(),
            json!({"upscale":"enabled","upscale_factor":null}),
        );
        source
            .arguments
            .insert("remove_background".into(), json!(true));
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert!(plan.body.get("aspect_ratio").is_none());
        assert!(plan.body.get("test_time_scaling").is_none());
        assert_eq!(
            plan.body["postprocessing"],
            json!([{"process":"upscale"},{"process":"remove_background"}])
        );
        Ok(())
    }
    #[test]
    fn reve_reference_order_and_count_check_follow_all_source_encodes() -> Result<(), Box<dyn Error>>
    {
        for case in oracle()?["preflight_errors"].as_array().ok_or("errors")? {
            let mut owner = SourceProbeOwner::default();
            assert!(request(&input_from(case)?, &mut owner).is_err());
            assert_eq!(json!(owner.operations), case["media"]);
        }
        let mut source = input("ReveImageRemixNode")?;
        let entries=(0..7).map(|index|json!({"key":format!("image_{}",9-index),"image":{"native_slot":format!("ref-{index}"),"shape":[1,720,720,3]}})).collect::<Vec<_>>();
        source.arguments.insert(
            "reference_images".into(),
            json!({"ordered_entries":entries}),
        );
        let mut owner = SourceProbeOwner::default();
        assert!(request(&source, &mut owner).is_err());
        assert_eq!(owner.operations.len(), 7);
        for (index, operation) in owner.operations.iter().enumerate() {
            assert_eq!(operation["image"]["native_slot"], format!("ref-{index}"));
        }
        Ok(())
    }
    #[test]
    fn reve_prompt_validates_stripped_unicode_length_and_preserves_body()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("ReveImageCreateNode")?;
        source
            .arguments
            .insert("prompt".into(), json!("\u{001c} α \u{001f}"));
        let plan = request(&source, &mut SourceProbeOwner::default())?;
        assert_eq!(plan.body["prompt"], source.arguments["prompt"]);
        for value in ["\u{001c}\u{001f}".to_owned(), "😀".repeat(2561)] {
            source.arguments.insert("prompt".into(), json!(value));
            assert!(request(&source, &mut SourceProbeOwner::default()).is_err());
        }
        Ok(())
    }
    #[test]
    fn reve_badge_prices_are_estimates_and_approval_uses_caller_ceiling()
    -> Result<(), Box<dyn Error>> {
        for node in NODES {
            for (enabled, factor, expected) in [
                (
                    false,
                    2,
                    if node.node_identifier == "ReveImageCreateNode" {
                        34320
                    } else {
                        57200
                    },
                ),
                (
                    true,
                    2,
                    if node.node_identifier == "ReveImageCreateNode" {
                        45700
                    } else {
                        68600
                    },
                ),
                (
                    true,
                    3,
                    if node.node_identifier == "ReveImageCreateNode" {
                        59100
                    } else {
                        81900
                    },
                ),
                (
                    true,
                    4,
                    if node.node_identifier == "ReveImageCreateNode" {
                        76200
                    } else {
                        99100
                    },
                ),
            ] {
                let mut source = input(node.node_identifier)?;
                source.arguments.insert("upscale".into(),json!({"upscale":if enabled{"enabled"}else{"disabled"},"upscale_factor":factor}));
                assert_eq!(estimated_cost_microunits(&source)?, expected);
            }
        }
        let mut source = input("ReveImageEditNode")?;
        source.arguments.get_mut("model").ok_or("model")?["model"] = json!("unknown-fast-version");
        assert_eq!(estimated_cost_microunits(&source)?, 10010);
        let mut transport = prepared(vec![]);
        invoke(
            &source,
            credentials(),
            1_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(
            transport.costs.first().ok_or("cost")?.operation,
            "reve.generation.maximum"
        );
        assert_eq!(
            transport.costs.first().ok_or("cost")?.maximum_microunits,
            1_000_000
        );
        Ok(())
    }
    #[test]
    fn reve_binary_credits_headers_preserve_source_values_and_errors_are_suppressed()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["header_prices"].as_array().ok_or("headers")? {
            let headers = case["headers"]
                .as_object()
                .ok_or("headers")?
                .iter()
                .map(|(name, value)| Ok(header(name, value.as_str().ok_or("value")?)))
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let actual = source_header_price(&headers);
            if case.get("suppressed_exception").is_some() || case["price_usd"].is_null() {
                assert!(actual.is_none());
            } else {
                let expected = case["price_usd"].as_str().ok_or("price")?;
                let actual = actual.ok_or("price")?;
                if expected == "nan" {
                    assert!(actual.is_nan());
                } else if expected == "inf" {
                    assert!(actual.is_infinite() && actual.is_sign_positive());
                } else {
                    assert_eq!(actual, expected.parse::<f64>()?);
                }
            }
        }
        Ok(())
    }
    #[test]
    fn reve_header_errors_precede_real_image_decode_and_error_code_wins()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["response_errors"].as_array().ok_or("errors")? {
            if case["response_headers"].is_null() {
                continue;
            }
            let headers = case["response_headers"]
                .as_object()
                .ok_or("headers")?
                .iter()
                .map(|(name, value)| Ok(header(name, value.as_str().ok_or("value")?)))
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let error = validate_binary_headers(&headers)
                .err()
                .ok_or("expected header failure")?;
            assert_eq!(error.to_string(), case["message"]);
        }
        assert!(validate_binary_headers(&[header("X-Reve-Content-Violation", " true ")]).is_ok());
        assert!(validate_binary_headers(&[header("X-Reve-Error-Code", "")]).is_ok());
        Ok(())
    }
    #[test]
    fn reve_direct_webp_bytes_receipts_auth_headers_and_source_policy_are_retained()
    -> Result<(), Box<dyn Error>> {
        let source = input("ReveImageCreateNode")?;
        let mut transport = prepared(vec![header("X-Reve-Credits-Used", "524.48")]);
        let result = invoke(
            &source,
            credentials(),
            1_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(
            result.response_bytes,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/reve/source-response.webp"
            ))
        );
        assert_eq!(result.authoritative_cost_usd, Some(1.0));
        assert_eq!(result.receipt, vec![7, 8, 9]);
        assert_eq!(result.approval_receipt, vec![42]);
        assert_eq!(transport.heads.len(), 1);
        let head = transport.heads.first().ok_or("head")?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.auth_token"));
        assert_eq!(
            head.headers.first().ok_or("accept")?,
            &header("Accept", "image/webp")
        );
        assert!(
            head.headers
                .contains(&header("Content-Type", "application/json"))
        );
        assert_eq!(
            transport.policies.first().ok_or("policy")?.1,
            ResponsePolicy::source()
        );
        assert!(transport.progresses.is_empty());
        Ok(())
    }
    #[test]
    fn reve_canonical_rgba_decode_rejection_is_atomic_for_real_webp_bytes()
    -> Result<(), Box<dyn Error>> {
        let expected = expected_outputs()?;
        assert_eq!(
            expected.first().ok_or("output")?.type_id.to_string(),
            "sim:image@1"
        );
        for (empty, cancelled) in [(false, false), (true, false), (false, true)] {
            let mut transport = prepared(vec![]);
            let mut decoder = ImageOwner {
                empty,
                cancelled,
                ..Default::default()
            };
            let mut image = SourceProbeOwner::default();
            let mut sink = Sink::default();
            let mut owners = MaterializationOwners {
                image: &mut image,
                decoder: &mut decoder,
                sink: &mut sink,
            };
            assert!(
                invoke_and_materialize(
                    &input("ReveImageCreateNode")?,
                    credentials(),
                    1_000_000,
                    &mut transport,
                    &mut owners
                )
                .is_err()
            );
            assert_eq!(sink.published, 0);
            assert_eq!(
                decoder.bytes.first().ok_or("bytes")?,
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/reve/source-response.webp"
                ))
            );
        }
        Ok(())
    }
    #[test]
    fn reve_auth_cost_native_encode_failure_prevent_body_and_api_key_is_supported()
    -> Result<(), Box<dyn Error>> {
        let source = input("ReveImageEditNode")?;
        for (credentials, deny_cost, fail, cancelled, ceiling) in [
            (
                CredentialAvailability {
                    auth_token: false,
                    api_key: false,
                },
                false,
                false,
                false,
                1_000_000,
            ),
            (credentials(), true, false, false, 1_000_000),
            (credentials(), false, true, false, 1_000_000),
            (credentials(), false, false, true, 1_000_000),
            (credentials(), false, false, false, 1),
        ] {
            let mut transport = Transport {
                deny_cost,
                ..Default::default()
            };
            let mut owner = SourceProbeOwner {
                fail,
                cancelled,
                ..Default::default()
            };
            assert!(invoke(&source, credentials, ceiling, &mut transport, &mut owner).is_err());
            assert!(transport.bodies.iter().all(Vec::is_empty));
        }
        let mut transport = prepared(vec![]);
        invoke(
            &source,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            1_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(
            transport.heads.first().ok_or("head")?.secret_id.as_deref(),
            Some("sim_org.api_key")
        );
        Ok(())
    }
    #[test]
    fn reve_retry_budgets_fresh_handles_body_replay_and_host_clock_remain_exact()
    -> Result<(), Box<dyn Error>> {
        for (status, count) in [(503, 3), (429, 16)] {
            let mut transport = Transport::default();
            for _ in 0..count {
                transport.events.extend(events(status, vec![], vec![]));
            }
            transport.events.extend(prepared(vec![]).events);
            invoke(
                &input("ReveImageCreateNode")?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut SourceProbeOwner::default(),
            )?;
            assert_eq!(transport.delays.len(), count);
            assert!(
                transport
                    .bodies
                    .windows(2)
                    .all(|pair| pair.first() == pair.get(1))
            );
            assert!(transport.delays.iter().all(|delay| *delay <= 150_000));
        }
        let mut transport = Transport::default();
        transport
            .events
            .extend(events(429, vec![], vec![header("Retry-After", "1000")]));
        transport.events.extend(prepared(vec![]).events);
        invoke(
            &input("ReveImageCreateNode")?,
            credentials(),
            1_000_000,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(transport.delays, vec![150_000]);
        let mut stale = Transport {
            events: events(429, vec![], vec![]),
            same_retry_handle: true,
            ..Default::default()
        };
        assert!(matches!(
            invoke(
                &input("ReveImageCreateNode")?,
                credentials(),
                1_000_000,
                &mut stale,
                &mut SourceProbeOwner::default()
            ),
            Err(ReveError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn reve_cancel_worker_loss_fresh_invoke_and_malformed_frames_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let source = input("ReveImageCreateNode")?;
        for event in [
            Err(ConformanceError::WorkerLost),
            Ok(ResponseEvent::Cancelled),
            Ok(ResponseEvent::Chunk(vec![1])),
            Ok(ResponseEvent::Completed(vec![])),
        ] {
            let mut transport = Transport {
                events: vec![event].into(),
                ..Default::default()
            };
            assert!(
                invoke(
                    &source,
                    credentials(),
                    1_000_000,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                )
                .is_err()
            );
            assert_eq!(transport.heads.len(), 1);
        }
        let mut cancelled = Transport {
            cancelled: true,
            ..Default::default()
        };
        assert!(
            invoke(
                &source,
                credentials(),
                1_000_000,
                &mut cancelled,
                &mut SourceProbeOwner::default()
            )
            .is_err()
        );
        assert!(cancelled.heads.is_empty());
        let mut fresh = prepared(vec![]);
        invoke(
            &source,
            credentials(),
            1_000_000,
            &mut fresh,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(fresh.heads.len(), 1);
        let mut malformed = Transport {
            events: events(200, vec![], vec![header("Bad\nHeader", "bad")]),
            ..Default::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                1_000_000,
                &mut malformed,
                &mut SourceProbeOwner::default()
            ),
            Err(ReveError::Transport(ConformanceError::HeaderLimit))
        ));
        let mut oversized = Transport {
            events: events(200, vec![1; MAXIMUM_CHUNK_BYTES + 1], vec![]),
            ..Default::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                1_000_000,
                &mut oversized,
                &mut SourceProbeOwner::default()
            ),
            Err(ReveError::Transport(ConformanceError::ChunkLimit))
        ));
        Ok(())
    }
    #[test]
    fn reve_aggregate_receipt_and_duplicate_head_limits_are_typed() -> Result<(), Box<dyn Error>> {
        let source = input("ReveImageCreateNode")?;
        for receipt in [vec![], vec![1; MAXIMUM_RECEIPT_BYTES + 1]] {
            let mut transport = Transport {
                events: vec![
                    Ok(ResponseEvent::Head {
                        status: 200,
                        headers: vec![],
                    }),
                    Ok(ResponseEvent::Completed(receipt)),
                ]
                .into(),
                ..Default::default()
            };
            assert!(matches!(
                invoke(
                    &source,
                    credentials(),
                    1_000_000,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                ),
                Err(ReveError::Transport(ConformanceError::InvalidResponse))
            ));
        }
        let mut aggregate = Transport::default();
        aggregate.events.push_back(Ok(ResponseEvent::Head {
            status: 200,
            headers: vec![],
        }));
        for _ in 0..=MAXIMUM_RESPONSE_BYTES / MAXIMUM_CHUNK_BYTES {
            aggregate
                .events
                .push_back(Ok(ResponseEvent::Chunk(vec![1; MAXIMUM_CHUNK_BYTES])));
        }
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                1_000_000,
                &mut aggregate,
                &mut SourceProbeOwner::default()
            ),
            Err(ReveError::Transport(ConformanceError::BodyLimit))
        ));
        let mut duplicate = Transport {
            events: vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
            ]
            .into(),
            ..Default::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                1_000_000,
                &mut duplicate,
                &mut SourceProbeOwner::default()
            ),
            Err(ReveError::Transport(ConformanceError::InvalidResponse))
        ));
        Ok(())
    }

    #[derive(Default)]
    struct UnavailableReferences {
        ordered_calls: usize,
        encode_calls: usize,
    }
    impl ReveImageOwner for UnavailableReferences {
        fn encode_base64(
            &mut self,
            _image: &Value,
            _policy: &ImagePolicy,
        ) -> Result<Base64Image, ConformanceError> {
            self.encode_calls += 1;
            Err(ConformanceError::HostFailure)
        }
        fn ordered_references(&mut self, _value: &Value) -> Result<Vec<Value>, ConformanceError> {
            self.ordered_calls += 1;
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn reve_empty_remix_references_preserve_source_error_before_image_owner()
    -> Result<(), Box<dyn Error>> {
        for references in [Value::Null, json!({})] {
            let mut source = input("ReveImageRemixNode")?;
            source
                .arguments
                .insert("reference_images".into(), references);
            source.arguments.insert("model".into(), json!({}));
            let mut owner = UnavailableReferences::default();
            assert!(matches!(
                request(&source, &mut owner),
                Err(ReveError::Input(
                    "At least one reference image is required."
                ))
            ));
            assert_eq!(owner.ordered_calls, 0);
            assert_eq!(owner.encode_calls, 0);
            source.arguments.insert("prompt".into(), json!(""));
            assert!(matches!(
                request(&source, &mut owner),
                Err(ReveError::Input("Invalid Reve prompt length"))
            ));
            assert_eq!(owner.ordered_calls, 0);
        }
        Ok(())
    }
    #[test]
    fn reve_empty_remix_references_stop_body_after_existing_authority_boundary()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("ReveImageRemixNode")?;
        source
            .arguments
            .insert("reference_images".into(), json!({}));
        let mut transport = Transport::default();
        let mut owner = UnavailableReferences::default();
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut owner
            ),
            Err(ReveError::Input(
                "At least one reference image is required."
            ))
        ));
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(
            transport.costs.first().ok_or("cost")?.operation,
            "reve.generation.maximum"
        );
        assert_eq!(transport.bodies.len(), 1);
        assert!(transport.bodies.first().ok_or("body")?.is_empty());
        assert_eq!(owner.ordered_calls, 0);
        assert_eq!(owner.encode_calls, 0);
        assert!(transport.policies.is_empty());
        Ok(())
    }
    #[test]
    fn reve_nonempty_owner_failure_and_prestart_cancellation_remain_required()
    -> Result<(), Box<dyn Error>> {
        let mut source = input("ReveImageRemixNode")?;
        source.arguments.insert(
            "reference_images".into(),
            json!({"ordered_entries":[{"image":{"native_slot":"one"}}]}),
        );
        let mut owner = UnavailableReferences::default();
        assert!(matches!(
            request(&source, &mut owner),
            Err(ReveError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(owner.ordered_calls, 1);
        assert_eq!(owner.encode_calls, 0);
        let mut transport = Transport {
            cancelled: true,
            ..Default::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                2_000_000,
                &mut transport,
                &mut owner
            ),
            Err(ReveError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        assert!(transport.costs.is_empty());
        assert!(transport.bodies.is_empty());
        assert_eq!(owner.ordered_calls, 1);
        Ok(())
    }
}
