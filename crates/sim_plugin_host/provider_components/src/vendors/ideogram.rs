use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TypeRegistry, publish_validated, validate_outputs,
};
use crate::ideogram_response::{GenerateResponse, ImageDatum};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
pub const NAMESPACE: &str = "zed.sim.provider.ideogram";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "8798f8a43b27656a98d68daf30b8120524f3191d00539919efb794d04ad2ccc3";
pub const IMAGE_COLOR_MODE: &str = "RGB";
pub const FILE_CONTENT_TYPE: &str = "application/octet-stream";
pub const TEXT_CONTENT_TYPE: &str = "text/plain; charset=utf-8";
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
        feature_id: "COMFY-NODE-0238",
        node_identifier: "IdeogramV3",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0239",
        node_identifier: "IdeogramV4",
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
        feature_id: "COMFY-API-EXT-0052",
        method: "POST",
        path: "/proxy/ideogram/ideogram-v3/edit",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0053",
        method: "POST",
        path: "/proxy/ideogram/ideogram-v3/generate",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0054",
        method: "POST",
        path: "/proxy/ideogram/ideogram-v4/generate",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum IdeogramError {
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
impl fmt::Display for IdeogramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Ideogram HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for IdeogramError {}
impl From<ConformanceError> for IdeogramError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for IdeogramError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, IdeogramError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/ideogram/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, IdeogramError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(IdeogramError::Input("Unknown Ideogram node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(IdeogramError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(IdeogramError::Input("Unknown Ideogram source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(IdeogramError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(IdeogramError::Input("Missing Ideogram source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, IdeogramError> {
    input
        .arguments
        .get(name)
        .ok_or(IdeogramError::Input("Missing Ideogram argument"))
}
fn text(value: &Value) -> Result<&str, IdeogramError> {
    value
        .as_str()
        .ok_or(IdeogramError::Input("Expected Ideogram string"))
}
fn integer(value: &Value, maximum: u64) -> Result<u64, IdeogramError> {
    let number = value
        .as_u64()
        .ok_or(IdeogramError::Input("Expected Ideogram unsigned integer"))?;
    if number > maximum {
        return Err(IdeogramError::Input("Ideogram integer bound exceeded"));
    }
    Ok(number)
}
fn choice<'a>(value: &'a Value, choices: &[&str]) -> Result<&'a str, IdeogramError> {
    let text = text(value)?;
    if !choices.contains(&text) {
        return Err(IdeogramError::Input("Unknown Ideogram source choice"));
    }
    Ok(text)
}
fn endpoint(value: &str) -> Result<bool, IdeogramError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(IdeogramError::Response(
            "Invalid Ideogram response endpoint",
        ));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(IdeogramError::Response(
            "Invalid Ideogram response endpoint",
        ))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(IdeogramError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(IdeogramError::Response(
            "Invalid Ideogram endpoint authority",
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
) -> Result<RequestHead, IdeogramError> {
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
            return Err(IdeogramError::Input("Missing Comfy API credentials"));
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

pub trait IdeogramTransport: ProviderTransport {
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
    fn set_content_type(
        &mut self,
        handle: StreamHandle,
        content_type: &str,
    ) -> Result<(), ConformanceError>;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparationKind {
    Edit,
    Character,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskPolicy {
    pub upscale_method: &'static str,
    pub crop: &'static str,
    pub threshold: f64,
    pub threshold_comparison: &'static str,
    pub allow_gradient: bool,
    pub invert: bool,
}
pub const MASK_POLICY: MaskPolicy = MaskPolicy {
    upscale_method: "nearest-exact",
    crop: "disabled",
    threshold: 0.5,
    threshold_comparison: "greater",
    allow_gradient: false,
    invert: true,
};
#[derive(Clone, Debug)]
pub struct PreparedPair {
    pub image_png: Vec<u8>,
    pub mask_png: Option<Vec<u8>>,
}
pub trait IdeogramInputOwner {
    fn prepare_png(
        &mut self,
        image: &Value,
        mask: Option<&Value>,
        kind: PreparationKind,
        policy: MaskPolicy,
    ) -> Result<PreparedPair, ConformanceError>;
}
#[derive(Clone, Debug, PartialEq)]
pub struct FilePart {
    pub name: String,
    pub file_name: String,
    pub source_image: Value,
    pub source_mask: Option<Value>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub path: String,
    pub body: Value,
    pub multipart: bool,
    pub form_order: Vec<String>,
    pub files: Vec<FilePart>,
}
fn present<'a>(input: &'a Input, name: &str) -> Result<Option<&'a Value>, IdeogramError> {
    Ok(Some(argument(input, name)?).filter(|value| !value.is_null()))
}
fn file(name: &str, file_name: &str, image: &Value, mask: Option<&Value>) -> FilePart {
    FilePart {
        name: name.into(),
        file_name: file_name.into(),
        source_image: image.clone(),
        source_mask: mask.cloned(),
    }
}
fn source_trim(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
pub fn request(input: &Input) -> Result<Request, IdeogramError> {
    let input = normalized(input)?;
    let prompt = text(argument(&input, "prompt")?)?;
    let speed = choice(
        argument(&input, "rendering_speed")?,
        if input.node == "IdeogramV3" {
            &["DEFAULT", "BALANCED", "TURBO", "QUALITY"]
        } else {
            &["DEFAULT", "TURBO", "QUALITY"]
        },
    )?;
    integer(argument(&input, "seed")?, 2147483647)?;
    if input.node == "IdeogramV4" {
        if source_trim(prompt).is_empty() {
            return Err(IdeogramError::Input(
                "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long.",
            ));
        }
        let resolution = text(argument(&input, "resolution")?)?;
        let mut body = json!({"text_prompt":prompt,"rendering_speed":speed});
        if resolution != "Auto" {
            body["resolution"] = json!(
                resolution
                    .split(' ')
                    .next()
                    .ok_or(IdeogramError::Input("Missing V4 resolution"))?
            );
        }
        return Ok(Request {
            path: "/proxy/ideogram/ideogram-v4/generate".into(),
            body,
            multipart: false,
            form_order: vec![],
            files: vec![],
        });
    }
    let image = present(&input, "image")?;
    let mask = present(&input, "mask")?;
    let character = present(&input, "character_image")?;
    let character_mask = present(&input, "character_mask")?;
    if character.is_none() && character_mask.is_some() {
        return Err(IdeogramError::Input(
            "Character mask requires character image to be present",
        ));
    }
    if image.is_some() != mask.is_some() {
        return Err(IdeogramError::Input(
            "Ideogram V3 image editing requires both an image AND a mask",
        ));
    }
    let editing = image.is_some();
    let speed = if speed == "BALANCED" {
        "DEFAULT"
    } else {
        speed
    };
    let mut body = json!({"prompt":prompt,"rendering_speed":speed});
    let magic = choice(
        argument(&input, "magic_prompt_option")?,
        &["AUTO", "ON", "OFF"],
    )?;
    if magic != "AUTO" {
        body["magic_prompt"] = json!(magic);
    }
    let seed = integer(argument(&input, "seed")?, 2147483647)?;
    if seed != 0 {
        body["seed"] = json!(seed);
    }
    let count = integer(argument(&input, "num_images")?, 8)?;
    if count == 0 {
        return Err(IdeogramError::Input("Invalid Ideogram image count"));
    }
    if count > 1 {
        body["num_images"] = json!(count);
    }
    let mut files = vec![];
    if let (Some(image), Some(mask)) = (image, mask) {
        files.push(file("image", "image.png", image, None));
        files.push(file("mask", "mask.png", image, Some(mask)));
    }
    if let Some(character) = character {
        files.push(file(
            "character_reference_images",
            "image.png",
            character,
            None,
        ));
        if let Some(mask) = character_mask {
            files.push(file(
                "character_mask_binary",
                "mask.png",
                character,
                Some(mask),
            ));
        }
    }
    if !editing {
        let resolution = text(argument(&input, "resolution")?)?;
        let aspect = text(argument(&input, "aspect_ratio")?)?;
        if resolution != "Auto" {
            body["resolution"] = json!(resolution);
        } else if aspect != "1:1"
            && let Some(mapped) = oracle()?["ratio_map"].get(aspect)
        {
            body["aspect_ratio"] = mapped.clone();
        }
        if !files.is_empty() {
            body["style_type"] = json!("AUTO");
        }
    }
    let order = if editing {
        &[
            "magic_prompt",
            "num_images",
            "prompt",
            "rendering_speed",
            "seed",
        ][..]
    } else {
        &[
            "aspect_ratio",
            "magic_prompt",
            "num_images",
            "prompt",
            "rendering_speed",
            "resolution",
            "seed",
            "style_type",
        ][..]
    };
    let form_order = order
        .iter()
        .filter(|name| body.get(**name).is_some())
        .map(|name| (*name).into())
        .collect();
    Ok(Request {
        path: format!(
            "/proxy/ideogram/ideogram-v3/{}",
            if editing { "edit" } else { "generate" }
        ),
        body,
        multipart: true,
        form_order,
        files,
    })
}
fn checked_pair(pair: PreparedPair, needs_mask: bool) -> Result<PreparedPair, IdeogramError> {
    if pair.image_png.is_empty()
        || pair.image_png.len() > MAXIMUM_MEDIA_BYTES
        || pair.mask_png.is_some() != needs_mask
        || pair
            .mask_png
            .as_ref()
            .is_some_and(|bytes| bytes.is_empty() || bytes.len() > MAXIMUM_MEDIA_BYTES)
    {
        return Err(IdeogramError::Input(
            "Invalid canonical image/mask PNG payload",
        ));
    }
    Ok(pair)
}
pub struct CanonicalMultipartEncoder {
    pub boundary: String,
}
impl CanonicalMultipartEncoder {
    fn encode(&self, fields: &[MultipartField]) -> Result<(Vec<u8>, String), ConformanceError> {
        let mut limits = crate::common::TransportLimits::conformance();
        limits.maximum_body_bytes = MAXIMUM_REQUEST_BYTES;
        Ok((
            MultipartField::encode_checked(fields, &self.boundary, limits)?,
            format!("multipart/form-data; boundary={}", self.boundary),
        ))
    }
}
#[derive(Clone, Debug)]
pub struct DownloadedImage {
    pub url: String,
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub response_bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub images: Vec<DownloadedImage>,
    pub color_mode: &'static str,
}
fn image_urls(bytes: &[u8]) -> Result<Vec<String>, IdeogramError> {
    let GenerateResponse {
        data,
        created: _created,
    } = serde_json::from_slice(bytes)?;
    let data = data
        .filter(|data| !data.is_empty())
        .ok_or(IdeogramError::Response(
            "No images were generated in the response",
        ))?;
    let mut urls = vec![];
    for item in data {
        let ImageDatum {
            url,
            is_image_safe,
            prompt,
            resolution,
            seed,
            style_type,
        } = item;
        let metadata = (is_image_safe, prompt, resolution, seed, style_type);
        drop(metadata);
        if let Some(url) = url.filter(|url| !url.is_empty()) {
            endpoint(&url)?;
            urls.push(url);
        }
    }
    if urls.is_empty() {
        return Err(IdeogramError::Response(
            "No image URLs were generated in the response",
        ));
    }
    Ok(urls)
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn IdeogramTransport,
    owner: &mut dyn IdeogramInputOwner,
    encoder: &CanonicalMultipartEncoder,
) -> Result<ResultValue, IdeogramError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    // Source prepares character references before reporting an incomplete edit pair.
    let character = if input.node == "IdeogramV3" {
        if let Some(image) = present(&input, "character_image")? {
            transport.check_cancelled(None)?;
            Some(checked_pair(
                owner.prepare_png(
                    image,
                    present(&input, "character_mask")?,
                    PreparationKind::Character,
                    MASK_POLICY,
                )?,
                present(&input, "character_mask")?.is_some(),
            )?)
        } else {
            None
        }
    } else {
        None
    };
    let plan = request(&input)?;
    let head = request_head(&plan.path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "ideogram.generation.maximum".into(),
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
    transport.check_cancelled(Some(handle))?;
    let body = if plan.multipart {
        let edit = if let Some(image) = present(&input, "image")? {
            Some(checked_pair(
                owner.prepare_png(
                    image,
                    present(&input, "mask")?,
                    PreparationKind::Edit,
                    MASK_POLICY,
                )?,
                true,
            )?)
        } else {
            None
        };
        let mut fields = vec![];
        for name in &plan.form_order {
            let value = plan
                .body
                .get(name)
                .ok_or(IdeogramError::Input("Missing source form field"))?;
            fields.push(MultipartField {
                name: name.clone(),
                file_name: None,
                media_type: Some(TEXT_CONTENT_TYPE.into()),
                bytes: if let Some(text) = value.as_str() {
                    text.as_bytes().to_vec()
                } else {
                    serde_json::to_vec(value)?
                },
            });
        }
        for part in &plan.files {
            let pair = if part.name == "image" || part.name == "mask" {
                edit.as_ref()
            } else {
                character.as_ref()
            }
            .ok_or(IdeogramError::Input("Missing prepared source file"))?;
            let bytes = if part.source_mask.is_some() {
                pair.mask_png
                    .as_ref()
                    .ok_or(IdeogramError::Input("Missing prepared mask"))?
            } else {
                &pair.image_png
            };
            fields.push(MultipartField {
                name: part.name.clone(),
                file_name: Some(part.file_name.clone()),
                media_type: Some(FILE_CONTENT_TYPE.into()),
                bytes: bytes.clone(),
            });
        }
        let (bytes, content_type) = encoder.encode(&fields)?;
        transport.set_content_type(handle, &content_type)?;
        bytes
    } else {
        serde_json::to_vec(&plan.body)?
    };
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
    let urls = image_urls(&response_bytes)?;
    let mut images = vec![];
    let mut total = 0usize;
    for url in urls {
        let head = request_head(&url, HttpMethod::Get, credentials, true)?;
        let (bytes, receipt, _) = exchange_with_policy(
            &head,
            &[],
            None,
            MAXIMUM_MEDIA_BYTES,
            DOWNLOAD_RETRY,
            transport,
        )?;
        total = total
            .checked_add(bytes.len())
            .filter(|total| *total <= MAXIMUM_MEDIA_BYTES)
            .ok_or(ConformanceError::BodyLimit)?;
        if bytes.is_empty() {
            return Err(IdeogramError::Response("Empty downloaded image"));
        }
        images.push(DownloadedImage {
            url,
            bytes,
            receipt,
        });
    }
    Ok(ResultValue {
        response_bytes,
        receipt,
        images,
        color_mode: IMAGE_COLOR_MODE,
    })
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
    transport: &mut dyn IdeogramTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), IdeogramError> {
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
    transport: &mut dyn IdeogramTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), IdeogramError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(IdeogramError::Http {
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
                    return Err(IdeogramError::Http {
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
    transport: &mut dyn IdeogramTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), IdeogramError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "ideogram.generation.maximum".into(),
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
                    return Err(IdeogramError::Http {
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

pub trait IdeogramMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub struct MaterializationOwners<'a> {
    pub input: &'a mut dyn IdeogramInputOwner,
    pub encoder: &'a CanonicalMultipartEncoder,
    pub materializer: &'a mut dyn IdeogramMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn IdeogramTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), IdeogramError> {
    let result = invoke(
        input,
        credentials,
        maximum_cost_microunits,
        transport,
        owners.input,
        owners.encoder,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.materializer.materialize(&result)?;
    let registry = TypeRegistry::built_in()
        .map_err(|_| IdeogramError::Response("Missing canonical type registry"))?;
    let type_id = registry
        .resolve("IMAGE")
        .map_err(|_| IdeogramError::Response("Missing canonical output type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| IdeogramError::Response("Missing canonical output family"))?;
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
    publish_validated(batch, owners.sink).map_err(ConformanceError::from)?;
    Ok(())
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
    if source["source_sha256"]["comfy_api_nodes/nodes_ideogram.py"] != SOURCE_SHA256 {
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
        let plan = request(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let expected = case["requests"]
            .as_array()
            .and_then(|requests| requests.first())
            .ok_or(ConformanceError::InvalidRequest)?;
        let files: Vec<_> = plan
            .files
            .iter()
            .map(|part| json!({"name":part.name,"file_name":part.file_name}))
            .collect();
        let expected_files: Vec<_> = expected["files"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?
            .iter()
            .map(|part| json!({"name":part["name"],"file_name":part["file_name"]}))
            .collect();
        if plan.body != expected["body"]
            || plan.path != expected["path"]
            || expected["max_retries"] != 1
            || files != expected_files
            || plan.multipart != (expected["content_type"] == "multipart/form-data")
        {
            return Err(ConformanceError::InvalidRequest);
        }
        if plan.multipart && json!(plan.form_order) != expected["form_order"] {
            return Err(ConformanceError::InvalidRequest);
        }
        if case["downloads"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?
            .iter()
            .any(|download| download["mode"] != "RGB")
        {
            return Err(ConformanceError::InvalidRequest);
        }
        if plan.multipart {
            let fields = expected["wire_fields"]
                .as_array()
                .ok_or(ConformanceError::InvalidRequest)?;
            if fields
                .iter()
                .filter(|field| !field["file_name"].is_null())
                .any(|field| field["media_type"] != FILE_CONTENT_TYPE)
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let names: Vec<_> = fields.iter().map(|field| field["name"].clone()).collect();
            let expected_names: Vec<_> = plan
                .form_order
                .iter()
                .cloned()
                .chain(plan.files.iter().map(|file| file.name.clone()))
                .map(Value::String)
                .collect();
            if names != expected_names {
                return Err(ConformanceError::InvalidRequest);
            }
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
        if request(&input).is_ok() || case["requests"] != json!([]) {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    if covered
        != NODES
            .iter()
            .map(|claim| claim.node_identifier.to_owned())
            .collect()
        || paths != ROUTES.iter().map(|claim| claim.path.to_owned()).collect()
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
    use crate::ideogram_response::ResponseCreated;
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
        content_types: Vec<(StreamHandle, String)>,
        fail_content_type: bool,
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
    impl IdeogramTransport for Transport {
        fn set_content_type(
            &mut self,
            handle: StreamHandle,
            content_type: &str,
        ) -> Result<(), ConformanceError> {
            if self.fail_content_type {
                return Err(ConformanceError::HostFailure);
            }
            self.content_types.push((handle, content_type.into()));
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
            "/../../sim_test_support/fixtures/providers/ideogram/downloaded-image.png"
        ))
        .to_vec()
    }
    fn mask_png() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/ideogram/prepared-mask.png"
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
    fn input(node: &str, arguments: Value) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(
            json!({"node":node,"arguments":arguments}),
        )?)
    }
    fn v3() -> Result<Input, Box<dyn Error>> {
        input("IdeogramV3", json!({"prompt":"source"}))
    }
    fn v4() -> Result<Input, Box<dyn Error>> {
        input(
            "IdeogramV4",
            json!({"prompt":"  source  ","resolution":"Auto","rendering_speed":"DEFAULT","seed":42}),
        )
    }
    fn response() -> Value {
        json!({"data":[{"url":"https://fixture.invalid/first.png","is_image_safe":false},{"url":null},{"url":"https://fixture.invalid/second.png"}]})
    }
    fn success_events()
    -> Result<VecDeque<Result<ResponseEvent, ConformanceError>>, serde_json::Error> {
        let mut sequence = events(200, serde_json::to_vec(&response())?, vec![]);
        sequence.extend(events(200, png(), vec![]));
        sequence.extend(events(200, png(), vec![]));
        Ok(sequence)
    }
    fn encoder() -> CanonicalMultipartEncoder {
        CanonicalMultipartEncoder {
            boundary: "source-boundary".into(),
        }
    }
    #[derive(Default)]
    struct Owner {
        calls: Vec<(Value, Option<Value>, PreparationKind, MaskPolicy)>,
        fail: bool,
        invalid: bool,
    }
    impl IdeogramInputOwner for Owner {
        fn prepare_png(
            &mut self,
            image: &Value,
            mask: Option<&Value>,
            kind: PreparationKind,
            policy: MaskPolicy,
        ) -> Result<PreparedPair, ConformanceError> {
            self.calls
                .push((image.clone(), mask.cloned(), kind, policy));
            if self.fail {
                return Err(ConformanceError::HostFailure);
            }
            Ok(PreparedPair {
                image_png: if self.invalid { vec![] } else { png() },
                mask_png: mask.map(|_| mask_png()),
            })
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
    #[derive(Default)]
    struct RejectDecoder {
        seen: bool,
    }
    impl IdeogramMaterializer for RejectDecoder {
        fn materialize(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            assert_eq!(result.color_mode, "RGB");
            assert_eq!(result.images.len(), 2);
            for image in &result.images {
                assert_eq!(image.bytes, png());
                assert_eq!(image.receipt, [7, 8, 9]);
            }
            self.seen = true;
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn ideogram_source_oracle_covers_exact_claims() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/ideogram/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 2);
        assert_eq!(ROUTES.len(), 3);
        assert!(ALIASES.is_empty());
        Ok(())
    }
    #[test]
    fn ideogram_v3_defaults_ratio_resolution_and_backward_speed_match_source()
    -> Result<(), Box<dyn Error>> {
        let plan = request(&input(
            "IdeogramV3",
            json!({"prompt":"","rendering_speed":"BALANCED","aspect_ratio":"1:1"}),
        )?)?;
        assert_eq!(plan.body, json!({"prompt":"","rendering_speed":"DEFAULT"}));
        assert!(plan.multipart);
        assert!(plan.files.is_empty());
        let plan = request(&input(
            "IdeogramV3",
            json!({"prompt":"source","resolution":"512x1536","aspect_ratio":"3:1","seed":42,"num_images":8,"magic_prompt_option":"ON"}),
        )?)?;
        assert_eq!(
            plan.body,
            json!({"prompt":"source","rendering_speed":"DEFAULT","resolution":"512x1536","seed":42,"num_images":8,"magic_prompt":"ON"})
        );
        assert_eq!(
            plan.form_order,
            [
                "magic_prompt",
                "num_images",
                "prompt",
                "rendering_speed",
                "resolution",
                "seed"
            ]
        );
        Ok(())
    }
    #[test]
    fn ideogram_v4_preserves_prompt_splits_resolution_and_omits_seed() -> Result<(), Box<dyn Error>>
    {
        let mut value = v4()?;
        value
            .arguments
            .insert("resolution".into(), json!("2048x2048 (1:1)"));
        let plan = request(&value)?;
        assert_eq!(
            plan.body,
            json!({"text_prompt":"  source  ","resolution":"2048x2048","rendering_speed":"DEFAULT"})
        );
        assert!(!plan.multipart);
        assert!(plan.files.is_empty());
        assert!(plan.body.get("seed").is_none());
        Ok(())
    }
    #[test]
    fn ideogram_edit_and_character_wire_files_preserve_source_order() -> Result<(), Box<dyn Error>>
    {
        let mut value = v3()?;
        for name in ["image", "mask", "character_image", "character_mask"] {
            value.arguments.insert(name.into(), json!(name));
        }
        let plan = request(&value)?;
        assert_eq!(
            plan.files
                .iter()
                .map(|part| part.name.as_str())
                .collect::<Vec<_>>(),
            [
                "image",
                "mask",
                "character_reference_images",
                "character_mask_binary"
            ]
        );
        assert!(plan.body.get("style_type").is_none());
        let mut transport = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        let mut owner = Owner::default();
        invoke(
            &value,
            credentials(),
            100,
            &mut transport,
            &mut owner,
            &encoder(),
        )?;
        assert_eq!(owner.calls.len(), 2);
        assert_eq!(
            owner.calls.first().ok_or("first")?.2,
            PreparationKind::Character
        );
        assert_eq!(owner.calls.get(1).ok_or("second")?.2, PreparationKind::Edit);
        assert!(owner.calls.iter().all(|call| call.3 == MASK_POLICY));
        let body = transport.bodies.first().ok_or("body")?;
        let text = String::from_utf8_lossy(body);
        let mut previous = 0;
        for name in [
            "image",
            "mask",
            "character_reference_images",
            "character_mask_binary",
        ] {
            let offset = text.find(&format!("name=\"{name}\"")).ok_or("file field")?;
            assert!(offset >= previous);
            previous = offset;
        }
        assert!(body.windows(png().len()).any(|bytes| bytes == png()));
        assert!(
            body.windows(mask_png().len())
                .any(|bytes| bytes == mask_png())
        );
        assert!(text.contains("Content-Type: application/octet-stream"));
        assert!(!text.contains("character_reference_images_mask"));
        Ok(())
    }
    #[test]
    fn ideogram_character_preparation_precedes_incomplete_edit_failure()
    -> Result<(), Box<dyn Error>> {
        let value = input(
            "IdeogramV3",
            json!({"prompt":"source","image":"image","character_image":"reference","character_mask":"mask"}),
        )?;
        let mut owner = Owner::default();
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(
                &value,
                credentials(),
                100,
                &mut transport,
                &mut owner,
                &encoder()
            ),
            Err(IdeogramError::Input(
                "Ideogram V3 image editing requires both an image AND a mask"
            ))
        ));
        assert_eq!(owner.calls.len(), 1);
        assert!(transport.heads.is_empty());
        Ok(())
    }
    #[test]
    fn ideogram_no_file_v3_still_uses_checked_multipart() -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        let result = invoke(
            &v3()?,
            credentials(),
            100,
            &mut transport,
            &mut Owner::default(),
            &encoder(),
        )?;
        assert_eq!(result.response_bytes, serde_json::to_vec(&response())?);
        assert_eq!(result.receipt, [7, 8, 9]);
        assert_eq!(
            transport.content_types.first().ok_or("content type")?.1,
            "multipart/form-data; boundary=source-boundary"
        );
        let body = String::from_utf8(transport.bodies.first().ok_or("body")?.clone())?;
        assert!(body.starts_with("--source-boundary\r\n"));
        assert!(
            body.contains(
                "Content-Type: text/plain; charset=utf-8\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nsource"
            )
        );
        assert!(body.ends_with("--source-boundary--\r\n"));
        Ok(())
    }
    #[test]
    fn ideogram_raw_bytes_receipts_order_and_vendor_cost_are_exact() -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        let result = invoke(
            &v4()?,
            credentials(),
            100,
            &mut transport,
            &mut Owner::default(),
            &encoder(),
        )?;
        assert_eq!(result.color_mode, "RGB");
        assert_eq!(result.images.len(), 2);
        assert_eq!(
            result.images.first().ok_or("image")?.url,
            "https://fixture.invalid/first.png"
        );
        assert_eq!(
            result.images.get(1).ok_or("image")?.url,
            "https://fixture.invalid/second.png"
        );
        assert_eq!(
            transport.costs.first().ok_or("cost")?.operation,
            "ideogram.generation.maximum"
        );
        assert_eq!(transport.costs.len(), 1);
        assert!(
            transport
                .heads
                .iter()
                .skip(1)
                .all(|head| head.secret_id.is_none() && head.headers.len() == 1)
        );
        assert!(transport.content_types.is_empty());
        Ok(())
    }
    #[test]
    fn ideogram_hidden_auth_and_download_secret_isolation() -> Result<(), Box<dyn Error>> {
        let head = request_head(
            ROUTES.first().ok_or("route")?.path,
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.auth_token"));
        assert_eq!(
            head.headers
                .iter()
                .map(|header| header.name.as_str())
                .collect::<Vec<_>>(),
            [
                "Accept",
                "Authorization",
                "Comfy-Env",
                "Comfy-Usage-Source",
                "Content-Type"
            ]
        );
        let head = request_head(
            "/proxy/ideogram/ideogram-v4/generate",
            HttpMethod::Post,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.api_key"));
        assert_eq!(head.headers.get(1).ok_or("api header")?.name, "X-API-KEY");
        assert!(
            request_head(
                "/proxy/ideogram/ideogram-v4/generate",
                HttpMethod::Post,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                false
            )
            .is_err()
        );
        for endpoint in [
            "http://fixture.invalid/image",
            "https://user:password@fixture.invalid/image",
            "//fixture.invalid/image",
            "https://fixture.invalid/image#fragment",
        ] {
            assert!(request_head(endpoint, HttpMethod::Get, credentials(), true).is_err());
        }
        Ok(())
    }
    #[test]
    fn ideogram_cost_native_and_content_type_gates_send_no_body() -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            deny_cost: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &v3()?,
                credentials(),
                100,
                &mut transport,
                &mut Owner::default(),
                &encoder()
            ),
            Err(IdeogramError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        let value = input(
            "IdeogramV3",
            json!({"prompt":"source","image":"image","mask":"mask"}),
        )?;
        for invalid in [false, true] {
            let mut owner = Owner {
                fail: !invalid,
                invalid,
                ..Owner::default()
            };
            let mut transport = Transport::default();
            assert!(
                invoke(
                    &value,
                    credentials(),
                    100,
                    &mut transport,
                    &mut owner,
                    &encoder()
                )
                .is_err()
            );
            assert!(transport.bodies.iter().all(Vec::is_empty));
        }
        let mut transport = Transport {
            fail_content_type: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &v3()?,
                credentials(),
                100,
                &mut transport,
                &mut Owner::default(),
                &encoder()
            ),
            Err(IdeogramError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn ideogram_source_preflight_and_response_failures_are_exact() -> Result<(), Box<dyn Error>> {
        for case in oracle()?["preflight_errors"]
            .as_array()
            .ok_or("preflight")?
        {
            let value: Input = serde_json::from_value(case["input"].clone())?;
            assert_eq!(
                request(&value).err().ok_or("error")?.to_string(),
                case["message"].as_str().ok_or("message")?
            );
        }
        for data in [json!({}), json!({"data":null}), json!({"data":[]})] {
            assert_eq!(
                image_urls(&serde_json::to_vec(&data)?)
                    .err()
                    .ok_or("error")?
                    .to_string(),
                "No images were generated in the response"
            );
        }
        for data in [json!({"data":[{"url":null}]}), json!({"data":[{"url":""}]})] {
            assert_eq!(
                image_urls(&serde_json::to_vec(&data)?)
                    .err()
                    .ok_or("error")?
                    .to_string(),
                "No image URLs were generated in the response"
            );
        }
        for data in [
            json!({"data":[{"url":7}]}),
            json!({"data":[{"url":"https://fixture.invalid/a","is_image_safe":"bad"}]}),
            json!({"data":[{"url":"https://fixture.invalid/a","seed":"bad"}]}),
        ] {
            assert!(image_urls(&serde_json::to_vec(&data)?).is_err());
        }
        Ok(())
    }
    #[test]
    fn ideogram_response_safety_matches_original_schema_coercion() -> Result<(), Box<dyn Error>> {
        for (value, expected) in [
            (json!(false), Some(false)),
            (json!(true), Some(true)),
            (Value::Null, None),
            (json!(0), Some(false)),
            (json!(1), Some(true)),
            (json!(0.0), Some(false)),
            (json!(-0.0), Some(false)),
            (json!(1.0), Some(true)),
        ] {
            let raw = serde_json::to_vec(&json!({"is_image_safe": value}))?;
            let datum: ImageDatum = serde_json::from_slice(&raw)?;
            assert_eq!(datum.is_image_safe, expected);
        }
        for (tokens, expected) in [
            (["0", "f", "n", "no", "off", "false"], false),
            (["1", "t", "y", "on", "yes", "true"], true),
        ] {
            for token in tokens {
                for value in [token.to_owned(), token.to_ascii_uppercase()] {
                    let raw = serde_json::to_vec(&json!({"is_image_safe": value}))?;
                    let datum: ImageDatum = serde_json::from_slice(&raw)?;
                    assert_eq!(datum.is_image_safe, Some(expected));
                }
            }
        }
        for (value, expected) in [("TrUe", true), ("FaLsE", false)] {
            let datum: ImageDatum = serde_json::from_value(json!({"is_image_safe": value}))?;
            assert_eq!(datum.is_image_safe, Some(expected));
        }
        let omitted: ImageDatum = serde_json::from_value(json!({}))?;
        assert_eq!(omitted.is_image_safe, None);
        Ok(())
    }
    #[test]
    fn ideogram_response_safety_preserves_raw_receipts_and_download_order()
    -> Result<(), Box<dyn Error>> {
        let mut payload = response();
        let data = payload
            .get_mut("data")
            .and_then(Value::as_array_mut)
            .ok_or("response data absent")?;
        for (position, safety) in [(0, json!("FaLsE")), (2, json!(1.0))] {
            data.get_mut(position)
                .and_then(Value::as_object_mut)
                .ok_or("response image datum absent")?
                .insert("is_image_safe".into(), safety);
        }
        let raw = serde_json::to_vec_pretty(&payload)?;
        for value in [v3()?, v4()?] {
            let mut sequence = events(200, raw.clone(), vec![]);
            sequence.extend(events(200, png(), vec![]));
            sequence.extend(events(200, png(), vec![]));
            let mut transport = Transport {
                events: sequence,
                ..Transport::default()
            };
            let mut owner = Owner::default();
            let result = invoke(
                &value,
                credentials(),
                100,
                &mut transport,
                &mut owner,
                &encoder(),
            )?;
            assert_eq!(result.response_bytes, raw);
            assert_eq!(result.receipt, [7, 8, 9]);
            assert_eq!(result.color_mode, IMAGE_COLOR_MODE);
            assert_eq!(
                result
                    .images
                    .iter()
                    .map(|image| image.url.as_str())
                    .collect::<Vec<_>>(),
                [
                    "https://fixture.invalid/first.png",
                    "https://fixture.invalid/second.png"
                ]
            );
            for image in &result.images {
                assert_eq!(image.bytes, png());
                assert_eq!(image.receipt, [7, 8, 9]);
            }
            assert_eq!(transport.heads.len(), 3);
            assert_eq!(transport.costs.len(), 1);
            assert!(transport.events.is_empty());
            assert!(owner.calls.is_empty());
            for head in transport.heads.iter().skip(1) {
                assert_eq!(head.method, HttpMethod::Get);
                assert_eq!(head.secret_id, None);
            }
        }
        Ok(())
    }
    #[test]
    fn ideogram_malformed_response_safety_stops_before_download() -> Result<(), Box<dyn Error>> {
        for safety in [
            json!(2),
            json!(-1),
            json!(0.5),
            json!(" true"),
            json!("false "),
            json!("\t1\n"),
            json!("bad"),
            json!(""),
            json!("１"),
            json!("ＴＲＵＥ"),
            json!([]),
            json!({}),
        ] {
            for value in [v3()?, v4()?] {
                let payload = json!({"data":[{"url":"https://fixture.invalid/image","is_image_safe":safety}]});
                let mut transport = Transport {
                    events: events(200, serde_json::to_vec(&payload)?, vec![]),
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                assert!(matches!(
                    invoke(
                        &value,
                        credentials(),
                        100,
                        &mut transport,
                        &mut owner,
                        &encoder()
                    ),
                    Err(IdeogramError::Json(_))
                ));
                assert_eq!(transport.heads.len(), 1);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
            }
        }
        Ok(())
    }
    #[test]
    fn ideogram_response_seed_text_matches_retained_source_schema() -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        let mut covered = 0;
        for case in source["api_schema_probe"]
            .as_array()
            .ok_or("source schema probes absent")?
        {
            let payload = &case["payload"];
            if case["accepted"] != true
                || !(payload["created"].is_string() || payload["created"].is_null())
            {
                continue;
            }
            let response: GenerateResponse = serde_json::from_value(payload.clone())?;
            let data = response.data.ok_or("source response data absent")?;
            let first = data.first().ok_or("source image datum absent")?;
            assert_eq!(payload["data"][0]["seed"], "42");
            assert_eq!(first.seed, case["parsed"]["data"][0]["seed"].as_i64());
            assert_eq!(first.seed, Some(42));
            assert_eq!(
                image_urls(&serde_json::to_vec(payload)?)?,
                ["https://fixture.invalid/image"]
            );
            covered += 1;
        }
        assert_eq!(covered, 2);
        for (seed, expected) in [
            (json!(42), Some(42)),
            (json!(-42), Some(-42)),
            (json!("-42"), Some(-42)),
            (json!("+42"), Some(42)),
            (Value::Null, None),
        ] {
            let datum: ImageDatum = serde_json::from_value(json!({"seed":seed}))?;
            assert_eq!(datum.seed, expected);
        }
        let omitted: ImageDatum = serde_json::from_value(json!({}))?;
        assert_eq!(omitted.seed, None);
        Ok(())
    }
    #[test]
    fn ideogram_response_seed_text_preserves_raw_receipts_and_download_order()
    -> Result<(), Box<dyn Error>> {
        let mut payload = response();
        let data = payload
            .get_mut("data")
            .and_then(Value::as_array_mut)
            .ok_or("response data absent")?;
        for position in [0, 2] {
            data.get_mut(position)
                .and_then(Value::as_object_mut)
                .ok_or("response image datum absent")?
                .insert("seed".into(), json!("42"));
        }
        let raw = serde_json::to_vec_pretty(&payload)?;
        for value in [v3()?, v4()?] {
            let mut sequence = events(200, raw.clone(), vec![]);
            sequence.extend(events(200, png(), vec![]));
            sequence.extend(events(200, png(), vec![]));
            let mut transport = Transport {
                events: sequence,
                ..Transport::default()
            };
            let mut owner = Owner::default();
            let result = invoke(
                &value,
                credentials(),
                100,
                &mut transport,
                &mut owner,
                &encoder(),
            )?;
            assert_eq!(result.response_bytes, raw);
            assert_eq!(result.receipt, [7, 8, 9]);
            assert_eq!(result.color_mode, IMAGE_COLOR_MODE);
            assert_eq!(
                result
                    .images
                    .iter()
                    .map(|image| image.url.as_str())
                    .collect::<Vec<_>>(),
                [
                    "https://fixture.invalid/first.png",
                    "https://fixture.invalid/second.png"
                ]
            );
            for image in &result.images {
                assert_eq!(image.bytes, png());
                assert_eq!(image.receipt, [7, 8, 9]);
            }
            assert_eq!(transport.heads.len(), 3);
            assert_eq!(transport.costs.len(), 1);
            assert!(transport.events.is_empty());
            assert!(owner.calls.is_empty());
            for head in transport.heads.iter().skip(1) {
                assert_eq!(head.method, HttpMethod::Get);
                assert_eq!(head.secret_id, None);
            }
        }
        Ok(())
    }
    #[test]
    fn ideogram_malformed_response_seed_stops_before_download() -> Result<(), Box<dyn Error>> {
        for seed in [
            json!("bad"),
            json!("42.5"),
            json!(42.5),
            json!([]),
            json!({}),
        ] {
            for value in [v3()?, v4()?] {
                let payload = json!({"data":[{"url":"https://fixture.invalid/image","seed":seed}]});
                let mut transport = Transport {
                    events: events(200, serde_json::to_vec(&payload)?, vec![]),
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                assert!(matches!(
                    invoke(
                        &value,
                        credentials(),
                        100,
                        &mut transport,
                        &mut owner,
                        &encoder()
                    ),
                    Err(IdeogramError::Json(_))
                ));
                assert_eq!(transport.heads.len(), 1);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
            }
        }
        Ok(())
    }
    #[test]
    fn ideogram_retry_budgets_delay_and_handle_identity_are_bounded() -> Result<(), Box<dyn Error>>
    {
        let head = request_head(
            "/proxy/ideogram/ideogram-v4/generate",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut sequence = events(
            503,
            vec![],
            vec![Header {
                name: "Retry-After".into(),
                value: "2".into(),
            }],
        );
        sequence.extend(events(200, b"{}".to_vec(), vec![]));
        let mut transport = Transport {
            events: sequence,
            ..Transport::default()
        };
        exchange_with_policy(&head, b"{}", None, 1024, GENERATION_RETRY, &mut transport)?;
        assert_eq!(transport.delays, [2000]);
        assert_eq!(transport.bodies, [b"{}".to_vec(), b"{}".to_vec()]);
        let mut sequence = events(503, vec![], vec![]);
        sequence.extend(events(503, vec![], vec![]));
        let mut transport = Transport {
            events: sequence,
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, b"{}", None, 1024, GENERATION_RETRY, &mut transport),
            Err(IdeogramError::Http { status: 503, .. })
        ));
        assert_eq!(transport.delays.len(), 1);
        let mut transport = Transport {
            events: events(503, vec![], vec![]),
            same_retry_handle: true,
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, b"{}", None, 1024, GENERATION_RETRY, &mut transport),
            Err(IdeogramError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn ideogram_rate_and_download_retries_follow_separate_source_budgets()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/ideogram/ideogram-v4/generate",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut sequence = VecDeque::new();
        for _ in 0..17 {
            sequence.extend(events(429, vec![], vec![]));
        }
        let mut transport = Transport {
            events: sequence,
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, &[], None, 1024, GENERATION_RETRY, &mut transport),
            Err(IdeogramError::Http { status: 429, .. })
        ));
        assert_eq!(transport.delays.len(), 16);
        assert_eq!(transport.delays.last().copied(), Some(30000));
        let mut sequence = VecDeque::new();
        for _ in 0..6 {
            sequence.extend(events(
                429,
                vec![],
                vec![Header {
                    name: "Retry-After".into(),
                    value: "999".into(),
                }],
            ));
        }
        let mut transport = Transport {
            events: sequence,
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, &[], None, 1024, DOWNLOAD_RETRY, &mut transport),
            Err(IdeogramError::Http { status: 429, .. })
        ));
        assert_eq!(transport.delays, [1000, 2000, 4000, 8000, 16000]);
        Ok(())
    }
    #[test]
    fn ideogram_response_order_chunk_receipt_and_body_limits_fail_closed()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/ideogram/ideogram-v4/generate",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        for sequence in [
            vec![Ok(ResponseEvent::Chunk(vec![1]))],
            vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Completed(vec![])),
            ],
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
                Ok(ResponseEvent::Chunk(vec![0; 9])),
            ],
        ] {
            let mut transport = Transport {
                events: sequence.into(),
                ..Transport::default()
            };
            assert!(
                exchange_with_policy(&head, &[], None, 8, GENERATION_RETRY, &mut transport)
                    .is_err()
            );
        }
        let mut transport = Transport {
            events: vec![Ok(ResponseEvent::Head {
                status: 200,
                headers: vec![
                    Header {
                        name: "a".into(),
                        value: "x".repeat(9000),
                    },
                    Header {
                        name: "b".into(),
                        value: "y".repeat(9000),
                    },
                ],
            })]
            .into(),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, &[], None, 1024, GENERATION_RETRY, &mut transport),
            Err(IdeogramError::Transport(ConformanceError::HeaderLimit))
        ));
        let fields = [MultipartField {
            name: "field".into(),
            file_name: None,
            media_type: None,
            bytes: vec![],
        }];
        assert!(
            CanonicalMultipartEncoder {
                boundary: "bad\r\nheader".into()
            }
            .encode(&fields)
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn ideogram_cancellation_worker_loss_and_fresh_invocation_preserve_supervisor_ownership()
    -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &v4()?,
                credentials(),
                100,
                &mut transport,
                &mut Owner::default(),
                &encoder()
            ),
            Err(IdeogramError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        let mut transport = Transport {
            events: vec![Err(ConformanceError::WorkerLost)].into(),
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &v4()?,
                credentials(),
                100,
                &mut transport,
                &mut Owner::default(),
                &encoder()
            ),
            Err(IdeogramError::Transport(ConformanceError::WorkerLost))
        ));
        assert!(transport.delays.is_empty());
        let mut fresh = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        assert_eq!(
            invoke(
                &v4()?,
                credentials(),
                100,
                &mut fresh,
                &mut Owner::default(),
                &encoder()
            )?
            .images
            .len(),
            2
        );
        Ok(())
    }
    #[test]
    fn ideogram_real_rgb_decode_rejection_publishes_no_placeholder() -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            events: success_events()?,
            ..Transport::default()
        };
        let mut decoder = RejectDecoder::default();
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &v4()?,
                credentials(),
                100,
                &mut transport,
                &mut MaterializationOwners {
                    input: &mut Owner::default(),
                    encoder: &encoder(),
                    materializer: &mut decoder,
                    sink: &mut sink
                }
            ),
            Err(IdeogramError::Transport(ConformanceError::HostFailure))
        ));
        assert!(decoder.seen);
        assert_eq!(sink.batches, 0);
        Ok(())
    }

    #[test]
    fn ideogram_v4_python_whitespace_rejects_before_request_authority() -> Result<(), Box<dyn Error>>
    {
        for prompt in [
            "",
            "\u{1c}",
            "\u{1d}",
            "\u{1e}",
            "\u{1f}",
            "\u{1c}\u{1f}",
            " \t\u{85}\u{a0}\u{3000}",
        ] {
            let mut value = v4()?;
            value.arguments.insert("prompt".into(), json!(prompt));
            assert!(matches!(
                request(&value),
                Err(IdeogramError::Input(
                    "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long."
                ))
            ));
            let mut transport = Transport {
                deny_cost: true,
                ..Transport::default()
            };
            let mut owner = Owner {
                fail: true,
                ..Owner::default()
            };
            assert!(matches!(
                invoke(
                    &value,
                    credentials(),
                    100,
                    &mut transport,
                    &mut owner,
                    &encoder()
                ),
                Err(IdeogramError::Input(
                    "Field 'prompt' cannot be shorter than 1 characters; was 0 characters long."
                ))
            ));
            assert!(transport.heads.is_empty());
            assert!(transport.costs.is_empty());
            assert!(transport.bodies.is_empty());
            assert!(owner.calls.is_empty());
        }
        Ok(())
    }
    #[test]
    fn ideogram_v4_python_whitespace_preserves_raw_json_and_source_unbounded_prompt()
    -> Result<(), Box<dyn Error>> {
        for prompt in [
            "\u{1c} scene\u{1f}inside \u{1f}".into(),
            "\u{200b}".into(),
            "\u{feff}".into(),
            format!("\u{1c}{}\u{1f}", "x".repeat(10000)),
            "x".repeat(10001),
        ] {
            let mut value = v4()?;
            value.arguments.insert("prompt".into(), json!(prompt));
            let plan = request(&value)?;
            assert_eq!(plan.body["text_prompt"], value.arguments["prompt"]);
            let mut transport = Transport {
                events: success_events()?,
                ..Transport::default()
            };
            let mut owner = Owner {
                fail: true,
                ..Owner::default()
            };
            let result = invoke(
                &value,
                credentials(),
                100,
                &mut transport,
                &mut owner,
                &encoder(),
            )?;
            assert_eq!(
                transport.bodies.first().ok_or("generation body absent")?,
                &serde_json::to_vec(&plan.body)?
            );
            assert_eq!(transport.costs.len(), 1);
            assert_eq!(
                transport.costs.first().ok_or("cost absent")?.operation,
                "ideogram.generation.maximum"
            );
            assert_eq!(transport.heads.len(), 3);
            assert!(transport.content_types.is_empty());
            assert!(owner.calls.is_empty());
            assert_eq!(result.images.len(), 2);
            assert_eq!(result.color_mode, IMAGE_COLOR_MODE);
        }
        Ok(())
    }
    #[test]
    fn ideogram_v4_whitespace_retains_cancellation_cost_and_v3_boundaries()
    -> Result<(), Box<dyn Error>> {
        let mut value = v4()?;
        value.arguments.insert("prompt".into(), json!("\u{1c}"));
        let mut cancelled = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &value,
                credentials(),
                100,
                &mut cancelled,
                &mut Owner::default(),
                &encoder()
            ),
            Err(IdeogramError::Transport(ConformanceError::Cancelled))
        ));
        assert!(cancelled.heads.is_empty());
        assert!(cancelled.costs.is_empty());
        assert!(cancelled.bodies.is_empty());
        let mut denied = Transport {
            deny_cost: true,
            ..Transport::default()
        };
        value
            .arguments
            .insert("prompt".into(), json!("\u{1c}source\u{1f}"));
        let mut owner = Owner {
            fail: true,
            ..Owner::default()
        };
        assert!(matches!(
            invoke(
                &value,
                credentials(),
                100,
                &mut denied,
                &mut owner,
                &encoder()
            ),
            Err(IdeogramError::Transport(ConformanceError::InvalidCost))
        ));
        assert_eq!(denied.heads.len(), 1);
        assert_eq!(denied.costs.len(), 1);
        assert_eq!(denied.bodies.len(), 1);
        assert!(denied.bodies.iter().all(Vec::is_empty));
        assert!(owner.calls.is_empty());
        let mut unauthenticated = Transport::default();
        assert!(
            invoke(
                &value,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                100,
                &mut unauthenticated,
                &mut owner,
                &encoder()
            )
            .is_err()
        );
        assert!(unauthenticated.heads.is_empty());
        assert!(unauthenticated.costs.is_empty());
        assert!(unauthenticated.bodies.is_empty());
        for prompt in ["", "\u{1c}", "\u{1f}", "\u{200b}", "\u{feff}"] {
            let mut old = v3()?;
            old.arguments.insert("prompt".into(), json!(prompt));
            let plan = request(&old)?;
            assert_eq!(plan.body["prompt"], old.arguments["prompt"]);
            assert!(plan.multipart);
        }
        Ok(())
    }
    #[derive(Deserialize)]
    struct CreatedSourceCase {
        id: String,
        raw_json_utf8_hex: String,
        json_nonfinite_extension: bool,
        schema_admitted: bool,
        datetime: Option<CreatedCalendarExpected>,
        nonfinite_f64: Option<String>,
    }
    #[derive(Deserialize)]
    struct CreatedCalendarExpected {
        year: u16,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        microsecond: u32,
        utc_offset_seconds: Option<f64>,
    }
    fn created_source_cases() -> Result<Vec<CreatedSourceCase>, Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/ideogram/created-datetime-source-observations.json"
        )))?;
        let cases: Vec<CreatedSourceCase> = serde_json::from_value(fixture["cases"].clone())?;
        assert_eq!(cases.len(), 63);
        Ok(cases)
    }
    fn created_source_bytes(case: &CreatedSourceCase) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut pairs = case.raw_json_utf8_hex.as_bytes().chunks_exact(2);
        let bytes = pairs
            .by_ref()
            .map(|pair| {
                let digits = std::str::from_utf8(pair)?;
                Ok(u8::from_str_radix(digits, 16)?)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        if !pairs.remainder().is_empty() {
            return Err("source payload hex has an incomplete byte".into());
        }
        Ok(bytes)
    }
    fn assert_created_calendar(
        actual: Option<speedate::DateTime>,
        expected: Option<&CreatedCalendarExpected>,
    ) -> Result<(), Box<dyn Error>> {
        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                assert_eq!(actual.date.year, expected.year);
                assert_eq!(actual.date.month, expected.month);
                assert_eq!(actual.date.day, expected.day);
                assert_eq!(actual.time.hour, expected.hour);
                assert_eq!(actual.time.minute, expected.minute);
                assert_eq!(actual.time.second, expected.second);
                assert_eq!(actual.time.microsecond, expected.microsecond);
                assert_eq!(
                    actual.time.tz_offset.map(f64::from),
                    expected.utc_offset_seconds
                );
            }
            (None, None) => {}
            _ => return Err("created presence differs from original Source".into()),
        }
        Ok(())
    }
    #[test]
    fn ideogram_created_calendar_matches_original_source_matrix() -> Result<(), Box<dyn Error>> {
        let mut admitted = 0;
        let mut typed = 0;
        let mut extensions = 0;
        for case in created_source_cases()? {
            let raw = created_source_bytes(&case)?;
            let parsed = serde_json::from_slice::<GenerateResponse>(&raw);
            assert_eq!(parsed.is_ok(), case.schema_admitted, "{}", case.id);
            if case.schema_admitted {
                assert_created_calendar(parsed?.created, case.datetime.as_ref())?;
                admitted += 1;
                typed += usize::from(case.datetime.is_some());
                assert_eq!(image_urls(&raw)?, ["https://fixture.invalid/image"]);
            } else {
                assert!(image_urls(&raw).is_err(), "{}", case.id);
            }
            if case.json_nonfinite_extension {
                let value = match case.nonfinite_f64.as_deref() {
                    Some("NaN") => f64::NAN,
                    Some("Infinity") => f64::INFINITY,
                    Some("-Infinity") => f64::NEG_INFINITY,
                    _ => return Err("unrecognized Source nonfinite input".into()),
                };
                let deserializer =
                    serde::de::value::F64Deserializer::<serde::de::value::Error>::new(value);
                assert!(
                    ResponseCreated::deserialize(deserializer).is_err(),
                    "{}",
                    case.id
                );
                extensions += 1;
            }
        }
        assert_eq!((admitted, typed, extensions), (35, 32, 3));
        Ok(())
    }
    #[test]
    fn ideogram_created_source_rejections_stop_before_download() -> Result<(), Box<dyn Error>> {
        let mut rejected = 0;
        for case in created_source_cases()? {
            if case.schema_admitted {
                continue;
            }
            for value in [v3()?, v4()?] {
                let mut transport = Transport {
                    events: events(200, created_source_bytes(&case)?, vec![]),
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                assert!(
                    matches!(
                        invoke(
                            &value,
                            credentials(),
                            100,
                            &mut transport,
                            &mut owner,
                            &encoder()
                        ),
                        Err(IdeogramError::Json(_))
                    ),
                    "{}",
                    case.id
                );
                assert_eq!(transport.heads.len(), 1);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
            }
            rejected += 1;
        }
        assert_eq!(rejected, 28);
        Ok(())
    }
    #[test]
    fn ideogram_created_preserves_raw_receipts_and_download_order() -> Result<(), Box<dyn Error>> {
        let mut admitted = 0;
        for case in created_source_cases()? {
            if !case.schema_admitted {
                continue;
            }
            let source_payload: Value = serde_json::from_slice(&created_source_bytes(&case)?)?;
            let mut payload = response();
            if let Some(created) = source_payload.get("created") {
                payload
                    .as_object_mut()
                    .ok_or("response object absent")?
                    .insert("created".into(), created.clone());
            }
            let raw = serde_json::to_vec_pretty(&payload)?;
            for value in [v3()?, v4()?] {
                let mut sequence = events(200, raw.clone(), vec![]);
                sequence.extend(events(200, png(), vec![]));
                sequence.extend(events(200, png(), vec![]));
                let mut transport = Transport {
                    events: sequence,
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                let result = invoke(
                    &value,
                    credentials(),
                    100,
                    &mut transport,
                    &mut owner,
                    &encoder(),
                )?;
                assert_eq!(result.response_bytes, raw);
                assert_eq!(result.receipt, [7, 8, 9]);
                assert_eq!(result.color_mode, IMAGE_COLOR_MODE);
                assert_eq!(
                    result
                        .images
                        .iter()
                        .map(|image| image.url.as_str())
                        .collect::<Vec<_>>(),
                    [
                        "https://fixture.invalid/first.png",
                        "https://fixture.invalid/second.png"
                    ]
                );
                for image in &result.images {
                    assert_eq!(image.bytes, png());
                    assert_eq!(image.receipt, [7, 8, 9]);
                }
                assert_eq!(transport.heads.len(), 3);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
                for head in transport.heads.iter().skip(1) {
                    assert_eq!(head.method, HttpMethod::Get);
                    assert_eq!(head.secret_id, None);
                    assert_eq!(head.headers.len(), 1);
                }
            }
            admitted += 1;
        }
        assert_eq!(admitted, 35);
        Ok(())
    }

    #[derive(Deserialize)]
    struct SeedSourceCase {
        id: String,
        raw_json_utf8_hex: String,
        schema_admitted: bool,
        typed_seed_decimal: Option<String>,
        parity_covered: bool,
    }
    fn seed_source_cases() -> Result<Vec<SeedSourceCase>, Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/ideogram/seed-source-observations.json"
        )))?;
        let cases: Vec<SeedSourceCase> = serde_json::from_value(fixture["cases"].clone())?;
        assert_eq!(cases.len(), 40);
        Ok(cases)
    }
    fn seed_source_bytes(case: &SeedSourceCase) -> Result<Vec<u8>, Box<dyn Error>> {
        let mut pairs = case.raw_json_utf8_hex.as_bytes().chunks_exact(2);
        let bytes = pairs
            .by_ref()
            .map(|pair| {
                let digits = std::str::from_utf8(pair)?;
                Ok(u8::from_str_radix(digits, 16)?)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        if !pairs.remainder().is_empty() {
            return Err("source seed payload hex has an incomplete byte".into());
        }
        Ok(bytes)
    }
    #[test]
    fn ideogram_seed_matches_bounded_original_source_matrix() -> Result<(), Box<dyn Error>> {
        let mut compared = 0;
        let mut admitted = 0;
        let mut open = 0;
        for case in seed_source_cases()? {
            if !case.parity_covered {
                open += 1;
                continue;
            }
            let raw = seed_source_bytes(&case)?;
            let parsed = serde_json::from_slice::<GenerateResponse>(&raw);
            assert_eq!(parsed.is_ok(), case.schema_admitted, "{}", case.id);
            if case.schema_admitted {
                let response = parsed?;
                let datum = response
                    .data
                    .as_deref()
                    .and_then(|data| data.first())
                    .ok_or("Source seed response data absent")?;
                let expected = case
                    .typed_seed_decimal
                    .as_deref()
                    .map(str::parse::<i64>)
                    .transpose()?;
                assert_eq!(datum.seed, expected, "{}", case.id);
                assert_eq!(image_urls(&raw)?, ["https://fixture.invalid/image"]);
                admitted += 1;
            } else {
                assert!(image_urls(&raw).is_err(), "{}", case.id);
            }
            compared += 1;
        }
        assert_eq!((compared, admitted, open), (27, 15, 13));
        Ok(())
    }
    #[test]
    fn ideogram_seed_source_rejections_stop_before_download() -> Result<(), Box<dyn Error>> {
        let mut rejected = 0;
        for case in seed_source_cases()? {
            if case.schema_admitted {
                continue;
            }
            for value in [v3()?, v4()?] {
                let mut transport = Transport {
                    events: events(200, seed_source_bytes(&case)?, vec![]),
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                assert!(
                    matches!(
                        invoke(
                            &value,
                            credentials(),
                            100,
                            &mut transport,
                            &mut owner,
                            &encoder()
                        ),
                        Err(IdeogramError::Json(_))
                    ),
                    "{}",
                    case.id
                );
                assert_eq!(transport.heads.len(), 1);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
            }
            rejected += 1;
        }
        assert_eq!(rejected, 15);
        Ok(())
    }
    #[test]
    fn ideogram_seed_bool_float_preserves_raw_receipts_and_download_order()
    -> Result<(), Box<dyn Error>> {
        let mut admitted = 0;
        for case in seed_source_cases()? {
            if !matches!(
                case.id.as_str(),
                "seed-bool-true"
                    | "seed-bool-false"
                    | "seed-float-integral"
                    | "seed-float-negative-integral"
            ) {
                continue;
            }
            let source_payload: Value = serde_json::from_slice(&seed_source_bytes(&case)?)?;
            let seed = source_payload
                .get("data")
                .and_then(Value::as_array)
                .and_then(|data| data.first())
                .and_then(|datum| datum.get("seed"))
                .ok_or("Source seed absent")?;
            let mut payload = response();
            for datum in payload
                .get_mut("data")
                .and_then(Value::as_array_mut)
                .ok_or("response data absent")?
            {
                if datum.get("url").and_then(Value::as_str).is_some() {
                    datum
                        .as_object_mut()
                        .ok_or("response datum absent")?
                        .insert("seed".into(), seed.clone());
                }
            }
            let raw = serde_json::to_vec_pretty(&payload)?;
            for value in [v3()?, v4()?] {
                let mut sequence = events(200, raw.clone(), vec![]);
                sequence.extend(events(200, png(), vec![]));
                sequence.extend(events(200, png(), vec![]));
                let mut transport = Transport {
                    events: sequence,
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                let result = invoke(
                    &value,
                    credentials(),
                    100,
                    &mut transport,
                    &mut owner,
                    &encoder(),
                )?;
                assert_eq!(result.response_bytes, raw);
                assert_eq!(result.receipt, [7, 8, 9]);
                assert_eq!(result.color_mode, IMAGE_COLOR_MODE);
                assert_eq!(
                    result
                        .images
                        .iter()
                        .map(|image| image.url.as_str())
                        .collect::<Vec<_>>(),
                    [
                        "https://fixture.invalid/first.png",
                        "https://fixture.invalid/second.png"
                    ]
                );
                for image in &result.images {
                    assert_eq!(image.bytes, png());
                    assert_eq!(image.receipt, [7, 8, 9]);
                }
                assert_eq!(transport.heads.len(), 3);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
                for head in transport.heads.iter().skip(1) {
                    assert_eq!(head.method, HttpMethod::Get);
                    assert_eq!(head.secret_id, None);
                    assert_eq!(head.headers.len(), 1);
                }
            }
            admitted += 1;
        }
        assert_eq!(admitted, 4);
        Ok(())
    }
    #[test]
    fn ideogram_seed_text_matches_remaining_original_source_cases() -> Result<(), Box<dyn Error>> {
        let mut compared = 0;
        for case in seed_source_cases()? {
            if !matches!(
                case.id.as_str(),
                "seed-integral-fraction-text"
                    | "seed-negative-integral-fraction-text"
                    | "seed-ascii-whitespace"
                    | "seed-underscore"
                    | "seed-trailing-space"
                    | "seed-unicode-whitespace"
            ) {
                continue;
            }
            assert!(case.schema_admitted, "{}", case.id);
            let raw = seed_source_bytes(&case)?;
            let parsed: GenerateResponse = serde_json::from_slice(&raw)?;
            let datum = parsed
                .data
                .as_deref()
                .and_then(|data| data.first())
                .ok_or("Source seed response data absent")?;
            let expected = case
                .typed_seed_decimal
                .as_deref()
                .map(str::parse::<i64>)
                .transpose()?;
            assert_eq!(datum.seed, expected, "{}", case.id);
            assert_eq!(image_urls(&raw)?, ["https://fixture.invalid/image"]);
            compared += 1;
        }
        assert_eq!(compared, 6);
        Ok(())
    }
    #[test]
    fn ideogram_seed_text_retains_integer_precision_and_rejects_before_download()
    -> Result<(), Box<dyn Error>> {
        for (text, expected) in [
            ("9223372036854775807.0", i64::MAX),
            ("-9223372036854775808.000", i64::MIN),
            ("9_007_199_254_740_993.0", 9_007_199_254_740_993),
        ] {
            let raw = serde_json::to_vec(&serde_json::json!({"data": [{
                "url": "https://fixture.invalid/image", "seed": text,
            }]}))?;
            let parsed: GenerateResponse = serde_json::from_slice(&raw)?;
            assert_eq!(
                parsed
                    .data
                    .as_deref()
                    .and_then(|data| data.first())
                    .ok_or("response datum absent")?
                    .seed,
                Some(expected),
                "{text}"
            );
        }
        for seed in [
            "42.0000000000000001",
            "42.",
            "+_42",
            "-+42",
            "4__2",
            "4.0_0",
            "\u{001c}42",
            "42\u{feff}",
            "9223372036854775808.0",
            "-9223372036854775809.0",
        ] {
            let raw = serde_json::to_vec(&serde_json::json!({"data": [{
                "url": "https://fixture.invalid/image", "seed": seed,
            }]}))?;
            for value in [v3()?, v4()?] {
                let mut transport = Transport {
                    events: events(200, raw.clone(), vec![]),
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                assert!(
                    matches!(
                        invoke(
                            &value,
                            credentials(),
                            100,
                            &mut transport,
                            &mut owner,
                            &encoder()
                        ),
                        Err(IdeogramError::Json(_))
                    ),
                    "{seed:?}"
                );
                assert_eq!(transport.heads.len(), 1);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
            }
        }
        Ok(())
    }
    #[test]
    fn ideogram_seed_text_preserves_raw_receipts_and_download_order() -> Result<(), Box<dyn Error>>
    {
        let mut admitted = 0;
        for case in seed_source_cases()? {
            if !matches!(
                case.id.as_str(),
                "seed-integral-fraction-text"
                    | "seed-negative-integral-fraction-text"
                    | "seed-ascii-whitespace"
                    | "seed-underscore"
                    | "seed-trailing-space"
                    | "seed-unicode-whitespace"
            ) {
                continue;
            }
            let source_payload: Value = serde_json::from_slice(&seed_source_bytes(&case)?)?;
            let seed = source_payload
                .get("data")
                .and_then(Value::as_array)
                .and_then(|data| data.first())
                .and_then(|datum| datum.get("seed"))
                .ok_or("Source seed absent")?;
            let mut payload = response();
            for datum in payload
                .get_mut("data")
                .and_then(Value::as_array_mut)
                .ok_or("response data absent")?
            {
                if datum.get("url").and_then(Value::as_str).is_some() {
                    datum
                        .as_object_mut()
                        .ok_or("response datum absent")?
                        .insert("seed".into(), seed.clone());
                }
            }
            let raw = serde_json::to_vec_pretty(&payload)?;
            for value in [v3()?, v4()?] {
                let mut sequence = events(200, raw.clone(), vec![]);
                sequence.extend(events(200, png(), vec![]));
                sequence.extend(events(200, png(), vec![]));
                let mut transport = Transport {
                    events: sequence,
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                let result = invoke(
                    &value,
                    credentials(),
                    100,
                    &mut transport,
                    &mut owner,
                    &encoder(),
                )?;
                assert_eq!(result.response_bytes, raw);
                assert_eq!(result.receipt, [7, 8, 9]);
                assert_eq!(result.color_mode, IMAGE_COLOR_MODE);
                assert_eq!(
                    result
                        .images
                        .iter()
                        .map(|image| image.url.as_str())
                        .collect::<Vec<_>>(),
                    [
                        "https://fixture.invalid/first.png",
                        "https://fixture.invalid/second.png"
                    ]
                );
                for image in &result.images {
                    assert_eq!(image.bytes, png());
                    assert_eq!(image.receipt, [7, 8, 9]);
                }
                assert_eq!(transport.heads.len(), 3);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
                for head in transport.heads.iter().skip(1) {
                    assert_eq!(head.method, HttpMethod::Get);
                    assert_eq!(head.secret_id, None);
                    assert_eq!(head.headers.len(), 1);
                }
            }
            admitted += 1;
        }
        assert_eq!(admitted, 6);
        Ok(())
    }
    #[test]
    fn ideogram_seed_float_lower_endpoint_preserves_integer_admission_and_precision()
    -> Result<(), Box<dyn Error>> {
        for (token, expected) in [
            ("-9223372036854775808.0", None),
            ("-9.223372036854776e+18", None),
            ("-9223372036854774784.0", Some(-9_223_372_036_854_774_784)),
            ("-9223372036854775808", Some(i64::MIN)),
            ("9007199254740993", Some(9_007_199_254_740_993)),
        ] {
            let raw =
                format!(r#"{{"data":[{{"url":"https://fixture.invalid/image","seed":{token}}}]}}"#);
            let parsed = serde_json::from_slice::<GenerateResponse>(raw.as_bytes());
            if let Some(expected) = expected {
                let response = parsed?;
                let datum = response
                    .data
                    .as_deref()
                    .and_then(|data| data.first())
                    .ok_or("response datum absent")?;
                assert_eq!(datum.seed, Some(expected), "{token}");
                assert_eq!(
                    image_urls(raw.as_bytes())?,
                    ["https://fixture.invalid/image"]
                );
            } else {
                assert!(parsed.is_err(), "{token}");
                assert!(image_urls(raw.as_bytes()).is_err(), "{token}");
            }
        }
        Ok(())
    }
    #[test]
    fn ideogram_seed_float_lower_endpoint_stops_before_download() -> Result<(), Box<dyn Error>> {
        for token in ["-9223372036854775808.0", "-9.223372036854776e+18"] {
            let raw =
                format!(r#"{{"data":[{{"url":"https://fixture.invalid/image","seed":{token}}}]}}"#)
                    .into_bytes();
            for value in [v3()?, v4()?] {
                let mut transport = Transport {
                    events: events(200, raw.clone(), vec![]),
                    ..Transport::default()
                };
                let mut owner = Owner::default();
                assert!(
                    matches!(
                        invoke(
                            &value,
                            credentials(),
                            100,
                            &mut transport,
                            &mut owner,
                            &encoder()
                        ),
                        Err(IdeogramError::Json(_))
                    ),
                    "{token}"
                );
                assert_eq!(transport.heads.len(), 1);
                assert_eq!(transport.costs.len(), 1);
                assert!(transport.events.is_empty());
                assert!(owner.calls.is_empty());
            }
        }
        Ok(())
    }
}
