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
pub const NAMESPACE: &str = "zed.sim.provider.grok";
pub const ALIASES: &[&str] = &["xai"];
pub const SOURCE_SHA256: &str = "b74dd5886965e2f98d8ae46ec207b13c820c4ee366d4b6486c545780b07c19c7";
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
        feature_id: "COMFY-NODE-0212",
        node_identifier: "GrokImageEditNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0213",
        node_identifier: "GrokImageEditNodeV2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0214",
        node_identifier: "GrokImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0215",
        node_identifier: "GrokVideoEditNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0216",
        node_identifier: "GrokVideoExtendNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0217",
        node_identifier: "GrokVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0218",
        node_identifier: "GrokVideoReferenceNode",
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
        feature_id: "COMFY-API-EXT-0211",
        method: "POST",
        path: "/proxy/xai/v1/images/edits",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0212",
        method: "POST",
        path: "/proxy/xai/v1/images/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0213",
        method: "GET",
        path: "/proxy/xai/v1/videos/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0214",
        method: "POST",
        path: "/proxy/xai/v1/videos/edits",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0215",
        method: "POST",
        path: "/proxy/xai/v1/videos/extensions",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0216",
        method: "POST",
        path: "/proxy/xai/v1/videos/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0217",
        method: "GET",
        path: "/proxy/xai/v1/videos/{initial_response.request_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum GrokError {
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
impl fmt::Display for GrokError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Grok HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for GrokError {}
impl From<ConformanceError> for GrokError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for GrokError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, GrokError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/grok/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, GrokError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(GrokError::Input("Unknown Grok node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(GrokError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(GrokError::Input("Unknown Grok source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(GrokError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(GrokError::Input("Missing Grok source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, GrokError> {
    input
        .arguments
        .get(name)
        .ok_or(GrokError::Input("Missing Grok argument"))
}
fn text(value: &Value) -> Result<&str, GrokError> {
    value
        .as_str()
        .ok_or(GrokError::Input("Expected Grok string"))
}
fn integer(value: &Value, maximum: u64) -> Result<u64, GrokError> {
    let number = value
        .as_u64()
        .ok_or(GrokError::Input("Expected Grok unsigned integer"))?;
    if number > maximum {
        return Err(GrokError::Input("Grok integer bound exceeded"));
    }
    Ok(number)
}
fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{001c}'..='\u{001f}')
}
fn prompt(value: &Value) -> Result<&str, GrokError> {
    let text = text(value)?;
    if text.trim_matches(python_whitespace).is_empty() {
        return Err(GrokError::Input("Empty Grok prompt"));
    }
    Ok(text)
}
fn choice<'a>(value: &'a Value, choices: &[&str]) -> Result<&'a str, GrokError> {
    let text = text(value)?;
    if !choices.contains(&text) {
        return Err(GrokError::Input("Unknown Grok source choice"));
    }
    Ok(text)
}
fn endpoint(value: &str) -> Result<bool, GrokError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(GrokError::Response("Invalid Grok response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(GrokError::Response("Invalid Grok response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(GrokError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(GrokError::Response("Invalid Grok endpoint authority"));
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
) -> Result<RequestHead, GrokError> {
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
            return Err(GrokError::Input("Missing Comfy API credentials"));
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

pub trait GrokTransport: ProviderTransport {
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
pub struct VideoInfo {
    pub source: Value,
    pub duration_seconds: Option<f64>,
    pub source_size_bytes: u64,
}
#[derive(Clone, Debug, Default)]
pub struct MediaInventory {
    pub images: Vec<Value>,
    pub video: Option<VideoInfo>,
}
pub trait GrokInputOwner {
    fn inventory(&mut self, input: &Input) -> Result<MediaInventory, ConformanceError>;
    fn inline_png(&mut self, source: &Value) -> Result<String, ConformanceError>;
    fn upload_references(
        &mut self,
        images: &[Value],
        mime_type: &str,
        wait_label: &str,
        max_images: usize,
    ) -> Result<Vec<String>, ConformanceError>;
    fn upload_video(
        &mut self,
        video: &Value,
        mime_type: &str,
        codec: &str,
    ) -> Result<String, ConformanceError>;
}
#[derive(Clone, Debug, Default)]
pub struct Resolution {
    pub image_urls: Vec<String>,
    pub video_url: Option<String>,
    pub image_count: usize,
    pub video_duration_seconds: Option<f64>,
    pub video_size_bytes: Option<u64>,
}
#[derive(Clone, Debug)]
pub struct PlannedRequest {
    pub path: &'static str,
    pub body: Value,
    pub is_video: bool,
    pub price_multiplier: f64,
}
fn model(input: &Input) -> Result<&Value, GrokError> {
    argument(input, "model")
}
fn model_id(input: &Input) -> Result<&str, GrokError> {
    let value = model(input)?;
    text(value.get("model").unwrap_or(value))
}
fn selected<'a>(input: &'a Input, name: &str) -> Result<&'a Value, GrokError> {
    if matches!(
        input.node.as_str(),
        "GrokImageEditNodeV2" | "GrokVideoReferenceNode" | "GrokVideoExtendNode"
    ) {
        model(input)?
            .get(name)
            .ok_or(GrokError::Input("Missing Grok model argument"))
    } else {
        argument(input, name)
    }
}
const IMAGE_MODELS: &[&str] = &[
    "grok-imagine-image-quality",
    "grok-imagine-image-pro",
    "grok-imagine-image",
];
const IMAGE_RATIOS: &[&str] = &[
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "9:19.5", "19.5:9", "9:20", "20:9",
    "1:2", "2:1",
];
const VIDEO_RATIOS: &[&str] = &["auto", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"];
fn duration(input: &Input, minimum: u64, maximum: u64) -> Result<u64, GrokError> {
    let value = integer(selected(input, "duration")?, maximum)?;
    if value < minimum {
        return Err(GrokError::Input("Grok duration is below source minimum"));
    }
    Ok(value)
}
fn optional_ratio(ratio: &str) -> Option<&str> {
    if ratio == "auto" { None } else { Some(ratio) }
}
fn validate_video(resolution: &Resolution, minimum: f64, maximum: f64) -> Result<(), GrokError> {
    if let Some(duration) = resolution.video_duration_seconds
        && (!duration.is_finite() || duration < minimum - 0.0001 || duration > maximum + 0.0001)
    {
        return Err(GrokError::Input(
            "Grok input video duration exceeds source limits",
        ));
    }
    if resolution
        .video_size_bytes
        .ok_or(GrokError::Input("Missing canonical video source size"))?
        > 50 * 1024 * 1024
    {
        return Err(GrokError::Input("Grok input video exceeds 50MB limit"));
    }
    Ok(())
}
pub fn request(input: &Input, resolution: &Resolution) -> Result<PlannedRequest, GrokError> {
    let input = normalized(input)?;
    let prompt_value = prompt(argument(&input, "prompt")?)?;
    let seed = integer(argument(&input, "seed")?, 2147483647)?;
    let identifier = model_id(&input)?;
    let mut multiplier = 1.0;
    let (path, body, is_video) = match input.node.as_str() {
        "GrokImageNode" => {
            choice(&json!(identifier), IMAGE_MODELS)?;
            let ratio = choice(argument(&input, "aspect_ratio")?, &IMAGE_RATIOS[1..])?;
            let count = integer(argument(&input, "number_of_images")?, 10)?;
            if count == 0 {
                return Err(GrokError::Input(
                    "Grok output image count below source minimum",
                ));
            }
            let size = choice(argument(&input, "resolution")?, &["1K", "2K"])?;
            (
                "/proxy/xai/v1/images/generations",
                json!({"model":identifier,"prompt":prompt_value,"aspect_ratio":ratio,"n":count,"seed":seed,"resolution":size.to_lowercase(),"response_format":"url"}),
                false,
            )
        }
        "GrokImageEditNode" | "GrokImageEditNodeV2" => {
            choice(&json!(identifier), IMAGE_MODELS)?;
            if input.node == "GrokImageEditNodeV2" && resolution.image_count == 0 {
                return Err(GrokError::Provider(
                    "At least one image is required for editing.".into(),
                ));
            }
            if identifier == "grok-imagine-image-pro" && resolution.image_count > 1 {
                return Err(GrokError::Provider(
                    "The pro model supports only 1 input image.".into(),
                ));
            }
            if identifier != "grok-imagine-image-pro" && resolution.image_count > 3 {
                return Err(GrokError::Provider(
                    "A maximum of 3 input images is supported.".into(),
                ));
            }
            let ratio_value = selected(&input, "aspect_ratio").ok();
            let ratio = match ratio_value {
                Some(value) => choice(value, IMAGE_RATIOS)?,
                None => "auto",
            };
            if ratio != "auto" && resolution.image_count == 1 {
                return Err(GrokError::Provider("Custom aspect ratio is only allowed when multiple images are connected to the image input.".into()));
            }
            let count = integer(selected(&input, "number_of_images")?, 10)?;
            if count == 0 {
                return Err(GrokError::Input(
                    "Grok output image count below source minimum",
                ));
            }
            let size = choice(selected(&input, "resolution")?, &["1K", "2K"])?;
            let mut body = json!({"model":identifier,"prompt":prompt_value,"images":resolution.image_urls.iter().map(|url|json!({"url":url})).collect::<Vec<_>>(),"resolution":size.to_lowercase(),"n":count,"seed":seed,"response_format":"url"});
            if let Some(ratio) = optional_ratio(ratio) {
                body["aspect_ratio"] = json!(ratio);
            }
            ("/proxy/xai/v1/images/edits", body, false)
        }
        "GrokVideoNode" => {
            choice(
                &json!(identifier),
                &["grok-imagine-video", "grok-imagine-video-1.5"],
            )?;
            if resolution.image_count == 0 && identifier == "grok-imagine-video-1.5" {
                return Err(GrokError::Provider(format!(
                    "The '{identifier}' model requires an input image; connect one to the 'image' input."
                )));
            }
            if resolution.image_count > 1 {
                return Err(GrokError::Provider(
                    "Only one input image is supported.".into(),
                ));
            }
            let size = choice(selected(&input, "resolution")?, &["480p", "720p", "1080p"])?;
            if size == "1080p" && identifier != "grok-imagine-video-1.5" {
                return Err(GrokError::Provider(format!(
                    "1080p resolution is only available for grok-imagine-video-1.5, not '{identifier}'."
                )));
            }
            let ratio = choice(selected(&input, "aspect_ratio")?, VIDEO_RATIOS)?;
            let mut body = json!({"model":identifier,"prompt":prompt_value,"resolution":size,"duration":duration(&input,1,15)?,"seed":seed});
            if let Some(ratio) = optional_ratio(ratio) {
                body["aspect_ratio"] = json!(ratio);
            }
            if let Some(url) = resolution.image_urls.first() {
                body["image"] = json!({"url":url});
            }
            if identifier == "grok-imagine-video-1.5" {
                multiplier = 1.43;
            }
            ("/proxy/xai/v1/videos/generations", body, true)
        }
        "GrokVideoEditNode" => {
            choice(&json!(identifier), &["grok-imagine-video"])?;
            validate_video(resolution, 1.0, 8.7)?;
            (
                "/proxy/xai/v1/videos/edits",
                json!({"model":identifier,"prompt":prompt_value,"video":{"url":resolution.video_url},"seed":seed}),
                true,
            )
        }
        "GrokVideoReferenceNode" => {
            choice(&json!(identifier), &["grok-imagine-video"])?;
            let size = choice(selected(&input, "resolution")?, &["480p", "720p"])?;
            let ratio = choice(selected(&input, "aspect_ratio")?, &VIDEO_RATIOS[1..])?;
            multiplier = 1.43;
            (
                "/proxy/xai/v1/videos/generations",
                json!({"model":identifier,"prompt":prompt_value,"reference_images":resolution.image_urls.iter().take(7).map(|url|json!({"url":url})).collect::<Vec<_>>(),"resolution":size,"duration":duration(&input,2,10)?,"aspect_ratio":ratio,"seed":seed}),
                true,
            )
        }
        "GrokVideoExtendNode" => {
            choice(&json!(identifier), &["grok-imagine-video"])?;
            validate_video(resolution, 2.0, 15.0)?;
            multiplier = 1.43;
            (
                "/proxy/xai/v1/videos/extensions",
                json!({"prompt":prompt_value,"video":{"url":resolution.video_url},"duration":duration(&input,2,10)?}),
                true,
            )
        }
        _ => return Err(GrokError::Input("Unknown Grok node")),
    };
    Ok(PlannedRequest {
        path,
        body,
        is_video,
        price_multiplier: multiplier,
    })
}
fn valid_base64(value: &str) -> bool {
    if value.is_empty() || !value.len().is_multiple_of(4) {
        return false;
    }
    let raw = value.trim_end_matches('=');
    value.len() - raw.len() <= 2
        && !raw.is_empty()
        && raw
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
}
fn check_url(url: &str) -> Result<(), GrokError> {
    if endpoint(url)? {
        return Err(GrokError::Input("Canonical upload returned a relative URL"));
    }
    Ok(())
}
fn prepare(
    input: &Input,
    inventory: &MediaInventory,
    owner: &mut dyn GrokInputOwner,
    transport: &mut dyn GrokTransport,
) -> Result<Resolution, GrokError> {
    let mut resolution = Resolution {
        image_count: inventory.images.len(),
        video_duration_seconds: inventory
            .video
            .as_ref()
            .and_then(|video| video.duration_seconds),
        video_size_bytes: inventory
            .video
            .as_ref()
            .map(|video| video.source_size_bytes),
        ..Resolution::default()
    };
    request(input, &resolution)?;
    match input.node.as_str() {
        "GrokImageEditNode" | "GrokImageEditNodeV2" | "GrokVideoNode" => {
            for source in &inventory.images {
                transport.check_cancelled(None)?;
                let bytes = owner.inline_png(source)?;
                if !valid_base64(&bytes) {
                    return Err(GrokError::Input(
                        "Canonical image encoder returned invalid base64",
                    ));
                }
                let url = format!("data:image/png;base64,{bytes}");
                if url.len() > MAXIMUM_REQUEST_BYTES {
                    return Err(ConformanceError::BodyLimit.into());
                }
                resolution.image_urls.push(url);
            }
        }
        "GrokVideoReferenceNode" => {
            transport.check_cancelled(None)?;
            let count = inventory.images.len().min(7);
            let urls = owner.upload_references(
                &inventory.images,
                "image/png",
                "Uploading base images",
                7,
            )?;
            if urls.len() != count {
                return Err(GrokError::Input(
                    "Canonical reference upload returned wrong count",
                ));
            }
            for url in &urls {
                check_url(url)?;
            }
            resolution.image_urls = urls;
        }
        "GrokVideoEditNode" | "GrokVideoExtendNode" => {
            transport.check_cancelled(None)?;
            let video = inventory
                .video
                .as_ref()
                .ok_or(GrokError::Input("Missing canonical video inventory"))?;
            let url = owner.upload_video(&video.source, "video/mp4", "h264")?;
            check_url(&url)?;
            resolution.video_url = Some(url);
        }
        "GrokImageNode" => {}
        _ => return Err(GrokError::Input("Unknown Grok media plan")),
    }
    Ok(resolution)
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
    pub media: Vec<DownloadedMedia>,
    pub usd: Option<f64>,
    pub response_bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub source_video_duration_seconds: Option<u64>,
}
fn price(response: &Value, multiplier: f64) -> Result<Option<f64>, GrokError> {
    let Some(usage) = response.get("usage").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    if !usage.is_object() {
        return Err(GrokError::Response("Invalid Grok usage object"));
    }
    let Some(ticks) = usage
        .get("cost_in_usd_ticks")
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let ticks = ticks
        .as_u64()
        .ok_or(GrokError::Response("Invalid Grok cost ticks"))?;
    let value = ticks as f64 / 10_000_000_000.0 * multiplier;
    if !value.is_finite() {
        return Err(ConformanceError::InvalidCost.into());
    }
    Ok(Some(value))
}
fn image_urls(response: &Value) -> Result<Vec<String>, GrokError> {
    let data = response["data"]
        .as_array()
        .ok_or(GrokError::Response("Missing Grok image data"))?;
    for image in data {
        if !image.is_object() {
            return Err(GrokError::Response("Invalid Grok image object"));
        }
        for name in ["url", "b64_json", "revised_prompt"] {
            if image
                .get(name)
                .is_some_and(|value| !value.is_null() && !value.is_string())
            {
                return Err(GrokError::Response("Invalid Grok image response field"));
            }
        }
    }
    let urls = if data.len() == 1 {
        vec![
            text(&data[0]["url"])
                .map_err(|_| GrokError::Response("Grok single image has no download URL"))?
                .to_owned(),
        ]
    } else {
        data.iter()
            .filter_map(|image| {
                image["url"]
                    .as_str()
                    .filter(|url| !url.is_empty())
                    .map(str::to_owned)
            })
            .collect()
    };
    if urls.is_empty() {
        return Err(GrokError::Response("Grok returned no downloadable images"));
    }
    if urls.len() > 10 {
        return Err(GrokError::Response(
            "Grok response exceeds output image limit",
        ));
    }
    for url in &urls {
        check_url(url)?;
    }
    Ok(urls)
}
fn validate_video_response(response: &Value) -> Result<(), GrokError> {
    if response
        .get("model")
        .is_some_and(|value| !value.is_null() && !value.is_string())
    {
        return Err(GrokError::Response("Invalid Grok result model"));
    }
    if let Some(video) = response.get("video").filter(|value| !value.is_null()) {
        if !video.is_object() || !video["url"].is_string() || video["duration"].as_u64().is_none() {
            return Err(GrokError::Response("Invalid Grok video object"));
        }
        if video
            .get("upsampled_prompt")
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err(GrokError::Response("Invalid Grok video prompt"));
        }
    }
    Ok(())
}
fn video_status(response: &Value) -> Result<String, GrokError> {
    if !response.is_object() {
        return Err(GrokError::Response("Invalid Grok video status response"));
    }
    match response.get("status").filter(|value| !value.is_null()) {
        Some(value) => Ok(text(value)
            .map_err(|_| GrokError::Response("Invalid Grok status"))?
            .trim_matches(python_whitespace)
            .to_lowercase()),
        None => Ok("complete".into()),
    }
}
fn source_status(status: &str, category: &str) -> Result<bool, GrokError> {
    Ok(oracle()?["polling"][category]
        .as_array()
        .ok_or(GrokError::Response("Missing source status constants"))?
        .iter()
        .any(|value| value.as_str() == Some(status)))
}
fn poll_path(response: &Value) -> Result<String, GrokError> {
    let identifier = text(&response["request_id"])
        .map_err(|_| GrokError::Response("Missing Grok request identifier"))?;
    if identifier.is_empty()
        || identifier.len() > 256
        || !identifier
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(GrokError::Response("Invalid Grok request identifier"));
    }
    Ok(format!("/proxy/xai/v1/videos/{identifier}"))
}
fn parse_json(bytes: &[u8]) -> Result<Value, GrokError> {
    let value: Value = serde_json::from_slice(bytes)?;
    if !value.is_object() {
        return Err(GrokError::Response("Expected Grok response object"));
    }
    Ok(value)
}
fn approve(
    handle: StreamHandle,
    maximum: u64,
    transport: &mut dyn GrokTransport,
) -> Result<(), GrokError> {
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "grok.generation.maximum".into(),
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
    transport: &mut dyn GrokTransport,
    owner: &mut dyn GrokInputOwner,
) -> Result<ResultValue, GrokError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    let head = request_head(
        "/proxy/xai/v1/images/generations",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let inventory = owner.inventory(&input)?;
    let preflight = Resolution {
        image_count: inventory.images.len(),
        video_duration_seconds: inventory
            .video
            .as_ref()
            .and_then(|video| video.duration_seconds),
        video_size_bytes: inventory
            .video
            .as_ref()
            .map(|video| video.source_size_bytes),
        ..Resolution::default()
    };
    let planned = request(&input, &preflight)?;
    let mut head = head;
    head.path = planned.path.into();
    let initial_handle = transport.start_request(&head)?;
    approve(initial_handle, maximum_cost_microunits, transport)?;
    let resolution = prepare(&input, &inventory, owner, transport)?;
    let planned = request(&input, &resolution)?;
    let body = serde_json::to_vec(&planned.body)?;
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let (mut bytes, mut receipt, mut final_handle) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        initial_handle,
        transport,
    )?;
    let mut response = parse_json(&bytes)?;
    if planned.is_video {
        let path = poll_path(&response)?;
        let head = request_head(&path, HttpMethod::Get, credentials, false)?;
        let mut active = 0;
        // Queued polls do not consume the source's active budget; the supervisor
        // supplies a finite total bound so perpetual queueing cannot retain a worker.
        for total in 0..4096 {
            let exchanged = exchange_with_policy(
                &head,
                &[],
                None,
                MAXIMUM_RESPONSE_BYTES,
                POLL_RETRY,
                transport,
            )?;
            bytes = exchanged.0;
            receipt = exchanged.1;
            final_handle = exchanged.2;
            response = parse_json(&bytes)?;
            validate_video_response(&response)?;
            let status = video_status(&response)?;
            price(&response, planned.price_multiplier)?;
            if source_status(&status, "COMPLETED_STATUSES")? {
                break;
            }
            if source_status(&status, "FAILED_STATUSES")? {
                return Err(GrokError::Provider(format!(
                    "Task failed: {}",
                    serde_json::to_string(&response)?
                )));
            }
            if !source_status(&status, "QUEUED_STATUSES")? {
                active += 1;
            }
            if active >= 480 || total == 4095 {
                return Err(ConformanceError::PollLimit.into());
            }
            transport.report_progress(
                final_handle,
                &Progress {
                    sequence: total as u64,
                    completed: 0,
                    total: 100,
                    message: Some(status),
                },
            )?;
            transport.check_cancelled(Some(final_handle))?;
            transport.poll_delay(final_handle, 5000)?;
        }
    }
    let usd = price(&response, planned.price_multiplier)?;
    let (urls, duration) = if planned.is_video {
        let video = &response["video"];
        let url = text(&video["url"])
            .map_err(|_| GrokError::Response("Missing Grok result video URL"))?;
        check_url(url)?;
        let duration = video["duration"]
            .as_u64()
            .ok_or(GrokError::Response("Invalid Grok result video duration"))?;
        if video
            .get("upsampled_prompt")
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err(GrokError::Response("Invalid Grok video prompt"));
        }
        (vec![url.to_owned()], Some(duration))
    } else {
        (image_urls(&response)?, None)
    };
    let mut media = Vec::new();
    let mut aggregate = 0usize;
    for url in urls {
        let head = request_head(&url, HttpMethod::Get, credentials, true)?;
        let (download, download_receipt, _handle) = exchange_with_policy(
            &head,
            &[],
            None,
            MAXIMUM_MEDIA_BYTES,
            DOWNLOAD_RETRY,
            transport,
        )?;
        if download.is_empty() {
            return Err(GrokError::Response("Empty Grok media download"));
        }
        aggregate = aggregate
            .checked_add(download.len())
            .ok_or(ConformanceError::BodyLimit)?;
        if aggregate > MAXIMUM_MEDIA_BYTES {
            return Err(ConformanceError::BodyLimit.into());
        }
        media.push(DownloadedMedia {
            url,
            bytes: download,
            receipt: download_receipt,
        });
    }
    transport.check_cancelled(Some(final_handle))?;
    Ok(ResultValue {
        node: input.node,
        is_video: planned.is_video,
        media,
        usd,
        response_bytes: bytes,
        receipt,
        source_video_duration_seconds: duration,
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
    transport: &mut dyn GrokTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), GrokError> {
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
    transport: &mut dyn GrokTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), GrokError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(GrokError::Http {
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
                    return Err(GrokError::Http {
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
    transport: &mut dyn GrokTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), GrokError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "grok.generation.maximum".into(),
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
                    return Err(GrokError::Http {
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

pub trait GrokMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn GrokTransport,
    owner: &mut dyn GrokInputOwner,
    materializer: &mut dyn GrokMaterializer,
    sink: &mut dyn MaterializationSink,
) -> Result<(), GrokError> {
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
        .map_err(|_| GrokError::Response("Missing canonical type registry"))?;
    let type_id = registry
        .resolve(if result.is_video { "VIDEO" } else { "IMAGE" })
        .map_err(|_| GrokError::Response("Missing canonical output type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| GrokError::Response("Missing canonical output family"))?;
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
fn fixture_resolution(case: &Value) -> Result<Resolution, GrokError> {
    let input: Input = serde_json::from_value(case["input"].clone())?;
    let mut resolution = Resolution::default();
    fn images(value: &Value) -> usize {
        if let Some(count) = value["image_count"].as_u64() {
            count as usize
        } else if let Some(values) = value.as_object() {
            values.values().map(images).sum()
        } else {
            0
        }
    }
    resolution.image_count = images(&json!(input.arguments));
    let video = input.arguments.get("video");
    resolution.video_duration_seconds = video.and_then(|value| value["duration"].as_f64());
    resolution.video_size_bytes = video.and_then(|value| value["source_size"].as_u64());
    if let Some(body) = case
        .get("requests")
        .and_then(Value::as_array)
        .and_then(|requests| requests.first())
        .and_then(|request| request.get("body"))
    {
        if let Some(images) = body["images"]
            .as_array()
            .or_else(|| body["reference_images"].as_array())
        {
            for image in images {
                resolution.image_urls.push(text(&image["url"])?.into());
            }
        } else if let Some(url) = body["image"]["url"].as_str() {
            resolution.image_urls.push(url.into());
        }
        resolution.video_url = body["video"]["url"].as_str().map(str::to_owned);
    }
    Ok(resolution)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({
            "schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,
            "aliases":ALIASES,"nodes":NODES,"routes":ROUTES,
            "scope":"source request and bounded transport; canonical native media and guest integration pending",
            "source_validation":{
                "python_strip_control_separators":[28,29,30,31],
                "zero_width_space_is_whitespace":false,
                "prompt_preserves_original":true,
                "validation_utils_sha256":"77554cc809f651f54943f73e6391b8899b3a52779d949edfc08166d8fdf86c66",
                "client_sha256":"939696eb63765794ecd8a1a90a076fd33118603045a9d55851fd8fb67db60ecc"
            }
        })
    {
        return Err(ConformanceError::InvalidRequest);
    }
    for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
        if prompt(&json!(separator.to_string())).is_ok()
            || video_status(&json!({"status":format!("{separator} COMPLETE {separator}")}))
                .map_err(|_| ConformanceError::InvalidRequest)?
                != "complete"
        {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    for original in ["\u{001c} Original prompt \u{001f}", "\u{200b}"] {
        let input = json!(original);
        if prompt(&input).map_err(|_| ConformanceError::InvalidRequest)? != original {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    if video_status(&json!({"status":"\u{200b}COMPLETE\u{200b}"}))
        .map_err(|_| ConformanceError::InvalidRequest)?
        != "\u{200b}complete\u{200b}"
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let oracle = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
    if oracle["source_sha256"]["comfy_api_nodes/nodes_grok.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in oracle["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let planned = request(
            &input,
            &fixture_resolution(case).map_err(|_| ConformanceError::InvalidRequest)?,
        )
        .map_err(|_| ConformanceError::InvalidRequest)?;
        if planned.body != case["requests"][0]["body"]
            || planned.path
                != case["requests"][0]["path"]
                    .as_str()
                    .ok_or(ConformanceError::InvalidRequest)?
        {
            return Err(ConformanceError::InvalidRequest);
        }
        if planned.is_video
            && case["requests"][1]["path"] != "/proxy/xai/v1/videos/source-request-42"
        {
            return Err(ConformanceError::InvalidRequest);
        }
        covered.insert(input.node);
    }
    for case in oracle["preflight_errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        if request(
            &input,
            &fixture_resolution(case).map_err(|_| ConformanceError::InvalidRequest)?,
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
    impl GrokTransport for Transport {
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
            "/../../sim_test_support/fixtures/providers/grok/downloaded-image.png"
        ))
        .to_vec()
    }
    fn png_base64() -> String {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/grok/downloaded-image.base64"
        ))
        .trim()
        .into()
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
    fn generation_response() -> Value {
        json!({"data":[{"url":"https://fixture.invalid/image.png"}],"usage":{"cost_in_usd_ticks":123456789}})
    }
    fn video_response(status: Option<&str>) -> Value {
        json!({"status":status,"video":{"url":"https://fixture.invalid/video.mp4","duration":6},"usage":{"cost_in_usd_ticks":123456789}})
    }
    fn image_events() -> Result<VecDeque<Result<ResponseEvent, ConformanceError>>, serde_json::Error>
    {
        let mut values = events(200, serde_json::to_vec(&generation_response())?, vec![]);
        values.extend(events(200, png(), vec![]));
        Ok(values)
    }
    #[derive(Default)]
    struct InputOwner {
        inventory: MediaInventory,
        inline_calls: Vec<Value>,
        reference_calls: Vec<Vec<Value>>,
        video_calls: Vec<Value>,
        inline_failure: bool,
        upload_failure: bool,
        invalid_inline: bool,
    }
    impl GrokInputOwner for InputOwner {
        fn inventory(&mut self, _input: &Input) -> Result<MediaInventory, ConformanceError> {
            Ok(self.inventory.clone())
        }
        fn inline_png(&mut self, source: &Value) -> Result<String, ConformanceError> {
            self.inline_calls.push(source.clone());
            if self.inline_failure {
                return Err(ConformanceError::HostFailure);
            }
            Ok(if self.invalid_inline {
                "canonical-not-encoded-png".into()
            } else {
                png_base64()
            })
        }
        fn upload_references(
            &mut self,
            images: &[Value],
            mime_type: &str,
            wait_label: &str,
            max_images: usize,
        ) -> Result<Vec<String>, ConformanceError> {
            assert_eq!(mime_type, "image/png");
            assert_eq!(wait_label, "Uploading base images");
            assert_eq!(max_images, 7);
            self.reference_calls.push(images.to_vec());
            if self.upload_failure {
                return Err(ConformanceError::HostFailure);
            }
            Ok(images
                .iter()
                .take(max_images)
                .enumerate()
                .map(|(index, _image)| format!("https://storage.fixture.invalid/reference/{index}"))
                .collect())
        }
        fn upload_video(
            &mut self,
            video: &Value,
            mime_type: &str,
            codec: &str,
        ) -> Result<String, ConformanceError> {
            assert_eq!((mime_type, codec), ("video/mp4", "h264"));
            self.video_calls.push(video.clone());
            Err(ConformanceError::HostFailure)
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
    struct RejectDecoder {
        seen: bool,
    }
    impl GrokMaterializer for RejectDecoder {
        fn materialize(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            assert!(!result.is_video);
            assert_eq!(result.media.len(), 1);
            assert_eq!(result.media[0].bytes, png());
            assert_eq!(result.media[0].receipt, vec![7, 8, 9]);
            self.seen = true;
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn grok_all_claims_and_executed_source_requests_are_exact() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/grok/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 7);
        assert_eq!(ROUTES.len(), 7);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            6
        );
        assert_eq!(ALIASES, ["xai"]);
        let source = oracle()?;
        assert_eq!(source["cases"].as_array().ok_or("cases")?.len(), 9);
        assert_eq!(
            source["preflight_errors"].as_array().ok_or("errors")?.len(),
            12
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
    fn grok_price_ticks_and_video_multiplier_match_executed_source() -> Result<(), Box<dyn Error>> {
        for case in oracle()?["price_cases"].as_array().ok_or("price cases")? {
            let response = json!({"usage":{"cost_in_usd_ticks":case["cost_in_usd_ticks"]}});
            assert_eq!(price(&response, 1.0)?, case["image_usd"].as_f64());
            assert_eq!(price(&response, 1.43)?, case["video_usd"].as_f64());
        }
        assert!(price(&json!({"usage":{"cost_in_usd_ticks":-1}}), 1.0).is_err());
        Ok(())
    }
    #[test]
    fn grok_seeds_resolution_and_extension_omissions_match_source() -> Result<(), Box<dyn Error>> {
        let mut input = fixture_input("GrokImageNode")?;
        let planned = request(&input, &Resolution::default())?;
        assert_eq!(planned.body["resolution"], "1k");
        assert_eq!(planned.body["seed"], 2147483647u64);
        input.arguments.insert("seed".into(), json!(2147483648u64));
        assert!(request(&input, &Resolution::default()).is_err());
        let case = fixture_case("GrokVideoExtendNode")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let planned = request(&input, &fixture_resolution(&case)?)?;
        assert!(planned.body.get("seed").is_none() && planned.body.get("model").is_none());
        assert_eq!(planned.price_multiplier, 1.43);
        Ok(())
    }
    #[test]
    fn grok_video_duration_epsilon_size_and_source_null_status_are_exact()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokVideoEditNode")?;
        let mut resolution = Resolution {
            video_duration_seconds: Some(8.70005),
            video_size_bytes: Some(50 * 1024 * 1024),
            ..Resolution::default()
        };
        assert!(request(&input, &resolution).is_ok());
        resolution.video_duration_seconds = Some(8.7002);
        assert!(request(&input, &resolution).is_err());
        resolution.video_duration_seconds = None;
        assert!(request(&input, &resolution).is_ok());
        resolution.video_size_bytes = Some(50 * 1024 * 1024 + 1);
        assert!(request(&input, &resolution).is_err());
        assert_eq!(video_status(&json!({}))?, "complete");
        assert_eq!(video_status(&json!({"status":" Completed "}))?, "completed");
        assert_eq!(
            poll_path(&json!({"request_id":"source-request-42"}))?,
            "/proxy/xai/v1/videos/source-request-42"
        );
        assert!(poll_path(&json!({"request_id":"../foreign"})).is_err());
        Ok(())
    }

    #[test]
    fn grok_python_whitespace_prompts_reject_before_paid_or_media_requests()
    -> Result<(), Box<dyn Error>> {
        for claim in NODES {
            for whitespace in [
                "\u{001c}",
                "\u{001d}",
                "\u{001e}",
                "\u{001f}",
                " \t\n\u{001c}\u{001d}\u{001e}\u{001f}\u{0085}\u{00a0}",
            ] {
                let mut input = fixture_input(claim.node_identifier)?;
                input.arguments.insert("prompt".into(), json!(whitespace));
                let mut transport = Transport::default();
                let mut owner = InputOwner::default();
                assert!(matches!(
                    invoke(&input, credentials(), 100, &mut transport, &mut owner),
                    Err(GrokError::Input("Empty Grok prompt"))
                ));
                assert!(transport.heads.is_empty());
                assert!(transport.bodies.is_empty());
                assert!(transport.costs.is_empty());
                assert!(owner.inline_calls.is_empty());
                assert!(owner.reference_calls.is_empty());
                assert!(owner.video_calls.is_empty());
            }
        }
        Ok(())
    }
    #[test]
    fn grok_nonempty_prompts_preserve_original_bytes_and_zero_width_space()
    -> Result<(), Box<dyn Error>> {
        for claim in NODES {
            let case = fixture_case(claim.node_identifier)?;
            let resolution = fixture_resolution(&case)?;
            for original in ["\u{001c} \tOriginal prompt\n\u{001f}", "\u{200b}"] {
                let mut input = fixture_input(claim.node_identifier)?;
                input.arguments.insert("prompt".into(), json!(original));
                assert_eq!(request(&input, &resolution)?.body["prompt"], original);
            }
        }
        let mut input = fixture_input("GrokImageNode")?;
        input.arguments.insert("prompt".into(), json!("\u{200b}"));
        let mut transport = Transport {
            events: image_events()?,
            ..Transport::default()
        };
        let result = invoke(
            &input,
            credentials(),
            100,
            &mut transport,
            &mut InputOwner::default(),
        )?;
        let body: Value = serde_json::from_slice(transport.bodies.first().ok_or("body")?)?;
        assert_eq!(body["prompt"], "\u{200b}");
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(result.media.first().ok_or("download")?.bytes, png());
        Ok(())
    }
    #[test]
    fn grok_python_whitespace_polling_normalizes_without_extra_downloads()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokVideoNode")?;
        for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
            for terminal in ["COMPLETE", "FAILED"] {
                let mut values = events(
                    200,
                    serde_json::to_vec(&json!({"request_id":"source-request-42"}))?,
                    vec![],
                );
                for phase in ["QUEUED", "PROCESSING", terminal] {
                    let status = format!("{separator} \t{phase}\n{separator}");
                    let response = if phase == "COMPLETE" {
                        video_response(Some(&status))
                    } else {
                        json!({"status":status})
                    };
                    values.extend(events(200, serde_json::to_vec(&response)?, vec![]));
                }
                values.push_back(Err(ConformanceError::HostFailure));
                let mut transport = Transport {
                    events: values,
                    ..Transport::default()
                };
                let result = invoke(
                    &input,
                    credentials(),
                    100,
                    &mut transport,
                    &mut InputOwner::default(),
                );
                if terminal == "COMPLETE" {
                    assert!(matches!(
                        result,
                        Err(GrokError::Transport(ConformanceError::HostFailure))
                    ));
                    assert_eq!(transport.heads.len(), 5);
                    assert_eq!(
                        transport.heads.last().ok_or("download head")?.path,
                        "https://fixture.invalid/video.mp4"
                    );
                    assert_eq!(
                        transport.heads.last().ok_or("download head")?.secret_id,
                        None
                    );
                } else {
                    assert!(
                        matches!(result, Err(GrokError::Provider(message)) if message.starts_with("Task failed:"))
                    );
                    assert_eq!(transport.heads.len(), 4);
                    assert!(
                        transport
                            .heads
                            .iter()
                            .all(|head| head.path.starts_with("/proxy/xai/"))
                    );
                }
                assert_eq!(transport.poll_delays, [5000, 5000]);
                assert_eq!(transport.progresses.len(), 2);
                assert_eq!(
                    transport
                        .progresses
                        .first()
                        .ok_or("queued progress")?
                        .message
                        .as_deref(),
                    Some("queued")
                );
                assert_eq!(
                    transport
                        .progresses
                        .get(1)
                        .ok_or("processing progress")?
                        .message
                        .as_deref(),
                    Some("processing")
                );
                assert_eq!(
                    transport
                        .heads
                        .iter()
                        .filter(|head| head.path == "/proxy/xai/v1/videos/source-request-42")
                        .count(),
                    3
                );
            }
        }
        Ok(())
    }
    #[test]
    fn grok_zero_width_space_status_remains_unknown_in_invoke() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokVideoNode")?;
        let mut values = events(
            200,
            serde_json::to_vec(&json!({"request_id":"source-request-42"}))?,
            vec![],
        );
        for status in ["\u{200b}COMPLETE\u{200b}", "FAILED"] {
            values.extend(events(
                200,
                serde_json::to_vec(&json!({"status":status}))?,
                vec![],
            ));
        }
        let mut transport = Transport {
            events: values,
            ..Transport::default()
        };
        assert!(
            matches!(invoke(&input, credentials(), 100, &mut transport, &mut InputOwner::default()), Err(GrokError::Provider(message)) if message.starts_with("Task failed:"))
        );
        assert_eq!(transport.heads.len(), 3);
        assert_eq!(transport.poll_delays, [5000]);
        assert_eq!(transport.progresses.len(), 1);
        assert_eq!(
            transport
                .progresses
                .first()
                .ok_or("unknown status progress")?
                .message
                .as_deref(),
            Some("\u{200b}complete\u{200b}")
        );
        assert!(
            transport
                .heads
                .iter()
                .all(|head| head.path.starts_with("/proxy/xai/"))
        );
        Ok(())
    }
    #[test]
    fn grok_source_preflight_and_cost_denial_precede_encoding() -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        for case in source["preflight_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            assert!(request(&input, &fixture_resolution(case)?).is_err());
        }
        let input = fixture_input("GrokImageEditNode")?;
        let mut owner = InputOwner {
            inventory: MediaInventory {
                images: vec![json!("first"), json!("second")],
                video: None,
            },
            ..InputOwner::default()
        };
        let mut transport = Transport {
            deny_cost: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(GrokError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(owner.inline_calls.is_empty() && transport.bodies.iter().all(Vec::is_empty));
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
        Ok(())
    }
    #[test]
    fn grok_inline_png_order_and_actual_bytes_are_preserved() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokImageEditNodeV2")?;
        let inventory = MediaInventory {
            images: vec![json!("image_2/0"), json!("image_2/1"), json!("image_1")],
            video: None,
        };
        let mut owner = InputOwner {
            inventory: inventory.clone(),
            ..InputOwner::default()
        };
        let mut transport = Transport {
            events: image_events()?,
            ..Transport::default()
        };
        let result = invoke(&input, credentials(), 100, &mut transport, &mut owner)?;
        assert_eq!(owner.inline_calls, inventory.images);
        let body: Value = serde_json::from_slice(transport.bodies.first().ok_or("body")?)?;
        let images = body["images"].as_array().ok_or("images")?;
        assert_eq!(images.len(), 3);
        for image in images {
            assert_eq!(
                image["url"],
                format!("data:image/png;base64,{}", png_base64())
            );
        }
        assert_eq!(result.media[0].bytes, png());
        assert_eq!(result.usd, Some(0.0123456789));
        assert_eq!(
            result.response_bytes,
            serde_json::to_vec(&generation_response())?
        );
        Ok(())
    }
    #[test]
    fn grok_reference_upload_caps_source_order_before_polling() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokVideoReferenceNode")?;
        let inventory = MediaInventory {
            images: (0..9)
                .map(|index| json!(format!("source-{index}")))
                .collect(),
            video: None,
        };
        let mut owner = InputOwner {
            inventory: inventory.clone(),
            ..InputOwner::default()
        };
        let mut values = events(
            200,
            serde_json::to_vec(&json!({"request_id":"source-request-42"}))?,
            vec![],
        );
        values.extend(events(
            200,
            serde_json::to_vec(&video_response(None))?,
            vec![],
        ));
        values.push_back(Err(ConformanceError::HostFailure));
        let mut transport = Transport {
            events: values,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(GrokError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(owner.reference_calls, [inventory.images]);
        let body: Value = serde_json::from_slice(transport.bodies.first().ok_or("body")?)?;
        assert_eq!(
            body["reference_images"].as_array().ok_or("images")?.len(),
            7
        );
        assert_eq!(
            transport.heads[1].path,
            "/proxy/xai/v1/videos/source-request-42"
        );
        assert_eq!(transport.heads[2].secret_id, None);
        Ok(())
    }
    #[test]
    fn grok_failed_native_encoders_do_not_submit_structural_media() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokImageEditNode")?;
        for invalid in [false, true] {
            let mut owner = InputOwner {
                inventory: MediaInventory {
                    images: vec![json!("first"), json!("second")],
                    video: None,
                },
                inline_failure: !invalid,
                invalid_inline: invalid,
                ..InputOwner::default()
            };
            let mut transport = Transport::default();
            assert!(invoke(&input, credentials(), 100, &mut transport, &mut owner).is_err());
            assert!(transport.bodies.iter().all(Vec::is_empty));
        }
        let input = fixture_input("GrokVideoEditNode")?;
        let mut owner = InputOwner {
            inventory: MediaInventory {
                images: vec![],
                video: Some(VideoInfo {
                    source: json!("source-video"),
                    duration_seconds: Some(3.0),
                    source_size_bytes: 1024,
                }),
            },
            ..InputOwner::default()
        };
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(GrokError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(owner.video_calls, [json!("source-video")]);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn grok_multiple_images_filter_missing_urls_but_single_missing_fails()
    -> Result<(), Box<dyn Error>> {
        assert_eq!(
            image_urls(
                &json!({"data":[{"url":null},{"url":"https://fixture.invalid/second.png"}]})
            )?,
            ["https://fixture.invalid/second.png"]
        );
        assert!(image_urls(&json!({"data":[{"b64_json":"opaque"}]})).is_err());
        assert!(image_urls(&json!({"data":[]})).is_err());
        assert!(
            image_urls(&json!({"data":[{"url":"https://secret@fixture.invalid/image.png"}]}))
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn grok_real_download_credentials_receipts_and_clean_reinvocation_are_exact()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokImageNode")?;
        for credentials in [
            credentials(),
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
        ] {
            let mut transport = Transport {
                events: image_events()?,
                ..Transport::default()
            };
            let result = invoke(
                &input,
                credentials,
                100,
                &mut transport,
                &mut InputOwner::default(),
            )?;
            assert_eq!(result.media[0].bytes, png());
            assert_eq!(result.media[0].receipt, [7, 8, 9]);
            assert_eq!(
                transport.heads[0].secret_id.as_deref(),
                Some(if credentials.auth_token {
                    "sim_org.auth_token"
                } else {
                    "sim_org.api_key"
                })
            );
            assert_eq!(transport.heads[1].secret_id, None);
            assert_eq!(transport.heads[1].headers.len(), 1);
        }
        Ok(())
    }
    #[test]
    fn grok_retries_replay_exact_body_on_fresh_handles() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokImageNode")?;
        let mut values = events(
            429,
            b"rate limited".to_vec(),
            vec![Header {
                name: "Retry-After".into(),
                value: "2".into(),
            }],
        );
        values.extend(image_events()?);
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
        assert_eq!(transport.delays, [2000]);
        assert_eq!(transport.bodies[0], transport.bodies[1]);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(
            transport.costs.first().map(|cost| cost.operation.as_str()),
            Some("grok.generation.maximum")
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
            Err(GrokError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn grok_queued_polling_delegates_delay_and_failed_status_stops() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokVideoNode")?;
        let mut values = events(
            200,
            serde_json::to_vec(&json!({"request_id":"source-request-42"}))?,
            vec![],
        );
        for status in [" queued ", "processing", "failed"] {
            values.extend(events(
                200,
                serde_json::to_vec(&json!({"status":status}))?,
                vec![],
            ));
        }
        let mut transport = Transport {
            events: values,
            ..Transport::default()
        };
        assert!(
            matches!(invoke(&input,credentials(),100,&mut transport,&mut InputOwner::default()),Err(GrokError::Provider(message)) if message.starts_with("Task failed:"))
        );
        assert_eq!(transport.poll_delays, [5000, 5000]);
        assert_eq!(transport.progresses.len(), 2);
        assert_eq!(transport.progresses[0].message.as_deref(), Some("queued"));
        assert_eq!(
            transport.progresses[1].message.as_deref(),
            Some("processing")
        );
        Ok(())
    }
    #[test]
    fn grok_worker_loss_cancellation_malformed_and_oversized_streams_are_typed()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokImageNode")?;
        let cases = [
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
        ];
        for (mut transport, expected) in cases {
            assert!(
                matches!(invoke(&input,credentials(),100,&mut transport,&mut InputOwner::default()),Err(GrokError::Transport(error))if error==expected)
            );
        }
        let mut transport = Transport {
            events: events(200, b"{\"data\":[]} trailing".to_vec(), vec![]),
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
            Err(GrokError::Json(_))
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
            Err(GrokError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn grok_actual_png_reaches_canonical_decoder_failure_with_zero_publication()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GrokImageNode")?;
        let mut transport = Transport {
            events: image_events()?,
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
            Err(GrokError::Transport(ConformanceError::HostFailure))
        ));
        assert!(decoder.seen);
        assert_eq!(sink.published, 0);
        Ok(())
    }

    #[test]
    fn grok_generation_poll_and_download_retry_budgets_are_source_specific()
    -> Result<(), Box<dyn Error>> {
        for (policy, status, retries, last_delay) in [
            (GENERATION_RETRY, 503, 3, 4000),
            (POLL_RETRY, 503, 10, 20661),
            (DOWNLOAD_RETRY, 429, 5, 16000),
        ] {
            let head = request_head(
                "https://fixture.invalid/media",
                HttpMethod::Get,
                credentials(),
                true,
            )?;
            let mut values = VecDeque::new();
            for _ in 0..=retries {
                values.extend(events(status, b"retry".to_vec(), vec![]));
            }
            let mut transport = Transport {
                events: values,
                ..Transport::default()
            };
            assert!(
                matches!(exchange_with_policy(&head,&[],None,1024,policy,&mut transport),Err(GrokError::Http{status:observed,..})if observed==status)
            );
            assert_eq!(transport.delays.len(), retries);
            assert_eq!(transport.delays.last().copied(), Some(last_delay));
        }
        Ok(())
    }
    #[test]
    fn grok_active_poll_limit_and_malformed_video_models_fail_closed() -> Result<(), Box<dyn Error>>
    {
        assert!(validate_video_response(&json!({"model":12})).is_err());
        assert!(
            validate_video_response(
                &json!({"video":{"url":"https://fixture.invalid/video.mp4","duration":"six"}})
            )
            .is_err()
        );
        let input = fixture_input("GrokVideoNode")?;
        let mut values = events(
            200,
            serde_json::to_vec(&json!({"request_id":"source-request-42"}))?,
            vec![],
        );
        for _ in 0..480 {
            values.extend(events(
                200,
                serde_json::to_vec(&json!({"status":"processing"}))?,
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
                &mut InputOwner::default()
            ),
            Err(GrokError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.poll_delays.len(), 479);
        assert!(transport.events.is_empty());
        Ok(())
    }
}
