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
pub const NAMESPACE: &str = "zed.sim.provider.kling";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "ad22ffbc4865b60c48b81a36be9d02b9b40f2cd9a4860b85dbaed328a9139e84";
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
        feature_id: "COMFY-NODE-0279",
        node_identifier: "KlingAvatarNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0280",
        node_identifier: "KlingCameraControlI2VNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0281",
        node_identifier: "KlingCameraControls",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0282",
        node_identifier: "KlingCameraControlT2VNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0283",
        node_identifier: "KlingDualCharacterVideoEffectNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0284",
        node_identifier: "KlingFirstLastFrameNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0285",
        node_identifier: "KlingImage2VideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0286",
        node_identifier: "KlingImageGenerationNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0287",
        node_identifier: "KlingImageToVideoWithAudio",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0288",
        node_identifier: "KlingLipSyncAudioToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0289",
        node_identifier: "KlingLipSyncTextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0290",
        node_identifier: "KlingMotionControl",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0291",
        node_identifier: "KlingOmniProEditVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0292",
        node_identifier: "KlingOmniProFirstLastFrameNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0293",
        node_identifier: "KlingOmniProImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0294",
        node_identifier: "KlingOmniProImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0295",
        node_identifier: "KlingOmniProTextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0296",
        node_identifier: "KlingOmniProVideoToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0297",
        node_identifier: "KlingSingleImageVideoEffectNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0298",
        node_identifier: "KlingStartEndFrameNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0299",
        node_identifier: "KlingTextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0300",
        node_identifier: "KlingTextToVideoWithAudio",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0301",
        node_identifier: "KlingVideoExtendNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0302",
        node_identifier: "KlingVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0303",
        node_identifier: "KlingVirtualTryOnNode",
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
        feature_id: "COMFY-API-EXT-0055",
        method: "POST",
        path: "/proxy/kling/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0056",
        method: "POST",
        path: "/proxy/kling/image-to-video/kling-3.0-turbo",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0057",
        method: "GET",
        path: "/proxy/kling/tasks",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0058",
        method: "POST",
        path: "/proxy/kling/text-to-video/kling-3.0-turbo",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0059",
        method: "POST",
        path: "/proxy/kling/v1/images/omni-image",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0060",
        method: "GET",
        path: "/proxy/kling/v1/images/omni-image/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0061",
        method: "GET",
        path: "/proxy/kling/v1/images/omni-image/{response.data.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0062",
        method: "POST",
        path: "/proxy/kling/v1/videos/avatar/image2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0063",
        method: "GET",
        path: "/proxy/kling/v1/videos/avatar/image2video/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0064",
        method: "GET",
        path: "/proxy/kling/v1/videos/avatar/image2video/{response.data.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0065",
        method: "POST",
        path: "/proxy/kling/v1/videos/image2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0066",
        method: "GET",
        path: "/proxy/kling/v1/videos/image2video/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0067",
        method: "GET",
        path: "/proxy/kling/v1/videos/image2video/{response.data.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0068",
        method: "POST",
        path: "/proxy/kling/v1/videos/motion-control",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0069",
        method: "GET",
        path: "/proxy/kling/v1/videos/motion-control/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0070",
        method: "GET",
        path: "/proxy/kling/v1/videos/motion-control/{response.data.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0071",
        method: "POST",
        path: "/proxy/kling/v1/videos/omni-video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0072",
        method: "GET",
        path: "/proxy/kling/v1/videos/omni-video/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0073",
        method: "GET",
        path: "/proxy/kling/v1/videos/omni-video/{response.data.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0074",
        method: "POST",
        path: "/proxy/kling/v1/videos/text2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0075",
        method: "GET",
        path: "/proxy/kling/v1/videos/text2video/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0076",
        method: "GET",
        path: "/proxy/kling/v1/videos/text2video/{response.data.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0077",
        method: "POST",
        path: "/proxy/kling/{KLING_API_VERSION}/images/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0078",
        method: "POST",
        path: "/proxy/kling/{KLING_API_VERSION}/images/kolors-virtual-try-on",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0079",
        method: "POST",
        path: "/proxy/kling/{KLING_API_VERSION}/videos/effects",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0080",
        method: "POST",
        path: "/proxy/kling/{KLING_API_VERSION}/videos/image2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0081",
        method: "POST",
        path: "/proxy/kling/{KLING_API_VERSION}/videos/lip-sync",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0082",
        method: "POST",
        path: "/proxy/kling/{KLING_API_VERSION}/videos/text2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0083",
        method: "POST",
        path: "/proxy/kling/{KLING_API_VERSION}/videos/video-extend",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0221",
        method: "GET",
        path: "/proxy/kling/v1/videos/effects/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0222",
        method: "GET",
        path: "/proxy/kling/v1/videos/lip-sync/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0223",
        method: "GET",
        path: "/proxy/kling/v1/videos/video-extend/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0224",
        method: "GET",
        path: "/proxy/kling/v1/images/kolors-virtual-try-on/{task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0225",
        method: "GET",
        path: "/proxy/kling/v1/images/generations/{task_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum KlingError {
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
impl fmt::Display for KlingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Kling HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for KlingError {}
impl From<ConformanceError> for KlingError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for KlingError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, KlingError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/kling/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, KlingError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(KlingError::Input("Unknown Kling node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(KlingError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(KlingError::Input("Unknown Kling source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(KlingError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(KlingError::Input("Missing Kling source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, KlingError> {
    input
        .arguments
        .get(name)
        .ok_or(KlingError::Input("Missing Kling argument"))
}
fn text(value: &Value) -> Result<&str, KlingError> {
    value
        .as_str()
        .ok_or(KlingError::Input("Expected Kling string"))
}
fn integer(value: &Value, maximum: u64) -> Result<u64, KlingError> {
    let number = value
        .as_u64()
        .ok_or(KlingError::Input("Expected Kling unsigned integer"))?;
    if number > maximum {
        return Err(KlingError::Input("Kling integer bound exceeded"));
    }
    Ok(number)
}
fn endpoint(value: &str) -> Result<bool, KlingError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(KlingError::Response("Invalid Kling response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(KlingError::Response("Invalid Kling response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(KlingError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(KlingError::Response("Invalid Kling endpoint authority"));
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
) -> Result<RequestHead, KlingError> {
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
            return Err(KlingError::Input("Missing Comfy API credentials"));
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

pub trait KlingTransport: ProviderTransport {
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
pub struct ImageInfo {
    pub width: u64,
    pub height: u64,
    pub count: u64,
}
#[derive(Clone, Debug)]
pub struct VideoInfo {
    pub width: u64,
    pub height: u64,
    pub duration_seconds: f64,
}
#[derive(Clone, Debug)]
pub struct AudioInfo {
    pub sample_rate: u64,
    pub samples: u64,
}
#[derive(Clone, Debug)]
pub struct UploadValue {
    pub urls: Vec<String>,
    pub receipts: Vec<Vec<u8>>,
}
pub trait KlingMediaOwner {
    fn image_info(&mut self, source: &Value) -> Result<ImageInfo, ConformanceError>;
    fn video_info(&mut self, source: &Value) -> Result<VideoInfo, ConformanceError>;
    fn audio_info(&mut self, source: &Value) -> Result<AudioInfo, ConformanceError>;
    fn inline_png(
        &mut self,
        source: &Value,
        total_pixels: u64,
    ) -> Result<Vec<u8>, ConformanceError>;
    fn upload_image(
        &mut self,
        source: &Value,
        wait_label: Option<&str>,
    ) -> Result<UploadValue, ConformanceError>;
    fn upload_images(
        &mut self,
        source: &Value,
        wait_label: Option<&str>,
    ) -> Result<UploadValue, ConformanceError>;
    fn upload_video(
        &mut self,
        source: &Value,
        wait_label: Option<&str>,
        container: &str,
        codec: &str,
    ) -> Result<UploadValue, ConformanceError>;
    fn upload_audio(
        &mut self,
        source: &Value,
        container: &str,
        codec: &str,
        mime_type: &str,
    ) -> Result<UploadValue, ConformanceError>;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputKind {
    Camera,
    Image,
    Video,
    VideoStrings,
    VideoDuration,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseKind {
    Camera,
    Legacy,
    Modern,
    Turbo,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: Option<String>,
    pub body: Value,
    pub poll_base: Option<String>,
    pub response_kind: ResponseKind,
    pub output_kind: OutputKind,
    pub maximum_polls: u32,
    pub estimated_duration: Option<u64>,
    pub upload_receipts: Vec<Vec<u8>>,
}
fn base64_png(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::new();
    for chunk in bytes.chunks(3) {
        let first = chunk.first().copied().unwrap_or_default();
        let second = chunk.get(1).copied().unwrap_or_default();
        let third = chunk.get(2).copied().unwrap_or_default();
        for index in [first >> 2, ((first & 3) << 4) | (second >> 4)] {
            output.push(ALPHABET[index as usize] as char);
        }
        output.push(if chunk.len() > 1 {
            ALPHABET[(((second & 15) << 2) | (third >> 6)) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            ALPHABET[(third & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}
struct VideoBounds {
    minimum_width: u64,
    maximum_width: u64,
    minimum_height: Option<u64>,
    maximum_height: Option<u64>,
    minimum_duration: f64,
    maximum_duration: f64,
}
struct Builder<'a> {
    input: Input,
    owner: &'a mut dyn KlingMediaOwner,
    receipts: Vec<Vec<u8>>,
}
impl Builder<'_> {
    fn value(&self, name: &str) -> Result<Value, KlingError> {
        Ok(argument(&self.input, name)?.clone())
    }
    fn text(&self, name: &str) -> Result<String, KlingError> {
        Ok(text(argument(&self.input, name)?)?.into())
    }
    fn number(&self, name: &str, minimum: f64, maximum: f64) -> Result<f64, KlingError> {
        let value = self
            .value(name)?
            .as_f64()
            .ok_or(KlingError::Input("Expected Kling number"))?;
        if !value.is_finite() || !(minimum..=maximum).contains(&value) {
            return Err(KlingError::Input("Kling numeric bound exceeded"));
        }
        Ok(value)
    }
    fn optional(&self, name: &str) -> Option<Value> {
        self.input
            .arguments
            .get(name)
            .filter(|value| !value.is_null())
            .cloned()
    }
    fn integer(&self, name: &str, maximum: u64) -> Result<u64, KlingError> {
        integer(&self.value(name)?, maximum)
    }
    fn boolean(&self, name: &str) -> Result<bool, KlingError> {
        self.value(name)?
            .as_bool()
            .ok_or(KlingError::Input("Expected Kling boolean"))
    }
    fn validate_prompt(
        &self,
        name: &str,
        minimum: usize,
        maximum: usize,
        strip: bool,
    ) -> Result<String, KlingError> {
        let value = self.text(name)?;
        let length = if strip { source_trim(&value) } else { &value }
            .chars()
            .count();
        if minimum > 0 && length < minimum {
            return Err(KlingError::Provider(format!(
                "Field '{name}' cannot be shorter than {minimum} characters; was {length} characters long."
            )));
        }
        if length > maximum {
            return Err(KlingError::Provider(format!(
                " Field '{name}' cannot be longer than {maximum} characters; was {length} characters long."
            )));
        }
        Ok(value)
    }
    fn old_prompts(&self, maximum: usize) -> Result<(String, String), KlingError> {
        let prompt = self.text("prompt")?;
        let negative = self.text("negative_prompt")?;
        if prompt.is_empty() {
            return Err(KlingError::Provider("Positive prompt is empty".into()));
        }
        if prompt.chars().count() > maximum {
            return Err(KlingError::Provider(format!(
                "Positive prompt is too long: {} characters",
                prompt.chars().count()
            )));
        }
        if negative.chars().count() > maximum {
            return Err(KlingError::Provider(format!(
                "Negative prompt is too long: {} characters",
                negative.chars().count()
            )));
        }
        Ok((prompt, negative))
    }
    fn enumeration(&self, value: &Value, name: &str) -> Result<String, KlingError> {
        let value = text(value)?;
        if !oracle()?["enums"][name]
            .as_array()
            .ok_or(KlingError::Input("Missing source enum"))?
            .iter()
            .any(|member| member == value)
        {
            return Err(KlingError::Input("Unknown Kling source enum"));
        }
        Ok(value.into())
    }
    fn check_image(&mut self, source: &Value, minimum: u64) -> Result<ImageInfo, KlingError> {
        let info = self.owner.image_info(source)?;
        if info.width < minimum || info.height < minimum || info.height == 0 || info.width == 0 {
            return Err(KlingError::Input(
                "Image dimensions below Kling source bound",
            ));
        }
        let ratio = info.height as f64 / info.width as f64;
        if ratio <= 0.4 || ratio >= 2.5 {
            return Err(KlingError::Input(
                "Image aspect ratio outside strict source bounds",
            ));
        }
        Ok(info)
    }
    fn check_video(&mut self, source: &Value, bounds: VideoBounds) -> Result<(), KlingError> {
        let info = self.owner.video_info(source)?;
        if info.width < bounds.minimum_width
            || info.width > bounds.maximum_width
            || bounds
                .minimum_height
                .is_some_and(|minimum| info.height < minimum)
            || bounds
                .maximum_height
                .is_some_and(|maximum| info.height > maximum)
            || !info.duration_seconds.is_finite()
            || info.duration_seconds < bounds.minimum_duration - 0.0001
            || info.duration_seconds > bounds.maximum_duration + 0.0001
        {
            return Err(KlingError::Input("Kling video inventory bound exceeded"));
        }
        Ok(())
    }
    fn upload(
        &mut self,
        value: UploadValue,
        count: Option<usize>,
    ) -> Result<Vec<String>, KlingError> {
        if value.urls.is_empty()
            || value.urls.len() != value.receipts.len()
            || count.is_some_and(|count| value.urls.len() != count)
        {
            return Err(KlingError::Response("Invalid canonical upload result"));
        }
        for (url, receipt) in value.urls.iter().zip(&value.receipts) {
            endpoint(url)?;
            if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
                return Err(KlingError::Response("Invalid canonical upload receipt"));
            }
        }
        self.receipts.extend(value.receipts);
        Ok(value.urls)
    }
    fn inline(&mut self, source: &Value) -> Result<String, KlingError> {
        let bytes = self.owner.inline_png(source, 4194304)?;
        if bytes.is_empty() || bytes.len() > MAXIMUM_MEDIA_BYTES {
            return Err(KlingError::Input("Invalid canonical inline PNG"));
        }
        Ok(base64_png(&bytes))
    }
    fn image_upload(&mut self, source: &Value, label: Option<&str>) -> Result<String, KlingError> {
        let result = self.owner.upload_image(source, label)?;
        self.upload(result, Some(1))?
            .into_iter()
            .next()
            .ok_or(KlingError::Response("Missing uploaded image"))
    }
    fn images_upload(
        &mut self,
        source: &Value,
        label: Option<&str>,
    ) -> Result<Vec<String>, KlingError> {
        let info = self.owner.image_info(source)?;
        let count =
            usize::try_from(info.count).map_err(|_| KlingError::Input("Invalid image count"))?;
        let result = self.owner.upload_images(source, label)?;
        self.upload(result, Some(count))
    }
    fn video_upload(&mut self, source: &Value, label: Option<&str>) -> Result<String, KlingError> {
        let result = self.owner.upload_video(source, label, "mp4", "h264")?;
        self.upload(result, Some(1))?
            .into_iter()
            .next()
            .ok_or(KlingError::Response("Missing uploaded video"))
    }
    fn audio_upload(&mut self, source: &Value) -> Result<String, KlingError> {
        let result = self
            .owner
            .upload_audio(source, "mp3", "libmp3lame", "audio/mpeg")?;
        self.upload(result, Some(1))?
            .into_iter()
            .next()
            .ok_or(KlingError::Response("Missing uploaded audio"))
    }
    fn reference_list(&mut self, source: &Value, maximum: u64) -> Result<Vec<Value>, KlingError> {
        let info = self.check_image(source, 300)?;
        if info.count > maximum {
            return Err(KlingError::Input("Kling reference count exceeded"));
        }
        Ok(self
            .images_upload(source, Some("Uploading reference image"))?
            .into_iter()
            .map(|url| json!({"image_url":url}))
            .collect())
    }
    fn camera(&self, source: &Value, simple: bool) -> Result<Value, KlingError> {
        let mut source = source.clone();
        let control = self.enumeration(
            source
                .get("type")
                .ok_or(KlingError::Input("Missing camera type"))?,
            "KlingCameraControlType",
        )?;
        if !source["config"].is_object() {
            return Err(KlingError::Input("Missing camera config"));
        }
        for name in ["horizontal", "vertical", "pan", "tilt", "roll", "zoom"] {
            if let Some(value) = source["config"].get(name).filter(|value| !value.is_null()) {
                let number = value
                    .as_f64()
                    .ok_or(KlingError::Input("Invalid camera config"))?;
                if !(-10.0..=10.0).contains(&number) {
                    return Err(KlingError::Input("Camera config bound exceeded"));
                }
            }
        }
        source["type"] = json!(if simple { "simple" } else { &control });
        Ok(source)
    }
    fn legacy_video(&mut self, image: bool) -> Result<(Value, String), KlingError> {
        let (prompt, negative) = self.old_prompts(if image { 500 } else { 2500 })?;
        let mut mode = self
            .optional("mode")
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or("std".into());
        let mut duration = self
            .optional("duration")
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or("5".into());
        let mut model = self
            .optional("model_name")
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or("kling-v1".into());
        if self.input.node == "KlingTextToVideoNode" || self.input.node == "KlingStartEndFrameNode"
        {
            let key = if image {
                "start_end_modes"
            } else {
                "model_modes"
            };
            let source = oracle()?;
            let values = source[key]
                .get(self.text("mode")?)
                .and_then(Value::as_array)
                .ok_or(KlingError::Input("Unknown Kling mode"))?;
            mode = values
                .first()
                .and_then(Value::as_str)
                .ok_or(KlingError::Input("Missing mode"))?
                .into();
            duration = values
                .get(1)
                .and_then(Value::as_str)
                .ok_or(KlingError::Input("Missing duration"))?
                .into();
            model = values
                .get(2)
                .and_then(Value::as_str)
                .ok_or(KlingError::Input("Missing model"))?
                .into();
        }
        if self.input.node == "KlingCameraControlI2VNode" {
            model = "kling-v1-5".into();
            mode = "pro".into();
        }
        self.enumeration(&json!(duration), "KlingVideoGenDuration")?;
        self.enumeration(&json!(mode), "KlingVideoGenMode")?;
        self.enumeration(&self.value("aspect_ratio")?, "KlingVideoGenAspectRatio")?;
        let mut body = json!({"aspect_ratio":if image{"16:9".to_owned()}else{self.text("aspect_ratio")?},"cfg_scale":self.number("cfg_scale",0.0,1.0)?,"duration":duration,"mode":mode,"model_name":model,"prompt":prompt});
        if !negative.is_empty() {
            body["negative_prompt"] = json!(negative);
        }
        if let Some(camera) = self.optional("camera_control") {
            body["camera_control"] = self.camera(&camera, image)?;
        }
        if image {
            self.enumeration(&json!(model), "KlingVideoGenModelName")?;
            let source = self.value("start_frame")?;
            self.check_image(&source, 300)?;
            body["image"] = json!(self.inline(&source)?);
            if let Some(end) = self.optional("end_frame") {
                body["image_tail"] = json!(self.inline(&end)?);
            }
            if model == "kling-v2-5-turbo" && mode == "std" {
                body["mode"] = json!("pro");
            }
        }
        Ok((
            body,
            format!(
                "/proxy/kling/v1/videos/{}",
                if image { "image2video" } else { "text2video" }
            ),
        ))
    }
    fn request(mut self) -> Result<Request, KlingError> {
        let node = self.input.node.clone();
        let mut output = OutputKind::Video;
        let mut dialect = ResponseKind::Modern;
        let mut maximum_polls = 480;
        let mut estimated = None;
        let (body, path) = match node.as_str() {
            "KlingCameraControls" => {
                let mut config = json!({});
                for (name, argument) in [
                    ("horizontal", "horizontal_movement"),
                    ("vertical", "vertical_movement"),
                    ("pan", "pan"),
                    ("tilt", "tilt"),
                    ("roll", "roll"),
                    ("zoom", "zoom"),
                ] {
                    config[name] = json!(self.number(argument, -10.0, 10.0)?);
                }
                let control = self.enumeration(
                    &self.value("camera_control_type")?,
                    "KlingCameraControlType",
                )?;
                output = OutputKind::Camera;
                dialect = ResponseKind::Camera;
                (json!({"type":control,"config":config}), None)
            }
            "KlingTextToVideoNode"
            | "KlingCameraControlT2VNode"
            | "KlingImage2VideoNode"
            | "KlingCameraControlI2VNode"
            | "KlingStartEndFrameNode" => {
                let image = matches!(
                    node.as_str(),
                    "KlingImage2VideoNode" | "KlingCameraControlI2VNode" | "KlingStartEndFrameNode"
                );
                let (body, path) = self.legacy_video(image)?;
                output = OutputKind::VideoStrings;
                dialect = ResponseKind::Legacy;
                estimated = Some(if image { 164 } else { 319 });
                (body, Some(path))
            }
            "KlingVideoExtendNode" => {
                let (prompt, negative) = self.old_prompts(2500)?;
                let mut body = json!({"prompt":prompt,"cfg_scale":self.number("cfg_scale",0.0,1.0)?,"video_id":self.text("video_id")?});
                if !negative.is_empty() {
                    body["negative_prompt"] = json!(negative);
                }
                output = OutputKind::VideoStrings;
                dialect = ResponseKind::Legacy;
                estimated = Some(320);
                (body, Some("/proxy/kling/v1/videos/video-extend".into()))
            }
            "KlingDualCharacterVideoEffectNode" | "KlingSingleImageVideoEffectNode" => {
                let dual = node == "KlingDualCharacterVideoEffectNode";
                let scene = self.enumeration(
                    &self.value("effect_scene")?,
                    if dual {
                        "KlingDualCharacterEffectsScene"
                    } else {
                        "KlingSingleImageEffectsScene"
                    },
                )?;
                let model = self.enumeration(
                    &self.value("model_name")?,
                    if dual {
                        "KlingCharacterEffectModelName"
                    } else {
                        "KlingSingleImageEffectModelName"
                    },
                )?;
                let duration = self.enumeration(
                    &self.value("duration")?,
                    if dual {
                        "KlingVideoGenDuration"
                    } else {
                        "KlingSingleImageEffectDuration"
                    },
                )?;
                let mut input = json!({"model_name":model,"duration":duration});
                if dual {
                    let left = self.value("image_left")?;
                    let right = self.value("image_right")?;
                    input["images"] = json!([self.inline(&left)?, self.inline(&right)?]);
                    input["mode"] =
                        json!(self.enumeration(&self.value("mode")?, "KlingVideoGenMode")?);
                } else {
                    let source = self.value("image")?;
                    input["image"] = json!(self.inline(&source)?);
                }
                output = if dual {
                    OutputKind::VideoDuration
                } else {
                    OutputKind::VideoStrings
                };
                dialect = ResponseKind::Legacy;
                estimated = Some(320);
                (
                    json!({"effect_scene":scene,"input":input}),
                    Some("/proxy/kling/v1/videos/effects".into()),
                )
            }
            "KlingLipSyncAudioToVideoNode" | "KlingLipSyncTextToVideoNode" => {
                let text_mode = node == "KlingLipSyncTextToVideoNode";
                let voice = if text_mode {
                    Some(
                        oracle()?["voices"]
                            .get(self.text("voice")?)
                            .and_then(Value::as_array)
                            .cloned()
                            .ok_or(KlingError::Input("Unknown Kling voice"))?,
                    )
                } else {
                    None
                };
                let text = if text_mode {
                    Some(self.validate_prompt("text", 0, 120, true)?)
                } else {
                    None
                };
                let source = self.value("video")?;
                self.check_video(
                    &source,
                    VideoBounds {
                        minimum_width: 720,
                        maximum_width: 1920,
                        minimum_height: None,
                        maximum_height: None,
                        minimum_duration: 2.0,
                        maximum_duration: 10.0,
                    },
                )?;
                let video = self.video_upload(&source, None)?;
                let mut input = json!({"video_url":video,"audio_type":"url","mode":if text_mode{"text2video"}else{"audio2video"}});
                if text_mode {
                    let voice = voice
                        .as_ref()
                        .ok_or(KlingError::Input("Missing Kling voice"))?;
                    input["voice_id"] = voice
                        .first()
                        .cloned()
                        .ok_or(KlingError::Input("Missing voice identifier"))?;
                    input["voice_language"] = voice
                        .get(1)
                        .cloned()
                        .ok_or(KlingError::Input("Missing voice language"))?;
                    input["voice_speed"] = json!(self.number("voice_speed", 0.8, 2.0)?);
                    input["text"] = json!(text);
                } else {
                    let source = self.value("audio")?;
                    input["audio_url"] = json!(self.audio_upload(&source)?);
                    input["voice_language"] = json!(self.enumeration(
                        &self.value("voice_language")?,
                        "KlingLipSyncVoiceLanguage"
                    )?);
                }
                output = OutputKind::VideoStrings;
                dialect = ResponseKind::Legacy;
                estimated = Some(455);
                (
                    json!({"input":input}),
                    Some("/proxy/kling/v1/videos/lip-sync".into()),
                )
            }
            "KlingVirtualTryOnNode" => {
                let human = self.value("human_image")?;
                let cloth = self.value("cloth_image")?;
                let body = json!({"human_image":self.inline(&human)?,"cloth_image":self.inline(&cloth)?,"model_name":self.enumeration(&self.value("model_name")?,"KlingVirtualTryOnModelName")?});
                output = OutputKind::Image;
                dialect = ResponseKind::Legacy;
                estimated = Some(19);
                (
                    body,
                    Some("/proxy/kling/v1/images/kolors-virtual-try-on".into()),
                )
            }
            "KlingImageGenerationNode" => {
                let mut body = json!({"model_name":self.text("model_name")?,"prompt":self.validate_prompt("prompt",1,500,true)?,"negative_prompt":self.validate_prompt("negative_prompt",0,500,true)?,"image_fidelity":self.number("image_fidelity",0.0,1.0)?,"human_fidelity":self.number("human_fidelity",0.0,1.0)?,"n":self.integer("n",9)?,"aspect_ratio":self.enumeration(&self.value("aspect_ratio")?,"KlingImageGenAspectRatio")?});
                if body["n"] == 0 {
                    return Err(KlingError::Input("Invalid Kling image count"));
                }
                if let Some(image) = self.optional("image") {
                    body["image"] = json!(self.inline(&image)?);
                    body["image_reference"] = json!(self.enumeration(
                        &self.value("image_type")?,
                        "KlingImageGenImageReferenceType"
                    )?);
                }
                if body["negative_prompt"]
                    .as_str()
                    .is_some_and(|prompt| prompt.chars().count() > 200)
                {
                    return Err(KlingError::Input(
                        "Kling API negative prompt exceeds 200 characters",
                    ));
                }
                if body["prompt"]
                    .as_str()
                    .is_some_and(|prompt| prompt.chars().count() > 500)
                {
                    return Err(KlingError::Input(
                        "Kling API positive prompt exceeds 500 characters",
                    ));
                }
                output = OutputKind::Image;
                dialect = ResponseKind::Legacy;
                estimated = Some(32);
                (body, Some("/proxy/kling/v1/images/generations".into()))
            }
            "KlingTextToVideoWithAudio" | "KlingImageToVideoWithAudio" => {
                let image = node == "KlingImageToVideoWithAudio";
                let mut body = json!({"model_name":self.text("model_name")?,"prompt":self.validate_prompt("prompt",1,2500,true)?,"mode":self.text("mode")?,"duration":self.integer("duration",15)?.to_string(),"sound":if self.boolean("generate_audio")?{"on"}else{"off"}});
                if image {
                    let source = self.value("start_frame")?;
                    self.check_image(&source, 300)?;
                    body["image"] = json!(
                        self.images_upload(&source, None)?
                            .into_iter()
                            .next()
                            .ok_or(KlingError::Response("Missing first image"))?
                    );
                } else {
                    body["aspect_ratio"] = json!(self.text("aspect_ratio")?);
                }
                (
                    body,
                    Some(format!(
                        "/proxy/kling/v1/videos/{}",
                        if image { "image2video" } else { "text2video" }
                    )),
                )
            }
            "KlingMotionControl" => {
                let prompt = self.validate_prompt("prompt", 0, 2500, true)?;
                let image = self.value("reference_image")?;
                self.check_image(&image, 340)?;
                let video = self.value("reference_video")?;
                let orientation = self.text("character_orientation")?;
                self.check_video(
                    &video,
                    VideoBounds {
                        minimum_width: 340,
                        maximum_width: 3850,
                        minimum_height: Some(340),
                        maximum_height: Some(3850),
                        minimum_duration: 3.0,
                        maximum_duration: if orientation == "image" { 10.0 } else { 30.0 },
                    },
                )?;
                let image_url = self
                    .images_upload(&image, None)?
                    .into_iter()
                    .next()
                    .ok_or(KlingError::Response("Missing first reference"))?;
                let video_url = self.video_upload(&video, None)?;
                (
                    json!({"prompt":prompt,"image_url":image_url,"video_url":video_url,"keep_original_sound":if self.boolean("keep_original_sound")?{"yes"}else{"no"},"character_orientation":orientation,"mode":self.text("mode")?,"model_name":self.text("model")?}),
                    Some("/proxy/kling/v1/videos/motion-control".into()),
                )
            }
            "KlingAvatarNode" => {
                let image = self.value("image")?;
                self.check_image(&image, 300)?;
                let sound = self.value("sound_file")?;
                let info = self.owner.audio_info(&sound)?;
                if info.sample_rate == 0 {
                    return Err(KlingError::Input("Invalid sample rate"));
                }
                let duration = info.samples as f64 / info.sample_rate as f64;
                let epsilon = 1.0 / info.sample_rate as f64;
                if duration + epsilon < 2.0 || duration - epsilon > 300.0 {
                    return Err(KlingError::Input("Kling avatar audio duration exceeded"));
                }
                let mut body = json!({"image":self.image_upload(&image,None)?,"sound_file":self.audio_upload(&sound)?,"mode":self.text("mode")?});
                let prompt = self.text("prompt")?;
                if !prompt.is_empty() {
                    body["prompt"] = json!(prompt);
                }
                maximum_polls = 800;
                (
                    body,
                    Some("/proxy/kling/v1/videos/avatar/image2video".into()),
                )
            }
            "KlingOmniProTextToVideoNode"
            | "KlingOmniProFirstLastFrameNode"
            | "KlingOmniProImageToVideoNode"
            | "KlingOmniProVideoToVideoNode"
            | "KlingOmniProEditVideoNode"
            | "KlingOmniProImageNode" => {
                let (body, path) = self.omni()?;
                if node == "KlingOmniProImageNode" {
                    output = OutputKind::Image;
                }
                (body, Some(path))
            }
            "KlingVideoNode" | "KlingFirstLastFrameNode" => {
                let (body, path, turbo) = self.version_three()?;
                if turbo {
                    dialect = ResponseKind::Turbo;
                }
                (body, Some(path))
            }
            _ => return Err(KlingError::Input("Unimplemented Kling source node")),
        };
        let poll_base = if dialect == ResponseKind::Turbo {
            Some("/proxy/kling/tasks".into())
        } else {
            path.clone()
        };
        Ok(Request {
            path,
            body,
            poll_base,
            response_kind: dialect,
            output_kind: output,
            maximum_polls,
            estimated_duration: estimated,
            upload_receipts: self.receipts,
        })
    }
    fn stories(
        &self,
        source: Option<&Value>,
        selector: &str,
        expected: Option<u64>,
    ) -> Result<Option<Vec<Value>>, KlingError> {
        let Some(source) = source else {
            return Ok(None);
        };
        let mode = source
            .get(selector)
            .and_then(Value::as_str)
            .ok_or(KlingError::Input("Missing storyboard selector"))?;
        if mode == "disabled" {
            return Ok(None);
        }
        let count = mode
            .split_whitespace()
            .next()
            .ok_or(KlingError::Input("Missing storyboard count"))?
            .parse::<usize>()
            .map_err(|_| KlingError::Input("Invalid storyboard count"))?;
        if !(1..=6).contains(&count) {
            return Err(KlingError::Input("Invalid storyboard count"));
        }
        let mut entries = vec![];
        let mut total = 0u64;
        for index in 1..=count {
            let name = format!("storyboard_{index}_prompt");
            let prompt = source
                .get(&name)
                .and_then(Value::as_str)
                .ok_or(KlingError::Input("Missing storyboard prompt"))?;
            let length = source_trim(prompt).chars().count();
            if length == 0 || length > 512 {
                return Err(KlingError::Input("Storyboard prompt outside source bounds"));
            }
            let duration = source
                .get(format!("storyboard_{index}_duration"))
                .and_then(Value::as_u64)
                .filter(|value| (1..=15).contains(value))
                .ok_or(KlingError::Input("Invalid storyboard duration"))?;
            total = total
                .checked_add(duration)
                .ok_or(KlingError::Input("Storyboard duration overflow"))?;
            entries.push(json!({"index":index,"prompt":prompt,"duration":duration.to_string()}));
        }
        if let Some(expected) = expected {
            if total != expected {
                return Err(KlingError::Provider(format!(
                    "Total storyboard duration ({total}s) must equal the global duration ({expected}s)."
                )));
            }
        } else if !(3..=15).contains(&total) {
            return Err(KlingError::Provider(format!(
                "Total storyboard duration ({total}s) must be between 3 and 15 seconds."
            )));
        }
        Ok(Some(entries))
    }
    fn omni(&mut self) -> Result<(Value, String), KlingError> {
        let node = self.input.node.clone();
        let model = self.text("model_name")?;
        let resolution = self.text("resolution")?;
        if node == "KlingOmniProImageNode" {
            if model == "kling-image-o1" && resolution == "4K" {
                return Err(KlingError::Input(
                    "4K resolution is not supported for kling-image-o1 model.",
                ));
            }
            let prompt = normalize_omni_prompt(&self.text("prompt")?)?;
            if source_trim(&prompt).is_empty() || source_trim(&prompt).chars().count() > 2500 {
                return Err(KlingError::Input(
                    "Normalized Omni prompt outside source bound",
                ));
            }
            let mut image_list = vec![];
            if let Some(images) = self.optional("reference_images") {
                let info = self.check_image(&images, 300)?;
                if info.count > 10 {
                    return Err(KlingError::Input(
                        "The maximum number of reference images is 10.",
                    ));
                }
                image_list = self
                    .images_upload(&images, Some("Uploading reference image"))?
                    .into_iter()
                    .map(|image| json!({"image":image}))
                    .collect();
            }
            let series = self.text("series_amount")?;
            if series != "disabled" && model == "kling-image-o1" {
                return Err(KlingError::Input(
                    "kling-image-o1 does not support series generation.",
                ));
            }
            let mut body = json!({"model_name":model,"prompt":prompt,"resolution":resolution.to_lowercase(),"aspect_ratio":self.text("aspect_ratio")?,"mode":"pro","n":1});
            if !image_list.is_empty() {
                body["image_list"] = json!(image_list);
            }
            if series != "disabled" {
                body["result_type"] = json!("series");
                body["series_amount"] = json!(
                    series
                        .parse::<u64>()
                        .map_err(|_| KlingError::Input("Invalid series amount"))?
                );
            }
            return Ok((body, "/proxy/kling/v1/images/omni-image".into()));
        }
        let text_node = node == "KlingOmniProTextToVideoNode";
        let first_last = node == "KlingOmniProFirstLastFrameNode";
        let references = node == "KlingOmniProImageToVideoNode";
        let video_node = node == "KlingOmniProVideoToVideoNode";
        let edit = node == "KlingOmniProEditVideoNode";
        let duration = if edit {
            None
        } else {
            Some(self.integer("duration", 15)?)
        };
        let stories_source = self.optional("storyboards");
        let stories_enabled = stories_source
            .as_ref()
            .is_some_and(|value| value["storyboards"] != "disabled");
        let sound = if self.input.arguments.contains_key("generate_audio") {
            self.boolean("generate_audio")?
        } else {
            false
        };
        if model == "kling-video-o1" && (text_node || first_last || references) {
            if text_node && !duration.is_some_and(|duration| [5, 10].contains(&duration)) {
                return Err(KlingError::Input(
                    "kling-video-o1 only supports durations of 5 or 10 seconds.",
                ));
            }
            if !text_node && duration.is_some_and(|duration| duration > 10) {
                return Err(KlingError::Input(
                    "kling-video-o1 does not support durations greater than 10 seconds.",
                ));
            }
            if sound {
                return Err(KlingError::Input(
                    "kling-video-o1 does not support audio generation.",
                ));
            }
            if resolution == "4k" {
                return Err(KlingError::Input(
                    "kling-video-o1 does not support 4k resolution.",
                ));
            }
            if stories_enabled {
                return Err(KlingError::Input(
                    "kling-video-o1 does not support storyboards.",
                ));
            }
        }
        let prompt = if text_node {
            self.text("prompt")?
        } else {
            normalize_omni_prompt(&self.text("prompt")?)?
        };
        let length = source_trim(&prompt).chars().count();
        if (!stories_enabled && length == 0) || length > 2500 {
            return Err(KlingError::Input(
                "Normalized Omni prompt outside source bound",
            ));
        }
        let end = self.optional("end_frame");
        let images = self.optional("reference_images");
        if first_last {
            if end.is_some() && images.is_some() {
                return Err(KlingError::Input(
                    "The 'end_frame' input cannot be used simultaneously with 'reference_images'.",
                ));
            }
            if end.is_some() && stories_enabled {
                return Err(KlingError::Input(
                    "The 'end_frame' input cannot be used simultaneously with storyboards.",
                ));
            }
            if model == "kling-video-o1"
                && !duration.is_some_and(|duration| [5, 10].contains(&duration))
                && end.is_none()
                && images.is_none()
            {
                return Err(KlingError::Input(
                    "Duration is only supported for 5 or 10 seconds if there is no end frame or reference images.",
                ));
            }
        }
        let mut body = json!({"model_name":model,"prompt":prompt,"mode":if resolution=="4k"&&!video_node&&!edit{"4k"}else if resolution=="1080p"{"pro"}else{"std"}});
        if let Some(duration) = duration {
            body["duration"] = json!(duration.to_string());
        }
        if !first_last && !edit {
            body["aspect_ratio"] = json!(self.text("aspect_ratio")?);
        }
        if text_node || first_last || references {
            body["sound"] = json!(if sound { "on" } else { "off" });
            if let Some(entries) = self.stories(stories_source.as_ref(), "storyboards", duration)? {
                body["multi_shot"] = json!(true);
                body["multi_prompt"] = json!(entries);
                body["shot_type"] = json!("customize");
            }
        }
        if first_last {
            let first = self.value("first_frame")?;
            self.check_image(&first, 300)?;
            let first_url = self
                .images_upload(&first, Some("Uploading first frame"))?
                .into_iter()
                .next()
                .ok_or(KlingError::Response("Missing first frame"))?;
            let mut image_list = vec![json!({"image_url":first_url,"type":"first_frame"})];
            if let Some(end) = end {
                self.check_image(&end, 300)?;
                let url = self
                    .images_upload(&end, Some("Uploading end frame"))?
                    .into_iter()
                    .next()
                    .ok_or(KlingError::Response("Missing end frame"))?;
                image_list.push(json!({"image_url":url,"type":"end_frame"}));
            }
            if let Some(images) = images {
                let info = self.check_image(&images, 300)?;
                if info.count > 6 {
                    return Err(KlingError::Input(
                        "The maximum number of reference images allowed is 6.",
                    ));
                }
                image_list.extend(
                    self.images_upload(&images, Some("Uploading reference frame(s)"))?
                        .into_iter()
                        .map(|url| json!({"image_url":url})),
                );
            }
            body["image_list"] = json!(image_list);
        } else if references {
            let images = self.value("reference_images")?;
            body["image_list"] = json!(self.reference_list(&images, 7)?);
        } else if video_node || edit {
            let video = self.value(if edit { "video" } else { "reference_video" })?;
            self.check_video(
                &video,
                VideoBounds {
                    minimum_width: 720,
                    maximum_width: 2160,
                    minimum_height: Some(720),
                    maximum_height: Some(2160),
                    minimum_duration: 3.0,
                    maximum_duration: 10.05,
                },
            )?;
            if let Some(images) = images {
                body["image_list"] = json!(self.reference_list(&images, 4)?);
            }
            let video_url = self.video_upload(
                &video,
                Some(if edit {
                    "Uploading base video"
                } else {
                    "Uploading reference video"
                }),
            )?;
            body["video_list"] = json!([{"video_url":video_url,"refer_type":if edit{"base"}else{"feature"},"keep_original_sound":if self.boolean("keep_original_sound")?{"yes"}else{"no"}}]);
        }
        Ok((body, "/proxy/kling/v1/videos/omni-video".into()))
    }
    fn version_three(&mut self) -> Result<(Value, String, bool), KlingError> {
        let first_last = self.input.node == "KlingFirstLastFrameNode";
        let model = self.value("model")?;
        let name = model["model"]
            .as_str()
            .ok_or(KlingError::Input("Missing Kling model"))?;
        let resolution = model["resolution"]
            .as_str()
            .ok_or(KlingError::Input("Missing Kling resolution"))?;
        let mode = if resolution == "4k" {
            "4k"
        } else if resolution == "1080p" {
            "pro"
        } else {
            "std"
        };
        if first_last {
            let prompt = self.validate_prompt("prompt", 1, 2500, true)?;
            let first = self.value("first_frame")?;
            let end = self.value("end_frame")?;
            self.check_image(&first, 300)?;
            self.check_image(&end, 300)?;
            let image = self.image_upload(&first, Some("Uploading first frame"))?;
            let tail = self.image_upload(&end, Some("Uploading end frame"))?;
            return Ok((
                json!({"model_name":name,"prompt":prompt,"image":image,"image_tail":tail,"mode":mode,"duration":self.integer("duration",15)?.to_string(),"sound":if self.boolean("generate_audio")?{"on"}else{"off"}}),
                "/proxy/kling/v1/videos/image2video".into(),
                false,
            ));
        }
        let shots = self.value("multi_shot")?;
        let entries = self.stories(Some(&shots), "multi_shot", None)?;
        let duration = if let Some(entries) = &entries {
            entries.iter().try_fold(0u64, |sum, entry| {
                let value = entry["duration"]
                    .as_str()
                    .ok_or(KlingError::Input("Invalid duration"))?
                    .parse::<u64>()
                    .map_err(|_| KlingError::Input("Invalid duration"))?;
                sum.checked_add(value)
                    .ok_or(KlingError::Input("Duration overflow"))
            })?
        } else {
            let prompt = shots["prompt"]
                .as_str()
                .ok_or(KlingError::Input("Missing multi-shot prompt"))?;
            if source_trim(prompt).is_empty() || source_trim(prompt).chars().count() > 2500 {
                return Err(KlingError::Input("Multi-shot prompt outside source bounds"));
            }
            shots["duration"]
                .as_u64()
                .ok_or(KlingError::Input("Invalid shot duration"))?
        };
        let frame = self.optional("start_frame");
        if name == "kling-3.0-turbo" {
            let prompt = if let Some(entries) = &entries {
                let mut prompt = String::new();
                for (index, entry) in entries.iter().enumerate() {
                    if index != 0 {
                        prompt.push_str("; ");
                    }
                    prompt.push_str(&format!(
                        "shot {}, {}, {}",
                        index + 1,
                        entry["duration"]
                            .as_str()
                            .ok_or(KlingError::Input("Invalid duration"))?,
                        entry["prompt"]
                            .as_str()
                            .ok_or(KlingError::Input("Invalid prompt"))?
                    ));
                }
                prompt.push(';');
                prompt
            } else {
                shots["prompt"]
                    .as_str()
                    .ok_or(KlingError::Input("Missing shot prompt"))?
                    .into()
            };
            let mut settings = json!({"resolution":resolution,"duration":duration});
            if let Some(frame) = frame {
                self.check_image(&frame, 300)?;
                let image = self.inline(&frame)?;
                let mut contents = vec![json!({"type":"first_frame","url":image})];
                if !prompt.is_empty() {
                    contents.insert(0, json!({"type":"prompt","text":prompt}));
                }
                return Ok((
                    json!({"contents":contents,"settings":settings}),
                    "/proxy/kling/image-to-video/kling-3.0-turbo".into(),
                    true,
                ));
            }
            settings["aspect_ratio"] = model
                .get("aspect_ratio")
                .cloned()
                .ok_or(KlingError::Input("Missing aspect ratio"))?;
            return Ok((
                json!({"prompt":prompt,"settings":settings}),
                "/proxy/kling/text-to-video/kling-3.0-turbo".into(),
                true,
            ));
        }
        let mut body = json!({"model_name":name,"mode":mode,"duration":duration.to_string(),"sound":if self.boolean("generate_audio")?{"on"}else{"off"}});
        if let Some(entries) = entries {
            body["multi_shot"] = json!(true);
            body["multi_prompt"] = json!(entries);
            body["shot_type"] = json!("customize");
        } else {
            body["prompt"] = shots
                .get("prompt")
                .cloned()
                .ok_or(KlingError::Input("Missing shot prompt"))?;
            body["negative_prompt"] = shots
                .get("negative_prompt")
                .cloned()
                .ok_or(KlingError::Input("Missing negative prompt"))?;
        }
        let path = if let Some(frame) = frame {
            self.check_image(&frame, 300)?;
            body["image"] = json!(self.image_upload(&frame, Some("Uploading start frame"))?);
            "/proxy/kling/v1/videos/image2video"
        } else {
            body["aspect_ratio"] = model
                .get("aspect_ratio")
                .cloned()
                .ok_or(KlingError::Input("Missing aspect ratio"))?;
            "/proxy/kling/v1/videos/text2video"
        };
        Ok((body, path.into(), false))
    }
}
fn unicode_class(character: char, ranges: &Value) -> Result<bool, KlingError> {
    let code = character as u64;
    let ranges = ranges
        .as_array()
        .ok_or(KlingError::Input("Missing pinned Unicode class"))?;
    let position = ranges.partition_point(|range| {
        range
            .get(1)
            .and_then(Value::as_u64)
            .is_some_and(|end| end < code)
    });
    Ok(ranges.get(position).is_some_and(|range| {
        range
            .get(0)
            .and_then(Value::as_u64)
            .is_some_and(|start| start <= code)
            && range
                .get(1)
                .and_then(Value::as_u64)
                .is_some_and(|end| end >= code)
    }))
}
pub fn normalize_omni_prompt(prompt: &str) -> Result<String, KlingError> {
    let contract = oracle()?;
    let characters: Vec<char> = prompt.chars().collect();
    let mut output = String::new();
    let mut index = 0;
    while let Some(character) = characters.get(index).copied() {
        if character == '@'
            && (index == 0
                || !unicode_class(
                    characters
                        .get(index - 1)
                        .copied()
                        .ok_or(KlingError::Input("Invalid prompt boundary"))?,
                    &contract["unicode_word_ranges"],
                )?)
        {
            let mut matched = false;
            for kind in ["image", "video"] {
                let start = index + 1;
                let end = start + kind.chars().count();
                if characters
                    .get(start..end)
                    .is_some_and(|slice| slice.iter().copied().eq(kind.chars()))
                {
                    let mut final_index = end;
                    while let Some(character) = characters.get(final_index) {
                        if !unicode_class(*character, &contract["unicode_decimal_ranges"])? {
                            break;
                        }
                        final_index += 1;
                    }
                    if characters.get(final_index).is_none()
                        || !unicode_class(
                            characters
                                .get(final_index)
                                .copied()
                                .ok_or(KlingError::Input("Invalid prompt boundary"))?,
                            &contract["unicode_word_ranges"],
                        )?
                    {
                        let digits = characters
                            .get(end..final_index)
                            .ok_or(KlingError::Input("Invalid prompt index"))?
                            .iter()
                            .collect::<String>();
                        output.push_str(&format!(
                            "<<<{kind}_{}>>>",
                            if digits.is_empty() { "1" } else { &digits }
                        ));
                        index = final_index;
                        matched = true;
                        break;
                    }
                }
            }
            if matched {
                continue;
            }
        }
        output.push(character);
        index += 1;
    }
    Ok(output)
}
pub fn request(input: &Input, owner: &mut dyn KlingMediaOwner) -> Result<Request, KlingError> {
    Builder {
        input: normalized(input)?,
        owner,
        receipts: vec![],
    }
    .request()
}

#[derive(Clone, Debug)]
pub struct DownloadedMedia {
    pub url: String,
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub creation_bytes: Vec<u8>,
    pub creation_receipt: Vec<u8>,
    pub response_bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub media: Vec<DownloadedMedia>,
    pub strings: Vec<String>,
    pub camera_control: Option<Value>,
    pub color_mode: &'static str,
}
fn task_identifier(value: &Value, kind: ResponseKind) -> Result<String, KlingError> {
    let identifier = value["data"][if kind == ResponseKind::Turbo {
        "id"
    } else {
        "task_id"
    }]
    .as_str()
    .filter(|value| !value.is_empty())
    .ok_or(KlingError::Response("Missing Kling task identifier"))?;
    if identifier.len() > 1024
        || !identifier
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(KlingError::Response("Invalid Kling task identifier"));
    }
    Ok(identifier.into())
}
fn source_status(value: &Value, kind: ResponseKind) -> Result<Option<String>, KlingError> {
    let status = if kind == ResponseKind::Turbo {
        &value["data"][0]["status"]
    } else {
        &value["data"]["task_status"]
    };
    if status.is_null() {
        return Ok(None);
    }
    Ok(Some(
        status
            .as_str()
            .ok_or(KlingError::Response("Invalid task status"))?
            .trim()
            .to_lowercase(),
    ))
}
fn response_urls(
    value: &Value,
    kind: ResponseKind,
    output: OutputKind,
) -> Result<(Vec<String>, Vec<String>), KlingError> {
    if kind == ResponseKind::Turbo {
        let rows = value["data"]
            .as_array()
            .and_then(|rows| rows.first())
            .and_then(|row| row["outputs"].as_array())
            .ok_or(KlingError::Response("Missing Turbo outputs"))?;
        let url = rows
            .iter()
            .find(|row| {
                row["type"] == "video" && row["url"].as_str().is_some_and(|url| !url.is_empty())
            })
            .and_then(|row| row["url"].as_str())
            .ok_or(KlingError::Response(
                "Kling 3.0 Turbo task finished without a video output",
            ))?;
        endpoint(url)?;
        return Ok((vec![url.into()], vec![]));
    }
    if output == OutputKind::Image {
        let result = &value["data"]["task_result"];
        let images = result["series_images"]
            .as_array()
            .filter(|values| !values.is_empty())
            .or_else(|| result["images"].as_array())
            .filter(|values| !values.is_empty())
            .ok_or(KlingError::Response("Kling completed without images"))?;
        let mut urls = vec![];
        for image in images {
            let url = image["url"]
                .as_str()
                .ok_or(KlingError::Response("Missing image URL"))?;
            endpoint(url)?;
            urls.push(url.into());
        }
        return Ok((urls, vec![]));
    }
    let video = value["data"]["task_result"]["videos"]
        .as_array()
        .and_then(|videos| videos.first())
        .ok_or(KlingError::Response("Kling completed without video data"))?;
    let url = video["url"]
        .as_str()
        .ok_or(KlingError::Response("Missing video URL"))?;
    endpoint(url)?;
    let string = |name: &str| -> Result<String, KlingError> {
        match video.get(name) {
            Some(Value::String(value)) => Ok(value.clone()),
            None | Some(Value::Null) => Ok("None".into()),
            _ => Err(KlingError::Response("Invalid video string metadata")),
        }
    };
    let strings = match output {
        OutputKind::VideoStrings => vec![string("id")?, string("duration")?],
        OutputKind::VideoDuration => vec![string("duration")?],
        _ => vec![],
    };
    Ok((vec![url.into()], strings))
}
fn relative_route(input: &Input) -> Result<Option<String>, KlingError> {
    let route = match input.node.as_str() {
        "KlingCameraControls" => return Ok(None),
        "KlingTextToVideoNode" | "KlingCameraControlT2VNode" | "KlingTextToVideoWithAudio" => {
            "/proxy/kling/v1/videos/text2video"
        }
        "KlingImage2VideoNode"
        | "KlingCameraControlI2VNode"
        | "KlingStartEndFrameNode"
        | "KlingFirstLastFrameNode"
        | "KlingImageToVideoWithAudio" => "/proxy/kling/v1/videos/image2video",
        "KlingVideoExtendNode" => "/proxy/kling/v1/videos/video-extend",
        "KlingDualCharacterVideoEffectNode" | "KlingSingleImageVideoEffectNode" => {
            "/proxy/kling/v1/videos/effects"
        }
        "KlingLipSyncAudioToVideoNode" | "KlingLipSyncTextToVideoNode" => {
            "/proxy/kling/v1/videos/lip-sync"
        }
        "KlingVirtualTryOnNode" => "/proxy/kling/v1/images/kolors-virtual-try-on",
        "KlingImageGenerationNode" => "/proxy/kling/v1/images/generations",
        "KlingMotionControl" => "/proxy/kling/v1/videos/motion-control",
        "KlingAvatarNode" => "/proxy/kling/v1/videos/avatar/image2video",
        "KlingOmniProImageNode" => "/proxy/kling/v1/images/omni-image",
        "KlingOmniProTextToVideoNode"
        | "KlingOmniProFirstLastFrameNode"
        | "KlingOmniProImageToVideoNode"
        | "KlingOmniProVideoToVideoNode"
        | "KlingOmniProEditVideoNode" => "/proxy/kling/v1/videos/omni-video",
        "KlingVideoNode" => {
            if argument(input, "model")?["model"] == "kling-3.0-turbo" {
                if input
                    .arguments
                    .get("start_frame")
                    .is_some_and(|value| !value.is_null())
                {
                    "/proxy/kling/image-to-video/kling-3.0-turbo"
                } else {
                    "/proxy/kling/text-to-video/kling-3.0-turbo"
                }
            } else if input
                .arguments
                .get("start_frame")
                .is_some_and(|value| !value.is_null())
            {
                "/proxy/kling/v1/videos/image2video"
            } else {
                "/proxy/kling/v1/videos/text2video"
            }
        }
        _ => return Err(KlingError::Input("Unknown Kling node")),
    };
    Ok(Some(route.into()))
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn KlingTransport,
    owner: &mut dyn KlingMediaOwner,
) -> Result<ResultValue, KlingError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    let Some(path) = relative_route(&input)? else {
        let plan = request(&input, owner)?;
        let camera = plan.body.clone();
        return Ok(ResultValue {
            plan,
            creation_bytes: vec![],
            creation_receipt: vec![],
            response_bytes: vec![],
            receipt: vec![],
            approval_receipt: vec![],
            media: vec![],
            strings: vec![],
            camera_control: Some(camera),
            color_mode: IMAGE_COLOR_MODE,
        });
    };
    let head = request_head(&path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "kling.generation.maximum".into(),
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
    let plan = request(&input, owner)?;
    if plan.path.as_deref() != Some(&path) {
        return Err(KlingError::Input("Kling request route drift"));
    }
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
    let creation: Value = serde_json::from_slice(&creation_bytes)?;
    if plan.response_kind == ResponseKind::Modern
        && creation["code"].as_i64().is_some_and(|code| code != 0)
    {
        return Err(KlingError::Provider(format!(
            "Kling request failed. Code: {}, Message: {}",
            creation["code"], creation["message"]
        )));
    }
    let identifier = task_identifier(&creation, plan.response_kind)?;
    let base = plan
        .poll_base
        .as_deref()
        .ok_or(KlingError::Response("Missing poll path"))?;
    let poll_path = if plan.response_kind == ResponseKind::Turbo {
        format!("{base}?task_ids={identifier}")
    } else {
        format!("{base}/{identifier}")
    };
    let mut active = 0u32;
    for sequence in 0..4096 {
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
        let (response_bytes, receipt, handle) = exchange_with_policy(
            &head,
            &[],
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            transport,
        )?;
        let response: Value = serde_json::from_slice(&response_bytes)?;
        let status = source_status(&response, plan.response_kind)?;
        let source = oracle()?;
        let member = |name: &str| {
            source["polling"][name].as_array().is_some_and(|values| {
                status
                    .as_deref()
                    .is_some_and(|status| values.iter().any(|value| value.as_str() == Some(status)))
            })
        };
        if member("FAILED_STATUSES") {
            return Err(KlingError::Provider(format!(
                "Task failed: {}",
                serde_json::to_string(&response)?
            )));
        }
        if member("COMPLETED_STATUSES") {
            let (urls, strings) = response_urls(&response, plan.response_kind, plan.output_kind)?;
            let mut media = vec![];
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
                    return Err(KlingError::Response("Empty canonical media download"));
                }
                media.push(DownloadedMedia {
                    url,
                    bytes,
                    receipt,
                });
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
                response_bytes,
                receipt,
                approval_receipt: approval.receipt,
                media,
                strings,
                camera_control: None,
                color_mode: IMAGE_COLOR_MODE,
            });
        }
        if !member("QUEUED_STATUSES") {
            active += 1;
        }
        if active >= plan.maximum_polls || sequence == 4095 {
            return Err(ConformanceError::PollLimit.into());
        }
        transport.report_progress(
            handle,
            &Progress {
                sequence: sequence as u64,
                completed: 0,
                total: 100,
                message: status,
            },
        )?;
        transport.check_cancelled(Some(handle))?;
        transport.poll_delay(handle, 5000)?;
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
pub trait KlingMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
    fn camera_control_output(&mut self) -> Result<ExpectedOutput, ConformanceError>;
}
pub struct MaterializationOwners<'a> {
    pub media: &'a mut dyn KlingMediaOwner,
    pub materializer: &'a mut dyn KlingMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn KlingTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), KlingError> {
    let result = invoke(
        input,
        credentials,
        maximum_cost_microunits,
        transport,
        owners.media,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.materializer.materialize(&result)?;
    let expected = if result.plan.output_kind == OutputKind::Camera {
        vec![owners.materializer.camera_control_output()?]
    } else {
        let registry = TypeRegistry::built_in()
            .map_err(|_| KlingError::Response("Missing canonical type registry"))?;
        let mut types = vec![if result.plan.output_kind == OutputKind::Image {
            "IMAGE"
        } else {
            "VIDEO"
        }];
        types.extend(vec!["STRING"; result.strings.len()]);
        let mut expected = vec![];
        for (index, name) in types.into_iter().enumerate() {
            let type_id = registry
                .resolve(name)
                .map_err(|_| KlingError::Response("Missing canonical output type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| KlingError::Response("Missing canonical output family"))?;
            expected.push(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            });
        }
        expected
    };
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, owners.sink).map_err(ConformanceError::from)?;
    Ok(())
}

fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn KlingTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), KlingError> {
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
    transport: &mut dyn KlingTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), KlingError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(KlingError::Http {
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
                    return Err(KlingError::Http {
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
    transport: &mut dyn KlingTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), KlingError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "kling.generation.maximum".into(),
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
                    return Err(KlingError::Http {
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
fn source_png() -> Vec<u8> {
    include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/kling/downloaded-image.png"
    ))
    .to_vec()
}
impl SourceProbeOwner {
    fn upload(
        &mut self,
        operation: &str,
        source: &Value,
        label: Option<&str>,
        urls: Vec<String>,
    ) -> Result<UploadValue, ConformanceError> {
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        let mut value = json!({"operation":operation});
        value[if operation == "upload_video" {
            "video"
        } else {
            "image"
        }] = source.clone();
        if let Some(label) = label {
            value["wait_label"] = json!(label);
        }
        if operation == "upload_video" {
            value["container"] = json!("mp4");
            value["codec"] = json!("h264");
        }
        self.operations.push(value);
        Ok(UploadValue {
            receipts: urls.iter().map(|_| vec![1, 2, 3]).collect(),
            urls,
        })
    }
    fn slot<'a>(source: &'a Value, key: &str) -> Result<&'a str, ConformanceError> {
        source[key].as_str().ok_or(ConformanceError::InvalidRequest)
    }
}
impl KlingMediaOwner for SourceProbeOwner {
    fn image_info(&mut self, source: &Value) -> Result<ImageInfo, ConformanceError> {
        let shape = source["shape"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        let dimension = |index| {
            shape
                .get(index)
                .and_then(Value::as_u64)
                .ok_or(ConformanceError::InvalidRequest)
        };
        match shape.len() {
            4 => Ok(ImageInfo {
                count: dimension(0)?,
                height: dimension(1)?,
                width: dimension(2)?,
            }),
            3 => Ok(ImageInfo {
                count: 1,
                height: dimension(0)?,
                width: dimension(1)?,
            }),
            _ => Err(ConformanceError::InvalidRequest),
        }
    }
    fn video_info(&mut self, source: &Value) -> Result<VideoInfo, ConformanceError> {
        Ok(VideoInfo {
            width: source["width"]
                .as_u64()
                .ok_or(ConformanceError::InvalidRequest)?,
            height: source["height"]
                .as_u64()
                .ok_or(ConformanceError::InvalidRequest)?,
            duration_seconds: source["duration"]
                .as_f64()
                .ok_or(ConformanceError::InvalidRequest)?,
        })
    }
    fn audio_info(&mut self, source: &Value) -> Result<AudioInfo, ConformanceError> {
        Ok(AudioInfo {
            sample_rate: source["sample_rate"]
                .as_u64()
                .ok_or(ConformanceError::InvalidRequest)?,
            samples: source["waveform"]["shape"]
                .as_array()
                .and_then(|shape| shape.last())
                .and_then(Value::as_u64)
                .ok_or(ConformanceError::InvalidRequest)?,
        })
    }
    fn inline_png(
        &mut self,
        source: &Value,
        total_pixels: u64,
    ) -> Result<Vec<u8>, ConformanceError> {
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        self.operations.push(json!({"operation":"inline_image","image":source,"mime_type":"image/png","total_pixels":total_pixels}));
        Ok(source_png())
    }
    fn upload_image(
        &mut self,
        source: &Value,
        label: Option<&str>,
    ) -> Result<UploadValue, ConformanceError> {
        let url = format!(
            "https://storage.fixture.invalid/{}",
            Self::slot(source, "native_slot")?
        );
        self.upload("upload_image", source, label, vec![url])
    }
    fn upload_images(
        &mut self,
        source: &Value,
        label: Option<&str>,
    ) -> Result<UploadValue, ConformanceError> {
        let count = self.image_info(source)?.count;
        if count > 16 {
            return Err(ConformanceError::InvalidRequest);
        }
        let slot = Self::slot(source, "native_slot")?;
        let urls = (0..count)
            .map(|index| format!("https://storage.fixture.invalid/{slot}/{index}"))
            .collect();
        self.upload("upload_images", source, label, urls)
    }
    fn upload_video(
        &mut self,
        source: &Value,
        label: Option<&str>,
        container: &str,
        codec: &str,
    ) -> Result<UploadValue, ConformanceError> {
        if container != "mp4" || codec != "h264" {
            return Err(ConformanceError::InvalidRequest);
        }
        let url = format!(
            "https://storage.fixture.invalid/{}",
            Self::slot(source, "native_video")?
        );
        self.upload("upload_video", source, label, vec![url])
    }
    fn upload_audio(
        &mut self,
        source: &Value,
        container: &str,
        codec: &str,
        mime_type: &str,
    ) -> Result<UploadValue, ConformanceError> {
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        self.operations.push(json!({"operation":"upload_audio","audio":source,"container_format":container,"codec_name":codec,"mime_type":mime_type}));
        Ok(UploadValue {
            urls: vec!["https://storage.fixture.invalid/audio".into()],
            receipts: vec![vec![1, 2, 3]],
        })
    }
}
fn source_trim(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"scope":"source request and bounded transport; Python prompt whitespace, raw positive-prompt500 API boundary and lip-sync source preparation order; canonical native media and guest integration pending"})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
    if source["source_sha256"]["comfy_api_nodes/nodes_kling.py"] != SOURCE_SHA256 {
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
        let requests = case["requests"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if requests.is_empty() {
            if plan.output_kind != OutputKind::Camera
                || plan.path.is_some()
                || case["outputs"] != json!([plan.body])
            {
                return Err(ConformanceError::InvalidRequest);
            }
        } else {
            let expected = requests.first().ok_or(ConformanceError::InvalidRequest)?;
            let poll = requests.get(1).ok_or(ConformanceError::InvalidRequest)?;
            let path = if plan.response_kind == ResponseKind::Turbo {
                plan.poll_base.clone()
            } else {
                plan.poll_base
                    .as_ref()
                    .map(|base| format!("{base}/source-task-42"))
            };
            if plan.path.as_deref() != expected["path"].as_str()
                || plan.body != expected["body"]
                || path.as_deref() != poll["path"].as_str()
                || plan.estimated_duration != poll["options"]["estimated_duration"].as_u64()
                || plan.maximum_polls
                    != poll["options"]["max_poll_attempts"].as_u64().unwrap_or(480) as u32
            {
                return Err(ConformanceError::InvalidRequest);
            }
            if plan.response_kind == ResponseKind::Turbo
                && poll["query_params"] != json!({"task_ids":"source-task-42"})
            {
                return Err(ConformanceError::InvalidRequest);
            }
        }
        if json!(owner.operations) != case["media"] {
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
        if request(&input, &mut SourceProbeOwner::default()).is_ok()
            || case["requests"] != json!([])
        {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    if covered
        != NODES
            .iter()
            .map(|node| node.node_identifier.to_owned())
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
    impl KlingTransport for Transport {
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
        fn poll_delay(
            &mut self,
            _handle: StreamHandle,
            delay: u64,
        ) -> Result<(), ConformanceError> {
            self.delays.push(delay);
            Ok(())
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
            "/../../sim_test_support/fixtures/providers/kling/downloaded-image.png"
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
    fn case(node: &str) -> Result<Value, Box<dyn Error>> {
        oracle()?["cases"]
            .as_array()
            .and_then(|cases| cases.iter().find(|case| case["input"]["node"] == node))
            .cloned()
            .ok_or_else(|| "source case missing".into())
    }
    fn input(node: &str) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(case(node)?["input"].clone())?)
    }
    fn source_events(
        node: &str,
        downloads: bool,
    ) -> Result<VecDeque<Result<ResponseEvent, ConformanceError>>, Box<dyn Error>> {
        let case = case(node)?;
        let mut sequence = events(
            200,
            serde_json::to_vec(&case["requests"][0]["response"])?,
            vec![],
        );
        sequence.extend(events(
            200,
            serde_json::to_vec(&case["requests"][1]["response"])?,
            vec![],
        ));
        if downloads {
            sequence.extend(events(200, png(), vec![]));
            sequence.extend(events(200, png(), vec![]));
        }
        Ok(sequence)
    }
    #[test]
    fn kling_source_oracle_covers_exact_claims() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/kling/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 25);
        assert_eq!(ROUTES.len(), 34);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition != "executable")
                .count(),
            7
        );
        Ok(())
    }
    #[test]
    fn each_source_request_matches_body_poll_options_and_media_order() -> Result<(), Box<dyn Error>>
    {
        for case in oracle()?["cases"].as_array().ok_or("missing cases")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            let plan = request(&input, &mut owner)?;
            if plan.path.is_some() {
                assert_eq!(plan.body, case["requests"][0]["body"], "{}", input.node);
                assert_eq!(
                    plan.path.as_deref(),
                    case["requests"][0]["path"].as_str(),
                    "{}",
                    input.node
                );
                assert_eq!(
                    plan.estimated_duration,
                    case["requests"][1]["options"]["estimated_duration"].as_u64(),
                    "{}",
                    input.node
                );
            }
            assert_eq!(json!(owner.operations), case["media"], "{}", input.node);
        }
        Ok(())
    }
    #[test]
    fn camera_control_requires_no_transport_auth_or_cost() -> Result<(), Box<dyn Error>> {
        let mut transport = Transport::default();
        let value = invoke(
            &input("KlingCameraControls")?,
            CredentialAvailability {
                auth_token: false,
                api_key: false,
            },
            9,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(value.plan.output_kind, OutputKind::Camera);
        assert!(value.camera_control.is_some());
        assert!(transport.heads.is_empty());
        assert!(transport.costs.is_empty());
        Ok(())
    }
    #[test]
    fn image_downloads_are_real_bytes_and_ordered_receipts() -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            events: source_events("KlingImageGenerationNode", true)?,
            ..Transport::default()
        };
        let value = invoke(
            &input("KlingImageGenerationNode")?,
            credentials(),
            19,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(value.media.len(), 2);
        assert_eq!(value.color_mode, "RGBA");
        assert_eq!(value.creation_receipt, [7, 8, 9]);
        assert_eq!(value.receipt, [7, 8, 9]);
        assert_eq!(value.approval_receipt, [42]);
        assert_eq!(value.media[0].url, "https://fixture.invalid/first.png");
        assert_eq!(value.media[1].url, "https://fixture.invalid/second.png");
        for media in &value.media {
            assert_eq!(media.bytes, png());
            assert_eq!(media.receipt, [7, 8, 9]);
        }
        assert_eq!(transport.costs[0].operation, "kling.generation.maximum");
        assert_eq!(
            transport.heads[0].secret_id.as_deref(),
            Some("sim_org.auth_token")
        );
        assert!(
            transport
                .heads
                .iter()
                .skip(2)
                .all(|head| head.secret_id.is_none()
                    && head.headers.iter().all(|header| header.name == "Accept"))
        );
        Ok(())
    }
    #[test]
    fn turbo_poll_uses_query_and_does_not_publish_invented_video() -> Result<(), Box<dyn Error>> {
        let case = oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["requests"][0]["response_model"] == "Kling3TurboCreateResponse")
            .cloned()
            .ok_or("turbo")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut transport = Transport {
            events: events(
                200,
                serde_json::to_vec(&case["requests"][0]["response"])?,
                vec![],
            ),
            ..Transport::default()
        };
        transport.events.extend(events(
            200,
            serde_json::to_vec(&case["requests"][1]["response"])?,
            vec![],
        ));
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                1,
                &mut transport,
                &mut SourceProbeOwner::default()
            ),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(
            transport.heads[1].path,
            "/proxy/kling/tasks?task_ids=source-task-42"
        );
        assert_eq!(
            transport.heads[2].path,
            "https://fixture.invalid/result.mp4"
        );
        Ok(())
    }
    #[test]
    fn auth_cost_and_upload_errors_precede_request_body() -> Result<(), Box<dyn Error>> {
        let source = input("KlingAvatarNode")?;
        let mut transport = Transport::default();
        let mut owner = SourceProbeOwner::default();
        assert!(
            invoke(
                &source,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                1,
                &mut transport,
                &mut owner
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        assert!(owner.operations.is_empty());
        transport.deny_cost = true;
        assert!(matches!(
            invoke(&source, credentials(), 1, &mut transport, &mut owner),
            Err(KlingError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(owner.operations.is_empty());
        assert!(transport.bodies.iter().all(Vec::is_empty));
        transport.deny_cost = false;
        owner.fail = true;
        assert!(matches!(
            invoke(&source, credentials(), 1, &mut transport, &mut owner),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn queued_and_null_statuses_delegate_five_second_poll_delay() -> Result<(), Box<dyn Error>> {
        let source = case("KlingImageGenerationNode")?;
        let creation = source["requests"][0]["response"].clone();
        let mut queued = source["requests"][1]["response"].clone();
        queued["data"]["task_status"] = json!("in_queue");
        let mut active = queued.clone();
        active["data"]["task_status"] = Value::Null;
        let mut transport = Transport {
            events: events(200, serde_json::to_vec(&creation)?, vec![]),
            ..Transport::default()
        };
        transport
            .events
            .extend(events(200, serde_json::to_vec(&queued)?, vec![]));
        transport
            .events
            .extend(events(200, serde_json::to_vec(&active)?, vec![]));
        transport.events.extend(events(
            200,
            serde_json::to_vec(&source["requests"][1]["response"])?,
            vec![],
        ));
        transport.events.extend(events(200, png(), vec![]));
        transport.events.extend(events(200, png(), vec![]));
        invoke(
            &input("KlingImageGenerationNode")?,
            credentials(),
            1,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(transport.delays, [5000, 5000]);
        assert_eq!(transport.progresses.len(), 3);
        Ok(())
    }
    #[test]
    fn active_and_avatar_poll_budgets_match_source() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            request(&input("KlingAvatarNode")?, &mut SourceProbeOwner::default())?.maximum_polls,
            800
        );
        let source = case("KlingImageGenerationNode")?;
        let mut active = source["requests"][1]["response"].clone();
        active["data"]["task_status"] = json!("processing");
        let mut transport = Transport {
            events: events(
                200,
                serde_json::to_vec(&source["requests"][0]["response"])?,
                vec![],
            ),
            ..Transport::default()
        };
        for _ in 0..480 {
            transport
                .events
                .extend(events(200, serde_json::to_vec(&active)?, vec![]));
        }
        assert!(matches!(
            invoke(
                &input("KlingImageGenerationNode")?,
                credentials(),
                1,
                &mut transport,
                &mut SourceProbeOwner::default()
            ),
            Err(KlingError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.delays.len(), 479);
        Ok(())
    }
    #[test]
    fn source_response_failures_fail_before_download() -> Result<(), Box<dyn Error>> {
        for case in oracle()?["response_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut transport = Transport::default();
            for request in case["requests"].as_array().ok_or("requests")? {
                transport.events.extend(events(
                    200,
                    serde_json::to_vec(&request["response"])?,
                    vec![],
                ));
            }
            assert!(
                invoke(
                    &input,
                    credentials(),
                    1,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                )
                .is_err(),
                "{}",
                input.node
            );
            assert!(
                transport
                    .heads
                    .iter()
                    .all(|head| !head.path.starts_with("https://"))
            );
        }
        Ok(())
    }
    #[test]
    fn rate_limit_retries_delegate_and_require_fresh_handle() -> Result<(), Box<dyn Error>> {
        let head = request_head("/proxy/kling/tasks", HttpMethod::Get, credentials(), false)?;
        let mut transport = Transport {
            events: events(
                429,
                b"rate".to_vec(),
                vec![Header {
                    name: "Retry-After".into(),
                    value: "3".into(),
                }],
            ),
            ..Transport::default()
        };
        transport.events.extend(events(200, b"{}".to_vec(), vec![]));
        assert_eq!(
            exchange_with_policy(&head, &[], None, 1024, POLL_RETRY, &mut transport)?.0,
            b"{}"
        );
        assert_eq!(transport.delays, [3000]);
        assert_eq!(transport.heads.len(), 2);
        let mut stale = Transport {
            same_retry_handle: true,
            events: events(503, b"retry".to_vec(), vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, &[], None, 1024, POLL_RETRY, &mut stale),
            Err(KlingError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn malformed_receipts_headers_and_oversized_chunks_are_rejected() -> Result<(), Box<dyn Error>>
    {
        let head = request_head("/proxy/kling/tasks", HttpMethod::Get, credentials(), false)?;
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
                    &[],
                    None,
                    MAXIMUM_RESPONSE_BYTES,
                    POLL_RETRY,
                    &mut transport
                )
                .is_err()
            );
        }
        Ok(())
    }
    #[test]
    fn cancellation_worker_loss_and_fresh_reinvocation_preserve_atomicity()
    -> Result<(), Box<dyn Error>> {
        let source = input("KlingImageGenerationNode")?;
        let mut cancelled = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                1,
                &mut cancelled,
                &mut SourceProbeOwner::default()
            ),
            Err(KlingError::Transport(ConformanceError::Cancelled))
        ));
        assert!(cancelled.heads.is_empty());
        let mut lost = Transport::default();
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                1,
                &mut lost,
                &mut SourceProbeOwner::default()
            ),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        let mut fresh = Transport {
            events: source_events("KlingImageGenerationNode", true)?,
            ..Transport::default()
        };
        assert_eq!(
            invoke(
                &source,
                credentials(),
                1,
                &mut fresh,
                &mut SourceProbeOwner::default()
            )?
            .media
            .len(),
            2
        );
        Ok(())
    }
    #[test]
    fn unicode_reference_boundaries_match_python_word_and_decimal_semantics()
    -> Result<(), Box<dyn Error>> {
        for probe in oracle()?["normalized_prompts"]
            .as_array()
            .ok_or("prompts")?
        {
            assert_eq!(
                normalize_omni_prompt(probe["input"].as_str().ok_or("prompt")?)?,
                probe["output"]
            );
        }
        assert_eq!(
            normalize_omni_prompt("email@image.com @image2 @video")?,
            "email@image.com <<<image_2>>> <<<video_1>>>"
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
    impl KlingMaterializer for RejectDecoder {
        fn materialize(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            assert_eq!(result.color_mode, "RGBA");
            assert_eq!(result.media.len(), 2);
            for media in &result.media {
                assert_eq!(media.bytes, png());
                assert_eq!(media.receipt, [7, 8, 9]);
            }
            self.seen = true;
            Err(ConformanceError::HostFailure)
        }
        fn camera_control_output(&mut self) -> Result<ExpectedOutput, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn missing_native_decoder_publishes_no_placeholder_outputs() -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            events: source_events("KlingImageGenerationNode", true)?,
            ..Transport::default()
        };
        let mut owner = SourceProbeOwner::default();
        let mut decoder = RejectDecoder::default();
        let mut sink = Sink::default();
        let mut owners = MaterializationOwners {
            media: &mut owner,
            materializer: &mut decoder,
            sink: &mut sink,
        };
        assert!(matches!(
            invoke_and_materialize(
                &input("KlingImageGenerationNode")?,
                credentials(),
                1,
                &mut transport,
                &mut owners
            ),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        assert!(decoder.seen);
        assert_eq!(sink.batches, 0);
        Ok(())
    }
    #[test]
    fn source_preflight_failures_preserve_media_preparation_order() -> Result<(), Box<dyn Error>> {
        for case in oracle()?["preflight_errors"]
            .as_array()
            .ok_or("preflight errors")?
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            assert!(request(&input, &mut owner).is_err(), "{}", input.node);
            assert_eq!(json!(owner.operations), case["media"], "{}", input.node);
        }
        Ok(())
    }

    fn assert_prompt_boundary(
        input: &Input,
        accepted: bool,
    ) -> Result<Option<Request>, Box<dyn Error>> {
        let mut owner = SourceProbeOwner::default();
        let prepared = request(input, &mut owner);
        assert_eq!(prepared.is_ok(), accepted, "{}", input.node);
        let mut invoked_owner = SourceProbeOwner::default();
        let mut transport = Transport::default();
        let result = invoke(input, credentials(), 1, &mut transport, &mut invoked_owner);
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(transport.costs[0].operation, "kling.generation.maximum");
        assert_eq!(owner.operations, invoked_owner.operations);
        match prepared {
            Ok(plan) => {
                assert!(matches!(
                    result,
                    Err(KlingError::Transport(ConformanceError::HostFailure))
                ));
                assert_eq!(transport.heads[0].path, plan.path.as_deref().ok_or("path")?);
                assert_eq!(transport.bodies, [serde_json::to_vec(&plan.body)?]);
                Ok(Some(plan))
            }
            Err(expected) => {
                let actual = match result {
                    Err(error) => error,
                    Ok(_) => return Err("invalid prompt published output".into()),
                };
                assert_eq!(actual.to_string(), expected.to_string());
                assert!(transport.bodies.iter().all(Vec::is_empty));
                Ok(None)
            }
        }
    }
    #[test]
    fn python_prompt_whitespace_preserves_validation_request_bytes() -> Result<(), Box<dyn Error>> {
        for (node, name, minimum, maximum) in [
            ("KlingLipSyncTextToVideoNode", "text", 0, 120),
            ("KlingImageGenerationNode", "prompt", 1, 498),
            ("KlingImageGenerationNode", "negative_prompt", 0, 198),
            ("KlingTextToVideoWithAudio", "prompt", 1, 2500),
            ("KlingImageToVideoWithAudio", "prompt", 1, 2500),
            ("KlingMotionControl", "prompt", 0, 2500),
            ("KlingFirstLastFrameNode", "prompt", 1, 2500),
        ] {
            for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                let mut input = input(node)?;
                input
                    .arguments
                    .insert(name.into(), json!(separator.to_string()));
                if let Some(plan) = assert_prompt_boundary(&input, minimum == 0)? {
                    let field = if name == "text" {
                        &plan.body["input"][name]
                    } else {
                        &plan.body[name]
                    };
                    assert_eq!(field, &input.arguments[name]);
                }
            }
            let mut too_long = input(node)?;
            too_long.arguments.insert(
                name.into(),
                json!("x".repeat(if name == "negative_prompt" {
                    201
                } else if node == "KlingImageGenerationNode" {
                    501
                } else {
                    maximum + 1
                })),
            );
            assert_prompt_boundary(&too_long, false)?;
            for prompt in [
                format!("\u{1c}{}\u{1f}", "x".repeat(maximum)),
                "\u{200b}".into(),
                "\u{feff}".into(),
                "\u{1c}x\u{1f}x\u{1c}".into(),
            ] {
                let mut input = input(node)?;
                input.arguments.insert(name.into(), json!(prompt));
                let plan = assert_prompt_boundary(&input, true)?.ok_or("plan")?;
                let field = if name == "text" {
                    &plan.body["input"][name]
                } else {
                    &plan.body[name]
                };
                assert_eq!(field, &input.arguments[name]);
            }
        }
        Ok(())
    }
    #[test]
    fn python_prompt_whitespace_applies_to_all_omni_branches() -> Result<(), Box<dyn Error>> {
        for node in [
            "KlingOmniProImageNode",
            "KlingOmniProTextToVideoNode",
            "KlingOmniProFirstLastFrameNode",
            "KlingOmniProImageToVideoNode",
            "KlingOmniProVideoToVideoNode",
            "KlingOmniProEditVideoNode",
        ] {
            for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                let mut input = input(node)?;
                input
                    .arguments
                    .insert("prompt".into(), json!(separator.to_string()));
                assert_prompt_boundary(&input, false)?;
            }
            let mut too_long = input(node)?;
            too_long
                .arguments
                .insert("prompt".into(), json!("x".repeat(2501)));
            assert_prompt_boundary(&too_long, false)?;
            for prompt in [
                format!("\u{1c}{}\u{1f}", "x".repeat(2500)),
                "\u{200b}".into(),
                "\u{feff}".into(),
                "\u{1c}x\u{1f}x\u{1c}".into(),
            ] {
                let mut input = input(node)?;
                input.arguments.insert("prompt".into(), json!(prompt));
                let plan = assert_prompt_boundary(&input, true)?.ok_or("plan")?;
                assert_eq!(plan.body["prompt"], input.arguments["prompt"]);
            }
        }
        Ok(())
    }
    #[test]
    fn python_storyboard_whitespace_checks_every_entry_and_preserves_order()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "KlingOmniProTextToVideoNode",
            "KlingOmniProFirstLastFrameNode",
            "KlingOmniProImageToVideoNode",
            "KlingVideoNode",
        ] {
            let argument = if node == "KlingVideoNode" {
                "multi_shot"
            } else {
                "storyboards"
            };
            let source_case = oracle()?["cases"]
                .as_array()
                .ok_or("cases")?
                .iter()
                .find(|case| {
                    case["input"]["node"] == node
                        && case["input"]["arguments"][argument].is_object()
                        && case["input"]["arguments"][argument][argument] != "disabled"
                })
                .cloned()
                .ok_or("storyboard case")?;
            let original: Input = serde_json::from_value(source_case["input"].clone())?;
            for index in 1..=2 {
                for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                    let mut input = original.clone();
                    input.arguments.get_mut(argument).ok_or("storyboards")?
                        [format!("storyboard_{index}_prompt")] = json!(separator.to_string());
                    assert_prompt_boundary(&input, false)?;
                }
            }
            for prompt in [
                format!("\u{1c}{}\u{1f}", "x".repeat(512)),
                "\u{200b}".into(),
                "\u{feff}".into(),
            ] {
                let mut input = original.clone();
                let shots = input.arguments.get_mut(argument).ok_or("storyboards")?;
                shots["storyboard_1_prompt"] = json!(prompt);
                shots["storyboard_2_prompt"] = json!("\u{1d}second\u{1e}");
                if node != "KlingVideoNode" {
                    input.arguments.insert("prompt".into(), json!("\u{1c}"));
                }
                let plan = assert_prompt_boundary(&input, true)?.ok_or("plan")?;
                assert_eq!(
                    plan.body["multi_prompt"][0]["prompt"],
                    input.arguments[argument]["storyboard_1_prompt"]
                );
                assert_eq!(
                    plan.body["multi_prompt"][1]["prompt"],
                    input.arguments[argument]["storyboard_2_prompt"]
                );
                assert_eq!(plan.body["multi_prompt"][0]["index"], 1);
                assert_eq!(plan.body["multi_prompt"][1]["index"], 2);
            }
        }
        Ok(())
    }
    #[test]
    fn python_disabled_multi_shot_whitespace_preserves_payload() -> Result<(), Box<dyn Error>> {
        for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            let mut input = input("KlingVideoNode")?;
            input.arguments.get_mut("multi_shot").ok_or("shots")?["prompt"] =
                json!(separator.to_string());
            assert_prompt_boundary(&input, false)?;
        }
        for prompt in [
            format!("\u{1c}{}\u{1f}", "x".repeat(2500)),
            "\u{200b}".into(),
            "\u{feff}".into(),
        ] {
            let mut input = input("KlingVideoNode")?;
            input.arguments.get_mut("multi_shot").ok_or("shots")?["prompt"] = json!(prompt);
            let plan = assert_prompt_boundary(&input, true)?.ok_or("plan")?;
            assert_eq!(plan.body["prompt"], input.arguments["multi_shot"]["prompt"]);
        }
        Ok(())
    }
    #[test]
    fn python_prompt_whitespace_preserves_cost_model_and_media_error_order()
    -> Result<(), Box<dyn Error>> {
        let mut image = input("KlingOmniProImageNode")?;
        image.arguments.insert("prompt".into(), json!("\u{1c}"));
        image.arguments.insert("resolution".into(), json!("4K"));
        assert!(matches!(
            request(&image, &mut SourceProbeOwner::default()),
            Err(KlingError::Input(
                "4K resolution is not supported for kling-image-o1 model."
            ))
        ));
        assert_prompt_boundary(&image, false)?;
        let mut lipsync = input("KlingLipSyncTextToVideoNode")?;
        lipsync
            .arguments
            .insert("text".into(), json!("x".repeat(121)));
        lipsync.arguments.get_mut("video").ok_or("video")?["width"] = json!(1);
        assert!(matches!(
            request(&lipsync, &mut SourceProbeOwner::default()),
            Err(KlingError::Provider(_))
        ));
        assert_prompt_boundary(&lipsync, false)?;
        for node in [
            "KlingTextToVideoWithAudio",
            "KlingOmniProImageNode",
            "KlingVideoNode",
        ] {
            let mut input = input(node)?;
            if node == "KlingVideoNode" {
                input.arguments.get_mut("multi_shot").ok_or("shots")?["prompt"] = json!("\u{1c}");
            } else {
                input.arguments.insert("prompt".into(), json!("\u{1c}"));
            }
            let mut denied = Transport {
                deny_cost: true,
                ..Transport::default()
            };
            let mut owner = SourceProbeOwner::default();
            assert!(matches!(
                invoke(&input, credentials(), 1, &mut denied, &mut owner),
                Err(KlingError::Transport(ConformanceError::InvalidCost))
            ));
            assert_eq!(denied.heads.len(), 1);
            assert_eq!(denied.costs.len(), 1);
            assert!(denied.bodies.iter().all(Vec::is_empty));
            assert!(owner.operations.is_empty());
            let mut cancelled = Transport {
                cancelled: true,
                ..Transport::default()
            };
            assert!(matches!(
                invoke(&input, credentials(), 1, &mut cancelled, &mut owner),
                Err(KlingError::Transport(ConformanceError::Cancelled))
            ));
            assert!(cancelled.heads.is_empty());
            assert!(cancelled.costs.is_empty());
            assert!(cancelled.bodies.is_empty());
            assert!(owner.operations.is_empty());
        }
        Ok(())
    }

    #[test]
    fn raw_positive_prompt_api_bound_preserves_original_admitted_bytes()
    -> Result<(), Box<dyn Error>> {
        for with_image in [false, true] {
            for prompt in [
                "x".repeat(499),
                "x".repeat(500),
                format!("\u{1c}{}\u{1f}", "x".repeat(498)),
                "\u{200b}".into(),
                "\u{feff}".into(),
            ] {
                let mut input = input("KlingImageGenerationNode")?;
                if !with_image {
                    input.arguments.remove("image");
                }
                input.arguments.insert("prompt".into(), json!(prompt));
                let plan = assert_prompt_boundary(&input, true)?.ok_or("plan")?;
                assert_eq!(plan.body["prompt"], input.arguments["prompt"]);
                assert!(
                    plan.body["prompt"]
                        .as_str()
                        .ok_or("prompt")?
                        .chars()
                        .count()
                        <= 500
                );
            }
            for amount in [499, 500] {
                let mut input = input("KlingImageGenerationNode")?;
                if !with_image {
                    input.arguments.remove("image");
                }
                input.arguments.insert(
                    "prompt".into(),
                    json!(format!("\u{1c}{}\u{1f}", "x".repeat(amount))),
                );
                assert!(matches!(
                    request(&input, &mut SourceProbeOwner::default()),
                    Err(KlingError::Input(
                        "Kling API positive prompt exceeds 500 characters"
                    ))
                ));
                assert_prompt_boundary(&input, false)?;
            }
        }
        Ok(())
    }
    #[test]
    fn raw_positive_prompt_api_error_follows_inline_and_negative_preparation()
    -> Result<(), Box<dyn Error>> {
        let mut input = input("KlingImageGenerationNode")?;
        input.arguments.insert(
            "prompt".into(),
            json!(format!("\u{1c}{}\u{1f}", "x".repeat(500))),
        );
        let mut owner = SourceProbeOwner::default();
        assert!(matches!(
            request(&input, &mut owner),
            Err(KlingError::Input(
                "Kling API positive prompt exceeds 500 characters"
            ))
        ));
        assert_eq!(owner.operations.len(), 1);
        assert_eq!(owner.operations[0]["operation"], "inline_image");
        assert_prompt_boundary(&input, false)?;
        let mut unavailable = SourceProbeOwner {
            fail: true,
            ..SourceProbeOwner::default()
        };
        assert!(matches!(
            request(&input, &mut unavailable),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut transport, &mut unavailable),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        input
            .arguments
            .insert("negative_prompt".into(), json!("x".repeat(201)));
        let mut owner = SourceProbeOwner::default();
        assert!(matches!(
            request(&input, &mut owner),
            Err(KlingError::Input(
                "Kling API negative prompt exceeds 200 characters"
            ))
        ));
        assert_eq!(owner.operations.len(), 1);
        assert_prompt_boundary(&input, false)?;
        input
            .arguments
            .insert("prompt".into(), json!("x".repeat(501)));
        let mut unavailable = SourceProbeOwner {
            fail: true,
            ..SourceProbeOwner::default()
        };
        assert!(matches!(
            request(&input, &mut unavailable),
            Err(KlingError::Provider(_))
        ));
        assert!(unavailable.operations.is_empty());
        assert_prompt_boundary(&input, false)?;
        Ok(())
    }
    #[test]
    fn lipsync_text_and_voice_reject_before_video_preparation() -> Result<(), Box<dyn Error>> {
        let mut input = input("KlingLipSyncTextToVideoNode")?;
        input
            .arguments
            .insert("text".into(), json!("x".repeat(121)));
        input.arguments.get_mut("video").ok_or("video")?["width"] = json!(1);
        let mut owner = SourceProbeOwner::default();
        assert!(matches!(
            request(&input, &mut owner),
            Err(KlingError::Provider(_))
        ));
        assert!(owner.operations.is_empty());
        let mut unavailable = SourceProbeOwner {
            fail: true,
            ..SourceProbeOwner::default()
        };
        assert!(matches!(
            request(&input, &mut unavailable),
            Err(KlingError::Provider(_))
        ));
        assert!(unavailable.operations.is_empty());
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut transport, &mut unavailable),
            Err(KlingError::Provider(_))
        ));
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(transport.bodies.len(), 1);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        input.arguments.insert("voice".into(), json!("UNKNOWN"));
        assert!(matches!(
            request(&input, &mut owner),
            Err(KlingError::Input("Unknown Kling voice"))
        ));
        assert!(owner.operations.is_empty());
        input.arguments.insert("voice".into(), json!("Melody"));
        input.arguments.insert("text".into(), json!("source"));
        assert!(matches!(
            request(&input, &mut owner),
            Err(KlingError::Input("Kling video inventory bound exceeded"))
        ));
        input.arguments.get_mut("video").ok_or("video")?["width"] = json!(1280);
        let mut unavailable = SourceProbeOwner {
            fail: true,
            ..SourceProbeOwner::default()
        };
        assert!(matches!(
            request(&input, &mut unavailable),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut transport, &mut unavailable),
            Err(KlingError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(transport.heads.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert_eq!(transport.bodies.len(), 1);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }

    #[test]
    fn lipsync_authority_remains_before_preparation() -> Result<(), Box<dyn Error>> {
        let mut input = input("KlingLipSyncTextToVideoNode")?;
        input
            .arguments
            .insert("text".into(), json!("x".repeat(121)));
        let mut owner = SourceProbeOwner::default();
        let mut denied = Transport {
            deny_cost: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut denied, &mut owner),
            Err(KlingError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(owner.operations.is_empty());
        assert_eq!(denied.heads.len(), 1);
        assert_eq!(denied.costs.len(), 1);
        assert_eq!(denied.bodies.len(), 1);
        assert!(denied.bodies.iter().all(Vec::is_empty));
        let mut cancelled = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(&input, credentials(), 1, &mut cancelled, &mut owner),
            Err(KlingError::Transport(ConformanceError::Cancelled))
        ));
        assert!(cancelled.heads.is_empty());
        assert!(cancelled.costs.is_empty());
        assert!(cancelled.bodies.is_empty());
        assert!(owner.operations.is_empty());
        Ok(())
    }
}
