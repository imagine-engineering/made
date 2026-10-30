use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, ProviderTransport, RequestHead, ResponseEvent, StreamHandle, TypeRegistry,
    publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.runway";
pub const SOURCE_SHA256: &str = "529998a3d2f51c60aff8e1095fff6a44ca8d481e761754ac24d7ad3e6e52b889";
pub const ALIASES: &[&str] = &[];
pub const MAXIMUM_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0557",
        node_identifier: "RunwayAleph2KeyframeNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0558",
        node_identifier: "RunwayAleph2PromptImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0559",
        node_identifier: "RunwayAleph2VideoToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0560",
        node_identifier: "RunwayFirstLastFrameNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0561",
        node_identifier: "RunwayImageToVideoNodeGen3a",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0562",
        node_identifier: "RunwayImageToVideoNodeGen4",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0563",
        node_identifier: "RunwayTextToImageNode",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0151",
        method: "POST",
        path: "/proxy/runway/image_to_video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0152",
        method: "GET",
        path: "/proxy/runway/tasks",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0153",
        method: "POST",
        path: "/proxy/runway/text_to_image",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0154",
        method: "POST",
        path: "/proxy/runway/video_to_video",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum RunwayError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    Http {
        status: u16,
        retry_after: Option<String>,
        rate_limited: bool,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for RunwayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "Runway job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Runway HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for RunwayError {}
impl From<ConformanceError> for RunwayError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for RunwayError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, RunwayError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(RunwayError::Input("Missing source string field"))
}
fn integer(value: &Value, key: &str) -> Result<i64, RunwayError> {
    match value.get(key) {
        Some(Value::Number(number)) => number.as_i64().or_else(|| {
            number
                .as_f64()
                .filter(|number| {
                    number.is_finite()
                        && number.fract() == 0.0
                        && *number >= i64::MIN as f64
                        && *number < i64::MAX as f64
                })
                .map(|number| number as i64)
        }),
        Some(Value::String(value)) => python_strip(value).parse().ok(),
        Some(Value::Bool(value)) => Some(i64::from(*value)),
        _ => None,
    }
    .ok_or(RunwayError::Input("Missing source integer field"))
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ImageInput {
    #[serde(default = "guidance_role")]
    pub role: String,
    pub batch: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default = "rgb_channels")]
    pub channels: u32,
}
fn guidance_role() -> String {
    "guidance".into()
}
fn rgb_channels() -> u32 {
    3
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VideoInput {
    pub duration: f64,
    pub fps: f64,
    #[serde(default)]
    pub duration_error: bool,
    #[serde(default)]
    pub rate_error: bool,
    #[serde(default)]
    pub save_error: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Guidance {
    pub image: ImageInput,
    pub mode: String,
    pub value: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    #[serde(default)]
    pub images: Vec<ImageInput>,
    pub video: Option<VideoInput>,
    pub keyframes: Option<Vec<Guidance>>,
    pub prompt_images: Option<Vec<Guidance>>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum UploadKind {
    ImageBatch,
    PairResizeEndToStartBilinearCenter,
    VideoMp4H264,
    GuidanceImage,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UploadPolicy {
    pub kind: UploadKind,
    pub maximum_images: u32,
    pub total_pixels: u64,
    pub mime_type: &'static str,
    pub show_batch_index: bool,
    pub wait_label: &'static str,
}
impl UploadPolicy {
    fn images(maximum_images: u32, pair: bool) -> Self {
        Self {
            kind: if pair {
                UploadKind::PairResizeEndToStartBilinearCenter
            } else {
                UploadKind::ImageBatch
            },
            maximum_images,
            total_pixels: 2048 * 2048,
            mime_type: "image/png",
            show_batch_index: true,
            wait_label: "Uploading",
        }
    }
    fn guidance() -> Self {
        Self {
            kind: UploadKind::GuidanceImage,
            maximum_images: 1,
            total_pixels: 2048 * 2048,
            mime_type: "image/png",
            show_batch_index: false,
            wait_label: "Uploading",
        }
    }
    fn video() -> Self {
        Self {
            kind: UploadKind::VideoMp4H264,
            maximum_images: 1,
            total_pixels: 0,
            mime_type: "video/mp4",
            show_batch_index: false,
            wait_label: "Uploading",
        }
    }
}
#[derive(Clone, Debug)]
pub struct UploadedMedia {
    pub urls: Vec<String>,
    pub receipt: Vec<u8>,
}
pub trait RunwayMediaOwner {
    fn verify_inputs(&mut self, _: &Input) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn upload(
        &mut self,
        _: &Input,
        _: &[ImageInput],
        _: &UploadPolicy,
    ) -> Result<UploadedMedia, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn video_duration(&mut self, _: &Input) -> Result<Option<f64>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn video_frame_rate(&mut self, _: &Input) -> Result<Option<f64>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub node: String,
    pub path: Option<&'static str>,
    pub body: Value,
    pub local_value: Option<Value>,
    pub estimated_duration_seconds: Option<u64>,
    pub uploads: Vec<UploadedMedia>,
}
fn is_local(node: &str) -> bool {
    matches!(
        node,
        "RunwayAleph2KeyframeNode" | "RunwayAleph2PromptImageNode"
    )
}
fn image_for<'a>(input: &'a Input, role: &str) -> Result<&'a ImageInput, RunwayError> {
    input
        .images
        .iter()
        .find(|image| image.role == role)
        .ok_or(RunwayError::Input("Missing canonical source image"))
}
fn image_valid(image: &ImageInput) -> Result<(), RunwayError> {
    if image.width > 7999
        || image.height > 7999
        || image.width == 0
        || image.height == 0
        || !(1..=4).contains(&image.channels)
    {
        return Err(RunwayError::Input("Source image dimensions"));
    }
    let ratio = image.width as f64 / image.height as f64;
    if !(0.5..2.0).contains(&ratio) || ratio == 0.5 {
        return Err(RunwayError::Input("Source strict image aspect ratio"));
    }
    Ok(())
}
fn preflight_prompt(input: &Input) -> Result<(), RunwayError> {
    let prompt = string(&input.settings, "prompt")?;
    let length = python_strip(prompt).chars().count();
    if length == 0 || input.node == "RunwayAleph2VideoToVideoNode" && length > 1000 {
        return Err(RunwayError::Input("Source prompt preflight"));
    }
    Ok(())
}
fn request_prompt(input: &Input) -> Result<&str, RunwayError> {
    let prompt = string(&input.settings, "prompt")?;
    if prompt.chars().count() > 1000 {
        return Err(RunwayError::Input("Source API prompt length"));
    }
    Ok(prompt)
}
fn upload(
    input: &Input,
    images: &[ImageInput],
    policy: &UploadPolicy,
    owner: &mut dyn RunwayMediaOwner,
    transport: &mut dyn RunwayTransport,
    retained: &mut Vec<UploadedMedia>,
) -> Result<Vec<String>, RunwayError> {
    transport.check_cancelled(None)?;
    let uploaded = owner.upload(input, images, policy)?;
    transport.check_cancelled(None)?;
    if uploaded.urls.len() > policy.maximum_images as usize
        || uploaded.receipt.is_empty()
        || uploaded.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ConformanceError::InvalidResponse.into());
    }
    for url in &uploaded.urls {
        if endpoint(url)? {
            return Err(RunwayError::Response("Upload URL must be absolute"));
        }
    }
    let urls = uploaded.urls.clone();
    retained.push(uploaded);
    Ok(urls)
}
fn first_url(urls: &[String]) -> Result<&str, RunwayError> {
    urls.first()
        .map(String::as_str)
        .ok_or(RunwayError::Input("Source uploaded image index missing"))
}
fn source_seed(input: &Input) -> Result<i64, RunwayError> {
    let seed = integer(&input.settings, "seed")?;
    if !(0..=4294967295).contains(&seed) {
        return Err(RunwayError::Input("Source API seed bound"));
    }
    Ok(seed)
}
fn guidance_model(input: &Input) -> Result<Value, RunwayError> {
    let keyframe = input.node == "RunwayAleph2KeyframeNode";
    let key = if keyframe { "timing" } else { "position" };
    let settings = input
        .settings
        .get(key)
        .ok_or(RunwayError::Input("Missing source guidance position"))?;
    let absolute = string(settings, key)? == "Absolute time (seconds)";
    let field = if absolute { "seconds" } else { "fraction" };
    let value = settings[field]
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or(RunwayError::Input("Source guidance scalar"))?;
    let mut items = if keyframe {
        input.keyframes.clone()
    } else {
        input.prompt_images.clone()
    }
    .unwrap_or_default();
    items.push(Guidance {
        image: image_for(input, "image")?.clone(),
        mode: match (keyframe, absolute) {
            (true, true) => "seconds",
            (true, false) => "at",
            (false, true) => "timestamp",
            (false, false) => "position",
        }
        .into(),
        value,
    });
    if items.len() > 128 {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(json!({"kind":if keyframe {"keyframe"} else {"prompt_image"},"items":items}))
}
pub fn prepare_request(
    input: &Input,
    owner: &mut dyn RunwayMediaOwner,
    transport: &mut dyn RunwayTransport,
) -> Result<RequestPlan, RunwayError> {
    transport.check_cancelled(None)?;
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(RunwayError::Input("Unknown source Runway node"));
    }
    owner.verify_inputs(input)?;
    transport.check_cancelled(None)?;
    if is_local(&input.node) {
        return Ok(RequestPlan {
            node: input.node.clone(),
            path: None,
            body: Value::Null,
            local_value: Some(guidance_model(input)?),
            estimated_duration_seconds: None,
            uploads: vec![],
        });
    }
    preflight_prompt(input)?;
    let mut uploads = vec![];
    let (path, body, estimate) = match input.node.as_str() {
        "RunwayImageToVideoNodeGen3a"
        | "RunwayImageToVideoNodeGen4"
        | "RunwayFirstLastFrameNode" => {
            let start = image_for(input, "start")?;
            image_valid(start)?;
            let pair = input.node == "RunwayFirstLastFrameNode";
            let images = if pair {
                let end = image_for(input, "end")?;
                image_valid(end)?;
                vec![start.clone(), end.clone()]
            } else {
                vec![start.clone()]
            };
            let urls = upload(
                input,
                &images,
                &UploadPolicy::images(if pair { 2 } else { 1 }, pair),
                owner,
                transport,
                &mut uploads,
            )?;
            if pair && urls.len() != 2 {
                return Err(RunwayError::Input("Source pair upload count"));
            }
            let duration = input
                .settings
                .get("duration")
                .and_then(Value::as_f64)
                .filter(|value| *value == 5.0 || *value == 10.0)
                .ok_or(RunwayError::Input("Source Duration enum"))?
                as u64;
            let ratio = string(&input.settings, "ratio")?;
            if ![
                "1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672", "1280:768",
                "768:1280",
            ]
            .contains(&ratio)
            {
                return Err(RunwayError::Input("Source AspectRatio enum"));
            }
            let mut prompt_images = vec![json!({"position":"first","uri":first_url(&urls)?})];
            if pair {
                prompt_images.push(json!({"position":"last","uri":urls.get(1).ok_or(RunwayError::Input("Source last URL missing"))?}));
            }
            (
                "/proxy/runway/image_to_video",
                json!({"duration":duration,"model":if input.node=="RunwayImageToVideoNodeGen4"{"gen4_turbo"}else{"gen3a_turbo"},"promptImage":prompt_images,"promptText":request_prompt(input)?,"ratio":ratio,"seed":source_seed(input)?}),
                if input.node == "RunwayImageToVideoNodeGen3a" {
                    None
                } else {
                    Some(256)
                },
            )
        }
        "RunwayTextToImageNode" => {
            let mut body = json!({"model":"gen4_image"});
            if let Some(reference) = input.images.iter().find(|image| image.role == "reference") {
                image_valid(reference)?;
                let urls = upload(
                    input,
                    std::slice::from_ref(reference),
                    &UploadPolicy::images(1, false),
                    owner,
                    transport,
                    &mut uploads,
                )?;
                body["referenceImages"] = json!([{"uri":first_url(&urls)?}]);
            }
            body["promptText"] = json!(request_prompt(input)?);
            let ratio = string(&input.settings, "ratio")?;
            if ![
                "1920:1080",
                "1080:1920",
                "1024:1024",
                "1360:768",
                "1080:1080",
                "1168:880",
                "1440:1080",
                "1080:1440",
                "1808:768",
                "2112:912",
            ]
            .contains(&ratio)
            {
                return Err(RunwayError::Input("Source text image ratio enum"));
            }
            body["ratio"] = json!(ratio);
            ("/proxy/runway/text_to_image", body, Some(41))
        }
        "RunwayAleph2VideoToVideoNode" => {
            if input.video.is_none() {
                return Err(RunwayError::Input("Missing native video"));
            }
            if let Some(duration) = owner.video_duration(input)?
                && !duration.is_nan()
                && !(2.0 - 0.0001..=30.0 + 0.0001).contains(&duration)
            {
                return Err(RunwayError::Input("Source Aleph duration bound"));
            }
            if let Some(rate) = owner.video_frame_rate(input)?
                && rate > 30.0 + 0.01
            {
                return Err(RunwayError::Input("Source Aleph frame rate bound"));
            }
            if input
                .keyframes
                .as_ref()
                .is_some_and(|items| !items.is_empty())
                && input
                    .prompt_images
                    .as_ref()
                    .is_some_and(|items| !items.is_empty())
            {
                return Err(RunwayError::Input("Source exclusive guidance chains"));
            }
            let duration = owner.video_duration(input)?;
            let urls = upload(
                input,
                &[],
                &UploadPolicy::video(),
                owner,
                transport,
                &mut uploads,
            )?;
            let video_url = first_url(&urls)?;
            let mut body = json!({"model":"aleph2","videoUri":video_url,"contentModeration":{"publicFigureThreshold":string(&input.settings,"public_figure_threshold")?}});
            for (key, items) in [
                ("keyframes", input.keyframes.as_ref()),
                ("promptImage", input.prompt_images.as_ref()),
            ] {
                if let Some(items) = items {
                    if items.len() > 5 {
                        return Err(RunwayError::Input(
                            "Source Aleph five guidance limit after video upload",
                        ));
                    }
                    let mut models = vec![];
                    for item in items {
                        let urls = upload(
                            input,
                            std::slice::from_ref(&item.image),
                            &UploadPolicy::guidance(),
                            owner,
                            transport,
                            &mut uploads,
                        )?;
                        let url = first_url(&urls)?;
                        let absolute = if key == "keyframes" {
                            item.mode == "seconds"
                        } else {
                            item.mode == "timestamp"
                        };
                        if absolute
                            && duration.is_some_and(|duration| item.value > duration + 0.0001)
                        {
                            return Err(RunwayError::Input(
                                "Source guidance timestamp bound after upload",
                            ));
                        }
                        if !item.value.is_finite()
                            || item.value < 0.0
                            || !absolute && item.value > 1.0
                        {
                            return Err(RunwayError::Input(
                                "Source guidance API bound after upload",
                            ));
                        }
                        models.push(if key=="keyframes" {if absolute {json!({"seconds":item.value,"uri":url})}else{json!({"at":item.value,"uri":url})}}else{json!({"position":if absolute{json!({"type":"timestamp","timestampSeconds":item.value})}else{json!({"type":"position","positionPercentage":item.value})},"uri":url})});
                    }
                    if !models.is_empty() {
                        body[key] = json!(models);
                    }
                }
            }
            body["promptText"] = json!(request_prompt(input)?);
            body["seed"] = json!(source_seed(input)?);
            ("/proxy/runway/video_to_video", body, None)
        }
        _ => return Err(RunwayError::Input("Unknown paid source node")),
    };
    Ok(RequestPlan {
        node: input.node.clone(),
        path: Some(path),
        body,
        local_value: None,
        estimated_duration_seconds: estimate,
        uploads,
    })
}
#[derive(Default)]
struct Session {
    invocation: Option<u64>,
    handles: BTreeSet<(u64, u32, u32)>,
}
impl Session {
    fn retain(&mut self, handle: StreamHandle) -> Result<(), RunwayError> {
        if handle.invocation == 0
            || handle.slot == 0
            || handle.generation == 0
            || self
                .invocation
                .is_some_and(|invocation| invocation != handle.invocation)
            || self.handles.len() >= 8192
            || !self
                .handles
                .insert((handle.invocation, handle.slot, handle.generation))
        {
            return Err(ConformanceError::InvalidHandle.into());
        }
        self.invocation = Some(handle.invocation);
        Ok(())
    }
}
fn endpoint(value: &str) -> Result<bool, RunwayError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(RunwayError::Response("Invalid Runway response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(RunwayError::Response("Invalid Runway response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(RunwayError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(RunwayError::Response("Invalid Runway endpoint authority"));
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
) -> Result<RequestHead, RunwayError> {
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
            return Err(RunwayError::Input("Missing Comfy API credentials"));
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
const DOWNLOAD_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 5,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
    timeout_milliseconds: None,
};
const POLL_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 10,
    rate_limit: 16,
    backoff: 1.4,
    download: false,
    timeout_milliseconds: Some(120_000),
};
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    session: &mut Session,
    transport: &mut dyn RunwayTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), RunwayError> {
    transport.check_cancelled(None)?;
    let handle = transport.start_request(head)?;
    session.retain(handle)?;
    exchange_existing(
        head,
        body,
        maximum_bytes,
        policy,
        handle,
        session,
        transport,
    )
}
fn exchange_existing(
    _head: &RequestHead,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    mut handle: StreamHandle,
    session: &mut Session,
    transport: &mut dyn RunwayTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), RunwayError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(RunwayError::Http {
                status,
                retry_after,
                rate_limited,
            }) => {
                let delay = if (status == 429 || rate_limited)
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
                    return Err(RunwayError::Http {
                        status,
                        retry_after,
                        rate_limited,
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
                session.retain(fresh)?;
                handle = fresh;
            }
            result => return result,
        }
    }
}

fn exchange_on_handle(
    handle: StreamHandle,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    transport: &mut dyn RunwayTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), RunwayError> {
    transport.begin_response_deadline(handle, policy.timeout_milliseconds)?;
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
                if !(100..=599).contains(&response_status)
                    || headers.len() > 16
                    || headers
                        .iter()
                        .map(|header| header.name.len() + header.value.len())
                        .sum::<usize>()
                        > 16 * 1024
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
                    return Err(RunwayError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
                        rate_limited: false,
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

pub trait RunwayTransport: ProviderTransport {
    fn validate_task_response(
        &mut self,
        _: &RetainedResponse,
        _: &Value,
    ) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn begin_response_deadline(
        &mut self,
        _: StreamHandle,
        _: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn report_task_progress(
        &mut self,
        _: StreamHandle,
        _: &str,
        _: bool,
        _: Option<f64>,
        _: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn retry_after_delay(&mut self, _: &str, _: u64) -> Result<u64, ConformanceError>;
    fn retry_request(&mut self, _: StreamHandle, _: u64) -> Result<StreamHandle, ConformanceError>;
    fn delay(&mut self, _: StreamHandle, _: u64) -> Result<(), ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct RetainedResponse {
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub handle: StreamHandle,
}
#[derive(Clone, Debug)]
pub struct InvocationResult {
    pub plan: RequestPlan,
    pub creation: Option<RetainedResponse>,
    pub polls: Vec<RetainedResponse>,
    pub downloaded: Option<RetainedResponse>,
    pub cost_receipt: Vec<u8>,
    pub url: Option<String>,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn RunwayTransport,
    media_owner: &mut dyn RunwayMediaOwner,
) -> Result<InvocationResult, RunwayError> {
    let plan = prepare_request(input, media_owner, transport)?;
    if plan.path.is_none() {
        return Ok(InvocationResult {
            plan,
            creation: None,
            polls: vec![],
            downloaded: None,
            cost_receipt: vec![],
            url: None,
        });
    }
    if maximum_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let body = serde_json::to_vec(&plan.body)?;
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let head = request_head(
        plan.path
            .ok_or(RunwayError::Input("Missing source route"))?,
        HttpMethod::Post,
        credentials,
        false,
    )?;
    transport.check_cancelled(None)?;
    let mut session = Session::default();
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    let cost = transport.request_cost(
        handle,
        &CostRequest {
            operation: "runway.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits,
        },
    )?;
    if !cost.accepted
        || cost.approved_microunits == 0
        || cost.approved_microunits > maximum_microunits
        || cost.receipt.is_empty()
        || cost.receipt.len() > MAXIMUM_RECEIPT_BYTES
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    let (bytes, receipt, handle) = exchange_existing(
        &head,
        &body,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let response: Value = serde_json::from_slice(&bytes)?;
    if !response.is_object() {
        return Err(RunwayError::Response("Invalid source creation model"));
    }
    // The source optional ID becomes the literal string None in its f-string;
    // it still cannot substitute a receipt or mint new transport authority.
    let id = match response.get("id") {
        Some(Value::String(id)) => id.as_str(),
        None | Some(Value::Null) => "None",
        _ => return Err(RunwayError::Response("Invalid source task ID")),
    };
    if id.is_empty()
        || id.len() > 1024
        || id.chars().any(|character| {
            character.is_control()
                || character.is_whitespace()
                || matches!(character, '/' | '\\' | '?' | '#' | '%')
        })
    {
        return Err(RunwayError::Response("Unsafe retained task ID"));
    }
    let poll_path = format!("/proxy/runway/tasks/{id}");
    let creation = RetainedResponse {
        bytes,
        receipt,
        handle,
    };
    let mut polls = vec![];
    let mut active_attempts = 0u32;
    for _ in 0..4096 {
        transport.check_cancelled(None)?;
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
        let (bytes, receipt, handle) = exchange_with_policy(
            &head,
            &[],
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            &mut session,
            transport,
        )?;
        let retained = RetainedResponse {
            bytes,
            receipt,
            handle,
        };
        let raw_response: Value = serde_json::from_slice(&retained.bytes)?;
        let response = transport.validate_task_response(&retained, &raw_response)?;
        transport.check_cancelled(Some(handle))?;
        string(&response, "id")?;
        if response
            .get("createdAt")
            .is_none_or(|value| value.is_null())
        {
            return Err(RunwayError::Response("Missing source task datetime"));
        }
        let status = python_strip(string(&response, "status")?).to_lowercase();
        let progress = match response.get("progress") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                    .ok_or(RunwayError::Response("Invalid source progress"))?
                    * 100.0,
            ),
        };
        let queued = [
            "created",
            "queued",
            "queueing",
            "submitted",
            "initializing",
            "wait",
            "in_queue",
        ]
        .contains(&status.as_str());
        transport.report_task_progress(
            handle,
            &status,
            queued,
            progress,
            plan.estimated_duration_seconds,
        )?;
        polls.push(retained);
        if [
            "succeeded",
            "succeed",
            "success",
            "completed",
            "finished",
            "done",
            "complete",
        ]
        .contains(&status.as_str())
        {
            let entries =
                response
                    .get("output")
                    .and_then(Value::as_array)
                    .ok_or(RunwayError::Response(
                        "Source task succeeded without output",
                    ))?;
            if entries.len() > 128 || entries.iter().any(|entry| !entry.is_string()) {
                return Err(RunwayError::Response("Invalid source output list"));
            }
            let url = entries
                .first()
                .and_then(Value::as_str)
                .ok_or(RunwayError::Response(
                    "Source task succeeded without output",
                ))?;
            if endpoint(url)? {
                return Err(RunwayError::Response(
                    "Source download requires absolute URL",
                ));
            }
            let head = request_head(url, HttpMethod::Get, credentials, true)?;
            let (bytes, receipt, handle) = exchange_with_policy(
                &head,
                &[],
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                &mut session,
                transport,
            )?;
            return Ok(InvocationResult {
                plan,
                creation: Some(creation),
                polls,
                downloaded: Some(RetainedResponse {
                    bytes,
                    receipt,
                    handle,
                }),
                cost_receipt: cost.receipt,
                url: Some(url.into()),
            });
        }
        if [
            "cancelled",
            "canceled",
            "canceling",
            "fail",
            "failed",
            "error",
        ]
        .contains(&status.as_str())
        {
            return Err(RunwayError::Response("Source task failed"));
        }
        if !queued {
            active_attempts += 1;
            if active_attempts >= 480 {
                return Err(ConformanceError::PollLimit.into());
            }
        }
        transport.check_cancelled(Some(handle))?;
        transport.delay(handle, 5000)?;
    }
    Err(ConformanceError::PollLimit.into())
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, RunwayError> {
    let name = match node {
        "RunwayAleph2KeyframeNode" => "RunwayAleph2IO.KEYFRAME",
        "RunwayAleph2PromptImageNode" => "RunwayAleph2IO.PROMPT_IMAGE",
        "RunwayTextToImageNode" => "IMAGE",
        _ if NODES.iter().any(|claim| claim.node_identifier == node) => "VIDEO",
        _ => return Err(RunwayError::Input("Unknown source output node")),
    };
    let registry = TypeRegistry::built_in()
        .map_err(|_| RunwayError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve(name)
        .map_err(|_| RunwayError::Response("Missing canonical source type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| RunwayError::Response("Missing canonical family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub trait RunwayOutputOwner {
    fn materialize(
        &mut self,
        _: &InvocationResult,
        _: &[ExpectedOutput],
    ) -> Result<Vec<EncodedOutput>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
pub fn materialize_and_publish(
    result: &InvocationResult,
    transport: &mut dyn RunwayTransport,
    owner: &mut dyn RunwayOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), RunwayError> {
    transport.check_cancelled(None)?;
    let expected = expected_outputs(&result.plan.node)?;
    let outputs = owner.materialize(result, &expected)?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}

pub fn source_price_badge(node: &str) -> Option<&'static str> {
    match node {
        "RunwayImageToVideoNodeGen3a"
        | "RunwayImageToVideoNodeGen4"
        | "RunwayFirstLastFrameNode" => Some(r#"{"type":"usd","usd": 0.0715 * widgets.duration}"#),
        "RunwayTextToImageNode" => Some(r#"{"type":"usd","usd":0.11}"#),
        "RunwayAleph2VideoToVideoNode" => {
            Some(r#"{"type":"usd","usd": 0.4004, "format":{"suffix":"/second"}}"#)
        }
        _ => None,
    }
}
fn source_oracle() -> Result<Value, ConformanceError> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/runway/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture["namespace"] != NAMESPACE
        || fixture["source_sha256"] != SOURCE_SHA256
        || fixture["nodes"] != json!(NODES)
        || fixture["routes"] != json!(ROUTES)
        || fixture["aliases"] != json!(ALIASES)
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = source_oracle()?;
    if source["sources"]["comfy_api_nodes/nodes_runway.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let covered: BTreeSet<&str> = source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
        .iter()
        .map(|case| {
            case["input"]["node"]
                .as_str()
                .ok_or(ConformanceError::InvalidRequest)
        })
        .collect::<Result<_, _>>()?;
    if covered != NODES.iter().map(|claim| claim.node_identifier).collect() {
        return Err(ConformanceError::InvalidRequest);
    }
    for claim in NODES {
        if json!(source_price_badge(claim.node_identifier))
            != source["price_badges"][claim.node_identifier]
        {
            return Err(ConformanceError::InvalidRequest);
        }
        expected_outputs(claim.node_identifier).map_err(|_| ConformanceError::InvalidRequest)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        CostRequest, CostResponse, Progress, SignedPutRequest, StorageCreateRequest,
        StorageCreateResponse,
    };
    use std::{cell::RefCell, collections::VecDeque, rc::Rc};
    type Events = Rc<RefCell<Vec<String>>>;
    type TaskProgress = (String, bool, Option<f64>, Option<u64>);
    struct Response {
        path: String,
        method: HttpMethod,
        body: Vec<u8>,
        status: u16,
        bytes: Vec<u8>,
        headers: Vec<Header>,
        receipt: Vec<u8>,
    }
    struct TestTransport {
        responses: VecDeque<Response>,
        active: Option<(StreamHandle, Response, u64, Vec<u8>)>,
        next_slot: u32,
        invocation: u64,
        events: Events,
        heads: Vec<RequestHead>,
        deadlines: Vec<Option<u64>>,
        progress: Vec<TaskProgress>,
        completed: Vec<RetainedResponse>,
        validate_models: bool,
        delays: Vec<u64>,
        cancelled: bool,
        failure: Option<ConformanceError>,
        stale_retry: bool,
        reject_cost: bool,
    }
    impl TestTransport {
        fn new(responses: Vec<Response>, events: Events) -> Self {
            Self {
                responses: responses.into(),
                active: None,
                next_slot: 1,
                invocation: 42,
                events,
                heads: vec![],
                deadlines: vec![],
                progress: vec![],
                completed: vec![],
                validate_models: true,
                delays: vec![],
                cancelled: false,
                failure: None,
                stale_retry: false,
                reject_cost: false,
            }
        }
        fn finish(&self) -> Result<(), Box<dyn Error>> {
            if self.active.is_some() || !self.responses.is_empty() {
                return Err("unconsumed script".into());
            }
            Ok(())
        }
    }
    impl ProviderTransport for TestTransport {
        fn check_cancelled(&mut self, _: Option<StreamHandle>) -> Result<(), ConformanceError> {
            if self.cancelled {
                return Err(ConformanceError::Cancelled);
            }
            if let Some(error) = &self.failure {
                return Err(error.clone());
            }
            Ok(())
        }
        fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
            if self.active.is_some() {
                return Err(ConformanceError::InvalidHandle);
            }
            let response = self
                .responses
                .pop_front()
                .ok_or(ConformanceError::InvalidRequest)?;
            if response.path != head.path || response.method != head.method {
                return Err(ConformanceError::InvalidRequest);
            }
            if head.path.starts_with("https://")
                && (head.secret_id.is_some()
                    || head.headers.iter().any(|header| {
                        matches!(
                            header.name.as_str(),
                            "Authorization" | "X-API-KEY" | "Comfy-Env" | "Comfy-Usage-Source"
                        )
                    }))
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let handle = StreamHandle {
                invocation: self.invocation,
                slot: self.next_slot,
                generation: 1,
            };
            self.next_slot += 1;
            self.events
                .borrow_mut()
                .push(if head.method == HttpMethod::Post {
                    format!("request:{}", head.path)
                } else {
                    format!("download:{}", head.path)
                });
            self.heads.push(head.clone());
            self.active = Some((handle, response, 0, vec![]));
            Ok(handle)
        }
        fn write_request_chunk(
            &mut self,
            handle: StreamHandle,
            sequence: u64,
            bytes: &[u8],
            end: bool,
        ) -> Result<(), ConformanceError> {
            let (active, response, _, received) = self
                .active
                .as_mut()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle
                || sequence != (received.len() / MAXIMUM_CHUNK_BYTES) as u64
                || bytes.len() > MAXIMUM_CHUNK_BYTES
            {
                return Err(ConformanceError::InvalidRequest);
            }
            received.extend_from_slice(bytes);
            if end
                && response.body != *received
                && !matches!((serde_json::from_slice::<Value>(&response.body),serde_json::from_slice::<Value>(received)),(Ok(expected),Ok(actual)) if expected==actual)
            {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(())
        }
        fn upload(&mut self, _: StreamHandle, _: &[u8]) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn create_storage(
            &mut self,
            _: StreamHandle,
            _: &StorageCreateRequest,
        ) -> Result<StorageCreateResponse, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn put_signed(
            &mut self,
            _: StreamHandle,
            _: &SignedPutRequest,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn request_cost(
            &mut self,
            _: StreamHandle,
            _: &CostRequest,
        ) -> Result<CostResponse, ConformanceError> {
            if self.reject_cost {
                return Err(ConformanceError::InvalidCost);
            }
            Ok(CostResponse {
                accepted: true,
                approved_microunits: 1,
                receipt: b"canonical test cost".to_vec(),
            })
        }
        fn report_progress(
            &mut self,
            _: StreamHandle,
            _: &Progress,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn wait_response(
            &mut self,
            handle: StreamHandle,
            after: Option<u64>,
            timeout: u64,
        ) -> Result<ResponseEvent, ConformanceError> {
            let (active, response, sequence, _) = self
                .active
                .as_mut()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle || timeout != 1000 || after != sequence.checked_sub(1) {
                return Err(ConformanceError::InvalidResponse);
            }
            let chunks = response.bytes.len().div_ceil(MAXIMUM_CHUNK_BYTES) as u64;
            let event = if *sequence == 0 {
                ResponseEvent::Head {
                    status: response.status,
                    headers: response.headers.clone(),
                }
            } else if *sequence <= chunks {
                ResponseEvent::Chunk(
                    response
                        .bytes
                        .chunks(MAXIMUM_CHUNK_BYTES)
                        .nth((*sequence - 1) as usize)
                        .ok_or(ConformanceError::InvalidResponse)?
                        .to_vec(),
                )
            } else if *sequence == chunks + 1 {
                ResponseEvent::Completed(response.receipt.clone())
            } else {
                return Err(ConformanceError::InvalidResponse);
            };
            *sequence += 1;
            if matches!(event, ResponseEvent::Completed(_)) {
                self.completed.push(RetainedResponse {
                    bytes: response.bytes.clone(),
                    receipt: response.receipt.clone(),
                    handle,
                });
                self.active = None;
            }
            Ok(event)
        }
    }
    impl RunwayTransport for TestTransport {
        fn begin_response_deadline(
            &mut self,
            handle: StreamHandle,
            timeout: Option<u64>,
        ) -> Result<(), ConformanceError> {
            if self
                .active
                .as_ref()
                .is_none_or(|(active, _, _, _)| *active != handle)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.deadlines.push(timeout);
            Ok(())
        }
        fn validate_task_response(
            &mut self,
            retained: &RetainedResponse,
            response: &Value,
        ) -> Result<Value, ConformanceError> {
            if !self.validate_models {
                return Err(ConformanceError::HostFailure);
            }
            if !self.completed.iter().any(|completed| {
                completed.handle == retained.handle
                    && completed.bytes == retained.bytes
                    && completed.receipt == retained.receipt
            }) || !matches!(serde_json::from_slice::<Value>(&retained.bytes), Ok(ref value) if value == response)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            // This hermetic owner admits only the independently source-validated fixture date;
            // production Pydantic datetime/coercion binding remains explicitly unbound.
            if response["createdAt"] != "2026-10-01T00:00:00Z"
                || !response["id"].is_string()
                || !response["status"].is_string()
                || response.get("output").is_some_and(|output| {
                    !output.is_null()
                        && output
                            .as_array()
                            .is_none_or(|items| items.iter().any(|item| !item.is_string()))
                })
            {
                return Err(ConformanceError::InvalidResponse);
            }
            Ok(response.clone())
        }
        fn report_task_progress(
            &mut self,
            handle: StreamHandle,
            status: &str,
            queued: bool,
            progress: Option<f64>,
            estimate: Option<u64>,
        ) -> Result<(), ConformanceError> {
            if !self
                .completed
                .iter()
                .any(|response| response.handle == handle)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.progress
                .push((status.into(), queued, progress, estimate));
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
                .map_or(fallback, |seconds| seconds.saturating_mul(1000)))
        }
        fn retry_request(
            &mut self,
            handle: StreamHandle,
            delay: u64,
        ) -> Result<StreamHandle, ConformanceError> {
            self.delays.push(delay);
            if self.stale_retry {
                return Ok(handle);
            }
            let head = self
                .heads
                .last()
                .cloned()
                .ok_or(ConformanceError::InvalidHandle)?;
            self.start_request(&head)
        }
        fn delay(&mut self, _: StreamHandle, delay: u64) -> Result<(), ConformanceError> {
            self.delays.push(delay);
            Ok(())
        }
    }
    struct MediaOwner {
        uploads: Vec<Value>,
        consumed: usize,
        prepared: Vec<UploadPolicy>,
        failed: bool,
    }
    impl MediaOwner {
        fn new(case: &Value) -> Result<Self, Box<dyn Error>> {
            Ok(Self {
                uploads: case["uploads"].as_array().ok_or("uploads")?.clone(),
                consumed: 0,
                prepared: vec![],
                failed: false,
            })
        }
    }
    impl RunwayMediaOwner for MediaOwner {
        fn verify_inputs(&mut self, _: &Input) -> Result<(), ConformanceError> {
            if self.failed {
                Err(ConformanceError::HostFailure)
            } else {
                Ok(())
            }
        }
        fn video_duration(&mut self, input: &Input) -> Result<Option<f64>, ConformanceError> {
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            Ok((!video.duration_error).then_some(video.duration))
        }
        fn video_frame_rate(&mut self, input: &Input) -> Result<Option<f64>, ConformanceError> {
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            Ok((!video.rate_error).then_some(video.fps))
        }
        fn upload(
            &mut self,
            input: &Input,
            images: &[ImageInput],
            policy: &UploadPolicy,
        ) -> Result<UploadedMedia, ConformanceError> {
            self.prepared.push(policy.clone());
            let count = match policy.kind {
                UploadKind::VideoMp4H264 => {
                    if !images.is_empty()
                        || input.video.as_ref().is_none_or(|video| video.save_error)
                        || *policy != UploadPolicy::video()
                    {
                        return Err(ConformanceError::HostFailure);
                    }
                    1usize
                }
                UploadKind::GuidanceImage => {
                    if images.len() != 1 || *policy != UploadPolicy::guidance() {
                        return Err(ConformanceError::InvalidRequest);
                    }
                    1usize
                }
                UploadKind::ImageBatch => {
                    if images.len() != 1 || *policy != UploadPolicy::images(1, false) {
                        return Err(ConformanceError::InvalidRequest);
                    }
                    images
                        .first()
                        .ok_or(ConformanceError::InvalidRequest)?
                        .batch
                        .min(1) as usize
                }
                UploadKind::PairResizeEndToStartBilinearCenter => {
                    if images.len() != 2 || *policy != UploadPolicy::images(2, true) {
                        return Err(ConformanceError::InvalidRequest);
                    }
                    images
                        .iter()
                        .map(|image| image.batch as u64)
                        .sum::<u64>()
                        .min(2) as usize
                }
            };
            let mut urls = vec![];
            for index in 0..count {
                let record = self
                    .uploads
                    .get(self.consumed)
                    .ok_or(ConformanceError::HostFailure)?;
                let label = if policy.show_batch_index && count > 1 {
                    format!("Uploading ({}/{})", index + 1, count)
                } else {
                    policy.wait_label.to_owned()
                };
                let file_name = format!(
                    "00000000-0000-4000-8000-000000000000.{}",
                    if policy.kind == UploadKind::VideoMp4H264 {
                        "mp4"
                    } else {
                        "png"
                    }
                );
                if record["mime_type"] != policy.mime_type
                    || record["file_name"] != file_name
                    || record["wait_label"] != label
                {
                    return Err(ConformanceError::InvalidRequest);
                }
                let bytes = decode_hex(
                    record["bytes_hex"]
                        .as_str()
                        .ok_or(ConformanceError::InvalidRequest)?,
                )?;
                if bytes.is_empty() || bytes.len() > MAXIMUM_MEDIA_BYTES {
                    return Err(ConformanceError::BodyLimit);
                }
                self.consumed += 1;
                urls.push(format!(
                    "https://source.fixture.invalid/upload-{}.bin",
                    self.consumed
                ));
            }
            Ok(UploadedMedia {
                urls,
                receipt: format!("canonical-upload-{}", self.prepared.len()).into_bytes(),
            })
        }
    }
    fn decode_hex(value: &str) -> Result<Vec<u8>, ConformanceError> {
        if !value.is_ascii() || !value.len().is_multiple_of(2) {
            return Err(ConformanceError::InvalidRequest);
        }
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(
                    std::str::from_utf8(pair).map_err(|_| ConformanceError::InvalidRequest)?,
                    16,
                )
                .map_err(|_| ConformanceError::InvalidRequest)
            })
            .collect()
    }
    fn response(
        path: String,
        method: HttpMethod,
        body: Value,
        bytes: Vec<u8>,
    ) -> Result<Response, Box<dyn Error>> {
        Ok(Response {
            path,
            method,
            body: if method == HttpMethod::Post {
                serde_json::to_vec(&body)?
            } else {
                vec![]
            },
            status: 200,
            bytes,
            headers: vec![],
            receipt: b"checked terminal test receipt".to_vec(),
        })
    }
    fn scripted(case: &Value) -> Result<(Input, TestTransport, MediaOwner), Box<dyn Error>> {
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut responses = vec![];
        for request in case["requests"].as_array().ok_or("requests")? {
            responses.push(response(request["path"].as_str().ok_or("path")?.into(),HttpMethod::Post,request["body"].clone(),serde_json::to_vec(&json!({"id":if case["input"]["oracle_options"]["missing_id"]==true {Value::Null}else{json!("task-42")}}))?)?);
        }
        for poll in case["polls"].as_array().ok_or("polls")? {
            for status in poll["statuses"].as_array().ok_or("statuses")? {
                let progress = status["progress"].as_f64().map(|number| number / 100.0);
                let output = if case["input"]["oracle_options"]["empty_output"] == true {
                    json!([])
                } else {
                    json!([
                        "https://source.fixture.invalid/result-first.bin",
                        "https://source.fixture.invalid/result-ignored.bin"
                    ])
                };
                responses.push(response(poll["path"].as_str().ok_or("poll path")?.into(),HttpMethod::Get,Value::Null,serde_json::to_vec(&json!({"createdAt":"2026-10-01T00:00:00Z","id":"task-42","status":status["status"],"progress":progress,"output":output}))?)?);
            }
        }
        for download in case["downloads"].as_array().ok_or("downloads")? {
            responses.push(response(
                download["url"].as_str().ok_or("download url")?.into(),
                HttpMethod::Get,
                Value::Null,
                decode_hex(download["bytes_hex"].as_str().ok_or("bytes")?)?,
            )?);
        }
        Ok((
            input,
            TestTransport::new(responses, Rc::new(RefCell::new(vec![]))),
            MediaOwner::new(case)?,
        ))
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn compare_local(plan: &RequestPlan, output: &Value) -> Result<(), Box<dyn Error>> {
        let local = plan.local_value.as_ref().ok_or("local plan")?;
        if local["kind"] != output["kind"] {
            return Err("local kind".into());
        }
        let items = local["items"].as_array().ok_or("local items")?;
        let expected = output["items"].as_array().ok_or("source items")?;
        if items.len() != expected.len() {
            return Err("local chain length".into());
        }
        for (item, expected) in items.iter().zip(expected) {
            let image: ImageInput = serde_json::from_value(item["image"].clone())?;
            if item["mode"] != expected["mode"]
                || item["value"] != expected["value"]
                || json!([image.batch, image.height, image.width, image.channels])
                    != expected["image"]["shape"]
            {
                return Err("local source chain".into());
            }
            let pixels = expected["image"]["pixels"]
                .as_array()
                .ok_or("source pixels")?;
            if pixels.len()
                != (image.batch as usize)
                    * (image.height as usize)
                    * (image.width as usize)
                    * (image.channels as usize)
            {
                return Err("source pixel count".into());
            }
            for (index, pixel) in pixels.iter().enumerate() {
                let channel = index % (image.channels as usize);
                let batch = index
                    / ((image.width as usize)
                        * (image.height as usize)
                        * (image.channels as usize));
                let base = [0.5f32, 0.25, 0.125, 1.0].get(channel).ok_or("channel")?;
                let value = *base
                    + if channel < 3 {
                        batch as f32 * 0.01
                    } else {
                        0.0
                    };
                if pixel.as_f64() != Some(value as f64) {
                    return Err("immutable source pixels".into());
                }
            }
        }
        Ok(())
    }
    #[test]
    fn exact_claims_price_badges_and_existing_model_ids() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/runway/conformance.fixture"
        )))?;
        assert_eq!(
            expected_outputs("RunwayAleph2KeyframeNode")?
                .first()
                .ok_or("output")?
                .type_id
                .to_string(),
            "sim:custom-runway-aleph2-keyframe@1"
        );
        assert_eq!(
            expected_outputs("RunwayAleph2PromptImageNode")?
                .first()
                .ok_or("output")?
                .type_id
                .to_string(),
            "sim:custom-runway-aleph2-prompt-image@1"
        );
        assert_eq!(NODES.len(), 7);
        assert_eq!(ROUTES.len(), 4);
        Ok(())
    }
    #[test]
    fn all_actual_source_successes_and_ordered_failures() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (group, success) in [("cases", true), ("errors", false)] {
            for (index, case) in source[group].as_array().ok_or("cases")?.iter().enumerate() {
                let (input, mut transport, mut owner) = scripted(case)?;
                let original_keyframes = input.keyframes.clone();
                let original_prompt_images = input.prompt_images.clone();
                let result = invoke(&input, credentials(), 100, &mut transport, &mut owner);
                if success {
                    let result = result
                        .as_ref()
                        .map_err(|error| format!("{group}[{index}] {}: {error:?}", input.node))?;
                    if is_local(&input.node) {
                        compare_local(&result.plan, &case["output"][0])?;
                    } else {
                        assert_eq!(
                            json!(result.plan.path),
                            case["requests"][0]["path"],
                            "{group}[{index}]"
                        );
                        assert_eq!(
                            result.plan.body, case["requests"][0]["body"],
                            "{group}[{index}]"
                        );
                        assert_eq!(
                            json!(result.plan.estimated_duration_seconds),
                            case["polls"][0]["estimated_duration"],
                            "{group}[{index}]"
                        );
                        let downloaded = result.downloaded.as_ref().ok_or("downloaded")?;
                        assert_eq!(
                            downloaded.bytes,
                            decode_hex(case["downloads"][0]["bytes_hex"].as_str().ok_or("hex")?)?
                        );
                        assert_eq!(
                            transport.progress,
                            vec![
                                (
                                    "pending".into(),
                                    false,
                                    None,
                                    result.plan.estimated_duration_seconds
                                ),
                                (
                                    "running".into(),
                                    false,
                                    Some(25.0),
                                    result.plan.estimated_duration_seconds
                                ),
                                (
                                    "succeeded".into(),
                                    false,
                                    Some(100.0),
                                    result.plan.estimated_duration_seconds
                                )
                            ]
                        );
                        assert_eq!(transport.delays, vec![5000, 5000]);
                        assert_eq!(
                            transport.deadlines,
                            vec![
                                Some(3_600_000),
                                Some(120_000),
                                Some(120_000),
                                Some(120_000),
                                None
                            ]
                        );
                    }
                } else {
                    assert!(
                        result.is_err(),
                        "{group}[{index}] {} accepted source failure",
                        input.node
                    );
                }
                assert_eq!(input.keyframes, original_keyframes);
                assert_eq!(input.prompt_images, original_prompt_images);
                assert_eq!(
                    owner.consumed,
                    owner.uploads.len(),
                    "{group}[{index}] {} wrong source upload prefix; result={result:?}",
                    input.node
                );
                transport
                    .finish()
                    .map_err(|error| format!("{group}[{index}] {}: {error}", input.node))?;
            }
        }
        Ok(())
    }
    #[test]
    fn absolute_relative_and_unknown_timing_preserve_source_clone_order()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let cases = source["cases"].as_array().ok_or("cases")?;
        let mut modes = BTreeSet::new();
        for case in cases
            .iter()
            .filter(|case| is_local(case["input"]["node"].as_str().unwrap_or_default()))
        {
            let (input, mut transport, mut owner) = scripted(case)?;
            let plan = prepare_request(&input, &mut owner, &mut transport)?;
            compare_local(&plan, &case["output"][0])?;
            for item in plan.local_value.as_ref().ok_or("local")?["items"]
                .as_array()
                .ok_or("items")?
            {
                modes.insert(item["mode"].as_str().ok_or("mode")?.to_owned());
            }
        }
        assert_eq!(
            modes,
            ["at", "position", "seconds", "timestamp"]
                .into_iter()
                .map(str::to_owned)
                .collect()
        );
        Ok(())
    }
    #[test]
    fn default_owners_fail_closed_before_publication_or_paid_request() -> Result<(), Box<dyn Error>>
    {
        struct Unbound;
        impl RunwayMediaOwner for Unbound {}
        impl RunwayOutputOwner for Unbound {}
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == "RunwayTextToImageNode")
            .ok_or("case")?;
        let (input, mut transport, mut owner) = scripted(case)?;
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut Unbound),
            Err(RunwayError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        let result = invoke(&input, credentials(), 100, &mut transport, &mut owner)?;
        struct Sink(usize);
        impl MaterializationSink for Sink {
            fn publish(
                &mut self,
                _: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.0 += 1;
                Ok(())
            }
        }
        let mut sink = Sink(0);
        assert!(materialize_and_publish(&result, &mut transport, &mut Unbound, &mut sink).is_err());
        assert_eq!(sink.0, 0);
        Ok(())
    }
    #[test]
    fn source_phase_uploads_precede_guidance_validation_and_use_real_pair_policy()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let errors = source["errors"].as_array().ok_or("errors")?;
        let oversized = errors
            .iter()
            .find(|case| {
                case["input"]["keyframes"]
                    .as_array()
                    .is_some_and(|items| items.len() == 6)
            })
            .ok_or("oversized chain")?;
        let (input, mut transport, mut owner) = scripted(oversized)?;
        assert!(prepare_request(&input, &mut owner, &mut transport).is_err());
        assert_eq!(
            owner
                .prepared
                .iter()
                .map(|policy| &policy.kind)
                .collect::<Vec<_>>(),
            vec![&UploadKind::VideoMp4H264]
        );
        assert_eq!(owner.consumed, 1);
        assert!(transport.heads.is_empty());
        let negative = errors
            .iter()
            .find(|case| {
                case["input"]["keyframes"].as_array().is_some_and(|items| {
                    items.len() == 1 && items.first().is_some_and(|item| item["value"] == -0.1)
                })
            })
            .ok_or("negative guidance")?;
        let (input, mut transport, mut owner) = scripted(negative)?;
        assert!(prepare_request(&input, &mut owner, &mut transport).is_err());
        assert_eq!(
            owner
                .prepared
                .iter()
                .map(|policy| &policy.kind)
                .collect::<Vec<_>>(),
            vec![&UploadKind::VideoMp4H264, &UploadKind::GuidanceImage]
        );
        assert_eq!(owner.consumed, 2);
        assert!(transport.heads.is_empty());
        let pair = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| {
                case["input"]["node"] == "RunwayFirstLastFrameNode"
                    && case["input"]["images"][1]["width"] == 3
            })
            .ok_or("resize pair")?;
        let (input, mut transport, mut owner) = scripted(pair)?;
        let plan = prepare_request(&input, &mut owner, &mut transport)?;
        assert_eq!(owner.prepared, vec![UploadPolicy::images(2, true)]);
        assert_eq!(plan.body, pair["requests"][0]["body"]);
        assert_eq!(
            pair["uploads"][0]["bytes_hex"],
            pair["uploads"][1]["bytes_hex"]
        );
        Ok(())
    }
    #[test]
    fn retained_response_validation_is_mandatory_and_checks_all_status_payloads()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == "RunwayTextToImageNode")
            .ok_or("case")?;
        for payload in [
            json!({"createdAt":"not a datetime","id":"task-42","status":"PENDING"}),
            json!({"createdAt":"2026-10-01T00:00:00Z","id":"task-42","status":"PENDING","output":[123]}),
            json!({"createdAt":"2026-10-01T00:00:00Z","id":"task-42","status":"PENDING","progress":1.1}),
        ] {
            let (input, mut transport, mut owner) = scripted(case)?;
            transport.responses.get_mut(1).ok_or("poll response")?.bytes =
                serde_json::to_vec(&payload)?;
            assert!(invoke(&input, credentials(), 100, &mut transport, &mut owner).is_err());
            assert_eq!(transport.heads.len(), 2);
            assert!(transport.progress.is_empty());
        }
        let (input, mut transport, mut owner) = scripted(case)?;
        transport.validate_models = false;
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(RunwayError::Transport(ConformanceError::HostFailure))
        ));
        let retained = transport
            .completed
            .last()
            .cloned()
            .ok_or("completed response")?;
        transport.validate_models = true;
        let value: Value = serde_json::from_slice(&retained.bytes)?;
        let mut stale = retained.clone();
        stale.handle.generation += 1;
        assert_eq!(
            transport.validate_task_response(&stale, &value),
            Err(ConformanceError::InvalidHandle)
        );
        let mut changed = retained;
        changed.receipt.push(1);
        assert_eq!(
            transport.validate_task_response(&changed, &value),
            Err(ConformanceError::InvalidHandle)
        );
        Ok(())
    }
    #[test]
    fn retries_use_fresh_handles_and_preserve_source_deadlines() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == "RunwayTextToImageNode")
            .ok_or("case")?;
        let (input, mut transport, mut owner) = scripted(case)?;
        let first = transport.responses.front().ok_or("first")?;
        let failed = Response {
            path: first.path.clone(),
            method: first.method,
            body: first.body.clone(),
            status: 503,
            bytes: b"unavailable".to_vec(),
            headers: vec![],
            receipt: b"failed scoped receipt".to_vec(),
        };
        transport.responses.push_front(failed);
        let result = invoke(&input, credentials(), 100, &mut transport, &mut owner)?;
        assert_eq!(result.creation.as_ref().ok_or("creation")?.handle.slot, 2);
        assert_eq!(transport.delays, vec![1000, 5000, 5000]);
        assert_eq!(transport.deadlines.first(), Some(&Some(3_600_000)));
        let (input, mut transport, mut owner) = scripted(case)?;
        let first = transport.responses.front().ok_or("first")?;
        let failed = Response {
            path: first.path.clone(),
            method: first.method,
            body: first.body.clone(),
            status: 429,
            bytes: b"limited".to_vec(),
            headers: vec![],
            receipt: b"failed scoped receipt".to_vec(),
        };
        transport.responses.push_front(failed);
        transport.stale_retry = true;
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(RunwayError::Transport(ConformanceError::InvalidHandle))
        ));
        let mut session = Session::default();
        session.retain(StreamHandle {
            invocation: 42,
            slot: 1,
            generation: 1,
        })?;
        assert!(
            session
                .retain(StreamHandle {
                    invocation: 43,
                    slot: 2,
                    generation: 1
                })
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn cancellation_cost_receipt_and_response_bounds_fail_before_download()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == "RunwayTextToImageNode")
            .ok_or("case")?;
        let (input, mut transport, mut owner) = scripted(case)?;
        transport.cancelled = true;
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(RunwayError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        let (input, mut transport, mut owner) = scripted(case)?;
        transport.failure = Some(ConformanceError::HostFailure);
        assert!(invoke(&input, credentials(), 100, &mut transport, &mut owner).is_err());
        assert!(transport.heads.is_empty());
        let (input, mut transport, mut owner) = scripted(case)?;
        transport.reject_cost = true;
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(RunwayError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(transport.deadlines.is_empty());
        for oversized in [false, true] {
            let (input, mut transport, mut owner) = scripted(case)?;
            let first = transport.responses.front_mut().ok_or("first")?;
            if oversized {
                first.bytes = vec![0; MAXIMUM_RESPONSE_BYTES + 1];
            } else {
                first.receipt = vec![];
            }
            assert!(invoke(&input, credentials(), 100, &mut transport, &mut owner).is_err());
            assert_eq!(transport.heads.len(), 1);
        }
        Ok(())
    }
    #[test]
    fn worker_loss_and_fresh_transport_cannot_replay_retained_response()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == "RunwayTextToImageNode")
            .ok_or("case")?;
        let (input, mut transport, mut owner) = scripted(case)?;
        transport.failure = Some(ConformanceError::WorkerLost);
        assert!(matches!(
            invoke(&input, credentials(), 100, &mut transport, &mut owner),
            Err(RunwayError::Transport(ConformanceError::WorkerLost))
        ));
        assert!(transport.heads.is_empty());
        let (input, mut first, mut owner) = scripted(case)?;
        let previous = invoke(&input, credentials(), 100, &mut first, &mut owner)?;
        let retained = previous.polls.last().ok_or("previous poll")?;
        let model: Value = serde_json::from_slice(&retained.bytes)?;
        let (input, mut fresh, mut owner) = scripted(case)?;
        fresh.invocation = 43;
        assert_eq!(
            fresh.validate_task_response(retained, &model),
            Err(ConformanceError::InvalidHandle)
        );
        let result = invoke(&input, credentials(), 100, &mut fresh, &mut owner)?;
        assert_eq!(
            result
                .creation
                .as_ref()
                .ok_or("creation")?
                .handle
                .invocation,
            43
        );
        assert_eq!(
            fresh.validate_task_response(retained, &model),
            Err(ConformanceError::InvalidHandle)
        );
        fresh.finish()?;
        Ok(())
    }
    #[test]
    fn actual_api_model_oracle_exposes_coercions_to_canonical_owner() -> Result<(), Box<dyn Error>>
    {
        let source = source_oracle()?;
        let cases = source["api_model_cases"]
            .as_array()
            .ok_or("API model cases")?;
        assert_eq!(cases.len(), 8);
        assert_eq!(
            cases.get(1).ok_or("string progress")?["normalized"]["progress"],
            0.25
        );
        assert_eq!(
            cases.get(2).ok_or("epoch datetime")?["normalized"]["progress"],
            1.0
        );
        assert_eq!(
            cases
                .iter()
                .filter(|case| case["error_type"] == "ValidationError")
                .count(),
            5
        );
        Ok(())
    }
    #[test]
    fn source_queue_classification_keeps_pending_and_throttled_active() -> Result<(), Box<dyn Error>>
    {
        let source = source_oracle()?;
        let case = source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == "RunwayTextToImageNode")
            .ok_or("case")?;
        for (status, queued) in [
            ("PENDING", false),
            ("THROTTLED", false),
            ("in_queue", true),
            ("CREATED", true),
        ] {
            let (input, mut transport, mut owner) = scripted(case)?;
            let first_poll = transport.responses.get_mut(1).ok_or("poll")?;
            let mut value: Value = serde_json::from_slice(&first_poll.bytes)?;
            value["status"] = json!(status);
            first_poll.bytes = serde_json::to_vec(&value)?;
            invoke(&input, credentials(), 100, &mut transport, &mut owner)?;
            assert_eq!(transport.progress.first().ok_or("progress")?.1, queued);
        }
        Ok(())
    }
}
