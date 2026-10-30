use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, ProviderTransport, RequestHead, ResponseEvent, StreamHandle, TypeRegistry,
    publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.vidu";
pub const SOURCE_SHA256: &str = "3bb3620fdd499270119770f78bbd6eaf70376c9d73dcf64ce764311f5c915998";
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
        feature_id: "COMFY-NODE-0728",
        node_identifier: "Vidu2ImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0729",
        node_identifier: "Vidu2ReferenceVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0730",
        node_identifier: "Vidu2StartEndToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0731",
        node_identifier: "Vidu2TextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0732",
        node_identifier: "Vidu3ImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0733",
        node_identifier: "Vidu3StartEndToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0734",
        node_identifier: "Vidu3TextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0735",
        node_identifier: "ViduExtendVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0736",
        node_identifier: "ViduImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0737",
        node_identifier: "ViduMultiFrameVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0738",
        node_identifier: "ViduReferenceVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0739",
        node_identifier: "ViduStartEndToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0740",
        node_identifier: "ViduTextToVideoNode",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0194",
        method: "POST",
        path: "/proxy/vidu/extend",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0195",
        method: "POST",
        path: "/proxy/vidu/img2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0196",
        method: "POST",
        path: "/proxy/vidu/multiframe",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0197",
        method: "POST",
        path: "/proxy/vidu/reference2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0198",
        method: "POST",
        path: "/proxy/vidu/start-end2video",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0199",
        method: "GET",
        path: "/proxy/vidu/tasks/%s/creations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0200",
        method: "POST",
        path: "/proxy/vidu/text2video",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum ViduError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    SourceIndex(&'static str),
    Http {
        status: u16,
        retry_after: Option<String>,
        rate_limited: bool,
    },
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for ViduError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) | Self::SourceIndex(message) => {
                formatter.write_str(message)
            }
            Self::Provider(status) => write!(formatter, "Vidu job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Vidu HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for ViduError {}
impl From<ConformanceError> for ViduError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for ViduError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageInput {
    pub batch: u32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubjectInput {
    pub id: String,
    pub image: ImageInput,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VideoInput {
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub container: String,
    #[serde(default)]
    pub duration_error: bool,
    #[serde(default)]
    pub conversion_error: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    pub images: std::collections::BTreeMap<String, Option<ImageInput>>,
    pub subjects: Vec<SubjectInput>,
    pub video: Option<VideoInput>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageUploadPolicy {
    pub maximum_images: u32,
    pub total_pixels: u64,
    pub content_type: Option<String>,
    pub wait_label: String,
    pub show_batch_index: bool,
}
#[derive(Clone, Debug)]
pub struct PreparedUpload {
    pub urls: Vec<String>,
    pub receipts: Vec<Vec<u8>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestModel {
    Creation,
    Extend,
    MultiFrame,
    Subject,
    Frame,
}
impl RequestModel {
    pub fn source_name(self) -> &'static str {
        match self {
            Self::Creation => "TaskCreationRequest",
            Self::Extend => "TaskExtendCreationRequest",
            Self::MultiFrame => "TaskMultiFrameCreationRequest",
            Self::Subject => "SubjectReference",
            Self::Frame => "FrameSetting",
        }
    }
}
pub trait ViduInputOwner {
    fn verify_inputs(&mut self, _: &Input) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn normalize_request(&mut self, _: RequestModel, _: &Value) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn source_integer(&mut self, _: &Value) -> Result<i64, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn image_upload(
        &mut self,
        _: &Input,
        _: &str,
        _: &ImageUploadPolicy,
    ) -> Result<PreparedUpload, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn validate_video_duration(
        &mut self,
        _: &Input,
        _: f64,
        _: f64,
        _: f64,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn video_upload_mp4_h264(
        &mut self,
        _: &Input,
        _: &str,
    ) -> Result<PreparedUpload, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub node: String,
    pub path: &'static str,
    pub body: Value,
    pub maximum_active_polls: i64,
    pub input_receipts: Vec<Vec<u8>>,
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, ViduError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(ViduError::Input("Missing source string field"))
}
fn required(value: &Value, key: &str) -> Result<Value, ViduError> {
    value
        .get(key)
        .cloned()
        .ok_or(ViduError::Input("Missing source required field"))
}
fn validate_prompt(settings: &Value, minimum: bool, legacy: bool) -> Result<(), ViduError> {
    let prompt = string(settings, "prompt")?;
    if legacy {
        if prompt.is_empty() {
            return Err(ViduError::Input("Prompt is required"));
        }
    } else {
        let length = python_strip(prompt).chars().count();
        if minimum && length == 0 || length > 2000 {
            return Err(ViduError::Input("Source prompt length rejected"));
        }
    }
    Ok(())
}
fn image<'a>(input: &'a Input, port: &str) -> Result<&'a ImageInput, ViduError> {
    input
        .images
        .get(port)
        .and_then(Option::as_ref)
        .ok_or(ViduError::Input("Missing source image"))
}
fn aspect(image: &ImageInput) -> Result<(), ViduError> {
    if image.width == 0 || image.height == 0 {
        return Err(ViduError::Input("Invalid source image dimensions"));
    }
    let ratio = f64::from(image.height) / f64::from(image.width);
    if !(ratio > 0.25 && ratio < 4.0) {
        return Err(ViduError::Input("Source image aspect ratio rejected"));
    }
    Ok(())
}
fn reference_aspect(image: &ImageInput) -> Result<(), ViduError> {
    aspect(image)?;
    if image.width < 128 || image.height < 128 {
        return Err(ViduError::Input(
            "Source reference image dimensions rejected",
        ));
    }
    Ok(())
}
fn closeness(first: &ImageInput, last: &ImageInput) -> Result<(), ViduError> {
    if first.width == 0 || first.height == 0 || last.width == 0 || last.height == 0 {
        return Err(ViduError::Input("Invalid source image dimensions"));
    }
    let first = f64::from(first.height) / f64::from(first.width);
    let last = f64::from(last.height) / f64::from(last.width);
    if first.max(last) / first.min(last) > 1.25 {
        return Err(ViduError::Input("Source frame aspect closeness rejected"));
    }
    Ok(())
}
fn upload_urls(
    prepared: PreparedUpload,
    maximum: u32,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Vec<String>, ViduError> {
    if prepared.urls.len() != prepared.receipts.len()
        || prepared.urls.len() > maximum as usize
        || receipts.len() + prepared.receipts.len() > 32
    {
        return Err(ConformanceError::InvalidResponse.into());
    }
    for url in &prepared.urls {
        if endpoint(url)? {
            return Err(ViduError::Input("Source upload requires absolute URL"));
        }
    }
    for receipt in prepared.receipts {
        bounded_receipt(&receipt)?;
        receipts.push(receipt);
    }
    Ok(prepared.urls)
}
fn image_policy(
    maximum_images: u32,
    mime: Option<&str>,
    wait_label: &str,
    single: bool,
) -> ImageUploadPolicy {
    ImageUploadPolicy {
        maximum_images,
        total_pixels: 2048 * 2048,
        content_type: mime.map(str::to_owned),
        wait_label: wait_label.into(),
        show_batch_index: !single,
    }
}
fn upload_images(
    input: &Input,
    port: &str,
    policy: &ImageUploadPolicy,
    owner: &mut dyn ViduInputOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Vec<String>, ViduError> {
    let urls = upload_urls(
        owner.image_upload(input, port, policy)?,
        policy.maximum_images,
        receipts,
    )?;
    if !policy.show_batch_index && urls.is_empty() {
        return Err(ViduError::SourceIndex("source image upload URLs[0]"));
    }
    Ok(urls)
}
fn first_url(urls: Vec<String>) -> Result<String, ViduError> {
    urls.into_iter()
        .next()
        .ok_or(ViduError::SourceIndex("source image upload URLs[0]"))
}
fn creation_fields(settings: &Value, modern: bool) -> Result<Value, ViduError> {
    let model = if modern {
        settings
            .get("model")
            .ok_or(ViduError::Input("Missing dynamic model"))?
    } else {
        settings
    };
    let mut raw = json!({"model":required(model,"model")?,"prompt":required(settings,"prompt")?,"duration":required(model,"duration")?,"seed":required(settings,"seed")?,"resolution":required(model,"resolution")?});
    if modern {
        raw["audio"] = required(model, "audio")?;
    }
    Ok(raw)
}
fn normalized_request(
    model: RequestModel,
    raw: &Value,
    owner: &mut dyn ViduInputOwner,
) -> Result<Value, ViduError> {
    let value = owner.normalize_request(model, raw)?;
    if !value.is_object() || serde_json::to_vec(&value)?.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(value)
}
pub fn prepare_request(
    input: &Input,
    owner: &mut dyn ViduInputOwner,
) -> Result<RequestPlan, ViduError> {
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(ViduError::Input("Unknown source node"));
    }
    owner.verify_inputs(input)?;
    let settings = &input.settings;
    let node = input.node.as_str();
    let modern = node.starts_with("Vidu3");
    let second = node.starts_with("Vidu2");
    let mut receipts = vec![];
    let mut maximum_active_polls = 480i64;
    let (path, body) = match node {
        "ViduTextToVideoNode" | "Vidu2TextToVideoNode" | "Vidu3TextToVideoNode" => {
            validate_prompt(settings, true, !second && !modern)?;
            let mut raw = creation_fields(settings, modern)?;
            raw["aspect_ratio"] = required(
                if modern {
                    settings
                        .get("model")
                        .ok_or(ViduError::Input("Missing dynamic model"))?
                } else {
                    settings
                },
                "aspect_ratio",
            )?;
            if modern {
                maximum_active_polls = 640;
            } else if second {
                raw["bgm"] = required(settings, "background_music")?;
            } else {
                raw["movement_amplitude"] = required(settings, "movement_amplitude")?;
            }
            (
                "/proxy/vidu/text2video",
                normalized_request(RequestModel::Creation, &raw, owner)?,
            )
        }
        "ViduImageToVideoNode" | "Vidu2ImageToVideoNode" | "Vidu3ImageToVideoNode" => {
            let descriptor = image(input, "image")?;
            if !modern && descriptor.batch > 1 {
                return Err(ViduError::Input("Only one source image is allowed"));
            }
            aspect(descriptor)?;
            if second || modern {
                validate_prompt(settings, false, false)?;
            }
            let mut raw = creation_fields(settings, modern)?;
            if !modern {
                raw["movement_amplitude"] = required(settings, "movement_amplitude")?;
            }
            let body = if !second && !modern {
                let mut body = normalized_request(RequestModel::Creation, &raw, owner)?;
                body["images"] = json!(upload_images(
                    input,
                    "image",
                    &image_policy(1, Some("image/png"), "Uploading", false),
                    owner,
                    &mut receipts
                )?);
                body
            } else {
                raw["images"] = json!(upload_images(
                    input,
                    "image",
                    &image_policy(
                        1,
                        if modern { None } else { Some("image/png") },
                        "Uploading",
                        modern
                    ),
                    owner,
                    &mut receipts
                )?);
                normalized_request(RequestModel::Creation, &raw, owner)?
            };
            if modern {
                maximum_active_polls = 720;
            }
            ("/proxy/vidu/img2video", body)
        }
        "ViduReferenceVideoNode" => {
            validate_prompt(settings, true, true)?;
            let descriptor = image(input, "images")?;
            if descriptor.batch > 7 {
                return Err(ViduError::Input("Maximum source references is seven"));
            }
            if descriptor.batch > 0 {
                reference_aspect(descriptor)?;
            }
            let mut raw = creation_fields(settings, false)?;
            for field in ["aspect_ratio", "movement_amplitude"] {
                raw[field] = required(settings, field)?;
            }
            let mut body = normalized_request(RequestModel::Creation, &raw, owner)?;
            body["images"] = json!(upload_images(
                input,
                "images",
                &image_policy(7, Some("image/png"), "Uploading", false),
                owner,
                &mut receipts
            )?);
            ("/proxy/vidu/reference2video", body)
        }
        "Vidu2ReferenceVideoNode" => {
            validate_prompt(settings, true, false)?;
            let mut total = 0u32;
            for subject in &input.subjects {
                if subject.image.batch > 3 {
                    return Err(ViduError::Input(
                        "Maximum source images per subject is three",
                    ));
                }
                total = total
                    .checked_add(subject.image.batch)
                    .ok_or(ConformanceError::BodyLimit)?;
                if subject.image.batch > 0 {
                    reference_aspect(&subject.image)?;
                }
            }
            if total > 7 {
                return Err(ViduError::Input("Maximum source references is seven"));
            }
            if input.subjects.len() > 32 {
                return Err(ConformanceError::BodyLimit.into());
            }
            let mut subjects = vec![];
            for (ordinal, subject) in input.subjects.iter().enumerate() {
                let urls = upload_images(
                    input,
                    &format!("subject:{ordinal}"),
                    &image_policy(
                        3,
                        Some("image/png"),
                        &format!("Uploading reference images for {}", subject.id),
                        false,
                    ),
                    owner,
                    &mut receipts,
                )?;
                subjects.push(normalized_request(
                    RequestModel::Subject,
                    &json!({"id":subject.id,"images":urls}),
                    owner,
                )?);
            }
            let mut raw = creation_fields(settings, false)?;
            for field in ["audio", "aspect_ratio", "movement_amplitude"] {
                raw[field] = required(settings, field)?;
            }
            raw["subjects"] = json!(subjects);
            (
                "/proxy/vidu/reference2video",
                normalized_request(RequestModel::Creation, &raw, owner)?,
            )
        }
        "ViduStartEndToVideoNode" | "Vidu2StartEndToVideoNode" | "Vidu3StartEndToVideoNode" => {
            if second || modern {
                validate_prompt(settings, false, false)?;
            }
            let first = image(input, "first_frame")?;
            let last = image(input, "end_frame")?;
            if second && (first.batch > 1 || last.batch > 1) {
                return Err(ViduError::Input("Only one source frame is allowed"));
            }
            closeness(first, last)?;
            let mut raw = creation_fields(settings, modern)?;
            if !modern {
                raw["movement_amplitude"] = required(settings, "movement_amplitude")?;
            }
            let early = !second && !modern;
            let mut body = if early {
                normalized_request(RequestModel::Creation, &raw, owner)?
            } else {
                Value::Null
            };
            let mut urls = vec![];
            for port in ["first_frame", "end_frame"] {
                urls.push(first_url(upload_images(
                    input,
                    port,
                    &image_policy(1, Some("image/png"), "Uploading", false),
                    owner,
                    &mut receipts,
                )?)?);
            }
            if early {
                body["images"] = json!(urls);
            } else {
                raw["images"] = json!(urls);
                body = normalized_request(RequestModel::Creation, &raw, owner)?;
            }
            ("/proxy/vidu/start-end2video", body)
        }
        "ViduExtendVideoNode" => {
            validate_prompt(settings, false, false)?;
            owner.validate_video_duration(input, 4.0, 55.0, 0.0001)?;
            let end_url = if let Some(Some(end)) = input.images.get("end_frame") {
                reference_aspect(end)?;
                Some(first_url(upload_images(
                    input,
                    "end_frame",
                    &image_policy(1, None, "Uploading end frame", true),
                    owner,
                    &mut receipts,
                )?)?)
            } else {
                None
            };
            let model = settings
                .get("model")
                .ok_or(ViduError::Input("Missing dynamic model"))?;
            let mut raw = json!({"model":required(model,"model")?,"prompt":required(settings,"prompt")?,"duration":required(model,"duration")?,"seed":required(settings,"seed")?,"resolution":required(model,"resolution")?});
            let video_url = first_url(upload_urls(
                owner.video_upload_mp4_h264(input, "Uploading video")?,
                1,
                &mut receipts,
            )?)?;
            raw["video_url"] = json!(video_url);
            raw["images"] = end_url.map(|url| json!([url])).unwrap_or(Value::Null);
            (
                "/proxy/vidu/extend",
                normalized_request(RequestModel::Extend, &raw, owner)?,
            )
        }
        "ViduMultiFrameVideoNode" => {
            aspect(image(input, "start_image")?)?;
            let frames = settings
                .get("frames")
                .ok_or(ViduError::Input("Missing source frame controls"))?;
            let count = owner.source_integer(&required(frames, "frames")?)?;
            if !(-16..=16).contains(&count) {
                return Err(ConformanceError::BodyLimit.into());
            }
            for ordinal in 1..=count {
                aspect(image(input, &format!("end_image{ordinal}"))?)?;
                validate_prompt(
                    &json!({"prompt":required(frames,&format!("prompt{ordinal}"))?}),
                    false,
                    false,
                )?;
            }
            let start_url = first_url(upload_images(
                input,
                "start_image",
                &image_policy(1, Some("image/png"), "Uploading start image", true),
                owner,
                &mut receipts,
            )?)?;
            let mut image_settings = vec![];
            for ordinal in 1..=count {
                let prompt = required(frames, &format!("prompt{ordinal}"))?;
                let key_image = first_url(upload_images(
                    input,
                    &format!("end_image{ordinal}"),
                    &image_policy(
                        1,
                        Some("image/png"),
                        &format!("Uploading end image({ordinal})"),
                        true,
                    ),
                    owner,
                    &mut receipts,
                )?)?;
                let duration = required(frames, &format!("duration{ordinal}"))?;
                image_settings.push(normalized_request(
                    RequestModel::Frame,
                    &json!({"prompt":prompt,"key_image":key_image,"duration":duration}),
                    owner,
                )?);
            }
            let raw = json!({"model":required(settings,"model")?,"seed":required(settings,"seed")?,"resolution":required(settings,"resolution")?,"start_image":start_url,"image_settings":image_settings});
            maximum_active_polls = 480 * count;
            (
                "/proxy/vidu/multiframe",
                normalized_request(RequestModel::MultiFrame, &raw, owner)?,
            )
        }
        _ => return Err(ViduError::Input("Unsupported source node")),
    };
    if serde_json::to_vec(&body)?.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(RequestPlan {
        node: input.node.clone(),
        path,
        body,
        maximum_active_polls,
        input_receipts: receipts,
    })
}
#[derive(Default)]
struct Session {
    invocation: Option<u64>,
    handles: BTreeSet<(u64, u32, u32)>,
}
impl Session {
    fn retain(&mut self, handle: StreamHandle) -> Result<(), ViduError> {
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
fn endpoint(value: &str) -> Result<bool, ViduError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(ViduError::Response("Invalid Vidu response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(ViduError::Response("Invalid Vidu response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(ViduError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(ViduError::Response("Invalid Vidu endpoint authority"));
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
) -> Result<RequestHead, ViduError> {
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
            return Err(ViduError::Input("Missing Comfy API credentials"));
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
    if matches!(method, HttpMethod::Post | HttpMethod::Patch) {
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
    transport: &mut dyn ViduTransport,
) -> Result<RetainedResponse, ViduError> {
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
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
    transport: &mut dyn ViduTransport,
) -> Result<RetainedResponse, ViduError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(ViduError::Http {
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
                    return Err(ViduError::Http {
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
    transport: &mut dyn ViduTransport,
) -> Result<RetainedResponse, ViduError> {
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
    let mut retained_headers = vec![];
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
                retained_headers = headers;
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
                    return Err(ViduError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
                        rate_limited: false,
                    });
                }
                transport.check_cancelled(Some(handle))?;
                return Ok(RetainedResponse {
                    bytes,
                    receipt,
                    handle,
                    headers: retained_headers,
                });
            }
            _ => return Err(ConformanceError::InvalidResponse.into()),
        }
    }
    Err(ConformanceError::PollLimit.into())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiModel {
    Creation,
    Status,
}
#[derive(Clone, Debug)]
pub struct RetainedResponse {
    pub handle: StreamHandle,
    pub bytes: Vec<u8>,
    pub headers: Vec<Header>,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct TaskProgress {
    pub status: String,
    pub queued: bool,
    pub percentage: Option<f64>,
    pub source_usd: Option<f64>,
    pub estimated_seconds: Option<u64>,
}
pub trait ViduTransport: ProviderTransport {
    fn validate_response(
        &mut self,
        _: &RetainedResponse,
        _: ApiModel,
        _: &Value,
    ) -> Result<Value, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn authorize_continuation(
        &mut self,
        _: &RetainedResponse,
        _: &RequestHead,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn begin_response_deadline(
        &mut self,
        _: StreamHandle,
        _: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn source_request_labels(
        &mut self,
        _: StreamHandle,
        _: Option<&str>,
        _: Option<&str>,
        _: bool,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn report_task_progress(
        &mut self,
        _: &RetainedResponse,
        _: &TaskProgress,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn retry_after_delay(&mut self, _: &str, _: u64) -> Result<u64, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn retry_request(&mut self, _: StreamHandle, _: u64) -> Result<StreamHandle, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn delay(&mut self, _: StreamHandle, _: u64) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn normalized(
    response: &RetainedResponse,
    model: ApiModel,
    transport: &mut dyn ViduTransport,
) -> Result<Value, ViduError> {
    transport.check_cancelled(Some(response.handle))?;
    let raw: Value = serde_json::from_slice(&response.bytes)?;
    if !raw.is_object() {
        return Err(ViduError::Response("Source response requires API object"));
    }
    let value = transport.validate_response(response, model, &raw)?;
    transport.check_cancelled(Some(response.handle))?;
    if !value.is_object() {
        return Err(ViduError::Response(
            "Invalid canonical normalized API object",
        ));
    }
    if serde_json::to_vec(&value)?.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(value)
}
fn retain_response(
    responses: &mut Vec<RetainedResponse>,
    response: RetainedResponse,
) -> Result<(), ViduError> {
    let retained_bytes = responses
        .iter()
        .chain(std::iter::once(&response))
        .try_fold(0usize, |total, response| {
            let header_bytes = response.headers.iter().try_fold(0usize, |total, header| {
                total
                    .checked_add(header.name.len())?
                    .checked_add(header.value.len())
            })?;
            total
                .checked_add(response.bytes.len())?
                .checked_add(response.receipt.len())?
                .checked_add(header_bytes)
        })
        .ok_or(ConformanceError::BodyLimit)?;
    if retained_bytes > MAXIMUM_MEDIA_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    responses.push(response);
    Ok(())
}
fn retained_id(value: &Value, key: &str) -> Result<String, ViduError> {
    let id = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(ViduError::Response("Missing source retained task ID"))?;
    if id.is_empty()
        || id.len() > 1024
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(ViduError::Response("Unsafe retained task ID"));
    }
    Ok(id.into())
}
fn bounded_receipt(receipt: &[u8]) -> Result<(), ViduError> {
    if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
        return Err(ConformanceError::InvalidResponse.into());
    }
    Ok(())
}
#[derive(Clone, Debug)]
pub struct InvocationResult {
    pub node: String,
    pub task_id: String,
    pub input_receipts: Vec<Vec<u8>>,
    pub responses: Vec<RetainedResponse>,
    pub downloaded: RetainedResponse,
    pub cost_receipt: Vec<u8>,
    pub url: String,
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn ViduTransport,
    owner: &mut dyn ViduInputOwner,
) -> Result<InvocationResult, ViduError> {
    transport.check_cancelled(None)?;
    let plan = prepare_request(input, owner)?;
    transport.check_cancelled(None)?;
    if maximum_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let head = request_head(plan.path, HttpMethod::Post, credentials, false)?;
    let mut session = Session::default();
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    transport.source_request_labels(handle, Some("Waiting for server"), Some("Completed"), true)?;
    let cost = transport.request_cost(
        handle,
        &CostRequest {
            operation: "vidu.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits,
        },
    )?;
    if !cost.accepted
        || cost.approved_microunits == 0
        || cost.approved_microunits > maximum_microunits
    {
        return Err(ConformanceError::InvalidCost.into());
    }
    bounded_receipt(&cost.receipt)?;
    let creation = exchange_existing(
        &head,
        &serde_json::to_vec(&plan.body)?,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let created = normalized(&creation, ApiModel::Creation, transport)?;
    if string(&created, "state")? == "failed" {
        return Err(ViduError::Provider("creation rejected".into()));
    }
    let task_id = retained_id(&created, "task_id")?;
    let mut responses = vec![];
    retain_response(&mut responses, creation)?;
    let poll_path = format!("/proxy/vidu/tasks/{task_id}/creations");
    let mut active_polls = 0i64;
    for _ in 0..8191 {
        if active_polls >= plan.maximum_active_polls {
            return Err(ConformanceError::PollLimit.into());
        }
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
        transport.check_cancelled(Some(prior.handle))?;
        transport.authorize_continuation(prior, &head)?;
        let handle = transport.start_request(&head)?;
        session.retain(handle)?;
        transport.source_request_labels(handle, Some("Checking"), None, false)?;
        let retained = exchange_existing(
            &head,
            &[],
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            handle,
            &mut session,
            transport,
        )?;
        let response = normalized(&retained, ApiModel::Status, transport)?;
        let status = python_strip(string(&response, "state")?).to_lowercase();
        let queued = matches!(
            status.as_str(),
            "created" | "queued" | "queueing" | "submitted" | "initializing" | "wait" | "in_queue"
        );
        let percentage = response.get("progress").and_then(Value::as_f64);
        let source_usd = response
            .get("credits")
            .and_then(Value::as_f64)
            .map(|value| value * 0.005);
        if percentage.is_some_and(|value| !value.is_finite())
            || source_usd.is_some_and(|value| !value.is_finite())
        {
            return Err(ConformanceError::InvalidResponse.into());
        }
        transport.report_task_progress(
            &retained,
            &TaskProgress {
                status: status.clone(),
                queued,
                percentage,
                source_usd,
                estimated_seconds: None,
            },
        )?;
        let completed = matches!(
            status.as_str(),
            "succeeded" | "succeed" | "success" | "completed" | "finished" | "done" | "complete"
        );
        if completed && percentage != Some(100.0) {
            transport.report_task_progress(
                &retained,
                &TaskProgress {
                    status: status.clone(),
                    queued: false,
                    percentage: Some(100.0),
                    source_usd,
                    estimated_seconds: None,
                },
            )?;
        }
        let url = if completed {
            let result = response
                .get("creations")
                .and_then(Value::as_array)
                .and_then(|values| values.first())
                .ok_or(ViduError::Response("Vidu request does not contain results"))?;
            Some(string(result, "url")?.to_owned())
        } else {
            None
        };
        retain_response(&mut responses, retained)?;
        if let Some(url) = url {
            if endpoint(&url)? {
                return Err(ViduError::Response(
                    "Source video download requires absolute URL",
                ));
            }
            let head = request_head(&url, HttpMethod::Get, credentials, true)?;
            let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
            transport.check_cancelled(Some(prior.handle))?;
            transport.authorize_continuation(prior, &head)?;
            let downloaded = exchange_with_policy(
                &head,
                &[],
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                &mut session,
                transport,
            )?;
            if downloaded.bytes.is_empty() {
                return Err(ConformanceError::InvalidResponse.into());
            }
            retain_response(&mut responses, downloaded.clone())?;
            transport.check_cancelled(None)?;
            return Ok(InvocationResult {
                node: input.node.clone(),
                task_id,
                input_receipts: plan.input_receipts,
                responses,
                downloaded,
                cost_receipt: cost.receipt,
                url,
            });
        }
        if matches!(
            status.as_str(),
            "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error"
        ) {
            return Err(ViduError::Provider(status));
        }
        let prior = responses.last().ok_or(ConformanceError::InvalidHandle)?;
        transport.check_cancelled(Some(prior.handle))?;
        transport.delay(prior.handle, 5000)?;
        if !queued {
            active_polls += 1;
        }
    }
    Err(ConformanceError::PollLimit.into())
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, ViduError> {
    if !NODES.iter().any(|claim| claim.node_identifier == node) {
        return Err(ViduError::Input("Unknown source output node"));
    }
    let registry =
        TypeRegistry::built_in().map_err(|_| ViduError::Response("Missing canonical registry"))?;
    let type_id = registry
        .resolve("VIDEO")
        .map_err(|_| ViduError::Response("Missing canonical VIDEO"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| ViduError::Response("Missing canonical video family"))?;
    Ok(vec![ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }])
}
pub trait ViduOutputOwner {
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
    transport: &mut dyn ViduTransport,
    owner: &mut dyn ViduOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), ViduError> {
    transport.check_cancelled(None)?;
    let expected = expected_outputs(&result.node)?;
    let outputs = owner.materialize(result, &expected)?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}

fn source_oracle() -> Result<Value, ConformanceError> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/vidu/source-contract.json"
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
    if source["sources"]["comfy_api_nodes/nodes_vidu.py"] != SOURCE_SHA256 {
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
    type ReportedProgress = (String, bool, Option<f64>, Option<f64>, Option<u64>);
    struct Response {
        path: String,
        method: HttpMethod,
        body: Vec<u8>,
        status: u16,
        bytes: Vec<u8>,
        headers: Vec<Header>,
        receipt: Vec<u8>,
        normalized: Option<(ApiModel, Value)>,
        failure: Option<ConformanceError>,
    }
    struct TestTransport {
        responses: VecDeque<Response>,
        active: Option<(StreamHandle, Response, u64, Vec<u8>)>,
        next_slot: u32,
        invocation: u64,
        events: Events,
        heads: Vec<RequestHead>,
        deadlines: Vec<Option<u64>>,
        progress: Vec<ReportedProgress>,
        completed: Vec<RetainedResponse>,
        validate_models: bool,
        normalized: Vec<(StreamHandle, ApiModel, Value)>,
        continuation_bound: bool,
        labels: Vec<(Option<String>, Option<String>, bool)>,
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
                normalized: vec![],
                continuation_bound: true,
                labels: vec![],
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
                .push(format!("request:{}", head.path));
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
            if let Some(error) = &response.failure {
                return Err(error.clone());
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
                    headers: response.headers.clone(),
                });
                if let Some((model, value)) = &response.normalized {
                    self.normalized.push((handle, *model, value.clone()));
                }
                self.active = None;
            }
            Ok(event)
        }
    }
    impl ViduTransport for TestTransport {
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
        fn validate_response(
            &mut self,
            retained: &RetainedResponse,
            model: ApiModel,
            response: &Value,
        ) -> Result<Value, ConformanceError> {
            if !self.validate_models {
                return Err(ConformanceError::HostFailure);
            }
            if !self.completed.iter().any(|completed| {
                completed.handle == retained.handle
                    && completed.bytes == retained.bytes
                    && completed.receipt == retained.receipt
                    && completed.headers == retained.headers
            }) || !matches!(serde_json::from_slice::<Value>(&retained.bytes),Ok(ref raw) if raw==response)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.normalized
                .iter()
                .find(|(handle, kind, _)| *handle == retained.handle && *kind == model)
                .map(|(_, _, normalized)| normalized.clone())
                .ok_or(ConformanceError::InvalidResponse)
        }
        fn authorize_continuation(
            &mut self,
            retained: &RetainedResponse,
            head: &RequestHead,
        ) -> Result<(), ConformanceError> {
            if !self.continuation_bound {
                return Err(ConformanceError::HostFailure);
            }
            if retained.handle.invocation != self.invocation
                || !self.completed.iter().any(|completed| {
                    completed.handle == retained.handle
                        && completed.bytes == retained.bytes
                        && completed.receipt == retained.receipt
                        && completed.headers == retained.headers
                })
                || self.responses.front().is_none_or(|response| {
                    response.path != head.path || response.method != head.method
                })
            {
                return Err(ConformanceError::InvalidHandle);
            }
            Ok(())
        }
        fn source_request_labels(
            &mut self,
            _: StreamHandle,
            wait: Option<&str>,
            final_label: Option<&str>,
            monitor: bool,
        ) -> Result<(), ConformanceError> {
            self.labels.push((
                wait.map(str::to_owned),
                final_label.map(str::to_owned),
                monitor,
            ));
            Ok(())
        }
        fn report_task_progress(
            &mut self,
            retained: &RetainedResponse,
            progress: &TaskProgress,
        ) -> Result<(), ConformanceError> {
            if !self.completed.iter().any(|response| {
                response.handle == retained.handle && response.receipt == retained.receipt
            }) {
                return Err(ConformanceError::InvalidHandle);
            }
            self.progress.push((
                progress.status.clone(),
                progress.queued,
                progress.percentage,
                progress.source_usd,
                progress.estimated_seconds,
            ));
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
    fn hex(value: &Value) -> Result<Vec<u8>, Box<dyn Error>> {
        let text = value.as_str().ok_or("hex string")?;
        if text.len() % 2 != 0 {
            return Err("odd hex".into());
        }
        text.as_bytes()
            .chunks_exact(2)
            .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
            .collect()
    }
    fn response(
        path: &str,
        method: HttpMethod,
        body: Vec<u8>,
        raw: &Value,
        normalized: Option<Value>,
    ) -> Result<Response, Box<dyn Error>> {
        Ok(Response {
            path: path.into(),
            method,
            body,
            status: 200,
            bytes: serde_json::to_vec(raw)?,
            headers: vec![],
            receipt: b"retained canonical test terminal".to_vec(),
            normalized: normalized.map(|value| {
                (
                    if method == HttpMethod::Post {
                        ApiModel::Creation
                    } else {
                        ApiModel::Status
                    },
                    value,
                )
            }),
            failure: None,
        })
    }
    fn scripts(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let mut script = vec![];
        for request in case["requests"].as_array().ok_or("requests")? {
            script.push(response(
                request["path"].as_str().ok_or("path")?,
                HttpMethod::Post,
                serde_json::to_vec(&request["body"])?,
                &request["response"],
                request.get("normalized_response").cloned(),
            )?);
        }
        for poll in case["polls"].as_array().ok_or("polls")? {
            for retained in poll["responses"].as_array().ok_or("poll responses")? {
                script.push(response(
                    poll["path"].as_str().ok_or("path")?,
                    HttpMethod::Get,
                    vec![],
                    &retained["raw"],
                    retained.get("normalized").cloned(),
                )?);
            }
        }
        for download in case["downloads"].as_array().ok_or("downloads")? {
            let mut retained = response(
                download["url"].as_str().ok_or("URL")?,
                HttpMethod::Get,
                vec![],
                &json!({}),
                None,
            )?;
            retained.bytes = hex(&download["bytes_hex"])?;
            script.push(retained);
        }
        Ok(script)
    }
    struct TestOwner {
        case: Value,
        constructor_count: usize,
        upload_count: usize,
        events: Events,
        fail: bool,
    }
    impl TestOwner {
        fn new(case: &Value, events: Events) -> Self {
            Self {
                case: case.clone(),
                constructor_count: 0,
                upload_count: 0,
                events,
                fail: false,
            }
        }
        fn uploaded(&mut self) -> Result<String, ConformanceError> {
            let record = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            if hex(&record["bytes_hex"])
                .map_err(|_| ConformanceError::InvalidRequest)?
                .is_empty()
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.events
                .borrow_mut()
                .push(format!("upload:{}", self.upload_count));
            self.upload_count += 1;
            Ok(format!(
                "https://source.fixture.invalid/input-{}.bin",
                self.upload_count
            ))
        }
    }
    impl ViduInputOwner for TestOwner {
        fn verify_inputs(&mut self, input: &Input) -> Result<(), ConformanceError> {
            let expected: Input = serde_json::from_value(self.case["input"].clone())
                .map_err(|_| ConformanceError::InvalidRequest)?;
            if self.fail
                || serde_json::to_value(input).map_err(|_| ConformanceError::InvalidRequest)?
                    != serde_json::to_value(expected)
                        .map_err(|_| ConformanceError::InvalidRequest)?
            {
                return Err(ConformanceError::HostFailure);
            }
            Ok(())
        }
        fn source_integer(&mut self, raw: &Value) -> Result<i64, ConformanceError> {
            let count = self.case["counts"]
                .get(0)
                .ok_or(ConformanceError::InvalidRequest)?;
            if count["raw"] != *raw {
                return Err(ConformanceError::InvalidRequest);
            }
            count["resolved"]
                .as_i64()
                .ok_or(ConformanceError::InvalidRequest)
        }
        fn normalize_request(
            &mut self,
            model: RequestModel,
            raw: &Value,
        ) -> Result<Value, ConformanceError> {
            let record = self.case["constructors"]
                .get(self.constructor_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            if record["model"] != model.source_name() || record["raw"] != *raw {
                return Err(ConformanceError::InvalidRequest);
            }
            self.events
                .borrow_mut()
                .push(format!("constructor:{}", self.constructor_count));
            self.constructor_count += 1;
            record
                .get("normalized")
                .cloned()
                .ok_or(ConformanceError::InvalidRequest)
        }
        fn image_upload(
            &mut self,
            input: &Input,
            port: &str,
            policy: &ImageUploadPolicy,
        ) -> Result<PreparedUpload, ConformanceError> {
            let descriptor = if let Some(ordinal) = port.strip_prefix("subject:") {
                &input
                    .subjects
                    .get(
                        ordinal
                            .parse::<usize>()
                            .map_err(|_| ConformanceError::InvalidRequest)?,
                    )
                    .ok_or(ConformanceError::InvalidRequest)?
                    .image
            } else {
                input
                    .images
                    .get(port)
                    .and_then(Option::as_ref)
                    .ok_or(ConformanceError::InvalidRequest)?
            };
            if policy.total_pixels != 2048 * 2048
                || policy.maximum_images == 0
                || policy.maximum_images > 7
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let count = descriptor.batch.min(policy.maximum_images);
            let mut urls = vec![];
            let mut receipts = vec![];
            for ordinal in 0..count {
                let record = self.case["uploads"]
                    .get(self.upload_count)
                    .ok_or(ConformanceError::InvalidRequest)?;
                let expected_label = if policy.show_batch_index && count > 1 {
                    format!("{} ({}/{})", policy.wait_label, ordinal + 1, count)
                } else {
                    policy.wait_label.clone()
                };
                let bytes =
                    hex(&record["bytes_hex"]).map_err(|_| ConformanceError::InvalidRequest)?;
                if record["mime_type"] != json!(policy.content_type)
                    || record["wait_label"] != expected_label
                    || !record["file_name"]
                        .as_str()
                        .is_some_and(|name| name.ends_with(".png"))
                    || !bytes.starts_with(b"\x89PNG\r\n\x1a\n")
                {
                    return Err(ConformanceError::InvalidRequest);
                }
                urls.push(self.uploaded()?);
                receipts.push(format!("retained upload {}", self.upload_count).into_bytes());
            }
            Ok(PreparedUpload { urls, receipts })
        }
        fn validate_video_duration(
            &mut self,
            input: &Input,
            minimum: f64,
            maximum: f64,
            epsilon: f64,
        ) -> Result<(), ConformanceError> {
            if (minimum, maximum, epsilon) != (4.0, 55.0, 0.0001) {
                return Err(ConformanceError::InvalidRequest);
            }
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            if !video.duration_error
                && (minimum - epsilon > video.duration || video.duration > maximum + epsilon)
            {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(())
        }
        fn video_upload_mp4_h264(
            &mut self,
            input: &Input,
            label: &str,
        ) -> Result<PreparedUpload, ConformanceError> {
            let video = input
                .video
                .as_ref()
                .ok_or(ConformanceError::InvalidRequest)?;
            if label != "Uploading video"
                || self.case["conversions"].get(0) != Some(&json!({"format":"mp4","codec":"h264"}))
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.events.borrow_mut().push("convert:video".into());
            if video.conversion_error {
                return Err(ConformanceError::HostFailure);
            }
            let record = self.case["uploads"]
                .get(self.upload_count)
                .ok_or(ConformanceError::InvalidRequest)?;
            if record["mime_type"] != "video/mp4"
                || record["wait_label"] != label
                || hex(&record["bytes_hex"]).map_err(|_| ConformanceError::InvalidRequest)?
                    != include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../sim_test_support/fixtures/providers/vidu/encoded-video.mp4"
                    ))
            {
                return Err(ConformanceError::InvalidRequest);
            }
            let url = self.uploaded()?;
            Ok(PreparedUpload {
                urls: vec![url],
                receipts: vec![b"retained MP4/H264 source boundary".to_vec()],
            })
        }
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn input(case: &Value) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(case["input"].clone())?)
    }
    fn first_case(node: &str) -> Result<Value, Box<dyn Error>> {
        source_oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == node)
            .cloned()
            .ok_or_else(|| "case".into())
    }
    fn source_events(case: &Value) -> Result<Vec<String>, Box<dyn Error>> {
        let mut values = vec![];
        for event in case["events"].as_array().ok_or("events")? {
            let pair = event
                .as_object()
                .ok_or("event")?
                .iter()
                .next()
                .ok_or("event field")?;
            if pair.0 == "poll"
                && case["polls"][0]["responses"]
                    .as_array()
                    .ok_or("poll responses")?
                    .is_empty()
            {
                continue;
            }
            values.push(format!(
                "{}:{}",
                pair.0,
                if let Some(text) = pair.1.as_str() {
                    text.into()
                } else {
                    pair.1.to_string()
                }
            ));
        }
        Ok(values)
    }
    fn transport_events(events: &Events) -> Vec<String> {
        let mut result = vec![];
        for event in events.borrow().iter() {
            let normalized = if let Some(path) = event.strip_prefix("request:https://") {
                format!("download:https://{path}")
            } else if event.starts_with("request:/proxy/vidu/tasks/") {
                event.replacen("request:", "poll:", 1)
            } else {
                event.clone()
            };
            if normalized.starts_with("poll:") && result.last() == Some(&normalized) {
                continue;
            }
            result.push(normalized);
        }
        result
    }
    #[test]
    fn all_actual_source_successes_preserve_constructor_upload_and_poll_phases()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (ordinal, case) in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .enumerate()
        {
            let events = Events::default();
            let mut owner = TestOwner::new(case, events.clone());
            let mut transport = TestTransport::new(scripts(case)?, events.clone());
            let result = invoke(&input(case)?, credentials(), 1, &mut transport, &mut owner)
                .map_err(|error| {
                    format!(
                        "case{ordinal}{}: {error}; constructor={},events={:?}",
                        case["input"]["node"],
                        owner.constructor_count,
                        events.borrow()
                    )
                })?;
            transport.finish()?;
            assert_eq!(
                owner.constructor_count,
                case["constructors"].as_array().ok_or("constructors")?.len()
            );
            assert_eq!(
                owner.upload_count,
                case["uploads"].as_array().ok_or("uploads")?.len()
            );
            assert_eq!(
                transport_events(&events),
                source_events(case)?,
                "case{ordinal}"
            );
            assert_eq!(
                result.downloaded.bytes,
                hex(&case["output"][0]["bytes_hex"])?
            );
            assert_eq!(result.url, case["downloads"][0]["url"]);
            assert_eq!(expected_outputs(&result.node)?.len(), 1);
            let poll = &case["polls"][0];
            let statuses = poll["statuses"].as_array().ok_or("statuses")?;
            for (actual, expected) in transport.progress.iter().zip(statuses) {
                assert_eq!(actual.0, expected["state"].as_str().ok_or("state")?);
                assert_eq!(actual.2, expected["progress"].as_f64());
                assert_eq!(actual.3, expected["price"].as_f64());
            }
            assert_eq!(transport.progress.len(), statuses.len() + 1);
            assert_eq!(transport.progress.last().ok_or("progress")?.2, Some(100.0));
            assert_eq!(
                poll["progress_updates"]
                    .as_array()
                    .ok_or("updates")?
                    .last()
                    .ok_or("final update")?["current"],
                100
            );
        }
        Ok(())
    }
    #[test]
    fn all_actual_source_errors_preserve_exact_paid_request_and_preparation_prefix()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (ordinal, case) in source["errors"]
            .as_array()
            .ok_or("errors")?
            .iter()
            .enumerate()
        {
            let events = Events::default();
            let mut owner = TestOwner::new(case, events.clone());
            let mut transport = TestTransport::new(scripts(case)?, events.clone());
            let result = invoke(&input(case)?, credentials(), 1, &mut transport, &mut owner);
            assert!(result.is_err(), "error{ordinal}:{}", case["input"]);
            assert_eq!(
                owner.constructor_count,
                case["constructors"].as_array().ok_or("constructors")?.len(),
                "error{ordinal}:{result:?}"
            );
            assert_eq!(
                owner.upload_count,
                case["uploads"].as_array().ok_or("uploads")?.len(),
                "error{ordinal}:{result:?}"
            );
            assert_eq!(
                transport_events(&events),
                source_events(case)?,
                "error{ordinal}:{result:?}"
            );
            assert!(
                !transport
                    .heads
                    .iter()
                    .any(|head| head.path.starts_with("https://")),
                "error{ordinal}"
            );
        }
        Ok(())
    }
    #[test]
    fn input_api_continuation_cost_and_lost_worker_fail_closed() -> Result<(), Box<dyn Error>> {
        let case = first_case("ViduTextToVideoNode")?;
        struct Unbound;
        impl ViduInputOwner for Unbound {}
        let mut transport = TestTransport::new(scripts(&case)?, Events::default());
        assert!(matches!(
            invoke(
                &input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut Unbound
            ),
            Err(ViduError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        for error in [ConformanceError::Cancelled, ConformanceError::WorkerLost] {
            let mut transport = TestTransport::new(scripts(&case)?, Events::default());
            transport.failure = Some(error.clone());
            assert!(
                matches!(invoke(&input(&case)?,credentials(),1,&mut transport,&mut TestOwner::new(&case,Events::default())),Err(ViduError::Transport(ref actual)) if *actual==error)
            );
            assert!(transport.heads.is_empty());
        }
        for (validate, continuation) in [(false, true), (true, false)] {
            let mut transport = TestTransport::new(scripts(&case)?, Events::default());
            transport.validate_models = validate;
            transport.continuation_bound = continuation;
            assert!(matches!(
                invoke(
                    &input(&case)?,
                    credentials(),
                    1,
                    &mut transport,
                    &mut TestOwner::new(&case, Events::default())
                ),
                Err(ViduError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(transport.heads.len(), 1);
        }
        let mut transport = TestTransport::new(scripts(&case)?, Events::default());
        transport.reject_cost = true;
        assert!(matches!(
            invoke(
                &input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut TestOwner::new(&case, Events::default())
            ),
            Err(ViduError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(transport.completed.is_empty());
        Ok(())
    }
    #[test]
    fn full_response_identity_and_cross_scope_replay_are_rejected() -> Result<(), Box<dyn Error>> {
        let case = first_case("ViduTextToVideoNode")?;
        let mut transport = TestTransport::new(scripts(&case)?, Events::default());
        let result = invoke(
            &input(&case)?,
            credentials(),
            1,
            &mut transport,
            &mut TestOwner::new(&case, Events::default()),
        )?;
        let original = result.responses.first().ok_or("response")?;
        for variant in 0..4 {
            let mut changed = original.clone();
            match variant {
                0 => changed.bytes.push(0),
                1 => changed.receipt.push(0),
                2 => changed.headers.push(Header {
                    name: "X-Test".into(),
                    value: "tamper".into(),
                }),
                _ => changed.handle.invocation += 1,
            }
            assert!(matches!(
                transport.validate_response(
                    &changed,
                    ApiModel::Creation,
                    &serde_json::from_slice(&original.bytes)?
                ),
                Err(ConformanceError::InvalidHandle)
            ));
        }
        let mut session = Session::default();
        session.retain(original.handle)?;
        assert!(session.retain(original.handle).is_err());
        let mut cross = original.handle;
        cross.invocation += 1;
        assert!(session.retain(cross).is_err());
        Ok(())
    }
    #[test]
    fn retries_limits_auth_and_fresh_invocation_recovery() -> Result<(), Box<dyn Error>> {
        let case = first_case("ViduTextToVideoNode")?;
        for stale in [false, true] {
            let mut script = scripts(&case)?;
            let mut transient = response(
                "/proxy/vidu/text2video",
                HttpMethod::Post,
                script.first().ok_or("first")?.body.clone(),
                &json!({}),
                None,
            )?;
            transient.status = 503;
            script.insert(0, transient);
            let mut transport = TestTransport::new(script, Events::default());
            transport.stale_retry = stale;
            let result = invoke(
                &input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut TestOwner::new(&case, Events::default()),
            );
            if stale {
                assert!(matches!(
                    result,
                    Err(ViduError::Transport(ConformanceError::InvalidHandle))
                ));
            } else {
                result?;
                transport.finish()?;
                assert_eq!(transport.delays.first(), Some(&1000));
            }
        }
        let mut script = scripts(&case)?;
        script.first_mut().ok_or("first")?.bytes = vec![b'x'; MAXIMUM_RESPONSE_BYTES + 1];
        let mut transport = TestTransport::new(script, Events::default());
        assert!(matches!(
            invoke(
                &input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut TestOwner::new(&case, Events::default())
            ),
            Err(ViduError::Transport(ConformanceError::BodyLimit))
        ));
        let mut script = scripts(&case)?;
        script.first_mut().ok_or("first")?.failure = Some(ConformanceError::WorkerLost);
        let mut transport = TestTransport::new(script, Events::default());
        assert!(matches!(
            invoke(
                &input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut TestOwner::new(&case, Events::default())
            ),
            Err(ViduError::Transport(ConformanceError::WorkerLost))
        ));
        let mut fresh = TestTransport::new(scripts(&case)?, Events::default());
        fresh.invocation = 43;
        invoke(
            &input(&case)?,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            1,
            &mut fresh,
            &mut TestOwner::new(&case, Events::default()),
        )?;
        assert_eq!(
            fresh.heads.first().ok_or("head")?.secret_id.as_deref(),
            Some("sim_org.api_key")
        );
        fresh.finish()?;
        let mut transport = TestTransport::new(scripts(&case)?, Events::default());
        assert!(matches!(
            invoke(
                &input(&case)?,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                1,
                &mut transport,
                &mut TestOwner::new(&case, Events::default())
            ),
            Err(ViduError::Input(_))
        ));
        assert!(transport.heads.is_empty());
        Ok(())
    }
    #[test]
    fn claims_video_identity_and_unbound_native_output_are_checked() -> Result<(), Box<dyn Error>> {
        assert_eq!(NODES.len(), 13);
        assert_eq!(ROUTES.len(), 7);
        assert!(ALIASES.is_empty());
        assert_eq!(
            expected_outputs("ViduTextToVideoNode")?[0]
                .type_id
                .to_string(),
            "sim:video@1"
        );
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/vidu/conformance.fixture"
        )))?;
        struct Output;
        impl ViduOutputOwner for Output {}
        #[derive(Default)]
        struct Sink {
            published: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(
                &mut self,
                _: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.published += 1;
                Ok(())
            }
        }
        let case = first_case("ViduTextToVideoNode")?;
        let mut transport = TestTransport::new(scripts(&case)?, Events::default());
        let result = invoke(
            &input(&case)?,
            credentials(),
            1,
            &mut transport,
            &mut TestOwner::new(&case, Events::default()),
        )?;
        let mut sink = Sink::default();
        assert!(matches!(
            materialize_and_publish(&result, &mut transport, &mut Output, &mut sink),
            Err(ViduError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(sink.published, 0);
        Ok(())
    }
}
