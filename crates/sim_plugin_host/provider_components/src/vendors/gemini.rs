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
pub const NAMESPACE: &str = "zed.sim.provider.gemini";
pub const ALIASES: &[&str] = &["vertexai"];
pub const SOURCE_SHA256: &str = "e760f41dad8e26c44b9e4d5c283c7ffafffb2d1839200c3b471da81992d463db";
pub const IMAGE_COLOR_MODE: &str = "RGBA";
pub const MAXIMUM_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 24 * 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_INPUT_FILE_BYTES: usize = 20 * 1024 * 1024;
pub const MAXIMUM_INLINE_BYTES: usize = 18 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
    pub source_class: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0195",
        node_identifier: "GeminiImage2Node",
        source_class: "GeminiImage2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0196",
        node_identifier: "GeminiImageNode",
        source_class: "GeminiImage",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0197",
        node_identifier: "GeminiInputFiles",
        source_class: "GeminiInputFiles",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0198",
        node_identifier: "GeminiNanoBanana2",
        source_class: "GeminiNanoBanana2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0199",
        node_identifier: "GeminiNanoBanana2V2",
        source_class: "GeminiNanoBanana2V2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0200",
        node_identifier: "GeminiNode",
        source_class: "GeminiNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0201",
        node_identifier: "GeminiNodeV2",
        source_class: "GeminiNodeV2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0202",
        node_identifier: "GeminiVideoOmni",
        source_class: "GeminiVideoOmni",
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
        feature_id: "COMFY-API-EXT-0190",
        method: "POST",
        path: "/proxy/vertexai/gemini",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0191",
        method: "POST",
        path: "/proxy/vertexai/gemini/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0192",
        method: "POST",
        path: "/proxy/vertexai/gemini/{model_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0193",
        method: "POST",
        path: "/proxy/vertexai/gemini/{model}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum GeminiError {
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
impl fmt::Display for GeminiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Gemini HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for GeminiError {}
impl From<ConformanceError> for GeminiError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for GeminiError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, GeminiError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/gemini/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, GeminiError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(GeminiError::Input("Unknown Gemini node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.source_class];
    let names = contract["arguments"]
        .as_array()
        .ok_or(GeminiError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(GeminiError::Input("Unknown Gemini source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(GeminiError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(GeminiError::Input("Missing Gemini source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, GeminiError> {
    input
        .arguments
        .get(name)
        .ok_or(GeminiError::Input("Missing Gemini argument"))
}
fn text(value: &Value) -> Result<&str, GeminiError> {
    value
        .as_str()
        .ok_or(GeminiError::Input("Expected Gemini string"))
}
fn number(value: &Value, minimum: f64, maximum: f64) -> Result<f64, GeminiError> {
    let number = value
        .as_f64()
        .ok_or(GeminiError::Input("Expected Gemini number"))?;
    if !number.is_finite() || number < minimum || number > maximum {
        return Err(GeminiError::Input("Gemini numeric bound exceeded"));
    }
    Ok(number)
}
fn integer(value: &Value, maximum: u64) -> Result<u64, GeminiError> {
    let number = value
        .as_u64()
        .ok_or(GeminiError::Input("Expected Gemini unsigned integer"))?;
    if number > maximum {
        return Err(GeminiError::Input("Gemini integer bound exceeded"));
    }
    Ok(number)
}
fn prompt(value: &Value) -> Result<&str, GeminiError> {
    let text = text(value)?;
    if text
        .trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
        })
        .is_empty()
    {
        return Err(GeminiError::Input("Empty Gemini prompt"));
    }
    Ok(text)
}
fn choice<'a>(value: &'a Value, choices: &[&str]) -> Result<&'a str, GeminiError> {
    let text = text(value)?;
    if !choices.contains(&text) {
        return Err(GeminiError::Input("Unknown Gemini source choice"));
    }
    Ok(text)
}
fn model_path(model: &str) -> Result<String, GeminiError> {
    if model.is_empty()
        || model.len() > 256
        || !model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(GeminiError::Input("Invalid Gemini model identifier"));
    }
    Ok(format!("/proxy/vertexai/gemini/{model}"))
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaKind {
    Image,
    Audio,
    Video,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MediaUnit {
    pub kind: MediaKind,
    pub source: Value,
    pub port: String,
    pub batch_index: usize,
    pub duration_seconds: Option<f64>,
}
#[derive(Clone, Debug, Default)]
pub struct MediaInventory {
    pub images: Vec<MediaUnit>,
    pub audios: Vec<MediaUnit>,
    pub videos: Vec<MediaUnit>,
}
#[derive(Clone, Debug, Default)]
pub struct Resolution {
    pub media_parts: Vec<Value>,
    pub file_parts: Vec<Value>,
    pub local_file_part: Option<Value>,
    pub image_count: usize,
    pub video_durations: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: Option<String>,
    pub body: Value,
}
fn system_instruction(body: &mut Value, input: &Input) -> Result<(), GeminiError> {
    if let Some(value) = input.arguments.get("system_prompt") {
        let text = text(value)?;
        if !text.is_empty() {
            body["systemInstruction"] = json!({"parts":[{"text":text}]});
        }
    }
    Ok(())
}
fn image_config(aspect: &str, resolution: Option<&str>) -> Value {
    let mut config = json!({"imageOutputOptions":{"mimeType":"image/png"}});
    if !aspect.is_empty() && aspect != "auto" {
        config["aspectRatio"] = json!(aspect);
    }
    if let Some(resolution) = resolution {
        config["imageSize"] = json!(resolution);
    }
    config
}
fn legacy_text_model_alias(model: &str) -> &str {
    match model {
        "gemini-3-pro-preview" | "gemini-3-1-pro" => "gemini-3.1-pro-preview",
        "gemini-3-1-flash-lite" => "gemini-3.1-flash-lite-preview",
        _ => model,
    }
}
fn image_model_alias(model: &str) -> &str {
    if model == "Nano Banana 2 (Gemini 3.1 Flash Image)" {
        "gemini-3.1-flash-image-preview"
    } else {
        model
    }
}
fn nano_v2_model_alias(model: &str) -> &str {
    if model == "Nano Banana 2 Lite" {
        "gemini-3.1-flash-lite-image"
    } else {
        image_model_alias(model)
    }
}
pub fn request(input: &Input, resolution: &Resolution) -> Result<Request, GeminiError> {
    let input = normalized(input)?;
    if input.node == "GeminiInputFiles" {
        let name = text(argument(&input, "file")?)?;
        if name.is_empty()
            || name.len() > 256
            || name.contains('/')
            || name.contains('\\')
            || name.chars().any(char::is_control)
            || !(name.ends_with(".txt") || name.ends_with(".pdf"))
        {
            return Err(GeminiError::Input("Invalid Gemini input file name"));
        }
        let part = resolution
            .local_file_part
            .clone()
            .ok_or(GeminiError::Input(
                "Canonical input file preparation is required",
            ))?;
        let mut files = vec![part];
        files.extend(resolution.file_parts.clone());
        return Ok(Request {
            path: None,
            body: json!(files),
        });
    }
    let seed = integer(
        argument(&input, "seed")?,
        if matches!(input.node.as_str(), "GeminiNodeV2" | "GeminiVideoOmni") {
            i32::MAX as u64
        } else {
            u64::MAX
        },
    )?;
    let mut parts = vec![];
    let mut body = json!({});
    let mut generation = None;
    let model = match input.node.as_str() {
        "GeminiNode" => {
            parts.push(json!({"text":text(argument(&input,"prompt")?)?}));
            legacy_text_model_alias(text(argument(&input, "model")?)?).to_owned()
        }
        "GeminiNodeV2" => {
            parts.push(json!({"text":prompt(argument(&input,"prompt")?)?}));
            let model = argument(&input, "model")?;
            let selected = choice(
                &model["model"],
                &["Gemini 3.1 Pro", "Gemini 3.1 Flash-Lite"],
            )?;
            let mut config = json!({"temperature":number(&model["temperature"],0.0,2.0)?,"topP":number(&model["top_p"],0.0,1.0)?,"maxOutputTokens":integer(&model["max_output_tokens"],65536)?,"thinkingConfig":{"thinkingLevel":choice(&model["thinking_level"],&["LOW","HIGH"])?}});
            if integer(&model["max_output_tokens"], 65536)? < 16 {
                return Err(GeminiError::Input(
                    "Gemini maximum output tokens below source range",
                ));
            }
            if seed > 0 {
                config["seed"] = json!(seed);
            }
            generation = Some(config);
            if selected == "Gemini 3.1 Pro" {
                "gemini-3.1-pro-preview".into()
            } else {
                "gemini-3.1-flash-lite-preview".into()
            }
        }
        "GeminiImageNode" | "GeminiImage2Node" | "GeminiNanoBanana2" => {
            parts.push(json!({"text":prompt(argument(&input,"prompt")?)?}));
            let aspect = text(argument(&input, "aspect_ratio")?)?;
            let size = input.arguments.get("resolution").map(text).transpose()?;
            let modalities = choice(
                argument(&input, "response_modalities")?,
                &["IMAGE", "IMAGE+TEXT"],
            )?;
            let mut config = json!({"responseModalities":if modalities=="IMAGE"{json!(["IMAGE"])}else{json!(["TEXT","IMAGE"])},"imageConfig":image_config(aspect,size)});
            if input.node == "GeminiNanoBanana2" {
                config["thinkingConfig"] = json!({"thinkingLevel":choice(argument(&input,"thinking_level")?,&["MINIMAL","HIGH"])?});
            }
            if input.node != "GeminiImageNode" && resolution.image_count > 14 {
                return Err(GeminiError::Input(
                    "The current maximum number of supported images is 14.",
                ));
            }
            generation = Some(config);
            body["uploadImagesToStorage"] = json!(true);
            image_model_alias(text(argument(&input, "model")?)?).into()
        }
        "GeminiNanoBanana2V2" => {
            parts.push(json!({"text":prompt(argument(&input,"prompt")?)?}));
            let model = argument(&input, "model")?;
            if resolution.image_count > 14 {
                return Err(GeminiError::Input(
                    "The current maximum number of supported images is 14.",
                ));
            }
            let modalities = choice(
                argument(&input, "response_modalities")?,
                &["IMAGE", "IMAGE+TEXT"],
            )?;
            generation = Some(
                json!({"responseModalities":if modalities=="IMAGE"{json!(["IMAGE"])}else{json!(["TEXT","IMAGE"])},"imageConfig":image_config(text(&model["aspect_ratio"])?,Some(text(&model["resolution"])?)),"thinkingConfig":{"thinkingLevel":choice(&model["thinking_level"],&["MINIMAL","HIGH"])?},"temperature":number(argument(&input,"temperature")?,0.0,2.0)?,"topP":number(argument(&input,"top_p")?,0.0,1.0)?}),
            );
            body["uploadImagesToStorage"] = json!(true);
            nano_v2_model_alias(text(&model["model"])?).into()
        }
        "GeminiVideoOmni" => {
            let model = argument(&input, "model")?;
            prompt(&model["prompt"])?;
            choice(&model["model"], &["Omni Flash"])?;
            if resolution.image_count > 14 || resolution.video_durations.len() > 3 {
                return Err(GeminiError::Input(
                    "Gemini Omni media count exceeds source limit",
                ));
            }
            if resolution
                .video_durations
                .iter()
                .any(|duration| !duration.is_finite() || *duration > 10.0 || *duration < 0.0)
            {
                return Err(GeminiError::Input(
                    "Gemini Omni video duration exceeds source limit",
                ));
            }
            generation = Some(
                json!({"responseModalities":["TEXT","VIDEO"],"temperature":number(model.get("temperature").unwrap_or(&json!(1.0)),0.0,2.0)?,"topP":number(model.get("top_p").unwrap_or(&json!(0.95)),0.0,1.0)?}),
            );
            "gemini-omni-flash-preview".into()
        }
        _ => return Err(GeminiError::Input("Unknown Gemini node")),
    };
    if matches!(
        input.node.as_str(),
        "GeminiNode" | "GeminiImageNode" | "GeminiImage2Node" | "GeminiNanoBanana2"
    ) && input
        .arguments
        .get("images")
        .is_some_and(|value| !value.is_null())
        && resolution.image_count == 0
    {
        return Err(GeminiError::Input(
            "No images provided to create_image_parts",
        ));
    }
    parts.extend(resolution.media_parts.clone());
    parts.extend(resolution.file_parts.clone());
    if input.node == "GeminiVideoOmni" {
        parts.push(json!({"text":text(&argument(&input,"model")?["prompt"])?}));
    }
    body["contents"] = json!([{"role":"user","parts":parts}]);
    if let Some(config) = generation {
        body["generationConfig"] = config;
    }
    system_instruction(&mut body, &input)?;
    Ok(Request {
        path: Some(model_path(&model)?),
        body,
    })
}
fn endpoint(value: &str) -> Result<bool, GeminiError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(GeminiError::Response("Invalid Gemini response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(GeminiError::Response("Invalid Gemini response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(GeminiError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(GeminiError::Response("Invalid Gemini endpoint authority"));
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
) -> Result<RequestHead, GeminiError> {
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
            return Err(GeminiError::Input("Missing Comfy API credentials"));
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

pub trait GeminiTransport: ProviderTransport {
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
pub trait GeminiInputOwner {
    fn inventory(&mut self, input: &Input) -> Result<MediaInventory, ConformanceError>;
    fn existing_file_parts(&mut self, input: &Input) -> Result<Vec<Value>, ConformanceError>;
    fn read_file_part(
        &mut self,
        name: &str,
        maximum_bytes: usize,
    ) -> Result<Value, ConformanceError>;
    fn upload_images(
        &mut self,
        images: &[MediaUnit],
        wait_label: &str,
        show_batch_index: bool,
    ) -> Result<Vec<String>, ConformanceError>;
    fn upload_media(
        &mut self,
        unit: &MediaUnit,
        mime_type: &str,
        container: Option<&str>,
        codec: Option<&str>,
        wait_label: Option<&str>,
    ) -> Result<String, ConformanceError>;
    fn inline_media(
        &mut self,
        unit: &MediaUnit,
        mime_type: &str,
        container: Option<&str>,
        codec: Option<&str>,
    ) -> Result<String, ConformanceError>;
}
#[derive(Clone, Debug)]
pub enum MediaAction {
    ImageBatch {
        units: Vec<MediaUnit>,
        wait_label: String,
        show_batch_index: bool,
    },
    Url {
        unit: MediaUnit,
        mime_type: &'static str,
        container: Option<&'static str>,
        codec: Option<&'static str>,
        wait_label: Option<&'static str>,
    },
    Inline {
        unit: MediaUnit,
        mime_type: &'static str,
        container: Option<&'static str>,
        codec: Option<&'static str>,
    },
}
fn inline_action(unit: MediaUnit, mime: &'static str) -> MediaAction {
    let (container, codec) = match unit.kind {
        MediaKind::Image => (None, None),
        MediaKind::Audio => (Some("mp3"), Some("libmp3lame")),
        MediaKind::Video => (Some("mp4"), Some("h264")),
    };
    MediaAction::Inline {
        unit,
        mime_type: mime,
        container,
        codec,
    }
}
pub fn media_actions(
    input: &Input,
    inventory: &MediaInventory,
) -> Result<Vec<MediaAction>, GeminiError> {
    let input = normalized(input)?;
    let mut actions = vec![];
    if matches!(input.node.as_str(), "GeminiNodeV2" | "GeminiVideoOmni") {
        let units = inventory
            .videos
            .iter()
            .chain(if input.node == "GeminiNodeV2" {
                inventory.audios.iter()
            } else {
                [].iter()
            })
            .chain(inventory.images.iter());
        for (index, unit) in units.enumerate() {
            if index < 10 {
                let (mime, container, codec, label) = match unit.kind {
                    MediaKind::Image => ("image/png", None, None, Some("Uploading image")),
                    MediaKind::Audio => ("audio/mp3", Some("mp3"), Some("libmp3lame"), None),
                    MediaKind::Video => ("video/mp4", None, None, Some("Uploading video")),
                };
                actions.push(MediaAction::Url {
                    unit: unit.clone(),
                    mime_type: mime,
                    container,
                    codec,
                    wait_label: label,
                });
            } else {
                let mime = match unit.kind {
                    MediaKind::Image => "image/webp",
                    MediaKind::Audio => "audio/mp3",
                    MediaKind::Video => "video/mp4",
                };
                actions.push(inline_action(unit.clone(), mime));
            }
        }
    } else {
        if !inventory.images.is_empty() {
            let count = inventory.images.len().min(10);
            actions.push(MediaAction::ImageBatch {
                units: inventory.images.iter().take(count).cloned().collect(),
                wait_label: if inventory.images.len() > 10 {
                    "Uploading reference images (10+)".into()
                } else {
                    "Uploading reference images".into()
                },
                show_batch_index: inventory.images.len() <= 10,
            });
            for unit in inventory.images.iter().skip(count) {
                actions.push(inline_action(unit.clone(), "image/png"));
            }
        }
        if input.node == "GeminiNode" {
            for unit in &inventory.audios {
                actions.push(inline_action(unit.clone(), "audio/mp3"));
            }
            for unit in &inventory.videos {
                actions.push(inline_action(unit.clone(), "video/mp4"));
            }
        }
    }
    Ok(actions)
}
fn valid_base64(value: &str) -> bool {
    if value.is_empty() || !value.len().is_multiple_of(4) {
        return false;
    }
    let unpadded = value.trim_end_matches('=');
    let padding = value.len() - unpadded.len();
    padding <= 2
        && !unpadded.is_empty()
        && unpadded
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/'))
}
fn validate_part(part: &Value) -> Result<(), GeminiError> {
    if let Some(inline) = part.get("inlineData") {
        let data = text(&inline["data"])?;
        if !valid_base64(data) {
            return Err(GeminiError::Input(
                "Canonical owner returned invalid inline base64",
            ));
        }
        text(&inline["mimeType"])?;
    } else if let Some(file) = part.get("fileData") {
        if endpoint(text(&file["fileUri"])?)? {
            return Err(GeminiError::Input(
                "Canonical storage URL must be absolute HTTPS",
            ));
        }
        text(&file["mimeType"])?;
    } else {
        return Err(GeminiError::Input("Canonical media part is missing data"));
    }
    Ok(())
}
fn prepare_parts(
    input: &Input,
    inventory: &MediaInventory,
    owner: &mut dyn GeminiInputOwner,
    transport: &mut dyn GeminiTransport,
    handle: StreamHandle,
) -> Result<Vec<Value>, GeminiError> {
    let mut parts = vec![];
    let mut inline_bytes = 0usize;
    let bounded_inline = matches!(input.node.as_str(), "GeminiNodeV2" | "GeminiVideoOmni");
    for action in media_actions(input, inventory)? {
        transport.check_cancelled(Some(handle))?;
        match action {
            MediaAction::ImageBatch {
                units,
                wait_label,
                show_batch_index,
            } => {
                let urls = owner.upload_images(&units, &wait_label, show_batch_index)?;
                if urls.len() != units.len() {
                    return Err(GeminiError::Input("Canonical image upload count differs"));
                }
                for url in urls {
                    let part = json!({"fileData":{"mimeType":"image/png","fileUri":url}});
                    validate_part(&part)?;
                    parts.push(part);
                }
            }
            MediaAction::Url {
                unit,
                mime_type,
                container,
                codec,
                wait_label,
            } => {
                let url = owner.upload_media(&unit, mime_type, container, codec, wait_label)?;
                let part = json!({"fileData":{"mimeType":mime_type,"fileUri":url}});
                validate_part(&part)?;
                parts.push(part);
            }
            MediaAction::Inline {
                unit,
                mime_type,
                container,
                codec,
            } => {
                let data = owner.inline_media(&unit, mime_type, container, codec)?;
                if bounded_inline {
                    inline_bytes = inline_bytes
                        .checked_add(data.len())
                        .ok_or(ConformanceError::BodyLimit)?;
                    if inline_bytes > MAXIMUM_INLINE_BYTES {
                        return Err(ConformanceError::BodyLimit.into());
                    }
                }
                let part = json!({"inlineData":{"mimeType":mime_type,"data":data}});
                validate_part(&part)?;
                parts.push(part);
            }
        }
    }
    Ok(parts)
}
#[derive(Clone, Debug)]
pub enum MediaPayload {
    Inline {
        mime_type: String,
        data: String,
    },
    Downloaded {
        mime_type: String,
        url: String,
        bytes: Vec<u8>,
        receipt: Vec<u8>,
    },
}
#[derive(Clone, Debug)]
pub struct MediaResult {
    pub port: String,
    pub payload: MediaPayload,
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub node: String,
    pub text: String,
    pub files: Vec<Value>,
    pub media: Vec<MediaResult>,
    pub empty_thought_shape: Option<[u64; 4]>,
    pub usd: Option<f64>,
    pub price_warning: Option<String>,
    pub response_bytes: Vec<u8>,
    pub response_receipt: Vec<u8>,
}
pub fn parts_by_type(response: &Value, kind: &str) -> Result<Vec<Value>, GeminiError> {
    let candidates=response["candidates"].as_array().filter(|candidates|!candidates.is_empty()).ok_or_else(||{if let Some(reason)=response["promptFeedback"]["blockReason"].as_str().filter(|reason|!reason.is_empty()){GeminiError::Provider(format!("Gemini API blocked the request. Reason: {reason} ({})",response["promptFeedback"]["blockReasonMessage"].as_str().unwrap_or("None")))}else{GeminiError::Provider("Gemini API returned no response candidates. If you are using the `IMAGE` modality, try changing it to `IMAGE+TEXT` to view the model's reasoning and understand why image generation failed.".into())}})?;
    let mut selected = vec![];
    let mut blocked = vec![];
    for candidate in candidates {
        if candidate["finishReason"]
            .as_str()
            .is_some_and(|value| value.eq_ignore_ascii_case("IMAGE_PROHIBITED_CONTENT"))
        {
            blocked.push(
                candidate["finishReason"]
                    .as_str()
                    .ok_or(GeminiError::Response("Invalid finish reason"))?,
            );
            continue;
        }
        if let Some(parts) = candidate["content"]["parts"].as_array() {
            for part in parts {
                let matches = if kind == "text" {
                    part["text"].as_str().is_some_and(|text| !text.is_empty())
                } else {
                    let prefix = kind.trim_end_matches('*');
                    part["inlineData"]["mimeType"]
                        .as_str()
                        .is_some_and(|mime| mime.starts_with(prefix))
                        || part["fileData"]["mimeType"]
                            .as_str()
                            .is_some_and(|mime| mime.starts_with(prefix))
                };
                if matches {
                    selected.push(part.clone());
                }
            }
        }
    }
    if selected.is_empty() && !blocked.is_empty() {
        let reasons = blocked
            .iter()
            .map(|value| format!("'{value}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(GeminiError::Provider(format!(
            "Gemini API blocked the request. Reasons: [{reasons}]"
        )));
    }
    Ok(selected)
}
pub fn response_text(response: &Value) -> Result<String, GeminiError> {
    Ok(parts_by_type(response, "text")?
        .iter()
        .map(|part| text(&part["text"]).map(str::to_owned))
        .collect::<Result<Vec<_>, GeminiError>>()?
        .join("\n"))
}
pub fn source_price(response: &Value) -> Result<Option<f64>, GeminiError> {
    let Some(model) = response["modelVersion"].as_str() else {
        return Ok(None);
    };
    let (input, text, image, video) = match model {
        "gemini-2.5-pro" => (1.25, 10.0, 0.0, 0.0),
        "gemini-2.5-flash" => (0.3, 2.5, 0.0, 0.0),
        "gemini-2.5-flash-image" => (0.3, 2.5, 30.0, 0.0),
        "gemini-3-pro-preview" | "gemini-3.1-pro-preview" => (2.0, 12.0, 0.0, 0.0),
        "gemini-3.1-flash-lite-preview" | "gemini-3.1-flash-lite" => (0.25, 1.5, 0.0, 0.0),
        "gemini-3-pro-image-preview" | "gemini-3-pro-image" => (2.0, 12.0, 120.0, 0.0),
        "gemini-3.1-flash-image-preview" | "gemini-3.1-flash-image" => (0.5, 3.0, 60.0, 0.0),
        "gemini-3.1-flash-lite-image" => (0.25, 1.5, 30.0, 0.0),
        "gemini-omni-flash-preview" => (2.145, 12.87, 0.0, 25.025),
        _ => return Ok(None),
    };
    let usage = &response["usageMetadata"];
    let mut price = integer(&usage["promptTokenCount"], u32::MAX as u64)? as f64 * input;
    if let Some(details) = usage["candidatesTokensDetails"].as_array() {
        for detail in details {
            let rate = match detail["modality"].as_str() {
                Some("IMAGE") => image,
                Some("VIDEO") => video,
                _ => text,
            };
            price += integer(&detail["tokenCount"], u32::MAX as u64)? as f64 * rate;
        }
    }
    if let Some(value) = usage
        .get("thoughtsTokenCount")
        .filter(|value| !value.is_null())
    {
        price += integer(value, u32::MAX as u64)? as f64 * text;
    }
    Ok(Some(price / 1_000_000.0))
}
fn media_payload(
    part: &Value,
    port: &str,
    transport: &mut dyn GeminiTransport,
) -> Result<MediaResult, GeminiError> {
    if let Some(inline) = part.get("inlineData").filter(|value| !value.is_null()) {
        let data = text(&inline["data"])?;
        if !valid_base64(data) || data.len() > MAXIMUM_RESPONSE_BYTES {
            return Err(GeminiError::Response("Invalid Gemini inline media"));
        }
        return Ok(MediaResult {
            port: port.into(),
            payload: MediaPayload::Inline {
                mime_type: text(&inline["mimeType"])?.into(),
                data: data.into(),
            },
        });
    }
    let file = &part["fileData"];
    let url = text(&file["fileUri"])?;
    if endpoint(url)? {
        return Err(GeminiError::Response("Invalid absolute Gemini media URL"));
    }
    let head = request_head(
        url,
        HttpMethod::Get,
        CredentialAvailability {
            auth_token: false,
            api_key: false,
        },
        true,
    )?;
    let (bytes, receipt, _) = exchange_with_policy(
        &head,
        &[],
        None,
        MAXIMUM_MEDIA_BYTES,
        DOWNLOAD_RETRY,
        transport,
    )?;
    if bytes.is_empty() {
        return Err(GeminiError::Response("Empty Gemini downloaded media"));
    }
    Ok(MediaResult {
        port: port.into(),
        payload: MediaPayload::Downloaded {
            mime_type: text(&file["mimeType"])?.into(),
            url: url.into(),
            bytes,
            receipt,
        },
    })
}
fn validate_response(response: &Value) -> Result<(), GeminiError> {
    if !response.is_object() {
        return Err(GeminiError::Response("Gemini response must be an object"));
    }
    if let Some(model) = response
        .get("modelVersion")
        .filter(|value| !value.is_null())
    {
        text(model)?;
    }
    if let Some(usage) = response
        .get("usageMetadata")
        .filter(|value| !value.is_null())
    {
        if !usage.is_object() {
            return Err(GeminiError::Response("Invalid Gemini usage metadata"));
        }
        for field in [
            "cachedContentTokenCount",
            "candidatesTokenCount",
            "promptTokenCount",
            "thoughtsTokenCount",
            "toolUsePromptTokenCount",
        ] {
            if let Some(value) = usage.get(field).filter(|value| !value.is_null())
                && !value.is_i64()
                && !value.is_u64()
            {
                return Err(GeminiError::Response("Invalid Gemini token count"));
            }
        }
        for field in ["candidatesTokensDetails", "promptTokensDetails"] {
            if let Some(details) = usage.get(field).filter(|value| !value.is_null()) {
                for detail in details.as_array().ok_or(GeminiError::Response(
                    "Invalid Gemini modality token details",
                ))? {
                    if !detail.is_object() {
                        return Err(GeminiError::Response(
                            "Invalid Gemini modality token detail",
                        ));
                    }
                    if let Some(modality) = detail.get("modality").filter(|value| !value.is_null())
                    {
                        choice(
                            modality,
                            &[
                                "MODALITY_UNSPECIFIED",
                                "TEXT",
                                "IMAGE",
                                "VIDEO",
                                "AUDIO",
                                "DOCUMENT",
                            ],
                        )?;
                    }
                    if let Some(count) = detail.get("tokenCount").filter(|value| !value.is_null())
                        && !count.is_i64()
                        && !count.is_u64()
                    {
                        return Err(GeminiError::Response("Invalid Gemini modality token count"));
                    }
                }
            }
        }
    }
    if let Some(candidates) = response.get("candidates").filter(|value| !value.is_null()) {
        let candidates = candidates
            .as_array()
            .ok_or(GeminiError::Response("Invalid Gemini candidates"))?;
        for candidate in candidates {
            if !candidate.is_object() {
                return Err(GeminiError::Response("Invalid Gemini candidate"));
            }
            if let Some(content) = candidate.get("content").filter(|value| !value.is_null()) {
                choice(&content["role"], &["user", "model"])?;
                if let Some(parts) = content.get("parts") {
                    for part in parts
                        .as_array()
                        .ok_or(GeminiError::Response("Invalid Gemini parts"))?
                    {
                        if !part.is_object() {
                            return Err(GeminiError::Response("Invalid Gemini part"));
                        }
                        if let Some(value) = part.get("text").filter(|value| !value.is_null()) {
                            text(value)?;
                        }
                        for name in ["inlineData", "fileData"] {
                            if let Some(media) = part.get(name).filter(|value| !value.is_null()) {
                                if !media.is_object() {
                                    return Err(GeminiError::Response(
                                        "Invalid Gemini media object",
                                    ));
                                }
                                if let Some(mime) =
                                    media.get("mimeType").filter(|value| !value.is_null())
                                {
                                    choice(
                                        mime,
                                        &[
                                            "application/pdf",
                                            "audio/mpeg",
                                            "audio/mp3",
                                            "audio/wav",
                                            "image/png",
                                            "image/jpeg",
                                            "image/webp",
                                            "text/plain",
                                            "video/mov",
                                            "video/mpeg",
                                            "video/mp4",
                                            "video/mpg",
                                            "video/avi",
                                            "video/wmv",
                                            "video/mpegps",
                                            "video/flv",
                                        ],
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
fn parse_result(
    input: &Input,
    bytes: Vec<u8>,
    receipt: Vec<u8>,
    transport: &mut dyn GeminiTransport,
) -> Result<ResultValue, GeminiError> {
    let response: Value = serde_json::from_slice(&bytes)?;
    validate_response(&response)?;
    let output_text = response_text(&response)?;
    let (usd, price_warning) = match source_price(&response) {
        Ok(price) => (price, None),
        Err(error) => (None, Some(error.to_string())),
    };
    let mut result = ResultValue {
        node: input.node.clone(),
        text: output_text.clone(),
        files: vec![],
        media: vec![],
        empty_thought_shape: None,
        usd,
        price_warning,
        response_bytes: bytes,
        response_receipt: receipt,
    };
    match input.node.as_str() {
        "GeminiNode" | "GeminiNodeV2" => {
            if result.text.is_empty() {
                result.text = "Empty response from Gemini model...".into();
            }
        }
        "GeminiVideoOmni" => {
            let video=parts_by_type(&response,"video/*")?.into_iter().find(|part|part["inlineData"]["data"].as_str().is_some_and(|value|!value.is_empty())||part["fileData"]["fileUri"].as_str().is_some_and(|value|!value.is_empty())).ok_or_else(||GeminiError::Provider(if output_text.trim().is_empty(){"Gemini did not generate a video. Try rephrasing your prompt, shortening the requested duration, or reducing the number of input images/videos.".into()}else{format!("Gemini did not generate a video. Model response: {}",output_text.trim())}))?;
            result
                .media
                .push(media_payload(&video, "output_0", transport)?);
        }
        "GeminiImageNode" | "GeminiImage2Node" | "GeminiNanoBanana2" | "GeminiNanoBanana2V2" => {
            let image_parts = parts_by_type(&response, "image/*")?;
            let generated = image_parts
                .iter()
                .filter(|part| part["thought"].as_bool() != Some(true))
                .collect::<Vec<_>>();
            if generated.is_empty() {
                return Err(GeminiError::Provider(if output_text.trim().is_empty() {
                    "Gemini did not generate an image. Try rephrasing your prompt or changing the response modality to 'IMAGE+TEXT' to see the model's reasoning.".into()
                } else {
                    format!(
                        "Gemini did not generate an image. Model response: {}",
                        output_text.trim()
                    )
                }));
            }
            for part in generated {
                result
                    .media
                    .push(media_payload(part, "output_0", transport)?);
            }
            if matches!(
                input.node.as_str(),
                "GeminiNanoBanana2" | "GeminiNanoBanana2V2"
            ) {
                let thoughts = image_parts
                    .iter()
                    .filter(|part| part["thought"].as_bool() == Some(true))
                    .collect::<Vec<_>>();
                if thoughts.is_empty() {
                    result.empty_thought_shape = Some([1, 1024, 1024, 4]);
                } else {
                    for part in thoughts {
                        result
                            .media
                            .push(media_payload(part, "output_2", transport)?);
                    }
                }
            }
        }
        _ => return Err(GeminiError::Input("Unknown Gemini response node")),
    }
    Ok(result)
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn GeminiTransport,
    owner: &mut dyn GeminiInputOwner,
) -> Result<ResultValue, GeminiError> {
    let input = normalized(input)?;
    if input.node == "GeminiInputFiles" {
        let name = text(argument(&input, "file")?)?;
        let preflight = Resolution {
            local_file_part: Some(
                json!({"inlineData":{"mimeType":if name.ends_with(".pdf"){"application/pdf"}else{"text/plain"},"data":"YQ=="}}),
            ),
            ..Resolution::default()
        };
        request(&input, &preflight)?;
        transport.check_cancelled(None)?;
        let part = owner.read_file_part(name, MAXIMUM_INPUT_FILE_BYTES)?;
        validate_part(&part)?;
        if part["inlineData"]["data"]
            .as_str()
            .is_none_or(|value| value.len() > MAXIMUM_INPUT_FILE_BYTES.div_ceil(3) * 4)
        {
            return Err(ConformanceError::BodyLimit.into());
        }
        let files = owner.existing_file_parts(&input)?;
        let local = request(
            &input,
            &Resolution {
                local_file_part: Some(part),
                file_parts: files,
                ..Resolution::default()
            },
        )?;
        return Ok(ResultValue {
            node: input.node.clone(),
            text: String::new(),
            files: local
                .body
                .as_array()
                .ok_or(GeminiError::Response("Invalid local Gemini files"))?
                .clone(),
            media: vec![],
            empty_thought_shape: None,
            usd: None,
            price_warning: None,
            response_bytes: vec![],
            response_receipt: vec![],
        });
    }
    request_head(
        "/proxy/vertexai/gemini/preflight",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    transport.check_cancelled(None)?;
    let inventory = owner.inventory(&input)?;
    let mut resolution = Resolution {
        image_count: inventory.images.len(),
        video_durations: if input.node == "GeminiVideoOmni" {
            inventory
                .videos
                .iter()
                .map(|unit| {
                    unit.duration_seconds
                        .ok_or(GeminiError::Input("Canonical video duration is required"))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![]
        },
        ..Resolution::default()
    };
    let preflight = request(&input, &resolution)?;
    let head = request_head(
        preflight
            .path
            .as_deref()
            .ok_or(GeminiError::Input("Missing Gemini endpoint"))?,
        HttpMethod::Post,
        credentials,
        false,
    )?;
    transport.check_cancelled(None)?;
    let handle = transport.start_request(&head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    approve(handle, maximum_cost_microunits, transport)?;
    resolution.media_parts = prepare_parts(&input, &inventory, owner, transport, handle)?;
    transport.check_cancelled(Some(handle))?;
    resolution.file_parts = owner.existing_file_parts(&input)?;
    for part in &resolution.file_parts {
        validate_part(part)?;
    }
    let request = request(&input, &resolution)?;
    let body = serde_json::to_vec(&request.body)?;
    if body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let (bytes, receipt, _) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    parse_result(&input, bytes, receipt, transport)
}
fn approve(
    handle: StreamHandle,
    maximum_microunits: u64,
    transport: &mut dyn GeminiTransport,
) -> Result<(), GeminiError> {
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "gemini.generation.maximum".into(),
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
    Ok(())
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
    transport: &mut dyn GeminiTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), GeminiError> {
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
    transport: &mut dyn GeminiTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), GeminiError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(GeminiError::Http {
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
                    return Err(GeminiError::Http {
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
    transport: &mut dyn GeminiTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), GeminiError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "gemini.generation.maximum".into(),
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
                    return Err(GeminiError::Http {
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

pub trait GeminiMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn GeminiTransport,
    owner: &mut dyn GeminiInputOwner,
    materializer: &mut dyn GeminiMaterializer,
    sink: &mut dyn MaterializationSink,
) -> Result<(), GeminiError> {
    let result = invoke(
        input,
        credentials,
        maximum_cost_microunits,
        transport,
        owner,
    )?;
    transport.check_cancelled(None)?;
    let outputs = materializer.materialize(&result)?;
    let aliases = match input.node.as_str() {
        "GeminiInputFiles" => vec!["GEMINI_INPUT_FILES"],
        "GeminiNode" | "GeminiNodeV2" => vec!["STRING"],
        "GeminiVideoOmni" => vec!["VIDEO", "STRING"],
        "GeminiImageNode" | "GeminiImage2Node" => vec!["IMAGE", "STRING"],
        "GeminiNanoBanana2" | "GeminiNanoBanana2V2" => vec!["IMAGE", "STRING", "IMAGE"],
        _ => return Err(GeminiError::Input("Unknown Gemini output schema")),
    };
    let registry =
        TypeRegistry::built_in().map_err(|_| GeminiError::Response("Missing type registry"))?;
    let expected = aliases
        .into_iter()
        .enumerate()
        .map(|(index, alias)| {
            let type_id = registry
                .resolve(alias)
                .map_err(|_| GeminiError::Response("Missing output type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| GeminiError::Response("Missing output family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect::<Result<Vec<_>, GeminiError>>()?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn fixture_resolution(case: &Value) -> Result<Resolution, GeminiError> {
    let input = &case["input"];
    let node = text(&input["node"])?;
    if node == "GeminiInputFiles" {
        let files = case["outputs"][0]
            .as_array()
            .ok_or(GeminiError::Input("Missing local file probe"))?;
        return Ok(Resolution {
            local_file_part: files.first().cloned(),
            file_parts: files.iter().skip(1).cloned().collect(),
            ..Resolution::default()
        });
    }
    let body = &case["requests"][0]["body"];
    let parts = body["contents"][0]["parts"]
        .as_array()
        .ok_or(GeminiError::Input("Missing parts"))?;
    let files = input["arguments"]
        .get("files")
        .or_else(|| input["arguments"]["model"].get("files"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let media_parts = parts
        .iter()
        .filter(|part| part.get("text").is_none())
        .take(
            parts
                .iter()
                .filter(|part| part.get("text").is_none())
                .count()
                .saturating_sub(files.len()),
        )
        .cloned()
        .collect();
    fn image_count(value: &Value) -> usize {
        if let Some(count) = value["image_count"].as_u64() {
            count as usize
        } else if let Some(values) = value.as_object() {
            values.values().map(image_count).sum()
        } else {
            0
        }
    }
    let images = input["arguments"]
        .get("images")
        .unwrap_or(&input["arguments"]["model"]["images"]);
    let videos = input["arguments"]["model"]["videos"]
        .as_object()
        .map(|values| {
            values
                .values()
                .map(|video| video["duration"].as_f64().unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    Ok(Resolution {
        media_parts,
        file_parts: files,
        image_count: image_count(images),
        video_durations: videos,
        local_file_part: None,
    })
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"scope":"source request and bounded transport; canonical native media, file ownership and guest integration pending"})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let oracle = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
    if oracle["source_sha256"]["comfy_api_nodes/nodes_gemini.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in oracle["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let observed = request(
            &input,
            &fixture_resolution(case).map_err(|_| ConformanceError::InvalidRequest)?,
        )
        .map_err(|_| ConformanceError::InvalidRequest)?;
        if input.node == "GeminiInputFiles" {
            if observed.path.is_some() || case["outputs"][0] != observed.body {
                return Err(ConformanceError::InvalidRequest);
            }
        } else if case["requests"][0]["path"] != observed.path.unwrap_or_default()
            || case["requests"][0]["body"] != observed.body
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
        let resolution = fixture_resolution(
            &json!({"input":case["input"],"requests":[{"body":{"contents":[{"parts":[]}]}}]}),
        )
        .map_err(|_| ConformanceError::InvalidRequest)?;
        if request(&input, &resolution).is_ok() || case["requests"] != json!([]) {
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
        CostResponse, MaterializationError, MaterializedBatch, PluginValue, Progress, ScalarValue,
        SignedPutRequest, StorageCreateRequest, StorageCreateResponse, ValueFamily,
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
            _progress: &Progress,
        ) -> Result<(), ConformanceError> {
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
    impl GeminiTransport for Transport {
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
            "/../../sim_test_support/fixtures/providers/gemini/downloaded-image.png"
        ))
        .to_vec()
    }
    fn png_base64() -> String {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/gemini/downloaded-image.base64"
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
    fn response(parts: Vec<Value>) -> Value {
        json!({"candidates":[{"content":{"role":"model","parts":parts}}]})
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
    #[derive(Default)]
    struct InputOwner {
        inventory: MediaInventory,
        files: Vec<Value>,
        operations: Vec<String>,
        deny_native: bool,
        inline_override: Option<String>,
    }
    impl GeminiInputOwner for InputOwner {
        fn inventory(&mut self, _input: &Input) -> Result<MediaInventory, ConformanceError> {
            Ok(self.inventory.clone())
        }
        fn existing_file_parts(&mut self, _input: &Input) -> Result<Vec<Value>, ConformanceError> {
            Ok(self.files.clone())
        }
        fn read_file_part(
            &mut self,
            name: &str,
            maximum: usize,
        ) -> Result<Value, ConformanceError> {
            self.operations.push(format!("read:{name}:{maximum}"));
            if name != "input.txt" {
                return Err(ConformanceError::HostFailure);
            }
            Ok(json!({"inlineData":{"mimeType":"text/plain","data":"c291cmNlIHRleHQgY2Fmw6kK"}}))
        }
        fn upload_images(
            &mut self,
            images: &[MediaUnit],
            label: &str,
            show_index: bool,
        ) -> Result<Vec<String>, ConformanceError> {
            self.operations
                .push(format!("batch:{}:{label}:{show_index}", images.len()));
            if self.deny_native {
                return Err(ConformanceError::HostFailure);
            }
            Ok((0..images.len())
                .map(|index| format!("https://storage.fixture.invalid/image/{index}"))
                .collect())
        }
        fn upload_media(
            &mut self,
            unit: &MediaUnit,
            mime: &str,
            container: Option<&str>,
            codec: Option<&str>,
            label: Option<&str>,
        ) -> Result<String, ConformanceError> {
            self.operations.push(format!(
                "url:{}:{mime}:{container:?}:{codec:?}:{label:?}",
                unit.port
            ));
            if self.deny_native {
                return Err(ConformanceError::HostFailure);
            }
            Ok(format!("https://storage.fixture.invalid/{}", unit.port))
        }
        fn inline_media(
            &mut self,
            unit: &MediaUnit,
            mime: &str,
            container: Option<&str>,
            codec: Option<&str>,
        ) -> Result<String, ConformanceError> {
            self.operations.push(format!(
                "inline:{}:{mime}:{container:?}:{codec:?}",
                unit.port
            ));
            if self.deny_native || mime != "image/png" && self.inline_override.is_none() {
                return Err(ConformanceError::HostFailure);
            }
            Ok(self.inline_override.clone().unwrap_or_else(png_base64))
        }
    }
    fn unit(kind: MediaKind, port: &str) -> MediaUnit {
        MediaUnit {
            kind,
            port: port.into(),
            source: json!({"canonical_native_input":port}),
            batch_index: 0,
            duration_seconds: if kind == MediaKind::Video {
                Some(1.0)
            } else {
                None
            },
        }
    }
    #[test]
    fn gemini_claims_defaults_requests_and_source_failures_are_exact() -> Result<(), Box<dyn Error>>
    {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/gemini/conformance.fixture"
        )))?;
        assert_eq!(NODES.len(), 8);
        assert_eq!(ROUTES.len(), 4);
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "executable")
                .count(),
            2
        );
        assert_eq!(ALIASES, &["vertexai"]);
        assert_eq!(oracle()?["cases"].as_array().ok_or("cases")?.len(), 10);
        assert_eq!(
            oracle()?["preflight_errors"]
                .as_array()
                .ok_or("errors")?
                .len(),
            5
        );
        Ok(())
    }
    #[test]
    fn gemini_only_text_v2_forwards_positive_seed_and_legacy_aliases_are_exact()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "GeminiNode",
            "GeminiImageNode",
            "GeminiImage2Node",
            "GeminiNanoBanana2",
            "GeminiNanoBanana2V2",
            "GeminiVideoOmni",
        ] {
            let case = fixture_case(node)?;
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let observed = request(&input, &fixture_resolution(&case)?)?;
            assert!(
                observed
                    .body
                    .get("generationConfig")
                    .and_then(|value| value.get("seed"))
                    .is_none()
            );
        }
        let case = fixture_case("GeminiNodeV2")?;
        let mut input: Input = serde_json::from_value(case["input"].clone())?;
        assert_eq!(
            request(&input, &fixture_resolution(&case)?)?.body["generationConfig"]["seed"],
            42
        );
        input.arguments.insert("seed".into(), json!(0));
        assert!(
            request(&input, &fixture_resolution(&case)?)?.body["generationConfig"]
                .get("seed")
                .is_none()
        );
        input.arguments.insert("seed".into(), json!(u64::MAX));
        assert!(request(&input, &fixture_resolution(&case)?).is_err());
        assert_eq!(
            legacy_text_model_alias("gemini-3-pro-preview"),
            "gemini-3.1-pro-preview"
        );
        assert_eq!(
            legacy_text_model_alias("gemini-3-1-flash-lite"),
            "gemini-3.1-flash-lite-preview"
        );
        Ok(())
    }
    #[test]
    fn gemini_media_planning_matches_actual_source_order_and_codec_branches()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["cases"].as_array().ok_or("cases")? {
            let node = text(&case["input"]["node"])?;
            if !matches!(node, "GeminiNodeV2" | "GeminiVideoOmni") {
                continue;
            }
            let mut inventory = MediaInventory::default();
            for operation in case["media"].as_array().ok_or("media")? {
                let kind = text(&operation["operation"])?;
                let slot = text(&operation["slot"])?;
                let unit = unit(
                    if kind.ends_with("image") {
                        MediaKind::Image
                    } else if kind.ends_with("audio") {
                        MediaKind::Audio
                    } else {
                        MediaKind::Video
                    },
                    slot,
                );
                match unit.kind {
                    MediaKind::Image => inventory.images.push(unit),
                    MediaKind::Audio => inventory.audios.push(unit),
                    MediaKind::Video => inventory.videos.push(unit),
                }
            }
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let actions = media_actions(&input, &inventory)?;
            let source = case["media"].as_array().ok_or("media")?;
            assert_eq!(actions.len(), source.len());
            for (action, expected) in actions.iter().zip(source) {
                match action {
                    MediaAction::Url {
                        unit,
                        mime_type,
                        container,
                        codec,
                        ..
                    } => {
                        assert!(text(&expected["operation"])?.starts_with("upload_"));
                        assert_eq!(unit.port, expected["slot"]);
                        if unit.kind == MediaKind::Audio {
                            assert_eq!(*mime_type, "audio/mp3");
                            assert_eq!(*container, Some("mp3"));
                            assert_eq!(*codec, Some("libmp3lame"));
                        }
                    }
                    MediaAction::Inline {
                        unit, mime_type, ..
                    } => {
                        assert!(text(&expected["operation"])?.starts_with("inline_"));
                        assert_eq!(unit.port, expected["slot"]);
                        if unit.kind == MediaKind::Image {
                            assert_eq!(*mime_type, "image/webp");
                        }
                    }
                    MediaAction::ImageBatch { .. } => return Err("unexpected image batch".into()),
                }
            }
        }
        let input = fixture_input("GeminiImage2Node")?;
        let inventory = MediaInventory {
            images: (0..12)
                .map(|index| unit(MediaKind::Image, &format!("image/{index}")))
                .collect(),
            ..MediaInventory::default()
        };
        let actions = media_actions(&input, &inventory)?;
        assert_eq!(actions.len(), 3);
        assert!(
            matches!(actions.first(),Some(MediaAction::ImageBatch{units,show_batch_index:false,wait_label})if units.len()==10&&wait_label=="Uploading reference images (10+)")
        );
        assert!(actions.iter().skip(1).all(|action| matches!(
            action,
            MediaAction::Inline {
                mime_type: "image/png",
                ..
            }
        )));
        Ok(())
    }
    #[test]
    fn gemini_token_prices_match_all_fifteen_actual_source_cases() -> Result<(), Box<dyn Error>> {
        for case in oracle()?["price_cases"].as_array().ok_or("prices")? {
            match (source_price(&case["response"])?, case["usd"].as_f64()) {
                (Some(observed), Some(expected)) => assert!((observed - expected).abs() < 1e-15),
                (None, None) => {}
                _ => return Err("source price differs".into()),
            }
        }
        let mut response = oracle()?["price_cases"][0]["response"].clone();
        response["usageMetadata"]["candidatesTokenCount"] = json!(999999);
        let first = source_price(&response)?;
        response["usageMetadata"]["candidatesTokenCount"] = json!(0);
        assert_eq!(source_price(&response)?, first);
        Ok(())
    }
    #[test]
    fn gemini_response_filtering_and_blocked_messages_match_actual_source()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["response_cases"].as_array().ok_or("responses")? {
            match response_text(&case["response"]) {
                Ok(observed) => {
                    assert_eq!(observed, case["result"]["text"]);
                    assert_eq!(
                        json!(parts_by_type(&case["response"], "image/*")?),
                        case["result"]["image_parts"]
                    );
                }
                Err(error) => assert_eq!(error.to_string(), case["result"]["error"]),
            }
        }
        Ok(())
    }
    #[test]
    fn gemini_local_input_files_prepend_real_content_without_auth_or_cost()
    -> Result<(), Box<dyn Error>> {
        let case = fixture_case("GeminiInputFiles")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut owner = InputOwner {
            files: fixture_resolution(&case)?.file_parts,
            ..InputOwner::default()
        };
        let mut transport = Transport::default();
        let result = invoke(
            &input,
            CredentialAvailability {
                auth_token: false,
                api_key: false,
            },
            0,
            &mut transport,
            &mut owner,
        )?;
        assert_eq!(json!(result.files), case["outputs"][0]);
        assert_eq!(
            owner.operations,
            vec![format!("read:input.txt:{MAXIMUM_INPUT_FILE_BYTES}")]
        );
        assert!(transport.heads.is_empty());
        assert!(transport.costs.is_empty());
        let registry = TypeRegistry::built_in()?;
        assert_eq!(
            registry.family(registry.resolve("GEMINI_INPUT_FILES")?)?,
            ValueFamily::Model
        );
        let mut input = input;
        input
            .arguments
            .insert("file".into(), json!("../escape.txt"));
        assert!(invoke(&input, credentials(), 0, &mut transport, &mut owner).is_err());
        Ok(())
    }
    #[test]
    fn gemini_source_empty_prompt_and_media_limits_precede_paid_body() -> Result<(), Box<dyn Error>>
    {
        for case in oracle()?["preflight_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let count = if matches!(
                input.node.as_str(),
                "GeminiImage2Node" | "GeminiNanoBanana2V2"
            ) {
                15
            } else {
                0
            };
            let videos = if input.node == "GeminiVideoOmni" {
                if case["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("duration"))
                {
                    vec![MediaUnit {
                        duration_seconds: Some(11.0),
                        ..unit(MediaKind::Video, "long")
                    }]
                } else {
                    (0..4)
                        .map(|index| unit(MediaKind::Video, &format!("video{index}")))
                        .collect()
                }
            } else {
                vec![]
            };
            let mut owner = InputOwner {
                inventory: MediaInventory {
                    images: (0..count)
                        .map(|index| unit(MediaKind::Image, &format!("image{index}")))
                        .collect(),
                    videos,
                    ..MediaInventory::default()
                },
                ..InputOwner::default()
            };
            let mut transport = Transport::default();
            assert!(invoke(&input, credentials(), 500, &mut transport, &mut owner).is_err());
            assert!(transport.heads.is_empty());
            assert!(owner.operations.is_empty());
        }
        Ok(())
    }
    #[test]
    fn gemini_missing_auth_and_denied_cost_prevent_media_actuation() -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GeminiImageNode")?;
        let mut owner = InputOwner {
            inventory: MediaInventory {
                images: vec![unit(MediaKind::Image, "image")],
                ..MediaInventory::default()
            },
            ..InputOwner::default()
        };
        let mut transport = Transport::default();
        assert!(
            invoke(
                &input,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                500,
                &mut transport,
                &mut owner
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        transport.deny_cost = true;
        assert!(matches!(
            invoke(&input, credentials(), 500, &mut transport, &mut owner),
            Err(GeminiError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(owner.operations.is_empty());
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn gemini_text_result_fallback_and_price_failure_continue_with_warning()
    -> Result<(), Box<dyn Error>> {
        let input = Input {
            node: "GeminiNodeV2".into(),
            arguments: BTreeMap::from([
                ("prompt".into(), json!("hello")),
                (
                    "model".into(),
                    json!({"model":"Gemini 3.1 Pro","thinking_level":"HIGH","temperature":1.0,"top_p":0.95,"max_output_tokens":32768}),
                ),
                ("seed".into(), json!(0)),
            ]),
        };
        let mut response = response(vec![]);
        response["modelVersion"] = json!("gemini-2.5-pro");
        let mut transport = Transport {
            events: events(200, serde_json::to_vec(&response)?, vec![]),
            ..Transport::default()
        };
        let result = invoke(
            &input,
            credentials(),
            500,
            &mut transport,
            &mut InputOwner::default(),
        )?;
        assert_eq!(result.text, "Empty response from Gemini model...");
        assert_eq!(result.usd, None);
        assert!(result.price_warning.is_some());
        let head = transport.heads.first().ok_or("head")?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.auth_token"));
        let body: Value = serde_json::from_slice(transport.bodies.first().ok_or("body")?)?;
        assert!(body["generationConfig"].get("seed").is_none());
        Ok(())
    }
    #[test]
    fn gemini_real_download_and_thought_empty_shape_preserve_source_semantics()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GeminiNanoBanana2V2")?;
        let response = response(vec![
            json!({"text":"reasoning","thought":true}),
            json!({"text":"final"}),
            json!({"fileData":{"mimeType":"image/png","fileUri":"https://download.fixture.invalid/image.png"}}),
        ]);
        let mut transport = Transport {
            events: events(200, png(), vec![]),
            ..Transport::default()
        };
        let result = parse_result(
            &input,
            serde_json::to_vec(&response)?,
            vec![42],
            &mut transport,
        )?;
        assert_eq!(result.text, "reasoning\nfinal");
        assert_eq!(result.empty_thought_shape, Some([1, 1024, 1024, 4]));
        assert_eq!(result.media.len(), 1);
        assert!(
            matches!(&result.media.first().ok_or("image")?.payload,MediaPayload::Downloaded{bytes,receipt,..}if bytes==&png()&&receipt==&vec![7,8,9])
        );
        let head = transport.heads.first().ok_or("head")?;
        assert_eq!(head.secret_id, None);
        assert!(head.headers.iter().all(|header| !matches!(
            header.name.as_str(),
            "Authorization" | "X-API-KEY" | "Comfy-Env"
        )));
        Ok(())
    }
    #[test]
    fn gemini_inline_images_and_first_video_are_kept_for_canonical_decoders()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GeminiNanoBanana2V2")?;
        let observed = response(vec![
            json!({"inlineData":{"mimeType":"image/png","data":png_base64()}}),
            json!({"inlineData":{"mimeType":"image/png","data":png_base64()},"thought":true}),
        ]);
        let mut transport = Transport::default();
        let result = parse_result(
            &input,
            serde_json::to_vec(&observed)?,
            vec![42],
            &mut transport,
        )?;
        assert_eq!(result.media.len(), 2);
        assert_eq!(result.media.get(1).ok_or("thought")?.port, "output_2");
        assert_eq!(result.empty_thought_shape, None);
        let input = fixture_input("GeminiVideoOmni")?;
        let observed = response(vec![
            json!({"fileData":{"mimeType":"video/mp4","fileUri":"https://download.fixture.invalid/first.mp4"}}),
            json!({"fileData":{"mimeType":"video/mp4","fileUri":"https://download.fixture.invalid/second.mp4"}}),
        ]);
        transport.events = vec![Err(ConformanceError::HostFailure)].into();
        assert!(
            parse_result(
                &input,
                serde_json::to_vec(&observed)?,
                vec![42],
                &mut transport
            )
            .is_err()
        );
        assert!(
            transport
                .heads
                .first()
                .is_some_and(|head| head.path.ends_with("first.mp4"))
        );
        Ok(())
    }
    #[test]
    fn gemini_missing_generated_image_and_video_return_source_model_messages()
    -> Result<(), Box<dyn Error>> {
        for node in ["GeminiImageNode", "GeminiVideoOmni"] {
            let input = fixture_input(node)?;
            let response = response(vec![json!({"text":" explain failure "})]);
            let error = parse_result(
                &input,
                serde_json::to_vec(&response)?,
                vec![42],
                &mut Transport::default(),
            )
            .err()
            .ok_or("expected error")?;
            assert!(
                error
                    .to_string()
                    .contains("Model response: explain failure")
            );
        }
        Ok(())
    }
    #[test]
    fn gemini_retries_require_fresh_handle_and_replay_unchanged_json() -> Result<(), Box<dyn Error>>
    {
        let mut input = fixture_input("GeminiNode")?;
        input.arguments = BTreeMap::from([
            ("prompt".into(), json!("retry")),
            ("model".into(), json!("gemini-3-1-pro")),
            ("seed".into(), json!(42)),
        ]);
        let mut transport = Transport {
            events: events(
                429,
                b"limit".to_vec(),
                vec![Header {
                    name: "Retry-After".into(),
                    value: "2".into(),
                }],
            ),
            ..Transport::default()
        };
        transport.events.extend(events(
            200,
            serde_json::to_vec(&response(vec![json!({"text":"success"})]))?,
            vec![],
        ));
        let result = invoke(
            &input,
            credentials(),
            500,
            &mut transport,
            &mut InputOwner::default(),
        )?;
        assert_eq!(result.text, "success");
        assert_eq!(transport.delays, vec![2000]);
        assert_eq!(transport.bodies.first(), transport.bodies.get(1));
        let mut transport = Transport {
            events: events(503, vec![1], vec![]),
            same_retry_handle: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                500,
                &mut transport,
                &mut InputOwner::default()
            ),
            Err(GeminiError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn gemini_cancel_worker_loss_malformed_and_oversized_responses_are_typed()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/vertexai/gemini/gemini-2.5-pro",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        for response_events in [
            vec![Err(ConformanceError::WorkerLost)],
            vec![Ok(ResponseEvent::Cancelled)],
            vec![Ok(ResponseEvent::Chunk(vec![1]))],
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
                events: response_events.into(),
                ..Transport::default()
            };
            assert!(
                exchange_with_policy(
                    &head,
                    b"body",
                    None,
                    MAXIMUM_RESPONSE_BYTES,
                    GENERATION_RETRY,
                    &mut transport
                )
                .is_err()
            );
        }
        let input = fixture_input("GeminiImageNode")?;
        assert!(
            parse_result(
                &input,
                b"{} trailing".to_vec(),
                vec![42],
                &mut Transport::default()
            )
            .is_err()
        );
        assert!(parse_result(&input,serde_json::to_vec(&json!({"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"unknown/image","data":"YQ=="}}]}}]}))?,vec![42],&mut Transport::default()).is_err());
        let mut transport = Transport {
            events: events(200, vec![0; 10], vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(&head, b"body", None, 5, GENERATION_RETRY, &mut transport),
            Err(GeminiError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn gemini_native_preparation_and_inline_overflow_fail_before_body_submission()
    -> Result<(), Box<dyn Error>> {
        let input = fixture_input("GeminiImageNode")?;
        let mut owner = InputOwner {
            inventory: MediaInventory {
                images: vec![unit(MediaKind::Image, "image")],
                ..MediaInventory::default()
            },
            deny_native: true,
            ..InputOwner::default()
        };
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 500, &mut transport, &mut owner),
            Err(GeminiError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        let input = fixture_input("GeminiNodeV2")?;
        owner.deny_native = false;
        owner.inventory = MediaInventory {
            images: (0..11)
                .map(|index| unit(MediaKind::Image, &format!("{index}")))
                .collect(),
            ..MediaInventory::default()
        };
        owner.inline_override = Some("A".repeat(MAXIMUM_INLINE_BYTES + 4));
        let mut transport = Transport::default();
        assert!(
            prepare_parts(
                &input,
                &owner.inventory.clone(),
                &mut owner,
                &mut transport,
                handle(1)
            )
            .is_err()
        );
        assert!(transport.bodies.is_empty());
        Ok(())
    }
    #[test]
    fn gemini_real_png_reaches_materializer_and_failure_publishes_nothing()
    -> Result<(), Box<dyn Error>> {
        struct Decoder;
        impl GeminiMaterializer for Decoder {
            fn materialize(
                &mut self,
                result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                assert!(
                    matches!(&result.media.first().ok_or(ConformanceError::InvalidResponse)?.payload,MediaPayload::Downloaded{bytes,..}if bytes==&png())
                );
                Err(ConformanceError::HostFailure)
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
        let input = Input {
            node: "GeminiImageNode".into(),
            arguments: BTreeMap::from([
                ("prompt".into(), json!("generate")),
                ("model".into(), json!("gemini-2.5-flash-image")),
                ("seed".into(), json!(42)),
            ]),
        };
        let mut transport = Transport {
            events: events(
                200,
                serde_json::to_vec(&response(vec![
                    json!({"fileData":{"mimeType":"image/png","fileUri":"https://download.fixture.invalid/image.png"}}),
                ]))?,
                vec![],
            ),
            ..Transport::default()
        };
        transport.events.extend(events(200, png(), vec![]));
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input,
                credentials(),
                500,
                &mut transport,
                &mut InputOwner::default(),
                &mut Decoder,
                &mut sink
            ),
            Err(GeminiError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(sink.batches, 0);
        Ok(())
    }
    #[test]
    fn gemini_text_uses_canonical_scalar_abi_and_publishes_atomically() -> Result<(), Box<dyn Error>>
    {
        struct TextOwner;
        impl GeminiMaterializer for TextOwner {
            fn materialize(
                &mut self,
                result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                let registry =
                    TypeRegistry::built_in().map_err(|_| ConformanceError::HostFailure)?;
                let type_id = registry
                    .resolve("STRING")
                    .map_err(|_| ConformanceError::HostFailure)?
                    .clone();
                let value = PluginValue::scalar(
                    type_id.clone(),
                    ScalarValue::String(result.text.clone()),
                    &registry,
                )
                .map_err(|_| ConformanceError::HostFailure)?;
                Ok(vec![EncodedOutput {
                    port_id: "output_0".into(),
                    type_id,
                    family: ValueFamily::Scalar,
                    bytes: value
                        .abi_bytes()
                        .map_err(|_| ConformanceError::HostFailure)?,
                }])
            }
        }
        #[derive(Default)]
        struct Sink {
            batches: Vec<MaterializedBatch>,
        }
        impl MaterializationSink for Sink {
            fn publish(&mut self, batch: MaterializedBatch) -> Result<(), MaterializationError> {
                self.batches.push(batch);
                Ok(())
            }
        }
        let input = Input {
            node: "GeminiNode".into(),
            arguments: BTreeMap::from([
                ("prompt".into(), json!("text")),
                ("model".into(), json!("gemini-2.5-pro")),
                ("seed".into(), json!(42)),
            ]),
        };
        let mut transport = Transport {
            events: events(
                200,
                serde_json::to_vec(&response(vec![json!({"text":"actual scalar text"})]))?,
                vec![],
            ),
            ..Transport::default()
        };
        let mut sink = Sink::default();
        invoke_and_materialize(
            &input,
            credentials(),
            500,
            &mut transport,
            &mut InputOwner::default(),
            &mut TextOwner,
            &mut sink,
        )?;
        assert_eq!(sink.batches.len(), 1);
        assert_eq!(sink.batches.first().ok_or("batch")?.outputs().len(), 1);
        Ok(())
    }
    #[test]
    fn gemini_python_prompt_strip_rejects_controls_in_six_branches_before_transport()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "GeminiNodeV2",
            "GeminiImageNode",
            "GeminiImage2Node",
            "GeminiNanoBanana2",
            "GeminiNanoBanana2V2",
            "GeminiVideoOmni",
        ] {
            for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                let mut input = fixture_input(node)?;
                let raw = format!("{separator} {separator}");
                if node == "GeminiVideoOmni" {
                    input.arguments.get_mut("model").ok_or("missing model")?["prompt"] = json!(raw);
                } else {
                    input.arguments.insert("prompt".into(), json!(raw));
                }
                let mut transport = Transport::default();
                let mut owner = InputOwner::default();
                assert!(matches!(
                    invoke(&input, credentials(), 500, &mut transport, &mut owner),
                    Err(GeminiError::Input("Empty Gemini prompt"))
                ));
                assert!(transport.heads.is_empty());
                assert!(transport.costs.is_empty());
                assert!(transport.bodies.is_empty());
                assert!(owner.operations.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn gemini_python_prompt_strip_preserves_raw_parts_and_negative_controls()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "GeminiNodeV2",
            "GeminiImageNode",
            "GeminiImage2Node",
            "GeminiNanoBanana2",
            "GeminiNanoBanana2V2",
            "GeminiVideoOmni",
        ] {
            let case = fixture_case(node)?;
            for raw in ["\u{1c} keep original ☃ \u{1f}", "\u{200b}", "\u{feff}"] {
                let mut input: Input = serde_json::from_value(case["input"].clone())?;
                if node == "GeminiVideoOmni" {
                    input.arguments.get_mut("model").ok_or("missing model")?["prompt"] = json!(raw);
                } else {
                    input.arguments.insert("prompt".into(), json!(raw));
                }
                let body = request(&input, &fixture_resolution(&case)?)?.body;
                let parts = body["contents"][0]["parts"]
                    .as_array()
                    .ok_or("missing parts")?;
                assert!(
                    parts
                        .iter()
                        .any(|part| part.get("text") == Some(&json!(raw)))
                );
            }
        }
        Ok(())
    }

    #[test]
    fn gemini_prompt_strip_keeps_legacy_admission_and_checked_cost_boundary()
    -> Result<(), Box<dyn Error>> {
        let case = fixture_case("GeminiNode")?;
        let mut input: Input = serde_json::from_value(case["input"].clone())?;
        input
            .arguments
            .insert("prompt".into(), json!("\u{1c} \u{1f}"));
        let legacy = request(&input, &fixture_resolution(&case)?)?;
        assert_eq!(
            legacy.body["contents"][0]["parts"][0]["text"],
            "\u{1c} \u{1f}"
        );
        for raw in ["\u{1c} keep original \u{1f}", "\u{200b}", "\u{feff}"] {
            let mut input = fixture_input("GeminiNodeV2")?;
            input.arguments.insert("prompt".into(), json!(raw));
            let mut transport = Transport {
                deny_cost: true,
                ..Transport::default()
            };
            let mut owner = InputOwner::default();
            assert!(matches!(
                invoke(&input, credentials(), 500, &mut transport, &mut owner),
                Err(GeminiError::Transport(ConformanceError::InvalidCost))
            ));
            assert_eq!(transport.heads.len(), 1);
            assert_eq!(transport.costs.len(), 1);
            assert_eq!(transport.bodies, vec![Vec::<u8>::new()]);
            assert!(owner.operations.is_empty());
        }
        Ok(())
    }
}
