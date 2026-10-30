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
pub const NAMESPACE: &str = "zed.sim.provider.luma";
pub const ALIASES: &[&str] = &["luma_2"];
pub const SOURCE_SHA256: &str = "6faf168ba42691705caef3e03f4894cc8e4b95dfac63ac9fedbe5f08fe083a36";
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
        feature_id: "COMFY-NODE-0376",
        node_identifier: "LumaConceptsNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0377",
        node_identifier: "LumaImageEditNode2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0378",
        node_identifier: "LumaImageModifyNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0379",
        node_identifier: "LumaImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0380",
        node_identifier: "LumaImageNode2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0381",
        node_identifier: "LumaImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0382",
        node_identifier: "LumaRay32ExtendVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0383",
        node_identifier: "LumaRay32ImageToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0384",
        node_identifier: "LumaRay32KeyframeNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0385",
        node_identifier: "LumaRay32KeyframesToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0386",
        node_identifier: "LumaRay32TextToVideoNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0387",
        node_identifier: "LumaRay32VideoEditNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0388",
        node_identifier: "LumaRay32VideoReframeNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0389",
        node_identifier: "LumaReferenceNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0390",
        node_identifier: "LumaVideoNode",
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
        feature_id: "COMFY-API-EXT-0092",
        method: "POST",
        path: "/proxy/luma/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0093",
        method: "GET",
        path: "/proxy/luma/generations/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0094",
        method: "POST",
        path: "/proxy/luma/generations/image",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0095",
        method: "GET",
        path: "/proxy/luma/generations/{response_api.id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0096",
        method: "POST",
        path: "/proxy/luma_2/generations",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0097",
        method: "GET",
        path: "/proxy/luma_2/generations/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0098",
        method: "GET",
        path: "/proxy/luma_2/generations/{initial.id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum LumaError {
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
impl fmt::Display for LumaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Luma HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for LumaError {}
impl From<ConformanceError> for LumaError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for LumaError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, LumaError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/luma/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, LumaError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(LumaError::Input("Unknown Luma node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(LumaError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(LumaError::Input("Unknown Luma source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(LumaError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(LumaError::Input("Missing Luma source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, LumaError> {
    input
        .arguments
        .get(name)
        .ok_or(LumaError::Input("Missing Luma argument"))
}
fn text(value: &Value) -> Result<&str, LumaError> {
    value
        .as_str()
        .ok_or(LumaError::Input("Expected Luma string"))
}
fn endpoint(value: &str) -> Result<bool, LumaError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(LumaError::Response("Invalid Luma response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(LumaError::Response("Invalid Luma response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(LumaError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(LumaError::Response("Invalid Luma endpoint authority"));
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
) -> Result<RequestHead, LumaError> {
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
            return Err(LumaError::Input("Missing Comfy API credentials"));
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum OutputKind {
    Image,
    Video,
    VideoId,
    Concepts,
    References,
    Keyframes,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VideoPreparation {
    Mp4H264,
}
#[derive(Clone, Debug, PartialEq)]
pub struct UploadPolicy {
    pub operation: &'static str,
    pub mime_type: Option<&'static str>,
    pub maximum_images: u64,
    pub total_pixels: u64,
    pub wait_label: &'static str,
    pub show_batch_index: bool,
    pub maximum_duration_seconds: Option<u64>,
    pub video_preparation: Option<VideoPreparation>,
}
impl UploadPolicy {
    fn images(maximum_images: u64) -> Self {
        Self {
            operation: "upload_images",
            mime_type: None,
            maximum_images,
            total_pixels: 4_194_304,
            wait_label: "Uploading",
            show_batch_index: true,
            maximum_duration_seconds: None,
            video_preparation: None,
        }
    }
    fn image(png: bool) -> Self {
        Self {
            operation: "upload_image",
            mime_type: png.then_some("image/png"),
            maximum_images: 1,
            total_pixels: 4_194_304,
            wait_label: "Uploading",
            show_batch_index: false,
            maximum_duration_seconds: None,
            video_preparation: None,
        }
    }
    fn video(maximum_duration_seconds: u64) -> Self {
        Self {
            operation: "upload_video",
            mime_type: Some("video/mp4"),
            maximum_images: 1,
            total_pixels: 0,
            wait_label: "Uploading",
            show_batch_index: false,
            maximum_duration_seconds: Some(maximum_duration_seconds),
            video_preparation: Some(VideoPreparation::Mp4H264),
        }
    }
}
#[derive(Clone, Debug)]
pub struct UploadedMedia {
    pub urls: Vec<String>,
    pub receipt: Vec<u8>,
}
pub trait LumaMediaOwner {
    fn upload(
        &mut self,
        value: &Value,
        policy: &UploadPolicy,
    ) -> Result<UploadedMedia, ConformanceError>;
    fn ordered_references(&mut self, value: &Value) -> Result<Vec<Value>, ConformanceError>;
    fn video_duration(&mut self, value: &Value) -> Result<f64, ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: Option<&'static str>,
    pub poll_base: Option<&'static str>,
    pub body: Value,
    pub output_kind: OutputKind,
    pub estimated_seconds: Option<u64>,
    pub uploads: Vec<Vec<u8>>,
}
fn source_route(node: &str) -> Option<&'static str> {
    match node {
        "LumaConceptsNode" | "LumaReferenceNode" | "LumaRay32KeyframeNode" => None,
        "LumaImageNode" | "LumaImageModifyNode" => Some("/proxy/luma/generations/image"),
        "LumaVideoNode" | "LumaImageToVideoNode" => Some("/proxy/luma/generations"),
        "LumaImageNode2"
        | "LumaImageEditNode2"
        | "LumaRay32TextToVideoNode"
        | "LumaRay32ImageToVideoNode"
        | "LumaRay32KeyframesToVideoNode"
        | "LumaRay32VideoEditNode"
        | "LumaRay32VideoReframeNode"
        | "LumaRay32ExtendVideoNode" => Some("/proxy/luma_2/generations"),
        _ => None,
    }
}
fn pure_kind(node: &str) -> Option<OutputKind> {
    match node {
        "LumaConceptsNode" => Some(OutputKind::Concepts),
        "LumaReferenceNode" => Some(OutputKind::References),
        "LumaRay32KeyframeNode" => Some(OutputKind::Keyframes),
        _ => None,
    }
}
fn number(value: &Value) -> Result<f64, LumaError> {
    let value = if let Some(value) = value.as_f64() {
        value
    } else if let Some(value) = value.as_str() {
        value
            .parse()
            .map_err(|_| LumaError::Input("Expected source number"))?
    } else {
        return Err(LumaError::Input("Expected source number"));
    };
    if !value.is_finite() {
        return Err(LumaError::Input("Non-finite source number"));
    }
    Ok(value)
}
fn python_round_two(value: f64) -> Result<f64, LumaError> {
    format!("{value:.2}")
        .parse()
        .map_err(|_| LumaError::Input("Invalid rounded source number"))
}
fn entries(value: &Value, name: &str) -> Result<Vec<Value>, LumaError> {
    if value.is_null() {
        return Ok(vec![]);
    }
    value
        .get(name)
        .and_then(Value::as_array)
        .cloned()
        .ok_or(LumaError::Input("Invalid native source chain descriptor"))
}
fn source_trim(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn preflight(input: &Input) -> Result<(), LumaError> {
    let node = input.node.as_str();
    if pure_kind(node).is_some() || matches!(node, "LumaImageModifyNode" | "LumaImageToVideoNode") {
        return Ok(());
    }
    let prompt = text(argument(input, "prompt")?)?;
    let trim = !matches!(
        node,
        "LumaVideoNode" | "LumaRay32VideoReframeNode" | "LumaRay32ExtendVideoNode"
    );
    let length = if trim {
        source_trim(prompt).chars().count()
    } else {
        prompt.chars().count()
    };
    let legacy = matches!(node, "LumaImageNode" | "LumaVideoNode");
    if length < if legacy { 3 } else { 1 } || (!legacy && length > 6000) {
        return Err(LumaError::Input("Prompt outside source bounds"));
    }
    Ok(())
}
fn upload(
    value: &Value,
    policy: UploadPolicy,
    owner: &mut dyn LumaMediaOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Vec<String>, LumaError> {
    let result = owner.upload(value, &policy)?;
    if result.receipt.is_empty()
        || result.receipt.len() > MAXIMUM_RECEIPT_BYTES
        || result.urls.len() > policy.maximum_images as usize
    {
        return Err(LumaError::Response("Invalid canonical media upload result"));
    }
    for url in &result.urls {
        if endpoint(url)? {
            return Err(LumaError::Response(
                "Expected absolute canonical upload URL",
            ));
        }
    }
    receipts.push(result.receipt);
    Ok(result.urls)
}
fn upload_one(
    value: &Value,
    policy: UploadPolicy,
    owner: &mut dyn LumaMediaOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<String, LumaError> {
    upload(value, policy, owner, receipts)?
        .into_iter()
        .next()
        .ok_or(LumaError::Response("Canonical upload returned no URL"))
}
fn put_nonnull(body: &mut Value, name: &str, value: Value) {
    if !value.is_null() {
        body[name] = value
    }
}
fn get_field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, LumaError> {
    value
        .get(name)
        .ok_or(LumaError::Input("Missing source dynamic field"))
}
fn concept_api(value: &Value) -> Result<Option<Value>, LumaError> {
    let concepts = entries(value, "concepts")?
        .into_iter()
        .filter(|concept| concept != "None")
        .map(|concept| json!({"key":concept}))
        .collect::<Vec<_>>();
    Ok((!concepts.is_empty()).then(|| json!(concepts)))
}
fn pure_request(input: &Input, kind: OutputKind) -> Result<Value, LumaError> {
    match kind {
        OutputKind::Concepts => {
            let mut concepts = entries(argument(input, "luma_concepts")?, "concepts")?;
            for name in ["concept1", "concept2", "concept3", "concept4"] {
                let concept = argument(input, name)?;
                if concept != "None" {
                    concepts.push(concept.clone())
                }
            }
            Ok(json!({"concepts":concepts}))
        }
        OutputKind::References => {
            let mut refs = entries(argument(input, "luma_ref")?, "refs")?;
            let weight = argument(input, "weight")?;
            let weight = if weight.is_i64() || weight.is_u64() {
                weight.clone()
            } else {
                json!(python_round_two(number(weight)?)?)
            };
            refs.push(json!({"image":argument(input,"image")?,"weight":weight}));
            Ok(json!({"refs":refs}))
        }
        OutputKind::Keyframes => {
            let mut items = entries(argument(input, "keyframes")?, "items")?;
            let position = argument(input, "position")?;
            let seconds = get_field(position, "position")? == "Absolute time (seconds)";
            items.push(json!({"image":argument(input,"image")?,"mode":if seconds{"seconds"}else{"fraction"},"value":number(get_field(position,if seconds{"seconds"}else{"fraction"})?)?}));
            Ok(json!({"items":items}))
        }
        _ => Err(LumaError::Input("Invalid pure Luma output")),
    }
}
fn legacy_references(
    value: &Value,
    maximum: usize,
    owner: &mut dyn LumaMediaOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Option<Value>, LumaError> {
    let references = entries(value, "refs")?
        .into_iter()
        .take(maximum)
        .collect::<Vec<_>>();
    let mut urls = vec![];
    for reference in &references {
        urls.push(upload_one(
            get_field(reference, "image")?,
            UploadPolicy::images(1),
            owner,
            receipts,
        )?);
    }
    let mut result = vec![];
    for (reference, url) in references.into_iter().zip(urls) {
        let weight = number(get_field(&reference, "weight")?)?;
        if !(0.0..=1.0).contains(&weight) {
            return Err(LumaError::Input(
                "Source reference weight outside API bounds",
            ));
        }
        result.push(json!({"url":url,"weight":weight}));
    }
    Ok((!result.is_empty()).then(|| json!(result)))
}
fn modern_refs(
    value: Option<&Value>,
    maximum: usize,
    owner: &mut dyn LumaMediaOwner,
    receipts: &mut Vec<Vec<u8>>,
) -> Result<Option<Value>, LumaError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let mut result = vec![];
    for image in owner.ordered_references(value)? {
        result.push(json!({"url":upload_one(&image,UploadPolicy::image(false),owner,receipts)?}));
    }
    if result.len() > maximum {
        return Err(LumaError::Provider(format!(
            "Maximum {maximum} reference images are allowed."
        )));
    }
    Ok((!result.is_empty()).then(|| json!(result)))
}
fn legacy_api_validate(body: &Value, image: bool) -> Result<(), LumaError> {
    let model = text(get_field(body, "model")?)?;
    if !if image {
        ["photon-1", "photon-flash-1"].contains(&model)
    } else {
        ["ray-2", "ray-flash-2", "ray-1-6"].contains(&model)
    } {
        return Err(LumaError::Input("Invalid source model enum"));
    }
    for (name, allowed) in [
        (
            "aspect_ratio",
            &["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"][..],
        ),
        ("resolution", &["540p", "720p", "1080p", "4k"][..]),
        ("duration", &["5s", "9s"][..]),
    ] {
        if let Some(value) = body.get(name)
            && !allowed.contains(&text(value)?)
        {
            return Err(LumaError::Input("Invalid source API enum"));
        }
    }
    Ok(())
}
pub fn request(input: &Input, owner: &mut dyn LumaMediaOwner) -> Result<Request, LumaError> {
    let input = normalized(input)?;
    preflight(&input)?;
    let node = input.node.as_str();
    if let Some(output_kind) = pure_kind(node) {
        return Ok(Request {
            path: None,
            poll_base: None,
            body: pure_request(&input, output_kind)?,
            output_kind,
            estimated_seconds: None,
            uploads: vec![],
        });
    }
    let mut uploads = vec![];
    let mut body = json!({"prompt":argument(&input,"prompt")?});
    let mut estimated_seconds = None;
    let path = source_route(node).ok_or(LumaError::Input("Missing paid source route"))?;
    let output_kind = if matches!(
        node,
        "LumaImageNode" | "LumaImageModifyNode" | "LumaImageNode2" | "LumaImageEditNode2"
    ) {
        OutputKind::Image
    } else if node.starts_with("LumaRay32") {
        OutputKind::VideoId
    } else {
        OutputKind::Video
    };
    match node {
        "LumaImageNode" => {
            if let Some(references) =
                legacy_references(argument(&input, "image_luma_ref")?, 4, owner, &mut uploads)?
            {
                body["image_ref"] = references
            }
            let style = argument(&input, "style_image")?;
            if !style.is_null() {
                let url = upload_one(style, UploadPolicy::images(1), owner, &mut uploads)?;
                let weight = number(argument(&input, "style_image_weight")?)?;
                if !(0.0..=1.0).contains(&weight) {
                    return Err(LumaError::Input("Source style weight outside API bounds"));
                }
                body["style_ref"] = json!([{"url":url,"weight":weight}])
            }
            let character = argument(&input, "character_image")?;
            if !character.is_null() {
                body["character_ref"] = json!({"identity0":{"images":upload(character,UploadPolicy::images(4),owner,&mut uploads)?}})
            }
            body["model"] = argument(&input, "model")?.clone();
            body["aspect_ratio"] = argument(&input, "aspect_ratio")?.clone();
            legacy_api_validate(&body, true)?;
        }
        "LumaImageModifyNode" => {
            let url = upload_one(
                argument(&input, "image")?,
                UploadPolicy::images(1),
                owner,
                &mut uploads,
            )?;
            body["model"] = argument(&input, "model")?.clone();
            body["aspect_ratio"] = json!("16:9");
            body["modify_image_ref"] = json!({"url":url,"weight":python_round_two((1.0-number(argument(&input,"image_weight")?)?).clamp(0.0,0.98))?});
            legacy_api_validate(&body, true)?;
        }
        "LumaVideoNode" | "LumaImageToVideoNode" => {
            if node == "LumaImageToVideoNode" {
                let first = argument(&input, "first_image")?;
                let last = argument(&input, "last_image")?;
                if first.is_null() && last.is_null() {
                    return Err(LumaError::Input(
                        "At least one of first_image and last_image requires an input.",
                    ));
                }
                let mut frames = json!({});
                for (name, value) in [("frame0", first), ("frame1", last)] {
                    if !value.is_null() {
                        frames[name] = json!({"type":"image","url":upload_one(value,UploadPolicy::images(1),owner,&mut uploads)?})
                    }
                }
                body["keyframes"] = frames;
                body["aspect_ratio"] = json!("16:9");
                estimated_seconds = Some(100);
            } else {
                body["aspect_ratio"] = argument(&input, "aspect_ratio")?.clone();
                estimated_seconds = Some(105);
            }
            let model = argument(&input, "model")?;
            body["model"] = model.clone();
            if model != "ray-1-6" {
                body["duration"] = argument(&input, "duration")?.clone();
                body["resolution"] = argument(&input, "resolution")?.clone();
            }
            body["loop"] = argument(&input, "loop")?.clone();
            if let Some(concepts) = concept_api(argument(&input, "luma_concepts")?)? {
                body["concepts"] = concepts
            }
            legacy_api_validate(&body, false)?;
        }
        "LumaImageNode2" | "LumaImageEditNode2" => {
            let dynamic = argument(&input, "model")?;
            let style = get_field(dynamic, "style")?;
            if node == "LumaImageNode2" {
                let ratio = get_field(dynamic, "aspect_ratio")?;
                if style == "manga"
                    && ratio != "auto"
                    && !["2:3", "9:16", "1:2", "1:3"].contains(&text(ratio)?)
                {
                    return Err(LumaError::Input(
                        "Manga style requires a portrait aspect ratio or auto",
                    ));
                }
                if ratio != "auto" {
                    body["aspect_ratio"] = ratio.clone();
                }
                body["type"] = json!("image");
            } else {
                body["source"] = json!({"url":upload_one(argument(&input,"source")?,UploadPolicy::image(false),owner,&mut uploads)?});
                body["type"] = json!("image_edit");
            }
            body["model"] = get_field(dynamic, "model")?.clone();
            if style != "auto" {
                body["style"] = style.clone();
            }
            body["output_format"] = json!("png");
            put_nonnull(
                &mut body,
                "web_search",
                get_field(dynamic, "web_search")?.clone(),
            );
            if let Some(refs) = modern_refs(
                dynamic.get("image_ref"),
                if node == "LumaImageNode2" { 9 } else { 8 },
                owner,
                &mut uploads,
            )? {
                body["image_ref"] = refs;
            }
        }
        "LumaRay32TextToVideoNode" => {
            let duration = argument(&input, "duration")?;
            let looping = argument(&input, "loop")?
                .as_bool()
                .ok_or(LumaError::Input("Invalid source loop"))?;
            if looping && duration == "10s" {
                return Err(LumaError::Input(
                    "Looping is only available with 5s duration on Ray 3.2.",
                ));
            }
            body["aspect_ratio"] = argument(&input, "aspect_ratio")?.clone();
            body["video"] =
                json!({"resolution":argument(&input,"resolution")?,"duration":duration});
            if looping {
                body["video"]["loop"] = json!(true)
            }
        }
        "LumaRay32ImageToVideoNode" => {
            let start = argument(&input, "start_frame")?;
            let end = argument(&input, "end_frame")?;
            if start.is_null() && end.is_null() {
                return Err(LumaError::Input(
                    "Provide at least one of start_frame / end_frame.",
                ));
            }
            let looping = argument(&input, "loop")?
                .as_bool()
                .ok_or(LumaError::Input("Invalid source loop"))?;
            if looping && !end.is_null() {
                return Err(LumaError::Input(
                    "Looping is not available when an end_frame is set.",
                ));
            }
            let mut video = json!({"resolution":argument(&input,"resolution")?,"duration":"5s"});
            if looping {
                video["loop"] = json!(true)
            }
            for (name, value) in [("start_frame", start), ("end_frame", end)] {
                if !value.is_null() {
                    video[name] = json!({"url":upload_one(value,UploadPolicy::image(true),owner,&mut uploads)?});
                }
            }
            body["video"] = video;
        }
        "LumaRay32KeyframesToVideoNode" => {
            let items = entries(argument(&input, "keyframes")?, "items")?;
            if !(2..=64).contains(&items.len()) {
                return Err(LumaError::Input("Ray 3.2 requires 2–64 keyframes"));
            }
            let duration = argument(&input, "duration")?;
            let maxframe = if duration == "5s" { 120 } else { 240 };
            let mut placed = vec![];
            for item in items {
                let value = number(get_field(&item, "value")?)?;
                let seconds = get_field(&item, "mode")? == "seconds";
                if seconds && value > maxframe as f64 / 24.0 {
                    return Err(LumaError::Input("Keyframe position past source video end"));
                }
                let frame = (value * if seconds { 24.0 } else { maxframe as f64 })
                    .round_ties_even()
                    .clamp(0.0, maxframe as f64) as i64;
                placed.push((frame, get_field(&item, "image")?.clone()));
            }
            placed.sort_by_key(|(frame, _)| *frame);
            if placed.windows(2).any(|pair| {
                pair.first()
                    .zip(pair.get(1))
                    .is_some_and(|(first, second)| first.0 == second.0)
            }) {
                return Err(LumaError::Input(
                    "Two keyframes resolve to the same output frame",
                ));
            }
            let mut refs = vec![];
            let mut indexes = vec![];
            for (frame, image) in placed {
                refs.push(
                    json!({"url":upload_one(&image,UploadPolicy::image(true),owner,&mut uploads)?}),
                );
                indexes.push(frame)
            }
            body["video"] = json!({"resolution":argument(&input,"resolution")?,"duration":duration,"keyframes":refs,"keyframe_indexes":indexes});
        }
        "LumaRay32VideoEditNode" => {
            let video = argument(&input, "video")?;
            let duration = match owner.video_duration(video) {
                Ok(duration) if duration <= 5.0 => "5s",
                Ok(_) => "10s",
                // The pinned source intentionally falls back to 10s when duration inspection fails.
                Err(_) => "10s",
            };
            let url = upload_one(video, UploadPolicy::video(18), owner, &mut uploads)?;
            body["source"] = json!({"url":url,"media_type":"video/mp4"});
            let strength = argument(&input, "strength")?;
            body["type"] = json!("video_edit");
            body["video"] = json!({"resolution":argument(&input,"resolution")?,"duration":duration,"edit":if strength=="auto"{json!({"auto_controls":true})}else{json!({"strength":strength})}});
        }
        "LumaRay32VideoReframeNode" => {
            let resolution = argument(&input, "resolution")?;
            let ratio = argument(&input, "aspect_ratio")?;
            if resolution == "1080p" && ["9:16", "3:4"].contains(&text(ratio)?) {
                return Err(LumaError::Input(
                    "1080p is not available for vertical aspect ratios when reframing.",
                ));
            }
            body["source"] = json!({"url":upload_one(argument(&input,"video")?,UploadPolicy::video(30),owner,&mut uploads)?,"media_type":"video/mp4"});
            body["type"] = json!("video_reframe");
            body["aspect_ratio"] = ratio.clone();
            body["video"] = json!({"resolution":resolution});
        }
        "LumaRay32ExtendVideoNode" => {
            let source = argument(&input, "source_generation_id")?;
            let identifier = source_trim(if source.is_null() { "" } else { text(source)? });
            if identifier.is_empty() {
                return Err(LumaError::Input("source_generation_id is required"));
            }
            let direction = argument(&input, "direction")?;
            let mut video = json!({"resolution":argument(&input,"resolution")?,"duration":"5s"});
            let forward = get_field(direction, "direction")? == "Forward (continue after)";
            video[if forward { "start_frame" } else { "end_frame" }] =
                json!({"generation_id":identifier});
            if forward && direction.get("loop").and_then(Value::as_bool) == Some(true) {
                video["loop"] = json!(true)
            }
            body["video"] = video;
        }
        _ => return Err(LumaError::Input("Unknown Luma node")),
    }
    if node.starts_with("LumaRay32") {
        body["model"] = json!("ray-3.2");
        if body.get("type").is_none() {
            body["type"] = json!("video")
        }
        estimated_seconds = Some(120)
    }
    if path.starts_with("/proxy/luma_2") {
        let length = text(get_field(&body, "prompt")?)?.chars().count();
        if !(1..=6000).contains(&length) {
            return Err(LumaError::Input("Prompt outside source API bounds"));
        }
    }
    Ok(Request {
        path: Some(path),
        poll_base: Some(if path.starts_with("/proxy/luma_2") {
            "/proxy/luma_2/generations"
        } else {
            "/proxy/luma/generations"
        }),
        body,
        output_kind,
        estimated_seconds,
        uploads,
    })
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
    pub estimated_seconds: Option<u64>,
    pub progress_extractor_none: bool,
    pub timeout_milliseconds: u64,
    pub maximum_active_attempts: u32,
    pub interval_milliseconds: u64,
}
pub trait LumaTransport: ProviderTransport {
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
    pub custom_chain: Option<Value>,
    pub color_mode: &'static str,
}
fn generation_response(bytes: &[u8], modern: bool) -> Result<Value, LumaError> {
    let value: Value = serde_json::from_slice(bytes)?;
    if !value.is_object() {
        return Err(LumaError::Response("Invalid source generation response"));
    }
    if modern {
        for name in [
            "id",
            "type",
            "state",
            "model",
            "created_at",
            "failure_reason",
            "failure_code",
        ] {
            if value
                .get(name)
                .is_some_and(|field| !field.is_null() && !field.is_string())
            {
                return Err(LumaError::Response("Invalid modern source response string"));
            }
        }
        if let Some(output) = value.get("output").filter(|field| !field.is_null()) {
            let output = output
                .as_array()
                .ok_or(LumaError::Response("Invalid source output list"))?;
            for item in output {
                if !item["url"].is_string() {
                    return Err(LumaError::Response("Invalid source output URL"));
                }
            }
        }
    } else {
        for name in ["id", "generation_type", "state", "created_at", "model"] {
            if !value[name].is_string() {
                return Err(LumaError::Response("Missing legacy source response field"));
            }
        }
        if !["queued", "dreaming", "completed", "failed"].contains(&text(&value["state"])?) {
            return Err(LumaError::Response("Invalid legacy source state"));
        }
        if !["image", "video"].contains(&text(&value["generation_type"])?) {
            return Err(LumaError::Response("Invalid legacy source generation type"));
        }
        if !value["request"].is_object() || !value["request"]["prompt"].is_string() {
            return Err(LumaError::Response("Missing legacy source request"));
        }
    }
    Ok(value)
}
fn generation_identifier(response: &Value, modern: bool) -> Result<&str, LumaError> {
    let id = response["id"].as_str().ok_or(LumaError::Response(
        "Luma API did not return a generation id.",
    ))?;
    if modern && id.is_empty() {
        return Err(LumaError::Response(
            "Luma API did not return a generation id.",
        ));
    }
    if id.len() > 256
        || id
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
        || id.contains(['/', '?', '#', '\\'])
    {
        return Err(LumaError::Response("Unsafe source generation identifier"));
    }
    Ok(id)
}
fn completed_output(
    response: &Value,
    kind: OutputKind,
    modern: bool,
) -> Result<(String, Vec<String>), LumaError> {
    let url = if modern {
        response["output"]
            .as_array()
            .and_then(|output| output.first())
            .and_then(|item| item["url"].as_str())
            .filter(|url| !url.is_empty())
            .ok_or_else(|| {
                let mut reason = response["failure_reason"]
                    .as_str()
                    .filter(|reason| !reason.is_empty())
                    .unwrap_or("no output returned")
                    .to_owned();
                if let Some(code) = response["failure_code"]
                    .as_str()
                    .filter(|code| !code.is_empty())
                {
                    reason.push_str(&format!(" [{code}]"));
                }
                LumaError::Provider(format!("Luma generation failed: {reason}"))
            })?
    } else {
        response["assets"][if kind == OutputKind::Image {
            "image"
        } else {
            "video"
        }]
        .as_str()
        .ok_or(LumaError::Response("Missing legacy source media URL"))?
    };
    if endpoint(url)? {
        return Err(LumaError::Response(
            "Source download requires an absolute URL without a node context",
        ));
    }
    let strings = if kind == OutputKind::VideoId {
        vec![response["id"].as_str().unwrap_or("").to_owned()]
    } else {
        vec![]
    };
    Ok((url.into(), strings))
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn LumaTransport,
    owner: &mut dyn LumaMediaOwner,
) -> Result<ResultValue, LumaError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    let Some(path) = source_route(&input.node) else {
        let plan = request(&input, owner)?;
        let custom_chain = Some(plan.body.clone());
        return Ok(ResultValue {
            plan,
            creation_bytes: vec![],
            creation_receipt: vec![],
            response_bytes: vec![],
            receipt: vec![],
            approval_receipt: vec![],
            media: vec![],
            strings: vec![],
            custom_chain,
            color_mode: "RGBA",
        });
    };
    preflight(&input)?;
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "luma.generation.maximum".into(),
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
    let modern = path.starts_with("/proxy/luma_2");
    let creation = generation_response(&creation_bytes, modern)?;
    let identifier = generation_identifier(&creation, modern)?;
    let poll_path = format!(
        "{}/{identifier}",
        plan.poll_base
            .ok_or(LumaError::Response("Missing poll base"))?
    );
    let source = oracle()?;
    let mut active = 0u32;
    for sequence in 0..4096 {
        let head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
        transport.check_cancelled(None)?;
        let handle = transport.start_request(&head)?;
        transport.configure_poll(
            handle,
            &PollPolicy {
                estimated_seconds: plan.estimated_seconds,
                progress_extractor_none: modern,
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
        let response = generation_response(&response_bytes, modern)?;
        let status = response["state"]
            .as_str()
            .map(|state| source_trim(state).to_lowercase());
        let member = |name: &str| {
            source["polling"][name].as_array().is_some_and(|values| {
                status
                    .as_deref()
                    .is_some_and(|status| values.iter().any(|value| value.as_str() == Some(status)))
            })
        };
        if member("FAILED_STATUSES") {
            return Err(LumaError::Provider(format!(
                "Task failed: {}",
                serde_json::to_string(&response)?
            )));
        }
        if member("COMPLETED_STATUSES") {
            let (url, strings) = completed_output(&response, plan.output_kind, modern)?;
            let head = request_head(&url, HttpMethod::Get, credentials, true)?;
            let (bytes, download_receipt, _) = exchange_with_policy(
                &head,
                &[],
                None,
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                transport,
            )?;
            if bytes.is_empty() {
                return Err(LumaError::Response("Empty canonical media download"));
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
                media: vec![DownloadedMedia {
                    url,
                    bytes,
                    receipt: download_receipt,
                }],
                strings,
                custom_chain: None,
                color_mode: "RGBA",
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
pub trait LumaMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
fn expected_outputs(kind: OutputKind) -> Result<Vec<ExpectedOutput>, LumaError> {
    let registry =
        TypeRegistry::built_in().map_err(|_| LumaError::Response("Missing canonical registry"))?;
    let mut types = vec![match kind {
        OutputKind::Image => "IMAGE",
        OutputKind::Video | OutputKind::VideoId => "VIDEO",
        OutputKind::Concepts => "LumaIO.LUMA_CONCEPTS",
        OutputKind::References => "LumaIO.LUMA_REF",
        OutputKind::Keyframes => "LumaIO.LUMA_RAY32_KEYFRAME",
    }];
    if kind == OutputKind::VideoId {
        types.push("STRING")
    }
    let mut expected = vec![];
    for (index, name) in types.into_iter().enumerate() {
        let type_id = registry
            .resolve(name)
            .map_err(|_| LumaError::Response("Missing canonical output type"))?
            .clone();
        let family = registry
            .family(&type_id)
            .map_err(|_| LumaError::Response("Missing canonical output family"))?;
        expected.push(ExpectedOutput {
            port_id: format!("output_{index}"),
            type_id,
            family,
        });
    }
    Ok(expected)
}
pub struct MaterializationOwners<'a> {
    pub media: &'a mut dyn LumaMediaOwner,
    pub materializer: &'a mut dyn LumaMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn LumaTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), LumaError> {
    let result = invoke(
        input,
        credentials,
        maximum_microunits,
        transport,
        owners.media,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.materializer.materialize(&result)?;
    let expected = expected_outputs(result.plan.output_kind)?;
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
    transport: &mut dyn LumaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), LumaError> {
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
    transport: &mut dyn LumaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), LumaError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        let mut response_policy = ResponsePolicy::source();
        if policy.download {
            response_policy.timeout_milliseconds = None;
            response_policy.monitor_progress = false;
            response_policy.final_label = None;
        } else if policy.ordinary_limit == 10 {
            response_policy.timeout_milliseconds = Some(120_000);
        }
        transport.configure_response(handle, &response_policy)?;
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(LumaError::Http {
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
                    return Err(LumaError::Http {
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
    transport: &mut dyn LumaTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), LumaError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "luma.generation.maximum".into(),
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
                    return Err(LumaError::Http {
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
    cancel_after: Option<usize>,
}
impl LumaMediaOwner for SourceProbeOwner {
    fn upload(
        &mut self,
        value: &Value,
        policy: &UploadPolicy,
    ) -> Result<UploadedMedia, ConformanceError> {
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        if self
            .cancel_after
            .is_some_and(|count| self.operations.len() >= count)
        {
            return Err(ConformanceError::Cancelled);
        }
        let urls = if policy.operation == "upload_video" {
            if policy.video_preparation != Some(VideoPreparation::Mp4H264) {
                return Err(ConformanceError::InvalidRequest);
            }
            self.operations.push(json!({"operation":"upload_video","video":value,"max_duration":policy.maximum_duration_seconds,"mime_type":"video/mp4"}));
            vec!["https://storage.fixture.invalid/video.mp4".into()]
        } else {
            let slot = value["native_slot"]
                .as_str()
                .ok_or(ConformanceError::InvalidRequest)?;
            if policy.operation == "upload_images" {
                self.operations.push(json!({"operation":"upload_images","image":value,"max_images":policy.maximum_images,"mime_type":policy.mime_type,"total_pixels":policy.total_pixels,"wait_label":policy.wait_label,"show_batch_index":policy.show_batch_index}));
                let count = value["shape"]
                    .as_array()
                    .and_then(|shape| shape.first())
                    .and_then(Value::as_u64)
                    .ok_or(ConformanceError::InvalidRequest)?;
                (0..count.min(policy.maximum_images))
                    .map(|index| format!("https://storage.fixture.invalid/{slot}-{index}.png"))
                    .collect()
            } else {
                self.operations.push(json!({"operation":"upload_image","image":value,"mime_type":policy.mime_type,"total_pixels":policy.total_pixels,"wait_label":policy.wait_label}));
                vec![format!("https://storage.fixture.invalid/{slot}-0.png")]
            }
        };
        Ok(UploadedMedia {
            urls,
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
    fn video_duration(&mut self, value: &Value) -> Result<f64, ConformanceError> {
        if value["duration_error"] == true {
            return Err(ConformanceError::HostFailure);
        }
        value["duration"]
            .as_f64()
            .ok_or(ConformanceError::InvalidRequest)
    }
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"scope":"source request and bounded transport; canonical native media and guest integration pending","source_generation_id_admission":"python_strip_before_direction_lookup_preserving_prompt_and_interior_bytes"})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
    if source["source_sha256"]["comfy_api_nodes/nodes_luma.py"] != SOURCE_SHA256 {
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
        let mut owner = SourceProbeOwner::default();
        let plan = request(&input, &mut owner).map_err(|_| ConformanceError::InvalidRequest)?;
        let expected = case["requests"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if let Some(expected) = expected.first() {
            if plan.path != expected["path"].as_str()
                || plan.body != expected["body"]
                || json!(plan.estimated_seconds) != case["polls"][0]["estimated_duration"]
            {
                return Err(ConformanceError::InvalidRequest);
            }
            if let Some(path) = plan.path {
                paths.insert(path);
            }
        } else if plan.path.is_some() || case["outputs"][0] != plan.body {
            return Err(ConformanceError::InvalidRequest);
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
        || paths
            != ROUTES
                .iter()
                .filter(|route| route.method == "POST")
                .map(|route| route.path)
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
    impl LumaTransport for Transport {
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
    fn completed(node: &str) -> Result<Transport, Box<dyn Error>> {
        let mut transport = Transport::default();
        if matches!(
            node,
            "LumaImageNode" | "LumaImageModifyNode" | "LumaVideoNode" | "LumaImageToVideoNode"
        ) {
            let initial = json!({"id":"source-id","generation_type":"image","state":"queued","created_at":"source-date","model":"photon-1","request":{"prompt":"source"}});
            append(&mut transport, 200, initial.clone());
            let mut response = initial;
            response["state"] = json!("completed");
            response["assets"] = json!({"image":"https://media.fixture.invalid/image.png","video":"https://media.fixture.invalid/video.mp4"});
            append(&mut transport, 200, response);
        } else {
            append(
                &mut transport,
                200,
                json!({"id":"source-id","state":"queued"}),
            );
            append(
                &mut transport,
                200,
                json!({"id":"final-id","state":"completed","output":[{"url":"https://media.fixture.invalid/result"}]}),
            );
        }
        transport.events.extend(events(200, png(), vec![]));
        Ok(transport)
    }
    fn png() -> Vec<u8> {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/luma/downloaded-image.png"
        ))
        .to_vec()
    }
    #[test]
    fn luma_source_generation_id_strip_preserves_request_bytes_and_direction()
    -> Result<(), Box<dyn Error>> {
        let node = "LumaRay32ExtendVideoNode";
        let prompt = "  preserved prompt  ";
        for (source, expected) in [
            ("id", "id"),
            ("\u{1c}id\u{1f}", "id"),
            (" \u{1c}id \u{1f}\t", "id"),
            ("i\u{1c}d", "i\u{1c}d"),
            ("\u{200b}id\u{200b}", "\u{200b}id\u{200b}"),
            ("\u{feff}id\u{feff}", "\u{feff}id\u{feff}"),
            ("\u{85}id\u{a0}", "id"),
            ("\u{2003}id\u{202f}", "id"),
        ] {
            for forward in [true, false] {
                let mut source_input = input(node)?;
                source_input
                    .arguments
                    .insert("prompt".into(), json!(prompt));
                source_input
                    .arguments
                    .insert("source_generation_id".into(), json!(source));
                source_input.arguments.insert(
                    "direction".into(),
                    if forward {
                        json!({"direction":"Forward (continue after)","loop":true})
                    } else {
                        json!({"direction":"Backward (lead-in before)"})
                    },
                );
                let mut owner = SourceProbeOwner::default();
                let plan = request(&source_input, &mut owner)?;
                let frame = if forward { "start_frame" } else { "end_frame" };
                let other_frame = if forward { "end_frame" } else { "start_frame" };
                assert_eq!(plan.path, Some("/proxy/luma_2/generations"));
                assert_eq!(plan.body["video"][frame]["generation_id"], expected);
                assert!(plan.body["video"].get(other_frame).is_none());
                if forward {
                    assert_eq!(plan.body["video"]["loop"], true);
                } else {
                    assert!(plan.body["video"].get("loop").is_none());
                }
                assert_eq!(plan.body["prompt"], prompt);
                assert!(owner.operations.is_empty());

                let mut transport = Transport::default();
                append(
                    &mut transport,
                    200,
                    json!({"id":"source-id","state":"queued"}),
                );
                append(
                    &mut transport,
                    200,
                    json!({"id":"final-id","state":"completed","output":[{"url":"https://media.fixture.invalid/result"}]}),
                );
                let video = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/luma/downloaded-video.mp4"
                ))
                .to_vec();
                transport.events.extend(events(200, video.clone(), vec![]));
                let result = invoke(
                    &source_input,
                    credentials(),
                    123,
                    &mut transport,
                    &mut owner,
                )?;
                let body = transport.bodies.first().ok_or("missing generation body")?;
                assert_eq!(serde_json::from_slice::<Value>(body)?, plan.body);
                assert_eq!(transport.heads.len(), 3);
                assert_eq!(transport.costs.len(), 1);
                assert_eq!(
                    transport.costs.first().ok_or("missing cost")?.operation,
                    "luma.generation.maximum"
                );
                assert_eq!(result.media.len(), 1);
                assert_eq!(
                    result
                        .media
                        .first()
                        .ok_or("missing downloaded media")?
                        .bytes,
                    video
                );
                assert_eq!(result.plan.body, plan.body);
                assert!(owner.operations.is_empty());
                assert!(transport.events.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn luma_source_generation_id_empty_checks_precede_direction_and_body()
    -> Result<(), Box<dyn Error>> {
        let node = "LumaRay32ExtendVideoNode";
        for identifier in [
            Value::Null,
            json!(""),
            json!(" \t\n"),
            json!("\u{1c}"),
            json!("\u{1d}"),
            json!("\u{1e}"),
            json!("\u{1f}"),
            json!("\u{1d}\u{1f}\u{1e}\u{1c}"),
        ] {
            let mut source_input = input(node)?;
            source_input
                .arguments
                .insert("source_generation_id".into(), identifier);
            source_input.arguments.insert("direction".into(), json!({}));
            let mut owner = SourceProbeOwner::default();
            assert!(matches!(
                request(&source_input, &mut owner),
                Err(LumaError::Input("source_generation_id is required"))
            ));
            let mut transport = Transport::default();
            assert!(matches!(
                invoke(
                    &source_input,
                    credentials(),
                    123,
                    &mut transport,
                    &mut owner
                ),
                Err(LumaError::Input("source_generation_id is required"))
            ));
            assert_eq!(transport.heads.len(), 1);
            assert_eq!(transport.costs.len(), 1);
            assert!(transport.bodies.iter().all(Vec::is_empty));
            assert!(owner.operations.is_empty());

            source_input.arguments.insert("prompt".into(), json!(""));
            let mut transport = Transport::default();
            assert!(matches!(
                invoke(
                    &source_input,
                    credentials(),
                    123,
                    &mut transport,
                    &mut owner
                ),
                Err(LumaError::Input("Prompt outside source bounds"))
            ));
            assert!(transport.heads.is_empty());
            assert!(transport.costs.is_empty());
            assert!(owner.operations.is_empty());
        }
        let mut source_input = input(node)?;
        source_input
            .arguments
            .insert("source_generation_id".into(), json!("id"));
        source_input.arguments.insert("direction".into(), json!({}));
        let mut owner = SourceProbeOwner::default();
        assert!(matches!(
            request(&source_input, &mut owner),
            Err(LumaError::Input("Missing source dynamic field"))
        ));
        assert!(owner.operations.is_empty());
        Ok(())
    }

    #[test]
    fn luma_source_claims_all_request_branches_and_custom_chains_match()
    -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        for (index, case) in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .enumerate()
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            let plan = request(&input, &mut owner)?;
            let expected = case["requests"].as_array().ok_or("requests")?.first();
            if let Some(expected) = expected {
                assert_eq!(plan.body, expected["body"], "case {index} {}", input.node);
                assert_eq!(plan.path, expected["path"].as_str());
                assert_eq!(
                    json!(plan.estimated_seconds),
                    case["polls"][0]["estimated_duration"]
                );
            } else {
                assert_eq!(plan.body, case["outputs"][0], "case {index} {}", input.node);
                assert!(plan.path.is_none());
            }
            assert_eq!(
                json!(owner.operations),
                case["media"],
                "case {index} {}",
                input.node
            );
        }
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/luma/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn luma_source_preflight_failure_upload_order_matches() -> Result<(), Box<dyn Error>> {
        for (index, case) in oracle()?["preflight_errors"]
            .as_array()
            .ok_or("errors")?
            .iter()
            .enumerate()
        {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut owner = SourceProbeOwner::default();
            assert!(
                request(&input, &mut owner).is_err(),
                "case {index} {}",
                input.node
            );
            assert_eq!(
                json!(owner.operations),
                case["media"],
                "case {index} {}",
                input.node
            );
        }
        Ok(())
    }
    #[test]
    fn luma_pure_nodes_have_no_credentials_cost_upload_or_http() -> Result<(), Box<dyn Error>> {
        for node in [
            "LumaConceptsNode",
            "LumaReferenceNode",
            "LumaRay32KeyframeNode",
        ] {
            let mut transport = Transport::default();
            let mut owner = SourceProbeOwner::default();
            let result = invoke(
                &input(node)?,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false,
                },
                0,
                &mut transport,
                &mut owner,
            )?;
            assert!(result.custom_chain.is_some());
            assert!(transport.heads.is_empty());
            assert!(transport.costs.is_empty());
            assert!(owner.operations.is_empty());
            assert!(result.creation_receipt.is_empty());
        }
        Ok(())
    }
    #[test]
    fn luma_all_paid_nodes_use_exact_routes_auth_vendor_cost_and_rgba() -> Result<(), Box<dyn Error>>
    {
        for claim in NODES
            .iter()
            .filter(|claim| pure_kind(claim.node_identifier).is_none())
        {
            let mut transport = completed(claim.node_identifier)?;
            let result = invoke(
                &input(claim.node_identifier)?,
                credentials(),
                1_234_567,
                &mut transport,
                &mut SourceProbeOwner::default(),
            )?;
            assert_eq!(result.color_mode, "RGBA");
            assert_eq!(result.media.first().ok_or("media")?.bytes, png());
            assert_eq!(result.media.first().ok_or("media")?.receipt, vec![7, 8, 9]);
            assert_eq!(result.approval_receipt, vec![42]);
            assert_eq!(transport.costs.len(), 1);
            assert_eq!(
                transport.costs.first().ok_or("cost")?.operation,
                "luma.generation.maximum"
            );
            assert_eq!(
                transport.costs.first().ok_or("cost")?.maximum_microunits,
                1_234_567
            );
            assert!(
                transport
                    .heads
                    .first()
                    .ok_or("head")?
                    .headers
                    .iter()
                    .any(|header| header.name == "Authorization")
            );
            assert!(
                transport
                    .heads
                    .last()
                    .ok_or("download")?
                    .secret_id
                    .is_none()
            );
            assert_eq!(
                transport
                    .poll_policies
                    .first()
                    .ok_or("poll")?
                    .1
                    .estimated_seconds,
                result.plan.estimated_seconds
            );
            assert_eq!(
                result.strings,
                if result.plan.output_kind == OutputKind::VideoId {
                    vec!["final-id".to_owned()]
                } else {
                    vec![]
                }
            );
        }
        Ok(())
    }
    #[test]
    fn luma_credential_priority_cost_denial_and_upload_failure_stop_body()
    -> Result<(), Box<dyn Error>> {
        let source = input("LumaImageEditNode2")?;
        let mut owner = SourceProbeOwner::default();
        let mut transport = Transport::default();
        assert!(
            invoke(
                &source,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                9,
                &mut transport,
                &mut owner
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        transport.deny_cost = true;
        assert!(matches!(
            invoke(&source, credentials(), 9, &mut transport, &mut owner),
            Err(LumaError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(owner.operations.is_empty());
        assert!(transport.bodies.iter().all(Vec::is_empty));
        let mut owner = SourceProbeOwner {
            fail: true,
            ..SourceProbeOwner::default()
        };
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(
                &source,
                CredentialAvailability {
                    auth_token: false,
                    api_key: true
                },
                9,
                &mut transport,
                &mut owner
            ),
            Err(LumaError::Transport(ConformanceError::HostFailure))
        ));
        assert!(
            transport
                .heads
                .first()
                .ok_or("head")?
                .headers
                .iter()
                .any(|header| header.name == "X-API-KEY")
        );
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn luma_modern_source_response_errors_preserve_submit_poll_and_no_download()
    -> Result<(), Box<dyn Error>> {
        for case in oracle()?["response_errors"].as_array().ok_or("errors")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let mut transport = Transport::default();
            let mut initial = json!({"id":"source-id","state":"queued"});
            if let Some(overrides) = case["response_override"]["initial"].as_object() {
                for (key, value) in overrides {
                    initial[key] = value.clone();
                }
            }
            append(&mut transport, 200, initial);
            let mut final_response = json!({"id":"final-id","state":"completed","output":[{"url":"https://media.fixture.invalid/result"}]});
            if let Some(overrides) = case["response_override"]["final"].as_object() {
                for (key, value) in overrides {
                    final_response[key] = value.clone();
                }
            }
            append(&mut transport, 200, final_response);
            let error = invoke(
                &input,
                credentials(),
                10,
                &mut transport,
                &mut SourceProbeOwner::default(),
            )
            .err()
            .ok_or("expected error")?;
            assert_eq!(
                error.to_string(),
                case["message"].as_str().ok_or("message")?
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
    fn luma_queued_null_and_dreaming_polls_use_canonical_delay_and_clock_policy()
    -> Result<(), Box<dyn Error>> {
        let mut transport = Transport::default();
        append(&mut transport, 200, json!({"id":"source-id"}));
        for state in [json!("QUEUED"), Value::Null, json!("dreaming")] {
            append(&mut transport, 200, json!({"state":state}));
        }
        append(
            &mut transport,
            200,
            json!({"state":"completed","output":[{"url":"https://media.fixture.invalid/result"}]}),
        );
        transport.events.extend(events(200, png(), vec![]));
        invoke(
            &input("LumaRay32TextToVideoNode")?,
            credentials(),
            10,
            &mut transport,
            &mut SourceProbeOwner::default(),
        )?;
        assert_eq!(transport.delays, vec![5000; 3]);
        assert_eq!(transport.poll_policies.len(), 4);
        assert!(
            transport
                .poll_policies
                .iter()
                .all(|(_, policy)| policy.estimated_seconds == Some(120)
                    && policy.timeout_milliseconds == 120000
                    && policy.progress_extractor_none)
        );
        assert!(
            transport
                .policies
                .iter()
                .any(|(_, policy)| policy.timeout_milliseconds == Some(120000))
        );
        assert!(
            transport
                .policies
                .iter()
                .any(|(_, policy)| policy.timeout_milliseconds.is_none())
        );
        Ok(())
    }
    #[test]
    fn luma_poll_failure_active_limit_and_hard_queued_limit_are_bounded()
    -> Result<(), Box<dyn Error>> {
        for (state, count) in [("failed", 1), ("dreaming", 480), ("queued", 4096)] {
            let mut transport = Transport::default();
            append(&mut transport, 200, json!({"id":"source-id"}));
            for _ in 0..count {
                append(
                    &mut transport,
                    200,
                    json!({"state":state,"failure_reason":"source-failure"}),
                );
            }
            let error = invoke(
                &input("LumaRay32TextToVideoNode")?,
                credentials(),
                10,
                &mut transport,
                &mut SourceProbeOwner::default(),
            )
            .err()
            .ok_or("failure")?;
            if state == "failed" {
                assert!(error.to_string().contains("source-failure"))
            } else {
                assert!(matches!(
                    error,
                    LumaError::Transport(ConformanceError::PollLimit)
                ));
                assert_eq!(transport.delays.len(), count - 1);
            }
        }
        Ok(())
    }
    #[test]
    fn luma_retry_budgets_and_retry_after_delegate_to_fresh_handles() -> Result<(), Box<dyn Error>>
    {
        for (policy, status, retries) in [
            (GENERATION_RETRY, 500, 3),
            (GENERATION_RETRY, 429, 16),
            (POLL_RETRY, 503, 10),
            (DOWNLOAD_RETRY, 429, 5),
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
            let head = request_head(
                "/proxy/luma_2/generations",
                HttpMethod::Post,
                credentials(),
                false,
            )?;
            assert!(matches!(
                exchange_with_policy(&head, b"{}", None, 1024, policy, &mut transport),
                Err(LumaError::Http { .. })
            ));
            assert_eq!(transport.delays.len(), retries);
            assert_eq!(transport.heads.len(), retries + 1);
            if !policy.download {
                assert!(transport.delays.iter().all(|delay| *delay == 150000));
            } else {
                assert_eq!(transport.delays, vec![1000, 2000, 4000, 8000, 16000]);
            }
        }
        let mut transport = Transport {
            same_retry_handle: true,
            events: events(500, vec![], vec![]),
            ..Transport::default()
        };
        assert!(matches!(
            exchange_with_policy(
                &request_head(
                    "/proxy/luma/generations",
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
            Err(LumaError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn luma_transport_rejects_malformed_headers_chunks_receipts_and_worker_loss()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/luma/generations",
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
            Transport {
                events: vec![Err(ConformanceError::HostFailure)].into(),
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
            Err(LumaError::Transport(ConformanceError::BodyLimit))
        ));
        Ok(())
    }
    #[test]
    fn luma_restart_without_response_evidence_does_not_replay_paid_generation()
    -> Result<(), Box<dyn Error>> {
        let source = input("LumaRay32TextToVideoNode")?;
        for _ in 0..2 {
            let mut transport = Transport::default();
            assert!(matches!(
                invoke(
                    &source,
                    credentials(),
                    10,
                    &mut transport,
                    &mut SourceProbeOwner::default()
                ),
                Err(LumaError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(transport.heads.len(), 1);
            assert!(transport.delays.is_empty());
        }
        Ok(())
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
        seen: Vec<Vec<u8>>,
        kind: Option<OutputKind>,
    }
    impl LumaMaterializer for RejectingMaterializer {
        fn materialize(
            &mut self,
            result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            self.kind = Some(result.plan.output_kind);
            assert_eq!(result.color_mode, "RGBA");
            self.seen = result
                .media
                .iter()
                .map(|media| media.bytes.clone())
                .collect();
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn luma_real_encoded_media_and_custom_values_require_canonical_allocation_before_publish()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "LumaImageNode2",
            "LumaRay32TextToVideoNode",
            "LumaConceptsNode",
        ] {
            let mut transport = completed(node)?;
            if node == "LumaRay32TextToVideoNode" {
                transport.events=transport.events.into_iter().enumerate().map(|(index,event)|if index==7{Ok(ResponseEvent::Chunk(include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),"/../../sim_test_support/fixtures/providers/luma/downloaded-video.mp4")).to_vec()))}else{event}).collect();
            }
            let mut media = SourceProbeOwner::default();
            let mut materializer = RejectingMaterializer::default();
            let mut sink = Sink::default();
            assert!(matches!(
                invoke_and_materialize(
                    &input(node)?,
                    credentials(),
                    10,
                    &mut transport,
                    &mut MaterializationOwners {
                        media: &mut media,
                        materializer: &mut materializer,
                        sink: &mut sink
                    }
                ),
                Err(LumaError::Transport(ConformanceError::HostFailure))
            ));
            assert_eq!(sink.published, 0);
            if node == "LumaImageNode2" {
                assert_eq!(materializer.seen, vec![png()]);
            }
            if node == "LumaRay32TextToVideoNode" {
                assert_eq!(materializer.seen.first().ok_or("video")?.len(), 6163);
            }
            if node == "LumaConceptsNode" {
                assert!(materializer.seen.is_empty());
            }
        }
        Ok(())
    }
    #[test]
    fn luma_cancelled_and_clock_owner_failure_have_no_native_publication()
    -> Result<(), Box<dyn Error>> {
        let source = input("LumaImageNode2")?;
        let mut transport = Transport {
            cancelled: true,
            ..Transport::default()
        };
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                10,
                &mut transport,
                &mut SourceProbeOwner::default()
            ),
            Err(LumaError::Transport(ConformanceError::Cancelled))
        ));
        assert!(transport.heads.is_empty());
        let mut transport = completed(&source.node)?;
        transport.fail_policy = true;
        assert!(matches!(
            invoke(
                &source,
                credentials(),
                10,
                &mut transport,
                &mut SourceProbeOwner::default()
            ),
            Err(LumaError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn luma_unsafe_ids_relative_downloads_and_invalid_response_types_fail()
    -> Result<(), Box<dyn Error>> {
        for id in ["bad/id", "bad?x", "bad#x", "bad\\id", " bad "] {
            assert!(generation_identifier(&json!({"id":id}), true).is_err())
        }
        for bytes in [
            b"[]".as_slice(),
            b"bad".as_slice(),
            b"{\"state\":1}".as_slice(),
        ] {
            assert!(generation_response(bytes, true).is_err())
        }
        assert!(generation_response(b"{\"id\":\"x\"}", false).is_err());
        assert!(
            completed_output(
                &json!({"output":[{"url":"/proxy/media/result"}]}),
                OutputKind::Image,
                true
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn luma_missing_canonical_output_batch_is_rejected_atomically() -> Result<(), Box<dyn Error>> {
        struct EmptyMaterializer;
        impl LumaMaterializer for EmptyMaterializer {
            fn materialize(
                &mut self,
                _result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                Ok(vec![])
            }
        }
        let mut transport = completed("LumaImageNode2")?;
        let mut owner = SourceProbeOwner::default();
        let mut materializer = EmptyMaterializer;
        let mut sink = Sink::default();
        assert!(matches!(
            invoke_and_materialize(
                &input("LumaImageNode2")?,
                credentials(),
                10,
                &mut transport,
                &mut MaterializationOwners {
                    media: &mut owner,
                    materializer: &mut materializer,
                    sink: &mut sink
                }
            ),
            Err(LumaError::Transport(ConformanceError::Materialization(
                MaterializationError::InvalidSchema
            )))
        ));
        assert_eq!(sink.published, 0);
        Ok(())
    }
    #[test]
    fn luma_canonical_upload_cancellation_stops_before_generation_body()
    -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        let case = source["cases"]
            .as_array()
            .and_then(|cases| {
                cases.iter().find(|case| {
                    case["input"]["node"] == "LumaImageNode"
                        && case["input"]["arguments"]["image_luma_ref"].is_object()
                })
            })
            .ok_or("reference case")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let mut owner = SourceProbeOwner {
            cancel_after: Some(1),
            ..SourceProbeOwner::default()
        };
        let mut transport = Transport::default();
        assert!(matches!(
            invoke(&input, credentials(), 10, &mut transport, &mut owner),
            Err(LumaError::Transport(ConformanceError::Cancelled))
        ));
        assert_eq!(owner.operations.len(), 1);
        assert_eq!(transport.costs.len(), 1);
        assert!(transport.bodies.iter().all(Vec::is_empty));
        Ok(())
    }
    #[test]
    fn luma_custom_chain_outputs_require_existing_canonical_model_types()
    -> Result<(), Box<dyn Error>> {
        for (kind, identifier) in [
            (OutputKind::Concepts, "sim:custom-luma-concepts@1"),
            (OutputKind::References, "sim:custom-luma-ref@1"),
            (OutputKind::Keyframes, "sim:custom-luma-ray32-keyframe@1"),
        ] {
            let outputs = expected_outputs(kind)?;
            let output = outputs.first().ok_or("custom output")?;
            assert_eq!(output.type_id.to_string(), identifier);
            assert_eq!(output.port_id, "output_0");
            assert_eq!(output.family, crate::common::ValueFamily::Model);
            let image = expected_outputs(OutputKind::Image)?;
            let wrong = image.first().ok_or("image")?;
            assert_ne!(output.type_id, wrong.type_id);
        }
        Ok(())
    }
}
