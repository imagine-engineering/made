use crate::common::{
    ConformanceError, CostResponse, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TransportLimits, TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.recraft";
pub const SOURCE_SHA256: &str = "35a4c957bd8f34691c26235ccd9ee3efacabee51f7132de9c536e344a98bf410";
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
        feature_id: "COMFY-NODE-0509",
        node_identifier: "RecraftColorRGB",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0510",
        node_identifier: "RecraftControls",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0511",
        node_identifier: "RecraftCreateStyleNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0512",
        node_identifier: "RecraftCreativeUpscaleNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0513",
        node_identifier: "RecraftCrispUpscaleNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0514",
        node_identifier: "RecraftImageInpaintingNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0515",
        node_identifier: "RecraftImageToImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0516",
        node_identifier: "RecraftRemoveBackgroundNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0517",
        node_identifier: "RecraftReplaceBackgroundNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0518",
        node_identifier: "RecraftStyleV3DigitalIllustration",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0519",
        node_identifier: "RecraftStyleV3InfiniteStyleLibrary",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0520",
        node_identifier: "RecraftStyleV3LogoRaster",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0521",
        node_identifier: "RecraftStyleV3RealisticImage",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0522",
        node_identifier: "RecraftTextToImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0523",
        node_identifier: "RecraftTextToVectorNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0524",
        node_identifier: "RecraftV4TextToImageNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0525",
        node_identifier: "RecraftV4TextToVectorNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0526",
        node_identifier: "RecraftVectorizeImageNode",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0136",
        method: "POST",
        path: "/proxy/recraft/image_generation",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0137",
        method: "POST",
        path: "/proxy/recraft/images/creativeUpscale",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0138",
        method: "POST",
        path: "/proxy/recraft/images/crispUpscale",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0139",
        method: "POST",
        path: "/proxy/recraft/images/imageToImage",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0140",
        method: "POST",
        path: "/proxy/recraft/images/inpaint",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0141",
        method: "POST",
        path: "/proxy/recraft/images/removeBackground",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0142",
        method: "POST",
        path: "/proxy/recraft/images/replaceBackground",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0143",
        method: "POST",
        path: "/proxy/recraft/images/vectorize",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0144",
        method: "POST",
        path: "/proxy/recraft/styles",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum RecraftError {
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
impl fmt::Display for RecraftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "Recraft job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Recraft HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for RecraftError {}
impl From<ConformanceError> for RecraftError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for RecraftError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, RecraftError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(RecraftError::Input("Missing source string field"))
}
fn integer(value: &Value, key: &str) -> Result<i64, RecraftError> {
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
    .ok_or(RecraftError::Input("Missing source integer field"))
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageInput {
    pub role: String,
    pub batch: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default = "rgb_channels")]
    pub channels: u32,
}
fn rgb_channels() -> u32 {
    3
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MaskInput {
    pub batch: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub value: f64,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StyleInput {
    pub style: Option<String>,
    pub substyle: Option<String>,
    pub style_id: Option<String>,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ControlsInput {
    pub colors: Option<Vec<[i64; 3]>>,
    pub background_color: Option<Vec<[i64; 3]>>,
    pub no_text: Option<bool>,
    pub artistic_level: Option<i64>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    #[serde(default)]
    pub images: Vec<ImageInput>,
    pub mask: Option<MaskInput>,
    pub style: Option<StyleInput>,
    pub controls: Option<ControlsInput>,
    pub colors: Option<Vec<[i64; 3]>>,
    pub background_color: Option<Vec<[i64; 3]>>,
}
fn is_local(node: &str) -> bool {
    matches!(
        node,
        "RecraftColorRGB"
            | "RecraftControls"
            | "RecraftStyleV3RealisticImage"
            | "RecraftStyleV3DigitalIllustration"
            | "RecraftStyleV3LogoRaster"
            | "RecraftStyleV3InfiniteStyleLibrary"
    )
}
fn color_value(colors: &[[i64; 3]]) -> Value {
    json!({"kind":"color","colors":colors,"api":if colors.is_empty(){Value::Null}else{json!(colors.iter().map(|color|json!({"rgb":color})).collect::<Vec<_>>())}})
}
fn controls_api(controls: &ControlsInput) -> Option<Value> {
    if controls.colors.is_none()
        && controls.background_color.is_none()
        && controls.no_text.is_none()
        && controls.artistic_level.is_none()
    {
        return None;
    }
    let mut value = json!({});
    if let Some(colors) = &controls.colors
        && !colors.is_empty()
    {
        value["colors"] = json!(
            colors
                .iter()
                .map(|color| json!({"rgb":color}))
                .collect::<Vec<_>>()
        );
    }
    if let Some(color) = controls
        .background_color
        .as_ref()
        .and_then(|colors| colors.first())
    {
        value["background_color"] = json!({"rgb":color});
    }
    if let Some(value_bool) = controls.no_text {
        value["no_text"] = json!(value_bool);
    }
    if let Some(level) = controls.artistic_level {
        value["artistic_level"] = json!(level);
    }
    Some(value)
}
fn optional_substyle(value: Option<&str>) -> Option<&str> {
    value.filter(|value| *value != "None")
}
pub fn local_value(input: &Input) -> Result<Value, RecraftError> {
    match input.node.as_str() {
        "RecraftColorRGB" => {
            let mut colors = input.colors.clone().unwrap_or_default();
            colors.push([
                integer(&input.settings, "r")?,
                integer(&input.settings, "g")?,
                integer(&input.settings, "b")?,
            ]);
            if colors.len() > 128 {
                return Err(ConformanceError::BodyLimit.into());
            }
            Ok(color_value(&colors))
        }
        "RecraftControls" => {
            let controls = ControlsInput {
                colors: input.colors.clone(),
                background_color: input.background_color.clone(),
                ..ControlsInput::default()
            };
            Ok(
                json!({"kind":"controls","colors":input.colors.as_ref().map(|colors|color_value(colors)),"background_color":input.background_color.as_ref().map(|colors|color_value(colors)),"api":controls_api(&controls)}),
            )
        }
        "RecraftStyleV3InfiniteStyleLibrary" => {
            let id = string(&input.settings, "style_id")?;
            if id.is_empty() {
                return Err(RecraftError::Input("Source style ID cannot be empty"));
            }
            Ok(json!({"kind":"style","style":null,"substyle":null,"style_id":id}))
        }
        "RecraftStyleV3RealisticImage"
        | "RecraftStyleV3DigitalIllustration"
        | "RecraftStyleV3LogoRaster" => {
            let style = match input.node.as_str() {
                "RecraftStyleV3RealisticImage" => "realistic_image",
                "RecraftStyleV3DigitalIllustration" => "digital_illustration",
                _ => "logo_raster",
            };
            Ok(
                json!({"kind":"style","style":style,"substyle":optional_substyle(Some(string(&input.settings,"substyle")?)),"style_id":null}),
            )
        }
        _ => Err(RecraftError::Input("Not a source local helper")),
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum PreparationPurpose {
    Image,
    StyleFirstFrameWebP,
    MaskNearestExactThresholdHalf,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MediaRequirement {
    pub role: String,
    pub batch_index: u32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub purpose: PreparationPurpose,
    pub total_pixels: u64,
    pub encoded_mime_type: &'static str,
    pub field: String,
    pub file_name: String,
}
#[derive(Clone, Debug)]
pub struct PreparedMedia {
    pub requirement: MediaRequirement,
    pub bytes: Vec<u8>,
}
pub trait RecraftInputOwner {
    fn validate_models(&mut self, _: &Input) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn prepare(
        &mut self,
        _: &Input,
        _: &[MediaRequirement],
    ) -> Result<Vec<PreparedMedia>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FormField {
    pub name: String,
    pub value: String,
}
#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub ordinal: u32,
    pub path: String,
    pub body: Value,
    pub multipart: bool,
    pub form_fields: Vec<FormField>,
    pub media: Vec<MediaRequirement>,
}
fn scalar_text(value: &Value) -> Result<String, RecraftError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Bool(true) => Ok("True".into()),
        Value::Bool(false) => Ok("False".into()),
        Value::Number(value) => Ok(value.to_string()),
        _ => Err(RecraftError::Input("Invalid source multipart scalar")),
    }
}
fn flatten_controls(value: &Value, fields: &mut Vec<FormField>) -> Result<(), RecraftError> {
    if let Some(colors) = value.get("colors").and_then(Value::as_array) {
        for color in colors {
            for channel in color
                .get("rgb")
                .and_then(Value::as_array)
                .ok_or(RecraftError::Input("Invalid source RGB controls"))?
            {
                fields.push(FormField {
                    name: "controls[colors][][rgb][]".into(),
                    value: scalar_text(channel)?,
                });
            }
        }
    }
    if let Some(color) = value.get("background_color") {
        for channel in color
            .get("rgb")
            .and_then(Value::as_array)
            .ok_or(RecraftError::Input("Invalid source background controls"))?
        {
            fields.push(FormField {
                name: "controls[background_color][rgb][]".into(),
                value: scalar_text(channel)?,
            });
        }
    }
    for key in ["no_text", "artistic_level"] {
        if let Some(value) = value.get(key) {
            fields.push(FormField {
                name: format!("controls[{key}]"),
                value: scalar_text(value)?,
            });
        }
    }
    Ok(())
}
fn ordered_fields(body: &Value) -> Result<Vec<FormField>, RecraftError> {
    let mut fields = vec![];
    for key in [
        "prompt",
        "size",
        "n",
        "negative_prompt",
        "model",
        "style",
        "substyle",
        "controls",
        "style_id",
        "strength",
        "random_seed",
    ] {
        if let Some(value) = body.get(key) {
            if key == "controls" {
                flatten_controls(value, &mut fields)?;
            } else {
                fields.push(FormField {
                    name: key.into(),
                    value: scalar_text(value)?,
                });
            }
        }
    }
    Ok(fields)
}
fn valid_prompt(input: &Input) -> Result<(), RecraftError> {
    let prompt = string(&input.settings, "prompt")?;
    let length = prompt.chars().count();
    let (minimum, maximum) = match input.node.as_str() {
        "RecraftTextToImageNode" => (1, 1000),
        "RecraftV4TextToImageNode" | "RecraftV4TextToVectorNode" => (1, 10000),
        "RecraftImageToImageNode" | "RecraftImageInpaintingNode" | "RecraftTextToVectorNode" => {
            (0, 1000)
        }
        "RecraftReplaceBackgroundNode" => return Ok(()),
        _ => return Err(RecraftError::Input("Unexpected source prompt node")),
    };
    if length < minimum || length > maximum {
        return Err(RecraftError::Input("Source prompt length"));
    }
    Ok(())
}
fn image_body(input: &Input) -> Result<Value, RecraftError> {
    valid_prompt(input)?;
    if input.controls.is_some()
        && matches!(
            input.node.as_str(),
            "RecraftImageInpaintingNode" | "RecraftReplaceBackgroundNode"
        )
    {
        return Err(RecraftError::Input("Source node has no controls argument"));
    }
    let mut body = json!({"prompt":string(&input.settings,"prompt")?,"n":integer(&input.settings,"n")?,"model":"recraftv3"});
    if let Some(value) = input.settings.get("negative_prompt").filter(|value| {
        !value.is_null()
            && **value != json!("")
            && **value != json!(false)
            && value.as_f64() != Some(0.0)
            && !value.as_array().is_some_and(Vec::is_empty)
            && !value.as_object().is_some_and(serde_json::Map::is_empty)
    }) {
        body["negative_prompt"] = json!(
            value
                .as_str()
                .ok_or(RecraftError::Input("Invalid source negative prompt"))?
        );
    }
    if input.node.starts_with("RecraftV4") {
        let model = input
            .settings
            .get("model")
            .ok_or(RecraftError::Input("Missing source dynamic model"))?;
        body["model"] = json!(string(model, "model")?);
        body["size"] = json!(string(model, "size")?);
        if input.node == "RecraftV4TextToVectorNode" {
            body["style"] = json!("vector_illustration");
        }
    } else if input.node == "RecraftTextToVectorNode" {
        body["style"] = json!("vector_illustration");
        body["size"] = json!(string(&input.settings, "size")?);
        if let Some(substyle) = optional_substyle(Some(string(&input.settings, "substyle")?)) {
            body["substyle"] = json!(substyle);
        }
    } else {
        let default_style = StyleInput {
            style: Some("realistic_image".into()),
            ..StyleInput::default()
        };
        let style = input.style.as_ref().unwrap_or(&default_style);
        if let Some(style) = &style.style {
            body["style"] = json!(style);
        }
        if let Some(substyle) = optional_substyle(style.substyle.as_deref()) {
            body["substyle"] = json!(substyle);
        }
        if let Some(id) = &style.style_id {
            body["style_id"] = json!(id);
        }
        if input.node == "RecraftTextToImageNode" {
            body["size"] = json!(string(&input.settings, "size")?);
        }
    }
    if let Some(controls) = &input.controls
        && let Some(value) = controls_api(controls)
    {
        body["controls"] = value;
    }
    if input.node == "RecraftImageToImageNode" {
        let strength = input
            .settings
            .get("strength")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or(RecraftError::Input("Invalid source strength"))?;
        // Decimal formatting rounds the actual binary input to even, unlike
        // multiplying by100 first, which would change source halfway cases.
        let rounded = format!("{strength:.2}")
            .parse::<f64>()
            .map_err(|_| RecraftError::Input("Source strength rounding"))?;
        body["strength"] = json!(rounded);
    }
    Ok(body)
}
pub fn request_plans(input: &Input) -> Result<Vec<RequestPlan>, RecraftError> {
    if !NODES
        .iter()
        .any(|claim| claim.node_identifier == input.node)
    {
        return Err(RecraftError::Input("Unknown Recraft node"));
    }
    if is_local(&input.node) {
        local_value(input)?;
        return Ok(vec![]);
    }
    let image_request = matches!(
        input.node.as_str(),
        "RecraftImageToImageNode"
            | "RecraftImageInpaintingNode"
            | "RecraftReplaceBackgroundNode"
            | "RecraftVectorizeImageNode"
            | "RecraftRemoveBackgroundNode"
            | "RecraftCrispUpscaleNode"
            | "RecraftCreativeUpscaleNode"
    );
    let style_request = input.node == "RecraftCreateStyleNode";
    let path = match input.node.as_str() {
        "RecraftCreateStyleNode" => "/proxy/recraft/styles",
        "RecraftImageToImageNode" => "/proxy/recraft/images/imageToImage",
        "RecraftImageInpaintingNode" => "/proxy/recraft/images/inpaint",
        "RecraftReplaceBackgroundNode" => "/proxy/recraft/images/replaceBackground",
        "RecraftVectorizeImageNode" => "/proxy/recraft/images/vectorize",
        "RecraftRemoveBackgroundNode" => "/proxy/recraft/images/removeBackground",
        "RecraftCrispUpscaleNode" => "/proxy/recraft/images/crispUpscale",
        "RecraftCreativeUpscaleNode" => "/proxy/recraft/images/creativeUpscale",
        _ => "/proxy/recraft/image_generation",
    };
    let body = if style_request {
        json!({"style":string(&input.settings,"style")?})
    } else if matches!(
        input.node.as_str(),
        "RecraftVectorizeImageNode"
            | "RecraftRemoveBackgroundNode"
            | "RecraftCrispUpscaleNode"
            | "RecraftCreativeUpscaleNode"
    ) {
        Value::Null
    } else {
        image_body(input)?
    };
    let fields = if style_request {
        vec![FormField {
            name: "style".into(),
            value: string(&input.settings, "style")?.into(),
        }]
    } else {
        ordered_fields(&body)?
    };
    let count = if image_request {
        input
            .images
            .first()
            .ok_or(RecraftError::Input("Missing canonical source image"))?
            .batch
    } else {
        1
    };
    if count > 128 || input.images.len() > 128 {
        return Err(ConformanceError::BodyLimit.into());
    }
    for image in &input.images {
        if image.width == 0
            || image.height == 0
            || !(1..=4).contains(&image.channels)
            || image.role.is_empty()
        {
            return Err(RecraftError::Input("Invalid canonical image requirement"));
        }
    }
    if input.node == "RecraftImageInpaintingNode" && input.mask.is_none() {
        return Err(RecraftError::Input("Missing source mask"));
    }
    let mut plans = vec![];
    for ordinal in 0..count {
        let mut media = vec![];
        for (index, image) in input.images.iter().enumerate() {
            if !image_request && !style_request {
                break;
            }
            if image_request && index != 0 {
                break;
            }
            let field = if style_request {
                format!("file{}", index + 1)
            } else {
                "image".into()
            };
            media.push(MediaRequirement {
                role: image.role.clone(),
                batch_index: if style_request { 0 } else { ordinal },
                width: image.width,
                height: image.height,
                channels: image.channels,
                purpose: if style_request {
                    PreparationPurpose::StyleFirstFrameWebP
                } else {
                    PreparationPurpose::Image
                },
                total_pixels: if style_request {
                    2048 * 2048
                } else {
                    4096 * 4096
                },
                encoded_mime_type: if style_request {
                    "image/webp"
                } else {
                    "image/png"
                },
                file_name: field.clone(),
                field,
            });
        }
        if input.node == "RecraftImageInpaintingNode" {
            let image = input
                .images
                .first()
                .ok_or(RecraftError::Input("Missing image"))?;
            media.push(MediaRequirement {
                role: "mask".into(),
                batch_index: ordinal,
                width: image.width,
                height: image.height,
                channels: 1,
                purpose: PreparationPurpose::MaskNearestExactThresholdHalf,
                total_pixels: 4096 * 4096,
                encoded_mime_type: "image/png",
                field: "mask".into(),
                file_name: "mask".into(),
            });
        }
        plans.push(RequestPlan {
            ordinal,
            path: path.into(),
            body: body.clone(),
            multipart: image_request || style_request,
            form_fields: if image_request || style_request {
                fields.clone()
            } else {
                vec![]
            },
            media,
        });
    }
    Ok(plans)
}
pub fn encode_request(
    plan: &RequestPlan,
    prepared: &[PreparedMedia],
) -> Result<Vec<u8>, RecraftError> {
    if !plan.multipart {
        let bytes = serde_json::to_vec(&plan.body)?;
        if bytes.len() > MAXIMUM_REQUEST_BYTES {
            return Err(ConformanceError::BodyLimit.into());
        }
        return Ok(bytes);
    }
    if plan.media.len() != prepared.len() {
        return Err(RecraftError::Input("Missing canonical encoded media"));
    }
    let text_bytes = plan.form_fields.iter().try_fold(0usize, |total, field| {
        total
            .checked_add(field.value.len())
            .ok_or(ConformanceError::BodyLimit)
    })?;
    let total = prepared.iter().try_fold(text_bytes, |total, media| {
        total
            .checked_add(media.bytes.len())
            .ok_or(ConformanceError::BodyLimit)
    })?;
    if total > MAXIMUM_REQUEST_BYTES
        || plan.path == "/proxy/recraft/styles" && total > 5 * 1024 * 1024
    {
        return Err(ConformanceError::BodyLimit.into());
    }
    let mut fields = plan
        .form_fields
        .iter()
        .map(|field| MultipartField {
            name: field.name.clone(),
            file_name: None,
            media_type: Some("text/plain; charset=utf-8".into()),
            bytes: field.value.as_bytes().to_vec(),
        })
        .collect::<Vec<_>>();
    for (requirement, media) in plan.media.iter().zip(prepared) {
        if requirement != &media.requirement || media.bytes.is_empty() {
            return Err(RecraftError::Input("Altered canonical media requirement"));
        }
        fields.push(MultipartField {
            name: requirement.field.clone(),
            file_name: Some(requirement.file_name.clone()),
            media_type: Some("application/octet-stream".into()),
            bytes: media.bytes.clone(),
        });
    }
    Ok(MultipartField::encode_checked(
        &fields,
        "sim-recraft-canonical-boundary",
        TransportLimits {
            maximum_body_bytes: MAXIMUM_REQUEST_BYTES,
            maximum_chunk_bytes: MAXIMUM_CHUNK_BYTES,
            ..TransportLimits::conformance()
        },
    )?)
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SourcePriceBadge {
    pub expression: &'static str,
    pub widgets: Value,
}
pub fn source_price_badge(input: &Input) -> Result<Option<SourcePriceBadge>, RecraftError> {
    let expression = match input.node.as_str() {
        "RecraftCreateStyleNode" | "RecraftReplaceBackgroundNode" => r#"{"type":"usd","usd":0.04}"#,
        "RecraftTextToImageNode" | "RecraftImageToImageNode" | "RecraftImageInpaintingNode" => {
            r#"{"type":"usd","usd":$round(0.04*widgets.n,2)}"#
        }
        "RecraftTextToVectorNode" => r#"{"type":"usd","usd":$round(0.08*widgets.n,2)}"#,
        "RecraftVectorizeImageNode" | "RecraftRemoveBackgroundNode" => {
            r#"{"type":"usd","usd":0.01}"#
        }
        "RecraftCrispUpscaleNode" => r#"{"type":"usd","usd":0.004}"#,
        "RecraftCreativeUpscaleNode" => r#"{"type":"usd","usd":0.25}"#,
        "RecraftV4TextToImageNode" => {
            r#"($prices:={"recraftv4":0.04,"recraftv4_pro":0.25};{"type":"usd","usd":$lookup($prices,widgets.model)*widgets.n})"#
        }
        "RecraftV4TextToVectorNode" => {
            r#"($prices:={"recraftv4":0.08,"recraftv4_pro":0.30};{"type":"usd","usd":$lookup($prices,widgets.model)*widgets.n})"#
        }
        _ if is_local(&input.node) => return Ok(None),
        _ => return Err(RecraftError::Input("Unknown source price badge")),
    };
    let mut widgets = json!({});
    if expression.contains("widgets.n") {
        widgets["n"] = json!(integer(&input.settings, "n")?);
    }
    if expression.contains("widgets.model") {
        widgets["model"] = json!(string(
            input
                .settings
                .get("model")
                .ok_or(RecraftError::Input("Missing source dynamic model"))?,
            "model"
        )?);
    }
    Ok(Some(SourcePriceBadge {
        expression,
        widgets,
    }))
}
pub trait RecraftTransport: ProviderTransport {
    fn begin_response_deadline(
        &mut self,
        _handle: StreamHandle,
        _timeout_milliseconds: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    // The trusted owner must reserve against one retained invocation cap across
    // all paid requests. A component cannot allocate that budget or remint it.
    fn reserve_paid_request(
        &mut self,
        _handle: StreamHandle,
        _maximum_invocation_microunits: u64,
        _ordinal: u32,
        _paid_request_count: u32,
        _source_quote: Option<&SourcePriceBadge>,
    ) -> Result<CostResponse, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn report_batch_frame(
        &mut self,
        _handle: StreamHandle,
        _completed: u32,
        _total: u32,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
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
    fn delay(&mut self, handle: StreamHandle, milliseconds: u64) -> Result<(), ConformanceError>;
}
#[derive(Default)]
struct Session {
    invocation: Option<u64>,
    handles: BTreeSet<(u64, u32, u32)>,
}
impl Session {
    fn retain(&mut self, handle: StreamHandle) -> Result<(), RecraftError> {
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
fn endpoint(value: &str) -> Result<bool, RecraftError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(RecraftError::Response("Invalid Recraft response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(RecraftError::Response("Invalid Recraft response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(RecraftError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(RecraftError::Response("Invalid Recraft endpoint authority"));
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
) -> Result<RequestHead, RecraftError> {
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
            return Err(RecraftError::Input("Missing Comfy API credentials"));
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
    ordinary_limit: 1,
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
    timeout_milliseconds: Some(1_024_000),
};
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    session: &mut Session,
    transport: &mut dyn RecraftTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), RecraftError> {
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
    transport: &mut dyn RecraftTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), RecraftError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(RecraftError::Http {
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
                    return Err(RecraftError::Http {
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
    transport: &mut dyn RecraftTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), RecraftError> {
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
                    return Err(RecraftError::Http {
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

#[derive(Clone, Debug)]
pub struct DownloadedMedia {
    pub request_ordinal: u32,
    pub response_index: usize,
    pub image_id: String,
    pub url: String,
    pub bytes: Vec<u8>,
    pub handle: StreamHandle,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct PaidResponse {
    pub ordinal: u32,
    pub body: Value,
    pub bytes: Vec<u8>,
    pub handle: StreamHandle,
    pub receipt: Vec<u8>,
    pub cost_receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct InvocationResult {
    pub node: String,
    pub local_value: Option<Value>,
    pub style_id: Option<String>,
    pub responses: Vec<PaidResponse>,
    pub media: Vec<DownloadedMedia>,
}
pub trait RecraftOutputOwner {
    fn complete_download(
        &mut self,
        _node: &str,
        _download: &DownloadedMedia,
        _parent: &PaidResponse,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn complete_frame(
        &mut self,
        _node: &str,
        _ordinal: u32,
        _media: &[DownloadedMedia],
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn materialize(
        &mut self,
        _: &InvocationResult,
        _: &[ExpectedOutput],
    ) -> Result<Vec<EncodedOutput>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn download_entries(
    response: &Value,
    file_helper: bool,
) -> Result<Vec<(String, String)>, RecraftError> {
    integer(response, "created")?;
    integer(response, "credits")?;
    if let Some(data) = response.get("data").filter(|value| !value.is_null()) {
        for entry in data
            .as_array()
            .ok_or(RecraftError::Response("Invalid source data model"))?
        {
            string(entry, "image_id")?;
            string(entry, "url")?;
        }
    }
    if let Some(image) = response.get("image").filter(|value| !value.is_null()) {
        string(image, "image_id")?;
        string(image, "url")?;
    }
    let entries = if let Some(image) = response
        .get("image")
        .filter(|image| file_helper && !image.is_null())
    {
        vec![image]
    } else {
        response
            .get("data")
            .and_then(Value::as_array)
            .ok_or(RecraftError::Response("Missing source returned images"))?
            .iter()
            .collect()
    };
    if entries.len() > 128 {
        return Err(ConformanceError::BodyLimit.into());
    }
    entries
        .into_iter()
        .map(|entry| {
            Ok((
                string(entry, "image_id")?.into(),
                string(entry, "url")?.into(),
            ))
        })
        .collect()
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_invocation_microunits: u64,
    transport: &mut dyn RecraftTransport,
    input_owner: &mut dyn RecraftInputOwner,
    output_owner: &mut dyn RecraftOutputOwner,
) -> Result<InvocationResult, RecraftError> {
    transport.check_cancelled(None)?;
    if input.style.is_some()
        || input.controls.is_some()
        || input.colors.is_some()
        || input.background_color.is_some()
    {
        input_owner.validate_models(input)?;
        transport.check_cancelled(None)?;
    }
    let plans = request_plans(input)?;
    if is_local(&input.node) {
        return Ok(InvocationResult {
            node: input.node.clone(),
            local_value: Some(local_value(input)?),
            style_id: None,
            responses: vec![],
            media: vec![],
        });
    }
    if plans.is_empty() && input.node == "RecraftVectorizeImageNode" {
        return Ok(InvocationResult {
            node: input.node.clone(),
            local_value: None,
            style_id: None,
            responses: vec![],
            media: vec![],
        });
    }
    if plans.is_empty() {
        return Err(RecraftError::Input(
            "Source cannot concatenate empty frame batch",
        ));
    }
    if maximum_invocation_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let count = u32::try_from(plans.len()).map_err(|_| ConformanceError::BodyLimit)?;
    let mut session = Session::default();
    let mut responses = vec![];
    let mut downloaded = vec![];
    let mut style_id = None;
    let mut downloaded_bytes = 0usize;
    for plan in &plans {
        transport.check_cancelled(None)?;
        let prepared = if input.node == "RecraftCreateStyleNode" {
            let mut prepared = vec![];
            let mut total = 0usize;
            for requirement in &plan.media {
                transport.check_cancelled(None)?;
                let next = input_owner.prepare(input, std::slice::from_ref(requirement))?;
                if next.len() != 1 {
                    return Err(RecraftError::Input("Missing canonical style part"));
                }
                let media = next
                    .into_iter()
                    .next()
                    .ok_or(RecraftError::Input("Missing style bytes"))?;
                total = total
                    .checked_add(media.bytes.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                if total > 5 * 1024 * 1024 {
                    return Err(ConformanceError::BodyLimit.into());
                }
                prepared.push(media);
            }
            prepared
        } else {
            let mut prepared = vec![];
            for requirement in &plan.media {
                transport.check_cancelled(None)?;
                // The source helper encodes the image before indexing the
                // resized mask, so an exhausted mask cannot skip that work.
                if requirement.purpose == PreparationPurpose::MaskNearestExactThresholdHalf
                    && input
                        .mask
                        .as_ref()
                        .is_none_or(|mask| requirement.batch_index >= mask.batch)
                {
                    return Err(RecraftError::Input("Source resized mask batch exhausted"));
                }
                let mut part = input_owner.prepare(input, std::slice::from_ref(requirement))?;
                if part.len() != 1 {
                    return Err(RecraftError::Input("Missing canonical frame part"));
                }
                prepared.append(&mut part);
            }
            prepared
        };
        transport.check_cancelled(None)?;
        let body = encode_request(plan, &prepared)?;
        let mut head = request_head(&plan.path, HttpMethod::Post, credentials, false)?;
        if plan.multipart {
            let header = head
                .headers
                .iter_mut()
                .find(|header| header.name == "Content-Type")
                .ok_or(ConformanceError::HeaderLimit)?;
            header.value = "multipart/form-data; boundary=sim-recraft-canonical-boundary".into();
        }
        let handle = transport.start_request(&head)?;
        session.retain(handle)?;
        let cost = transport.reserve_paid_request(
            handle,
            maximum_invocation_microunits,
            plan.ordinal,
            count,
            source_price_badge(input)?.as_ref(),
        )?;
        if !cost.accepted
            || cost.approved_microunits == 0
            || cost.approved_microunits > maximum_invocation_microunits
            || cost.receipt.is_empty()
            || cost.receipt.len() > MAXIMUM_RECEIPT_BYTES
        {
            return Err(ConformanceError::InvalidCost.into());
        }
        let (bytes, receipt, response_handle) = exchange_existing(
            &head,
            &body,
            MAXIMUM_RESPONSE_BYTES,
            GENERATION_RETRY,
            handle,
            &mut session,
            transport,
        )?;
        let response: Value = serde_json::from_slice(&bytes)?;
        let parent = PaidResponse {
            ordinal: plan.ordinal,
            body: response.clone(),
            bytes,
            handle: response_handle,
            receipt,
            cost_receipt: cost.receipt,
        };
        let first_download = downloaded.len();
        if input.node == "RecraftCreateStyleNode" {
            style_id = Some(string(&response, "id")?.into());
        } else {
            for (response_index, (image_id, url)) in download_entries(&response, plan.multipart)?
                .into_iter()
                .enumerate()
            {
                if endpoint(&url)? {
                    return Err(RecraftError::Response(
                        "Source download requires absolute URL without class",
                    ));
                }
                let head = request_head(&url, HttpMethod::Get, credentials, true)?;
                let (bytes, receipt, handle) = exchange_with_policy(
                    &head,
                    &[],
                    MAXIMUM_MEDIA_BYTES,
                    DOWNLOAD_RETRY,
                    &mut session,
                    transport,
                )?;
                downloaded_bytes = downloaded_bytes
                    .checked_add(bytes.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                if downloaded_bytes > MAXIMUM_MEDIA_BYTES {
                    return Err(ConformanceError::BodyLimit.into());
                }
                let download = DownloadedMedia {
                    request_ordinal: plan.ordinal,
                    response_index,
                    image_id,
                    url,
                    bytes,
                    handle,
                    receipt,
                };
                transport.check_cancelled(Some(handle))?;
                if !plan.multipart {
                    output_owner.complete_download(&input.node, &download, &parent)?;
                }
                transport.check_cancelled(Some(handle))?;
                downloaded.push(download);
            }
        }
        if input.node != "RecraftCreateStyleNode" {
            let frame = downloaded
                .get(first_download..)
                .ok_or(ConformanceError::InvalidResponse)?;
            if plan.multipart {
                for download in frame {
                    transport.check_cancelled(Some(download.handle))?;
                    output_owner.complete_download(&input.node, download, &parent)?;
                    transport.check_cancelled(Some(download.handle))?;
                }
            }
            transport.check_cancelled(Some(response_handle))?;
            output_owner.complete_frame(&input.node, plan.ordinal, frame)?;
        }
        responses.push(parent);
        if matches!(
            input.node.as_str(),
            "RecraftImageToImageNode"
                | "RecraftImageInpaintingNode"
                | "RecraftReplaceBackgroundNode"
                | "RecraftVectorizeImageNode"
                | "RecraftRemoveBackgroundNode"
                | "RecraftCrispUpscaleNode"
                | "RecraftCreativeUpscaleNode"
        ) {
            transport.report_batch_frame(response_handle, plan.ordinal + 1, count)?;
        }
        transport.check_cancelled(Some(response_handle))?;
    }
    if plans.is_empty() {
        return Err(RecraftError::Input(
            "Source cannot concatenate empty frame batch",
        ));
    }
    Ok(InvocationResult {
        node: input.node.clone(),
        local_value: None,
        style_id,
        responses,
        media: downloaded,
    })
}
pub fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, RecraftError> {
    let names = match node {
        "RecraftColorRGB" => vec!["RecraftIO.COLOR"],
        "RecraftControls" => vec!["RecraftIO.CONTROLS"],
        "RecraftStyleV3RealisticImage"
        | "RecraftStyleV3DigitalIllustration"
        | "RecraftStyleV3LogoRaster"
        | "RecraftStyleV3InfiniteStyleLibrary" => vec!["RecraftIO.STYLEV3"],
        "RecraftCreateStyleNode" => vec!["STRING"],
        "RecraftTextToVectorNode" | "RecraftVectorizeImageNode" | "RecraftV4TextToVectorNode" => {
            vec!["SVG"]
        }
        "RecraftRemoveBackgroundNode" => vec!["IMAGE", "MASK"],
        _ if NODES.iter().any(|claim| claim.node_identifier == node) => vec!["IMAGE"],
        _ => return Err(RecraftError::Input("Unknown source output node")),
    };
    let registry = TypeRegistry::built_in()
        .map_err(|_| RecraftError::Response("Missing canonical registry"))?;
    names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let type_id = registry
                .resolve(name)
                .map_err(|_| RecraftError::Response("Missing source type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| RecraftError::Response("Missing source family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect()
}
pub fn materialize_and_publish(
    result: &InvocationResult,
    transport: &mut dyn RecraftTransport,
    owner: &mut dyn RecraftOutputOwner,
    sink: &mut dyn MaterializationSink,
) -> Result<(), RecraftError> {
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
        "/../../sim_test_support/fixtures/providers/recraft/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
fn decode_hex(value: &str) -> Result<Vec<u8>, ConformanceError> {
    if !value.is_ascii() || !value.len().is_multiple_of(2) {
        return Err(ConformanceError::InvalidRequest);
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).map_err(|_| ConformanceError::InvalidRequest)?;
            u8::from_str_radix(text, 16).map_err(|_| ConformanceError::InvalidRequest)
        })
        .collect()
}
fn oracle_prepared(
    plan: &RequestPlan,
    request: &Value,
) -> Result<Vec<PreparedMedia>, ConformanceError> {
    let files = request["files"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?;
    if files.len() != plan.media.len() {
        return Err(ConformanceError::InvalidRequest);
    }
    plan.media
        .iter()
        .zip(files)
        .map(|(requirement, file)| {
            if file["field"] != requirement.field
                || file["file_name"] != requirement.file_name
                || file["mime_type"] != "application/octet-stream"
            {
                return Err(ConformanceError::InvalidRequest);
            }
            Ok(PreparedMedia {
                requirement: requirement.clone(),
                bytes: decode_hex(
                    file["bytes_hex"]
                        .as_str()
                        .ok_or(ConformanceError::InvalidRequest)?,
                )?,
            })
        })
        .collect()
}
fn compare_request(plan: &RequestPlan, request: &Value) -> Result<(), ConformanceError> {
    if json!(plan.path) != request["path"]
        || request["method"] != "POST"
        || plan.body != request["body"]
        || json!(plan.form_fields) != request["form_fields"]
        || plan.multipart != (request["content_type"] == "multipart/form-data")
        || request["max_retries"] != 1
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let bytes = encode_request(plan, &oracle_prepared(plan, request)?)
        .map_err(|_| ConformanceError::InvalidRequest)?;
    if plan.multipart
        && bytes
            != decode_hex(
                request["wire_hex"]
                    .as_str()
                    .ok_or(ConformanceError::InvalidRequest)?,
            )?
    {
        return Err(ConformanceError::InvalidRequest);
    }
    Ok(())
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
    if source["sources"]["comfy_api_nodes/nodes_recraft.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let plans = request_plans(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let requests = case["requests"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if plans.len() != requests.len() {
            return Err(ConformanceError::InvalidRequest);
        }
        if is_local(&input.node)
            && local_value(&input).map_err(|_| ConformanceError::InvalidRequest)?
                != case["output"][0]
        {
            return Err(ConformanceError::InvalidRequest);
        }
        for (plan, request) in plans.iter().zip(requests) {
            compare_request(plan, request)?;
        }
        covered.insert(input.node);
    }
    for case in source["errors"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let requests = case["requests"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        match request_plans(&input) {
            Ok(plans) if !requests.is_empty() => {
                if plans.len() < requests.len() {
                    return Err(ConformanceError::InvalidRequest);
                }
                for (plan, request) in plans.iter().zip(requests) {
                    compare_request(plan, request)?;
                }
            }
            Ok(plans)
                if plans.is_empty()
                    && requests.is_empty()
                    && !is_local(&input.node)
                    && input.node != "RecraftVectorizeImageNode" => {}
            Err(_) if requests.is_empty() => {}
            _ => return Err(ConformanceError::InvalidRequest),
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
        CostRequest, Progress, SignedPutRequest, StorageCreateRequest, StorageCreateResponse,
    };
    use std::{cell::RefCell, collections::VecDeque, rc::Rc};
    type Events = Rc<RefCell<Vec<String>>>;
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
        events: Events,
        heads: Vec<RequestHead>,
        deadlines: Vec<Option<u64>>,
        frames: Vec<(u32, u32)>,
        delays: Vec<u64>,
        reservations: Vec<(u32, u32, u64)>,
        reserved: u64,
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
                events,
                heads: vec![],
                deadlines: vec![],
                frames: vec![],
                delays: vec![],
                reservations: vec![],
                reserved: 0,
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
                invocation: 42,
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
            Err(ConformanceError::HostFailure)
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
            let event = match *sequence {
                0 => ResponseEvent::Head {
                    status: response.status,
                    headers: response.headers.clone(),
                },
                1 => ResponseEvent::Chunk(response.bytes.clone()),
                2 => ResponseEvent::Completed(response.receipt.clone()),
                _ => return Err(ConformanceError::InvalidResponse),
            };
            *sequence += 1;
            if matches!(event, ResponseEvent::Completed(_)) {
                self.active = None;
            }
            Ok(event)
        }
    }
    impl RecraftTransport for TestTransport {
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
        fn reserve_paid_request(
            &mut self,
            handle: StreamHandle,
            cap: u64,
            ordinal: u32,
            count: u32,
            quote: Option<&SourcePriceBadge>,
        ) -> Result<CostResponse, ConformanceError> {
            if quote.is_none()
                || handle.invocation != 42
                || ordinal as usize != self.reservations.len()
                || ordinal >= count
                || count == 0
                || self
                    .reservations
                    .first()
                    .is_some_and(|(_, previous_count, previous_cap)| {
                        *previous_count != count || *previous_cap != cap
                    })
            {
                return Err(ConformanceError::InvalidCost);
            }
            let next = self
                .reserved
                .checked_add(1)
                .ok_or(ConformanceError::InvalidCost)?;
            if next > cap || self.reject_cost {
                return Err(ConformanceError::InvalidCost);
            }
            self.reserved = next;
            self.reservations.push((ordinal, count, cap));
            Ok(CostResponse {
                accepted: true,
                approved_microunits: 1,
                receipt: b"checked test reservation".to_vec(),
            })
        }
        fn report_batch_frame(
            &mut self,
            handle: StreamHandle,
            completed: u32,
            total: u32,
        ) -> Result<(), ConformanceError> {
            if handle.invocation != 42
                || completed != self.frames.len() as u32 + 1
                || completed > total
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.frames.push((completed, total));
            self.events.borrow_mut().push("progress".into());
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
    struct InputOwner {
        parts: VecDeque<PreparedMedia>,
        prepared: Vec<MediaRequirement>,
    }
    impl RecraftInputOwner for InputOwner {
        fn validate_models(&mut self, _: &Input) -> Result<(), ConformanceError> {
            Ok(())
        }
        fn prepare(
            &mut self,
            _: &Input,
            requirements: &[MediaRequirement],
        ) -> Result<Vec<PreparedMedia>, ConformanceError> {
            requirements
                .iter()
                .map(|requirement| {
                    let part = self
                        .parts
                        .pop_front()
                        .ok_or(ConformanceError::HostFailure)?;
                    if &part.requirement != requirement {
                        return Err(ConformanceError::InvalidRequest);
                    }
                    self.prepared.push(requirement.clone());
                    Ok(part)
                })
                .collect()
        }
    }
    struct OutputOwner {
        events: Events,
        frames: usize,
        fail_frame: bool,
    }
    impl RecraftOutputOwner for OutputOwner {
        fn complete_download(
            &mut self,
            node: &str,
            download: &DownloadedMedia,
            parent: &PaidResponse,
        ) -> Result<(), ConformanceError> {
            if download.handle.invocation != parent.handle.invocation
                || download.request_ordinal != parent.ordinal
                || download.receipt.is_empty()
                || parent.receipt.is_empty()
            {
                return Err(ConformanceError::InvalidHandle);
            }
            if !matches!(
                node,
                "RecraftTextToVectorNode"
                    | "RecraftVectorizeImageNode"
                    | "RecraftV4TextToVectorNode"
            ) {
                self.events.borrow_mut().push("decode".into());
            }
            if download.bytes == b"not an encoded image" {
                return Err(ConformanceError::InvalidResponse);
            }
            Ok(())
        }
        fn complete_frame(
            &mut self,
            node: &str,
            _: u32,
            downloads: &[DownloadedMedia],
        ) -> Result<(), ConformanceError> {
            if self.fail_frame
                || downloads.is_empty()
                    && !matches!(
                        node,
                        "RecraftTextToVectorNode"
                            | "RecraftVectorizeImageNode"
                            | "RecraftV4TextToVectorNode"
                    )
            {
                return Err(ConformanceError::HostFailure);
            }
            self.frames += 1;
            Ok(())
        }
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn scripted(
        case: &Value,
    ) -> Result<(Input, TestTransport, InputOwner, OutputOwner), Box<dyn Error>> {
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let events = Rc::new(RefCell::new(vec![]));
        let mut parts = vec![];
        let mut script = vec![];
        let plans = match request_plans(&input) {
            Ok(plans) => plans,
            Err(_) if case.get("error_type").is_some() && case["requests"] == json!([]) => vec![],
            Err(error) => return Err(error.into()),
        };
        for (ordinal, request) in case["requests"]
            .as_array()
            .ok_or("requests")?
            .iter()
            .enumerate()
        {
            let plan = plans.get(ordinal).ok_or("plan")?;
            let prepared = oracle_prepared(plan, request)?;
            let body = encode_request(plan, &prepared)?;
            parts.extend(prepared);
            let n = request["body"]["n"].as_i64().unwrap_or(1).max(0);
            let data=(0..n).map(|index|json!({"image_id":format!("image-{index}"),"url":format!("https://source.fixture.invalid/{}-{index}.bin",ordinal+1)})).collect::<Vec<_>>();
            let mut response = if input.node == "RecraftCreateStyleNode" {
                json!({"id":"style-42"})
            } else {
                json!({"created":12,"credits":34,"data":data})
            };
            if case["input"]["oracle_options"]["single_image"] == true {
                response["image"] = json!({"image_id":"preferred-single","url":"https://source.fixture.invalid/preferred-single.bin"});
            }
            if case["input"]["oracle_options"]["missing_data"] == true {
                response["data"] = Value::Null;
            }
            if case["input"]["oracle_options"]["malformed_image"] == true {
                response["image"] = json!({"url":"https://source.fixture.invalid/invalid.bin"});
            }
            script.push(Response {
                path: plan.path.clone(),
                method: HttpMethod::Post,
                body,
                status: 200,
                bytes: serde_json::to_vec(&response)?,
                headers: vec![],
                receipt: b"response receipt".to_vec(),
            });
            let returned = match if input.node == "RecraftCreateStyleNode" {
                Ok(vec![])
            } else {
                download_entries(&response, plan.multipart)
            } {
                Ok(returned) => returned,
                Err(_) if case.get("error_type").is_some() => vec![],
                Err(error) => return Err(error.into()),
            };
            for (_, url) in returned {
                if let Some(download) = case["downloads"]
                    .as_array()
                    .ok_or("downloads")?
                    .iter()
                    .find(|download| download["url"] == url)
                {
                    script.push(Response {
                        path: url,
                        method: HttpMethod::Get,
                        body: vec![],
                        status: 200,
                        bytes: decode_hex(download["bytes_hex"].as_str().ok_or("bytes")?)?,
                        headers: vec![],
                        receipt: b"download receipt".to_vec(),
                    });
                }
            }
        }
        if input.node == "RecraftImageInpaintingNode"
            && input.mask.as_ref().is_some_and(|mask| {
                input
                    .images
                    .first()
                    .is_some_and(|image| mask.batch < image.batch)
            })
        {
            let next_plan = plans
                .get(case["requests"].as_array().ok_or("requests")?.len())
                .ok_or("late mask plan")?;
            let requirement = next_plan.media.first().ok_or("late image")?;
            let bytes = parts
                .first()
                .ok_or("first encoded source image")?
                .bytes
                .clone();
            parts.push(PreparedMedia {
                requirement: requirement.clone(),
                bytes,
            });
        }
        Ok((
            input,
            TestTransport::new(script, events.clone()),
            InputOwner {
                parts: parts.into(),
                prepared: vec![],
            },
            OutputOwner {
                events,
                frames: 0,
                fail_frame: false,
            },
        ))
    }
    fn source_events(case: &Value) -> Result<Vec<String>, Box<dyn Error>> {
        case["phases"]
            .as_array()
            .ok_or("phases")?
            .iter()
            .filter(|phase| phase.get("progress_total").is_none())
            .map(|phase| {
                if let Some(path) = phase["request"].as_str() {
                    Ok(format!("request:{path}"))
                } else if let Some(url) = phase["download"].as_str() {
                    Ok(format!("download:{url}"))
                } else if phase.get("decode_sha256").is_some() {
                    Ok("decode".into())
                } else if phase.get("progress_update").is_some() {
                    Ok("progress".into())
                } else {
                    Err("unknown phase".into())
                }
            })
            .collect()
    }
    #[test]
    fn exact_pinned_source_requests_helpers_and_multipart() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/recraft/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn all_source_successes_and_ordered_failures_execute_checked_callbacks()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for (expected_success, key) in [(true, "cases"), (false, "errors")] {
            for (index, case) in source[key]
                .as_array()
                .ok_or("source cases")?
                .iter()
                .enumerate()
            {
                let (input, mut transport, mut input_owner, mut output_owner) = scripted(case)
                    .map_err(|error| format!("{key}[{index}] fixture harness: {error}"))?;
                let result = invoke(
                    &input,
                    credentials(),
                    1000,
                    &mut transport,
                    &mut input_owner,
                    &mut output_owner,
                );
                if result.is_ok() != expected_success {
                    return Err(format!("{key}[{index}] {}: {result:?}", input.node).into());
                }
                if *transport.events.borrow() != source_events(case)? {
                    return Err(format!(
                        "{key}[{index}] {} phase mismatch {:?} vs {:?}",
                        input.node,
                        transport.events.borrow(),
                        source_events(case)?
                    )
                    .into());
                }
                transport.finish()?;
                if let Ok(result) = result {
                    if is_local(&input.node) {
                        assert_eq!(result.local_value, Some(case["output"][0].clone()));
                    }
                    assert!(transport.deadlines.iter().zip(&transport.heads).all(
                        |(timeout, head)| *timeout
                            == Some(if head.method == HttpMethod::Get {
                                1_024_000
                            } else {
                                3_600_000
                            })
                    ));
                    assert_eq!(transport.reservations.len(), result.responses.len());
                }
            }
        }
        Ok(())
    }
    impl RecraftTransport for crate::common::ScriptedActuator {
        fn retry_after_delay(&mut self, _: &str, _: u64) -> Result<u64, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn retry_request(
            &mut self,
            _: StreamHandle,
            _: u64,
        ) -> Result<StreamHandle, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn delay(&mut self, _: StreamHandle, _: u64) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
    }
    fn source_case(node: &str) -> Result<Value, Box<dyn Error>> {
        source_oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| case["input"]["node"] == node)
            .cloned()
            .ok_or_else(|| "missing node".into())
    }
    #[test]
    fn exact_source_badges_are_informational_not_budget_authority() -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        for case in source["cases"].as_array().ok_or("cases")? {
            let input: Input = serde_json::from_value(case["input"].clone())?;
            let badge = source_price_badge(&input)?;
            if let Some(badge) = badge {
                assert_eq!(json!(badge.expression), source["price_badges"][&input.node]);
            } else {
                assert!(is_local(&input.node));
            }
        }
        let case = source_oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| {
                case["input"]["node"] == "RecraftImageToImageNode"
                    && case["input"]["images"][0]["batch"] == 2
            })
            .cloned()
            .ok_or("batch case")?;
        let (input, mut transport, mut input_owner, mut output_owner) = scripted(&case)?;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                1,
                &mut transport,
                &mut input_owner,
                &mut output_owner
            ),
            Err(RecraftError::Transport(ConformanceError::InvalidCost))
        ));
        assert_eq!(transport.reservations, vec![(0, 2, 1)]);
        assert_eq!(transport.frames, vec![(1, 2)]);
        assert!(
            transport
                .active
                .as_ref()
                .is_some_and(|(_, _, _, bytes)| bytes.is_empty())
        );
        Ok(())
    }
    #[test]
    fn canonical_types_and_unbound_owners_fail_without_publication() -> Result<(), Box<dyn Error>> {
        let mut unbound = crate::common::ScriptedActuator::new(vec![]);
        assert_eq!(
            unbound.reserve_paid_request(
                StreamHandle {
                    invocation: 42,
                    slot: 1,
                    generation: 1
                },
                100,
                0,
                2,
                None
            ),
            Err(ConformanceError::HostFailure)
        );
        struct MissingOwner;
        impl RecraftInputOwner for MissingOwner {}
        impl RecraftOutputOwner for MissingOwner {}
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
        for (node, id) in [
            ("RecraftColorRGB", "sim:custom-recraft-color@1"),
            ("RecraftControls", "sim:custom-recraft-controls@1"),
            (
                "RecraftStyleV3RealisticImage",
                "sim:custom-recraft-style-v3@1",
            ),
            ("RecraftTextToVectorNode", "sim:svg@1"),
        ] {
            assert_eq!(expected_outputs(node)?[0].type_id.to_string(), id);
        }
        let case = source_case("RecraftImageToImageNode")?;
        let (input, mut transport, _, mut output_owner) = scripted(&case)?;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut MissingOwner,
                &mut output_owner
            ),
            Err(RecraftError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        let (input, mut transport, mut input_owner, _) = scripted(&case)?;
        assert!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut input_owner,
                &mut MissingOwner
            )
            .is_err()
        );
        let result = InvocationResult {
            node: "RecraftColorRGB".into(),
            local_value: Some(json!({})),
            style_id: None,
            responses: vec![],
            media: vec![],
        };
        let mut sink = Sink(0);
        assert!(
            materialize_and_publish(&result, &mut transport, &mut MissingOwner, &mut sink).is_err()
        );
        assert_eq!(sink.0, 0);
        Ok(())
    }
    #[test]
    fn style_cap_stops_later_preparations_and_preserves_first_frame_contract()
    -> Result<(), Box<dyn Error>> {
        let mut input: Input =
            serde_json::from_value(source_case("RecraftCreateStyleNode")?["input"].clone())?;
        let first = input.images.first_mut().ok_or("image")?;
        first.batch = 4;
        let mut second = first.clone();
        second.role = "imageSecond".into();
        let mut third = first.clone();
        third.role = "imageThird".into();
        input.images.extend([second, third]);
        let plans = request_plans(&input)?;
        let plan = plans.first().ok_or("plan")?;
        assert!(plan.media.iter().all(|media| media.batch_index == 0
            && media.purpose == PreparationPurpose::StyleFirstFrameWebP
            && media.total_pixels == 2048 * 2048));
        let parts = plan
            .media
            .iter()
            .map(|requirement| PreparedMedia {
                requirement: requirement.clone(),
                bytes: vec![1; 3 * 1024 * 1024],
            })
            .collect::<Vec<_>>();
        let mut input_owner = InputOwner {
            parts: parts.into(),
            prepared: vec![],
        };
        let events = Rc::new(RefCell::new(vec![]));
        let mut transport = TestTransport::new(vec![], events.clone());
        let mut output_owner = OutputOwner {
            events,
            frames: 0,
            fail_frame: false,
        };
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut input_owner,
                &mut output_owner
            ),
            Err(RecraftError::Transport(ConformanceError::BodyLimit))
        ));
        assert_eq!(input_owner.prepared.len(), 2);
        assert_eq!(input_owner.parts.len(), 1);
        assert!(transport.heads.is_empty());
        Ok(())
    }
    #[test]
    fn aggregate_owner_and_frame_failures_stop_later_paid_requests() -> Result<(), Box<dyn Error>> {
        let case = source_oracle()?["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .find(|case| {
                case["input"]["node"] == "RecraftImageToImageNode"
                    && case["input"]["images"][0]["batch"] == 2
            })
            .cloned()
            .ok_or("batch case")?;
        let (input, mut transport, mut input_owner, mut output_owner) = scripted(&case)?;
        output_owner.fail_frame = true;
        assert!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut input_owner,
                &mut output_owner
            )
            .is_err()
        );
        assert_eq!(transport.reservations.len(), 1);
        assert!(transport.frames.is_empty());
        assert_eq!(
            transport
                .heads
                .iter()
                .filter(|head| head.method == HttpMethod::Post)
                .count(),
            1
        );
        let (input, mut transport, mut input_owner, mut output_owner) = scripted(&case)?;
        transport.reject_cost = true;
        assert!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut input_owner,
                &mut output_owner
            )
            .is_err()
        );
        assert!(
            transport
                .active
                .as_ref()
                .is_some_and(|(_, _, _, bytes)| bytes.is_empty())
        );
        Ok(())
    }
    #[test]
    fn source_retry_download_timeout_and_stale_handles_are_checked() -> Result<(), Box<dyn Error>> {
        for (status, download, retries) in [
            (500, false, 1),
            (429, false, 16),
            (500, true, 5),
            (429, true, 5),
        ] {
            let events = Rc::new(RefCell::new(vec![]));
            let path = if download {
                "https://source.fixture.invalid/retry"
            } else {
                "/proxy/recraft/image_generation"
            };
            let script = (0..retries + 1)
                .map(|_| Response {
                    path: path.into(),
                    method: HttpMethod::Get,
                    body: vec![],
                    status,
                    bytes: vec![],
                    headers: vec![],
                    receipt: b"receipt".to_vec(),
                })
                .collect();
            let mut transport = TestTransport::new(script, events);
            let mut session = Session::default();
            let head = request_head(path, HttpMethod::Get, credentials(), download)?;
            assert!(matches!(
                exchange_with_policy(
                    &head,
                    &[],
                    MAXIMUM_MEDIA_BYTES,
                    if download {
                        DOWNLOAD_RETRY
                    } else {
                        GENERATION_RETRY
                    },
                    &mut session,
                    &mut transport
                ),
                Err(RecraftError::Http { .. })
            ));
            assert_eq!(transport.delays.len(), retries);
            transport.finish()?;
        }
        let case = source_case("RecraftTextToImageNode")?;
        let (input, mut transport, mut input_owner, mut output_owner) = scripted(&case)?;
        transport.responses.front_mut().ok_or("response")?.status = 500;
        transport.stale_retry = true;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut input_owner,
                &mut output_owner
            ),
            Err(RecraftError::Transport(ConformanceError::InvalidHandle))
        ));
        let mut session = Session::default();
        let handle = StreamHandle {
            invocation: 42,
            slot: 1,
            generation: 1,
        };
        session.retain(handle)?;
        assert!(session.retain(handle).is_err());
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
    fn cancellation_worker_loss_response_receipt_and_body_limits() -> Result<(), Box<dyn Error>> {
        let case = source_case("RecraftTextToImageNode")?;
        for error in [ConformanceError::Cancelled, ConformanceError::WorkerLost] {
            let (input, mut transport, mut input_owner, mut output_owner) = scripted(&case)?;
            transport.failure = Some(error.clone());
            assert!(
                matches!(invoke(&input,credentials(),100,&mut transport,&mut input_owner,&mut output_owner),Err(RecraftError::Transport(actual)) if actual==error)
            );
            assert!(transport.heads.is_empty());
        }
        for mode in 0..4 {
            let (input, mut transport, mut input_owner, mut output_owner) = scripted(&case)?;
            let response = transport.responses.front_mut().ok_or("response")?;
            match mode {
                0 => response.bytes = b"{}".to_vec(),
                1 => response.receipt.clear(),
                2 => response.bytes = vec![0; MAXIMUM_CHUNK_BYTES + 1],
                _ => {
                    response.headers = vec![Header {
                        name: "X-Test".into(),
                        value: "bad\nheader".into(),
                    }]
                }
            }
            assert!(
                invoke(
                    &input,
                    credentials(),
                    100,
                    &mut transport,
                    &mut input_owner,
                    &mut output_owner
                )
                .is_err()
            );
            assert_eq!(output_owner.frames, 0);
        }
        for url in [
            "http://x/y",
            "https://user:secret@x/y",
            "//x/y",
            "https://x/y#fragment",
            "https://x/ bad",
        ] {
            assert!(endpoint(url).is_err());
        }
        assert!(download_entries(&json!({"data":[]}), false).is_err());
        assert!(decode_hex("ffxz").is_err());
        assert!(decode_hex("f").is_err());
        Ok(())
    }
    #[test]
    fn optional_models_mask_and_auth_remain_canonical_requirements() -> Result<(), Box<dyn Error>> {
        let case = source_case("RecraftImageInpaintingNode")?;
        let input: Input = serde_json::from_value(case["input"].clone())?;
        let plans = request_plans(&input)?;
        let mask = plans
            .first()
            .and_then(|plan| plan.media.get(1))
            .ok_or("mask")?;
        assert_eq!(
            mask.purpose,
            PreparationPurpose::MaskNearestExactThresholdHalf
        );
        assert_eq!((mask.width, mask.height, mask.channels), (2, 2, 1));
        let head = request_head(
            "/proxy/recraft/image_generation",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.auth_token"));
        let head = request_head(
            "/proxy/recraft/image_generation",
            HttpMethod::Post,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(head.secret_id.as_deref(), Some("sim_org.api_key"));
        assert!(
            request_head(
                "/proxy/recraft/image_generation",
                HttpMethod::Post,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                false
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn short_mask_batch_encodes_second_image_before_failing_without_second_post()
    -> Result<(), Box<dyn Error>> {
        let source = source_oracle()?;
        let case = source["errors"]
            .as_array()
            .ok_or("errors")?
            .iter()
            .find(|case| {
                case["input"]["node"] == "RecraftImageInpaintingNode"
                    && case["input"]["images"][0]["batch"] == 2
                    && case["input"]["mask"]["batch"] == 1
            })
            .ok_or("source late mask failure")?;
        assert_eq!(case["error_type"], "IndexError");
        let (input, mut transport, mut input_owner, mut output_owner) = scripted(case)?;
        assert!(matches!(
            invoke(
                &input,
                credentials(),
                100,
                &mut transport,
                &mut input_owner,
                &mut output_owner
            ),
            Err(RecraftError::Input("Source resized mask batch exhausted"))
        ));
        assert_eq!(
            input_owner
                .prepared
                .iter()
                .map(|requirement| (requirement.field.as_str(), requirement.batch_index))
                .collect::<Vec<_>>(),
            vec![("image", 0), ("mask", 0), ("image", 1)]
        );
        assert_eq!(*transport.events.borrow(), source_events(case)?);
        assert_eq!(transport.reservations.len(), 1);
        assert_eq!(output_owner.frames, 1);
        assert_eq!(transport.frames, vec![(1, 2)]);
        transport.finish()?;
        Ok(())
    }

    const NEGATIVE_GENERATION_NODES: &[&str] = &[
        "RecraftTextToImageNode",
        "RecraftImageToImageNode",
        "RecraftImageInpaintingNode",
        "RecraftTextToVectorNode",
        "RecraftReplaceBackgroundNode",
        "RecraftV4TextToImageNode",
        "RecraftV4TextToVectorNode",
    ];
    #[test]
    fn recraft_empty_negative_containers_match_omitted_source_body_on_all_generation_paths()
    -> Result<(), Box<dyn Error>> {
        for node in NEGATIVE_GENERATION_NODES {
            let case = source_case(node)?;
            let mut input: Input = serde_json::from_value(case["input"].clone())?;
            input.settings["negative_prompt"] = Value::Null;
            let baseline = serde_json::to_vec(&image_body(&input)?)?;
            for empty in [json!([]), json!({}), json!(0.0), json!(-0.0)] {
                input.settings["negative_prompt"] = empty;
                let body = image_body(&input)?;
                assert!(body.get("negative_prompt").is_none());
                assert_eq!(serde_json::to_vec(&body)?, baseline);
            }
        }
        Ok(())
    }
    #[test]
    fn recraft_nonempty_negative_containers_fail_and_raw_strings_remain_exact()
    -> Result<(), Box<dyn Error>> {
        for node in NEGATIVE_GENERATION_NODES {
            let case = source_case(node)?;
            let mut input: Input = serde_json::from_value(case["input"].clone())?;
            for malformed in [
                json!(["nonempty"]),
                json!({"key":"value"}),
                json!(true),
                json!(1),
                json!(1.5),
            ] {
                input.settings["negative_prompt"] = malformed;
                assert!(matches!(
                    image_body(&input),
                    Err(RecraftError::Input("Invalid source negative prompt"))
                ));
            }
            for raw in [" keep negative ", " ", "\u{1c}raw\u{1f}", "\u{200b}"] {
                input.settings["negative_prompt"] = json!(raw);
                assert_eq!(image_body(&input)?["negative_prompt"], json!(raw));
            }
        }
        Ok(())
    }
    #[test]
    fn recraft_empty_negative_containers_execute_existing_transport_without_new_media_owner()
    -> Result<(), Box<dyn Error>> {
        for node in [
            "RecraftTextToImageNode",
            "RecraftV4TextToImageNode",
            "RecraftV4TextToVectorNode",
        ] {
            let mut case = source_case(node)?;
            case["input"]["settings"]["negative_prompt"] = Value::Null;
            for request in case["requests"].as_array_mut().ok_or("requests")? {
                request["body"]
                    .as_object_mut()
                    .ok_or("body")?
                    .remove("negative_prompt");
            }
            for empty in [json!([]), json!({}), json!(0.0), json!(-0.0)] {
                let (mut input, mut transport, mut input_owner, mut output_owner) =
                    scripted(&case)?;
                input.settings["negative_prompt"] = empty;
                let result = invoke(
                    &input,
                    credentials(),
                    1000,
                    &mut transport,
                    &mut input_owner,
                    &mut output_owner,
                )?;
                assert!(!result.responses.is_empty());
                assert_eq!(transport.reservations.len(), result.responses.len());
                assert!(
                    transport
                        .heads
                        .iter()
                        .any(|head| head.method == HttpMethod::Post)
                );
                transport.finish()?;
            }
            let (mut input, mut transport, mut input_owner, mut output_owner) = scripted(&case)?;
            input.settings["negative_prompt"] = json!([]);
            transport.cancelled = true;
            assert!(matches!(
                invoke(
                    &input,
                    credentials(),
                    1000,
                    &mut transport,
                    &mut input_owner,
                    &mut output_owner
                ),
                Err(RecraftError::Transport(ConformanceError::Cancelled))
            ));
            assert!(transport.heads.is_empty());
            assert!(transport.reservations.is_empty());
        }
        Ok(())
    }
}
