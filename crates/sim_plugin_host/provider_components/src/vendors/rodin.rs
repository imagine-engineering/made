use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, MultipartField, Progress, ProviderTransport, RequestHead, ResponseEvent,
    StreamHandle, TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};
pub const NAMESPACE: &str = "zed.sim.provider.rodin";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "e833ab2c68c6c619b4666b269e04e2f3c36a076f6a9e2b44f1bc845ba3614555";
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
        feature_id: "COMFY-NODE-0549",
        node_identifier: "Rodin3D_Detail",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0550",
        node_identifier: "Rodin3D_Gen2",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0551",
        node_identifier: "Rodin3D_Gen25_Image",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0552",
        node_identifier: "Rodin3D_Gen25_Text",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0553",
        node_identifier: "Rodin3D_Regular",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0554",
        node_identifier: "Rodin3D_Sketch",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0555",
        node_identifier: "Rodin3D_Smooth",
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
        feature_id: "COMFY-API-EXT-0148",
        method: "POST",
        path: "/proxy/rodin/api/v2/download",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0149",
        method: "POST",
        path: "/proxy/rodin/api/v2/rodin",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0150",
        method: "POST",
        path: "/proxy/rodin/api/v2/status",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum RodinError {
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
impl fmt::Display for RodinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Http { status, .. } => write!(formatter, "Rodin HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for RodinError {}
impl From<ConformanceError> for RodinError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for RodinError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, RodinError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/rodin/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, RodinError> {
    let claim = NODES
        .iter()
        .find(|claim| claim.node_identifier == input.node)
        .ok_or(RodinError::Input("Unknown Rodin node"))?;
    let oracle = oracle()?;
    let contract = &oracle["contracts"][claim.node_identifier];
    let names = contract["arguments"]
        .as_array()
        .ok_or(RodinError::Input("Missing source argument set"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(RodinError::Input("Unknown Rodin source argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(RodinError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            let default = contract["defaults"]
                .get(name)
                .ok_or(RodinError::Input("Missing Rodin source argument"))?;
            input.arguments.insert(name.into(), default.clone());
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, RodinError> {
    input
        .arguments
        .get(name)
        .ok_or(RodinError::Input("Missing Rodin argument"))
}
fn text(value: &Value) -> Result<&str, RodinError> {
    value
        .as_str()
        .ok_or(RodinError::Input("Expected Rodin string"))
}
fn endpoint(value: &str) -> Result<bool, RodinError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(RodinError::Response("Invalid Rodin response endpoint"));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(RodinError::Response("Invalid Rodin response endpoint"))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(RodinError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(RodinError::Response("Invalid Rodin endpoint authority"));
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
) -> Result<RequestHead, RodinError> {
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
            return Err(RodinError::Input("Missing Comfy API credentials"));
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
    pub source_function: &'static str,
    pub media_type: &'static str,
    pub total_pixels: u64,
    pub output_mode: &'static str,
    pub resampling: &'static str,
}
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
}
pub trait RodinImageOwner {
    fn batch_images(&mut self, image: &Value) -> Result<Vec<Value>, ConformanceError>;
    fn ordered_references(&mut self, value: &Value) -> Result<Vec<Value>, ConformanceError>;
    fn prepare_png(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<PreparedImage, ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct Request {
    pub node: String,
    pub body: Value,
    pub fields: Vec<MultipartField>,
    pub encoding_receipts: Vec<Vec<u8>>,
    pub geometry_format: String,
    pub gen25: bool,
}
fn python_whitespace(value: char) -> bool {
    value.is_whitespace() || matches!(value, '\u{001c}'..='\u{001f}')
}
fn integer(value: &Value) -> Result<i64, RodinError> {
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    let value = value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= i64::MIN as f64 && *value < i64::MAX as f64)
        .ok_or(RodinError::Input("Invalid Rodin integer"))?;
    Ok(value.trunc() as i64)
}
fn api_integer(value: &Value) -> Result<i64, RodinError> {
    if let Some(value) = value.as_bool() {
        return Ok(i64::from(value));
    }
    if value.as_f64().is_some_and(|value| value.fract() != 0.0) {
        return Err(RodinError::Input("Invalid Rodin API integer"));
    }
    integer(value)
}
fn truth(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}
fn present<'a>(value: &'a Value, name: &str) -> Result<&'a Value, RodinError> {
    value
        .get(name)
        .ok_or(RodinError::Input("Missing Rodin source option"))
}
fn quality(label: &str) -> Result<(&'static str, i64), RodinError> {
    let source = oracle()?;
    if let Some(value) = source["quality_options"].get(label) {
        let mode = if value[0] == "Raw" { "Raw" } else { "Quad" };
        return Ok((
            mode,
            value[1]
                .as_i64()
                .ok_or(RodinError::Input("Missing source quality"))?,
        ));
    }
    Ok(("Quad", 18000))
}
fn source_model_body(input: &Input, gen25: bool) -> Result<Value, RodinError> {
    let mut body = serde_json::Map::new();
    if !gen25 {
        let (tier, sketch) = match input.node.as_str() {
            "Rodin3D_Regular" => ("Regular", false),
            "Rodin3D_Detail" => ("Detail", false),
            "Rodin3D_Smooth" => ("Smooth", false),
            "Rodin3D_Sketch" => ("Sketch", true),
            "Rodin3D_Gen2" => ("Gen-2", false),
            _ => return Err(RodinError::Input("Unknown legacy Rodin node")),
        };
        let (mesh, faces) = if sketch {
            ("Quad", 18000)
        } else {
            quality(text(argument(input, "Polygon_count")?)?)?
        };
        body.insert("seed".into(), json!(api_integer(argument(input, "Seed")?)?));
        body.insert("tier".into(), json!(tier));
        body.insert(
            "material".into(),
            if sketch {
                json!("PBR")
            } else {
                json!(text(argument(input, "Material_Type")?)?)
            },
        );
        body.insert("quality_override".into(), json!(faces));
        body.insert("mesh_mode".into(), json!(mesh));
        body.insert(
            "TAPose".into(),
            json!(if input.node == "Rodin3D_Gen2" {
                argument(input, "TAPose")?
                    .as_bool()
                    .ok_or(RodinError::Input("Invalid Rodin TAPose"))?
            } else {
                false
            }),
        );
    } else {
        let mode = argument(input, "mode")?;
        let selected = text(present(mode, "mode")?)?;
        match selected {
            "Regular" => {
                body.insert("tier".into(), json!(text(present(mode, "tier")?)?));
                let label = mode
                    .get("polygon_count")
                    .map(text)
                    .transpose()?
                    .unwrap_or("Default");
                if label != "Default" {
                    let (mesh, faces) = quality(label)?;
                    body.insert("mesh_mode".into(), json!(mesh));
                    body.insert("quality_override".into(), json!(faces));
                }
                if mode.get("creative").is_some_and(truth) {
                    body.insert("geometry_instruct_mode".into(), json!("creative"));
                }
            }
            "Fast" => {
                body.insert("tier".into(), json!(text(present(mode, "tier")?)?));
                body.insert("mesh_mode".into(), json!("Raw"));
                body.insert(
                    "quality_override".into(),
                    json!(integer(present(mode, "mesh_faces")?)?),
                );
            }
            "Extreme-High" => {
                body.insert("tier".into(), json!("Gen-2.5-Extreme-High"));
                body.insert(
                    "mesh_mode".into(),
                    json!(text(present(mode, "mesh_mode")?)?),
                );
                body.insert(
                    "quality_override".into(),
                    json!(integer(present(mode, "mesh_faces")?)?),
                );
                for (field, target) in [
                    ("creative", "geometry_instruct_mode"),
                    ("is_micro", "is_micro"),
                ] {
                    if mode.get(field).is_some_and(truth) {
                        body.insert(
                            target.into(),
                            if field == "creative" {
                                json!("creative")
                            } else {
                                json!(true)
                            },
                        );
                    }
                }
            }
            _ => return Err(RodinError::Input("Missing Rodin source tier")),
        }
        if input.node == "Rodin3D_Gen25_Text" {
            body.insert("prompt".into(), json!(text(argument(input, "prompt")?)?));
        }
        for field in ["seed", "material", "geometry_file_format", "texture_mode"] {
            let value = argument(input, field)?;
            if value.is_null() || field == "texture_mode" && value == "Default" {
                continue;
            }
            body.insert(
                field.into(),
                if field == "seed" {
                    json!(api_integer(value)?)
                } else {
                    json!(text(value)?)
                },
            );
        }
        let dimensions = [
            api_integer(argument(input, "bbox_width")?)?,
            api_integer(argument(input, "bbox_height")?)?,
            api_integer(argument(input, "bbox_length")?)?,
        ];
        if dimensions.iter().all(|value| *value != 0) {
            body.insert("bbox_condition".into(), json!(dimensions));
        }
        let height = api_integer(argument(input, "height_cm")?)?;
        if height != 0 {
            body.insert("height".into(), json!(height));
        }
        for field in ["TAPose", "hd_texture", "texture_delight"] {
            if argument(input, field)?
                .as_bool()
                .ok_or(RodinError::Input("Invalid Rodin flag"))?
            {
                body.insert(field.into(), json!(true));
            }
        }
        if input.node == "Rodin3D_Gen25_Image"
            && argument(input, "use_original_alpha")?
                .as_bool()
                .ok_or(RodinError::Input("Invalid Rodin alpha flag"))?
        {
            body.insert("use_original_alpha".into(), json!(true));
        }
        if argument(input, "addon_highpack")?
            .as_bool()
            .ok_or(RodinError::Input("Invalid Rodin addon"))?
        {
            body.insert("addons".into(), json!(["HighPack"]));
        }
    }
    Ok(Value::Object(body))
}
fn preflight(input: &Input) -> Result<(), RodinError> {
    if input.node == "Rodin3D_Gen25_Text" {
        let prompt = text(argument(input, "prompt")?)?;
        if !(1..=2500).contains(&prompt.trim_matches(python_whitespace).chars().count()) {
            return Err(RodinError::Input("Invalid Rodin prompt length"));
        }
    }
    Ok(())
}
const LEGACY_FIELDS: &[&str] = &[
    "seed",
    "tier",
    "material",
    "quality_override",
    "mesh_mode",
    "TAPose",
];
const GEN25_FIELDS: &[&str] = &[
    "tier",
    "prompt",
    "seed",
    "material",
    "geometry_file_format",
    "texture_mode",
    "mesh_mode",
    "quality_override",
    "geometry_instruct_mode",
    "bbox_condition",
    "height",
    "TAPose",
    "hd_texture",
    "texture_delight",
    "is_micro",
    "use_original_alpha",
    "preview_render",
    "addons",
];
fn multipart_fields(body: &Value, gen25: bool) -> Result<Vec<MultipartField>, RodinError> {
    let mut fields = vec![];
    for name in if gen25 { GEN25_FIELDS } else { LEGACY_FIELDS } {
        let Some(value) = body.get(*name).filter(|value| !value.is_null()) else {
            continue;
        };
        let values = if gen25 {
            value
                .as_array()
                .cloned()
                .unwrap_or_else(|| vec![value.clone()])
        } else {
            vec![value.clone()]
        };
        for value in values {
            let serialized = match value {
                Value::Bool(value) => {
                    if gen25 {
                        value.to_string()
                    } else if value {
                        "True".into()
                    } else {
                        "False".into()
                    }
                }
                Value::String(value) => value,
                Value::Number(value) => value.to_string(),
                _ => return Err(RodinError::Input("Unsupported source multipart scalar")),
            };
            fields.push(MultipartField {
                name: (*name).into(),
                file_name: None,
                media_type: Some("text/plain; charset=utf-8".into()),
                bytes: serialized.into_bytes(),
            });
        }
    }
    Ok(fields)
}
pub fn request(input: &Input, owner: &mut dyn RodinImageOwner) -> Result<Request, RodinError> {
    let input = normalized(input)?;
    preflight(&input)?;
    let gen25 = input.node.starts_with("Rodin3D_Gen25");
    let mut images = vec![];
    if input.node == "Rodin3D_Gen25_Image" {
        let references = owner
            .ordered_references(argument(&input, "images")?)?
            .into_iter()
            .filter(|value| !value.is_null())
            .collect::<Vec<_>>();
        if references.is_empty() {
            return Err(RodinError::Input(
                "Rodin Gen-2.5 Image-to-3D requires at least one image.",
            ));
        }
        for reference in references {
            images.extend(owner.batch_images(&reference)?);
        }
    } else if !gen25 {
        images = owner.batch_images(argument(&input, "Images")?)?;
    }
    if images.len() > 5 {
        return Err(RodinError::Input(
            "Rodin source accepts at most five images",
        ));
    }
    let body = source_model_body(&input, gen25)?;
    let mut fields = multipart_fields(&body, gen25)?;
    let mut receipts = vec![];
    for image in images {
        let encoded = owner.prepare_png(
            &image,
            &ImagePolicy {
                source_function: "tensor_to_filelike",
                media_type: "image/png",
                total_pixels: 4194304,
                output_mode: "RGB",
                resampling: "PIL.LANCZOS",
            },
        )?;
        if encoded.bytes.is_empty()
            || encoded.bytes.len() > MAXIMUM_MEDIA_BYTES
            || encoded.receipt.is_empty()
            || encoded.receipt.len() > MAXIMUM_RECEIPT_BYTES
        {
            return Err(ConformanceError::InvalidResponse.into());
        }
        fields.push(MultipartField {
            name: "images".into(),
            file_name: Some("images".into()),
            media_type: Some("application/octet-stream".into()),
            bytes: encoded.bytes,
        });
        receipts.push(encoded.receipt);
    }
    let geometry_format = if gen25 {
        text(argument(&input, "geometry_file_format")?)?.to_owned()
    } else {
        "glb".into()
    };
    Ok(Request {
        node: input.node,
        body,
        fields,
        encoding_receipts: receipts,
        geometry_format,
        gen25,
    })
}
#[derive(Clone, Debug)]
pub struct CanonicalMultipartEncoder {
    pub boundary: String,
}
impl CanonicalMultipartEncoder {
    pub fn encode(&self, fields: &[MultipartField]) -> Result<(Vec<u8>, String), ConformanceError> {
        let mut limits = crate::common::TransportLimits::conformance();
        limits.maximum_body_bytes = MAXIMUM_REQUEST_BYTES;
        Ok((
            MultipartField::encode_checked(fields, &self.boundary, limits)?,
            format!("multipart/form-data; boundary={}", self.boundary),
        ))
    }
}
pub fn estimated_cost_microunits(input: &Input) -> Result<u64, RodinError> {
    let input = normalized(input)?;
    if !input.node.starts_with("Rodin3D_Gen25") {
        return Ok(400_000);
    }
    let mode = text(present(argument(&input, "mode")?, "mode")?)?;
    let addon = argument(&input, "addon_highpack")?
        .as_bool()
        .ok_or(RodinError::Input("Invalid Rodin addon"))?;
    Ok((if mode == "extreme-high" {
        1_500_000
    } else {
        750_000
    }) + if addon { 800_000 } else { 0 })
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResponsePolicy {
    pub timeout_milliseconds: Option<u64>,
    pub monitor_progress: bool,
    pub wait_label: Option<&'static str>,
    pub final_label: Option<&'static str>,
    pub source_read_chunk_bytes: Option<usize>,
}
pub trait RodinTransport: ProviderTransport {
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
    fn set_content_type(
        &mut self,
        handle: StreamHandle,
        content_type: &str,
    ) -> Result<(), ConformanceError>;
    fn poll_delay(
        &mut self,
        handle: StreamHandle,
        milliseconds: u64,
    ) -> Result<(), ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct Submission {
    pub uuid: Option<String>,
    pub subscription_key: String,
}
fn optional_string(value: &Value, name: &str) -> Result<Option<String>, RodinError> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        _ => Err(RodinError::Response("Invalid Rodin response string")),
    }
}
pub fn submission(bytes: &[u8], gen25: bool) -> Result<Submission, RodinError> {
    let value: Value = serde_json::from_slice(bytes)?;
    if !value.is_object() {
        return Err(RodinError::Response("Invalid Rodin generation response"));
    }
    for field in ["message", "prompt", "submit_time"] {
        optional_string(&value, field)?;
    }
    let uuid = optional_string(&value, "uuid")?;
    let jobs = value["jobs"]
        .as_object()
        .ok_or(RodinError::Response("Missing Rodin jobs"))?;
    if !jobs
        .get("uuids")
        .and_then(Value::as_array)
        .is_some_and(|values| values.iter().all(Value::is_string))
    {
        return Err(RodinError::Response("Missing Rodin job UUIDs"));
    }
    let key = jobs
        .get("subscription_key")
        .and_then(Value::as_str)
        .ok_or(RodinError::Response("Missing Rodin subscription key"))?
        .to_owned();
    if gen25 && (uuid.as_ref().is_none_or(String::is_empty) || key.is_empty()) {
        return Err(RodinError::Response("Rodin Gen-2.5 submit failed"));
    }
    Ok(Submission {
        uuid,
        subscription_key: key,
    })
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PollStatus {
    pub complete: bool,
    pub progress: Option<u64>,
    pub suppressed_status_exception: bool,
}
pub fn poll_status(bytes: &[u8]) -> Result<PollStatus, RodinError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let jobs = value["jobs"]
        .as_array()
        .ok_or(RodinError::Response("Missing Rodin status jobs"))?;
    if jobs.len() > 4096 {
        return Err(ConformanceError::BodyLimit.into());
    }
    let mut done = 0usize;
    let mut failed = false;
    for job in jobs {
        if job["uuid"].as_str().is_none() {
            return Err(RodinError::Response("Missing Rodin status job UUID"));
        }
        match job["status"].as_str() {
            Some("Done") => done += 1,
            Some("Failed") => failed = true,
            Some("Generating" | "Waiting") => {}
            _ => return Err(RodinError::Response("Invalid Rodin status enum")),
        }
    }
    Ok(PollStatus {
        complete: done == jobs.len(),
        progress: (!jobs.is_empty()).then(|| (done as f64 / jobs.len() as f64 * 100.0) as u64),
        suppressed_status_exception: failed,
    })
}
#[derive(Clone, Debug)]
pub struct Resource {
    pub url: String,
    pub name: String,
}
pub fn resources(bytes: &[u8]) -> Result<Vec<Resource>, RodinError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let list = value["list"]
        .as_array()
        .ok_or(RodinError::Response("Missing Rodin download list alias"))?;
    if list.len() > 256 {
        return Err(ConformanceError::BodyLimit.into());
    }
    list.iter()
        .map(|item| {
            Ok(Resource {
                url: text(&item["url"])?.to_owned(),
                name: text(&item["name"])?.to_owned(),
            })
        })
        .collect()
}
#[derive(Clone, Debug)]
pub struct DownloadedFile {
    pub resource: Resource,
    pub bytes: Vec<u8>,
    pub receipt: Vec<u8>,
    pub handle: StreamHandle,
    pub source_format: Option<String>,
    pub compatibility_path: Option<String>,
    pub fallback: bool,
}
pub trait RodinFileOwner {
    fn complete_download(&mut self, file: &DownloadedFile) -> Result<(), ConformanceError>;
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub submission: Submission,
    pub generation_bytes: Vec<u8>,
    pub generation_receipt: Vec<u8>,
    pub status_bytes: Vec<u8>,
    pub status_receipt: Vec<u8>,
    pub list_bytes: Vec<u8>,
    pub list_receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub files: Vec<DownloadedFile>,
    pub selected_file: Option<usize>,
    pub compatibility_model_path: Option<String>,
}
fn extension(name: &str) -> String {
    let basename = name.rsplit('/').next().unwrap_or(name);
    basename
        .rsplit_once('.')
        .filter(|(prefix, _)| prefix.chars().any(|character| character != '.'))
        .map(|(_, extension)| extension.to_lowercase())
        .unwrap_or_default()
}
fn previewable(format: &str) -> bool {
    matches!(format, "glb" | "obj" | "fbx" | "stl" | "gltf")
}
struct DownloadPlan<'a> {
    format: Option<String>,
    save: bool,
    fallback: bool,
    folder: &'a str,
    credentials: CredentialAvailability,
}
fn downloaded(
    resource: &Resource,
    plan: DownloadPlan<'_>,
    transport: &mut dyn RodinTransport,
    owner: &mut dyn RodinFileOwner,
) -> Result<DownloadedFile, RodinError> {
    if endpoint(&resource.url)? {
        return Err(RodinError::Response(
            "Relative Rodin download requires source cls",
        ));
    }
    let head = request_head(&resource.url, HttpMethod::Get, plan.credentials, true)?;
    let handle = transport.start_request(&head)?;
    let (bytes, receipt, handle, _) = exchange_existing(
        &head,
        &[],
        None,
        MAXIMUM_MEDIA_BYTES,
        DOWNLOAD_RETRY,
        handle,
        transport,
    )?;
    let file = DownloadedFile {
        resource: resource.clone(),
        bytes,
        receipt,
        handle,
        source_format: plan.format,
        compatibility_path: plan
            .save
            .then(|| format!("{}/{}", plan.folder, resource.name)),
        fallback: plan.fallback,
    };
    transport.check_cancelled(Some(handle))?;
    owner.complete_download(&file)?;
    Ok(file)
}

pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn RodinTransport,
    image: &mut dyn RodinImageOwner,
    file_owner: &mut dyn RodinFileOwner,
    encoder: &CanonicalMultipartEncoder,
) -> Result<ResultValue, RodinError> {
    let input = normalized(input)?;
    transport.check_cancelled(None)?;
    preflight(&input)?;
    let mut head = request_head(
        "/proxy/rodin/api/v2/rodin",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    let handle = transport.start_request(&head)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "rodin.generation.maximum".into(),
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
    let (body, content_type) = encoder.encode(&plan.fields)?;
    if let Some(header) = head
        .headers
        .iter_mut()
        .find(|header| header.name.eq_ignore_ascii_case("Content-Type"))
    {
        header.value = content_type;
    } else {
        return Err(ConformanceError::InvalidRequest.into());
    }
    let (generation_bytes, generation_receipt, _, _) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    let submission = submission(&generation_bytes, plan.gen25)?;
    let status_head = request_head(
        "/proxy/rodin/api/v2/status",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    let status_body = serde_json::to_vec(&json!({"subscription_key":submission.subscription_key}))?;
    let mut status_result = None;
    let mut last_progress = None;
    let mut progress_sequence = 0u64;
    for _ in 0..480 {
        transport.check_cancelled(None)?;
        let handle = transport.start_request(&status_head)?;
        let (bytes, receipt, handle, _) = exchange_existing(
            &status_head,
            &status_body,
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            handle,
            transport,
        )?;
        let status = poll_status(&bytes)?;
        if let Some(value) = status
            .progress
            .filter(|value| Some(*value) != last_progress)
        {
            transport.report_progress(
                handle,
                &Progress {
                    sequence: progress_sequence,
                    completed: value,
                    total: 100,
                    message: None,
                },
            )?;
            progress_sequence += 1;
            last_progress = Some(value);
        }
        if status.complete {
            if last_progress != Some(100) {
                transport.report_progress(
                    handle,
                    &Progress {
                        sequence: progress_sequence,
                        completed: 100,
                        total: 100,
                        message: None,
                    },
                )?;
            }
            status_result = Some((bytes, receipt));
            break;
        }
        transport.check_cancelled(Some(handle))?;
        transport.poll_delay(handle, 5000)?;
    }
    let (status_bytes, status_receipt) = status_result.ok_or(ConformanceError::PollLimit)?;
    let uuid = submission.uuid.as_deref().ok_or(RodinError::Response(
        "Missing Rodin task UUID before download request",
    ))?;
    let list_head = request_head(
        "/proxy/rodin/api/v2/download",
        HttpMethod::Post,
        credentials,
        false,
    )?;
    let list_body = serde_json::to_vec(&json!({"task_uuid":uuid}))?;
    let handle = transport.start_request(&list_head)?;
    let (list_bytes, list_receipt, _, _) = exchange_existing(
        &list_head,
        &list_body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        LIST_RETRY,
        handle,
        transport,
    )?;
    let resources = resources(&list_bytes)?;
    let folder = format!(
        "{}{}",
        if plan.gen25 {
            "Rodin3D_Gen25_"
        } else {
            "Rodin3D_"
        },
        uuid
    );
    let target = plan
        .geometry_format
        .to_lowercase()
        .trim_start_matches('.')
        .to_owned();
    let mut files = vec![];
    let mut selected = None;
    let mut path = None;
    for resource in &resources {
        let format = extension(&resource.name);
        let choose = if plan.gen25 {
            selected.is_none() && format == target && previewable(&format)
        } else {
            resource.name.to_lowercase().ends_with(".glb")
        };
        let file = downloaded(
            resource,
            DownloadPlan {
                format: choose.then_some(if plan.gen25 { format } else { "glb".into() }),
                save: true,
                fallback: false,
                folder: &folder,
                credentials,
            },
            transport,
            file_owner,
        )?;
        if choose {
            selected = Some(files.len());
            if !plan.gen25 {
                path = file.compatibility_path.clone();
            }
        }
        files.push(file);
    }
    if plan.gen25
        && selected.is_none()
        && let Some(resource) = resources
            .iter()
            .find(|resource| previewable(&extension(&resource.name)))
    {
        let file = downloaded(
            resource,
            DownloadPlan {
                format: Some(extension(&resource.name)),
                save: false,
                fallback: true,
                folder: &folder,
                credentials,
            },
            transport,
            file_owner,
        )?;
        selected = Some(files.len());
        files.push(file);
    }
    Ok(ResultValue {
        plan,
        submission,
        generation_bytes,
        generation_receipt,
        status_bytes,
        status_receipt,
        list_bytes,
        list_receipt,
        approval_receipt: approval.receipt,
        files,
        selected_file: selected,
        compatibility_model_path: path,
    })
}
pub trait RodinMaterializer {
    fn materialize(&mut self, result: &ResultValue)
    -> Result<Vec<EncodedOutput>, ConformanceError>;
}
fn expected_outputs(node: &str) -> Result<Vec<ExpectedOutput>, RodinError> {
    let registry =
        TypeRegistry::built_in().map_err(|_| RodinError::Response("Missing canonical registry"))?;
    let types = if node.starts_with("Rodin3D_Gen25") {
        vec!["FILE_3D"]
    } else {
        vec!["STRING", "FILE_3D_GLB"]
    };
    types
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let type_id = registry
                .resolve(name)
                .map_err(|_| RodinError::Response("Missing canonical Rodin output type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| RodinError::Response("Missing canonical Rodin output family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect()
}
pub struct MaterializationOwners<'a> {
    pub image: &'a mut dyn RodinImageOwner,
    pub file: &'a mut dyn RodinFileOwner,
    pub result: &'a mut dyn RodinMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    ceiling_microunits: u64,
    transport: &mut dyn RodinTransport,
    owners: &mut MaterializationOwners<'_>,
    encoder: &CanonicalMultipartEncoder,
) -> Result<(), RodinError> {
    let result = invoke(
        input,
        credentials,
        ceiling_microunits,
        transport,
        owners.image,
        owners.file,
        encoder,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.result.materialize(&result)?;
    let batch = validate_outputs(
        &expected_outputs(&result.plan.node)?,
        outputs,
        MAXIMUM_MEDIA_BYTES,
    )
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
    multipart: bool,
    response: ResponsePolicy,
}
const GENERATION_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 3,
    rate_limit: 16,
    backoff: 2.0,
    download: false,
    multipart: true,
    response: ResponsePolicy {
        timeout_milliseconds: Some(3_600_000),
        monitor_progress: true,
        wait_label: Some("Waiting for server"),
        final_label: Some("Completed"),
        source_read_chunk_bytes: None,
    },
};
const POLL_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 10,
    rate_limit: 16,
    backoff: 1.4,
    download: false,
    multipart: false,
    response: ResponsePolicy {
        timeout_milliseconds: Some(120_000),
        monitor_progress: false,
        wait_label: Some("Checking"),
        final_label: None,
        source_read_chunk_bytes: None,
    },
};
const LIST_RETRY: RetryPolicy = RetryPolicy {
    multipart: false,
    response: ResponsePolicy {
        monitor_progress: false,
        ..GENERATION_RETRY.response
    },
    ..GENERATION_RETRY
};
const DOWNLOAD_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 5,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
    multipart: false,
    response: ResponsePolicy {
        timeout_milliseconds: None,
        monitor_progress: false,
        wait_label: None,
        final_label: None,
        source_read_chunk_bytes: Some(1024 * 1024),
    },
};
type BinaryResponse = (Vec<u8>, Vec<u8>, StreamHandle, Vec<Header>);
fn exchange_existing(
    _head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    mut handle: StreamHandle,
    transport: &mut dyn RodinTransport,
) -> Result<BinaryResponse, RodinError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        transport.configure_response(handle, &policy.response)?;
        if policy.multipart {
            let content_type = _head
                .headers
                .iter()
                .find(|header| header.name.eq_ignore_ascii_case("Content-Type"))
                .ok_or(ConformanceError::InvalidRequest)?;
            transport.set_content_type(handle, &content_type.value)?;
        }
        match exchange_on_handle(handle, body, cost, maximum_bytes, transport) {
            Err(RodinError::Http {
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
                    return Err(RodinError::Http {
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
    transport: &mut dyn RodinTransport,
) -> Result<BinaryResponse, RodinError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "rodin.generation.maximum".into(),
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
                    return Err(RodinError::Http {
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
impl RodinImageOwner for SourceProbeOwner {
    fn batch_images(&mut self, image: &Value) -> Result<Vec<Value>, ConformanceError> {
        let shape = image["shape"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if shape.len() != 4 {
            return Ok(vec![image.clone()]);
        }
        let count = shape
            .first()
            .and_then(Value::as_u64)
            .ok_or(ConformanceError::InvalidRequest)?;
        if count > 128 {
            return Err(ConformanceError::BodyLimit);
        }
        let slot = image["native_slot"]
            .as_str()
            .ok_or(ConformanceError::InvalidRequest)?;
        let pixels = image["pixels"].clone();
        Ok((0..count).map(|index|json!({"native_slot":format!("{slot}:{index}"),"shape":shape.iter().skip(1).cloned().collect::<Vec<_>>(),"pixels":pixels})).collect())
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
    fn prepare_png(
        &mut self,
        image: &Value,
        policy: &ImagePolicy,
    ) -> Result<PreparedImage, ConformanceError> {
        if self.cancelled {
            return Err(ConformanceError::Cancelled);
        }
        if self.fail {
            return Err(ConformanceError::HostFailure);
        }
        let source = oracle().map_err(|_| ConformanceError::InvalidRequest)?;
        let bytes_hex = source["cases"]
            .as_array()
            .and_then(|cases| {
                cases
                    .iter()
                    .flat_map(|case| case["media"].as_array().into_iter().flatten())
                    .next()
            })
            .and_then(|value| value["bytes_hex"].as_str())
            .ok_or(ConformanceError::InvalidResponse)?;
        let bytes = decode_hex(bytes_hex).map_err(|_| ConformanceError::InvalidResponse)?;
        self.operations.push(json!({"operation":policy.source_function,"image":image,"mime_type":policy.media_type,"total_pixels":policy.total_pixels,"resampling":policy.resampling,"bytes_hex":bytes_hex}));
        Ok(PreparedImage {
            bytes,
            receipt: b"source-probe-only-not-native".to_vec(),
        })
    }
}
fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    if !value.is_ascii() || !value.len().is_multiple_of(2) {
        return Err("invalid hex".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            Ok(u8::from_str_radix(
                value.get(index..index + 2).ok_or("hex")?,
                16,
            )?)
        })
        .collect()
}
fn input_from(case: &Value) -> Result<Input, Box<dyn Error>> {
    Ok(serde_json::from_value(case["input"].clone())?)
}
fn compare_case(case: &Value) -> Result<(), Box<dyn Error>> {
    let input = input_from(case)?;
    let mut owner = SourceProbeOwner::default();
    let plan = request(&input, &mut owner)?;
    let expected = case["requests"]
        .as_array()
        .and_then(|values| values.first())
        .ok_or("request")?;
    let source = oracle()?;
    let encoder = CanonicalMultipartEncoder {
        boundary: source["multipart_boundary"]
            .as_str()
            .ok_or("boundary")?
            .to_owned(),
    };
    let (bytes, _) = encoder.encode(&plan.fields)?;
    if plan.body != expected["body"]
        || json!(owner.operations) != case["media"]
        || bytes != decode_hex(expected["multipart_body_hex"].as_str().ok_or("wire")?)?
    {
        return Err(format!("Rodin request/media/wire differs for {}", input.node).into());
    }
    Ok(())
}
fn source_conformance(fixture: &str) -> Result<(), Box<dyn Error>> {
    let value: Value = serde_json::from_str(fixture)?;
    if value
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"scope":"source request and multipart/poll/download transport; canonical native model and guest integration pending"})
    {
        return Err("Rodin claims differ".into());
    }
    let source = oracle()?;
    if source["source_sha256"]["comfy_api_nodes/nodes_rodin.py"] != SOURCE_SHA256 {
        return Err("Rodin source differs".into());
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"].as_array().ok_or("cases")? {
        compare_case(case)?;
        covered.insert(input_from(case)?.node);
    }
    for case in source["preflight_errors"].as_array().ok_or("errors")? {
        let mut owner = SourceProbeOwner::default();
        if request(&input_from(case)?, &mut owner).is_ok()
            || json!(owner.operations) != case["media"]
        {
            return Err("Rodin source preflight/order differs".into());
        }
    }
    if covered
        != NODES
            .iter()
            .map(|claim| claim.node_identifier.to_owned())
            .collect()
    {
        return Err("Rodin node coverage differs".into());
    }
    Ok(())
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    source_conformance(fixture).map_err(|_| ConformanceError::InvalidRequest)
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
        mime: Vec<(StreamHandle, String)>,
        poll_delays: Vec<u64>,
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
    impl RodinTransport for Transport {
        fn set_content_type(
            &mut self,
            handle: StreamHandle,
            content_type: &str,
        ) -> Result<(), ConformanceError> {
            self.mime.push((handle, content_type.into()));
            Ok(())
        }
        fn poll_delay(
            &mut self,
            _handle: StreamHandle,
            milliseconds: u64,
        ) -> Result<(), ConformanceError> {
            self.poll_delays.push(milliseconds);
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
            self.policies.push((handle, *policy));
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
    #[derive(Default)]
    struct FileOwner {
        files: Vec<DownloadedFile>,
        reject: bool,
    }
    impl RodinFileOwner for FileOwner {
        fn complete_download(&mut self, file: &DownloadedFile) -> Result<(), ConformanceError> {
            if self.reject {
                return Err(ConformanceError::HostFailure);
            }
            self.files.push(file.clone());
            Ok(())
        }
    }
    fn encoder() -> CanonicalMultipartEncoder {
        CanonicalMultipartEncoder {
            boundary: "rodin-source-boundary".into(),
        }
    }
    fn append(transport: &mut Transport, value: Value) -> Result<(), Box<dyn Error>> {
        transport
            .events
            .extend(events(200, serde_json::to_vec(&value)?, vec![]));
        Ok(())
    }
    fn setup(names: &[&str]) -> Result<Transport, Box<dyn Error>> {
        let mut transport = Transport::default();
        append(
            &mut transport,
            json!({"uuid":"task-1","jobs":{"uuids":["job-1"],"subscription_key":"sub-1"}}),
        )?;
        append(
            &mut transport,
            json!({"jobs":[{"uuid":"job-1","status":"Done"}]}),
        )?;
        append(
            &mut transport,
            json!({"list":names.iter().map(|name|json!({"name":name,"url":format!("https://fixture.invalid/{name}")})).collect::<Vec<_>>()}),
        )?;
        for name in names {
            let bytes = if name.ends_with("obj") {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/rodin/source-response.obj"
                ))
                .to_vec()
            } else {
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../sim_test_support/fixtures/providers/rodin/source-response.glb"
                ))
                .to_vec()
            };
            transport.events.extend(events(200, bytes, vec![]));
        }
        Ok(transport)
    }
    fn execute(
        node: &str,
        transport: &mut Transport,
        owner: &mut FileOwner,
    ) -> Result<ResultValue, Box<dyn Error>> {
        Ok(invoke(
            &input(node)?,
            credentials(),
            2_500_000,
            transport,
            &mut SourceProbeOwner::default(),
            owner,
            &encoder(),
        )?)
    }
    #[test]
    fn rodin_source_requests_and_real_multipart_match() -> Result<(), Box<dyn Error>> {
        source_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/rodin/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn rodin_hidden_auth_and_cost_prevent_preparation_and_network() -> Result<(), Box<dyn Error>> {
        let original = input("Rodin3D_Regular")?;
        let mut transport = Transport::default();
        let mut image = SourceProbeOwner::default();
        let mut files = FileOwner::default();
        assert!(
            invoke(
                &original,
                CredentialAvailability {
                    auth_token: false,
                    api_key: false
                },
                2_500_000,
                &mut transport,
                &mut image,
                &mut files,
                &encoder()
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        assert!(image.operations.is_empty());
        transport.deny_cost = true;
        assert!(
            invoke(
                &original,
                credentials(),
                2_500_000,
                &mut transport,
                &mut image,
                &mut files,
                &encoder()
            )
            .is_err()
        );
        assert!(image.operations.is_empty());
        assert!(transport.bodies.iter().all(Vec::is_empty));
        assert_eq!(
            transport.costs.first().ok_or("approval")?.operation,
            "rodin.generation.maximum"
        );
        let fallback = request_head(
            "/proxy/rodin/api/v2/status",
            HttpMethod::Post,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            false,
        )?;
        assert_eq!(fallback.secret_id.as_deref(), Some("sim_org.api_key"));
        let absolute = request_head(
            "https://fixture.invalid/a",
            HttpMethod::Get,
            credentials(),
            true,
        )?;
        assert!(absolute.secret_id.is_none());
        assert!(
            !absolute
                .headers
                .iter()
                .any(|header| header.name == "Authorization")
        );
        Ok(())
    }
    #[test]
    fn rodin_caller_ceiling_requires_canonical_approval_not_badge_policy()
    -> Result<(), Box<dyn Error>> {
        let value = input("Rodin3D_Gen25_Text")?;
        assert!(estimated_cost_microunits(&value)? > 1);
        let mut transport = setup(&[])?;
        let approved = invoke(
            &value,
            credentials(),
            1,
            &mut transport,
            &mut SourceProbeOwner::default(),
            &mut FileOwner::default(),
            &encoder(),
        )?;
        assert_eq!(approved.approval_receipt, vec![42]);
        assert_eq!(
            transport
                .costs
                .first()
                .ok_or("approval")?
                .maximum_microunits,
            1
        );
        assert!(!transport.bodies.first().ok_or("paid body")?.is_empty());
        let mut denied = Transport {
            deny_cost: true,
            ..Default::default()
        };
        assert!(
            invoke(
                &value,
                credentials(),
                1,
                &mut denied,
                &mut SourceProbeOwner::default(),
                &mut FileOwner::default(),
                &encoder()
            )
            .is_err()
        );
        assert_eq!(
            denied
                .costs
                .first()
                .ok_or("denied approval")?
                .maximum_microunits,
            1
        );
        assert!(denied.bodies.iter().all(Vec::is_empty));
        assert!(denied.policies.is_empty());
        Ok(())
    }
    #[test]
    fn rodin_badge_preserves_literal_extreme_high_case() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            estimated_cost_microunits(&input("Rodin3D_Regular")?)?,
            400_000
        );
        let mut value = input("Rodin3D_Gen25_Text")?;
        assert_eq!(estimated_cost_microunits(&value)?, 750_000);
        value
            .arguments
            .insert("mode".into(), json!({"mode":"Extreme-High"}));
        assert_eq!(estimated_cost_microunits(&value)?, 750_000);
        value
            .arguments
            .insert("mode".into(), json!({"mode":"extreme-high"}));
        value.arguments.insert("addon_highpack".into(), json!(true));
        assert_eq!(estimated_cost_microunits(&value)?, 2_300_000);
        Ok(())
    }
    #[test]
    fn rodin_all_actual_source_generation_failures_stop_before_poll() -> Result<(), Box<dyn Error>>
    {
        let source = oracle()?;
        for case in source["response_errors"]
            .as_array()
            .ok_or("response failures")?
        {
            let mut transport = Transport::default();
            append(&mut transport, case["override"]["generate"].clone())?;
            let mut image = SourceProbeOwner::default();
            assert!(
                invoke(
                    &input_from(case)?,
                    credentials(),
                    2_500_000,
                    &mut transport,
                    &mut image,
                    &mut FileOwner::default(),
                    &encoder()
                )
                .is_err()
            );
            assert_eq!(
                transport.heads.len(),
                case["requests"].as_array().ok_or("requests")?.len()
            );
            assert_eq!(json!(image.operations), case["media"]);
        }
        Ok(())
    }
    #[test]
    fn rodin_submission_legacy_empty_and_gen25_required_identity() -> Result<(), Box<dyn Error>> {
        let bytes = serde_json::to_vec(
            &json!({"uuid":"","jobs":{"uuids":[],"subscription_key":""},"error":"ignored"}),
        )?;
        assert_eq!(submission(&bytes, false)?.uuid.as_deref(), Some(""));
        assert!(submission(&bytes, true).is_err());
        assert!(submission(br#"{"jobs":null}"#, false).is_err());
        assert!(resources(br#"{"items":[]}"#).is_err());
        assert!(resources(br#"{"list":[]}"#)?.is_empty());
        Ok(())
    }
    #[test]
    fn rodin_pinned_poll_extractors_and_float_progress_match() -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        for case in source["poll_cases"].as_array().ok_or("poll cases")? {
            let value = &case["response"];
            if value.is_null() {
                continue;
            }
            let result = poll_status(&serde_json::to_vec(value)?);
            if value["jobs"]
                .as_array()
                .is_some_and(|jobs| jobs.iter().any(|job| job["status"] == "unknown"))
            {
                assert!(result.is_err());
            }
        }
        let failed = poll_status(br#"{"jobs":[{"uuid":"j","status":"Failed"}]}"#)?;
        assert!(failed.suppressed_status_exception);
        assert!(!failed.complete);
        assert_eq!(failed.progress, Some(0));
        let empty = poll_status(br#"{"jobs":[]}"#)?;
        assert!(empty.complete);
        assert_eq!(empty.progress, None);
        let jobs=(0..100).map(|index|json!({"uuid":index.to_string(),"status":if index<57 {"Done"}else{"Generating"}})).collect::<Vec<_>>();
        assert_eq!(
            poll_status(&serde_json::to_vec(&json!({"jobs":jobs}))?)?.progress,
            Some(56)
        );
        assert!(poll_status(br#"{"jobs":[{"uuid":"j","status":"unknown"}]}"#).is_err());
        Ok(())
    }
    #[test]
    fn rodin_actual_poll_traces_match_source_with_caller_clock() -> Result<(), Box<dyn Error>> {
        let source = oracle()?;
        for case in source["poll_cases"].as_array().ok_or("poll cases")? {
            let mut transport = Transport::default();
            append(
                &mut transport,
                json!({"uuid":"task-1","jobs":{"uuids":[],"subscription_key":"s"}}),
            )?;
            let count = case["requests"].as_u64().ok_or("poll request count")?;
            for _ in 0..count {
                append(&mut transport, case["response"].clone())?;
            }
            if case.get("result").is_some() {
                append(&mut transport, json!({"list":[]}))?;
            }
            let result = execute("Rodin3D_Regular", &mut transport, &mut FileOwner::default());
            assert_eq!(result.is_ok(), case.get("result").is_some());
            assert_eq!(
                transport
                    .heads
                    .iter()
                    .filter(|head| head.path.ends_with("/status"))
                    .count() as u64,
                count
            );
            assert_eq!(
                json!(
                    transport
                        .poll_delays
                        .iter()
                        .map(|value| *value as f64 / 1000.0)
                        .collect::<Vec<_>>()
                ),
                case["delays"]
            );
            assert_eq!(
                json!(
                    transport
                        .progresses
                        .iter()
                        .map(|value| json!({"completed":value.completed,"total":value.total}))
                        .collect::<Vec<_>>()
                ),
                case["progress"]
            );
        }
        Ok(())
    }
    #[test]
    fn rodin_legacy_last_glb_and_all_downloads_preserve_real_bytes() -> Result<(), Box<dyn Error>> {
        let mut transport = setup(&["first.glb", "cube.obj", "LAST.GLB"])?;
        let mut owner = FileOwner::default();
        let result = execute("Rodin3D_Gen2", &mut transport, &mut owner)?;
        assert_eq!(result.selected_file, Some(2));
        assert_eq!(
            result.compatibility_model_path.as_deref(),
            Some("Rodin3D_task-1/LAST.GLB")
        );
        assert_eq!(owner.files.len(), 3);
        assert!(owner.files.iter().all(|file| file.receipt == [7, 8, 9]));
        assert_eq!(
            result.files.first().ok_or("file")?.bytes,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/rodin/source-response.glb"
            ))
        );
        assert_eq!(
            transport
                .heads
                .iter()
                .map(|head| head.path.as_str())
                .take(3)
                .collect::<Vec<_>>(),
            vec![
                "/proxy/rodin/api/v2/rodin",
                "/proxy/rodin/api/v2/status",
                "/proxy/rodin/api/v2/download"
            ]
        );
        assert!(
            transport
                .heads
                .iter()
                .skip(3)
                .all(|head| head.secret_id.is_none() && head.method == HttpMethod::Get)
        );
        assert_eq!(transport.mime.len(), 1);
        assert!(
            transport
                .mime
                .first()
                .ok_or("mime")?
                .1
                .contains("boundary=rodin-source-boundary")
        );
        assert_eq!(
            transport
                .policies
                .last()
                .ok_or("policy")?
                .1
                .source_read_chunk_bytes,
            Some(1024 * 1024)
        );
        Ok(())
    }
    #[test]
    fn rodin_gen25_prefers_first_target_and_fallback_redownloads_after_raw_saves()
    -> Result<(), Box<dyn Error>> {
        let mut transport = setup(&["first.glb", "second.glb"])?;
        let mut owner = FileOwner::default();
        let result = execute("Rodin3D_Gen25_Text", &mut transport, &mut owner)?;
        assert_eq!(result.selected_file, Some(0));
        assert_eq!(
            owner.files.first().ok_or("first")?.source_format.as_deref(),
            Some("glb")
        );
        assert!(owner.files.get(1).ok_or("second")?.source_format.is_none());
        let mut transport = setup(&["cube.obj"])?;
        transport.events.extend(events(
            200,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/rodin/source-response.obj"
            ))
            .to_vec(),
            vec![],
        ));
        let mut owner = FileOwner::default();
        let result = execute("Rodin3D_Gen25_Text", &mut transport, &mut owner)?;
        assert_eq!(result.selected_file, Some(1));
        let first = owner.files.first().ok_or("raw")?;
        let fallback = owner.files.get(1).ok_or("fallback")?;
        assert!(first.compatibility_path.is_some());
        assert!(first.source_format.is_none());
        assert!(fallback.compatibility_path.is_none());
        assert!(fallback.fallback);
        assert_eq!(fallback.source_format.as_deref(), Some("obj"));
        assert_eq!(first.bytes, fallback.bytes);
        assert_eq!(transport.heads.len(), 5);
        Ok(())
    }
    #[test]
    fn rodin_empty_files_and_dotfiles_preserve_source_nullable_gate() -> Result<(), Box<dyn Error>>
    {
        let mut transport = setup(&[])?;
        let result = execute(
            "Rodin3D_Gen25_Text",
            &mut transport,
            &mut FileOwner::default(),
        )?;
        assert_eq!(result.selected_file, None);
        assert_eq!(extension(".glb"), "");
        assert_eq!(extension("directory/MODEL.GLB"), "glb");
        let mut transport = setup(&[".glb"])?;
        let result = execute("Rodin3D_Regular", &mut transport, &mut FileOwner::default())?;
        assert_eq!(
            result.files.first().ok_or("dot")?.source_format.as_deref(),
            Some("glb")
        );
        Ok(())
    }
    #[test]
    fn rodin_file_owner_failure_and_relative_url_stop_later_resources() -> Result<(), Box<dyn Error>>
    {
        let mut transport = setup(&["first.glb", "cube.obj"])?;
        let mut owner = FileOwner {
            reject: true,
            ..Default::default()
        };
        assert!(execute("Rodin3D_Regular", &mut transport, &mut owner).is_err());
        assert_eq!(transport.heads.len(), 4);
        let resource = Resource {
            url: "/relative.glb".into(),
            name: "a.glb".into(),
        };
        let mut transport = Transport::default();
        assert!(
            downloaded(
                &resource,
                DownloadPlan {
                    format: Some("glb".into()),
                    save: true,
                    fallback: false,
                    folder: "f",
                    credentials: credentials()
                },
                &mut transport,
                &mut FileOwner::default()
            )
            .is_err()
        );
        assert!(transport.heads.is_empty());
        Ok(())
    }
    #[test]
    fn rodin_retries_replay_body_bind_mime_and_preserve_source_budgets()
    -> Result<(), Box<dyn Error>> {
        for (policy, attempts, status) in [
            (GENERATION_RETRY, 3, 500),
            (POLL_RETRY, 10, 500),
            (DOWNLOAD_RETRY, 5, 429),
        ] {
            let head = request_head(
                "/proxy/rodin/api/v2/rodin",
                HttpMethod::Post,
                credentials(),
                false,
            )?;
            let mut transport = Transport::default();
            for _ in 0..attempts {
                transport.events.extend(events(
                    status,
                    vec![],
                    vec![Header {
                        name: "Retry-After".into(),
                        value: "600".into(),
                    }],
                ));
            }
            transport.events.extend(events(200, vec![1], vec![]));
            let handle = transport.start_request(&head)?;
            exchange_existing(&head, b"body", None, 1024, policy, handle, &mut transport)?;
            assert_eq!(transport.delays.len(), attempts);
            assert!(transport.bodies.iter().all(|body| body == b"body"));
            assert_eq!(
                transport.mime.len(),
                if policy.multipart { attempts + 1 } else { 0 }
            );
            assert_eq!(
                transport.delays.first().copied(),
                Some(if policy.download { 1000 } else { 150_000 })
            );
        }
        let head = request_head(
            "/proxy/rodin/api/v2/status",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut transport = Transport::default();
        for _ in 0..17 {
            transport.events.extend(events(429, vec![], vec![]));
        }
        let handle = transport.start_request(&head)?;
        assert!(
            exchange_existing(&head, b"{}", None, 1024, POLL_RETRY, handle, &mut transport)
                .is_err()
        );
        assert_eq!(transport.delays.len(), 16);
        assert_eq!(transport.delays.last(), Some(&30_000));
        Ok(())
    }
    #[test]
    fn rodin_bad_frames_oversized_receipts_and_stale_retry_handles_fail()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "https://fixture.invalid/model.glb",
            HttpMethod::Get,
            credentials(),
            true,
        )?;
        for events in [
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
            vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Completed(vec![0; MAXIMUM_RECEIPT_BYTES + 1])),
            ],
        ] {
            let mut transport = Transport {
                events: events.into(),
                ..Default::default()
            };
            let handle = transport.start_request(&head)?;
            assert!(
                exchange_existing(
                    &head,
                    &[],
                    None,
                    1024,
                    DOWNLOAD_RETRY,
                    handle,
                    &mut transport
                )
                .is_err()
            );
        }
        let mut transport = Transport {
            same_retry_handle: true,
            events: events(500, vec![], vec![]),
            ..Default::default()
        };
        let handle = transport.start_request(&head)?;
        assert!(matches!(
            exchange_existing(
                &head,
                &[],
                None,
                1024,
                DOWNLOAD_RETRY,
                handle,
                &mut transport
            ),
            Err(RodinError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn rodin_cancellation_worker_loss_and_fresh_injection_do_not_publish()
    -> Result<(), Box<dyn Error>> {
        let mut transport = Transport {
            cancelled: true,
            ..Default::default()
        };
        assert!(execute("Rodin3D_Regular", &mut transport, &mut FileOwner::default()).is_err());
        assert!(transport.heads.is_empty());
        let mut transport = Transport {
            events: vec![Err(ConformanceError::HostFailure)].into(),
            ..Default::default()
        };
        assert!(execute("Rodin3D_Regular", &mut transport, &mut FileOwner::default()).is_err());
        assert!(transport.delays.is_empty());
        let mut transport = setup(&[])?;
        assert!(execute("Rodin3D_Regular", &mut transport, &mut FileOwner::default()).is_ok());
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
    struct RejectMaterializer;
    impl RodinMaterializer for RejectMaterializer {
        fn materialize(
            &mut self,
            _result: &ResultValue,
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
    }
    #[test]
    fn rodin_actual_encoded_model_never_substitutes_for_native_publication()
    -> Result<(), Box<dyn Error>> {
        let mut transport = setup(&["model.glb"])?;
        let mut image = SourceProbeOwner::default();
        let mut file = FileOwner::default();
        let mut result = RejectMaterializer;
        let mut sink = Sink::default();
        let mut owners = MaterializationOwners {
            image: &mut image,
            file: &mut file,
            result: &mut result,
            sink: &mut sink,
        };
        assert!(
            invoke_and_materialize(
                &input("Rodin3D_Regular")?,
                credentials(),
                2_500_000,
                &mut transport,
                &mut owners,
                &encoder()
            )
            .is_err()
        );
        assert_eq!(sink.published, 0);
        assert_eq!(file.files.len(), 1);
        assert_eq!(expected_outputs("Rodin3D_Regular")?.len(), 2);
        assert_eq!(expected_outputs("Rodin3D_Gen25_Text")?.len(), 1);
        Ok(())
    }
}
