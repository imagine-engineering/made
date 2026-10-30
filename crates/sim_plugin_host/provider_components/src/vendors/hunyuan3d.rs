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
pub const NAMESPACE: &str = "zed.sim.provider.hunyuan3d";
pub const SOURCE_SHA256: &str = "b40684e319579f7f1fb43fc513c0a94298602468d777c11a765de728256fbab0";
pub const ALIASES: &[&str] = &["tencent"];
pub const MAXIMUM_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;
pub const MAXIMUM_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
const MAXIMUM_POLLS: u32 = 480;
const MAXIMUM_TOTAL_POLLS: u32 = 4096;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0658",
        node_identifier: "Tencent3DPartNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0659",
        node_identifier: "Tencent3DTextureEditNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0660",
        node_identifier: "TencentImageToModelNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0661",
        node_identifier: "TencentModelTo3DUVNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0662",
        node_identifier: "TencentSmartTopologyNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0663",
        node_identifier: "TencentTextToModelNode",
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
        feature_id: "COMFY-API-EXT-0164",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-part",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0165",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-part/query",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0166",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-pro",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0167",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-pro/query",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0168",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-smart-topology",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0169",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-smart-topology/query",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0170",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-texture-edit",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0171",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-texture-edit/query",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0172",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-uv",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0173",
        method: "POST",
        path: "/proxy/tencent/hunyuan/3d-uv/query",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum HunyuanError {
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
impl fmt::Display for HunyuanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => write!(formatter, "Hunyuan3D task failed: {message}"),
            Self::Http { status, .. } => write!(formatter, "Hunyuan3D HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for HunyuanError {}
impl From<ConformanceError> for HunyuanError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for HunyuanError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub model: String,
    pub prompt: String,
    pub face_count: u32,
    pub generate_type: GenerateType,
    pub seed: u32,
    pub model_format: String,
    pub polygon_type: String,
    pub face_level: String,
    pub images: BTreeMap<String, ImageDimensions>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateType {
    pub generate_type: String,
    pub polygon_type: Option<String>,
    pub pbr: Option<bool>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UploadRequirement {
    pub role: String,
    pub content_type: String,
    pub format: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub total_pixels: Option<u64>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedMedia {
    pub requirement: UploadRequirement,
    pub file_name: String,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct UploadedMedia {
    pub role: String,
    pub url: String,
}
fn validate_prompt(prompt: &str) -> Result<(), HunyuanError> {
    // Python str.strip includes these four separators beyond Unicode White_Space.
    let length = prompt
        .trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
        })
        .chars()
        .count();
    if !(1..=1024).contains(&length) {
        return Err(HunyuanError::Input(
            "Prompt must contain 1 to 1024 stripped characters",
        ));
    }
    Ok(())
}
fn is_generation(input: &Input) -> bool {
    matches!(
        input.node.as_str(),
        "TencentTextToModelNode" | "TencentImageToModelNode"
    )
}
pub fn upload_requirements(input: &Input) -> Result<Vec<UploadRequirement>, HunyuanError> {
    if input.seed > 2147483647 || input.prompt.len() > 65536 {
        return Err(HunyuanError::Input("Hunyuan3D input exceeds its bound"));
    }
    if is_generation(input) {
        if input.node == "TencentTextToModelNode" {
            validate_prompt(&input.prompt)?;
        }
        if input.model == "3.1"
            && input
                .generate_type
                .generate_type
                .eq_ignore_ascii_case("lowpoly")
        {
            return Err(HunyuanError::Input(
                "The LowPoly option is currently unavailable for the 3.1 model.",
            ));
        }
        if !matches!(input.model.as_str(), "3.0" | "3.1")
            || !(3000..=1500000).contains(&input.face_count)
            || !matches!(
                input.generate_type.generate_type.as_str(),
                "Normal" | "LowPoly" | "Geometry"
            )
            || input
                .generate_type
                .polygon_type
                .as_deref()
                .is_some_and(|polygon| !matches!(polygon, "triangle" | "quadrilateral"))
        {
            return Err(HunyuanError::Input("Invalid Hunyuan3D generation controls"));
        }
        if input.node == "TencentTextToModelNode" {
            return Ok(vec![]);
        }
        let front = input
            .images
            .get("image")
            .ok_or(HunyuanError::Input("Missing front image"))?;
        validate_image(front)?;
        if input.images.keys().any(|role| {
            !matches!(
                role.as_str(),
                "image" | "image_left" | "image_right" | "image_back"
            )
        }) {
            return Err(HunyuanError::Input("Unknown image role"));
        }
        let mut requirements = vec![];
        for role in ["image_left", "image_right", "image_back", "image"] {
            if let Some(dimensions) = input.images.get(role) {
                validate_image(dimensions)?;
                let (width, height) = source_downscale(dimensions)?;
                requirements.push(UploadRequirement {
                    role: role.into(),
                    content_type: "image/webp".into(),
                    format: "webp".into(),
                    width: Some(width),
                    height: Some(height),
                    total_pixels: Some(24_010_000),
                });
            }
        }
        return Ok(requirements);
    }
    let format = input.model_format.to_lowercase();
    let allowed = match input.node.as_str() {
        "TencentModelTo3DUVNode" => matches!(format.as_str(), "glb" | "obj" | "fbx"),
        "Tencent3DTextureEditNode" | "Tencent3DPartNode" => format == "fbx",
        "TencentSmartTopologyNode" => matches!(format.as_str(), "glb" | "obj"),
        _ => false,
    };
    if !allowed {
        return Err(HunyuanError::Input(
            "Unsupported Hunyuan3D node or model format",
        ));
    }
    if input.node == "Tencent3DTextureEditNode" {
        validate_prompt(&input.prompt)?;
    }
    if input.node == "TencentSmartTopologyNode"
        && (!matches!(input.polygon_type.as_str(), "triangle" | "quadrilateral")
            || !matches!(input.face_level.as_str(), "medium" | "high" | "low"))
    {
        return Err(HunyuanError::Input("Invalid topology controls"));
    }
    let content_type = match format.as_str() {
        "glb" => "model/gltf-binary",
        "obj" => "model/obj",
        _ => "application/octet-stream",
    };
    Ok(vec![UploadRequirement {
        role: "model_3d".into(),
        content_type: content_type.into(),
        format,
        width: None,
        height: None,
        total_pixels: None,
    }])
}
fn validate_image(dimensions: &ImageDimensions) -> Result<(), HunyuanError> {
    if dimensions.width < 128
        || dimensions.height < 128
        || dimensions.width > 1_000_000
        || dimensions.height > 1_000_000
    {
        return Err(HunyuanError::Input("Invalid Hunyuan3D image dimensions"));
    }
    Ok(())
}
pub fn source_downscale(dimensions: &ImageDimensions) -> Result<(u32, u32), HunyuanError> {
    validate_image(dimensions)?;
    let maximum = dimensions.width.max(dimensions.height);
    if maximum <= 4900 {
        return Ok((dimensions.width, dimensions.height));
    }
    let scale = 4900.0 / f64::from(maximum);
    let width = (f64::from(dimensions.width) * scale).round_ties_even() as u32;
    let height = (f64::from(dimensions.height) * scale).round_ties_even() as u32;
    if width == 0 || height == 0 {
        return Err(HunyuanError::Input("Downscaled image is empty"));
    }
    Ok((width, height))
}
fn route(input: &Input) -> Result<&'static str, HunyuanError> {
    match input.node.as_str() {
        "TencentTextToModelNode" | "TencentImageToModelNode" => Ok("/proxy/tencent/hunyuan/3d-pro"),
        "TencentModelTo3DUVNode" => Ok("/proxy/tencent/hunyuan/3d-uv"),
        "Tencent3DTextureEditNode" => Ok("/proxy/tencent/hunyuan/3d-texture-edit"),
        "Tencent3DPartNode" => Ok("/proxy/tencent/hunyuan/3d-part"),
        "TencentSmartTopologyNode" => Ok("/proxy/tencent/hunyuan/3d-smart-topology"),
        _ => Err(HunyuanError::Input("Unknown Hunyuan3D node")),
    }
}
pub fn request(
    input: &Input,
    uploads: &[UploadedMedia],
) -> Result<(&'static str, Vec<u8>), HunyuanError> {
    let requirements = upload_requirements(input)?;
    if requirements.len() != uploads.len() {
        return Err(HunyuanError::Input("Missing uploaded media"));
    }
    for (requirement, upload) in requirements.iter().zip(uploads) {
        if requirement.role != upload.role || endpoint(&upload.url)? {
            return Err(HunyuanError::Input("Invalid uploaded media role or URL"));
        }
    }
    let mut body = serde_json::Map::new();
    if is_generation(input) {
        body.insert("Model".into(), json!(input.model));
        body.insert("FaceCount".into(), json!(input.face_count));
        body.insert(
            "GenerateType".into(),
            json!(input.generate_type.generate_type),
        );
        if let Some(pbr) = input.generate_type.pbr {
            body.insert("EnablePBR".into(), json!(pbr));
        }
        if let Some(polygon) = &input.generate_type.polygon_type {
            body.insert("PolygonType".into(), json!(polygon));
        }
        if input.node == "TencentTextToModelNode" {
            body.insert("Prompt".into(), json!(input.prompt));
        } else {
            let front = uploads
                .iter()
                .find(|upload| upload.role == "image")
                .ok_or(HunyuanError::Input("Missing uploaded front image"))?;
            body.insert("ImageUrl".into(), json!(front.url));
            let views = uploads.iter().filter(|upload| upload.role != "image").map(|upload| json!({"ViewType":upload.role.trim_start_matches("image_"),"ViewImageUrl":upload.url})).collect::<Vec<_>>();
            if !views.is_empty() {
                body.insert("MultiViewImages".into(), json!(views));
            }
        }
    } else {
        let upload = uploads
            .first()
            .ok_or(HunyuanError::Input("Missing uploaded model"))?;
        let file = json!({"Type":input.model_format.to_uppercase(),"Url":upload.url});
        let field = if matches!(
            input.node.as_str(),
            "TencentModelTo3DUVNode" | "Tencent3DPartNode"
        ) {
            "File"
        } else {
            "File3D"
        };
        body.insert(field.into(), file);
        if input.node == "Tencent3DTextureEditNode" {
            body.insert("Prompt".into(), json!(input.prompt));
            body.insert("EnablePBR".into(), json!(true));
        }
        if input.node == "TencentSmartTopologyNode" {
            body.insert("PolygonType".into(), json!(input.polygon_type));
            body.insert("FaceLevel".into(), json!(input.face_level));
        }
    }
    let bytes = serde_json::to_vec(&body)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok((route(input)?, bytes))
}
pub fn price_badge_microunits(input: &Input) -> Result<u64, HunyuanError> {
    upload_requirements(input)?;
    Ok(match input.node.as_str() {
        "TencentModelTo3DUVNode" => 200_000,
        "Tencent3DPartNode" | "Tencent3DTextureEditNode" => 600_000,
        "TencentSmartTopologyNode" => 1_000_000,
        _ => {
            // The source badge compares lowercase literals despite titlecase schema options.
            let base = match input.generate_type.generate_type.as_str() {
                "normal" => 25,
                "lowpoly" => 30,
                _ => 15,
            };
            let pbr = if input.generate_type.pbr == Some(true) {
                10
            } else {
                0
            };
            let face = if input.face_count != 500000 { 10 } else { 0 };
            let multi = if input.node == "TencentImageToModelNode" && input.images.len() > 1 {
                10
            } else {
                0
            };
            (base + pbr + face + multi) * 20_000
        }
    })
}
fn unwrap_response(value: Value) -> Value {
    match value.get("Response") {
        Some(response) if response.is_object() => response.clone(),
        _ => value,
    }
}
pub fn submitted_task(bytes: &[u8]) -> Result<String, HunyuanError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let response = unwrap_response(serde_json::from_slice(bytes)?);
    if let Some(error) = response.get("Error").filter(|error| !error.is_null()) {
        return Err(HunyuanError::Provider(error.to_string()));
    }
    let task = response
        .get("JobId")
        .and_then(Value::as_str)
        .ok_or(HunyuanError::Response("Missing Hunyuan3D JobId"))?;
    if task.len() > 1024
        || task
            .chars()
            .any(|character| character.is_control() || matches!(character, '/' | '\\'))
    {
        return Err(HunyuanError::Response("Unsafe Hunyuan3D JobId"));
    }
    Ok(task.into())
}
#[derive(Clone, Debug, Deserialize)]
pub struct ResultFile {
    #[serde(rename = "Type")]
    pub file_type: String,
    #[serde(rename = "Url")]
    pub url: String,
    #[serde(rename = "PreviewImageUrl", default)]
    pub preview_image_url: String,
}
#[derive(Debug, Deserialize)]
pub struct TaskStatus {
    #[serde(rename = "Status")]
    pub status: String,
    #[serde(rename = "ResultFile3Ds", default)]
    pub files: Vec<ResultFile>,
    #[serde(rename = "ErrorCode", default)]
    pub error_code: String,
    #[serde(rename = "ErrorMessage", default)]
    pub error_message: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Completed,
    Queued,
    Processing,
}
pub fn task_status(bytes: &[u8]) -> Result<(TaskStatus, Phase), HunyuanError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let response: Value = serde_json::from_slice(bytes)?;
    let status: TaskStatus = serde_json::from_value(unwrap_response(response.clone()))?;
    if status.files.len() > 256 || status.status.len() > 1024 {
        return Err(ConformanceError::BodyLimit.into());
    }
    let phase =
        match status
            .status
            .trim_matches(|character: char| {
                character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
            })
            .to_lowercase()
            .as_str()
        {
            "succeeded" | "succeed" | "success" | "completed" | "finished" | "done"
            | "complete" => Phase::Completed,
            "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error" => {
                return Err(HunyuanError::Provider(response.to_string()));
            }
            "created" | "queued" | "queueing" | "submitted" | "initializing" | "wait"
            | "in_queue" => Phase::Queued,
            _ => Phase::Processing,
        };
    Ok((status, phase))
}
pub fn tencent_rate_limited(status: u16, body: &Value) -> bool {
    status == 400
        && body
            .get("Response")
            .and_then(|response| response.get("Error"))
            .and_then(|error| error.get("Code"))
            .is_some_and(|code| {
                code.as_str()
                    .map_or_else(|| code.to_string(), str::to_owned)
                    .contains("RequestLimitExceeded")
            })
}
fn get_file<'a>(
    files: &'a [ResultFile],
    kind: &str,
    required: bool,
) -> Result<Option<&'a ResultFile>, HunyuanError> {
    let file = files
        .iter()
        .find(|file| file.file_type.eq_ignore_ascii_case(kind));
    if required && file.is_none() {
        return Err(HunyuanError::Response("Required file type is missing"));
    }
    Ok(file)
}
fn endpoint(value: &str) -> Result<bool, HunyuanError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(HunyuanError::Response(
            "Invalid Hunyuan3D response endpoint",
        ));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(HunyuanError::Response(
            "Invalid Hunyuan3D response endpoint",
        ))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(HunyuanError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(HunyuanError::Response(
            "Invalid Hunyuan3D endpoint authority",
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
) -> Result<RequestHead, HunyuanError> {
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
            return Err(HunyuanError::Input("Missing Comfy API credentials"));
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

pub trait HunyuanTransport: ProviderTransport {
    fn begin_response_deadline(
        &mut self,
        _handle: StreamHandle,
        _timeout_milliseconds: Option<u64>,
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
    fn upload_prepared(
        &mut self,
        handle: StreamHandle,
        media: &PreparedMedia,
        wait_label: &str,
    ) -> Result<UploadedMedia, ConformanceError>;
}

#[derive(Clone, Debug)]
pub struct DownloadedAsset {
    pub bytes: Vec<u8>,
    pub response_receipt: Vec<u8>,
    pub source_url: String,
    pub handle: StreamHandle,
}
#[derive(Clone, Debug)]
pub struct DecodedObjArchiveEntry {
    pub name: String,
    pub bytes: Vec<u8>,
}
pub trait HunyuanMediaOwner {
    fn validate_rgba_image(&mut self, _asset: &DownloadedAsset) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn stage_zero_rgb(
        &mut self,
        _handle: StreamHandle,
        _receipt: &[u8],
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }

    // The canonical owner must enforce ZIP framing, decompression budgets and cancellation;
    // None means a non-ZIP response, not a malformed recognized archive.
    fn read_obj_archive(
        &mut self,
        asset: &DownloadedAsset,
    ) -> Result<Option<Vec<DecodedObjArchiveEntry>>, ConformanceError>;
    fn validate_rgb_image(
        &mut self,
        asset: &DownloadedAsset,
        entry_index: usize,
        entry: &DecodedObjArchiveEntry,
    ) -> Result<(), ConformanceError>;
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceOutput {
    Absent,
    Text(String),
    File {
        asset: usize,
        format: &'static str,
        persist_task_id: Option<String>,
    },
    Image {
        asset: usize,
        entry: Option<usize>,
        mode: &'static str,
    },
    Obj {
        asset: usize,
        entry: Option<usize>,
    },
    ZeroRgb,
}
impl SourceOutput {
    fn alias(&self) -> Option<&'static str> {
        match self {
            Self::Absent => None,
            Self::Text(_) => Some("STRING"),
            Self::Image { .. } | Self::ZeroRgb => Some("IMAGE"),
            Self::Obj { .. } => Some("FILE_3D_OBJ"),
            Self::File { format, .. } => match *format {
                "obj" => Some("FILE_3D_OBJ"),
                "fbx" => Some("FILE_3D_FBX"),
                _ => Some("FILE_3D_GLB"),
            },
        }
    }
}
#[derive(Clone, Debug)]
pub struct DownloadedResult {
    pub node: String,
    pub task_id: String,
    pub submission_receipt: Vec<u8>,
    pub status_receipt: Vec<u8>,
    pub assets: Vec<DownloadedAsset>,
    pub archives: BTreeMap<usize, Vec<DecodedObjArchiveEntry>>,
    pub outputs: Vec<SourceOutput>,
}
#[derive(Default, Debug)]
pub struct ObjSelection {
    pub obj: Option<usize>,
    pub texture: Option<usize>,
    pub metallic: Option<usize>,
    pub normal: Option<usize>,
    pub roughness: Option<usize>,
    pub all_images: Vec<usize>,
}
fn scan_obj_entries(entries: &[DecodedObjArchiveEntry]) -> Result<ObjSelection, HunyuanError> {
    if entries.len() > 256 {
        return Err(ConformanceError::BodyLimit.into());
    }
    let mut selected = ObjSelection::default();
    let mut total = 0usize;
    for (index, entry) in entries.iter().enumerate() {
        total = total
            .checked_add(entry.bytes.len())
            .ok_or(ConformanceError::BodyLimit)?;
        if total > MAXIMUM_MEDIA_BYTES || entry.name.len() > 8192 || entry.name.contains('\0') {
            return Err(ConformanceError::BodyLimit.into());
        }
        let lower = entry.name.to_lowercase();
        if lower.ends_with(".obj") {
            selected.obj = Some(index);
        } else if [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]
            .iter()
            .any(|extension| lower.ends_with(extension))
        {
            selected.all_images.push(index);
            let stem = lower
                .rsplit_once('.')
                .map_or(lower.as_str(), |(stem, _)| stem);
            if stem.ends_with("_metallic") {
                selected.metallic = Some(index);
            } else if stem.ends_with("_normal") {
                selected.normal = Some(index);
            } else if stem.ends_with("_roughness") {
                selected.roughness = Some(index);
            } else {
                selected.texture = Some(index);
            }
        }
    }
    Ok(selected)
}
pub fn select_obj_entries(
    entries: &[DecodedObjArchiveEntry],
) -> Result<ObjSelection, HunyuanError> {
    let selected = scan_obj_entries(entries)?;
    if selected.obj.is_none() {
        return Err(HunyuanError::Response(
            "ZIP archive does not contain an OBJ file.",
        ));
    }
    Ok(selected)
}
fn checked_handle(
    handle: StreamHandle,
    previous: Option<StreamHandle>,
) -> Result<(), HunyuanError> {
    if handle.invocation == 0
        || handle.slot == 0
        || handle.generation == 0
        || previous.is_some_and(|old| old == handle || old.invocation != handle.invocation)
    {
        return Err(ConformanceError::InvalidHandle.into());
    }
    Ok(())
}
pub fn invoke(
    input: &Input,
    media: &[PreparedMedia],
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn HunyuanTransport,
    media_owner: &mut dyn HunyuanMediaOwner,
) -> Result<DownloadedResult, HunyuanError> {
    let requirements = upload_requirements(input)?;
    if requirements.len() != media.len() {
        return Err(HunyuanError::Input("Missing canonical prepared media"));
    }
    let mut size = 0usize;
    for (requirement, prepared) in requirements.iter().zip(media) {
        if *requirement != prepared.requirement
            || prepared.bytes.is_empty()
            || prepared.file_name.is_empty()
            || prepared.file_name.len() > 1024
            || prepared
                .file_name
                .chars()
                .any(|character| character.is_control() || matches!(character, '/' | '\\'))
        {
            return Err(HunyuanError::Input("Invalid canonical media preparation"));
        }
        size = size
            .checked_add(prepared.bytes.len())
            .ok_or(ConformanceError::BodyLimit)?;
    }
    if size > MAXIMUM_MEDIA_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    if maximum_cost_microunits < price_badge_microunits(input)? || maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    let path = route(input)?;
    let head = request_head(path, HttpMethod::Post, credentials, false)?;
    transport.check_cancelled(None)?;
    let handle = transport.start_request(&head)?;
    checked_handle(handle, None)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "hunyuan3d.generation.maximum".into(),
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
    let mut uploaded = vec![];
    for prepared in media {
        transport.check_cancelled(Some(handle))?;
        let upload = transport.upload_prepared(handle, prepared, "Uploading")?;
        if upload.role != prepared.requirement.role || endpoint(&upload.url)? {
            return Err(HunyuanError::Response("Invalid canonical upload response"));
        }
        uploaded.push(upload);
    }
    let (_, body) = request(input, &uploaded)?;
    let (bytes, submission_receipt, mut previous) = exchange_existing(
        &head,
        &body,
        None,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        transport,
    )?;
    let task_id = submitted_task(&bytes)?;
    let query_head = request_head(
        &format!("{path}/query"),
        HttpMethod::Post,
        credentials,
        false,
    )?;
    let query_body = serde_json::to_vec(&json!({"JobId":task_id}))?;
    let mut consumed = 0;
    for _ in 0..MAXIMUM_TOTAL_POLLS {
        let (bytes, status_receipt, poll_handle) = exchange_with_policy(
            &query_head,
            &query_body,
            None,
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            Some(previous),
            transport,
        )?;
        previous = poll_handle;
        let (status, phase) = task_status(&bytes)?;
        if phase == Phase::Completed {
            let mut result = DownloadedResult {
                node: input.node.clone(),
                task_id,
                submission_receipt,
                status_receipt,
                assets: vec![],
                archives: BTreeMap::new(),
                outputs: vec![],
            };
            materialization_plan(
                input,
                &status.files,
                credentials,
                &mut previous,
                transport,
                media_owner,
                &mut result,
            )?;
            return Ok(result);
        }
        transport.check_cancelled(Some(poll_handle))?;
        transport.delay(poll_handle, 5000)?;
        if phase != Phase::Queued {
            consumed += 1;
            if consumed >= MAXIMUM_POLLS {
                break;
            }
        }
    }
    Err(ConformanceError::PollLimit.into())
}
fn download(
    file: &ResultFile,
    credentials: CredentialAvailability,
    previous: &mut StreamHandle,
    transport: &mut dyn HunyuanTransport,
    result: &mut DownloadedResult,
) -> Result<usize, HunyuanError> {
    if endpoint(&file.url)? {
        return Err(HunyuanError::Response("Downloads require absolute URLs"));
    }
    let head = request_head(&file.url, HttpMethod::Get, credentials, true)?;
    let (bytes, response_receipt, handle) = exchange_with_policy(
        &head,
        &[],
        None,
        MAXIMUM_MEDIA_BYTES,
        DOWNLOAD_RETRY,
        Some(*previous),
        transport,
    )?;
    *previous = handle;
    let total = result.assets.iter().try_fold(bytes.len(), |total, asset| {
        total
            .checked_add(asset.bytes.len())
            .ok_or(ConformanceError::BodyLimit)
    })?;
    let total = result
        .archives
        .values()
        .flatten()
        .try_fold(total, |total, entry| {
            total
                .checked_add(entry.bytes.len())
                .ok_or(ConformanceError::BodyLimit)
        })?;
    if bytes.is_empty() || total > MAXIMUM_MEDIA_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let index = result.assets.len();
    result.assets.push(DownloadedAsset {
        bytes,
        response_receipt,
        source_url: file.url.clone(),
        handle,
    });
    Ok(index)
}
fn required_file<'a>(files: &'a [ResultFile], kind: &str) -> Result<&'a ResultFile, HunyuanError> {
    get_file(files, kind, true)?.ok_or(HunyuanError::Response("Required file is missing"))
}
fn download_file(
    files: &[ResultFile],
    kind: &'static str,
    credentials: CredentialAvailability,
    previous: &mut StreamHandle,
    transport: &mut dyn HunyuanTransport,
    result: &mut DownloadedResult,
    persist_task_id: Option<String>,
) -> Result<SourceOutput, HunyuanError> {
    let asset = download(
        required_file(files, kind)?,
        credentials,
        previous,
        transport,
        result,
    )?;
    Ok(SourceOutput::File {
        asset,
        format: kind,
        persist_task_id,
    })
}
fn materialization_plan(
    input: &Input,
    files: &[ResultFile],
    credentials: CredentialAvailability,
    previous: &mut StreamHandle,
    transport: &mut dyn HunyuanTransport,
    media_owner: &mut dyn HunyuanMediaOwner,
    result: &mut DownloadedResult,
) -> Result<(), HunyuanError> {
    if is_generation(input) {
        let mut obj = SourceOutput::Absent;
        let mut texture = SourceOutput::Absent;
        let mut maps = vec![SourceOutput::Absent; 3];
        if let Some(file) = get_file(files, "obj", false)? {
            let asset = download(file, credentials, previous, transport, result)?;
            transport.check_cancelled(Some(*previous))?;
            let downloaded = result
                .assets
                .get(asset)
                .ok_or(HunyuanError::Response("Missing OBJ asset"))?;
            match media_owner.read_obj_archive(downloaded)? {
                None => {
                    obj = SourceOutput::Obj { asset, entry: None };
                    maps.fill(SourceOutput::ZeroRgb);
                }
                Some(entries) => {
                    let expanded = entries.iter().try_fold(0usize, |total, entry| {
                        total
                            .checked_add(entry.bytes.len())
                            .ok_or(ConformanceError::BodyLimit)
                    })?;
                    let total = result.assets.iter().try_fold(expanded, |total, asset| {
                        total
                            .checked_add(asset.bytes.len())
                            .ok_or(ConformanceError::BodyLimit)
                    })?;
                    if total > MAXIMUM_MEDIA_BYTES {
                        return Err(ConformanceError::BodyLimit.into());
                    }
                    let selected = scan_obj_entries(&entries)?;
                    for entry_index in &selected.all_images {
                        transport.check_cancelled(Some(*previous))?;
                        let entry = entries
                            .get(*entry_index)
                            .ok_or(HunyuanError::Response("Missing archive image"))?;
                        media_owner.validate_rgb_image(downloaded, *entry_index, entry)?;
                    }
                    if selected.obj.is_none() {
                        return Err(HunyuanError::Response(
                            "ZIP archive does not contain an OBJ file.",
                        ));
                    }
                    obj = SourceOutput::Obj {
                        asset,
                        entry: selected.obj,
                    };
                    texture = selected.texture.map_or(SourceOutput::Absent, |entry| {
                        SourceOutput::Image {
                            asset,
                            entry: Some(entry),
                            mode: "RGB",
                        }
                    });
                    maps = [selected.metallic, selected.normal, selected.roughness]
                        .into_iter()
                        .map(|entry| {
                            entry.map_or(SourceOutput::ZeroRgb, |entry| SourceOutput::Image {
                                asset,
                                entry: Some(entry),
                                mode: "RGB",
                            })
                        })
                        .collect();
                    result.archives.insert(asset, entries);
                }
            }
        }
        let persist_task_id = Some(result.task_id.clone());
        let glb = download_file(
            files,
            "glb",
            credentials,
            previous,
            transport,
            result,
            persist_task_id,
        )?;
        result.outputs = vec![
            SourceOutput::Text(format!("{}.glb", result.task_id)),
            glb,
            obj,
            texture,
        ];
        if input.node == "TencentImageToModelNode" {
            result.outputs.extend(maps);
        }
    } else if input.node == "TencentModelTo3DUVNode" {
        let image = if let Some(file) = get_file(files, "uv_image", false)? {
            let asset = download(file, credentials, previous, transport, result)?;
            transport.check_cancelled(Some(*previous))?;
            media_owner.validate_rgba_image(
                result
                    .assets
                    .get(asset)
                    .ok_or(HunyuanError::Response("Missing UV image asset"))?,
            )?;
            SourceOutput::Image {
                asset,
                entry: None,
                mode: "RGBA",
            }
        } else {
            transport.check_cancelled(Some(*previous))?;
            media_owner.stage_zero_rgb(*previous, &result.status_receipt)?;
            SourceOutput::ZeroRgb
        };
        let obj = download_file(files, "obj", credentials, previous, transport, result, None)?;
        let fbx = download_file(files, "fbx", credentials, previous, transport, result, None)?;
        result.outputs = vec![obj, fbx, image];
    } else if input.node == "Tencent3DTextureEditNode" {
        let glb = download_file(files, "glb", credentials, previous, transport, result, None)?;
        let obj = download_file(files, "obj", credentials, previous, transport, result, None)?;
        let asset = download(
            required_file(files, "texture_image")?,
            credentials,
            previous,
            transport,
            result,
        )?;
        transport.check_cancelled(Some(*previous))?;
        media_owner.validate_rgba_image(
            result
                .assets
                .get(asset)
                .ok_or(HunyuanError::Response("Missing texture image asset"))?,
        )?;
        let image = SourceOutput::Image {
            asset,
            entry: None,
            mode: "RGBA",
        };
        result.outputs = vec![glb, obj, image];
    } else {
        let kind = if input.node == "Tencent3DPartNode" {
            "fbx"
        } else {
            "obj"
        };
        let output = download_file(files, kind, credentials, previous, transport, result, None)?;
        result.outputs = vec![output];
    }
    Ok(())
}
#[derive(Clone, Copy)]
struct RetryPolicy {
    ordinary_limit: u32,
    rate_limit: u32,
    backoff: f64,
    download: bool,
    tencent_create: bool,
    timeout_milliseconds: Option<u64>,
}
const GENERATION_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 3,
    rate_limit: 16,
    backoff: 2.0,
    download: false,
    tencent_create: true,
    timeout_milliseconds: Some(3_600_000),
};
const POLL_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 10,
    rate_limit: 16,
    backoff: 1.4,
    download: false,
    tencent_create: false,
    timeout_milliseconds: Some(120_000),
};
const DOWNLOAD_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 5,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
    tencent_create: false,
    timeout_milliseconds: None,
};
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    previous: Option<StreamHandle>,
    transport: &mut dyn HunyuanTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), HunyuanError> {
    transport.check_cancelled(None)?;
    let handle = transport.start_request(head)?;
    checked_handle(handle, previous)?;
    exchange_existing(head, body, cost, maximum_bytes, policy, handle, transport)
}
fn exchange_existing(
    _head: &RequestHead,
    body: &[u8],
    cost: Option<u64>,
    maximum_bytes: usize,
    policy: RetryPolicy,
    mut handle: StreamHandle,
    transport: &mut dyn HunyuanTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), HunyuanError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, cost, maximum_bytes, policy, transport) {
            Err(HunyuanError::Http {
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
                    return Err(HunyuanError::Http {
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
                checked_handle(fresh, Some(handle))?;
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
    policy: RetryPolicy,
    transport: &mut dyn HunyuanTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), HunyuanError> {
    if let Some(maximum_microunits) = cost {
        let approval = transport.request_cost(
            handle,
            &CostRequest {
                operation: "hunyuan3d.generation.maximum".into(),
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
                    return Err(HunyuanError::Http {
                        status: status.unwrap_or_default(),
                        retry_after,
                        rate_limited: policy.tencent_create
                            && serde_json::from_slice::<Value>(&bytes).is_ok_and(|body| {
                                tencent_rate_limited(status.unwrap_or_default(), &body)
                            }),
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

pub trait HunyuanMediaMaterializer {
    // Validation includes every recognized archive image, even overwritten ones, in RGB mode.
    // ZeroRgb is a real [1,1,1,3] F32 allocation. Absent ports stay absent; all staging is atomic.
    fn materialize(
        &mut self,
        result: &DownloadedResult,
        archive_images: &[(usize, usize)],
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub struct HunyuanMaterializationOwners<'a> {
    pub decoded_media: &'a mut dyn HunyuanMediaOwner,
    pub media: &'a mut dyn HunyuanMediaMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    media: &[PreparedMedia],
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn HunyuanTransport,
    owners: &mut HunyuanMaterializationOwners<'_>,
) -> Result<(), HunyuanError> {
    let result = invoke(
        input,
        media,
        credentials,
        maximum_cost_microunits,
        transport,
        owners.decoded_media,
    )?;
    transport.check_cancelled(None)?;
    let archive_images =
        result
            .archives
            .iter()
            .try_fold(vec![], |mut indices, (asset, entries)| {
                indices.extend(
                    select_obj_entries(entries)?
                        .all_images
                        .into_iter()
                        .map(|entry| (*asset, entry)),
                );
                Ok::<_, HunyuanError>(indices)
            })?;
    let outputs = owners.media.materialize(&result, &archive_images)?;
    let registry = TypeRegistry::built_in()
        .map_err(|_| HunyuanError::Response("Missing canonical type registry"))?;
    let expected = result
        .outputs
        .iter()
        .enumerate()
        .filter_map(|(index, output)| output.alias().map(|alias| (index, alias)))
        .map(|(index, alias)| {
            let type_id = registry
                .resolve(alias)
                .map_err(|_| HunyuanError::Response("Missing canonical output type"))?
                .clone();
            let family = registry
                .family(&type_id)
                .map_err(|_| HunyuanError::Response("Missing canonical output family"))?;
            Ok(ExpectedOutput {
                port_id: format!("output_{index}"),
                type_id,
                family,
            })
        })
        .collect::<Result<Vec<_>, HunyuanError>>()?;
    let batch = validate_outputs(&expected, outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, owners.sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn source_oracle() -> Result<Value, ConformanceError> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/hunyuan3d/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture
        != json!({"schema_version":1,"namespace":NAMESPACE,"source_sha256":SOURCE_SHA256,"aliases":ALIASES,"nodes":NODES,"routes":ROUTES,"scope":"source request and bounded transport; canonical media integration pending"})
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let oracle = source_oracle()?;
    if oracle["source_sha256"]["comfy_api_nodes/nodes_hunyuan3d.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    if oracle["client_defaults"]["sync_op_raw"]["timeout"] != 3600.0
        || oracle["client_defaults"]["poll_op_raw"]["timeout_per_poll"] != 120.0
        || !oracle["client_defaults"]["download_url_to_bytesio"]["timeout"].is_null()
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in oracle["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let requirements =
            upload_requirements(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let uploads = requirements
            .iter()
            .map(|requirement| UploadedMedia {
                role: requirement.role.clone(),
                url: format!("https://storage.fixture.invalid/{}", requirement.role),
            })
            .collect::<Vec<_>>();
        let (path, body) =
            request(&input, &uploads).map_err(|_| ConformanceError::InvalidRequest)?;
        if case["request"]["path"] != path
            || case["request"]["method"] != "POST"
            || case["request"]["body"]
                != serde_json::from_slice::<Value>(&body)
                    .map_err(|_| ConformanceError::InvalidRequest)?
            || case["poll"]
                != json!({"path":format!("{path}/query"),"method":"POST","body":{"JobId":"source-task-42"}})
        {
            return Err(ConformanceError::InvalidRequest);
        }
        let observed = case["uploads"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if observed.len() != requirements.len()
            || requirements
                .iter()
                .zip(observed)
                .any(|(requirement, upload)| {
                    upload["role"] != requirement.role
                        || requirement
                            .width
                            .is_some_and(|width| upload["width"] != width)
                        || requirement
                            .height
                            .is_some_and(|height| upload["height"] != height)
                        || requirement.total_pixels.is_some_and(|pixels| {
                            upload["total_pixels"] != pixels
                                || upload["mime_type"] != requirement.content_type
                        })
                        || requirement.role == "model_3d" && upload["format"] != requirement.format
                })
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
        if upload_requirements(&input).is_ok() || case["requests"] != 0 || case["uploads"] != 0 {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    for case in oracle["rate_limit_cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let status = u16::try_from(
            case["status"]
                .as_u64()
                .ok_or(ConformanceError::InvalidRequest)?,
        )
        .map_err(|_| ConformanceError::InvalidRequest)?;
        if case["rate_limited"] != tencent_rate_limited(status, &case["body"]) {
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
        CostResponse, Progress, SignedPutRequest, StorageCreateRequest, StorageCreateResponse,
    };
    use std::collections::VecDeque;
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
        active: Option<(StreamHandle, Response, u64)>,
        next_slot: u32,
        uploads: Vec<PreparedMedia>,
        delays: Vec<u64>,
        heads: Vec<RequestHead>,
        costs: usize,
        deadlines: Vec<Option<u64>>,
        last_head: Option<RequestHead>,
        cancelled: bool,
        reject_cost: bool,
        stale_retry: bool,
        failure: Option<ConformanceError>,
    }
    impl TestTransport {
        fn new(responses: Vec<Response>) -> Self {
            Self {
                responses: responses.into(),
                active: None,
                next_slot: 1,
                uploads: vec![],
                delays: vec![],
                heads: vec![],
                costs: 0,
                deadlines: vec![],
                last_head: None,
                cancelled: false,
                reject_cost: false,
                stale_retry: false,
                failure: None,
            }
        }
        fn finish(&self) -> Result<(), Box<dyn Error>> {
            if !self.responses.is_empty() || self.active.is_some() {
                return Err("unconsumed provider script".into());
            }
            Ok(())
        }
    }

    #[test]
    fn test_transport_rejects_unequal_non_json_bodies() -> Result<(), Box<dyn Error>> {
        for (expected, actual, accepted) in [
            (b"{bad".as_slice(), b"{worse".as_slice(), false),
            (&[0, 255, 1], &[0, 255, 2], false),
            (&[0, 255, 1], &[0, 255, 1], true),
            (
                b"{\"one\":1,\"two\":2}".as_slice(),
                b"{\"two\":2,\"one\":1}".as_slice(),
                true,
            ),
        ] {
            let mut transport =
                TestTransport::new(vec![response("/fixture", expected.to_vec(), vec![])]);
            let handle = transport.start_request(&RequestHead {
                path: "/fixture".into(),
                method: HttpMethod::Post,
                secret_id: None,
                headers: vec![],
            })?;
            let result = transport.write_request_chunk(handle, 0, actual, true);
            if accepted {
                result?;
            } else {
                assert_eq!(result, Err(ConformanceError::InvalidRequest));
            }
        }
        Ok(())
    }
    impl ProviderTransport for TestTransport {
        fn check_cancelled(
            &mut self,
            _handle: Option<StreamHandle>,
        ) -> Result<(), ConformanceError> {
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
            if head.path != response.path || head.method != response.method {
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
            self.heads.push(head.clone());
            self.last_head = Some(head.clone());
            self.active = Some((handle, response, 0));
            Ok(handle)
        }
        fn write_request_chunk(
            &mut self,
            handle: StreamHandle,
            sequence: u64,
            bytes: &[u8],
            end: bool,
        ) -> Result<(), ConformanceError> {
            let (active, response, _) = self
                .active
                .as_ref()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle
                || sequence != 0
                || !end
                || response.body != bytes
                    && (!matches!(
                        (serde_json::from_slice::<Value>(&response.body), serde_json::from_slice::<Value>(bytes)),
                        (Ok(expected), Ok(actual)) if expected == actual
                    ))
            {
                return Err(ConformanceError::InvalidRequest);
            }
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
            self.costs += 1;
            Ok(CostResponse {
                accepted: !self.reject_cost,
                approved_microunits: request.maximum_microunits,
                receipt: b"cost receipt".to_vec(),
            })
        }
        fn report_progress(
            &mut self,
            _handle: StreamHandle,
            _progress: &Progress,
        ) -> Result<(), ConformanceError> {
            Err(ConformanceError::HostFailure)
        }
        fn wait_response(
            &mut self,
            handle: StreamHandle,
            after_sequence: Option<u64>,
            timeout: u64,
        ) -> Result<ResponseEvent, ConformanceError> {
            let (active, response, sequence) = self
                .active
                .as_mut()
                .ok_or(ConformanceError::InvalidHandle)?;
            if *active != handle || timeout != 1000 || after_sequence != sequence.checked_sub(1) {
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
    impl HunyuanTransport for TestTransport {
        fn begin_response_deadline(
            &mut self,
            handle: StreamHandle,
            timeout_milliseconds: Option<u64>,
        ) -> Result<(), ConformanceError> {
            if self
                .active
                .as_ref()
                .is_none_or(|(active, _, _)| *active != handle)
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.deadlines.push(timeout_milliseconds);
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
                .last_head
                .clone()
                .ok_or(ConformanceError::InvalidRequest)?;
            self.start_request(&head)
        }
        fn delay(
            &mut self,
            _handle: StreamHandle,
            milliseconds: u64,
        ) -> Result<(), ConformanceError> {
            self.delays.push(milliseconds);
            Ok(())
        }
        fn upload_prepared(
            &mut self,
            _handle: StreamHandle,
            media: &PreparedMedia,
            wait_label: &str,
        ) -> Result<UploadedMedia, ConformanceError> {
            if wait_label != "Uploading" || media.bytes.is_empty() {
                return Err(ConformanceError::InvalidRequest);
            }
            self.uploads.push(media.clone());
            Ok(UploadedMedia {
                role: media.requirement.role.clone(),
                url: format!("https://storage.fixture.invalid/{}", media.requirement.role),
            })
        }
    }
    struct ArchiveOwner {
        entries: Option<Vec<DecodedObjArchiveEntry>>,
        fail: bool,
        observed_receipts: Vec<Vec<u8>>,
        fail_rgba: bool,
        rgba_urls: Vec<String>,
        zeros: usize,
    }
    impl HunyuanMediaOwner for ArchiveOwner {
        fn validate_rgba_image(&mut self, asset: &DownloadedAsset) -> Result<(), ConformanceError> {
            if self.fail_rgba {
                return Err(ConformanceError::HostFailure);
            }
            if asset.bytes.is_empty() || asset.response_receipt.is_empty() {
                return Err(ConformanceError::InvalidResponse);
            }
            self.rgba_urls.push(asset.source_url.clone());
            Ok(())
        }
        fn stage_zero_rgb(
            &mut self,
            handle: StreamHandle,
            receipt: &[u8],
        ) -> Result<(), ConformanceError> {
            if handle.invocation != 42 || receipt.is_empty() {
                return Err(ConformanceError::InvalidHandle);
            }
            self.zeros += 1;
            Ok(())
        }

        fn read_obj_archive(
            &mut self,
            asset: &DownloadedAsset,
        ) -> Result<Option<Vec<DecodedObjArchiveEntry>>, ConformanceError> {
            if self.fail {
                return Err(ConformanceError::BodyLimit);
            }
            if asset.bytes.is_empty()
                || asset.response_receipt.is_empty()
                || asset.handle.invocation != 42
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.observed_receipts.push(asset.response_receipt.clone());
            Ok(self.entries.clone())
        }
        fn validate_rgb_image(
            &mut self,
            asset: &DownloadedAsset,
            _entry_index: usize,
            entry: &DecodedObjArchiveEntry,
        ) -> Result<(), ConformanceError> {
            if asset.response_receipt.is_empty() || entry.bytes.is_empty() {
                return Err(ConformanceError::InvalidResponse);
            }
            Ok(())
        }
    }
    fn owner(raw: bool) -> ArchiveOwner {
        ArchiveOwner {
            entries: if raw {
                None
            } else {
                Some(vec![
                    DecodedObjArchiveEntry {
                        name: "model.obj".into(),
                        bytes: b"actual OBJ mesh bytes".to_vec(),
                    },
                    DecodedObjArchiveEntry {
                        name: "texture.png".into(),
                        bytes: vec![137, 80, 78, 71],
                    },
                ])
            },
            fail: false,
            observed_receipts: vec![],
            fail_rgba: false,
            rgba_urls: vec![],
            zeros: 0,
        }
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn response(path: &str, body: Vec<u8>, bytes: Vec<u8>) -> Response {
        Response {
            path: path.into(),
            method: if path.starts_with("https://") {
                HttpMethod::Get
            } else {
                HttpMethod::Post
            },
            body,
            status: 200,
            bytes,
            headers: vec![],
            receipt: format!("receipt:{path}").into_bytes(),
        }
    }
    fn case_input(case: &Value) -> Result<Input, Box<dyn Error>> {
        Ok(serde_json::from_value(case["input"].clone())?)
    }
    fn prepared(input: &Input) -> Result<Vec<PreparedMedia>, HunyuanError> {
        Ok(upload_requirements(input)?
            .into_iter()
            .map(|requirement| PreparedMedia {
                file_name: format!("source.{}", requirement.format),
                requirement,
                bytes: b"actual caller prepared bytes".to_vec(),
            })
            .collect())
    }
    fn script(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let mut responses = vec![
            response(
                case["request"]["path"].as_str().ok_or("missing path")?,
                serde_json::to_vec(&case["request"]["body"])?,
                serde_json::to_vec(&json!({"Response":{"JobId":"source-task-42"}}))?,
            ),
            response(
                case["poll"]["path"].as_str().ok_or("missing poll")?,
                serde_json::to_vec(&case["poll"]["body"])?,
                serde_json::to_vec(
                    &json!({"Response":{"Status":"Done","ResultFile3Ds":case["files"]}}),
                )?,
            ),
        ];
        for download in case["downloads"].as_array().ok_or("missing downloads")? {
            let url = download["url"].as_str().ok_or("missing URL")?;
            responses.push(response(
                url,
                vec![],
                format!("actual bytes from {url}").into_bytes(),
            ));
        }
        Ok(responses)
    }
    fn normalized_output(
        output: &SourceOutput,
        result: &DownloadedResult,
    ) -> Result<Value, Box<dyn Error>> {
        Ok(match output {
            SourceOutput::Absent => Value::Null,
            SourceOutput::Text(value) => json!(value),
            SourceOutput::File {
                asset,
                format,
                persist_task_id,
            } => {
                let url = &result
                    .assets
                    .get(*asset)
                    .ok_or("missing file asset")?
                    .source_url;
                let mut value = json!({"download":url,"format":format});
                if let Some(task_id) = persist_task_id {
                    value["task_id"] = json!(task_id);
                }
                value
            }
            SourceOutput::Image { asset, entry, mode } => {
                if entry.is_some() {
                    json!("source RGB texture")
                } else {
                    json!({"download":result.assets.get(*asset).ok_or("missing image asset")?.source_url,"mode":mode})
                }
            }
            SourceOutput::Obj { .. } => {
                json!({"file_format":"obj","bytes_hex":"61637475616c20736f75726365206d6f64656c206279746573"})
            }
            SourceOutput::ZeroRgb => json!({"shape":[1,1,1,3],"pixels":[0.0,0.0,0.0]}),
        })
    }
    #[test]
    fn pinned_reverse_request_upload_and_output_order_all_six_nodes() -> Result<(), Box<dyn Error>>
    {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/hunyuan3d/conformance.fixture"
        )))?;
        let oracle = source_oracle()?;
        for case in oracle["cases"].as_array().ok_or("missing cases")? {
            let input = case_input(case)?;
            let media = prepared(&input)?;
            let mut transport = TestTransport::new(script(case)?);
            let mut archive = owner(case["obj_mode"] == "raw");
            let result = invoke(
                &input,
                &media,
                credentials(),
                1_000_000,
                &mut transport,
                &mut archive,
            )?;
            assert_eq!(transport.costs, 1);
            assert_eq!(transport.deadlines.first(), Some(&Some(3_600_000)));
            assert_eq!(transport.deadlines.get(1), Some(&Some(120_000)));
            assert!(transport.deadlines.iter().skip(2).all(Option::is_none));
            assert_eq!(transport.uploads, media);
            let normalized = result
                .outputs
                .iter()
                .map(|output| normalized_output(output, &result))
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(json!(normalized), case["outputs"], "{}", input.node);
            assert_eq!(
                result.assets.len(),
                case["downloads"]
                    .as_array()
                    .ok_or("missing downloads")?
                    .len()
            );
            assert!(!result.submission_receipt.is_empty());
            assert!(!result.status_receipt.is_empty());
            transport.finish()?;
        }
        Ok(())
    }
    #[test]
    fn source_preflight_seeds_nulls_unicode_and_resize_ties() -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        for case in oracle["preflight_errors"]
            .as_array()
            .ok_or("missing invalid cases")?
        {
            assert!(upload_requirements(&case_input(case)?).is_err());
        }
        assert_eq!(
            source_downscale(&ImageDimensions {
                width: 9800,
                height: 257
            })?,
            (4900, 128)
        );
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .find(|case| case["input"]["node"] == "TencentTextToModelNode")
            .ok_or("missing text")?;
        let mut input = case_input(case)?;
        let (_, first) = request(&input, &[])?;
        input.seed = 2147483647;
        assert_eq!(request(&input, &[])?.1, first);
        input.generate_type = GenerateType {
            generate_type: "Geometry".into(),
            pbr: None,
            polygon_type: None,
        };
        let body: Value = serde_json::from_slice(&request(&input, &[])?.1)?;
        assert!(body.get("EnablePBR").is_none());
        assert!(body.get("PolygonType").is_none());
        assert!(body.get("seed").is_none());
        assert!(validate_prompt("☁".repeat(1024).as_str()).is_ok());
        assert!(validate_prompt(" \u{1c} ").is_err());
        Ok(())
    }
    #[test]
    fn response_wrapper_first_match_errors_and_status_aliases() -> Result<(), Box<dyn Error>> {
        assert_eq!(
            submitted_task(br#"{"Response":{"JobId":"wrapped"},"JobId":"outer"}"#)?,
            "wrapped"
        );
        assert!(submitted_task(br#"{"Response":{"Error":{},"JobId":"ignored"}}"#).is_err());
        assert!(submitted_task(br#"{"JobId":null}"#).is_err());
        let files: Vec<ResultFile> = serde_json::from_value(
            json!([{"Type":"Obj","Url":"https://first.invalid"},{"Type":"OBJ","Url":"https://second.invalid"}]),
        )?;
        assert_eq!(required_file(&files, "obj")?.url, "https://first.invalid");
        assert!(get_file(&files, "fbx", true).is_err());
        assert!(get_file(&files, "fbx", false)?.is_none());
        for status in [
            " Succeeded ",
            "succeed",
            "success",
            "completed",
            "finished",
            "done",
            "complete",
        ] {
            assert_eq!(
                task_status(&serde_json::to_vec(&json!({"Status":status}))?)?.1,
                Phase::Completed
            );
        }
        for status in [
            "created",
            "queued",
            "queueing",
            "submitted",
            "initializing",
            "wait",
            "in_queue",
        ] {
            assert_eq!(
                task_status(&serde_json::to_vec(&json!({"Status":status}))?)?.1,
                Phase::Queued
            );
        }
        for status in [
            "cancelled",
            "canceled",
            "canceling",
            "fail",
            "failed",
            "error",
        ] {
            assert!(task_status(&serde_json::to_vec(&json!({"Status":status}))?).is_err());
        }
        assert_eq!(
            task_status(br#"{"Status":"working"}"#)?.1,
            Phase::Processing
        );
        Ok(())
    }
    #[test]
    fn archive_source_order_suffixes_and_mandatory_mesh() -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let case = oracle["archive_cases"]
            .as_array()
            .ok_or("missing archives")?
            .first()
            .ok_or("missing ordered archive")?;
        let entries = case["entries"]
            .as_array()
            .ok_or("missing entries")?
            .iter()
            .map(|entry| {
                Ok(DecodedObjArchiveEntry {
                    name: entry["name"].as_str().ok_or("missing name")?.into(),
                    bytes: vec![1],
                })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let selected = select_obj_entries(&entries)?;
        assert_eq!(selected.obj, Some(7));
        assert_eq!(selected.texture, Some(6));
        assert_eq!(selected.metallic, Some(3));
        assert_eq!(selected.normal, Some(4));
        assert_eq!(selected.roughness, Some(5));
        assert_eq!(selected.all_images, vec![2, 3, 4, 5, 6]);
        assert_eq!(case["result"]["obj"]["bytes_hex"], "6c617374206d657368");
        assert_eq!(case["result"]["texture"]["shape"], json!([1, 1, 1, 3]));
        assert!(
            select_obj_entries(&[DecodedObjArchiveEntry {
                name: "texture.PNG".into(),
                bytes: vec![1]
            }])
            .is_err()
        );
        assert!(
            select_obj_entries(&vec![
                DecodedObjArchiveEntry {
                    name: "mesh.obj".into(),
                    bytes: vec![1]
                };
                257
            ])
            .is_err()
        );
        Ok(())
    }
    fn exchange_script(statuses: &[u16], body: &[u8], tencent: bool) -> TestTransport {
        let path = "/proxy/tencent/hunyuan/3d-part";
        TestTransport::new(
            statuses
                .iter()
                .map(|status| {
                    let mut response = response(
                        path,
                        body.to_vec(),
                        if tencent {
                            br#"{"Response":{"Error":{"Code":"RequestLimitExceeded.test"}}}"#
                                .to_vec()
                        } else {
                            b"response".to_vec()
                        },
                    );
                    response.status = *status;
                    response
                })
                .collect(),
        )
    }
    #[test]
    fn tencent_400_rate_limit_only_creates_and_receipt_preserving_retries()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/tencent/hunyuan/3d-part",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut transport = exchange_script(&[400, 400, 200], b"{}", true);
        let (_, receipt, handle) = exchange_with_policy(
            &head,
            b"{}",
            None,
            1024,
            GENERATION_RETRY,
            None,
            &mut transport,
        )?;
        assert_eq!(transport.delays, vec![1000, 2000]);
        assert_eq!(handle.slot, 3);
        assert!(!receipt.is_empty());
        transport.finish()?;
        let mut transport = exchange_script(&[400], b"{}", true);
        assert!(matches!(
            exchange_with_policy(&head, b"{}", None, 1024, POLL_RETRY, None, &mut transport),
            Err(HunyuanError::Http {
                status: 400,
                rate_limited: false,
                ..
            })
        ));
        assert!(transport.delays.is_empty());
        let mut transport = exchange_script(&[400, 200], b"{}", true);
        transport.stale_retry = true;
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                1024,
                GENERATION_RETRY,
                None,
                &mut transport
            ),
            Err(HunyuanError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn retries_limits_retry_after_and_download_credential_isolation() -> Result<(), Box<dyn Error>>
    {
        let head = request_head(
            "/proxy/tencent/hunyuan/3d-part",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut transport = exchange_script(&[503, 503, 503, 503], b"{}", false);
        assert!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                1024,
                GENERATION_RETRY,
                None,
                &mut transport
            )
            .is_err()
        );
        assert_eq!(transport.delays, vec![1000, 2000, 4000]);
        transport.finish()?;
        let mut transport = exchange_script(&[429, 200], b"{}", false);
        transport
            .responses
            .front_mut()
            .ok_or("missing response")?
            .headers
            .push(Header {
                name: "Retry-After".into(),
                value: "999".into(),
            });
        exchange_with_policy(
            &head,
            b"{}",
            None,
            1024,
            GENERATION_RETRY,
            None,
            &mut transport,
        )?;
        assert_eq!(transport.delays, vec![150000]);
        transport.finish()?;
        let downloaded = request_head(
            "https://download.fixture.invalid/actual.obj",
            HttpMethod::Get,
            credentials(),
            true,
        )?;
        assert!(downloaded.secret_id.is_none());
        assert_eq!(downloaded.headers.len(), 1);
        assert!(
            request_head(
                "https://user@evil.invalid/x",
                HttpMethod::Get,
                credentials(),
                true
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn queued_polls_do_not_consume_active_budget_and_processing_is_bounded()
    -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing cases")?
            .first()
            .ok_or("missing first")?;
        let input = case_input(case)?;
        let mut script = script(case)?;
        let query = script.remove(1);
        let download = script.remove(1);
        for _ in 0..481 {
            script.push(response(
                &query.path,
                query.body.clone(),
                br#"{"Response":{"Status":"queued"}}"#.to_vec(),
            ));
        }
        script.push(query);
        script.push(download);
        let mut transport = TestTransport::new(script);
        let result = invoke(
            &input,
            &prepared(&input)?,
            credentials(),
            1_000_000,
            &mut transport,
            &mut owner(false),
        )?;
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(transport.delays.len(), 481);
        transport.finish()?;
        let mut script = script_for_processing(case)?;
        let mut transport = TestTransport::new(std::mem::take(&mut script));
        assert!(matches!(
            invoke(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut owner(false)
            ),
            Err(HunyuanError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.delays.len(), 480);
        transport.finish()?;
        Ok(())
    }
    fn script_for_processing(case: &Value) -> Result<Vec<Response>, Box<dyn Error>> {
        let mut responses = script(case)?;
        responses.truncate(1);
        for _ in 0..480 {
            responses.push(response(
                case["poll"]["path"].as_str().ok_or("missing poll")?,
                serde_json::to_vec(&case["poll"]["body"])?,
                br#"{"Response":{"Status":"processing"}}"#.to_vec(),
            ));
        }
        Ok(responses)
    }
    #[test]
    fn cancellation_worker_loss_cost_and_bad_response_fail_closed() -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing cases")?
            .first()
            .ok_or("missing first")?;
        let input = case_input(case)?;
        let mut transport = TestTransport::new(script(case)?);
        transport.cancelled = true;
        assert!(matches!(
            invoke(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut owner(false)
            ),
            Err(HunyuanError::Transport(ConformanceError::Cancelled))
        ));
        assert_eq!(transport.costs, 0);
        assert!(transport.heads.is_empty());
        let mut transport = TestTransport::new(script(case)?);
        transport.failure = Some(ConformanceError::WorkerLost);
        assert!(matches!(
            invoke(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut owner(false)
            ),
            Err(HunyuanError::Transport(ConformanceError::WorkerLost))
        ));
        let mut transport = TestTransport::new(script(case)?);
        transport.reject_cost = true;
        assert!(matches!(
            invoke(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut owner(false)
            ),
            Err(HunyuanError::Transport(ConformanceError::InvalidCost))
        ));
        assert!(transport.uploads.is_empty());
        let head = request_head(
            "/proxy/tencent/hunyuan/3d-part",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        let mut transport = exchange_script(&[200], b"{}", false);
        transport
            .responses
            .front_mut()
            .ok_or("missing response")?
            .receipt
            .clear();
        assert!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                1024,
                GENERATION_RETRY,
                None,
                &mut transport
            )
            .is_err()
        );
        let mut transport = exchange_script(&[200], b"{}", false);
        transport
            .responses
            .front_mut()
            .ok_or("missing response")?
            .bytes = vec![0; MAXIMUM_CHUNK_BYTES + 1];
        assert!(matches!(
            exchange_with_policy(
                &head,
                b"{}",
                None,
                MAXIMUM_MEDIA_BYTES,
                GENERATION_RETRY,
                None,
                &mut transport
            ),
            Err(HunyuanError::Transport(ConformanceError::ChunkLimit))
        ));
        Ok(())
    }
    struct FailingMaterializer {
        observed_images: Vec<(usize, usize)>,
    }
    impl HunyuanMediaMaterializer for FailingMaterializer {
        fn materialize(
            &mut self,
            result: &DownloadedResult,
            archive_images: &[(usize, usize)],
        ) -> Result<Vec<EncodedOutput>, ConformanceError> {
            if result
                .assets
                .iter()
                .any(|asset| asset.bytes.is_empty() || asset.response_receipt.is_empty())
            {
                return Err(ConformanceError::InvalidHandle);
            }
            self.observed_images = archive_images.to_vec();
            Err(ConformanceError::HostFailure)
        }
    }
    #[derive(Default)]
    struct Sink {
        publications: usize,
    }
    impl MaterializationSink for Sink {
        fn publish(
            &mut self,
            _outputs: crate::common::MaterializedBatch,
        ) -> Result<(), crate::common::MaterializationError> {
            self.publications += 1;
            Ok(())
        }
    }
    #[test]
    fn actual_receipts_and_all_archive_images_reach_owner_without_partial_publication()
    -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .find(|case| case["input"]["node"] == "TencentImageToModelNode")
            .ok_or("missing image case")?;
        let input = case_input(case)?;
        let mut archive = owner(false);
        archive.entries.as_mut().ok_or("missing archive")?.insert(
            1,
            DecodedObjArchiveEntry {
                name: "old.png".into(),
                bytes: b"invalid overwritten image".to_vec(),
            },
        );
        let mut transport = TestTransport::new(script(case)?);
        let mut materializer = FailingMaterializer {
            observed_images: vec![],
        };
        let mut sink = Sink::default();
        assert!(
            invoke_and_materialize(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut HunyuanMaterializationOwners {
                    decoded_media: &mut archive,
                    media: &mut materializer,
                    sink: &mut sink
                }
            )
            .is_err()
        );
        assert_eq!(materializer.observed_images, vec![(0, 1), (0, 2)]);
        assert_eq!(sink.publications, 0);
        assert_eq!(archive.observed_receipts.len(), 1);
        transport.finish()?;
        let mut archive = owner(false);
        archive.fail = true;
        let mut transport = TestTransport::new(script(case)?);
        assert!(
            invoke_and_materialize(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut HunyuanMaterializationOwners {
                    decoded_media: &mut archive,
                    media: &mut materializer,
                    sink: &mut sink
                }
            )
            .is_err()
        );
        assert_eq!(sink.publications, 0);
        Ok(())
    }
    #[test]
    fn uv_rgba_failure_aborts_before_obj_fbx_downloads() -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .find(|case| {
                case["input"]["node"] == "TencentModelTo3DUVNode"
                    && case["downloads"].as_array().is_some_and(|downloads| {
                        downloads
                            .first()
                            .is_some_and(|download| download["kind"] == "image")
                    })
            })
            .ok_or("missing UV image case")?;
        let input = case_input(case)?;
        let mut transport = TestTransport::new(script(case)?);
        let mut media_owner = owner(false);
        media_owner.fail_rgba = true;
        let mut materializer = FailingMaterializer {
            observed_images: vec![],
        };
        let mut sink = Sink::default();
        assert!(
            invoke_and_materialize(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut HunyuanMaterializationOwners {
                    decoded_media: &mut media_owner,
                    media: &mut materializer,
                    sink: &mut sink
                }
            )
            .is_err()
        );
        assert_eq!(transport.heads.len(), 3);
        assert_eq!(transport.responses.len(), 2);
        assert_eq!(sink.publications, 0);
        Ok(())
    }
    #[test]
    fn hunyuan_python_status_strip_preserves_raw_fields_and_negative_controls()
    -> Result<(), Box<dyn Error>> {
        for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            for (label, expected) in [
                (" CoMpLeTeD ", Phase::Completed),
                (" QuEuEd ", Phase::Queued),
            ] {
                let raw = format!("{separator}{label}{separator}");
                for wrapped in [false, true] {
                    let response = json!({"Status": raw, "ErrorCode":"raw-code", "ErrorMessage":"raw-message"});
                    let body = if wrapped {
                        json!({"Response":response})
                    } else {
                        response
                    };
                    let (status, phase) = task_status(&serde_json::to_vec(&body)?)?;
                    assert_eq!(phase, expected);
                    assert_eq!(status.status, raw);
                    assert_eq!(status.error_code, "raw-code");
                    assert_eq!(status.error_message, "raw-message");
                }
            }
            let raw = format!("{separator} FaIlEd {separator}");
            let body = json!({"Status": raw});
            match task_status(&serde_json::to_vec(&body)?) {
                Err(HunyuanError::Provider(value)) => {
                    assert_eq!(serde_json::from_str::<Value>(&value)?, body);
                }
                result => return Err(format!("expected failed response, got {result:?}").into()),
            }
        }
        for separator in ['\u{200b}', '\u{feff}'] {
            for label in ["completed", "failed", "queued"] {
                let raw = format!("{separator}{label}{separator}");
                let (status, phase) = task_status(&serde_json::to_vec(&json!({"Status":raw}))?)?;
                assert_eq!(phase, Phase::Processing);
                assert_eq!(status.status, raw);
            }
        }
        Ok(())
    }

    #[test]
    fn hunyuan_python_status_strip_drives_six_node_invoke_success_and_failure()
    -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let mut seen = std::collections::BTreeSet::new();
        for case in oracle["cases"].as_array().ok_or("missing cases")? {
            let input = case_input(case)?;
            if !seen.insert(input.node.clone()) {
                continue;
            }
            for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
                for failed in [false, true] {
                    let raw = format!(
                        "{separator} {} {separator}",
                        if failed { "FaIlEd" } else { "CoMpLeTeD" }
                    );
                    let mut responses = script(case)?;
                    let poll = responses.get_mut(1).ok_or("missing poll response")?;
                    let mut body: Value = serde_json::from_slice(&poll.bytes)?;
                    body["Response"]["Status"] = json!(raw);
                    poll.bytes = serde_json::to_vec(&body)?;
                    let mut transport = TestTransport::new(responses);
                    let result = invoke(
                        &input,
                        &prepared(&input)?,
                        credentials(),
                        1_000_000,
                        &mut transport,
                        &mut owner(case["obj_mode"] == "raw"),
                    );
                    assert_eq!(transport.costs, 1);
                    assert!(transport.delays.is_empty());
                    if failed {
                        match result {
                            Err(HunyuanError::Provider(value)) => {
                                assert_eq!(serde_json::from_str::<Value>(&value)?, body);
                            }
                            result => {
                                return Err(
                                    format!("expected failed response, got {result:?}").into()
                                );
                            }
                        }
                        assert_eq!(transport.heads.len(), 2);
                        assert_eq!(
                            transport.responses.len(),
                            case["downloads"]
                                .as_array()
                                .ok_or("missing downloads")?
                                .len()
                        );
                    } else {
                        let result = result?;
                        let outputs = result
                            .outputs
                            .iter()
                            .map(|output| normalized_output(output, &result))
                            .collect::<Result<Vec<_>, _>>()?;
                        assert_eq!(json!(outputs), case["outputs"]);
                        assert!(!result.status_receipt.is_empty());
                        transport.finish()?;
                    }
                }
            }
        }
        assert_eq!(seen.len(), 6);
        Ok(())
    }

    #[test]
    fn hunyuan_python_status_strip_retains_queued_budget_and_unstripped_processing()
    -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing cases")?
            .first()
            .ok_or("missing case")?;
        let input = case_input(case)?;
        for (raw, poll_count) in [
            ("\u{1c} QuEuEd \u{1f}", 481usize),
            ("\u{200b}completed\u{200b}", 1),
            ("\u{feff}completed\u{feff}", 1),
        ] {
            let mut responses = script(case)?;
            let query = responses.get(1).ok_or("missing query")?;
            let bytes = serde_json::to_vec(&json!({"Response":{"Status":raw}}))?;
            let polls = (0..poll_count)
                .map(|_| response(&query.path, query.body.clone(), bytes.clone()))
                .collect::<Vec<_>>();
            responses.splice(1..1, polls);
            let mut transport = TestTransport::new(responses);
            let result = invoke(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut owner(case["obj_mode"] == "raw"),
            )?;
            assert!(!result.outputs.is_empty());
            assert_eq!(transport.costs, 1);
            assert_eq!(transport.delays, vec![5000; poll_count]);
            transport.finish()?;
        }
        Ok(())
    }

    #[test]
    fn hunyuan_failed_poll_retains_complete_wrapped_and_unwrapped_responses()
    -> Result<(), Box<dyn Error>> {
        for wrapped in [false, true] {
            for raw_status in ["Failed", "\u{1c} FaIlEd \u{1f}", "Canceled"] {
                let response = json!({
                    "Status": raw_status,
                    "ErrorCode": "Provider.InvalidGeometry",
                    "ErrorMessage": "mesh \"bad\"\n非ASCII 💠",
                    "ResultFile3Ds": [],
                    "Details": {"count": 42, "missing": null, "keys": ["one", "two"]}
                });
                let body = if wrapped {
                    json!({"Response": response, "RequestId": "raw-outer-request", "Trace": ["outer", null]})
                } else {
                    response
                };
                match task_status(&serde_json::to_vec(&body)?) {
                    Err(HunyuanError::Provider(value)) => {
                        assert_eq!(serde_json::from_str::<Value>(&value)?, body);
                        assert_eq!(
                            HunyuanError::Provider(value.clone()).to_string(),
                            format!("Hunyuan3D task failed: {value}")
                        );
                    }
                    result => {
                        return Err(format!("expected failed response, got {result:?}").into());
                    }
                }
            }
        }
        let creation_error = json!({"Error": {"Code": "creation", "Message": "unchanged"}});
        match submitted_task(&serde_json::to_vec(&creation_error)?) {
            Err(HunyuanError::Provider(value)) => {
                assert_eq!(serde_json::from_str::<Value>(&value)?, creation_error["Error"]);
            }
            result => return Err(format!("expected creation error, got {result:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn hunyuan_failed_poll_aborts_six_node_download_and_publication_with_service_details()
    -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let mut seen = BTreeSet::new();
        for case in oracle["cases"].as_array().ok_or("missing cases")? {
            let input = case_input(case)?;
            if !seen.insert(input.node.clone()) {
                continue;
            }
            for wrapped in [false, true] {
                let response = json!({
                    "Status": "\u{1c} FAILED \u{1f}",
                    "ErrorCode": "Provider.InvalidGeometry",
                    "ErrorMessage": "mesh \"bad\"\n非ASCII 💠",
                    "Details": {"node": input.node, "missing": null}
                });
                let body = if wrapped {
                    json!({"Response": response, "RequestId": "raw-outer-request"})
                } else {
                    response
                };
                let mut responses = script(case)?;
                responses.get_mut(1).ok_or("missing query response")?.bytes =
                    serde_json::to_vec(&body)?;
                let mut transport = TestTransport::new(responses);
                let mut archive = owner(case["obj_mode"] == "raw");
                let mut materializer = FailingMaterializer {
                    observed_images: vec![],
                };
                let mut sink = Sink::default();
                let result = invoke_and_materialize(
                    &input,
                    &prepared(&input)?,
                    credentials(),
                    1_000_000,
                    &mut transport,
                    &mut HunyuanMaterializationOwners {
                        decoded_media: &mut archive,
                        media: &mut materializer,
                        sink: &mut sink,
                    },
                );
                match result {
                    Err(HunyuanError::Provider(value)) => {
                        assert_eq!(serde_json::from_str::<Value>(&value)?, body);
                    }
                    result => {
                        return Err(format!("expected failed response, got {result:?}").into());
                    }
                }
                assert_eq!(transport.costs, 1);
                assert_eq!(transport.heads.len(), 2);
                assert!(transport.delays.is_empty());
                assert!(transport.active.is_none());
                assert_eq!(
                    transport.responses.len(),
                    case["downloads"]
                        .as_array()
                        .ok_or("missing downloads")?
                        .len()
                );
                assert!(archive.observed_receipts.is_empty());
                assert!(materializer.observed_images.is_empty());
                assert_eq!(sink.publications, 0);
            }
        }
        assert_eq!(seen.len(), 6);
        Ok(())
    }

    #[test]
    fn hunyuan_failed_poll_retention_preserves_response_limits_cancellation_and_total_queue_bound()
    -> Result<(), Box<dyn Error>> {
        assert!(matches!(
            task_status(&vec![b' '; MAXIMUM_RESPONSE_BYTES + 1]),
            Err(HunyuanError::Transport(ConformanceError::BodyLimit))
        ));
        assert!(matches!(task_status(b"{bad"), Err(HunyuanError::Json(_))));
        assert!(matches!(
            task_status(br#"{"Status":"Failed","ErrorMessage":null}"#),
            Err(HunyuanError::Json(_))
        ));
        let oracle = source_oracle()?;
        let case = oracle["cases"]
            .as_array()
            .ok_or("missing cases")?
            .first()
            .ok_or("missing case")?;
        let input = case_input(case)?;
        let mut responses = script(case)?;
        let query = responses.get(1).ok_or("missing query response")?;
        let path = query.path.clone();
        let body = query.body.clone();
        responses.truncate(1);
        let total_polls = usize::try_from(MAXIMUM_TOTAL_POLLS)?;
        for _ in 0..total_polls {
            responses.push(response(&path, body.clone(), serde_json::to_vec(&json!({
                "Response": {"Status": "Queued", "ErrorCode": "not-terminal", "ErrorMessage": "not a failure"}
            }))?));
        }
        let mut transport = TestTransport::new(responses);
        assert!(matches!(
            invoke(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut owner(false)
            ),
            Err(HunyuanError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.costs, 1);
        assert_eq!(transport.heads.len(), total_polls + 1);
        assert_eq!(transport.delays, vec![5000; total_polls]);
        transport.finish()?;
        let mut transport = TestTransport::new(script(case)?);
        transport.cancelled = true;
        assert!(matches!(
            invoke(
                &input,
                &prepared(&input)?,
                credentials(),
                1_000_000,
                &mut transport,
                &mut owner(false)
            ),
            Err(HunyuanError::Transport(ConformanceError::Cancelled))
        ));
        assert_eq!(transport.costs, 0);
        assert!(transport.heads.is_empty());
        assert!(transport.delays.is_empty());
        Ok(())
    }
}
