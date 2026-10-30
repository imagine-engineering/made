use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, ProviderTransport, RequestHead, ResponseEvent, StreamHandle, TypeRegistry,
    publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, error::Error, fmt};
pub const NAMESPACE: &str = "zed.sim.provider.magnific";
pub const SOURCE_SHA256: &str = "316eddd16e8b8f9862e1f7ac95a355717c24a4f8be72eee5f2a0b2844dc731de";
pub const ALIASES: &[&str] = &["freepik"];
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0391",
        node_identifier: "MagnificImageRelightNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0392",
        node_identifier: "MagnificImageSkinEnhancerNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0393",
        node_identifier: "MagnificImageStyleTransferNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0394",
        node_identifier: "MagnificImageUpscalerCreativeNode",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0395",
        node_identifier: "MagnificImageUpscalerPreciseV2Node",
    },
];
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0034",
        method: "POST",
        path: "/proxy/freepik/v1/ai/image-relight",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0035",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-relight/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0036",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-relight/{initial_res.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0037",
        method: "POST",
        path: "/proxy/freepik/v1/ai/image-style-transfer",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0038",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-style-transfer/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0039",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-style-transfer/{initial_res.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0040",
        method: "POST",
        path: "/proxy/freepik/v1/ai/image-upscaler",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0041",
        method: "POST",
        path: "/proxy/freepik/v1/ai/image-upscaler-precision-v2",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0042",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-upscaler-precision-v2/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0043",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-upscaler-precision-v2/{initial_res.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0044",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-upscaler/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0045",
        method: "GET",
        path: "/proxy/freepik/v1/ai/image-upscaler/{initial_res.task_id}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0046",
        method: "POST",
        path: "/proxy/freepik/v1/ai/skin-enhancer/",
        disposition: "synthetic_prefix_tombstone",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0047",
        method: "POST",
        path: "/proxy/freepik/v1/ai/skin-enhancer/{endpoint}",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0048",
        method: "GET",
        path: "/proxy/freepik/v1/ai/skin-enhancer/{initial_res.task_id}",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum MagnificError {
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
impl fmt::Display for MagnificError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(status) => write!(formatter, "Magnific job failed: {status}"),
            Self::Http { status, .. } => write!(formatter, "Magnific HTTP {status}"),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for MagnificError {}
impl From<ConformanceError> for MagnificError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for MagnificError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
    pub batch: u32,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Input {
    pub node: String,
    pub settings: Value,
    pub image: ImageDimensions,
    pub reference_image: Option<ImageDimensions>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UploadRequirement {
    pub role: String,
    pub source: ImageDimensions,
    pub target_width: u32,
    pub target_height: u32,
    pub explicit_resize_pixels: Option<u64>,
    pub encoding_pixels: Option<u64>,
    pub encoded_mime_type: &'static str,
    pub declared_content_type: Option<&'static str>,
}
#[derive(Clone, Debug)]
pub struct PreparationPlan {
    pub uploads: Vec<UploadRequirement>,
    pub scale: u32,
    pub creative_scale_factor: Option<String>,
    pub progress_price_microunits: Option<u64>,
}
fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, MagnificError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(MagnificError::Input("Missing source string field"))
}
fn integer(value: &Value, key: &str) -> Result<i64, MagnificError> {
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
    .ok_or(MagnificError::Input("Missing source integer field"))
}
fn boolean(value: &Value, key: &str) -> Result<bool, MagnificError> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or(MagnificError::Input("Missing source boolean field"))
}
fn validate_image(image: &ImageDimensions) -> Result<(), MagnificError> {
    if image.batch != 1
        || image.width < 160
        || image.height < 160
        || u64::from(image.width) > 3 * u64::from(image.height)
        || u64::from(image.height) > 3 * u64::from(image.width)
    {
        return Err(MagnificError::Input(
            "Source requires one image, minimum 160 and inclusive 1:3..3:1 aspect",
        ));
    }
    Ok(())
}
pub fn source_downscale(
    width: u32,
    height: u32,
    total_pixels: u64,
) -> Result<(u32, u32), MagnificError> {
    if width == 0 || height == 0 || total_pixels == 0 {
        return Err(MagnificError::Input("Invalid canonical resize dimensions"));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels <= total_pixels {
        return Ok((width, height));
    }
    let scale = (total_pixels as f64 / pixels as f64).sqrt();
    Ok((
        ((f64::from(width) * scale) as u32).max(2) & !1,
        ((f64::from(height) * scale) as u32).max(2) & !1,
    ))
}
pub fn upscale_progress_price_microunits(
    width: u32,
    height: u32,
    scale: u32,
) -> Result<u64, MagnificError> {
    if scale == 0 {
        return Err(MagnificError::Input("Invalid upscale factor"));
    }
    let mut pixels = u64::from(width) * u64::from(height);
    let mut tier_units = 0u64;
    for _ in 0..scale.ilog2() {
        tier_units += if pixels <= 1_300_000 {
            1
        } else if pixels <= 3_000_000 {
            2
        } else if pixels <= 6_400_000 {
            3
        } else {
            12
        };
        pixels = pixels.checked_mul(4).ok_or(ConformanceError::BodyLimit)?;
    }
    // Each EUR tier is an integer multiple of .143; the four source steps cannot
    // reach a half-cent tie after the 1.19 conversion. Preserve source cents.
    let cents = (tier_units * 17017 + 500) / 1000;
    Ok(cents * 10_000)
}
pub fn preparation_plan(input: &Input) -> Result<PreparationPlan, MagnificError> {
    if !NODES.iter().any(|node| node.node_identifier == input.node) {
        return Err(MagnificError::Input("Unknown Magnific node"));
    }
    validate_image(&input.image)?;
    let reference = match input.node.as_str() {
        "MagnificImageStyleTransferNode" => Some(
            input
                .reference_image
                .as_ref()
                .ok_or(MagnificError::Input("Missing reference image"))?,
        ),
        "MagnificImageRelightNode" => input.reference_image.as_ref(),
        _ => None,
    };
    if let Some(reference) = reference {
        validate_image(reference)?;
    }
    let mut scale = 1u32;
    let mut creative_scale_factor = None;
    let mut resize = None;
    let mut width = input.image.width;
    let mut height = input.image.height;
    let creative = input.node == "MagnificImageUpscalerCreativeNode";
    let precise = input.node == "MagnificImageUpscalerPreciseV2Node";
    if creative || precise {
        let source_scale = string(&input.settings, "scale_factor")?;
        if creative {
            creative_scale_factor = Some(source_scale.to_owned());
        }
        let source_scale = if creative {
            source_scale.trim_end_matches('x')
        } else {
            source_scale.trim_matches('x')
        };
        scale = python_strip(source_scale)
            .parse()
            .map_err(|_| MagnificError::Input("Invalid upscale factor"))?;
        if scale == 0 {
            return Err(MagnificError::Input("Invalid upscale factor"));
        }
        let pixels = u64::from(width) * u64::from(height);
        let axis = u64::from(width.max(height));
        let output_pixels = pixels
            .checked_mul(u64::from(scale))
            .and_then(|pixels| pixels.checked_mul(u64::from(scale)))
            .ok_or(ConformanceError::BodyLimit)?;
        let output_axis = axis * u64::from(scale);
        let exceeds = if creative {
            output_pixels > 25_300_000
        } else {
            output_axis > 10060
        };
        if exceeds {
            if !boolean(&input.settings, "auto_downscale")? {
                return Err(MagnificError::Input(
                    "Requested source upscale exceeds its output bound",
                ));
            }
            let requested_scale = scale;
            scale = 2;
            resize = Some(if creative {
                25_300_000 / 4
            } else {
                let ratio = 5030.0 / axis as f64;
                (pixels as f64 * ratio * ratio) as u64
            });
            for candidate in [16u32, 8, 4, 2] {
                if candidate > requested_scale {
                    continue;
                }
                let candidate_pixels = pixels
                    .checked_mul(u64::from(candidate * candidate))
                    .ok_or(ConformanceError::BodyLimit)?;
                let candidate_axis = axis * u64::from(candidate);
                let fits = if creative {
                    candidate_pixels <= 25_300_000
                } else {
                    candidate_axis <= 10060
                };
                if fits {
                    scale = candidate;
                    resize = None;
                    break;
                }
                let ratio = if creative {
                    (candidate_pixels as f64 / 25_300_000.0).sqrt()
                } else {
                    candidate_axis as f64 / 10060.0
                };
                if ratio <= 2.0 {
                    scale = candidate;
                    resize = Some(if creative {
                        25_300_000 / u64::from(candidate * candidate)
                    } else {
                        let ratio = f64::from(10060 / candidate) / axis as f64;
                        (pixels as f64 * ratio * ratio) as u64
                    });
                    break;
                }
            }
            if creative {
                creative_scale_factor = Some(format!("{scale}x"));
            }
            if let Some(total_pixels) = resize {
                (width, height) = source_downscale(width, height, total_pixels)?;
            }
        }
    }
    let mut uploads = vec![UploadRequirement {
        role: "image".into(),
        source: input.image.clone(),
        target_width: width,
        target_height: height,
        explicit_resize_pixels: resize,
        encoding_pixels: if creative || precise {
            None
        } else if input.node == "MagnificImageSkinEnhancerNode" {
            Some(4096 * 4096)
        } else {
            Some(2048 * 2048)
        },
        encoded_mime_type: "image/png",
        declared_content_type: None,
    }];
    if let Some(reference) = reference {
        uploads.push(UploadRequirement {
            role: "reference_image".into(),
            source: reference.clone(),
            target_width: reference.width,
            target_height: reference.height,
            explicit_resize_pixels: None,
            encoding_pixels: Some(2048 * 2048),
            encoded_mime_type: "image/png",
            declared_content_type: None,
        });
    }
    let progress_price_microunits = if creative || precise {
        Some(upscale_progress_price_microunits(width, height, scale)?)
    } else {
        None
    };
    // Source parses portrait controls before uploads, while relight/skin controls
    // are constructed after uploads. Keep that source ordering visible to owners.
    if input.node == "MagnificImageStyleTransferNode" {
        string(
            input
                .settings
                .get("portrait_mode")
                .ok_or(MagnificError::Input("Missing portrait mode"))?,
            "portrait_mode",
        )?;
    }
    Ok(PreparationPlan {
        uploads,
        scale,
        creative_scale_factor,
        progress_price_microunits,
    })
}
#[derive(Clone, Debug)]
pub struct PreparedMedia {
    pub requirement: UploadRequirement,
    pub file_name: String,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct CompletedUploadRequest {
    pub handle: StreamHandle,
    pub response_receipt: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct UploadedMedia {
    pub role: String,
    pub url: String,
    pub completed_requests: Vec<CompletedUploadRequest>,
}
pub trait MagnificUploadOwner {
    // Source None is a real omitted storage field and omitted signed-PUT header.
    // The legacy String DTO cannot represent it; an unbound owner must fail closed.
    fn upload(
        &mut self,
        _media: &PreparedMedia,
        _transport: &mut dyn MagnificTransport,
    ) -> Result<UploadedMedia, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
fn copy_integers(body: &mut Value, settings: &Value, names: &[&str]) -> Result<(), MagnificError> {
    for name in names {
        body[*name] = json!(integer(settings, name)?);
    }
    Ok(())
}
fn copy_strings(body: &mut Value, settings: &Value, names: &[&str]) -> Result<(), MagnificError> {
    for name in names {
        body[*name] = json!(string(settings, name)?);
    }
    Ok(())
}
fn copy_booleans(body: &mut Value, settings: &Value, names: &[&str]) -> Result<(), MagnificError> {
    for name in names {
        body[*name] = json!(boolean(settings, name)?);
    }
    Ok(())
}
pub fn request(
    input: &Input,
    plan: &PreparationPlan,
    uploaded: &[UploadedMedia],
) -> Result<(String, Vec<u8>), MagnificError> {
    if plan.uploads.len() != uploaded.len() {
        return Err(MagnificError::Input("Missing checked upload results"));
    }
    for (expected, actual) in plan.uploads.iter().zip(uploaded) {
        if expected.role != actual.role || endpoint(&actual.url)? {
            return Err(MagnificError::Response("Invalid canonical upload result"));
        }
    }
    let image = uploaded
        .first()
        .ok_or(MagnificError::Input("Missing uploaded image"))?;
    let mut body = json!({"image":image.url});
    let settings = &input.settings;
    let path = match input.node.as_str() {
        "MagnificImageUpscalerCreativeNode" => {
            body["scale_factor"] = json!(
                plan.creative_scale_factor
                    .as_ref()
                    .ok_or(MagnificError::Input("Missing source creative scale"))?
            );
            copy_strings(&mut body, settings, &["optimized_for", "engine"])?;
            copy_integers(
                &mut body,
                settings,
                &["creativity", "hdr", "resemblance", "fractality"],
            )?;
            "/proxy/freepik/v1/ai/image-upscaler".into()
        }
        "MagnificImageUpscalerPreciseV2Node" => {
            body["scale_factor"] = json!(plan.scale);
            copy_strings(&mut body, settings, &["flavor"])?;
            copy_integers(
                &mut body,
                settings,
                &["sharpen", "smart_grain", "ultra_detail"],
            )?;
            "/proxy/freepik/v1/ai/image-upscaler-precision-v2".into()
        }
        "MagnificImageStyleTransferNode" => {
            body["reference_image"] = json!(
                uploaded
                    .get(1)
                    .ok_or(MagnificError::Input("Missing reference upload"))?
                    .url
            );
            let portrait = settings
                .get("portrait_mode")
                .ok_or(MagnificError::Input("Missing portrait mode"))?;
            let enabled = string(portrait, "portrait_mode")? == "enabled";
            body["is_portrait"] = json!(enabled);
            if enabled {
                match portrait.get("portrait_style") {
                    Some(Value::Null) => {}
                    Some(Value::String(style)) => body["portrait_style"] = json!(style),
                    None => body["portrait_style"] = json!("standard"),
                    _ => return Err(MagnificError::Input("Invalid portrait style")),
                };
                match portrait.get("portrait_beautifier") {
                    Some(Value::String(value)) if value != "none" => {
                        body["portrait_beautifier"] = json!(value)
                    }
                    None | Some(Value::Null) | Some(Value::String(_)) => {}
                    _ => return Err(MagnificError::Input("Invalid portrait beautifier")),
                }
            }
            copy_strings(&mut body, settings, &["flavor", "engine"])?;
            copy_integers(
                &mut body,
                settings,
                &["style_strength", "structure_strength"],
            )?;
            copy_booleans(&mut body, settings, &["fixed_generation"])?;
            "/proxy/freepik/v1/ai/image-style-transfer".into()
        }
        "MagnificImageRelightNode" => {
            if let Some(reference) = uploaded.get(1) {
                body["transfer_light_from_reference_image"] = json!(reference.url);
            }
            copy_strings(&mut body, settings, &["style"])?;
            copy_integers(&mut body, settings, &["light_transfer_strength"])?;
            copy_booleans(
                &mut body,
                settings,
                &[
                    "interpolate_from_original",
                    "change_background",
                    "preserve_details",
                ],
            )?;
            let advanced = settings
                .get("advanced_settings")
                .ok_or(MagnificError::Input("Missing advanced settings"))?;
            if string(advanced, "advanced_settings")? == "enabled" {
                let mut fields = json!({});
                copy_integers(
                    &mut fields,
                    advanced,
                    &["whites", "blacks", "brightness", "contrast", "saturation"],
                )?;
                copy_strings(
                    &mut fields,
                    advanced,
                    &["engine", "transfer_light_a", "transfer_light_b"],
                )?;
                copy_booleans(&mut fields, advanced, &["fixed_generation"])?;
                body["advanced_settings"] = fields;
            }
            "/proxy/freepik/v1/ai/image-relight".into()
        }
        "MagnificImageSkinEnhancerNode" => {
            copy_integers(&mut body, settings, &["sharpen", "smart_grain"])?;
            let mode = settings
                .get("mode")
                .ok_or(MagnificError::Input("Missing skin mode"))?;
            let mode = match string(mode, "mode")? {
                "creative" => "creative",
                "faithful" => {
                    copy_integers(&mut body, mode, &["skin_detail"])?;
                    "faithful"
                }
                _ => {
                    copy_strings(&mut body, mode, &["optimized_for"])?;
                    "flexible"
                }
            };
            format!("/proxy/freepik/v1/ai/skin-enhancer/{mode}")
        }
        _ => return Err(MagnificError::Input("Unknown Magnific node")),
    };
    if input.node != "MagnificImageUpscalerPreciseV2Node"
        && input.node != "MagnificImageSkinEnhancerNode"
    {
        let prompt = string(settings, "prompt")?;
        if !prompt.is_empty() {
            body["prompt"] = json!(prompt);
        }
    }
    let bytes = serde_json::to_vec(&body)?;
    if bytes.len() > MAXIMUM_REQUEST_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok((path, bytes))
}
pub trait MagnificTransport: ProviderTransport {
    fn begin_response_deadline(
        &mut self,
        _handle: StreamHandle,
        _timeout_milliseconds: Option<u64>,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn report_poll_status(
        &mut self,
        _handle: StreamHandle,
        _status: &str,
        _queued: bool,
        _price_microunits: Option<u64>,
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
    fn retain(&mut self, handle: StreamHandle) -> Result<(), MagnificError> {
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
fn endpoint(value: &str) -> Result<bool, MagnificError> {
    if value.is_empty()
        || value.len() > 8192
        || value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(MagnificError::Response(
            "Invalid Magnific response endpoint",
        ));
    }
    if value.starts_with('/') && !value.starts_with("//") {
        return Ok(true);
    }
    let absolute = value
        .strip_prefix("https://")
        .ok_or(MagnificError::Response(
            "Invalid Magnific response endpoint",
        ))?;
    let authority = absolute
        .split('/')
        .next()
        .ok_or(MagnificError::Response("Missing endpoint authority"))?;
    if authority.is_empty() || authority.contains('@') || authority.contains('?') {
        return Err(MagnificError::Response(
            "Invalid Magnific endpoint authority",
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
) -> Result<RequestHead, MagnificError> {
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
            return Err(MagnificError::Input("Missing Comfy API credentials"));
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
const POLL_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 10,
    rate_limit: 16,
    backoff: 1.4,
    download: false,
    timeout_milliseconds: Some(120_000),
};
const DOWNLOAD_RETRY: RetryPolicy = RetryPolicy {
    ordinary_limit: 5,
    rate_limit: 0,
    backoff: 2.0,
    download: true,
    timeout_milliseconds: None,
};
fn exchange_with_policy(
    head: &RequestHead,
    body: &[u8],
    maximum_bytes: usize,
    policy: RetryPolicy,
    session: &mut Session,
    transport: &mut dyn MagnificTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MagnificError> {
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
    transport: &mut dyn MagnificTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MagnificError> {
    let mut ordinary_retries = 0u32;
    let mut rate_limit_retries = 0u32;
    loop {
        match exchange_on_handle(handle, body, maximum_bytes, policy, transport) {
            Err(MagnificError::Http {
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
                    return Err(MagnificError::Http {
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
    transport: &mut dyn MagnificTransport,
) -> Result<(Vec<u8>, Vec<u8>, StreamHandle), MagnificError> {
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
                    return Err(MagnificError::Http {
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
pub struct TaskResponse {
    pub task_id: String,
    pub status: String,
    pub generated: Option<Vec<String>>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Queued,
    Processing,
    Completed,
}
pub fn task_response(bytes: &[u8]) -> Result<TaskResponse, MagnificError> {
    if bytes.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    let value: Value = serde_json::from_slice(bytes)?;
    let value = value
        .get("data")
        .filter(|value| value.is_object())
        .unwrap_or(&value);
    let task_id = value
        .get("task_id")
        .and_then(Value::as_str)
        .ok_or(MagnificError::Response("Missing response task ID"))?
        .to_owned();
    let status = value
        .get("status")
        .or_else(|| value.get("task_status"))
        .and_then(Value::as_str)
        .ok_or(MagnificError::Response("Missing response task status"))?
        .to_owned();
    let generated = match value.get("generated") {
        None | Some(Value::Null) => None,
        Some(value) => Some(serde_json::from_value::<Vec<String>>(value.clone())?),
    };
    Ok(TaskResponse {
        task_id,
        status,
        generated,
    })
}
pub fn task_phase(status: &str) -> Result<Phase, MagnificError> {
    match python_strip(status).to_lowercase().as_str() {
        "succeeded" | "succeed" | "success" | "completed" | "finished" | "done" | "complete" => {
            Ok(Phase::Completed)
        }
        "cancelled" | "canceled" | "canceling" | "fail" | "failed" | "error" => {
            Err(MagnificError::Provider(status.to_owned()))
        }
        "created" | "queued" | "queueing" | "submitted" | "initializing" | "wait" | "in_queue" => {
            Ok(Phase::Queued)
        }
        _ => Ok(Phase::Processing),
    }
}
#[derive(Clone, Debug)]
pub struct DownloadedResult {
    pub node: String,
    pub task_id: String,
    pub source_url: String,
    pub bytes: Vec<u8>,
    pub uploaded: Vec<UploadedMedia>,
    pub cost_receipt: Vec<u8>,
    pub submission_receipt: Vec<u8>,
    pub status_receipt: Vec<u8>,
    pub download_receipt: Vec<u8>,
    pub handle: StreamHandle,
    pub progress_price_microunits: Option<u64>,
}
pub fn invoke(
    input: &Input,
    media: &[PreparedMedia],
    credentials: CredentialAvailability,
    maximum_cost_microunits: u64,
    transport: &mut dyn MagnificTransport,
    uploads: &mut dyn MagnificUploadOwner,
) -> Result<DownloadedResult, MagnificError> {
    let plan = preparation_plan(input)?;
    if plan.uploads.len() != media.len() {
        return Err(MagnificError::Input("Missing canonical prepared images"));
    }
    let mut size = 0usize;
    for (requirement, media) in plan.uploads.iter().zip(media) {
        if *requirement != media.requirement
            || media.bytes.is_empty()
            || media.file_name.is_empty()
            || media.file_name.len() > 1024
            || !media.file_name.ends_with(".png")
            || media
                .file_name
                .chars()
                .any(|character| character.is_control() || matches!(character, '/' | '\\'))
        {
            return Err(MagnificError::Input(
                "Invalid canonical source PNG preparation",
            ));
        }
        size = size
            .checked_add(media.bytes.len())
            .ok_or(ConformanceError::BodyLimit)?;
    }
    if size > MAXIMUM_MEDIA_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    if maximum_cost_microunits == 0 {
        return Err(ConformanceError::InvalidCost.into());
    }
    transport.check_cancelled(None)?;
    let mut session = Session::default();
    let mut uploaded = Vec::with_capacity(media.len());
    for media in media {
        transport.check_cancelled(None)?;
        let result = uploads.upload(media, transport)?;
        if result.role != media.requirement.role
            || endpoint(&result.url)?
            || !(2..=32).contains(&result.completed_requests.len())
        {
            return Err(MagnificError::Response("Invalid retained upload lineage"));
        }
        for completed in &result.completed_requests {
            if completed.response_receipt.is_empty()
                || completed.response_receipt.len() > MAXIMUM_RECEIPT_BYTES
            {
                return Err(ConformanceError::InvalidResponse.into());
            }
            session.retain(completed.handle)?;
        }
        uploaded.push(result);
    }
    let (path, body) = request(input, &plan, &uploaded)?;
    transport.check_cancelled(None)?;
    let head = request_head(&path, HttpMethod::Post, credentials, false)?;
    let handle = transport.start_request(&head)?;
    session.retain(handle)?;
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "magnific.generation.maximum".into(),
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
    let (bytes, submission_receipt, _handle) = exchange_existing(
        &head,
        &body,
        MAXIMUM_RESPONSE_BYTES,
        GENERATION_RETRY,
        handle,
        &mut session,
        transport,
    )?;
    let initial = task_response(&bytes)?;
    if initial.task_id.is_empty()
        || initial.task_id.len() > 1024
        || !initial
            .task_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(MagnificError::Response("Unsafe task route selector"));
    }
    let poll_path = if input.node == "MagnificImageSkinEnhancerNode" {
        format!("/proxy/freepik/v1/ai/skin-enhancer/{}", initial.task_id)
    } else {
        format!("{path}/{}", initial.task_id)
    };
    let poll_head = request_head(&poll_path, HttpMethod::Get, credentials, false)?;
    let mut consumed = 0;
    for _ in 0..MAXIMUM_TOTAL_POLLS {
        let (bytes, status_receipt, poll_handle) = exchange_with_policy(
            &poll_head,
            &[],
            MAXIMUM_RESPONSE_BYTES,
            POLL_RETRY,
            &mut session,
            transport,
        )?;
        let final_response = task_response(&bytes)?;
        if final_response.task_id != initial.task_id {
            return Err(MagnificError::Response("Response task identity changed"));
        }
        let phase = task_phase(&final_response.status)?;
        let normalized_status = python_strip(&final_response.status).to_lowercase();
        transport.report_poll_status(
            poll_handle,
            if normalized_status.is_empty() {
                "Processing"
            } else {
                &normalized_status
            },
            phase == Phase::Queued,
            plan.progress_price_microunits,
        )?;
        if phase == Phase::Completed {
            let source_url = final_response
                .generated
                .and_then(|values| values.into_iter().next())
                .ok_or(MagnificError::Response(
                    "Completed task has no generated image",
                ))?;
            if endpoint(&source_url)? {
                return Err(MagnificError::Response(
                    "Expected absolute generated image URL",
                ));
            }
            let download_head = request_head(&source_url, HttpMethod::Get, credentials, true)?;
            let (bytes, download_receipt, handle) = exchange_with_policy(
                &download_head,
                &[],
                MAXIMUM_MEDIA_BYTES,
                DOWNLOAD_RETRY,
                &mut session,
                transport,
            )?;
            if bytes.is_empty() {
                return Err(MagnificError::Response("Generated image bytes are empty"));
            }
            return Ok(DownloadedResult {
                node: input.node.clone(),
                task_id: initial.task_id,
                source_url,
                bytes,
                uploaded,
                cost_receipt: approval.receipt,
                submission_receipt,
                status_receipt,
                download_receipt,
                handle,
                progress_price_microunits: plan.progress_price_microunits,
            });
        }
        transport.check_cancelled(Some(poll_handle))?;
        transport.delay(poll_handle, 10_000)?;
        if phase != Phase::Queued {
            consumed += 1;
            if consumed >= MAXIMUM_POLLS {
                break;
            }
        }
    }
    Err(ConformanceError::PollLimit.into())
}
pub trait MagnificImageMaterializer {
    // Receipts are checked against retained caller context before canonical RGBA
    // decoding/allocation; these bytes never stand in for a NativePayload or grant.
    fn materialize(
        &mut self,
        result: &DownloadedResult,
        expected: &ExpectedOutput,
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub fn materialize_and_publish(
    result: &DownloadedResult,
    transport: &mut dyn MagnificTransport,
    owner: &mut dyn MagnificImageMaterializer,
    sink: &mut dyn MaterializationSink,
) -> Result<(), MagnificError> {
    transport.check_cancelled(None)?;
    let registry = TypeRegistry::built_in()
        .map_err(|_| MagnificError::Response("Missing canonical IMAGE registry"))?;
    let type_id = registry
        .resolve("IMAGE")
        .map_err(|_| MagnificError::Response("Missing canonical IMAGE type"))?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| MagnificError::Response("Missing canonical IMAGE family"))?;
    let expected = ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    };
    let outputs = owner.materialize(result, &expected)?;
    let batch = validate_outputs(&[expected], outputs, MAXIMUM_MEDIA_BYTES)
        .map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn source_oracle() -> Result<Value, ConformanceError> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/magnific/source-contract.json"
    )))
    .map_err(|_| ConformanceError::InvalidRequest)
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture: Value =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture["namespace"] != NAMESPACE
        || fixture["source_sha256"] != SOURCE_SHA256
        || fixture["aliases"] != json!(ALIASES)
        || fixture["nodes"] != json!(NODES)
        || fixture["routes"] != json!(ROUTES)
    {
        return Err(ConformanceError::InvalidRequest);
    }
    let source = source_oracle()?;
    if source["sources"]["comfy_api_nodes/nodes_magnific.py"] != SOURCE_SHA256 {
        return Err(ConformanceError::InvalidRequest);
    }
    let mut covered = BTreeSet::new();
    for case in source["cases"]
        .as_array()
        .ok_or(ConformanceError::InvalidRequest)?
    {
        let input: Input = serde_json::from_value(case["input"].clone())
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let plan = preparation_plan(&input).map_err(|_| ConformanceError::InvalidRequest)?;
        let uploads = case["uploads"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?;
        if uploads.len() != plan.uploads.len() {
            return Err(ConformanceError::InvalidRequest);
        }
        let mut uploaded = vec![];
        for (source_upload, requirement) in uploads.iter().zip(&plan.uploads) {
            if source_upload["role"] != requirement.role
                || source_upload["width"] != requirement.target_width
                || source_upload["height"] != requirement.target_height
                || source_upload["total_pixels"] != json!(requirement.encoding_pixels)
                || source_upload["declared_content_type"]
                    != json!(requirement.declared_content_type)
                || source_upload["encoded_mime_type"] != requirement.encoded_mime_type
            {
                return Err(ConformanceError::InvalidRequest);
            }
            uploaded.push(UploadedMedia {
                role: requirement.role.clone(),
                url: format!("https://storage.fixture.invalid/{}.png", requirement.role),
                completed_requests: vec![],
            });
        }
        let (path, bytes) =
            request(&input, &plan, &uploaded).map_err(|_| ConformanceError::InvalidRequest)?;
        let body: Value =
            serde_json::from_slice(&bytes).map_err(|_| ConformanceError::InvalidRequest)?;
        if case["request"]["path"] != path
            || case["request"]["body"] != body
            || case["poll"]["poll_interval"] != 10.0
            || case["poll"]["progress_price_usd"]
                != json!(
                    plan.progress_price_microunits
                        .map(|price| price as f64 / 1_000_000.0)
                )
        {
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
        if preparation_plan(&input).is_ok() {
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
        delays: Vec<u64>,
        heads: Vec<RequestHead>,
        costs: usize,
        deadlines: Vec<Option<u64>>,
        statuses: Vec<(String, bool, Option<u64>)>,
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
                delays: vec![],
                heads: vec![],
                costs: 0,
                deadlines: vec![],
                statuses: vec![],
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
                    && !matches!((serde_json::from_slice::<Value>(&response.body), serde_json::from_slice::<Value>(bytes)), (Ok(expected),Ok(actual)) if expected == actual)
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
    impl MagnificTransport for TestTransport {
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

        fn report_poll_status(
            &mut self,
            handle: StreamHandle,
            status: &str,
            queued: bool,
            price: Option<u64>,
        ) -> Result<(), ConformanceError> {
            if handle.invocation != 42 {
                return Err(ConformanceError::InvalidHandle);
            }
            self.statuses.push((status.into(), queued, price));
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
    }
    fn credentials() -> CredentialAvailability {
        CredentialAvailability {
            auth_token: true,
            api_key: true,
        }
    }
    fn response(path: &str, method: HttpMethod, body: Vec<u8>, bytes: Vec<u8>) -> Response {
        Response {
            path: path.into(),
            method,
            body,
            status: 200,
            bytes,
            headers: vec![],
            receipt: b"retained request receipt".to_vec(),
        }
    }
    fn baseline(index: usize) -> Result<(Input, Value), Box<dyn Error>> {
        let oracle = source_oracle()?;
        let case = oracle["cases"]
            .get(index)
            .ok_or("missing source case")?
            .clone();
        Ok((serde_json::from_value(case["input"].clone())?, case))
    }
    fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        if !value.len().is_multiple_of(2) {
            return Err("odd source hex".into());
        }
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| Ok(u8::from_str_radix(std::str::from_utf8(pair)?, 16)?))
            .collect()
    }
    fn prepared(input: &Input) -> Result<Vec<PreparedMedia>, Box<dyn Error>> {
        let oracle = source_oracle()?;
        preparation_plan(input)?
            .uploads
            .into_iter()
            .map(|requirement| {
                let index = if requirement.target_width == 320 {
                    1
                } else {
                    0
                };
                let hex = oracle["real_png_upload"]["puts"][index]["bytes_hex"]
                    .as_str()
                    .ok_or("missing actual source PNG")?;
                Ok(PreparedMedia {
                    file_name: format!("{}.png", requirement.role),
                    requirement,
                    bytes: decode_hex(hex)?,
                })
            })
            .collect()
    }
    struct UploadOwner {
        calls: Vec<UploadRequirement>,
        next_slot: u32,
        replay: bool,
        cross_invocation: bool,
        empty_receipt: bool,
    }
    impl Default for UploadOwner {
        fn default() -> Self {
            Self {
                calls: vec![],
                next_slot: 1,
                replay: false,
                cross_invocation: false,
                empty_receipt: false,
            }
        }
    }
    impl MagnificUploadOwner for UploadOwner {
        fn upload(
            &mut self,
            media: &PreparedMedia,
            transport: &mut dyn MagnificTransport,
        ) -> Result<UploadedMedia, ConformanceError> {
            transport.check_cancelled(None)?;
            if media.requirement.declared_content_type.is_some()
                || media.requirement.encoded_mime_type != "image/png"
                || !media.bytes.starts_with(b"\x89PNG\r\n\x1a\n")
            {
                return Err(ConformanceError::InvalidRequest);
            }
            self.calls.push(media.requirement.clone());
            let mut completed_requests = vec![];
            for _ in 0..2 {
                completed_requests.push(CompletedUploadRequest {
                    handle: StreamHandle {
                        invocation: if self.cross_invocation && self.next_slot == 2 {
                            43
                        } else {
                            42
                        },
                        slot: if self.replay { 1 } else { self.next_slot },
                        generation: 1,
                    },
                    response_receipt: if self.empty_receipt {
                        vec![]
                    } else {
                        b"trusted upload owner completion".to_vec()
                    },
                });
                self.next_slot += 1;
            }
            Ok(UploadedMedia {
                role: media.requirement.role.clone(),
                url: format!(
                    "https://storage.fixture.invalid/{}.png",
                    media.requirement.role
                ),
                completed_requests,
            })
        }
    }
    fn script(case: &Value, download: &[u8]) -> Result<Vec<Response>, Box<dyn Error>> {
        let path = case["request"]["path"]
            .as_str()
            .ok_or("missing source request path")?;
        let poll = case["poll"]["path"]
            .as_str()
            .ok_or("missing source poll path")?;
        Ok(vec![
            response(
                path,
                HttpMethod::Post,
                serde_json::to_vec(&case["request"]["body"])?,
                serde_json::to_vec(&json!({"data":{"task_id":"task-42","task_status":"queued"}}))?,
            ),
            response(
                poll,
                HttpMethod::Get,
                vec![],
                serde_json::to_vec(
                    &json!({"data":{"task_id":"task-42","status":"done","generated":["https://output.fixture.invalid/first.png","https://output.fixture.invalid/ignored.png"]}}),
                )?,
            ),
            response(
                "https://output.fixture.invalid/first.png",
                HttpMethod::Get,
                vec![],
                download.to_vec(),
            ),
        ])
    }
    fn transport_for(script: Vec<Response>, media: &[PreparedMedia]) -> TestTransport {
        let mut transport = TestTransport::new(script);
        transport.next_slot = 1 + media.len() as u32 * 2;
        transport
    }
    #[test]
    fn pinned_execute_oracle_covers_all_controls_resize_decisions_and_preflight()
    -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/magnific/conformance.fixture"
        )))?;
        let oracle = source_oracle()?;
        assert_eq!(oracle["cases"].as_array().ok_or("cases")?.len(), 227);
        assert_eq!(
            oracle["preflight_errors"].as_array().ok_or("errors")?.len(),
            24
        );
        for case in oracle["price_cases"].as_array().ok_or("prices")? {
            let price = upscale_progress_price_microunits(
                serde_json::from_value(case["width"].clone())?,
                serde_json::from_value(case["height"].clone())?,
                serde_json::from_value(case["scale"].clone())?,
            )?;
            assert_eq!(json!(price as f64 / 1_000_000.), case["usd"]);
        }
        assert_eq!(
            ROUTES
                .iter()
                .filter(|route| route.disposition == "synthetic_prefix_tombstone")
                .count(),
            5
        );
        assert_eq!(ALIASES, &["freepik"]);
        Ok(())
    }
    #[test]
    fn actual_source_none_content_type_png_and_all_five_generation_poll_download_paths()
    -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        for body in oracle["real_png_upload"]["storage_bodies"]
            .as_array()
            .ok_or("storage bodies")?
        {
            assert!(body.get("content_type").is_none());
        }
        for put in oracle["real_png_upload"]["puts"].as_array().ok_or("puts")? {
            assert!(put["declared_content_type"].is_null());
        }
        for index in 0..5 {
            let (input, case) = baseline(index)?;
            let media = prepared(&input)?;
            let download = &media.first().ok_or("missing prepared PNG")?.bytes;
            let mut transport = transport_for(script(&case, download)?, &media);
            let mut owner = UploadOwner::default();
            let result = invoke(
                &input,
                &media,
                credentials(),
                900_000,
                &mut transport,
                &mut owner,
            )?;
            assert_eq!(result.bytes, *download);
            assert_eq!(result.node, input.node);
            assert_eq!(result.task_id, "task-42");
            assert_eq!(result.uploaded.len(), media.len());
            assert_eq!(owner.calls, preparation_plan(&input)?.uploads);
            assert_eq!(transport.costs, 1);
            assert_eq!(
                transport.deadlines,
                vec![Some(3_600_000), Some(120_000), None]
            );
            assert_eq!(
                transport.statuses,
                vec![("done".into(), false, result.progress_price_microunits)]
            );
            let last = transport.heads.last().ok_or("missing download head")?;
            assert!(last.secret_id.is_none());
            assert_eq!(
                last.headers,
                vec![Header {
                    name: "Accept".into(),
                    value: "*/*".into()
                }]
            );
            assert!(!result.download_receipt.is_empty());
            assert!(!result.status_receipt.is_empty());
            assert!(!result.submission_receipt.is_empty());
            transport.finish()?;
        }
        Ok(())
    }
    #[test]
    fn actual_pydantic_alias_precedence_and_source_status_sets() -> Result<(), Box<dyn Error>> {
        let oracle = source_oracle()?;
        for case in oracle["response_cases"]
            .as_array()
            .ok_or("response cases")?
        {
            let parsed = task_response(&serde_json::to_vec(&case["input"])?);
            if let Some(model) = case.get("model") {
                let parsed = parsed?;
                assert_eq!(parsed.task_id, model["task_id"]);
                assert_eq!(parsed.status, model["status"]);
                assert_eq!(
                    json!(parsed.generated),
                    model.get("generated").cloned().unwrap_or(Value::Null)
                );
            } else {
                assert!(parsed.is_err());
            }
        }
        for (name, phase) in [
            ("COMPLETED_STATUSES", Phase::Completed),
            ("QUEUED_STATUSES", Phase::Queued),
        ] {
            for status in oracle["client_constants"][name]
                .as_array()
                .ok_or("source statuses")?
            {
                let status = status.as_str().ok_or("status")?;
                assert_eq!(
                    task_phase(&format!("\u{1c} {} \n", status.to_uppercase()))?,
                    phase
                );
            }
        }
        for status in oracle["client_constants"]["FAILED_STATUSES"]
            .as_array()
            .ok_or("failures")?
        {
            assert!(matches!(
                task_phase(status.as_str().ok_or("failure")?),
                Err(MagnificError::Provider(_))
            ));
        }
        assert_eq!(task_phase("unknown")?, Phase::Processing);
        Ok(())
    }
    #[test]
    fn source_queued_polls_preserve_ten_second_delay_and_progress_price()
    -> Result<(), Box<dyn Error>> {
        let (input, case) = baseline(4)?;
        let media = prepared(&input)?;
        let mut responses = script(&case, &media[0].bytes)?;
        responses.insert(
            1,
            response(
                case["poll"]["path"].as_str().ok_or("path")?,
                HttpMethod::Get,
                vec![],
                serde_json::to_vec(&json!({"task_id":"task-42","task_status":" \u{1c}QUEUED\n"}))?,
            ),
        );
        let mut transport = transport_for(responses, &media);
        let mut owner = UploadOwner::default();
        let result = invoke(
            &input,
            &media,
            credentials(),
            900_000,
            &mut transport,
            &mut owner,
        )?;
        assert_eq!(transport.delays, vec![10_000]);
        assert_eq!(
            transport.statuses,
            vec![
                ("queued".into(), true, result.progress_price_microunits),
                ("done".into(), false, result.progress_price_microunits)
            ]
        );
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn source_retry_limits_deadlines_rate_backoff_and_stale_replay() -> Result<(), Box<dyn Error>> {
        for (path, method, policy, limit) in [
            ("/proxy/freepik/test", HttpMethod::Post, GENERATION_RETRY, 3),
            ("/proxy/freepik/test/x", HttpMethod::Get, POLL_RETRY, 10),
            (
                "https://image.fixture.invalid/result",
                HttpMethod::Get,
                DOWNLOAD_RETRY,
                5,
            ),
        ] {
            for status in [503, 429] {
                let allowed = if status == 429 && !policy.download {
                    16
                } else {
                    limit
                };
                let mut responses = vec![];
                for _ in 0..=allowed {
                    let mut reply = response(path, method, vec![], b"error".to_vec());
                    reply.status = status;
                    responses.push(reply);
                }
                let mut transport = TestTransport::new(responses);
                let head = request_head(path, method, credentials(), policy.download)?;
                let result = exchange_with_policy(
                    &head,
                    &[],
                    1024,
                    policy,
                    &mut Session::default(),
                    &mut transport,
                );
                assert!(
                    matches!(result,Err(MagnificError::Http{status:value,..}) if value==status)
                );
                assert_eq!(transport.delays.len(), allowed as usize);
                assert!(
                    transport
                        .deadlines
                        .iter()
                        .all(|value| *value == policy.timeout_milliseconds)
                );
                transport.finish()?;
            }
        }
        let mut first = response(
            "/proxy/freepik/test",
            HttpMethod::Post,
            vec![],
            b"busy".to_vec(),
        );
        first.status = 429;
        first.headers = vec![Header {
            name: "Retry-After".into(),
            value: "9999".into(),
        }];
        let mut transport = TestTransport::new(vec![
            first,
            response(
                "/proxy/freepik/test",
                HttpMethod::Post,
                vec![],
                b"ok".to_vec(),
            ),
        ]);
        let head = request_head(
            "/proxy/freepik/test",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        exchange_with_policy(
            &head,
            &[],
            1024,
            GENERATION_RETRY,
            &mut Session::default(),
            &mut transport,
        )?;
        assert_eq!(transport.delays, vec![150_000]);
        transport.finish()?;
        let mut reply = response(
            "/proxy/freepik/test",
            HttpMethod::Post,
            vec![],
            b"busy".to_vec(),
        );
        reply.status = 503;
        let mut transport = TestTransport::new(vec![reply]);
        transport.stale_retry = true;
        assert!(matches!(
            exchange_with_policy(
                &head,
                &[],
                1024,
                GENERATION_RETRY,
                &mut Session::default(),
                &mut transport
            ),
            Err(MagnificError::Transport(ConformanceError::InvalidHandle))
        ));
        Ok(())
    }
    #[test]
    fn unbound_upload_owner_and_invalid_lineage_fail_before_generation()
    -> Result<(), Box<dyn Error>> {
        struct Unbound;
        impl MagnificUploadOwner for Unbound {}
        let (input, _) = baseline(0)?;
        let media = prepared(&input)?;
        let mut transport = transport_for(vec![], &media);
        assert!(matches!(
            invoke(
                &input,
                &media,
                credentials(),
                1,
                &mut transport,
                &mut Unbound
            ),
            Err(MagnificError::Transport(ConformanceError::HostFailure))
        ));
        assert!(transport.heads.is_empty());
        for (replay, cross_invocation, empty_receipt) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let mut owner = UploadOwner {
                replay,
                cross_invocation,
                empty_receipt,
                ..UploadOwner::default()
            };
            let mut transport = transport_for(vec![], &media);
            assert!(invoke(&input, &media, credentials(), 1, &mut transport, &mut owner).is_err());
            assert!(transport.heads.is_empty());
        }
        let mut media = media;
        media[0].requirement.declared_content_type = Some("image/png");
        let mut owner = UploadOwner::default();
        let mut transport = TestTransport::new(vec![]);
        assert!(invoke(&input, &media, credentials(), 1, &mut transport, &mut owner).is_err());
        assert!(owner.calls.is_empty());
        Ok(())
    }
    #[test]
    fn cancellation_worker_failure_cost_rejection_and_response_residue()
    -> Result<(), Box<dyn Error>> {
        let (input, case) = baseline(0)?;
        let media = prepared(&input)?;
        for error in [ConformanceError::Cancelled, ConformanceError::WorkerLost] {
            let mut transport = transport_for(vec![], &media);
            transport.failure = Some(error.clone());
            let mut owner = UploadOwner::default();
            assert!(
                matches!(invoke(&input,&media,credentials(),1,&mut transport,&mut owner),Err(MagnificError::Transport(value)) if value==error)
            );
            assert!(owner.calls.is_empty());
            assert!(transport.heads.is_empty());
        }
        let mut transport = transport_for(script(&case, &media[0].bytes)?, &media);
        transport.reject_cost = true;
        let mut owner = UploadOwner::default();
        assert!(matches!(
            invoke(&input, &media, credentials(), 1, &mut transport, &mut owner),
            Err(MagnificError::Transport(ConformanceError::InvalidCost))
        ));
        assert_eq!(transport.heads.len(), 1);
        assert!(transport.deadlines.is_empty());
        for mode in 0..6 {
            let mut replies = script(&case, &media[0].bytes)?;
            match mode {
                0 => replies[0].bytes = b"malformed".to_vec(),
                1 => {
                    replies[0].bytes =
                        serde_json::to_vec(&json!({"task_id":"../escape","status":"queued"}))?
                }
                2 => {
                    replies[1].bytes = serde_json::to_vec(
                        &json!({"task_id":"other","status":"done","generated":["https://output.fixture.invalid/first.png"]}),
                    )?
                }
                3 => replies[1].receipt = vec![],
                4 => {
                    replies[1].headers = vec![Header {
                        name: "invalid\r\nname".into(),
                        value: "x".into(),
                    }]
                }
                _ => replies[1].bytes = vec![0; MAXIMUM_CHUNK_BYTES + 1],
            }
            let mut transport = transport_for(replies, &media);
            let mut owner = UploadOwner::default();
            assert!(invoke(&input, &media, credentials(), 1, &mut transport, &mut owner).is_err());
            assert!(
                transport
                    .heads
                    .iter()
                    .all(|head| !head.path.starts_with("https://output"))
            );
        }
        Ok(())
    }
    #[test]
    fn strict_test_transport_rejects_unequal_binary_and_malformed_json()
    -> Result<(), Box<dyn Error>> {
        let head = request_head(
            "/proxy/freepik/test",
            HttpMethod::Post,
            credentials(),
            false,
        )?;
        for (expected, actual, accepted) in [
            (b"{bad".as_slice(), b"{other".as_slice(), false),
            (b"\x89PNG\x00".as_slice(), b"\x89PNG\x01".as_slice(), false),
            (b"\x89PNG\x00".as_slice(), b"\x89PNG\x00".as_slice(), true),
            (
                br#"{"a":1,"b":2}"#.as_slice(),
                br#"{"b":2,"a":1}"#.as_slice(),
                true,
            ),
        ] {
            let mut transport = TestTransport::new(vec![response(
                &head.path,
                HttpMethod::Post,
                expected.to_vec(),
                vec![],
            )]);
            let handle = transport.start_request(&head)?;
            assert_eq!(
                transport
                    .write_request_chunk(handle, 0, actual, true)
                    .is_ok(),
                accepted
            );
        }
        Ok(())
    }
    #[test]
    fn native_materializer_gate_has_zero_publication() -> Result<(), Box<dyn Error>> {
        struct Missing;
        impl MagnificImageMaterializer for Missing {
            fn materialize(
                &mut self,
                _: &DownloadedResult,
                _: &ExpectedOutput,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                Err(ConformanceError::HostFailure)
            }
        }
        struct Sink {
            count: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(
                &mut self,
                _: crate::common::MaterializedBatch,
            ) -> Result<(), crate::common::MaterializationError> {
                self.count += 1;
                Ok(())
            }
        }
        let (input, case) = baseline(0)?;
        let media = prepared(&input)?;
        let mut transport = transport_for(script(&case, &media[0].bytes)?, &media);
        let mut owner = UploadOwner::default();
        let result = invoke(&input, &media, credentials(), 1, &mut transport, &mut owner)?;
        let mut sink = Sink { count: 0 };
        assert!(matches!(
            materialize_and_publish(&result, &mut transport, &mut Missing, &mut sink),
            Err(MagnificError::Transport(ConformanceError::HostFailure))
        ));
        assert_eq!(sink.count, 0);
        transport.cancelled = true;
        assert!(matches!(
            materialize_and_publish(&result, &mut transport, &mut Missing, &mut sink),
            Err(MagnificError::Transport(ConformanceError::Cancelled))
        ));
        assert_eq!(sink.count, 0);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn processing_poll_budget_excludes_source_queued_states() -> Result<(), Box<dyn Error>> {
        let (input, case) = baseline(0)?;
        let media = prepared(&input)?;
        let mut responses = script(&case, &media[0].bytes)?;
        responses.truncate(1);
        for index in 0..483 {
            responses.push(response(case["poll"]["path"].as_str().ok_or("poll path")?, HttpMethod::Get, vec![], serde_json::to_vec(&json!({"task_id":"task-42", "status":if index<3 {"in_queue"}else{"processing"}}))?));
        }
        let mut transport = transport_for(responses, &media);
        let mut owner = UploadOwner::default();
        assert!(matches!(
            invoke(&input, &media, credentials(), 1, &mut transport, &mut owner),
            Err(MagnificError::Transport(ConformanceError::PollLimit))
        ));
        assert_eq!(transport.statuses.len(), 483);
        assert_eq!(
            transport
                .statuses
                .iter()
                .filter(|(_, queued, _)| *queued)
                .count(),
            3
        );
        assert_eq!(transport.delays.len(), 483);
        transport.finish()?;
        Ok(())
    }
    #[test]
    fn source_control_parse_order_precedes_or_follows_uploads() -> Result<(), Box<dyn Error>> {
        for (index, field, expected_uploads) in [
            (0, "advanced_settings", 1),
            (1, "mode", 1),
            (2, "portrait_mode", 0),
        ] {
            let (mut input, _) = baseline(index)?;
            let media = prepared(&input)?;
            input
                .settings
                .as_object_mut()
                .ok_or("settings")?
                .remove(field);
            let mut transport = transport_for(vec![], &media);
            let mut owner = UploadOwner::default();
            assert!(invoke(&input, &media, credentials(), 1, &mut transport, &mut owner).is_err());
            assert_eq!(owner.calls.len(), expected_uploads);
            assert!(transport.heads.is_empty());
        }
        Ok(())
    }
}
