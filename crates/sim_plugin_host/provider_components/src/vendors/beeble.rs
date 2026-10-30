use std::{error::Error, fmt};

use crate::common::ConformanceError;
use serde::{Deserialize, Serialize};

pub const NAMESPACE: &str = "zed.sim.provider.beeble";
pub const SOURCE_SHA256: &str = "a7d98292ba01564a4a8519ba3c8caa30d9e0aaf42bf16ca55e29944222896812";
pub const GENERATIONS_PATH: &str = "/proxy/beeble/v1/switchx/generations";
pub const MAXIMUM_PIXELS: u64 = 2_770_000;
pub const MAXIMUM_FRAMES: u64 = 240;
pub const MAXIMUM_PROMPT_CHARACTERS: usize = 2000;
pub const MAXIMUM_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub enum BeebleError {
    Input(&'static str),
    MissingRender(String),
    ResponseLimit,
    Json(serde_json::Error),
}

impl fmt::Display for BeebleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) => formatter.write_str(message),
            Self::MissingRender(id) => write!(
                formatter,
                "Beeble job {id} completed without a 'render' output URL."
            ),
            Self::ResponseLimit => formatter.write_str("Beeble response exceeds its byte limit"),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}

impl Error for BeebleError {}

impl From<serde_json::Error> for BeebleError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationType {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaMode {
    Auto,
    Fill,
    Select,
    Custom,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Resolution {
    #[serde(rename = "1080p")]
    FullHd,
    #[serde(rename = "720p")]
    Hd,
}

impl Resolution {
    pub fn pixels(self) -> u16 {
        match self {
            Self::FullHd => 1080,
            Self::Hd => 720,
        }
    }

    pub fn price_badge_microunits(self) -> u64 {
        match self {
            Self::FullHd => 429_000,
            Self::Hd => 143_000,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwitchXInput {
    pub generation_type: GenerationType,
    pub prompt: Option<String>,
    pub reference_image: bool,
    pub alpha_mode: AlphaMode,
    pub max_resolution: Resolution,
    pub seed: i64,
    pub source_frame_count: u64,
    pub mask_frame_count: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadRole {
    Keyframe,
    AlphaImage,
    AlphaVideo,
    SourceImage,
    SourceVideo,
    Reference,
}

impl UploadRole {
    pub fn wait_label(self) -> &'static str {
        match self {
            Self::Keyframe => "Uploading keyframe",
            Self::AlphaImage => "Uploading alpha",
            Self::AlphaVideo => "Uploading alpha video",
            Self::SourceImage | Self::SourceVideo => "Uploading source",
            Self::Reference => "Uploading reference",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreparedInput {
    generation_type: GenerationType,
    prompt: Option<String>,
    reference_image: bool,
    alpha_mode: AlphaMode,
    max_resolution: Resolution,
    uploads: Vec<UploadRole>,
}

impl SwitchXInput {
    pub fn prepare(&self) -> Result<PreparedInput, BeebleError> {
        let prompt = self
            .prompt
            .as_deref()
            .unwrap_or_default()
            .trim_matches(|character: char| {
                character.is_whitespace() || matches!(character, '\u{001c}'..='\u{001f}')
            });
        if prompt.is_empty() && !self.reference_image {
            return Err(BeebleError::Input(
                "At least one of 'prompt' or 'reference_image' must be provided.",
            ));
        }
        if prompt.chars().count() > MAXIMUM_PROMPT_CHARACTERS {
            return Err(BeebleError::Input("Beeble prompt exceeds 2000 characters"));
        }
        if !(0..=2_147_483_647).contains(&self.seed) {
            return Err(BeebleError::Input(
                "Beeble seed is outside its source schema",
            ));
        }
        if self.generation_type == GenerationType::Video && self.source_frame_count > MAXIMUM_FRAMES
        {
            return Err(BeebleError::Input("Beeble video exceeds 240 frames"));
        }
        let mut uploads = Vec::new();
        match self.alpha_mode {
            AlphaMode::Auto | AlphaMode::Fill => {}
            AlphaMode::Select | AlphaMode::Custom => {
                let mask_frames = self
                    .mask_frame_count
                    .ok_or(BeebleError::Input("Beeble alpha mode requires a mask"))?;
                if self.alpha_mode == AlphaMode::Select {
                    uploads.push(UploadRole::Keyframe);
                } else if self.generation_type == GenerationType::Video {
                    if mask_frames != self.source_frame_count {
                        return Err(BeebleError::Input(
                            "Custom alpha video frame count does not match the source video",
                        ));
                    }
                    uploads.push(UploadRole::AlphaVideo);
                } else {
                    uploads.push(UploadRole::AlphaImage);
                }
            }
        }
        uploads.push(match self.generation_type {
            GenerationType::Image => UploadRole::SourceImage,
            GenerationType::Video => UploadRole::SourceVideo,
        });
        if self.reference_image {
            uploads.push(UploadRole::Reference);
        }
        Ok(PreparedInput {
            generation_type: self.generation_type,
            prompt: (!prompt.is_empty()).then(|| prompt.to_owned()),
            reference_image: self.reference_image,
            alpha_mode: self.alpha_mode,
            max_resolution: self.max_resolution,
            uploads,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadedInputs {
    pub source_uri: String,
    pub alpha_uri: Option<String>,
    pub reference_image_uri: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateSwitchXRequest<'a> {
    generation_type: GenerationType,
    source_uri: &'a str,
    alpha_mode: AlphaMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_image_uri: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    alpha_uri: Option<&'a str>,
    max_resolution: u16,
}

impl PreparedInput {
    pub fn uploads(&self) -> &[UploadRole] {
        &self.uploads
    }

    pub fn request<'a>(
        &'a self,
        uploaded: &'a UploadedInputs,
    ) -> Result<CreateSwitchXRequest<'a>, BeebleError> {
        if uploaded.alpha_uri.is_some()
            != matches!(self.alpha_mode, AlphaMode::Select | AlphaMode::Custom)
            || uploaded.reference_image_uri.is_some() != self.reference_image
        {
            return Err(BeebleError::Input(
                "Beeble uploaded inputs do not match the prepared inputs",
            ));
        }
        Ok(CreateSwitchXRequest {
            generation_type: self.generation_type,
            source_uri: &uploaded.source_uri,
            alpha_mode: self.alpha_mode,
            prompt: self.prompt.as_deref(),
            reference_image_uri: uploaded.reference_image_uri.as_deref(),
            alpha_uri: uploaded.alpha_uri.as_deref(),
            max_resolution: self.max_resolution.pixels(),
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct SwitchXOutputUrls {
    pub render: Option<String>,
    pub source: Option<String>,
    pub alpha: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SwitchXStatusResponse {
    pub id: String,
    pub status: String,
    pub progress: Option<i64>,
    pub generation_type: Option<String>,
    pub alpha_mode: Option<String>,
    pub output: Option<SwitchXOutputUrls>,
    pub error: Option<String>,
    pub created_at: Option<String>,
    pub modified_at: Option<String>,
    pub completed_at: Option<String>,
}

impl SwitchXStatusResponse {
    pub fn parse(bytes: &[u8]) -> Result<Self, BeebleError> {
        if bytes.len() > MAXIMUM_RESPONSE_BYTES {
            return Err(BeebleError::ResponseLimit);
        }
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn output_urls(&self, mode: AlphaMode) -> Result<(&str, Option<&str>), BeebleError> {
        let render = self
            .output
            .as_ref()
            .and_then(|output| output.render.as_deref())
            .ok_or_else(|| BeebleError::MissingRender(self.id.clone()))?;
        // Fill responses can contain a signed alpha URL for an asset that does not exist.
        let alpha = if mode == AlphaMode::Fill {
            None
        } else {
            self.output
                .as_ref()
                .and_then(|output| output.alpha.as_deref())
        };
        Ok((render, alpha))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceFixture {
    schema_version: u32,
    namespace: String,
    source_sha256: String,
    validation_sha256: String,
    api_sha256: String,
    responses: Vec<SourceResponse>,
    cases: Vec<SourceCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceResponse {
    mode: AlphaMode,
    response: serde_json::Value,
    expected_urls: Option<(String, Option<String>)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCase {
    name: String,
    input: SwitchXInput,
    expected_request: Option<serde_json::Value>,
    expected_uploads: Vec<UploadRole>,
    expected_error: bool,
}

pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    if fixture.len() > MAXIMUM_RESPONSE_BYTES {
        return Err(ConformanceError::BodyLimit);
    }
    let fixture: SourceFixture =
        serde_json::from_str(fixture).map_err(|_| ConformanceError::InvalidRequest)?;
    if fixture.schema_version != 1
        || fixture.namespace != NAMESPACE
        || fixture.source_sha256 != SOURCE_SHA256
        || fixture.validation_sha256
            != "77554cc809f651f54943f73e6391b8899b3a52779d949edfc08166d8fdf86c66"
        || fixture.api_sha256 != "19a0a82f85ee6bc1f514a0af5174029ee868c4650d0e5f35343a8bca1188e356"
        || fixture.cases.len() != 25
        || fixture.responses.len() != 16
    {
        return Err(ConformanceError::InvalidRequest);
    }
    for case in fixture.responses {
        let bytes =
            serde_json::to_vec(&case.response).map_err(|_| ConformanceError::InvalidResponse)?;
        let response =
            SwitchXStatusResponse::parse(&bytes).map_err(|_| ConformanceError::InvalidResponse)?;
        match (response.output_urls(case.mode), case.expected_urls) {
            (Ok((render, alpha)), Some((expected_render, expected_alpha)))
                if render == expected_render && alpha == expected_alpha.as_deref() => {}
            (Err(BeebleError::MissingRender(_)), None) => {}
            _ => return Err(ConformanceError::InvalidResponse),
        }
    }
    let mut names = std::collections::BTreeSet::new();
    for case in fixture.cases {
        if !names.insert(case.name) {
            return Err(ConformanceError::InvalidRequest);
        }
        let prepared = case.input.prepare();
        if case.expected_error {
            if prepared.is_ok() || case.expected_request.is_some() {
                return Err(ConformanceError::InvalidRequest);
            }
            continue;
        }
        let prepared = prepared.map_err(|_| ConformanceError::InvalidRequest)?;
        let uploaded = UploadedInputs {
            source_uri: "https://fixture.invalid/source".into(),
            alpha_uri: matches!(case.input.alpha_mode, AlphaMode::Select | AlphaMode::Custom)
                .then(|| "https://fixture.invalid/alpha".into()),
            reference_image_uri: case
                .input
                .reference_image
                .then(|| "https://fixture.invalid/reference".into()),
        };
        let request = prepared
            .request(&uploaded)
            .map_err(|_| ConformanceError::InvalidRequest)?;
        let request =
            serde_json::to_value(request).map_err(|_| ConformanceError::InvalidRequest)?;
        if Some(request) != case.expected_request || prepared.uploads() != case.expected_uploads {
            return Err(ConformanceError::InvalidRequest);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_source_control_flow_and_output_cases_match() -> Result<(), Box<dyn Error>> {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/beeble/conformance.fixture"
        )))?;
        Ok(())
    }

    #[test]
    fn malformed_trailing_data_and_oversized_responses_fail() {
        for bytes in [
            b"{}".as_slice(),
            b"{\"id\":\"job\",\"status\":\"done\"} {}".as_slice(),
            b"{\"id\":\"job\",\"status\":\"done\",\"output\":{\"render\":4}}".as_slice(),
        ] {
            assert!(SwitchXStatusResponse::parse(bytes).is_err());
        }
        assert!(matches!(
            SwitchXStatusResponse::parse(&vec![b' '; MAXIMUM_RESPONSE_BYTES + 1]),
            Err(BeebleError::ResponseLimit)
        ));
    }

    #[test]
    fn uploaded_slots_and_seed_remain_exact() -> Result<(), Box<dyn Error>> {
        let mut input = SwitchXInput {
            generation_type: GenerationType::Image,
            prompt: Some("edit".into()),
            reference_image: false,
            alpha_mode: AlphaMode::Auto,
            max_resolution: Resolution::FullHd,
            seed: 0,
            source_frame_count: 1,
            mask_frame_count: None,
        };
        let mut uploaded = UploadedInputs {
            source_uri: "https://fixture.invalid/source".into(),
            alpha_uri: None,
            reference_image_uri: None,
        };
        let first = serde_json::to_value(input.prepare()?.request(&uploaded)?)?;
        input.seed = 2_147_483_647;
        assert_eq!(
            first,
            serde_json::to_value(input.prepare()?.request(&uploaded)?)?
        );
        input.seed += 1;
        assert!(input.prepare().is_err());
        input.seed = -1;
        assert!(input.prepare().is_err());
        input.seed = 0;
        uploaded.alpha_uri = Some("https://fixture.invalid/alpha".into());
        assert!(input.prepare()?.request(&uploaded).is_err());
        uploaded.alpha_uri = None;
        uploaded.reference_image_uri = Some("https://fixture.invalid/reference".into());
        assert!(input.prepare()?.request(&uploaded).is_err());
        assert_eq!(Resolution::FullHd.price_badge_microunits(), 429_000);
        assert_eq!(Resolution::Hd.price_badge_microunits(), 143_000);
        Ok(())
    }
}
