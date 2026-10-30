use std::{collections::HashMap, fmt, panic::AssertUnwindSafe, sync::Arc, time::Duration};

use graphite_core_types::{
    Color,
    transform::{Footprint, RenderQuality},
};
use graphite_editor::messages::portfolio::document::{
    DocumentMessageHandler,
    utility_types::network_interface::{InputConnector, NodeTemplate},
};
use graphite_glam::{DAffine2, DVec2, UVec2};
use graphite_graph_craft::{
    application_io::resource::{ResourceHash, ResourceId},
    document::{NodeId, NodeInput, value::TaggedValue},
};
use graphite_std::{
    application_io::{ExportFormat, RenderConfig, TimingInformation},
    vector_types::vector::style::RenderMode,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    evaluate::{
        CpuSvgArtboardDocument, CpuSvgArtboardParameters, CpuSvgDocumentError,
        CpuSvgEncodedPngInput, CpuSvgEvaluationBudget, encoded_png_node_ids,
    },
    resources::{GRAPHITE_IMAGE_SOURCE_REVISION, PreparedEncodedPngSource},
};

const PROFILE: &str = "sim.graphite.finite-linked-artboard.v1";
const ORIGINAL_PROFILE: &str = "sim.graphite.source-linked-artboard.v2";

#[derive(Debug)]
pub enum ArtboardPersistenceError {
    Json(serde_json::Error),
    Document(CpuSvgDocumentError),
    MissingEncodedInput,
    PayloadBudgetExceeded,
    UnsupportedPayload,
    SourceMismatch,
    OriginalSourcePanic,
}

impl fmt::Display for ArtboardPersistenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(
                formatter,
                "Cannot read the Graphite engine payload: {error}"
            ),
            Self::Document(error) => error.fmt(formatter),
            Self::MissingEncodedInput => {
                formatter.write_str("The editable payload requires a linked encoded PNG.")
            }
            Self::PayloadBudgetExceeded => {
                formatter.write_str("The editable payload exceeds its byte budget.")
            }
            Self::UnsupportedPayload => formatter
                .write_str("The Graphite document contains unsupported graph or editor state."),
            Self::SourceMismatch => {
                formatter.write_str("The verified PNG does not match the saved input link.")
            }
            Self::OriginalSourcePanic => {
                formatter.write_str("Original Graphite document reconstruction panicked.")
            }
        }
    }
}

impl std::error::Error for ArtboardPersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Document(error) => Some(error),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for ArtboardPersistenceError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
impl From<CpuSvgDocumentError> for ArtboardPersistenceError {
    fn from(error: CpuSvgDocumentError) -> Self {
        Self::Document(error)
    }
}

#[derive(Clone)]
pub struct PreparedArtboardEnginePayload {
    bytes: Arc<[u8]>,
    legacy_document: Value,
    resource_id: ResourceId,
}

impl PreparedArtboardEnginePayload {
    pub fn serialized_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn legacy_document(&self) -> &Value {
        &self.legacy_document
    }
    pub fn resource_id(&self) -> ResourceId {
        self.resource_id
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    profile: String,
    source_revision: String,
    legacy_document: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    legacy_serialization: Option<String>,
    capture: Capture,
}

// These integer bit patterns belong to the opaque capture record, not the
// original document schema or an assertion of durable host version authority.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Capture {
    document_id_encoding: u64,
    node_id: u64,
    revision: u64,
    location: [u64; 2],
    dimensions: [u64; 2],
    background: [u32; 4],
    clip: bool,
    configuration: Configuration,
    placement: [u64; 6],
    resource_id: u64,
    input: InputLink,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputLink {
    asset_id: String,
    version_id: String,
    encoded_sha256: [u8; 32],
    width: u32,
    height: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    viewport_transform: [u64; 6],
    viewport_resolution: [u32; 2],
    quality: Quality,
    scale: u64,
    time: u64,
    animation_seconds: u64,
    animation_nanoseconds: u32,
    pointer: [u64; 2],
    render_mode: RenderMode,
    export_format: ExportFormat,
    for_export: bool,
    for_eyedropper: bool,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Quality {
    Preview,
    Full,
    Scale(u32),
    Probability(u32),
}

impl Configuration {
    fn capture(configuration: RenderConfig) -> Self {
        Self {
            viewport_transform: configuration
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits),
            viewport_resolution: configuration.viewport.resolution.to_array(),
            quality: match configuration.viewport.quality {
                RenderQuality::Preview => Quality::Preview,
                RenderQuality::Full => Quality::Full,
                RenderQuality::Scale(value) => Quality::Scale(value.to_bits()),
                RenderQuality::Probability(value) => Quality::Probability(value.to_bits()),
            },
            scale: configuration.scale.to_bits(),
            time: configuration.time.time.to_bits(),
            animation_seconds: configuration.time.animation_time.as_secs(),
            animation_nanoseconds: configuration.time.animation_time.subsec_nanos(),
            pointer: configuration.pointer.to_array().map(f64::to_bits),
            render_mode: configuration.render_mode,
            export_format: configuration.export_format,
            for_export: configuration.for_export,
            for_eyedropper: configuration.for_eyedropper,
        }
    }

    fn restore(&self) -> Result<RenderConfig, ArtboardPersistenceError> {
        if self.animation_nanoseconds >= 1_000_000_000 {
            return Err(ArtboardPersistenceError::UnsupportedPayload);
        }
        Ok(RenderConfig {
            viewport: Footprint {
                transform: DAffine2::from_cols_array(&self.viewport_transform.map(f64::from_bits)),
                resolution: UVec2::from_array(self.viewport_resolution),
                quality: match self.quality {
                    Quality::Preview => RenderQuality::Preview,
                    Quality::Full => RenderQuality::Full,
                    Quality::Scale(bits) => RenderQuality::Scale(f32::from_bits(bits)),
                    Quality::Probability(bits) => RenderQuality::Probability(f32::from_bits(bits)),
                },
            },
            scale: f64::from_bits(self.scale),
            time: TimingInformation {
                time: f64::from_bits(self.time),
                animation_time: Duration::new(self.animation_seconds, self.animation_nanoseconds),
            },
            pointer: DVec2::from_array(self.pointer.map(f64::from_bits)),
            render_mode: self.render_mode,
            export_format: self.export_format,
            for_export: self.for_export,
            for_eyedropper: self.for_eyedropper,
        })
    }
}

impl Capture {
    fn parameters(&self) -> CpuSvgArtboardParameters {
        let [red, green, blue, alpha] = self.background.map(f32::from_bits);
        CpuSvgArtboardParameters {
            location: DVec2::from_array(self.location.map(f64::from_bits)),
            dimensions: DVec2::from_array(self.dimensions.map(f64::from_bits)),
            background: Color::from_rgbaf32_unchecked(red, green, blue, alpha),
            clip: self.clip,
        }
    }
    fn placement(&self) -> DAffine2 {
        DAffine2::from_cols_array(&self.placement.map(f64::from_bits))
    }
    fn source_matches(&self, source: &PreparedEncodedPngSource) -> bool {
        let identity = source.identity();
        self.input.asset_id == identity.asset_id.as_ref()
            && self.input.version_id == identity.version_id.as_ref()
            && self.input.encoded_sha256 == identity.content_sha256
            && self.input.width == source.width()
            && self.input.height == source.height()
    }
}

fn capture_document(
    document: &CpuSvgArtboardDocument,
    resource_id: ResourceId,
) -> Result<Capture, ArtboardPersistenceError> {
    if document.node_id().0 == u64::MAX {
        return Err(ArtboardPersistenceError::UnsupportedPayload);
    }
    let input = document
        .encoded_png_input()
        .ok_or(ArtboardPersistenceError::MissingEncodedInput)?;
    let source = input.source();
    let parameters = document.parameters();
    let configuration = document.configuration();
    Ok(Capture {
        document_id_encoding: document.document_id_encoding(),
        node_id: document.node_id().0,
        revision: document.revision(),
        location: parameters.location.to_array().map(f64::to_bits),
        dimensions: parameters.dimensions.to_array().map(f64::to_bits),
        background: [
            parameters.background.r(),
            parameters.background.g(),
            parameters.background.b(),
            parameters.background.a(),
        ]
        .map(f32::to_bits),
        clip: parameters.clip,
        configuration: Configuration::capture(configuration),
        placement: input.placement().to_cols_array().map(f64::to_bits),
        resource_id: u64::from(resource_id),
        input: InputLink {
            asset_id: source.identity().asset_id.to_string(),
            version_id: source.identity().version_id.to_string(),
            encoded_sha256: source.identity().content_sha256,
            width: source.width(),
            height: source.height(),
        },
    })
}

pub fn capture_artboard_engine_payload(
    document: &CpuSvgArtboardDocument,
    resource_id: ResourceId,
    max_payload_bytes: usize,
) -> Result<PreparedArtboardEnginePayload, ArtboardPersistenceError> {
    let capture = capture_document(document, resource_id)?;
    let source = document
        .encoded_png_input()
        .ok_or(ArtboardPersistenceError::MissingEncodedInput)?
        .source();
    let original = persistent_document(&capture, source)?;
    let legacy_document = serde_json::to_value(original)?;
    let payload = Payload {
        profile: PROFILE.into(),
        source_revision: GRAPHITE_IMAGE_SOURCE_REVISION.into(),
        legacy_document: legacy_document.clone(),
        legacy_serialization: None,
        capture,
    };
    let bytes = serde_json::to_vec(&payload)?;
    if bytes.len() > max_payload_bytes {
        return Err(ArtboardPersistenceError::PayloadBudgetExceeded);
    }
    Ok(PreparedArtboardEnginePayload {
        bytes: bytes.into(),
        legacy_document,
        resource_id,
    })
}

pub fn capture_original_artboard_engine_payload(
    document: &CpuSvgArtboardDocument,
    resource_id: ResourceId,
    legacy_bytes: &[u8],
    max_payload_bytes: usize,
) -> Result<PreparedArtboardEnginePayload, ArtboardPersistenceError> {
    if legacy_bytes.len() > max_payload_bytes {
        return Err(ArtboardPersistenceError::PayloadBudgetExceeded);
    }
    let capture = capture_document(document, resource_id)?;
    let source = document
        .encoded_png_input()
        .ok_or(ArtboardPersistenceError::MissingEncodedInput)?
        .source();
    let legacy_document: Value = serde_json::from_slice(legacy_bytes)?;
    validate_original_document(&legacy_document, &capture, source)?;
    let legacy_serialization = std::str::from_utf8(legacy_bytes)
        .map_err(|_| ArtboardPersistenceError::UnsupportedPayload)?
        .to_owned();
    let payload = Payload {
        profile: ORIGINAL_PROFILE.into(),
        source_revision: GRAPHITE_IMAGE_SOURCE_REVISION.into(),
        legacy_document: legacy_document.clone(),
        legacy_serialization: Some(legacy_serialization),
        capture,
    };
    let bytes = serde_json::to_vec(&payload)?;
    if bytes.len() > max_payload_bytes {
        return Err(ArtboardPersistenceError::PayloadBudgetExceeded);
    }
    Ok(PreparedArtboardEnginePayload {
        bytes: bytes.into(),
        legacy_document,
        resource_id,
    })
}

pub struct ReopenedArtboardEnginePayload {
    legacy_bytes: Arc<[u8]>,
    document: CpuSvgArtboardDocument,
    resource_id: ResourceId,
}

impl ReopenedArtboardEnginePayload {
    pub fn legacy_document_bytes(&self) -> &[u8] {
        &self.legacy_bytes
    }
    pub fn resource_id(&self) -> ResourceId {
        self.resource_id
    }
    pub fn into_document(self) -> CpuSvgArtboardDocument {
        self.document
    }
}

pub fn reopen_artboard_engine_payload(
    bytes: &[u8],
    source: Arc<PreparedEncodedPngSource>,
    budget: CpuSvgEvaluationBudget,
    max_payload_bytes: usize,
) -> Result<CpuSvgArtboardDocument, ArtboardPersistenceError> {
    reopen_artboard_engine_payload_with_resource(bytes, source, budget, max_payload_bytes)
        .map(ReopenedArtboardEnginePayload::into_document)
}

pub fn reopen_artboard_engine_payload_with_resource(
    bytes: &[u8],
    source: Arc<PreparedEncodedPngSource>,
    budget: CpuSvgEvaluationBudget,
    max_payload_bytes: usize,
) -> Result<ReopenedArtboardEnginePayload, ArtboardPersistenceError> {
    reopen_artboard_engine_payload_for_profile(bytes, source, budget, max_payload_bytes, PROFILE)
}

pub fn reopen_original_artboard_engine_payload_with_resource(
    bytes: &[u8],
    source: Arc<PreparedEncodedPngSource>,
    budget: CpuSvgEvaluationBudget,
    max_payload_bytes: usize,
) -> Result<ReopenedArtboardEnginePayload, ArtboardPersistenceError> {
    reopen_artboard_engine_payload_for_profile(
        bytes,
        source,
        budget,
        max_payload_bytes,
        ORIGINAL_PROFILE,
    )
}

fn reopen_artboard_engine_payload_for_profile(
    bytes: &[u8],
    source: Arc<PreparedEncodedPngSource>,
    budget: CpuSvgEvaluationBudget,
    max_payload_bytes: usize,
    expected_profile: &str,
) -> Result<ReopenedArtboardEnginePayload, ArtboardPersistenceError> {
    if bytes.len() > max_payload_bytes {
        return Err(ArtboardPersistenceError::PayloadBudgetExceeded);
    }
    let payload: Payload = serde_json::from_slice(bytes)?;
    if payload.profile != expected_profile
        || payload.source_revision != GRAPHITE_IMAGE_SOURCE_REVISION
    {
        return Err(ArtboardPersistenceError::UnsupportedPayload);
    }
    if !payload.capture.source_matches(&source) {
        return Err(ArtboardPersistenceError::SourceMismatch);
    }
    if payload.capture.node_id == u64::MAX {
        return Err(ArtboardPersistenceError::UnsupportedPayload);
    }
    let parameters = payload.capture.parameters();
    let configuration = payload.capture.configuration.restore()?;
    let input = CpuSvgEncodedPngInput::new(source.clone(), payload.capture.placement())?;
    let document = CpuSvgArtboardDocument::from_persisted_checkpoint(
        payload.capture.document_id_encoding,
        NodeId(payload.capture.node_id),
        payload.capture.revision,
        parameters,
        configuration,
        budget,
        input,
    )?;
    let legacy_bytes = match payload.profile.as_str() {
        PROFILE if payload.legacy_serialization.is_none() => {
            let expected = serde_json::to_value(persistent_document(&payload.capture, &source)?)?;
            if !same_original_json(&payload.legacy_document, &expected) {
                return Err(ArtboardPersistenceError::UnsupportedPayload);
            }
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                serde_json::from_value::<DocumentMessageHandler>(payload.legacy_document.clone())
            }))
            .map_err(|_| ArtboardPersistenceError::OriginalSourcePanic)??;
            serde_json::to_vec(&payload.legacy_document)?
        }
        ORIGINAL_PROFILE => {
            let serialization = payload
                .legacy_serialization
                .ok_or(ArtboardPersistenceError::UnsupportedPayload)?;
            let original: Value = serde_json::from_str(&serialization)?;
            if !same_original_json(&payload.legacy_document, &original) {
                return Err(ArtboardPersistenceError::UnsupportedPayload);
            }
            validate_original_document(&original, &payload.capture, &source)?;
            serialization.into_bytes()
        }
        _ => return Err(ArtboardPersistenceError::UnsupportedPayload),
    };
    Ok(ReopenedArtboardEnginePayload {
        legacy_bytes: legacy_bytes.into(),
        document,
        resource_id: ResourceId::from(payload.capture.resource_id),
    })
}

fn validate_original_document(
    original: &Value,
    capture: &Capture,
    source: &PreparedEncodedPngSource,
) -> Result<(), ArtboardPersistenceError> {
    let expected = serde_json::to_value(persistent_document(capture, source)?)?;
    let network = original
        .get("network_interface")
        .and_then(|interface| interface.get("network"))
        .ok_or(ArtboardPersistenceError::UnsupportedPayload)?;
    let expected_network = expected
        .get("network_interface")
        .and_then(|interface| interface.get("network"))
        .ok_or(ArtboardPersistenceError::UnsupportedPayload)?;
    if !same_original_json(network, expected_network)
        || original.get("resources") != expected.get("resources")
        || original
            .get("pending_gradient_bbox_bake")
            .is_some_and(|pending| pending.as_array().is_none_or(|pending| !pending.is_empty()))
    {
        return Err(ArtboardPersistenceError::UnsupportedPayload);
    }
    // Resource deserialization can decode embedded bytes. Exclude those and foreign
    // declarations before asking the original owner to read otherwise native state.
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut owner: DocumentMessageHandler = serde_json::from_value(original.clone())?;
        // The original loader intentionally skips this historical build hint. It is
        // retained here as source data, without treating it as runtime authority.
        owner.commit_hash = original
            .get("commit_hash")
            .and_then(Value::as_str)
            .ok_or(ArtboardPersistenceError::UnsupportedPayload)?
            .to_owned();
        let roundtrip = serde_json::to_value(owner)?;
        if !same_original_json(original, &roundtrip) {
            return Err(ArtboardPersistenceError::UnsupportedPayload);
        }
        Ok(())
    }))
    .map_err(|_| ArtboardPersistenceError::OriginalSourcePanic)?
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Templates {
    artboard: NodeTemplate,
    replace_transform: NodeTemplate,
    resource: NodeTemplate,
    decode_image: NodeTemplate,
}

fn persistent_document(
    capture: &Capture,
    source: &PreparedEncodedPngSource,
) -> Result<DocumentMessageHandler, ArtboardPersistenceError> {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let mut templates: Templates =
            serde_json::from_str(include_str!("persistent-artboard-templates.json"))?;
        let node_id = NodeId(capture.node_id);
        let [_, decode_id, placement_id] = encoded_png_node_ids(node_id)?;
        let parameters = capture.parameters();
        for (index, value) in [
            (2, TaggedValue::DVec2(parameters.location)),
            (3, TaggedValue::DVec2(parameters.dimensions)),
            (4, TaggedValue::Color(parameters.background)),
            (5, TaggedValue::Bool(parameters.clip)),
        ] {
            let input = templates
                .artboard
                .inputs
                .get_mut(index)
                .ok_or(ArtboardPersistenceError::UnsupportedPayload)?;
            *input = NodeInput::value(value, false);
        }
        *templates
            .artboard
            .inputs
            .get_mut(1)
            .ok_or(ArtboardPersistenceError::UnsupportedPayload)? =
            NodeInput::node(placement_id, 0);
        *templates
            .decode_image
            .inputs
            .get_mut(0)
            .ok_or(ArtboardPersistenceError::UnsupportedPayload)? = NodeInput::value(
            TaggedValue::Resource(ResourceId::from(capture.resource_id)),
            false,
        );
        templates.replace_transform.inputs = vec![
            NodeInput::node(decode_id, 0),
            NodeInput::value(TaggedValue::DAffine2(capture.placement()), false),
        ];
        // ResourceHash is runtime-only. The original preprocessor resolves this
        // persistent ResourceId and inserts the Resource node during evaluation.
        if templates.resource.inputs.len() != 2 {
            return Err(ArtboardPersistenceError::UnsupportedPayload);
        }
        let mut original = DocumentMessageHandler::default();
        let nodes = vec![
            (node_id, templates.artboard),
            (decode_id, templates.decode_image),
            (placement_id, templates.replace_transform),
        ];
        let identifiers: HashMap<_, _> = nodes
            .iter()
            .map(|(identifier, _)| (*identifier, *identifier))
            .collect();
        original
            .network_interface
            .insert_node_group(nodes, identifiers, &[]);
        original.network_interface.set_input(
            &InputConnector::Export(0),
            NodeInput::node(node_id, 0),
            &[],
        );
        let resource_id = ResourceId::from(capture.resource_id);
        if original
            .resources
            .registry
            .resolve(
                &resource_id,
                ResourceHash::from(source.encoded_bytes().as_ref()),
            )
            .is_some()
        {
            return Err(ArtboardPersistenceError::UnsupportedPayload);
        }
        original.commit_hash = GRAPHITE_IMAGE_SOURCE_REVISION.into();
        Ok(original)
    }))
    .map_err(|_| ArtboardPersistenceError::OriginalSourcePanic)?
}

fn same_original_json(first: &Value, second: &Value) -> bool {
    match (first, second) {
        (Value::Number(first), Value::Number(second)) if first.is_f64() || second.is_f64() => {
            first.as_f64().map(f64::to_bits) == second.as_f64().map(f64::to_bits)
        }
        (Value::Array(first), Value::Array(second)) => {
            first.len() == second.len()
                && first
                    .iter()
                    .zip(second)
                    .all(|(first, second)| same_original_json(first, second))
        }
        (Value::Object(first), Value::Object(second)) => {
            first.len() == second.len()
                && first.iter().all(|(key, first)| {
                    second
                        .get(key)
                        .is_some_and(|second| same_original_json(first, second))
                })
        }
        _ => first == second,
    }
}

#[cfg(test)]
mod original_source_tests {
    use super::*;
    use crate::resources::{EncodedPngBudget, ImageResourceIdentity, prepare_encoded_png_source};
    use sha2::{Digest, Sha256};

    const RESOURCE_ID: u64 = 904;
    const PAYLOAD_LIMIT: usize = 512 * 1024;

    fn encoded_source(version: &str) -> Arc<PreparedEncodedPngSource> {
        let encoded = crate::export::prepare_rgba8_png(
            2,
            2,
            vec![
                231, 17, 83, 0, 19, 223, 71, 1, 11, 53, 241, 127, 201, 31, 97, 255,
            ],
            4096,
        )
        .expect("actual shared PNG encoder");
        let bytes: Arc<[u8]> = encoded.png().encoded_bytes().into();
        Arc::new(
            prepare_encoded_png_source(
                ImageResourceIdentity {
                    asset_id: "original-owner-format-fixture".into(),
                    version_id: version.into(),
                    content_sha256: Sha256::digest(bytes.as_ref()).into(),
                },
                bytes,
                EncodedPngBudget {
                    max_encoded_bytes: 4096,
                    max_pixel_bytes: 4 * (4 + 2 * std::mem::size_of::<Color>()),
                    max_metadata_bytes: 4096,
                },
            )
            .expect("genuine encoded envelope"),
        )
    }

    fn document(source: Arc<PreparedEncodedPngSource>) -> CpuSvgArtboardDocument {
        let configuration = RenderConfig {
            viewport: Footprint {
                transform: DAffine2::from_cols_array(&[1., -0., 0., 1., -0., 2.]),
                resolution: UVec2::new(31, 17),
                quality: RenderQuality::Preview,
            },
            scale: 1.5,
            time: TimingInformation {
                time: -0.,
                animation_time: Duration::new(2, 13),
            },
            pointer: DVec2::new(-0., 5.),
            render_mode: RenderMode::Normal,
            export_format: ExportFormat::Svg,
            for_export: true,
            for_eyedropper: false,
        };
        CpuSvgArtboardDocument::new_with_encoded_png(
            81,
            NodeId(7001),
            CpuSvgArtboardParameters {
                location: DVec2::new(-0., 2.),
                dimensions: DVec2::new(6., 4.),
                background: Color::from_rgbaf32_unchecked(-0., 0.25, 0.75, 0.5),
                clip: true,
            },
            configuration,
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
            CpuSvgEncodedPngInput::new(
                source,
                DAffine2::from_cols_array(&[2., -0., 0., 2., 3., 2.]),
            )
            .expect("explicit finite original image placement"),
        )
        .expect("admitted finite checkpoint")
    }

    fn original_owner(document: &CpuSvgArtboardDocument) -> DocumentMessageHandler {
        let capture = capture_document(document, ResourceId::from(RESOURCE_ID))
            .expect("exact admitted checkpoint");
        let source = document
            .encoded_png_input()
            .expect("retained image")
            .source();
        let mut owner = persistent_document(&capture, source).expect("authentic source templates");
        owner.document_ptz.pan = DVec2::new(-0., 17.25);
        owner.document_ptz.set_tilt(0.375);
        owner.document_ptz.set_zoom(1.5);
        owner.document_ptz.flip = true;
        owner.rulers_visible = false;
        owner.graph_view_overlay_open = true;
        owner.graph_fade_artwork_percentage = 0.125;
        owner.properties_panel_collapsed_sections = vec![document.node_id()];
        owner.network_interface.set_display_name(
            &document.node_id(),
            "Original layer — 保存".into(),
            &[],
        );
        owner
    }

    fn reopen(
        payload: &PreparedArtboardEnginePayload,
        source: Arc<PreparedEncodedPngSource>,
    ) -> ReopenedArtboardEnginePayload {
        reopen_original_artboard_engine_payload_with_resource(
            payload.serialized_bytes(),
            source,
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
            PAYLOAD_LIMIT,
        )
        .expect("genuine supported saved source")
    }

    #[test]
    fn at_r06_9_1_original_serializer_bytes_metadata_resources_and_bits_are_retained() {
        let source = encoded_source("first");
        let document = document(source.clone());
        let owner = original_owner(&document);
        let raw = format!(" \n{}\n\t", owner.serialize_document()).into_bytes();
        let original: Value = serde_json::from_slice(&raw).expect("actual owner serialization");
        let network = document.network().clone();
        let payload = capture_original_artboard_engine_payload(
            &document,
            ResourceId::from(RESOURCE_ID),
            &raw,
            PAYLOAD_LIMIT,
        )
        .expect("admitted original source serialization");
        assert!(same_original_json(payload.legacy_document(), &original));
        let envelope: Value =
            serde_json::from_slice(payload.serialized_bytes()).expect("typed envelope");
        assert_eq!(envelope["profile"], ORIGINAL_PROFILE);
        assert_eq!(
            envelope["legacy_serialization"]
                .as_str()
                .expect("owned original serialization")
                .as_bytes(),
            raw
        );
        let restored = reopen(&payload, source.clone());
        assert_eq!(restored.legacy_document_bytes(), raw);
        assert_eq!(restored.resource_id(), ResourceId::from(RESOURCE_ID));
        let actual_original: DocumentMessageHandler =
            serde_json::from_slice(restored.legacy_document_bytes())
                .expect("original source deserializer accepts retained owner state");
        assert_eq!(actual_original.network_interface, owner.network_interface);
        assert_eq!(actual_original.resources, owner.resources);
        assert_eq!(
            actual_original.document_ptz.pan.x.to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            actual_original.document_ptz.pan.y.to_bits(),
            17.25_f64.to_bits()
        );
        assert_eq!(
            actual_original.document_ptz.unmodified_tilt().to_bits(),
            0.375_f64.to_bits()
        );
        assert_eq!(
            actual_original.document_ptz.zoom().to_bits(),
            1.5_f64.to_bits()
        );
        assert!(actual_original.document_ptz.flip);
        assert_eq!(
            actual_original.properties_panel_collapsed_sections,
            vec![NodeId(7001)]
        );
        assert!(!actual_original.rulers_visible);
        assert!(actual_original.graph_view_overlay_open);
        assert_eq!(
            actual_original.graph_fade_artwork_percentage.to_bits(),
            0.125_f64.to_bits()
        );
        let restored = restored.into_document();
        assert_eq!(
            restored.document_id_encoding(),
            document.document_id_encoding()
        );
        assert_eq!(restored.node_id(), document.node_id());
        assert_eq!(restored.revision(), document.revision());
        assert_eq!(
            restored.parameters().location.x.to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            restored.parameters().background.r().to_bits(),
            (-0_f32).to_bits()
        );
        assert_eq!(
            restored
                .encoded_png_input()
                .expect("retained source")
                .placement()
                .to_cols_array()
                .map(f64::to_bits),
            document
                .encoded_png_input()
                .expect("original input")
                .placement()
                .to_cols_array()
                .map(f64::to_bits)
        );
        assert!(Arc::ptr_eq(
            restored
                .encoded_png_input()
                .expect("retained source")
                .source(),
            &source
        ));
        assert!(same_original_json(
            &serde_json::to_value(Configuration::capture(restored.configuration()))
                .expect("captured configuration"),
            &serde_json::to_value(Configuration::capture(document.configuration()))
                .expect("original configuration")
        ));
        assert_eq!(document.network(), &network);
        assert_eq!(
            format!(" \n{}\n\t", owner.serialize_document()).as_bytes(),
            raw
        );
    }

    #[test]
    fn at_r06_9_1_original_source_graph_resource_and_unknown_state_fail_closed() {
        let source = encoded_source("first");
        let document = document(source);
        let owner = original_owner(&document);
        let raw = owner.serialize_document();
        let mut changed = owner.clone();
        changed.network_interface.set_input(
            &InputConnector::Node {
                node_id: document.node_id(),
                input_index: 2,
            },
            NodeInput::value(TaggedValue::DVec2(DVec2::new(19., 2.)), false),
            &[],
        );
        assert!(matches!(
            capture_original_artboard_engine_payload(
                &document,
                ResourceId::from(RESOURCE_ID),
                changed.serialize_document().as_bytes(),
                PAYLOAD_LIMIT
            ),
            Err(ArtboardPersistenceError::UnsupportedPayload)
        ));
        let mut changed = owner.clone();
        assert!(
            changed
                .resources
                .registry
                .resolve(
                    &ResourceId::from(RESOURCE_ID),
                    ResourceHash::from(&b"wrong encoded source"[..]),
                )
                .is_some()
        );
        assert!(matches!(
            capture_original_artboard_engine_payload(
                &document,
                ResourceId::from(RESOURCE_ID),
                changed.serialize_document().as_bytes(),
                PAYLOAD_LIMIT
            ),
            Err(ArtboardPersistenceError::UnsupportedPayload)
        ));
        for (field, value) in [
            (
                "future_editor_state",
                serde_json::json!({"state":"must not be discarded"}),
            ),
            (
                "pending_gradient_bbox_bake",
                serde_json::json!([[[], 7001, {}]]),
            ),
            (
                "resources",
                serde_json::json!({"embedded":{"untrusted":"not base64"}}),
            ),
        ] {
            let mut changed: Value = serde_json::from_str(&raw).expect("actual source document");
            changed[field] = value;
            let changed = serde_json::to_vec(&changed).expect("negative payload fixture");
            assert!(
                matches!(
                    capture_original_artboard_engine_payload(
                        &document,
                        ResourceId::from(RESOURCE_ID),
                        &changed,
                        PAYLOAD_LIMIT
                    ),
                    Err(ArtboardPersistenceError::UnsupportedPayload)
                ),
                "unsupported {field}"
            );
        }
        assert_eq!(owner.serialize_document(), raw);
        assert!(
            capture_original_artboard_engine_payload(
                &document,
                ResourceId::from(RESOURCE_ID),
                raw.as_bytes(),
                PAYLOAD_LIMIT
            )
            .is_ok()
        );
    }

    #[test]
    fn at_r06_9_1_original_snapshots_keep_native_metadata_and_independent_finite_context() {
        let source = encoded_source("first");
        let mut document = document(source.clone());
        let mut owner = original_owner(&document);
        let first_raw = owner.serialize_document();
        let first = capture_original_artboard_engine_payload(
            &document,
            ResourceId::from(RESOURCE_ID),
            first_raw.as_bytes(),
            PAYLOAD_LIMIT,
        )
        .expect("first actual original snapshot");
        let mut parameters = document.parameters();
        parameters.location.x = 12.;
        let mut configuration = document.configuration();
        configuration.pointer.x = 7.;
        assert!(
            document
                .apply(0, parameters, configuration)
                .expect("finite captured source successor")
        );
        owner.network_interface.set_input(
            &InputConnector::Node {
                node_id: document.node_id(),
                input_index: 2,
            },
            NodeInput::value(TaggedValue::DVec2(parameters.location), false),
            &[],
        );
        let second_raw = owner.serialize_document();
        assert_ne!(first_raw, second_raw);
        let second = capture_original_artboard_engine_payload(
            &document,
            ResourceId::from(RESOURCE_ID),
            second_raw.as_bytes(),
            PAYLOAD_LIMIT,
        )
        .expect("second actual original snapshot");
        let first_reopened = reopen(&first, source.clone());
        let second_reopened = reopen(&second, source.clone());
        assert_eq!(first_reopened.legacy_document_bytes(), first_raw.as_bytes());
        assert_eq!(
            second_reopened.legacy_document_bytes(),
            second_raw.as_bytes()
        );
        let first_document = first_reopened.into_document();
        let second_document = second_reopened.into_document();
        assert_eq!(first_document.revision(), 0);
        assert_eq!(second_document.revision(), 1);
        assert_eq!(
            first_document.parameters().location.x.to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            second_document.parameters().location.x.to_bits(),
            12_f64.to_bits()
        );
        assert_eq!(
            first_document.configuration().pointer.x.to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            second_document.configuration().pointer.x.to_bits(),
            7_f64.to_bits()
        );
        for payload in [&first, &second] {
            assert_eq!(
                payload.legacy_document()["document_ptz"],
                first.legacy_document()["document_ptz"]
            );
            assert_eq!(
                payload.legacy_document()["resources"],
                first.legacy_document()["resources"]
            );
            assert!(
                payload
                    .legacy_document()
                    .to_string()
                    .contains("Original layer — 保存")
            );
        }
        assert!(Arc::ptr_eq(
            first_document
                .encoded_png_input()
                .expect("first source")
                .source(),
            &source
        ));
        assert!(Arc::ptr_eq(
            second_document
                .encoded_png_input()
                .expect("second source")
                .source(),
            &source
        ));
        assert_eq!(owner.serialize_document(), second_raw);
    }

    #[test]
    fn at_r06_9_1_original_payload_caps_integrity_recovery_and_v1_compatibility() {
        let source = encoded_source("first");
        let document = document(source.clone());
        let owner = original_owner(&document);
        let raw = owner.serialize_document();
        let original = capture_original_artboard_engine_payload(
            &document,
            ResourceId::from(RESOURCE_ID),
            raw.as_bytes(),
            PAYLOAD_LIMIT,
        )
        .expect("actual supported source capture");
        let exact = original.serialized_bytes().len();
        for maximum in [0, exact - 1] {
            assert!(matches!(
                capture_original_artboard_engine_payload(
                    &document,
                    ResourceId::from(RESOURCE_ID),
                    raw.as_bytes(),
                    maximum
                ),
                Err(ArtboardPersistenceError::PayloadBudgetExceeded)
            ));
            assert!(matches!(
                reopen_original_artboard_engine_payload_with_resource(
                    original.serialized_bytes(),
                    source.clone(),
                    CpuSvgEvaluationBudget {
                        max_viewport_pixels: 31 * 17,
                        max_output_bytes: 64 * 1024
                    },
                    maximum
                ),
                Err(ArtboardPersistenceError::PayloadBudgetExceeded)
            ));
        }
        let recovery = capture_original_artboard_engine_payload(
            &document,
            ResourceId::from(RESOURCE_ID),
            raw.as_bytes(),
            exact,
        )
        .expect("exact output cap recovery");
        assert_eq!(recovery.serialized_bytes(), original.serialized_bytes());
        assert!(matches!(
            reopen_original_artboard_engine_payload_with_resource(
                original.serialized_bytes(),
                encoded_source("different-version"),
                CpuSvgEvaluationBudget {
                    max_viewport_pixels: 31 * 17,
                    max_output_bytes: 64 * 1024
                },
                PAYLOAD_LIMIT
            ),
            Err(ArtboardPersistenceError::SourceMismatch)
        ));
        for (field, value) in [
            (
                "profile",
                Value::String("sim.graphite.source-linked-artboard.v999".into()),
            ),
            ("legacy_serialization", Value::String("{}".into())),
            (
                "source_revision",
                Value::String("foreign-source-pin".into()),
            ),
        ] {
            let mut envelope: Value =
                serde_json::from_slice(original.serialized_bytes()).expect("actual envelope");
            envelope[field] = value;
            assert!(matches!(
                reopen_original_artboard_engine_payload_with_resource(
                    &serde_json::to_vec(&envelope).expect("negative fixture"),
                    source.clone(),
                    CpuSvgEvaluationBudget {
                        max_viewport_pixels: 31 * 17,
                        max_output_bytes: 64 * 1024
                    },
                    PAYLOAD_LIMIT
                ),
                Err(ArtboardPersistenceError::UnsupportedPayload)
            ));
        }
        let legacy = capture_artboard_engine_payload(
            &document,
            ResourceId::from(RESOURCE_ID),
            PAYLOAD_LIMIT,
        )
        .expect("existing v1 writer");
        let envelope: Value =
            serde_json::from_slice(legacy.serialized_bytes()).expect("v1 envelope");
        assert_eq!(envelope["profile"], PROFILE);
        assert!(envelope.get("legacy_serialization").is_none());
        assert!(matches!(
            reopen_artboard_engine_payload_with_resource(
                original.serialized_bytes(),
                source.clone(),
                CpuSvgEvaluationBudget {
                    max_viewport_pixels: 31 * 17,
                    max_output_bytes: 64 * 1024
                },
                PAYLOAD_LIMIT
            ),
            Err(ArtboardPersistenceError::UnsupportedPayload)
        ));
        let restored = reopen_artboard_engine_payload_with_resource(
            legacy.serialized_bytes(),
            source,
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
            PAYLOAD_LIMIT,
        )
        .expect("existing finite v1 reader remains compatible");
        assert!(same_original_json(
            &serde_json::from_slice::<Value>(restored.legacy_document_bytes())
                .expect("old serialized native state"),
            legacy.legacy_document()
        ));
        assert_eq!(restored.into_document().revision(), document.revision());
        assert_eq!(owner.serialize_document(), raw);
    }
}
