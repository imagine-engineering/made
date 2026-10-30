use std::{fmt, panic::AssertUnwindSafe, sync::Arc};

use futures::FutureExt as _;
use graphite_core_types::{Color, Context, list::List};
use graphite_glam::{DAffine2, DVec2};
use graphite_graph_craft::{
    application_io::{
        PlatformApplicationIo, PlatformEditorApi,
        resource::{HashMapResourceStorage, ResourceHash, ResourceStorage},
    },
    document::{
        DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
        value::{RenderOutput, RenderOutputType, TaggedValue},
    },
    graphene_compiler::{Compiler, Executor},
};
use graphite_interpreted_executor::dynamic_executor::DynamicExecutor;
use graphite_std::{
    ATTR_TRANSFORM, Artboard, Graphic,
    application_io::{ExportFormat, GetEditorPreferences, RenderConfig},
    memo::IORecord,
    platform_application_io::{decode_image, editor_api, resource},
    transform_nodes::replace_transform,
};

use crate::resources::PreparedEncodedPngSource;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuSvgArtboardParameters {
    pub location: DVec2,
    pub dimensions: DVec2,
    pub background: Color,
    pub clip: bool,
}

impl Default for CpuSvgArtboardParameters {
    fn default() -> Self {
        Self {
            location: DVec2::ZERO,
            dimensions: DVec2::new(1920., 1080.),
            background: Color::WHITE,
            clip: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuSvgEvaluationBudget {
    pub max_viewport_pixels: u32,
    pub max_output_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuSvgDocumentError {
    InvalidConfiguration,
    InvalidGeometry,
    InvalidBackground,
    InvalidPlacement,
    ViewportBudgetExceeded,
    InvalidSourceTemplate,
    StaleRevision,
    RevisionExhausted,
}

impl fmt::Display for CpuSvgDocumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => {
                "The CPU SVG export requires a finite explicit render context."
            }
            Self::InvalidGeometry => "The Artboard geometry exceeds the finite SVG contract.",
            Self::InvalidBackground => "The Artboard background exceeds the finite SVG contract.",
            Self::InvalidPlacement => "The image placement exceeds the finite SVG contract.",
            Self::ViewportBudgetExceeded => "The viewport exceeds the supplied rendering budget.",
            Self::InvalidSourceTemplate => "The pinned original Artboard template is invalid.",
            Self::StaleRevision => "The edit belongs to an earlier document revision.",
            Self::RevisionExhausted => "The document revision is exhausted.",
        })
    }
}

impl std::error::Error for CpuSvgDocumentError {}

// The pinned DynamicExecutor boxes only String failures. Owning the message
// lets genuine completions cross the host worker boundary without moving its
// local interpreter future or erasing the execution error stage.
#[derive(Debug)]
pub struct CpuSvgExecutionError(String);

impl fmt::Display for CpuSvgExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CpuSvgExecutionError {}

#[derive(Debug)]
pub enum CpuSvgEvaluationError {
    Preprocessing(String),
    Compilation(String),
    Construction(Box<graphite_graph_craft::proto::GraphErrors>),
    Execution(CpuSvgExecutionError),
    UnexpectedContract,
    UnexpectedOutput,
    SourceDecodeFailed,
    UnexpectedDecodedImage,
    OutputBudgetExceeded,
    SourcePanic(String),
}

impl fmt::Display for CpuSvgEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preprocessing(error) => {
                write!(formatter, "Original preprocessing failed: {error}")
            }
            Self::Compilation(error) => write!(formatter, "Original compilation failed: {error}"),
            Self::Construction(errors) => {
                write!(formatter, "Original construction failed: {errors:?}")
            }
            Self::Execution(error) => write!(formatter, "Original evaluation failed: {error}"),
            Self::UnexpectedContract => {
                formatter.write_str("The original executor returned an unexpected type contract.")
            }
            Self::UnexpectedOutput => formatter
                .write_str("The original executor did not return SVG with embedded resources."),
            Self::SourceDecodeFailed => {
                formatter.write_str("The original image decoder returned its empty fallback.")
            }
            Self::UnexpectedDecodedImage => formatter
                .write_str("The evaluated image disagrees with its admitted source contract."),
            Self::OutputBudgetExceeded => {
                formatter.write_str("The generated SVG exceeds the supplied output budget.")
            }
            Self::SourcePanic(error) => {
                write!(formatter, "Original SVG evaluation panicked: {error}")
            }
        }
    }
}

impl std::error::Error for CpuSvgEvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum CpuSvgAcceptanceError {
    ForeignOrStaleCompletion,
    Evaluation(CpuSvgEvaluationError),
}

impl fmt::Display for CpuSvgAcceptanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignOrStaleCompletion => formatter
                .write_str("The SVG completion is no longer owned by this document request."),
            Self::Evaluation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CpuSvgAcceptanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct CpuSvgEncodedPngInput {
    source: Arc<PreparedEncodedPngSource>,
    placement: DAffine2,
}

impl CpuSvgEncodedPngInput {
    pub fn new(
        source: Arc<PreparedEncodedPngSource>,
        placement: DAffine2,
    ) -> Result<Self, CpuSvgDocumentError> {
        validate_placement(placement)?;
        Ok(Self { source, placement })
    }

    pub fn source(&self) -> &Arc<PreparedEncodedPngSource> {
        &self.source
    }

    pub fn placement(&self) -> DAffine2 {
        self.placement
    }
}

struct SvgCheckpoint {
    document_id_encoding: u64,
    node_id: NodeId,
    revision: u64,
    network: Arc<NodeNetwork>,
    encoded_png: Option<CpuSvgEncodedPngInput>,
    parameters: CpuSvgArtboardParameters,
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
}

pub struct CpuSvgArtboardDocument {
    checkpoint: Arc<SvgCheckpoint>,
    pending: Option<Arc<()>>,
    accepted: Option<AcceptedCpuSvgOutput>,
}

impl CpuSvgArtboardDocument {
    pub fn new(
        document_id_encoding: u64,
        node_id: NodeId,
        parameters: CpuSvgArtboardParameters,
        configuration: RenderConfig,
        budget: CpuSvgEvaluationBudget,
    ) -> Result<Self, CpuSvgDocumentError> {
        Self::with_input(
            document_id_encoding,
            node_id,
            parameters,
            configuration,
            budget,
            None,
        )
    }

    pub fn new_with_encoded_png(
        document_id_encoding: u64,
        node_id: NodeId,
        parameters: CpuSvgArtboardParameters,
        configuration: RenderConfig,
        budget: CpuSvgEvaluationBudget,
        input: CpuSvgEncodedPngInput,
    ) -> Result<Self, CpuSvgDocumentError> {
        Self::with_input(
            document_id_encoding,
            node_id,
            parameters,
            configuration,
            budget,
            Some(input),
        )
    }

    fn with_input(
        document_id_encoding: u64,
        node_id: NodeId,
        parameters: CpuSvgArtboardParameters,
        configuration: RenderConfig,
        budget: CpuSvgEvaluationBudget,
        encoded_png: Option<CpuSvgEncodedPngInput>,
    ) -> Result<Self, CpuSvgDocumentError> {
        validate(parameters, configuration, budget)?;
        validate_input(encoded_png.as_ref(), parameters, configuration)?;
        Ok(Self {
            checkpoint: Arc::new(SvgCheckpoint {
                document_id_encoding,
                node_id,
                revision: 0,
                network: Arc::new(source_network(node_id, parameters, encoded_png.as_ref())?),
                encoded_png,
                parameters,
                configuration,
                budget,
            }),
            pending: None,
            accepted: None,
        })
    }

    pub(crate) fn from_persisted_checkpoint(
        document_id_encoding: u64,
        node_id: NodeId,
        revision: u64,
        parameters: CpuSvgArtboardParameters,
        configuration: RenderConfig,
        budget: CpuSvgEvaluationBudget,
        input: CpuSvgEncodedPngInput,
    ) -> Result<Self, CpuSvgDocumentError> {
        let mut document = Self::with_input(
            document_id_encoding,
            node_id,
            parameters,
            configuration,
            budget,
            Some(input),
        )?;
        Arc::get_mut(&mut document.checkpoint)
            .ok_or(CpuSvgDocumentError::InvalidSourceTemplate)?
            .revision = revision;
        Ok(document)
    }

    pub fn detach_for_edit(&self) -> Self {
        Self {
            checkpoint: self.checkpoint.clone(),
            pending: None,
            accepted: None,
        }
    }

    pub fn revision(&self) -> u64 {
        self.checkpoint.revision
    }
    pub fn node_id(&self) -> NodeId {
        self.checkpoint.node_id
    }
    pub fn document_id_encoding(&self) -> u64 {
        self.checkpoint.document_id_encoding
    }
    pub fn network(&self) -> &NodeNetwork {
        &self.checkpoint.network
    }
    pub fn parameters(&self) -> CpuSvgArtboardParameters {
        self.checkpoint.parameters
    }
    pub fn configuration(&self) -> RenderConfig {
        self.checkpoint.configuration
    }

    pub fn encoded_png_input(&self) -> Option<&CpuSvgEncodedPngInput> {
        self.checkpoint.encoded_png.as_ref()
    }

    pub fn apply_encoded_png(
        &mut self,
        expected_revision: u64,
        parameters: CpuSvgArtboardParameters,
        configuration: RenderConfig,
        input: CpuSvgEncodedPngInput,
    ) -> Result<bool, CpuSvgDocumentError> {
        self.apply_input(expected_revision, parameters, configuration, Some(input))
    }

    pub fn apply(
        &mut self,
        expected_revision: u64,
        parameters: CpuSvgArtboardParameters,
        configuration: RenderConfig,
    ) -> Result<bool, CpuSvgDocumentError> {
        self.apply_input(
            expected_revision,
            parameters,
            configuration,
            self.checkpoint.encoded_png.clone(),
        )
    }

    fn apply_input(
        &mut self,
        expected_revision: u64,
        parameters: CpuSvgArtboardParameters,
        configuration: RenderConfig,
        encoded_png: Option<CpuSvgEncodedPngInput>,
    ) -> Result<bool, CpuSvgDocumentError> {
        if expected_revision != self.revision() {
            return Err(CpuSvgDocumentError::StaleRevision);
        }
        validate(parameters, configuration, self.checkpoint.budget)?;
        validate_input(encoded_png.as_ref(), parameters, configuration)?;
        let same_input = same_input(encoded_png.as_ref(), self.checkpoint.encoded_png.as_ref());
        if same_input
            && same_parameters(parameters, self.parameters())
            && same_configuration(configuration, self.configuration())
        {
            return Ok(false);
        }
        let revision = self
            .revision()
            .checked_add(1)
            .ok_or(CpuSvgDocumentError::RevisionExhausted)?;
        let network = if same_input && same_parameters(parameters, self.parameters()) {
            self.checkpoint.network.clone()
        } else {
            Arc::new(source_network(
                self.node_id(),
                parameters,
                encoded_png.as_ref(),
            )?)
        };
        self.checkpoint = Arc::new(SvgCheckpoint {
            document_id_encoding: self.document_id_encoding(),
            node_id: self.node_id(),
            revision,
            network,
            encoded_png,
            parameters,
            configuration,
            budget: self.checkpoint.budget,
        });
        self.pending = None;
        Ok(true)
    }

    pub fn submit(&mut self) -> CpuSvgEvaluation {
        let submission = Arc::new(());
        self.pending = Some(submission.clone());
        CpuSvgEvaluation {
            checkpoint: self.checkpoint.clone(),
            submission,
        }
    }

    pub fn cancel(&mut self) -> bool {
        self.pending.take().is_some()
    }
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }
    pub fn accepted(&self) -> Option<&AcceptedCpuSvgOutput> {
        self.accepted.as_ref()
    }
    pub fn accepted_is_current(&self) -> bool {
        self.accepted
            .as_ref()
            .is_some_and(|accepted| Arc::ptr_eq(&accepted.checkpoint, &self.checkpoint))
    }

    pub fn owns_current_output(&self, output: &AcceptedCpuSvgOutput) -> bool {
        Arc::ptr_eq(&output.checkpoint, &self.checkpoint)
            && self
                .accepted
                .as_ref()
                .is_some_and(|accepted| Arc::ptr_eq(&accepted.output, &output.output))
    }

    pub fn accept(
        &mut self,
        completion: CpuSvgCompletion,
    ) -> Result<&AcceptedCpuSvgOutput, CpuSvgAcceptanceError> {
        if !Arc::ptr_eq(&completion.checkpoint, &self.checkpoint)
            || !self
                .pending
                .as_ref()
                .is_some_and(|pending| Arc::ptr_eq(pending, &completion.submission))
        {
            return Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion);
        }
        self.pending = None;
        let output = completion
            .result
            .map_err(CpuSvgAcceptanceError::Evaluation)?;
        let accepted = self.accepted.insert(AcceptedCpuSvgOutput {
            checkpoint: completion.checkpoint,
            output: Arc::new(output),
        });
        Ok(accepted)
    }
}

#[derive(Clone)]
pub struct AcceptedCpuSvgOutput {
    checkpoint: Arc<SvgCheckpoint>,
    output: Arc<RenderOutput>,
}

impl AcceptedCpuSvgOutput {
    pub fn output(&self) -> &RenderOutput {
        &self.output
    }
    pub fn output_lease(&self) -> Arc<RenderOutput> {
        self.output.clone()
    }
    pub fn revision(&self) -> u64 {
        self.checkpoint.revision
    }
    pub fn configuration(&self) -> RenderConfig {
        self.checkpoint.configuration
    }
    pub fn parameters(&self) -> CpuSvgArtboardParameters {
        self.checkpoint.parameters
    }
    pub fn encoded_png_input(&self) -> Option<&CpuSvgEncodedPngInput> {
        self.checkpoint.encoded_png.as_ref()
    }
    pub fn document_id_encoding(&self) -> u64 {
        self.checkpoint.document_id_encoding
    }
    pub fn node_id(&self) -> NodeId {
        self.checkpoint.node_id
    }
}

pub struct CpuSvgEvaluation {
    checkpoint: Arc<SvgCheckpoint>,
    submission: Arc<()>,
}

pub struct CpuSvgCompletion {
    checkpoint: Arc<SvgCheckpoint>,
    submission: Arc<()>,
    result: Result<RenderOutput, CpuSvgEvaluationError>,
}

impl CpuSvgCompletion {
    pub fn result(&self) -> Result<&RenderOutput, &CpuSvgEvaluationError> {
        self.result.as_ref()
    }
}

struct SvgPreferences(u32);
impl GetEditorPreferences for SvgPreferences {
    fn max_render_region_area(&self) -> u32 {
        self.0
    }
}

impl CpuSvgEvaluation {
    pub fn network(&self) -> &NodeNetwork {
        &self.checkpoint.network
    }

    pub async fn evaluate(self) -> CpuSvgCompletion {
        let result = AssertUnwindSafe(async {
            // The private network has either no resource or exactly one sealed PNG.
            // Its original memory proxy grants no filesystem or host-service authority.
            let mut application_io = PlatformApplicationIo::default();
            if let Some(input) = &self.checkpoint.encoded_png {
                let storage = HashMapResourceStorage::new();
                storage.store(input.source.encoded_bytes().as_ref());
                application_io.inject_resource_proxy(Box::new(storage));
            }
            let api = Arc::new(PlatformEditorApi {
                application_io: Some(Arc::new(application_io)),
                editor_preferences: Box::new(SvgPreferences(
                    self.checkpoint.budget.max_viewport_pixels,
                )),
                ..PlatformEditorApi::default()
            });
            let mut network = graphite_interpreted_executor::util::wrap_network_in_scope(
                (*self.checkpoint.network).clone(),
                api,
            );
            graphite_preprocessor::Preprocessor::new()
                .preprocess(&mut network, &|_| None)
                .map_err(|error| CpuSvgEvaluationError::Preprocessing(error.to_string()))?;
            let proto = Compiler {}
                .compile_single(network)
                .map_err(CpuSvgEvaluationError::Compilation)?;
            let executor = DynamicExecutor::new(proto)
                .await
                .map_err(|errors| CpuSvgEvaluationError::Construction(Box::new(errors)))?;
            if executor.input_type() != Some(graphite_graph_craft::concrete!(Context<'static>))
                || executor.output_type() != Some(graphite_std::future!(Item<RenderOutput>))
            {
                return Err(CpuSvgEvaluationError::UnexpectedContract);
            }
            let value = Executor::execute(&&executor, self.checkpoint.configuration.into_context())
                .await
                .map_err(|error| {
                    CpuSvgEvaluationError::Execution(CpuSvgExecutionError(error.to_string()))
                })?;
            if let Some(input) = &self.checkpoint.encoded_png {
                validate_decoded_image(&executor, self.checkpoint.node_id, input)?;
            }
            let TaggedValue::RenderOutput(output) = value else {
                return Err(CpuSvgEvaluationError::UnexpectedOutput);
            };
            let RenderOutputType::Svg { svg, image_data } = &output.data else {
                return Err(CpuSvgEvaluationError::UnexpectedOutput);
            };
            if !image_data.is_empty() {
                return Err(CpuSvgEvaluationError::UnexpectedOutput);
            }
            if svg.len() > self.checkpoint.budget.max_output_bytes {
                return Err(CpuSvgEvaluationError::OutputBudgetExceeded);
            }
            Ok(output)
        })
        .catch_unwind()
        .await
        .unwrap_or_else(|payload| {
            let detail = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|detail| (*detail).to_owned())
                })
                .unwrap_or_else(|| "Non-string panic payload".to_owned());
            Err(CpuSvgEvaluationError::SourcePanic(detail))
        });
        CpuSvgCompletion {
            checkpoint: self.checkpoint,
            submission: self.submission,
            result,
        }
    }
}

fn source_network(
    node_id: NodeId,
    parameters: CpuSvgArtboardParameters,
    encoded_png: Option<&CpuSvgEncodedPngInput>,
) -> Result<NodeNetwork, CpuSvgDocumentError> {
    // This fixture is emitted and round-tripped by the pinned editor's public template
    // API. Loading it avoids a second handwritten definition or production editor owner.
    let mut network: NodeNetwork = serde_json::from_str(include_str!("artboard-template.json"))
        .map_err(|_| CpuSvgDocumentError::InvalidSourceTemplate)?;
    let source_id = NodeId(7001);
    if network.nodes.len() != 1 || network.exports != vec![NodeInput::node(source_id, 0)] {
        return Err(CpuSvgDocumentError::InvalidSourceTemplate);
    }
    let mut node = network
        .nodes
        .remove(&source_id)
        .ok_or(CpuSvgDocumentError::InvalidSourceTemplate)?;
    for (index, value) in [
        (2, TaggedValue::DVec2(parameters.location)),
        (3, TaggedValue::DVec2(parameters.dimensions)),
        (4, TaggedValue::Color(parameters.background)),
        (5, TaggedValue::Bool(parameters.clip)),
    ] {
        let input = node
            .inputs
            .get_mut(index)
            .ok_or(CpuSvgDocumentError::InvalidSourceTemplate)?;
        let Some(original) = input.as_value() else {
            return Err(CpuSvgDocumentError::InvalidSourceTemplate);
        };
        if std::mem::discriminant(original) != std::mem::discriminant(&value) {
            return Err(CpuSvgDocumentError::InvalidSourceTemplate);
        }
        *input = NodeInput::value(value, false);
    }
    if let Some(input) = encoded_png {
        let [resource_id, decode_id, placement_id] = encoded_png_node_ids(node_id)?;
        let mut placement: DocumentNode =
            serde_json::from_str(include_str!("replace-transform-template.json"))
                .map_err(|_| CpuSvgDocumentError::InvalidSourceTemplate)?;
        if placement.implementation
            != DocumentNodeImplementation::ProtoNode(replace_transform::IDENTIFIER)
            || placement.inputs.len() != 2
            || placement
                .inputs
                .get(1)
                .and_then(NodeInput::as_value)
                .is_none_or(|value| !matches!(value, TaggedValue::DAffine2(_)))
        {
            return Err(CpuSvgDocumentError::InvalidSourceTemplate);
        }
        placement.inputs = vec![
            NodeInput::node(decode_id, 0),
            NodeInput::value(TaggedValue::DAffine2(input.placement), false),
        ];
        let content = node
            .inputs
            .get_mut(1)
            .ok_or(CpuSvgDocumentError::InvalidSourceTemplate)?;
        *content = NodeInput::node(placement_id, 0);
        network.nodes.extend([
            (
                resource_id,
                DocumentNode {
                    inputs: vec![
                        NodeInput::scope(editor_api::IDENTIFIER.as_static_str()),
                        NodeInput::value(
                            TaggedValue::ResourceHash(ResourceHash::from(
                                input.source.encoded_bytes().as_ref(),
                            )),
                            false,
                        ),
                    ],
                    implementation: DocumentNodeImplementation::ProtoNode(resource::IDENTIFIER),
                    ..DocumentNode::default()
                },
            ),
            (
                decode_id,
                DocumentNode {
                    inputs: vec![NodeInput::node(resource_id, 0)],
                    implementation: DocumentNodeImplementation::ProtoNode(decode_image::IDENTIFIER),
                    ..DocumentNode::default()
                },
            ),
            (placement_id, placement),
        ]);
    }
    network.nodes.insert(node_id, node);
    network.exports = vec![NodeInput::node(node_id, 0)];
    Ok(network)
}

fn validate_decoded_image(
    executor: &DynamicExecutor,
    node_id: NodeId,
    input: &CpuSvgEncodedPngInput,
) -> Result<(), CpuSvgEvaluationError> {
    // WriteAttribute in the authentic Artboard template promotes its Item to List.
    // Inspecting the existing monitor after execution avoids another pixel decode.
    let monitored = executor
        .introspect(&[node_id, NodeId(3)])
        .map_err(|_| CpuSvgEvaluationError::UnexpectedDecodedImage)?;
    let observed = monitored
        .downcast_ref::<IORecord<Context<'static>, List<Artboard>>>()
        .ok_or(CpuSvgEvaluationError::UnexpectedDecodedImage)?;
    if observed.output.len() != 1 {
        return Err(CpuSvgEvaluationError::UnexpectedDecodedImage);
    }
    let content = observed
        .output
        .element(0)
        .ok_or(CpuSvgEvaluationError::UnexpectedDecodedImage)?
        .as_graphic_list();
    if content.len() != 1 {
        return Err(CpuSvgEvaluationError::UnexpectedDecodedImage);
    }
    let (raster, placement) = match content.element(0) {
        Some(Graphic::RasterCPU(item)) => {
            (item.element(), item.attribute::<DAffine2>(ATTR_TRANSFORM))
        }
        Some(Graphic::RasterCPUList(list)) if list.len() == 1 => (
            list.element(0)
                .ok_or(CpuSvgEvaluationError::UnexpectedDecodedImage)?,
            list.attribute::<DAffine2>(ATTR_TRANSFORM, 0),
        ),
        _ => return Err(CpuSvgEvaluationError::UnexpectedDecodedImage),
    };
    let image = raster.data();
    if image.width == 0 && image.height == 0 && image.data.is_empty() {
        return Err(CpuSvgEvaluationError::SourceDecodeFailed);
    }
    let pixels = u64::from(input.source.width()) * u64::from(input.source.height());
    if image.width != input.source.width()
        || image.height != input.source.height()
        || u64::try_from(image.data.len()).ok() != Some(pixels)
        || placement.is_none_or(|placement| {
            placement.to_cols_array().map(f64::to_bits)
                != input.placement.to_cols_array().map(f64::to_bits)
        })
    {
        return Err(CpuSvgEvaluationError::UnexpectedDecodedImage);
    }
    Ok(())
}

fn same_input(
    first: Option<&CpuSvgEncodedPngInput>,
    second: Option<&CpuSvgEncodedPngInput>,
) -> bool {
    match (first, second) {
        (None, None) => true,
        (Some(first), Some(second)) => {
            Arc::ptr_eq(&first.source, &second.source)
                && first.placement.to_cols_array().map(f64::to_bits)
                    == second.placement.to_cols_array().map(f64::to_bits)
        }
        _ => false,
    }
}

fn validate_placement(placement: DAffine2) -> Result<(), CpuSvgDocumentError> {
    if !placement.is_finite() {
        return Err(CpuSvgDocumentError::InvalidPlacement);
    }
    Ok(())
}

fn validate_input(
    input: Option<&CpuSvgEncodedPngInput>,
    parameters: CpuSvgArtboardParameters,
    configuration: RenderConfig,
) -> Result<(), CpuSvgDocumentError> {
    let Some(input) = input else {
        return Ok(());
    };
    validate_placement(input.placement)?;
    let physical =
        DAffine2::from_scale(DVec2::splat(configuration.scale)) * configuration.viewport.transform;
    let logical = DAffine2::from_scale(DVec2::splat(1. / configuration.scale)) * physical;
    let location = parameters
        .location
        .min(parameters.location + parameters.dimensions);
    let artwork = DAffine2::from_translation(location) * input.placement;
    let transformations = [input.placement, artwork, logical * artwork];
    if transformations.into_iter().any(|transform| {
        !transform.is_finite()
            || [DVec2::ZERO, DVec2::X, DVec2::Y, DVec2::ONE]
                .into_iter()
                .any(|point| !transform.transform_point2(point).is_finite())
    }) {
        return Err(CpuSvgDocumentError::InvalidPlacement);
    }
    Ok(())
}

// Original CacheHash and serialization distinguish signed zeros. Numeric equality
// would discard an edit and could incorrectly reuse a different source network.
fn same_parameters(first: CpuSvgArtboardParameters, second: CpuSvgArtboardParameters) -> bool {
    first.location.to_array().map(f64::to_bits) == second.location.to_array().map(f64::to_bits)
        && first.dimensions.to_array().map(f64::to_bits)
            == second.dimensions.to_array().map(f64::to_bits)
        && [
            first.background.r(),
            first.background.g(),
            first.background.b(),
            first.background.a(),
        ]
        .map(f32::to_bits)
            == [
                second.background.r(),
                second.background.g(),
                second.background.b(),
                second.background.a(),
            ]
            .map(f32::to_bits)
        && first.clip == second.clip
}

fn same_configuration(first: RenderConfig, second: RenderConfig) -> bool {
    first.viewport.transform.to_cols_array().map(f64::to_bits)
        == second.viewport.transform.to_cols_array().map(f64::to_bits)
        && first.viewport.resolution == second.viewport.resolution
        && first.viewport.quality == second.viewport.quality
        && first.scale.to_bits() == second.scale.to_bits()
        && first.time.time.to_bits() == second.time.time.to_bits()
        && first.time.animation_time == second.time.animation_time
        && first.pointer.to_array().map(f64::to_bits) == second.pointer.to_array().map(f64::to_bits)
        && first.render_mode == second.render_mode
        && first.export_format == second.export_format
        && first.for_export == second.for_export
        && first.for_eyedropper == second.for_eyedropper
}

fn validate(
    parameters: CpuSvgArtboardParameters,
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
) -> Result<(), CpuSvgDocumentError> {
    let scale = configuration.scale;
    let viewport = configuration.viewport;
    let reciprocal = 1. / scale;
    let physical = DAffine2::from_scale(DVec2::splat(scale)) * viewport.transform;
    let logical = DAffine2::from_scale(DVec2::splat(reciprocal)) * physical;
    let resolution = viewport.resolution.as_dvec2() / scale;
    if configuration.export_format != ExportFormat::Svg
        || configuration.render_mode
            != graphite_std::vector_types::vector::style::RenderMode::Normal
        || !configuration.for_export
        || configuration.for_eyedropper
        || !scale.is_finite()
        || scale <= 0.
        || !reciprocal.is_finite()
        || !viewport.transform.is_finite()
        || !physical.is_finite()
        || !logical.is_finite()
        || !viewport.transform.matrix2.x_axis.length().is_finite()
        || !resolution.is_finite()
        || resolution.min_element() <= 0.
        || !configuration.time.time.is_finite()
        || !configuration.pointer.is_finite()
    {
        return Err(CpuSvgDocumentError::InvalidConfiguration);
    }
    let pixels = u64::from(viewport.resolution.x) * u64::from(viewport.resolution.y);
    if pixels > u64::from(budget.max_viewport_pixels) {
        return Err(CpuSvgDocumentError::ViewportBudgetExceeded);
    }
    if !parameters.location.is_finite()
        || !parameters.dimensions.is_finite()
        || !(parameters.location + parameters.dimensions).is_finite()
    {
        return Err(CpuSvgDocumentError::InvalidGeometry);
    }
    let location = parameters
        .location
        .min(parameters.location + parameters.dimensions);
    let dimensions = parameters.dimensions.abs().max(DVec2::ONE);
    let translated = physical * DAffine2::from_translation(location);
    let source_content = DAffine2::from_translation(parameters.location) * physical;
    if !(location + dimensions).is_finite()
        || !translated.is_finite()
        || !source_content.is_finite()
        || [
            location,
            location + dimensions,
            DVec2::new(location.x, location.y + dimensions.y),
            DVec2::new(location.x + dimensions.x, location.y),
        ]
        .into_iter()
        .any(|point| {
            !physical.transform_point2(point).is_finite()
                || !logical.transform_point2(point).is_finite()
        })
    {
        return Err(CpuSvgDocumentError::InvalidGeometry);
    }
    let color = parameters.background;
    if [color.r(), color.g(), color.b(), color.a()]
        .into_iter()
        .any(|channel| !channel.is_finite())
        || color
            .to_gamma_srgb_channels()
            .into_iter()
            .any(|channel| !channel.is_finite())
        || (color.a() < 1. && !(color.a() * 1000.).is_finite())
    {
        return Err(CpuSvgDocumentError::InvalidBackground);
    }
    Ok(())
}

#[cfg(test)]
#[path = "evaluate_tests.rs"]
mod tests;

pub(crate) fn encoded_png_node_ids(node_id: NodeId) -> Result<[NodeId; 3], CpuSvgDocumentError> {
    let mut identifiers = [7102, 7103, 7104, 7105]
        .into_iter()
        .map(NodeId)
        .filter(|identifier| *identifier != node_id);
    Ok([
        identifiers
            .next()
            .ok_or(CpuSvgDocumentError::InvalidSourceTemplate)?,
        identifiers
            .next()
            .ok_or(CpuSvgDocumentError::InvalidSourceTemplate)?,
        identifiers
            .next()
            .ok_or(CpuSvgDocumentError::InvalidSourceTemplate)?,
    ])
}
