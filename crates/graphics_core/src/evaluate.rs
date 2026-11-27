use std::{
    collections::BTreeMap, fmt, future::Future, ops::Sub, panic::AssertUnwindSafe, sync::Arc,
};

use futures::FutureExt as _;
use graphite_core_types::{
    Color, Context, Node, Type,
    list::{Item, List, NodeIdPath},
    ops::Convert as _,
    render_complexity::RenderComplexity as _,
};
use graphite_glam::{DAffine2, DVec2, UVec2};
use graphite_graph_craft::{
    application_io::{
        PlatformApplicationIo, PlatformEditorApi,
        resource::{
            HashMapResourceStorage, LoadResource, Resource, ResourceFuture, ResourceHash,
            ResourceId, ResourceStorage,
        },
    },
    document::{
        DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork,
        value::{RenderOutput, RenderOutputType, TaggedValue},
    },
    graphene_compiler::{Compiler, Executor},
    proto::{GraphErrors, ProtoNetwork, TypingContext},
};
use graphite_interpreted_executor::{
    dynamic_executor::{DynamicExecutor, NodeTypes},
    node_registry::NODE_REGISTRY,
};
use graphite_math_nodes::{NthRoot, SafeDivide};
use graphite_raster_types::Raster;
use graphite_std::{
    ATTR_EDITOR_LAYER_PATH, ATTR_TRANSFORM, Artboard, Graphic,
    application_io::{ApplicationIo as _, ExportFormat, GetEditorPreferences, RenderConfig},
    bounds::RenderBoundingBox,
    memo::IORecord,
    platform_application_io::{decode_image, editor_api, resource},
    renderer::{
        Render as _, RenderParams, RenderSvgSegmentList as _, SvgRender, graphic_list_bounding_box,
    },
    transform::{Footprint, RenderQuality},
    transform_nodes::replace_transform,
    vector_types::vector::style::RenderMode,
};

use crate::{
    AddEvaluationError, Input, resources::PreparedEncodedPngSource, vector_elements, vector_items,
};

pub fn evaluate_subtract(minuend: f64, subtrahend: f64) -> Result<f64, AddEvaluationError> {
    subtract_value(minuend, subtrahend)
}

pub fn evaluate_subtract_lists(
    minuends: &[f64],
    subtrahends: &[f64],
) -> Result<Vec<f64>, AddEvaluationError> {
    subtract_list_items(numeric_items(minuends), numeric_items(subtrahends))
        .map(|items| items.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_subtract_i64(minuend: i64, subtrahend: i64) -> Result<i64, AddEvaluationError> {
    subtract_value(minuend, subtrahend)
}

pub fn evaluate_subtract_i64_lists(
    minuends: &[i64],
    subtrahends: &[i64],
) -> Result<Vec<i64>, AddEvaluationError> {
    subtract_list_items(numeric_items(minuends), numeric_items(subtrahends))
        .map(|items| items.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_subtract_vector(
    minuend: [f64; 2],
    subtrahend: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    subtract_value(DVec2::from_array(minuend), DVec2::from_array(subtrahend))
        .map(|result| result.to_array())
}

pub fn evaluate_subtract_scalar_vector(
    minuend: f64,
    subtrahend: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    subtract_value(minuend, DVec2::from_array(subtrahend)).map(|result| result.to_array())
}

pub fn evaluate_subtract_vector_scalar(
    minuend: [f64; 2],
    subtrahend: f64,
) -> Result<[f64; 2], AddEvaluationError> {
    subtract_value(DVec2::from_array(minuend), subtrahend).map(|result| result.to_array())
}

pub fn evaluate_subtract_vector_lists(
    minuends: &[[f64; 2]],
    subtrahends: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    subtract_list_items(vector_items(minuends), vector_items(subtrahends)).map(vector_elements)
}

pub fn evaluate_subtract_scalar_vector_lists(
    minuends: &[f64],
    subtrahends: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    subtract_list_items(numeric_items(minuends), vector_items(subtrahends)).map(vector_elements)
}

pub fn evaluate_subtract_vector_scalar_lists(
    minuends: &[[f64; 2]],
    subtrahends: &[f64],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    subtract_list_items(vector_items(minuends), numeric_items(subtrahends)).map(vector_elements)
}

fn numeric_items<Value: Copy>(values: &[Value]) -> List<Value> {
    values.iter().copied().map(Item::new_from_element).collect()
}

fn subtract_value<Left, Right>(
    minuend: Left,
    subtrahend: Right,
) -> Result<Left::Output, AddEvaluationError>
where
    Left: Sub<Right> + Clone + Send + Sync + 'static,
    Right: Clone + Send + Sync + 'static,
    Left::Output: Send + Sync + 'static,
{
    subtract_item(
        Item::new_from_element(minuend),
        Item::new_from_element(subtrahend),
    )
    .map(Item::into_element)
}

fn subtract_item<Left, Right>(
    minuend: Item<Left>,
    subtrahend: Item<Right>,
) -> Result<Item<Left::Output>, AddEvaluationError>
where
    Left: Sub<Right> + Clone + Send + Sync + 'static,
    Right: Clone + Send + Sync + 'static,
    Left::Output: Send + Sync + 'static,
{
    let node = graphite_math_nodes::SubtractNode::new(Input(minuend), Input(subtrahend));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

fn subtract_list_items<Left, Right>(
    minuends: List<Left>,
    subtrahends: List<Right>,
) -> Result<List<Left::Output>, AddEvaluationError>
where
    Left: Sub<Right> + Clone + Send + Sync + 'static,
    Right: Clone + Send + Sync + 'static,
    Left::Output: Clone + Send + Sync + 'static,
{
    let node = graphite_math_nodes::SubtractNodeMapped::new(Input(minuends), Input(subtrahends));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

pub fn evaluate_divide(numerator: f64, denominator: f64) -> Result<f64, AddEvaluationError> {
    divide_value(numerator, denominator)
}

pub fn evaluate_divide_lists(
    numerators: &[f64],
    denominators: &[f64],
) -> Result<Vec<f64>, AddEvaluationError> {
    divide_list_items(numeric_items(numerators), numeric_items(denominators))
        .map(|items| items.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_divide_i64(numerator: i64, denominator: i64) -> Result<i64, AddEvaluationError> {
    divide_value(numerator, denominator)
}

pub fn evaluate_divide_i64_lists(
    numerators: &[i64],
    denominators: &[i64],
) -> Result<Vec<i64>, AddEvaluationError> {
    divide_list_items(numeric_items(numerators), numeric_items(denominators))
        .map(|items| items.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_divide_vector(
    numerator: [f64; 2],
    denominator: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    divide_value(DVec2::from_array(numerator), DVec2::from_array(denominator))
        .map(|value| value.to_array())
}

pub fn evaluate_divide_scalar_vector(
    numerator: f64,
    denominator: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    divide_value(numerator, DVec2::from_array(denominator)).map(|value| value.to_array())
}

pub fn evaluate_divide_vector_scalar(
    numerator: [f64; 2],
    denominator: f64,
) -> Result<[f64; 2], AddEvaluationError> {
    divide_value(DVec2::from_array(numerator), denominator).map(|value| value.to_array())
}

pub fn evaluate_divide_vector_lists(
    numerators: &[[f64; 2]],
    denominators: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    divide_list_items(vector_items(numerators), vector_items(denominators)).map(vector_elements)
}

pub fn evaluate_divide_scalar_vector_lists(
    numerators: &[f64],
    denominators: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    divide_list_items(numeric_items(numerators), vector_items(denominators)).map(vector_elements)
}

pub fn evaluate_divide_vector_scalar_lists(
    numerators: &[[f64; 2]],
    denominators: &[f64],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    divide_list_items(vector_items(numerators), numeric_items(denominators)).map(vector_elements)
}

fn divide_value<Numerator, Denominator>(
    numerator: Numerator,
    denominator: Denominator,
) -> Result<Numerator::Output, AddEvaluationError>
where
    Numerator: SafeDivide<Denominator> + Clone + Send + Sync + 'static,
    Denominator: Clone + Send + Sync + 'static,
    Numerator::Output: Send + Sync + 'static,
{
    let node = graphite_math_nodes::DivideNode::new(
        Input(Item::new_from_element(numerator)),
        Input(Item::new_from_element(denominator)),
    );
    evaluate_source_node(&node).map(Item::into_element)
}

fn divide_list_items<Numerator, Denominator>(
    numerators: List<Numerator>,
    denominators: List<Denominator>,
) -> Result<List<Numerator::Output>, AddEvaluationError>
where
    Numerator: SafeDivide<Denominator> + Clone + Send + Sync + 'static,
    Denominator: Clone + Send + Sync + 'static,
    Numerator::Output: Clone + Send + Sync + 'static,
{
    let node = graphite_math_nodes::DivideNodeMapped::new(Input(numerators), Input(denominators));
    evaluate_source_node(&node)
}

pub fn evaluate_reciprocal(value: f64) -> Result<f64, AddEvaluationError> {
    let node = graphite_math_nodes::ReciprocalNode::new(Input(Item::new_from_element(value)));
    evaluate_source_node(&node).map(Item::into_element)
}

pub fn evaluate_reciprocal_vector(value: [f64; 2]) -> Result<[f64; 2], AddEvaluationError> {
    let node = graphite_math_nodes::ReciprocalNode::new(Input(Item::new_from_element(
        DVec2::from_array(value),
    )));
    evaluate_source_node(&node).map(|item| item.into_element().to_array())
}

pub fn evaluate_reciprocal_list(values: &[f64]) -> Result<Vec<f64>, AddEvaluationError> {
    let node = graphite_math_nodes::ReciprocalNodeMapped::new(Input(numeric_items(values)));
    evaluate_source_node(&node).map(|items| items.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_reciprocal_vector_list(
    values: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    let node = graphite_math_nodes::ReciprocalNodeMapped::new(Input(vector_items(values)));
    evaluate_source_node(&node).map(vector_elements)
}

fn evaluate_source_node<'node, Output, Evaluation: Future<Output = Output>>(
    node: &'node impl Node<'node, Context<'node>, Output = Evaluation>,
) -> Result<Output, AddEvaluationError> {
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryNumericOperation {
    Ceiling,
    Floor,
    Round,
    AbsoluteValue,
    Sign,
}

macro_rules! evaluate_unary_numeric {
    ($operation:expr, $input:expr, $ceiling:ident, $floor:ident, $round:ident, $absolute_value:ident, $sign:ident) => {{
        let input = Input($input);
        match $operation {
            UnaryNumericOperation::Ceiling => {
                evaluate_source_node(&graphite_math_nodes::$ceiling::new(input))
            }
            UnaryNumericOperation::Floor => {
                evaluate_source_node(&graphite_math_nodes::$floor::new(input))
            }
            UnaryNumericOperation::Round => {
                evaluate_source_node(&graphite_math_nodes::$round::new(input))
            }
            UnaryNumericOperation::AbsoluteValue => {
                evaluate_source_node(&graphite_math_nodes::$absolute_value::new(input))
            }
            UnaryNumericOperation::Sign => {
                evaluate_source_node(&graphite_math_nodes::$sign::new(input))
            }
        }
    }};
}

impl UnaryNumericOperation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ceiling => "Ceiling",
            Self::Floor => "Floor",
            Self::Round => "Round",
            Self::AbsoluteValue => "Absolute Value",
            Self::Sign => "Sign",
        }
    }

    pub fn evaluate_scalar(self, value: f64) -> Result<f64, AddEvaluationError> {
        evaluate_unary_numeric!(
            self,
            Item::new_from_element(value),
            CeilingNode,
            FloorNode,
            RoundNode,
            AbsoluteValueNode,
            SignNode
        )
        .map(Item::into_element)
    }

    pub fn evaluate_vector(self, value: [f64; 2]) -> Result<[f64; 2], AddEvaluationError> {
        evaluate_unary_numeric!(
            self,
            Item::new_from_element(DVec2::from_array(value)),
            CeilingNode,
            FloorNode,
            RoundNode,
            AbsoluteValueNode,
            SignNode
        )
        .map(|item| item.into_element().to_array())
    }

    pub fn evaluate_list(self, values: &[f64]) -> Result<Vec<f64>, AddEvaluationError> {
        evaluate_unary_numeric!(
            self,
            numeric_items(values),
            CeilingNodeMapped,
            FloorNodeMapped,
            RoundNodeMapped,
            AbsoluteValueNodeMapped,
            SignNodeMapped
        )
        .map(|items| items.into_iter().map(Item::into_element).collect())
    }

    pub fn evaluate_vector_list(
        self,
        values: &[[f64; 2]],
    ) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
        evaluate_unary_numeric!(
            self,
            vector_items(values),
            CeilingNodeMapped,
            FloorNodeMapped,
            RoundNodeMapped,
            AbsoluteValueNodeMapped,
            SignNodeMapped
        )
        .map(vector_elements)
    }
}

pub(crate) struct NumericGraphEvaluator(DynamicExecutor);

#[derive(Debug, thiserror::Error)]
pub(crate) enum NumericGraphPreparationError {
    #[error("{0}")]
    Compiler(String),
    #[error("{0:?}")]
    Nodes(GraphErrors),
}

pub(crate) const NUMERIC_NETWORK_WRAPPER: NodeId = NodeId(0);

impl NumericGraphEvaluator {
    pub(crate) async fn prepare(
        network: NodeNetwork,
    ) -> Result<Self, NumericGraphPreparationError> {
        AssertUnwindSafe(async {
            let proto = compile_detached_numeric_network(network)
                .map_err(NumericGraphPreparationError::Compiler)?;
            let executor = DynamicExecutor::new(proto)
                .await
                .map_err(NumericGraphPreparationError::Nodes)?;
            if executor.input_type() != Some(graphite_graph_craft::concrete!(Context<'static>)) {
                return Err(NumericGraphPreparationError::Compiler(
                    "Unexpected Graphite evaluation context.".into(),
                ));
            }
            Ok(Self(executor))
        })
        .catch_unwind()
        .await
        .unwrap_or_else(|_| {
            Err(NumericGraphPreparationError::Compiler(
                "The pinned Graphite evaluator stopped while preparing the numeric graph.".into(),
            ))
        })
    }

    pub(crate) fn node_types(&self, node: NodeId) -> Option<NodeTypes> {
        self.0
            .document_node_types(std::iter::once([NUMERIC_NETWORK_WRAPPER, node].into()))
            .next()
            .map(|(_, types)| types)
    }

    pub(crate) async fn evaluate(&self) -> Result<TaggedValue, String> {
        AssertUnwindSafe(async { Executor::execute(&&self.0, Context::default()).await })
            .catch_unwind()
            .await
            .map_err(|_| "The pinned Graphite evaluator stopped while running the numeric graph.")?
            .map_err(|error| error.to_string())
    }
}

pub(crate) fn typecheck_detached_numeric_network(network: NodeNetwork) -> Result<(), String> {
    typecheck_detached_ranked_network(
        network,
        &[
            graphite_core_types::concrete!(f64),
            graphite_core_types::concrete!(i64),
            graphite_core_types::concrete!(DVec2),
        ],
        "The selected output has an unsupported Graphite numeric type.",
    )
    .map(|_| ())
}

pub(crate) fn typecheck_detached_ranked_network(
    network: NodeNetwork,
    elements: &[Type],
    incompatible_output: &'static str,
) -> Result<NodeTypes, String> {
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        let proto = compile_detached_numeric_network(network)?;
        let mut typing = TypingContext::new(&NODE_REGISTRY);
        typing
            .update(&proto)
            .map_err(|error| format!("{error:?}"))?;
        let types = typing
            .type_of(proto.output)
            .ok_or_else(|| "Graphite did not resolve the selected output type.".to_owned())?;
        let supported = match types.return_value.nested_type() {
            Type::Item(element) | Type::List(element) => elements.contains(element.as_ref()),
            _ => false,
        };
        if types.call_argument != graphite_core_types::concrete!(Context<'static>) || !supported {
            return Err(incompatible_output.into());
        }
        Ok(NodeTypes {
            inputs: types.inputs.clone(),
            output: types.return_value.clone(),
        })
    }))
    .unwrap_or_else(|_| {
        Err("The pinned Graphite compiler stopped while typing the selected output.".into())
    })
}

fn compile_detached_numeric_network(network: NodeNetwork) -> Result<ProtoNetwork, String> {
    let branch_id = NUMERIC_NETWORK_WRAPPER;
    let mut network = NodeNetwork {
        nodes: [(
            branch_id,
            DocumentNode {
                implementation: DocumentNodeImplementation::Network(network),
                ..Default::default()
            },
        )]
        .into_iter()
        .collect(),
        exports: vec![NodeInput::node(branch_id, 0)],
        ..Default::default()
    };
    graphite_preprocessor::Preprocessor::new()
        .preprocess(&mut network, &|_| None)
        .map_err(|error| error.to_string())?;
    if !network.nodes.get(&branch_id).is_some_and(|branch| {
        matches!(
            branch.implementation,
            DocumentNodeImplementation::Network(_)
        )
    }) {
        return Err("Graphite changed the selected numeric branch during preprocessing.".into());
    }
    // The preprocessor injects registered scopes that the pure branch never uses.
    network.remove_dead_nodes(0);
    Compiler {}.compile_single(network)
}

pub fn evaluate_absolute_value_i64(value: i64) -> Result<i64, AddEvaluationError> {
    let node = graphite_math_nodes::AbsoluteValueNode::new(Input(Item::new_from_element(value)));
    evaluate_source_node(&node).map(Item::into_element)
}

pub fn evaluate_absolute_value_i64_list(values: &[i64]) -> Result<Vec<i64>, AddEvaluationError> {
    let node = graphite_math_nodes::AbsoluteValueNodeMapped::new(Input(numeric_items(values)));
    evaluate_source_node(&node).map(|items| items.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_root(radicand: f64, degree: f64) -> Result<f64, AddEvaluationError> {
    root_value(radicand, degree)
}

pub fn evaluate_root_vector(
    radicand: [f64; 2],
    degree: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    root_value(DVec2::from_array(radicand), DVec2::from_array(degree))
        .map(|result| result.to_array())
}

pub fn evaluate_root_scalar_vector(
    radicand: f64,
    degree: [f64; 2],
) -> Result<[f64; 2], AddEvaluationError> {
    root_value(radicand, DVec2::from_array(degree)).map(|result| result.to_array())
}

pub fn evaluate_root_vector_scalar(
    radicand: [f64; 2],
    degree: f64,
) -> Result<[f64; 2], AddEvaluationError> {
    root_value(DVec2::from_array(radicand), degree).map(|result| result.to_array())
}

pub fn evaluate_root_lists(
    radicands: &[f64],
    degrees: &[f64],
) -> Result<Vec<f64>, AddEvaluationError> {
    root_list_items(numeric_items(radicands), numeric_items(degrees))
        .map(|items| items.into_iter().map(Item::into_element).collect())
}

pub fn evaluate_root_vector_lists(
    radicands: &[[f64; 2]],
    degrees: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    root_list_items(vector_items(radicands), vector_items(degrees)).map(vector_elements)
}

pub fn evaluate_root_scalar_vector_lists(
    radicands: &[f64],
    degrees: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    root_list_items(numeric_items(radicands), vector_items(degrees)).map(vector_elements)
}

pub fn evaluate_root_vector_scalar_lists(
    radicands: &[[f64; 2]],
    degrees: &[f64],
) -> Result<Vec<[f64; 2]>, AddEvaluationError> {
    root_list_items(vector_items(radicands), numeric_items(degrees)).map(vector_elements)
}

fn root_value<Radicand, Degree>(
    radicand: Radicand,
    degree: Degree,
) -> Result<Radicand::Output, AddEvaluationError>
where
    Radicand: NthRoot<Degree> + Clone + Send + Sync + 'static,
    Degree: Clone + Send + Sync + 'static,
    Radicand::Output: Send + Sync + 'static,
{
    root_item(
        Item::new_from_element(radicand),
        Item::new_from_element(degree),
    )
    .map(Item::into_element)
}

fn root_item<Radicand, Degree>(
    radicand: Item<Radicand>,
    degree: Item<Degree>,
) -> Result<Item<Radicand::Output>, AddEvaluationError>
where
    Radicand: NthRoot<Degree> + Clone + Send + Sync + 'static,
    Degree: Clone + Send + Sync + 'static,
    Radicand::Output: Send + Sync + 'static,
{
    let node = graphite_math_nodes::RootNode::new(Input(radicand), Input(degree));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

fn root_list_items<Radicand, Degree>(
    radicands: List<Radicand>,
    degrees: List<Degree>,
) -> Result<List<Radicand::Output>, AddEvaluationError>
where
    Radicand: NthRoot<Degree> + Clone + Send + Sync + 'static,
    Degree: Clone + Send + Sync + 'static,
    Radicand::Output: Clone + Send + Sync + 'static,
{
    let node = graphite_math_nodes::RootNodeMapped::new(Input(radicands), Input(degrees));
    futures::executor::block_on(AssertUnwindSafe(node.eval(Context::default())).catch_unwind())
        .map_err(|_| AddEvaluationError)
}

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
    UnexpectedLayerMonitor,
    LayerThumbnailComplexity,
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
            Self::UnexpectedLayerMonitor => formatter
                .write_str("Graphite could not verify the saved layer's thumbnail monitor."),
            Self::LayerThumbnailComplexity => {
                formatter.write_str("The layer is too complex for a Graphite thumbnail.")
            }
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

#[derive(Clone, Copy, Debug)]
pub struct CpuSvgArtboardEditState {
    pub parameters: CpuSvgArtboardParameters,
    pub image_placement: Option<DAffine2>,
}

impl PartialEq for CpuSvgArtboardEditState {
    fn eq(&self, other: &Self) -> bool {
        let placement_bits = |placement: Option<DAffine2>| {
            placement.map(|placement| placement.to_cols_array().map(f64::to_bits))
        };
        same_parameters(self.parameters, other.parameters)
            && placement_bits(self.image_placement) == placement_bits(other.image_placement)
    }
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

    pub fn edit_state(&self) -> CpuSvgArtboardEditState {
        CpuSvgArtboardEditState {
            parameters: self.parameters(),
            image_placement: self
                .encoded_png_input()
                .map(CpuSvgEncodedPngInput::placement),
        }
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
        let result = evaluate_cpu_svg(
            &self.checkpoint.network,
            self.checkpoint.configuration,
            self.checkpoint.budget,
            self.checkpoint
                .encoded_png
                .as_ref()
                .map(|input| (self.checkpoint.node_id, input)),
        )
        .await;
        CpuSvgCompletion {
            checkpoint: self.checkpoint,
            submission: self.submission,
            result,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct SavedGraphResources {
    hashes: BTreeMap<ResourceId, ResourceHash>,
    bytes: BTreeMap<ResourceHash, Resource>,
}

impl SavedGraphResources {
    pub(crate) fn insert(&mut self, id: ResourceId, resource: Resource) {
        let hash = resource.hash();
        self.hashes.insert(id, hash);
        self.bytes.entry(hash).or_insert(resource);
    }

    pub(crate) fn resource(&self, id: ResourceId) -> Option<&Resource> {
        self.hashes.get(&id).and_then(|hash| self.bytes.get(hash))
    }

    fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

impl LoadResource for SavedGraphResources {
    fn load(&self, hash: ResourceHash) -> ResourceFuture<'_> {
        Box::pin(std::future::ready(self.bytes.get(&hash).cloned()))
    }
}

pub(crate) async fn evaluate_cpu_svg(
    network: &NodeNetwork,
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
    encoded_png: Option<(NodeId, &CpuSvgEncodedPngInput)>,
) -> Result<RenderOutput, CpuSvgEvaluationError> {
    evaluate_cpu_svg_with_resources(
        network,
        configuration,
        budget,
        encoded_png,
        &SavedGraphResources::default(),
    )
    .await
}

pub(crate) async fn evaluate_cpu_svg_with_resources(
    network: &NodeNetwork,
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
    encoded_png: Option<(NodeId, &CpuSvgEncodedPngInput)>,
    resources: &SavedGraphResources,
) -> Result<RenderOutput, CpuSvgEvaluationError> {
    AssertUnwindSafe(async {
        // Both resource paths use memory-only proxies to keep rendering independent
        // of filesystem or host-service access.
        let mut application_io = PlatformApplicationIo::default();
        if let Some((_, input)) = encoded_png {
            if !resources.is_empty() {
                return Err(CpuSvgEvaluationError::UnexpectedContract);
            }
            let storage = HashMapResourceStorage::new();
            storage.store(input.source.encoded_bytes().as_ref());
            application_io.inject_resource_proxy(Box::new(storage));
        } else if !resources.is_empty() {
            application_io.inject_resource_proxy(Box::new(resources.clone()));
        }
        let (output, _) = execute_source_render(
            network,
            configuration,
            budget,
            Arc::new(application_io),
            encoded_png,
            resources,
        )
        .await?;
        let RenderOutputType::Svg { svg, image_data } = &output.data else {
            return Err(CpuSvgEvaluationError::UnexpectedOutput);
        };
        if !image_data.is_empty() {
            return Err(CpuSvgEvaluationError::UnexpectedOutput);
        }
        if svg.len() > budget.max_output_bytes {
            return Err(CpuSvgEvaluationError::OutputBudgetExceeded);
        }
        Ok(output)
    })
    .catch_unwind()
    .await
    .unwrap_or_else(|payload| Err(source_render_panic(payload)))
}

pub(crate) async fn evaluate_merge_thumbnail(
    network: &NodeNetwork,
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
    owner: NodeId,
    monitor: NodeId,
    resources: &SavedGraphResources,
) -> Result<String, CpuSvgEvaluationError> {
    AssertUnwindSafe(async {
        let mut application_io = PlatformApplicationIo::default();
        if !resources.is_empty() {
            application_io.inject_resource_proxy(Box::new(resources.clone()));
        }
        let (_, executor) = execute_source_render(
            network,
            configuration,
            budget,
            Arc::new(application_io),
            None,
            resources,
        )
        .await?;
        let observed = executor
            .introspect(&[owner, monitor])
            .map_err(|_| CpuSvgEvaluationError::UnexpectedLayerMonitor)?;
        let content = &observed
            .downcast_ref::<IORecord<Context<'static>, List<Graphic>>>()
            .ok_or(CpuSvgEvaluationError::UnexpectedLayerMonitor)?
            .output;
        let owner_path = NodeIdPath::from(vec![owner]);
        if (0..content.len()).any(|index| {
            content.attribute::<NodeIdPath>(ATTR_EDITOR_LAYER_PATH, index) != Some(&owner_path)
        }) {
            return Err(CpuSvgEvaluationError::UnexpectedLayerMonitor);
        }
        if content.render_complexity() > 1000 {
            return Err(CpuSvgEvaluationError::LayerThumbnailComplexity);
        }
        let bounds = match graphic_list_bounding_box(content, DAffine2::IDENTITY) {
            RenderBoundingBox::Rectangle(bounds) if bounds[0] != bounds[1] => bounds,
            _ => [DVec2::ZERO, DVec2::ONE],
        };
        let size = bounds[1] - bounds[0];
        let center = bounds[0] + size / 2.;
        let half_extent = DVec2::new(size.x.max(size.y * 1.5), size.y.max(size.x / 1.5)) / 2.;
        let [min, max] = [center - half_extent, center + half_extent];
        if !min.is_finite() || !max.is_finite() || (max - min).min_element() <= 0. {
            return Err(CpuSvgEvaluationError::UnexpectedLayerMonitor);
        }
        let mut render = SvgRender::new();
        content.render_svg(
            &mut render,
            &RenderParams {
                footprint: Footprint {
                    transform: DAffine2::from_translation(min),
                    resolution: UVec2::new((max.x - min.x) as u32, (max.y - min.y) as u32),
                    quality: RenderQuality::Full,
                },
                thumbnail: true,
                ..Default::default()
            },
        );
        render.format_svg(min, max);
        let resolution = configuration.viewport.resolution;
        // Native rasterization needs explicit dimensions to fit the source viewBox.
        let svg = render.svg.to_svg_string().replacen(
            "<svg ",
            &format!(
                r#"<svg width="{}" height="{}" "#,
                resolution.x, resolution.y
            ),
            1,
        );
        if svg.len() > budget.max_output_bytes {
            return Err(CpuSvgEvaluationError::OutputBudgetExceeded);
        }
        Ok(svg)
    })
    .catch_unwind()
    .await
    .unwrap_or_else(|payload| Err(source_render_panic(payload)))
}

fn source_render_panic(payload: Box<dyn std::any::Any + Send>) -> CpuSvgEvaluationError {
    let detail = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|detail| (*detail).to_owned())
        })
        .unwrap_or_else(|| "Non-string panic payload".to_owned());
    CpuSvgEvaluationError::SourcePanic(detail)
}

async fn execute_source_render(
    network: &NodeNetwork,
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
    application_io: Arc<PlatformApplicationIo>,
    encoded_png: Option<(NodeId, &CpuSvgEncodedPngInput)>,
    resources: &SavedGraphResources,
) -> Result<(RenderOutput, DynamicExecutor), CpuSvgEvaluationError> {
    let api = Arc::new(PlatformEditorApi {
        application_io: Some(application_io),
        editor_preferences: Box::new(SvgPreferences(budget.max_viewport_pixels)),
        ..PlatformEditorApi::default()
    });
    let mut network =
        graphite_interpreted_executor::util::wrap_network_in_scope(network.clone(), api);
    graphite_preprocessor::Preprocessor::new()
        .preprocess(&mut network, &|id| resources.hashes.get(&id).copied())
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
    let value = Executor::execute(&&executor, configuration.into_context())
        .await
        .map_err(|error| {
            CpuSvgEvaluationError::Execution(CpuSvgExecutionError(error.to_string()))
        })?;
    if let Some((node, input)) = encoded_png {
        validate_decoded_image(&executor, node, input)?;
    }
    match value {
        TaggedValue::RenderOutput(output) => Ok((output, executor)),
        _ => Err(CpuSvgEvaluationError::UnexpectedOutput),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CanvasGpuError {
    #[error("Graphite could not initialize the GPU Canvas renderer.")]
    Unavailable,
    #[error("Graphite requires a finite raster Canvas view.")]
    InvalidConfiguration,
    #[error("The Graphite viewport exceeds the raster rendering budget.")]
    ViewportBudgetExceeded,
    #[error("Graphite did not return pixels for the requested raster viewport.")]
    UnexpectedOutput,
    #[error("{0}")]
    Evaluation(#[from] CpuSvgEvaluationError),
}

pub struct CanvasGpuRenderer {
    application_io: Arc<PlatformApplicationIo>,
}

impl CanvasGpuRenderer {
    pub async fn new() -> Result<Self, CanvasGpuError> {
        let application_io = AssertUnwindSafe(PlatformApplicationIo::new())
            .catch_unwind()
            .await
            .map_err(|_| CanvasGpuError::Unavailable)?;
        if application_io.gpu_executor().is_none() {
            return Err(CanvasGpuError::Unavailable);
        }
        Ok(Self {
            application_io: Arc::new(application_io),
        })
    }

    pub(crate) async fn render(
        &self,
        network: &NodeNetwork,
        configuration: RenderConfig,
        budget: CpuSvgEvaluationBudget,
        resources: &SavedGraphResources,
    ) -> Result<RenderOutput, CanvasGpuError> {
        if configuration.export_format != ExportFormat::Raster
            || configuration.for_export
            || configuration.for_eyedropper
            || configuration.render_mode == RenderMode::SvgPreview
        {
            return Err(CanvasGpuError::InvalidConfiguration);
        }
        validate_viewport_configuration(configuration, budget).map_err(|error| match error {
            CpuSvgDocumentError::ViewportBudgetExceeded => CanvasGpuError::ViewportBudgetExceeded,
            _ => CanvasGpuError::InvalidConfiguration,
        })?;
        let executor = self
            .application_io
            .gpu_executor()
            .ok_or(CanvasGpuError::Unavailable)?;
        let resolution = configuration.viewport.resolution;
        if resolution.max_element() > executor.context().device.limits().max_texture_dimension_2d {
            return Err(CanvasGpuError::ViewportBudgetExceeded);
        }
        AssertUnwindSafe(async {
            let application_io = if resources.is_empty() {
                self.application_io.clone()
            } else {
                let mut application_io =
                    PlatformApplicationIo::new_with_context(executor.context().clone());
                application_io.inject_resource_proxy(Box::new(resources.clone()));
                Arc::new(application_io)
            };
            let (RenderOutput { data, metadata }, _) = execute_source_render(
                network,
                configuration,
                budget,
                application_io,
                None,
                resources,
            )
            .await?;
            let RenderOutputType::Texture(texture) = data else {
                return Err(CanvasGpuError::UnexpectedOutput);
            };
            if texture.width() != resolution.x || texture.height() != resolution.y {
                return Err(CanvasGpuError::UnexpectedOutput);
            }
            let raster = Raster::new_gpu(texture)
                .convert(Footprint::BOUNDLESS, executor)
                .await;
            let (data, width, height) = raster.to_flat_u8();
            if width != resolution.x
                || height != resolution.y
                || u64::try_from(data.len()).ok() != Some(u64::from(width) * u64::from(height) * 4)
            {
                return Err(CanvasGpuError::UnexpectedOutput);
            }
            Ok(RenderOutput {
                data: RenderOutputType::Buffer {
                    data,
                    width,
                    height,
                },
                metadata,
            })
        })
        .catch_unwind()
        .await
        .unwrap_or_else(|payload| Err(source_render_panic(payload).into()))
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

pub(crate) fn validate_svg_configuration(
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
) -> Result<(), CpuSvgDocumentError> {
    if configuration.export_format != ExportFormat::Svg
        || configuration.render_mode != RenderMode::Normal
        || !configuration.for_export
        || configuration.for_eyedropper
    {
        return Err(CpuSvgDocumentError::InvalidConfiguration);
    }
    validate_viewport_configuration(configuration, budget)
}

fn validate_viewport_configuration(
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
) -> Result<(), CpuSvgDocumentError> {
    let scale = configuration.scale;
    let viewport = configuration.viewport;
    let reciprocal = 1. / scale;
    let physical = DAffine2::from_scale(DVec2::splat(scale)) * viewport.transform;
    let logical = DAffine2::from_scale(DVec2::splat(reciprocal)) * physical;
    let resolution = viewport.resolution.as_dvec2() / scale;
    if !scale.is_finite()
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
    Ok(())
}

fn validate(
    parameters: CpuSvgArtboardParameters,
    configuration: RenderConfig,
    budget: CpuSvgEvaluationBudget,
) -> Result<(), CpuSvgDocumentError> {
    validate_svg_configuration(configuration, budget)?;
    let scale = configuration.scale;
    let physical = DAffine2::from_scale(DVec2::splat(scale)) * configuration.viewport.transform;
    let logical = DAffine2::from_scale(DVec2::splat(1. / scale)) * physical;
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
